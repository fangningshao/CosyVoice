# encoding: utf-8
# Copyright (c) 2026 (authors: Fangning Shao)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for CosyVoice3 embedding model.
Supports multi-task training with multiple JSONL files.
"""

import os
import sys

# Set memory optimization and cache settings BEFORE any imports
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MODELSCOPE_CACHE'] = os.environ.get('MODELSCOPE_CACHE', r'D:\\models\\modelscope_cache')
os.environ['MODELSCOPE_OFFLINE'] = os.environ.get('MODELSCOPE_OFFLINE', '1')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

sys.path.append('.')
import argparse
import glob
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset, WeightedRandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from torch.utils.tensorboard import SummaryWriter
from embedding.model import CosyVoice3Embedding
from embedding.parquet_dataset import ParquetVoiceEmbeddingDataset, collate_fn
from embedding.loss import InfoNCELoss, MultiTaskContrastiveLoss


# Fix Unicode encoding for Windows console output redirection
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr:
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')



def worker_init_fn(worker_id):
    """Initialize worker with unique random seed. Must be a top-level function for Windows pickling."""
    import numpy as np
    # Get seed from global scope (set by main)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def setup_logging(output_dir, rank=0):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"train_rank{rank}.log"
    
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints: keeps best and recent N checkpoints."""
    
    def __init__(self, output_dir, keep_recent=10, rank=0):
        """
        Args:
            output_dir: Output directory for checkpoints
            keep_recent: Number of recent checkpoints to keep
            rank: Process rank (only rank 0 saves)
        """
        self.checkpoint_dir = Path(output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_recent = keep_recent
        self.rank = rank
        
        self.best_val_loss = float('inf')
        self.best_checkpoint = None
        self.recent_checkpoints = []
        
        # Load existing checkpoint info if resuming
        self.info_file = self.checkpoint_dir / "checkpoint_info.txt"
        if self.info_file.exists():
            self._load_info()
    
    def _load_info(self):
        """Load checkpoint info from file."""
        try:
            with open(self.info_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    self.best_val_loss = float(lines[0].split(':')[1].strip())
                    self.best_checkpoint = lines[1].split(':')[1].strip() if len(lines) > 1 else None
                    if len(lines) > 2:
                        self.recent_checkpoints = [line.strip() for line in lines[2:]]
        except Exception as e:
            logging.warning(f"Failed to load checkpoint info: {e}")
    
    def _save_info(self):
        """Save checkpoint info to file."""
        if self.rank != 0:
            return
        
        with open(self.info_file, 'w') as f:
            f.write(f"best_val_loss: {self.best_val_loss}\n")
            f.write(f"best_checkpoint: {self.best_checkpoint}\n")
            for ckpt in self.recent_checkpoints:
                f.write(f"{ckpt}\n")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, val_loss=None, use_lora=False):
        """Save checkpoint and manage old checkpoints."""
        if self.rank != 0:
            logging.info(f"Rank {self.rank} skipping checkpoint save")
            return
        
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        logging.info(f"Saving checkpoint to: {checkpoint_path}")
        
        # Unwrap DDP if needed
        model_to_save = model.module if hasattr(model, 'module') else model
        
        try:
            # Prepare state dict for saving
            if use_lora:
                # Only save LoRA parameters (trainable parameters)
                logging.info("Extracting LoRA weights only (not saving frozen base model parameters)...")
                
                # Get the PEFT model state dict (includes only LoRA adapters)
                from peft import get_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(model_to_save.llm)
                
                # Create a clean state dict with proper key names
                # Remove the PEFT wrapper prefixes for easier loading
                clean_state_dict = {}
                for key, value in model_to_save.state_dict().items():
                    if key.startswith('llm.'):
                        # Only save LoRA weights (skip base_layer weights)
                        if 'lora_' in key:
                            # Remove 'base_model.model.' prefix from PEFT wrapper
                            clean_key = key.replace('llm.base_model.model.', 'llm.')
                            clean_state_dict[clean_key] = value
                            # logging.debug(f"Saving LoRA weight: {clean_key}")
                    else:
                        # Save non-LLM parts (like llm_decoder, speech_embedding if trainable)
                        clean_state_dict[key] = value
                
                state_dict_to_save = clean_state_dict
                
                logging.info(f"@ Extracted {len(state_dict_to_save)} LoRA parameter tensors")
                logging.info(f"  (Frozen base model parameters not saved - reducing checkpoint size)")
            else:
                # Save full model state dict (standard training without LoRA)
                state_dict_to_save = model_to_save.state_dict()
                logging.info(f"Saving full model state dict with {len(state_dict_to_save)} parameters")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': state_dict_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'use_lora': use_lora,  # Store whether LoRA was used
            }, checkpoint_path)
            
            logging.info(f"@ Checkpoint saved successfully: {checkpoint_path}")
            
            # Log checkpoint size
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            logging.info(f"  Checkpoint size: {checkpoint_size_mb:.2f} MB")
            
        except Exception as e:
            logging.error(f"✗ Failed to save checkpoint: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return
        
        # Add to recent checkpoints
        self.recent_checkpoints.append(checkpoint_name)
        
        # Update best checkpoint if this is better
        is_best = False
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_checkpoint = checkpoint_name
            is_best = True
            
            # Create symlink to best checkpoint
            best_link = self.checkpoint_dir / "checkpoint_best.pt"
            if best_link.exists():
                best_link.unlink()
            try:
                best_link.symlink_to(checkpoint_path.name)
            except:
                # On Windows, symlinks may not work, so copy instead
                import shutil
                shutil.copy2(checkpoint_path, best_link)
            
            logging.info(f"@ New best checkpoint! Val loss: {val_loss:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Save checkpoint info
        self._save_info()
        
        return checkpoint_path, is_best
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping best and recent N."""
        if self.rank != 0:
            return
        
        # Keep only recent N checkpoints
        if len(self.recent_checkpoints) > self.keep_recent:
            to_remove = self.recent_checkpoints[:-self.keep_recent]
            self.recent_checkpoints = self.recent_checkpoints[-self.keep_recent:]
            
            for ckpt_name in to_remove:
                # Don't remove if it's the best checkpoint
                if ckpt_name == self.best_checkpoint:
                    continue
                
                ckpt_path = self.checkpoint_dir / ckpt_name
                if ckpt_path.exists():
                    ckpt_path.unlink()
                    logging.info(f"Removed old checkpoint: {ckpt_name}")


def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir, rank=0):
    """Save training checkpoint (legacy function for compatibility)."""
    if rank != 0:
        return
    
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    
    # Unwrap DDP if needed
    model_to_save = model.module if hasattr(model, 'module') else model
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    # Unwrap DDP if needed
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    
    logging.info(f"Checkpoint loaded: epoch {epoch}, step {step}")
    return epoch, step


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, 
                epoch, gradient_accumulation_steps, use_mixed_precision, rank=0, 
                save_steps=100, output_dir=None, global_step=0, writer=None, 
                checkpoint_manager=None, val_loader=None, use_lora=False, val_max_batches=32,
                max_grad_norm=1.0):
    """Train for one epoch with detailed loss logging and periodic validation."""
    import time  # Import at function level
    
    model.train()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    task_losses = {}
    num_batches = 0
    
    # Add timing metrics
    data_loading_time = 0
    gpu_forward_time = 0
    gpu_backward_time = 0
    optimizer_step_time = 0
    total_iteration_time = 0
    
    # Track gradient norms
    total_grad_norm = 0
    grad_norm_count = 0
    
    optimizer.zero_grad()
    
    # Use ncols=None to auto-detect terminal width, or set a fixed width
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0), ncols=160)
    
    iter_start_time = time.time()
    
    for batch_idx, batch in enumerate(pbar):
        # Measure data loading time (time to get batch from dataloader)
        batch_ready_time = time.time()
        data_load_time = batch_ready_time - iter_start_time
        data_loading_time += data_load_time
        
        # Skip None batches (all samples failed)
        if batch is None:
            logging.warning(f"Skipping batch {batch_idx} - all samples failed to load")
            iter_start_time = time.time()
            continue
        
        # Move batch to device
        device_transfer_start = time.time()
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        device_transfer_time = time.time() - device_transfer_start
        
        # Forward pass with mixed precision
        forward_start_time = time.time()
        if use_mixed_precision:
            with autocast():
                # Extract embeddings from model
                query_output = model(
                    batch['query_text_token'],
                    batch['query_text_token_len'],
                    batch['query_speech_token'],
                    batch['query_speech_token_len']
                )
                query_embeddings = query_output['embedding']
                
                positive_output = model(
                    batch['positive_text_token'],
                    batch['positive_text_token_len'],
                    batch['positive_speech_token'],
                    batch['positive_speech_token_len']
                )
                positive_embeddings = positive_output['embedding']
                
                # Extract negative embeddings if provided
                negative_embeddings = None
                if 'negative_speech_token' in batch:
                    negative_output = model(
                        batch['negative_text_token'],
                        batch['negative_text_token_len'],
                        batch['negative_speech_token'],
                        batch['negative_speech_token_len']
                    )
                    negative_embeddings = negative_output['embedding']
                
                # Compute loss
                loss_dict = criterion(
                    query_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                    batch.get('negative_counts', None)
                )
                
                loss = loss_dict['total_loss'] / gradient_accumulation_steps
        else:
            # Extract embeddings from model
            query_output = model(
                batch['query_text_token'],
                batch['query_text_token_len'],
                batch['query_speech_token'],
                batch['query_speech_token_len']
            )
            query_embeddings = query_output['embedding']
            
            positive_output = model(
                batch['positive_text_token'],
                batch['positive_text_token_len'],
                batch['positive_speech_token'],
                batch['positive_speech_token_len']
            )
            positive_embeddings = positive_output['embedding']
            
            # Extract negative embeddings if provided
            negative_embeddings = None
            if 'negative_speech_token' in batch:
                negative_output = model(
                    batch['negative_text_token'],
                    batch['negative_text_token_len'],
                    batch['negative_speech_token'],
                    batch['negative_speech_token_len']
                )
                negative_embeddings = negative_output['embedding']
            
            # Compute loss
            loss_dict = criterion(
                query_embeddings,
                positive_embeddings,
                negative_embeddings,
                batch.get('negative_counts', None)
            )
            
            loss = loss_dict['total_loss'] / gradient_accumulation_steps
        
        forward_time = time.time() - forward_start_time
        gpu_forward_time += forward_time
        
        # Backward pass
        backward_start_time = time.time()
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        backward_time = time.time() - backward_start_time
        gpu_backward_time += backward_time
        
        # Free memory
        del query_output, positive_output
        if negative_embeddings is not None:
            del negative_output
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optim_start_time = time.time()
            
            # Gradient clipping
            grad_norm = None
            if max_grad_norm > 0:
                if use_mixed_precision:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=max_grad_norm
                )
                
                # Track gradient norms
                total_grad_norm += grad_norm.item()
                grad_norm_count += 1
            
            # Optimizer step
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            optim_time = time.time() - optim_start_time
            optimizer_step_time += optim_time
            
            global_step += 1
            
            # Log to TensorBoard
            if writer is not None and rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss_dict['total_loss'].item(), global_step)
                writer.add_scalar('train/softmax_loss', loss_dict.get('softmax_loss', 0), global_step)
                writer.add_scalar('train/hard_negative_loss', loss_dict.get('hard_negative_loss', 0), global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                # Log gradient norm
                if grad_norm is not None:
                    writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                
                # Log task-specific losses
                if 'task_losses' in loss_dict:
                    for task_type, task_loss in loss_dict['task_losses'].items():
                        writer.add_scalar(f'train/task_{task_type}', task_loss, global_step)
                
                # Log timing metrics (every 10 steps)
                if global_step % 50 == 0:
                    avg_data_time = data_loading_time / (batch_idx + 1)
                    avg_forward_time = gpu_forward_time / (batch_idx + 1)
                    avg_backward_time = gpu_backward_time / (batch_idx + 1)
                    total_gpu_time = avg_forward_time + avg_backward_time
                    
                    writer.add_scalar('timing/data_loading_ms', avg_data_time * 1000, global_step)
                    writer.add_scalar('timing/gpu_forward_ms', avg_forward_time * 1000, global_step)
                    writer.add_scalar('timing/gpu_backward_ms', avg_backward_time * 1000, global_step)
                    writer.add_scalar('timing/total_gpu_ms', total_gpu_time * 1000, global_step)
                    writer.add_scalar('timing/data_to_gpu_ratio', avg_data_time / (total_gpu_time + 1e-6), global_step)
            
            # Run validation and save checkpoint every save_steps
            if save_steps > 0 and global_step % save_steps == 0:
                # Run validation if val_loader is provided
                val_loss = None
                if val_loader is not None and rank == 0:
                    logging.info(f"\nRunning validation at step {global_step}...")
                    val_losses = validate(
                        model, val_loader, criterion, device, use_mixed_precision, 
                        rank, writer, global_step, max_batches=val_max_batches
                    )
                    val_loss = val_losses['total_loss']
                    
                    logging.info(f"Validation results at step {global_step}:")
                    logging.info(f"  Total Loss: {val_losses['total_loss']:.4f}")
                    logging.info(f"  Softmax Loss: {val_losses['softmax_loss']:.4f}")
                    logging.info(f"  Hard Negative Loss: {val_losses['hard_negative_loss']:.4f}")
                
                # Save checkpoint
                if checkpoint_manager is not None and rank == 0:
                    logging.info(f"Saving checkpoint at step {global_step}")
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step, val_loss=val_loss, use_lora=use_lora
                    )
                
                # Return to training mode
                model.train()
            
            # Clear cache periodically
            if (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()
        
        # Accumulate losses
        total_loss += loss_dict['total_loss'].item()
        total_softmax_loss += loss_dict.get('softmax_loss', 0)
        total_hard_neg_loss += loss_dict.get('hard_negative_loss', 0)
        
        # Accumulate task-specific losses
        if 'task_losses' in loss_dict:
            for task_type, task_loss in loss_dict['task_losses'].items():
                if task_type not in task_losses:
                    task_losses[task_type] = 0
                task_losses[task_type] += task_loss
        
        num_batches += 1
        
        # Measure total iteration time
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        total_iteration_time += iter_time
        
        # Update progress bar with timing info
        current_lr = scheduler.get_last_lr()[0]
        avg_data_time = data_loading_time / num_batches
        avg_gpu_time = (gpu_forward_time + gpu_backward_time) / num_batches
        
        pbar.set_postfix({
            'L': f"{loss_dict['total_loss'].item():.4f}",
            'avg': f"{total_loss / num_batches:.4f}",
            'sm': f"{loss_dict.get('softmax_loss', 0):.4f}",
            'hn': f"{loss_dict.get('hard_negative_loss', 0):.4f}",
            'lr': f"{current_lr:.6f}",
            'step': global_step,
            'data_ms': f"{avg_data_time*1000:.0f}",
            'gpu_ms': f"{avg_gpu_time*1000:.0f}",
            'ratio': f"{avg_data_time/avg_gpu_time:.1f}x"
        })
        
        # Print detailed timing info every 10 batches
        if rank == 0 and (batch_idx + 1) % 50 == 0:
            avg_data_time = data_loading_time / num_batches
            avg_forward_time = gpu_forward_time / num_batches
            avg_backward_time = gpu_backward_time / num_batches
            avg_optim_time = optimizer_step_time / max(1, (num_batches // gradient_accumulation_steps))
            total_gpu_time = avg_forward_time + avg_backward_time
            
            logging.info(f"\n{'='*80}")
            logging.info(f"#  TIMING ANALYSIS (Batch {batch_idx + 1}/{len(train_loader)})")
            logging.info(f"{'='*80}")
            logging.info(f"Data Loading:     {avg_data_time*1000:>8.1f} ms  ({avg_data_time/iter_time*100:>5.1f}%)")
            logging.info(f"GPU Forward:      {avg_forward_time*1000:>8.1f} ms  ({avg_forward_time/iter_time*100:>5.1f}%)")
            logging.info(f"GPU Backward:     {avg_backward_time*1000:>8.1f} ms  ({avg_backward_time/iter_time*100:>5.1f}%)")
            logging.info(f"Optimizer Step:   {avg_optim_time*1000:>8.1f} ms  ({avg_optim_time/iter_time*100:>5.1f}%)")
            logging.info(f"Device Transfer:  {device_transfer_time*1000:>8.1f} ms  (last batch)")
            logging.info(f"─────────────────────────────────────────")
            logging.info(f"Total GPU Time:   {total_gpu_time*1000:>8.1f} ms")
            logging.info(f"Total Iter Time:  {(total_iteration_time/num_batches)*1000:>8.1f} ms")
            logging.info(f"")
            logging.info(f"# Data-to-GPU Ratio: {avg_data_time/total_gpu_time:.2f}x")
            if avg_data_time > total_gpu_time:
                logging.warning(f"#  DATA LOADING IS BLOCKING GPU! ({avg_data_time/total_gpu_time:.1f}x slower)")
                logging.warning(f"   Suggestions:")
                logging.warning(f"   • Increase --num_workers (currently: {train_loader.num_workers})")
                logging.warning(f"   • Use --batch_softmax_only to skip loading hard negatives")
                logging.warning(f"   • Check disk I/O (SSD vs HDD)")
                logging.warning(f"   • Verify audio files are not corrupted/slow to read")
            else:
                logging.info(f"@ GPU is well-fed by data pipeline")
            logging.info(f"{'='*80}\n")
        
        # Start timing next iteration
        iter_start_time = time.time()
    
    # Calculate average losses
    avg_total_loss = total_loss / num_batches
    avg_softmax_loss = total_softmax_loss / num_batches
    avg_hard_neg_loss = total_hard_neg_loss / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
    
    # Log final timing summary
    if rank == 0 and num_batches > 0:
        avg_data_time = data_loading_time / num_batches
        avg_forward_time = gpu_forward_time / num_batches
        avg_backward_time = gpu_backward_time / num_batches
        total_gpu_time = avg_forward_time + avg_backward_time
        
        logging.info(f"\n{'='*80}")
        logging.info(f"# EPOCH {epoch} TIMING SUMMARY")
        logging.info(f"{'='*80}")
        logging.info(f"Average Data Loading Time:  {avg_data_time*1000:.1f} ms/batch")
        logging.info(f"Average GPU Forward Time:   {avg_forward_time*1000:.1f} ms/batch")
        logging.info(f"Average GPU Backward Time:  {avg_backward_time*1000:.1f} ms/batch")
        logging.info(f"Average Total GPU Time:     {total_gpu_time*1000:.1f} ms/batch")
        logging.info(f"")
        logging.info(f"Data-to-GPU Ratio: {avg_data_time/total_gpu_time:.2f}x")
        if avg_data_time > total_gpu_time * 1.5:
            logging.warning(f"❌ SEVERE DATA LOADING BOTTLENECK DETECTED!")
        elif avg_data_time > total_gpu_time:
            logging.warning(f"⚠️  Data loading is slower than GPU processing")
        else:
            logging.info(f"@ Data pipeline is keeping up with GPU")
        logging.info(f"{'='*80}\n")
    
    return {
        'total_loss': avg_total_loss,
        'softmax_loss': avg_softmax_loss,
        'hard_negative_loss': avg_hard_neg_loss,
        'task_losses': avg_task_losses,
        'global_step': global_step
    }


def validate(model, val_loader, criterion, device, use_mixed_precision, rank=0, writer=None, global_step=0, max_batches=64):
    """Validate the model with detailed loss logging (limited to max_batches examples)."""
    model.eval()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    task_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", disable=(rank != 0), total=min(len(val_loader), max_batches))
        
        for batch_idx, batch in enumerate(pbar):
            # Limit validation to max_batches
            if batch_idx >= max_batches:
                break
            
            # Move batch to device
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            if use_mixed_precision:
                with autocast():
                    # Extract embeddings from model
                    query_output = model(
                        batch['query_text_token'],
                        batch['query_text_token_len'],
                        batch['query_speech_token'],
                        batch['query_speech_token_len']
                    )
                    query_embeddings = query_output['embedding']
                    
                    positive_output = model(
                        batch['positive_text_token'],
                        batch['positive_text_token_len'],
                        batch['positive_speech_token'],
                        batch['positive_speech_token_len']
                    )
                    positive_embeddings = positive_output['embedding']
                    
                    # Extract negative embeddings if provided
                    negative_embeddings = None
                    if 'negative_speech_token' in batch:
                        negative_output = model(
                            batch['negative_text_token'],
                            batch['negative_text_token_len'],
                            batch['negative_speech_token'],
                            batch['negative_speech_token_len']
                        )
                        negative_embeddings = negative_output['embedding']
                    
                    # Compute loss
                    loss_dict = criterion(
                        query_embeddings,
                        positive_embeddings,
                        negative_embeddings,
                        batch.get('negative_counts', None)
                    )
            else:
                # Extract embeddings from model
                query_output = model(
                    batch['query_text_token'],
                    batch['query_text_token_len'],
                    batch['query_speech_token'],
                    batch['query_speech_token_len']
                )
                query_embeddings = query_output['embedding']
                
                positive_output = model(
                    batch['positive_text_token'],
                    batch['positive_text_token_len'],
                    batch['positive_speech_token'],
                    batch['positive_speech_token_len']
                )
                positive_embeddings = positive_output['embedding']
                
                # Extract negative embeddings if provided
                negative_embeddings = None
                if 'negative_speech_token' in batch:
                    negative_output = model(
                        batch['negative_text_token'],
                        batch['negative_text_token_len'],
                        batch['negative_speech_token'],
                        batch['negative_speech_token_len']
                    )
                    negative_embeddings = negative_output['embedding']
                
                # Compute loss
                loss_dict = criterion(
                    query_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                    batch.get('negative_counts', None)
                )
            
            # Free memory
            del query_output, positive_output
            if negative_embeddings is not None:
                del negative_output
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_softmax_loss += loss_dict.get('softmax_loss', 0)
            total_hard_neg_loss += loss_dict.get('hard_negative_loss', 0)
            
            # Accumulate task-specific losses
            if 'task_losses' in loss_dict:
                for task_type, task_loss in loss_dict['task_losses'].items():
                    if task_type not in task_losses:
                        task_losses[task_type] = 0
                    task_losses[task_type] += task_loss
            
            num_batches += 1
            
            pbar.set_postfix({'val_loss': f"{loss_dict['total_loss'].item():.4f}", 'batch': f"{batch_idx+1}/{max_batches}"})
    
    # Calculate average losses
    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_softmax_loss = total_softmax_loss / num_batches if num_batches > 0 else 0
    avg_hard_neg_loss = total_hard_neg_loss / num_batches if num_batches > 0 else 0
    avg_task_losses = {k: v / num_batches for k, v in task_losses.items()} if num_batches > 0 else {}
    
    # Log to TensorBoard
    if writer is not None and rank == 0:
        writer.add_scalar('val/loss', avg_total_loss, global_step)
        writer.add_scalar('val/softmax_loss', avg_softmax_loss, global_step)
        writer.add_scalar('val/hard_negative_loss', avg_hard_neg_loss, global_step)
        
        # Log task-specific losses
        for task_type, task_loss in avg_task_losses.items():
            writer.add_scalar(f'val/task_{task_type}', task_loss, global_step)
    
    return {
        'total_loss': avg_total_loss,
        'softmax_loss': avg_softmax_loss,
        'hard_negative_loss': avg_hard_neg_loss,
        'task_losses': avg_task_losses
    }


def main():
    parser = argparse.ArgumentParser(description='Train CosyVoice3 embedding model')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--llm_path', type=str, required=True, help='LLM checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='exp/embedding', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--loss_type', type=str, default='infonce', choices=['infonce', 'triplet'], help='Loss type')
    parser.add_argument('--use_hard_negatives', action='store_true', help='Use hard negatives')
    parser.add_argument('--batch_softmax_only', action='store_true', help='Use ONLY batch softmax (in-batch negatives), ignore hard negatives from data. Faster and simpler. Larger batch size recommended!')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every N steps')
    parser.add_argument('--keep_recent', type=int, default=10, help='Number of recent checkpoints to keep')
    parser.add_argument('--val_max_batches', type=int, default=32, help='Maximum number of validation batches (default: 32)')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Gradient clipping and warmup
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping (default: 1.0, set to 0 to disable)')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps for learning rate (default: 100)')
    parser.add_argument('--lr_decay_style', type=str, default='linear', choices=['cosine', 'linear'],
                       help='Learning rate decay style after warmup: cosine or linear (default: cosine)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                       help='Minimum learning rate as a ratio of initial LR (default: 0.01, i.e., decay to 1%% of initial LR)')
    
    # Hard negatives sampling
    parser.add_argument('--max_hard_negatives', type=int, default=7,
                       help='Maximum number of hard negatives per sample for training (default: 7). '
                            'Used to initialize the dataset. For validation, uses first N deterministically.')
    parser.add_argument('--max_num_negatives', type=int, default=-1, 
                       help='Maximum number of hard negatives to sample per query (default: -1 = use all). '
                            'If set to a positive value and fewer than available, randomly sample from hard negatives each epoch.')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for training (recommended for larger batches)')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling (default: 32)')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout (default: 0.1)')
    parser.add_argument('--lora_target_modules', type=str, 
                       default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                       help='Comma-separated list of modules to apply LoRA')
    
    # Dataset sampling arguments
    parser.add_argument('--dataset_ratios', type=str, default=None,
                       help='Comma-separated ratios for each dataset file (e.g., "1.0,0.5,2.0"). '
                            'If not provided, all datasets are shuffled together uniformly.')
    
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = args.local_rank
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0
        local_rank = 0

    # Setup logging
    logger = setup_logging(args.output_dir, rank)

    # Handle --batch_softmax_only flag
    if args.batch_softmax_only:
        if args.use_hard_negatives:
            if rank == 0:
                logger.warning("⚠️  --batch_softmax_only is enabled, ignoring --use_hard_negatives flag")
        # Override to disable hard negatives
        args.use_hard_negatives = False
    
    # Setup TensorBoard
    writer = None
    if rank == 0:
        tensorboard_dir = Path(args.output_dir) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        logger.info(f"TensorBoard logs: {tensorboard_dir}")
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(args.output_dir, keep_recent=args.keep_recent, rank=rank)
    
    if rank == 0:
        logger.info(f"Starting training with:")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Batch size per GPU: {args.batch_size}")
        logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {args.use_mixed_precision}")
        logger.info(f"  Hard negatives: {args.use_hard_negatives}")
        logger.info(f"  Save every: {args.save_steps} steps")
        logger.info(f"  Keep recent: {args.keep_recent} checkpoints")
        logger.info(f"  Data workers: {args.num_workers}")
        logger.info(f"  Random seed: {args.seed}")
    
    # Load config
    config_path = os.path.join(args.model_dir, 'cosyvoice3.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f)
    
    # Initialize frontend (not passed to dataset due to pickle issues)
    if rank == 0:
        logger.info("Frontend will be initialized per worker...")
    
    # Initialize model
    model = CosyVoice3Embedding(
        llm_config=configs['llm'],
        speech_tokenizer_path=os.path.join(args.model_dir, 'speech_tokenizer_v3.onnx')
    )
    
    # Load LLM weights
    if rank == 0:
        logger.info(f"Loading LLM weights from {args.llm_path}")
    model.load_llm(args.llm_path, strict=False)
    
    # Apply LoRA if enabled
    if args.use_lora:
        if rank == 0:
            logger.info("Applying LoRA to model...")
            logger.info(f"  LoRA rank (r): {args.lora_r}")
            logger.info(f"  LoRA alpha: {args.lora_alpha}")
            logger.info(f"  LoRA dropout: {args.lora_dropout}")
            logger.info(f"  Target modules: {args.lora_target_modules}")
        
        from peft import LoraConfig, get_peft_model
        
        # Parse target modules
        target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION"  # Use FEATURE_EXTRACTION for embedding models
        )
        
        # Apply LoRA to the LLM part of the model
        model.llm = get_peft_model(model.llm, lora_config)
        
        if rank == 0:
            logger.info("LoRA applied successfully!")
            model.llm.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory savings
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        if rank == 0:
            logger.info("Gradient checkpointing enabled")
    
    model.to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Initialize loss function
    if args.loss_type == 'infonce':
        criterion = InfoNCELoss(temperature=args.temperature, use_hard_negatives=args.use_hard_negatives)
    else:
        criterion = MultiTaskContrastiveLoss(temperature=args.temperature, use_hard_negatives=args.use_hard_negatives)    

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Load training dataset from parquet files
    # Expected structure: data_dir/train/ and data_dir/val/
    if rank == 0:
        logger.info(f"Loading datasets from parquet directory: {args.data_dir}")
    
    # Load training dataset
    train_dataset = ParquetVoiceEmbeddingDataset(
        parquet_dir=args.data_dir,
        split='train',
        datasets=None,  # Auto-discover all datasets
        use_hard_negatives=args.use_hard_negatives,
        max_negatives=args.max_hard_negatives,
        shuffle_datasets=True,
        seed=args.seed
    )
    
    if rank == 0:
        logger.info(f"Total training samples: {len(train_dataset)}")
        logger.info(f"Dataset info: {train_dataset.dataset_info}")
    
    # Load validation dataset if available
    val_dataset = None
    val_loader = None
    try:
        val_dataset = ParquetVoiceEmbeddingDataset(
            parquet_dir=args.data_dir,
            split='val',
            datasets=None,  # Auto-discover all datasets
            use_hard_negatives=args.use_hard_negatives,
            max_negatives=args.max_hard_negatives,
            shuffle_datasets=False,  # No shuffle for validation
            seed=args.seed
        )
        
        if rank == 0:
            logger.info(f"Total validation samples: {len(val_dataset)}")
    except Exception as e:
        if rank == 0:
            logger.warning(f"No validation dataset found or error loading: {e}")
    
    # Parse dataset ratios if provided (note: not applicable for single parquet dataset)
    if args.dataset_ratios:
        if rank == 0:
            logger.warning("⚠️  --dataset_ratios is not applicable when using parquet format with auto-discovery.")
            logger.warning("    If you need dataset weighting, specify individual dataset names in the code.")
    
    # Calculate total steps for scheduler
    total_steps = (len(train_dataset) // (args.batch_size * world_size * args.gradient_accumulation_steps)) * args.epochs
    
    # Initialize scheduler with linear warmup and configurable decay
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        """Learning rate schedule with linear warmup and configurable decay (cosine or linear)."""
        if current_step < args.warmup_steps:
            # Linear warmup: 0 -> 1.0 over warmup_steps
            return float(current_step) / float(max(1, args.warmup_steps))
        
        # Decay after warmup
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        
        if args.lr_decay_style == 'linear':
            # Linear decay: 1.0 -> min_lr_ratio
            return max(args.min_lr_ratio, 1.0 - (1.0 - args.min_lr_ratio) * progress)
        else:
            # Cosine decay: 1.0 -> min_lr_ratio
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(args.min_lr_ratio, args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine_decay)
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if args.use_mixed_precision else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    if args.resume_checkpoint:
        start_epoch, start_step = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler)
    
    if rank == 0:
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {args.warmup_steps} ({args.warmup_steps/total_steps*100:.1f}% of total)")
        logger.info(f"Learning rate schedule: Linear warmup -> {args.lr_decay_style.capitalize()} decay")
        logger.info(f"  Initial LR: {args.learning_rate}")
        logger.info(f"  Min LR: {args.learning_rate * args.min_lr_ratio} (ratio: {args.min_lr_ratio})")
        logger.info(f"Gradient clipping: {'Enabled (max_norm=' + str(args.max_grad_norm) + ')' if args.max_grad_norm > 0 else 'Disabled'}")
        if args.max_num_negatives > 0:
            logger.info(f"Max hard negatives: {args.max_num_negatives} (randomly sampled each epoch)")
        else:
            logger.info(f"Max hard negatives: All available (no sampling)")
    
    # Setup data loaders with shuffling
    # For distributed training, use DistributedSampler which handles shuffling per epoch
    # For single GPU, use shuffle=True in DataLoader
    train_sampler = None
    shuffle = True
    
    if world_size > 1:
        # Distributed training: use DistributedSampler with shuffling
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True  # Shuffle enabled
        )
        shuffle = False  # Disable DataLoader shuffle when using sampler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,  # Shuffle if no sampler (single GPU)
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn  # Use the new worker_init_fn
    )
    
    # Setup validation loader with epoch-based shuffling for better representativeness
    if val_dataset:
        val_sampler = None
        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True  # Enable shuffle for validation to get different batches each time
            )
        else:
            # Single GPU: use custom sampler with deterministic shuffling
            # Create a shuffled sampler that changes each epoch
            from torch.utils.data import RandomSampler
            val_sampler = RandomSampler(val_dataset, replacement=False)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,  # Don't use DataLoader shuffle when using sampler
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        if rank == 0:
            logger.info(f"Validation loader: Using shuffled sampler for better representativeness")
            logger.info(f"  Each validation run will see different {args.val_max_batches} batches (shuffled deterministically per epoch)")
    
    # Training loop
    global_step = start_step
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler to ensure different shuffling each epoch
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting Epoch {epoch}")
            logger.info(f"  Shuffling with seed: {args.seed + epoch}")
            logger.info(f"{'='*80}\n")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            epoch, args.gradient_accumulation_steps, args.use_mixed_precision, rank,
            args.save_steps, args.output_dir, global_step, writer, checkpoint_manager, val_loader, args.use_lora, args.val_max_batches,
            max_grad_norm=args.max_grad_norm
        )
        
        # Update global step
        global_step = train_losses['global_step']
        
        # Log training losses
        if rank == 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch} Training Results (Step {global_step}):")
            logger.info(f"  Total Loss: {train_losses['total_loss']:.4f}")
            logger.info(f"  Softmax Loss: {train_losses['softmax_loss']:.4f}")
            logger.info(f"  Hard Negative Loss: {train_losses['hard_negative_loss']:.4f}")
            logger.info(f"  Task-specific Losses:")
            for task_type, task_loss in train_losses['task_losses'].items():
                logger.info(f"    {task_type}: {task_loss:.4f}")
            logger.info(f"{'='*80}\n")
            
            # Log epoch-level metrics to TensorBoard
            writer.add_scalar('epoch/train_loss', train_losses['total_loss'], epoch)
        
        # Validate
        if val_loader:
            # Set epoch for validation sampler to get different batches each validation
            if hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(epoch)
            
            val_losses = validate(model, val_loader, criterion, device, args.use_mixed_precision, 
                                 rank, writer, global_step, max_batches=args.val_max_batches)
            
            if rank == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {epoch} Validation Results (Step {global_step}):")
                logger.info(f"  Total Loss: {val_losses['total_loss']:.4f}")
                logger.info(f"  Softmax Loss: {val_losses['softmax_loss']:.4f}")
                logger.info(f"  Hard Negative Loss: {val_losses['hard_negative_loss']:.4f}")
                logger.info(f"  Task-specific Losses:")
                for task_type, task_loss in val_losses['task_losses'].items():
                    logger.info(f"    {task_type}: {task_loss:.4f}")
                logger.info(f"{'='*80}\n")
                
                # Log epoch-level metrics to TensorBoard
                writer.add_scalar('epoch/val_loss', val_losses['total_loss'], epoch)
            
            # Save checkpoint at end of epoch with validation loss
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, 
                val_loss=val_losses['total_loss'], use_lora=args.use_lora
            )
        else:
            # Save checkpoint at end of epoch without validation loss
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, use_lora=args.use_lora
            )
        
        # Clear cache at end of epoch
        torch.cuda.empty_cache()
    
    if rank == 0:
        logger.info("Training completed!")
        writer.close()
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
