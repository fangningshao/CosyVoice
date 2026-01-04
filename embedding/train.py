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
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from torch.utils.tensorboard import SummaryWriter
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from embedding.model import CosyVoice3Embedding
from embedding.dataset import VoiceEmbeddingDataset, collate_fn
from embedding.loss import InfoNCELoss, MultiTaskContrastiveLoss


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
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, val_loss=None):
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
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
            logging.info(f"✓ Checkpoint saved successfully: {checkpoint_path}")
        except Exception as e:
            logging.error(f"✗ Failed to save checkpoint: {e}")
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
            
            logging.info(f"✓ New best checkpoint! Val loss: {val_loss:.4f}")
        
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
                checkpoint_manager=None):
    """Train for one epoch with detailed loss logging."""
    model.train()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    task_losses = {}
    num_batches = 0
    
    optimizer.zero_grad()
    
    # Use ncols=None to auto-detect terminal width, or set a fixed width
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0), ncols=160)
    
    for batch_idx, batch in enumerate(pbar):
        # Skip None batches (all samples failed)
        if batch is None:
            logging.warning(f"Skipping batch {batch_idx} - all samples failed to load")
            continue
        
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Forward pass with mixed precision
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
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Free memory
        del query_output, positive_output
        if negative_embeddings is not None:
            del negative_output
        
        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            
            # Log to TensorBoard
            if writer is not None and rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss_dict['total_loss'].item(), global_step)
                writer.add_scalar('train/softmax_loss', loss_dict.get('softmax_loss', 0), global_step)
                writer.add_scalar('train/hard_negative_loss', loss_dict.get('hard_negative_loss', 0), global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                # Log task-specific losses
                if 'task_losses' in loss_dict:
                    for task_type, task_loss in loss_dict['task_losses'].items():
                        writer.add_scalar(f'train/task_{task_type}', task_loss, global_step)
            
            # Save checkpoint every N steps (FIX: pass None for val_loss during training)
            if save_steps > 0 and global_step % save_steps == 0 and checkpoint_manager is not None and rank == 0:
                logging.info(f"Saving checkpoint at step {global_step}")
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, val_loss=None
                )
            
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
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'L': f"{loss_dict['total_loss'].item():.4f}",  # Loss
            'avg': f"{total_loss / num_batches:.4f}",       # Average
            'sm': f"{loss_dict.get('softmax_loss', 0):.4f}",  # Softmax
            'hn': f"{loss_dict.get('hard_negative_loss', 0):.4f}",  # Hard neg
            'lr': f"{current_lr:.6f}",
            'step': global_step
        })
    
    # Calculate average losses
    avg_total_loss = total_loss / num_batches
    avg_softmax_loss = total_softmax_loss / num_batches
    avg_hard_neg_loss = total_hard_neg_loss / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
    
    return {
        'total_loss': avg_total_loss,
        'softmax_loss': avg_softmax_loss,
        'hard_negative_loss': avg_hard_neg_loss,
        'task_losses': avg_task_losses,
        'global_step': global_step
    }


def validate(model, val_loader, criterion, device, use_mixed_precision, rank=0, writer=None, global_step=0):
    """Validate the model with detailed loss logging."""
    model.eval()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    task_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", disable=(rank != 0))
        
        for batch in pbar:
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
            
            pbar.set_postfix({'val_loss': f"{loss_dict['total_loss'].item():.4f}"})
    
    # Calculate average losses
    avg_total_loss = total_loss / num_batches
    avg_softmax_loss = total_softmax_loss / num_batches
    avg_hard_neg_loss = total_hard_neg_loss / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
    
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
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every N steps')
    parser.add_argument('--keep_recent', type=int, default=10, help='Number of recent checkpoints to keep')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
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
        logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
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
    
    # Find all training files
    train_files = sorted(glob.glob(os.path.join(args.data_dir, '**/*.train.jsonl'), recursive=True))
    val_files = sorted(glob.glob(os.path.join(args.data_dir, '**/*.val.jsonl'), recursive=True))
    
    if rank == 0:
        logger.info(f"Found {len(train_files)} training files")
        logger.info(f"Found {len(val_files)} validation files")
    
    # Load and shuffle training datasets
    train_datasets = []
    for train_file in train_files:
        dataset = VoiceEmbeddingDataset(
            train_file,
            frontend=None,
            model_dir=args.model_dir,
            use_hard_negatives=args.use_hard_negatives
        )
        train_datasets.append(dataset)
    
    train_dataset = ConcatDataset(train_datasets)
    
    if rank == 0:
        logger.info(f"Total training samples: {len(train_dataset)}")
    
    # Calculate total steps for scheduler
    total_steps = (len(train_dataset) // (args.batch_size * world_size * args.gradient_accumulation_steps)) * args.epochs
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if args.use_mixed_precision else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    if args.resume_checkpoint:
        start_epoch, start_step = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler)
    
    if rank == 0:
        logger.info(f"Total training steps: {total_steps}")
    
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
            shuffle=True,  # Shuffle enabled
            seed=args.seed  # Use seed for reproducibility
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
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id)  # Seed per worker
    )
    
    # Setup validation loader (no shuffling for validation)
    val_loader = None
    if val_files:
        val_datasets = []
        for val_file in val_files:
            dataset = VoiceEmbeddingDataset(
                val_file,
                frontend=None,
                model_dir=args.model_dir,
                use_hard_negatives=False
            )
            val_datasets.append(dataset)
        
        val_dataset = ConcatDataset(val_datasets)
        
        if rank == 0:
            logger.info(f"Total validation samples: {len(val_dataset)}")
        
        val_sampler = None
        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False  # No shuffling for validation
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,  # No shuffling for validation
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if args.num_workers > 0 else False
        )
    
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
            args.save_steps, args.output_dir, global_step, writer, checkpoint_manager
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
            val_losses = validate(model, val_loader, criterion, device, args.use_mixed_precision, rank, writer, global_step)
            
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
                val_loss=val_losses['total_loss']
            )
        else:
            # Save checkpoint at end of epoch without validation loss
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, global_step
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
