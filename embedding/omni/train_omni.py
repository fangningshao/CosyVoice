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

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embedding.model_omni import OmniEmbeddingModel
from embedding.dataset_omni import OmniEmbeddingDataset, collate_fn_omni
from embedding.loss import InfoNCELoss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_dataset_type(dataset_path: str) -> str:
    """
    Determine dataset type from the dataset path.

    Returns:
        'speakerid' if path ends with 'speakerid'
        'text2speech' if path ends with 'text2speech'
        'semantic' if path ends with 'semantic'
        'other' otherwise
    """
    dataset_path = dataset_path.lower()
    if dataset_path.endswith('speakerid'):
        return 'speakerid'
    elif dataset_path.endswith('text2speech'):
        return 'text2speech'
    elif dataset_path.endswith('semantic'):
        return 'semantic'
    return 'other'


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, global_step,
                val_loader=None, save_steps=100, output_dir=None, best_val_loss=float('inf'),
                scheduler=None, use_lora=False):
    """Train for one epoch with step-based validation and checkpointing."""
    import time  # Import at function level

    model.train()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    total_pos_sim = 0
    total_neg_sim = 0
    total_accuracy = 0
    total_mean_rank = 0
    num_batches = 0

    # Add timing metrics
    data_loading_time = 0
    gpu_forward_time = 0
    gpu_backward_time = 0
    optimizer_step_time = 0
    total_iteration_time = 0

    # Track losses by dataset type
    loss_by_type = {
        'speakerid': {'total': 0.0, 'count': 0},
        'text2speech': {'total': 0.0, 'count': 0},
        'semantic': {'total': 0.0, 'count': 0},
        'other': {'total': 0.0, 'count': 0}
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    iter_start_time = time.time()

    for batch_idx, batch in enumerate(pbar):
        # Measure data loading time (time to get batch from dataloader)
        batch_ready_time = time.time()
        data_load_time = batch_ready_time - iter_start_time
        data_loading_time += data_load_time

        # Get dataset paths for this batch
        dataset_paths = batch.get('dataset_paths', [])

        # Move tensors to device
        device_transfer_start = time.time()
        query_input_ids = batch['query_input_ids'].to(device)
        query_attention_mask = batch['query_attention_mask'].to(device)
        query_audio_values = batch['query_audio_values'].to(device)

        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        positive_audio_values = batch['positive_audio_values'].to(device)

        # Handle optional negatives
        negative_input_ids = batch.get('negative_input_ids', None)
        negative_attention_mask = batch.get('negative_attention_mask', None)
        negative_audio_values = batch.get('negative_audio_values', None)
        negative_counts = batch.get('negative_counts', None)

        if negative_input_ids is not None:
            negative_input_ids = negative_input_ids.to(device)
            negative_attention_mask = negative_attention_mask.to(device)
            negative_audio_values = negative_audio_values.to(device)

        device_transfer_time = time.time() - device_transfer_start

        # Extract embeddings using pre-tokenized inputs
        forward_start_time = time.time()
        anchor_output = model.forward_from_tensors(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            audio_values=query_audio_values
        )
        positive_output = model.forward_from_tensors(
            input_ids=positive_input_ids,
            attention_mask=positive_attention_mask,
            audio_values=positive_audio_values
        )

        # Extract embeddings from output dictionaries
        anchor_emb = anchor_output['embedding']
        positive_emb = positive_output['embedding']

        # Extract negative embeddings if provided
        negative_emb = None
        if negative_input_ids is not None and negative_input_ids.size(0) > 0:
            negative_output = model.forward_from_tensors(
                input_ids=negative_input_ids,
                attention_mask=negative_attention_mask,
                audio_values=negative_audio_values
            )
            negative_emb = negative_output['embedding']

        # Convert negative_counts to tensor if provided
        if negative_counts is not None:
            negative_counts = torch.tensor(negative_counts, device=device)

        # Compute loss using InfoNCE
        loss_dict = criterion(
            query_embeddings=anchor_emb,
            positive_embeddings=positive_emb,
            negative_embeddings=negative_emb,
            negative_counts=negative_counts
        )

        loss = loss_dict['total_loss']

        forward_time = time.time() - forward_start_time
        gpu_forward_time += forward_time

        # Backward
        backward_start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        backward_time = time.time() - backward_start_time
        gpu_backward_time += backward_time

        optim_start_time = time.time()
        optimizer.step()
        optim_time = time.time() - optim_start_time
        optimizer_step_time += optim_time

        # Compute similarities for logging
        with torch.no_grad():
            anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
            positive_norm = F.normalize(positive_emb, p=2, dim=1)
            pos_sim = (anchor_norm * positive_norm).sum(dim=1).mean().item()

            # Compute negative similarity (in-batch)
            batch_size = anchor_norm.size(0)
            if batch_size > 1:
                # Use off-diagonal elements as negatives
                all_sims = torch.matmul(anchor_norm, positive_norm.T)
                mask = torch.eye(batch_size, device=device).bool()
                all_sims = all_sims.masked_fill(mask, float('-inf'))
                neg_sim = all_sims.max(dim=1)[0].mean().item()
            else:
                neg_sim = 0.0

        # Track losses by dataset type
        if dataset_paths:
            # Get the dataset type for this batch (use first sample's path as representative)
            dataset_type = get_dataset_type(dataset_paths[0])
            loss_by_type[dataset_type]['total'] += loss.item()
            loss_by_type[dataset_type]['count'] += 1

        # Update metrics
        total_loss += loss.item()
        total_softmax_loss += loss_dict['softmax_loss']
        total_hard_neg_loss += loss_dict['hard_negative_loss']
        total_pos_sim += pos_sim
        total_neg_sim += neg_sim
        total_accuracy += loss_dict.get('accuracy', 0)
        total_mean_rank += loss_dict.get('mean_rank', 0)
        num_batches += 1
        global_step += 1

        # Update progress bar
        avg_loss = total_loss / num_batches
        avg_data_time = data_loading_time / num_batches
        avg_gpu_time = (gpu_forward_time + gpu_backward_time) / num_batches

        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'softmax': f'{loss_dict["softmax_loss"]:.4f}',
            'hard_neg': f'{loss_dict["hard_negative_loss"]:.4f}',
            'pos_sim': f'{pos_sim:.3f}',
            'neg_sim': f'{neg_sim:.3f}',
            'acc': f'{loss_dict.get("accuracy", 0):.3f}',
            'rank': f'{loss_dict.get("mean_rank", 0):.1f}',
            'step': global_step,
            'data_ms': f"{avg_data_time*1000:.0f}",
            'gpu_ms': f"{avg_gpu_time*1000:.0f}",
            'ratio': f"{avg_data_time/avg_gpu_time:.1f}x"
        })

        # Log to TensorBoard
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/SoftmaxLoss', loss_dict['softmax_loss'], global_step)
        writer.add_scalar('Train/HardNegLoss', loss_dict['hard_negative_loss'], global_step)
        writer.add_scalar('Train/Positive_Similarity', pos_sim, global_step)
        writer.add_scalar('Train/Negative_Similarity', neg_sim, global_step)
        writer.add_scalar('Train/Accuracy', loss_dict.get('accuracy', 0), global_step)
        writer.add_scalar('Train/Mean_Rank', loss_dict.get('mean_rank', 0), global_step)
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)

        # Log timing metrics (every 50 steps)
        if global_step % 50 == 0:
            avg_data_time = data_loading_time / num_batches
            avg_forward_time = gpu_forward_time / num_batches
            avg_backward_time = gpu_backward_time / num_batches
            total_gpu_time = avg_forward_time + avg_backward_time

            writer.add_scalar('timing/data_loading_ms', avg_data_time * 1000, global_step)
            writer.add_scalar('timing/gpu_forward_ms', avg_forward_time * 1000, global_step)
            writer.add_scalar('timing/gpu_backward_ms', avg_backward_time * 1000, global_step)
            writer.add_scalar('timing/total_gpu_ms', total_gpu_time * 1000, global_step)
            writer.add_scalar('timing/data_to_gpu_ratio', avg_data_time / (total_gpu_time + 1e-6), global_step)

        # Log losses by dataset type to TensorBoard
        if dataset_paths:
            dataset_type = get_dataset_type(dataset_paths[0])
            if dataset_type == 'speakerid':
                writer.add_scalar('Train/Loss_SpeakerID', loss.item(), global_step)
            elif dataset_type == 'text2speech':
                writer.add_scalar('Train/Loss_Text2Speech', loss.item(), global_step)
            elif dataset_type == 'semantic':
                writer.add_scalar('Train/Loss_Semantic', loss.item(), global_step)

        # Print detailed timing info every 50 batches
        if (batch_idx + 1) % 50 == 0:
            avg_data_time = data_loading_time / num_batches
            avg_forward_time = gpu_forward_time / num_batches
            avg_backward_time = gpu_backward_time / num_batches
            avg_optim_time = optimizer_step_time / num_batches
            total_gpu_time = avg_forward_time + avg_backward_time

            logger.info(f"\n{'='*80}")
            logger.info(f"#  TIMING ANALYSIS (Batch {batch_idx + 1}/{len(dataloader)})")
            logger.info(f"{'='*80}")
            logger.info(f"Data Loading:     {avg_data_time*1000:>8.1f} ms  ({avg_data_time/(avg_data_time+total_gpu_time+avg_optim_time)*100:>5.1f}%)")
            logger.info(f"GPU Forward:      {avg_forward_time*1000:>8.1f} ms  ({avg_forward_time/(avg_data_time+total_gpu_time+avg_optim_time)*100:>5.1f}%)")
            logger.info(f"GPU Backward:     {avg_backward_time*1000:>8.1f} ms  ({avg_backward_time/(avg_data_time+total_gpu_time+avg_optim_time)*100:>5.1f}%)")
            logger.info(f"Optimizer Step:   {avg_optim_time*1000:>8.1f} ms  ({avg_optim_time/(avg_data_time+total_gpu_time+avg_optim_time)*100:>5.1f}%)")
            logger.info(f"Device Transfer:  {device_transfer_time*1000:>8.1f} ms  (last batch)")
            logger.info(f"─────────────────────────────────────────")
            logger.info(f"Total GPU Time:   {total_gpu_time*1000:>8.1f} ms")
            logger.info(f"Total Iter Time:  {(data_loading_time/num_batches + total_gpu_time + avg_optim_time)*1000:>8.1f} ms")
            logger.info(f"")
            logger.info(f"# Data-to-GPU Ratio: {avg_data_time/total_gpu_time:.2f}x")
            if avg_data_time > total_gpu_time:
                logger.warning(f"#  DATA LOADING IS BLOCKING GPU! ({avg_data_time/total_gpu_time:.1f}x slower)")
                logger.warning(f"   Suggestions:")
                logger.warning(f"   • Increase --num_workers (currently: {dataloader.num_workers})")
                logger.warning(f"   • Use --batch_softmax_only to skip loading hard negatives")
                logger.warning(f"   • Check disk I/O (SSD vs HDD)")
                logger.warning(f"   • Verify audio files are not corrupted/slow to read")
            else:
                logger.info(f"@ GPU is well-fed by data pipeline")
            logger.info(f"{'='*80}\n")

        # Step-based validation and checkpointing
        if save_steps > 0 and global_step % save_steps == 0:
            logger.info(f"\n>>> Step {global_step}: Running validation and saving checkpoint...")

            # Log averaged losses by dataset type
            for dtype in ['speakerid', 'text2speech', 'semantic']:
                if loss_by_type[dtype]['count'] > 0:
                    avg_loss_type = loss_by_type[dtype]['total'] / loss_by_type[dtype]['count']
                    logger.info(f"  {dtype} loss: {avg_loss_type:.4f} ({loss_by_type[dtype]['count']} batches)")

            # Run validation if val_loader is provided
            if val_loader is not None:
                val_loss, val_metrics = validate_step(
                    model, val_loader, criterion, device, epoch, writer, global_step
                )
                logger.info(f"Step {global_step} - Val Loss: {val_loss:.4f}, "
                           f"Pos Sim: {val_metrics['pos_sim']:.3f}, "
                           f"Neg Sim: {val_metrics['neg_sim']:.3f}, "
                           f"Acc: {val_metrics['accuracy']:.3f}, "
                           f"Mean Rank: {val_metrics['mean_rank']:.1f}")

                # Log validation metrics to TensorBoard
                writer.add_scalar('Validation/Loss', val_loss, global_step)
                writer.add_scalar('Validation/SoftmaxLoss', val_metrics['softmax_loss'], global_step)
                writer.add_scalar('Validation/HardNegLoss', val_metrics['hard_negative_loss'], global_step)
                writer.add_scalar('Validation/Positive_Similarity', val_metrics['pos_sim'], global_step)
                writer.add_scalar('Validation/Negative_Similarity', val_metrics['neg_sim'], global_step)
                writer.add_scalar('Validation/Accuracy', val_metrics['accuracy'], global_step)
                writer.add_scalar('Validation/Mean_Rank', val_metrics['mean_rank'], global_step)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if output_dir is not None:
                        checkpoint_path = output_dir / 'best_model.pt'
                        save_checkpoint(
                            model, optimizer, scheduler, epoch,
                            checkpoint_path, val_loss, val_metrics, use_lora, global_step
                        )
                        logger.info(f"@ New best model saved at step {global_step} with val_loss={val_loss:.4f}")

                # Switch back to training mode
                model.train()

            # Save step checkpoint
            if output_dir is not None:
                checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
                current_metrics = {
                    'total_loss': avg_loss,
                    'softmax_loss': total_softmax_loss / num_batches,
                    'hard_negative_loss': total_hard_neg_loss / num_batches,
                    'pos_sim': total_pos_sim / num_batches,
                    'neg_sim': total_neg_sim / num_batches
                }
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    checkpoint_path, avg_loss, current_metrics, use_lora, global_step
                )
                logger.info(f"@ Checkpoint saved at step {global_step}")

        # Measure total iteration time
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        total_iteration_time += iter_time

        # Start timing next iteration
        iter_start_time = time.time()

    # Log final averaged losses by dataset type
    logger.info(f"\nEpoch {epoch} - Losses by dataset type:")
    for dtype in ['speakerid', 'text2speech', 'semantic']:
        if loss_by_type[dtype]['count'] > 0:
            avg_loss_type = loss_by_type[dtype]['total'] / loss_by_type[dtype]['count']
            logger.info(f"  l_{dtype[:3] if dtype != 'text2speech' else 't2s'}: {avg_loss_type:.4f} ({loss_by_type[dtype]['count']} batches)")
            # Log epoch-level averages to TensorBoard
            writer.add_scalar(f'Train_Epoch/Loss_{dtype.capitalize()}', avg_loss_type, epoch)

    # Log final timing summary
    if num_batches > 0:
        avg_data_time = data_loading_time / num_batches
        avg_forward_time = gpu_forward_time / num_batches
        avg_backward_time = gpu_backward_time / num_batches
        total_gpu_time = avg_forward_time + avg_backward_time

        logger.info(f"\n{'='*80}")
        logger.info(f"# EPOCH {epoch} TIMING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Average Data Loading Time:  {avg_data_time*1000:.1f} ms/batch")
        logger.info(f"Average GPU Forward Time:   {avg_forward_time*1000:.1f} ms/batch")
        logger.info(f"Average GPU Backward Time:  {avg_backward_time*1000:.1f} ms/batch")
        logger.info(f"Average Total GPU Time:     {total_gpu_time*1000:.1f} ms/batch")
        logger.info(f"")
        logger.info(f"Data-to-GPU Ratio: {avg_data_time/total_gpu_time:.2f}x")
        if avg_data_time > total_gpu_time * 1.5:
            logger.warning(f"❌ SEVERE DATA LOADING BOTTLENECK DETECTED!")
        elif avg_data_time > total_gpu_time:
            logger.warning(f"⚠️  Data loading is slower than GPU processing")
        else:
            logger.info(f"@ Data pipeline is keeping up with GPU")
        logger.info(f"{'='*80}\n")

    # Average metrics
    metrics = {
        'total_loss': total_loss / num_batches,
        'softmax_loss': total_softmax_loss / num_batches,
        'hard_negative_loss': total_hard_neg_loss / num_batches,
        'pos_sim': total_pos_sim / num_batches,
        'neg_sim': total_neg_sim / num_batches,
        'accuracy': total_accuracy / num_batches,
        'mean_rank': total_mean_rank / num_batches
    }

    return metrics['total_loss'], metrics, global_step, best_val_loss


def validate_step(model, dataloader, criterion, device, epoch, writer, global_step):
    """Quick validation without TensorBoard per-step logging (used for step-based validation)."""
    model.eval()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    total_pos_sim = 0
    total_neg_sim = 0
    total_accuracy = 0
    total_mean_rank = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move tensors to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            query_audio_values = batch['query_audio_values'].to(device)

            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            positive_audio_values = batch['positive_audio_values'].to(device)

            # Handle optional negatives
            negative_input_ids = batch.get('negative_input_ids', None)
            negative_attention_mask = batch.get('negative_attention_mask', None)
            negative_audio_values = batch.get('negative_audio_values', None)
            negative_counts = batch.get('negative_counts', None)

            if negative_input_ids is not None:
                negative_input_ids = negative_input_ids.to(device)
                negative_attention_mask = negative_attention_mask.to(device)
                negative_audio_values = negative_audio_values.to(device)

            # Extract embeddings using pre-tokenized inputs
            anchor_output = model.forward_from_tensors(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                audio_values=query_audio_values
            )
            positive_output = model.forward_from_tensors(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                audio_values=positive_audio_values
            )

            # Extract embeddings from output dictionaries
            anchor_emb = anchor_output['embedding']
            positive_emb = positive_output['embedding']

            # Extract negative embeddings if provided
            negative_emb = None
            if negative_input_ids is not None and negative_input_ids.size(0) > 0:
                negative_output = model.forward_from_tensors(
                    input_ids=negative_input_ids,
                    attention_mask=negative_attention_mask,
                    audio_values=negative_audio_values
                )
                negative_emb = negative_output['embedding']

            # Convert negative_counts to tensor if provided
            if negative_counts is not None:
                negative_counts = torch.tensor(negative_counts, device=device)

            # Compute loss
            loss_dict = criterion(
                query_embeddings=anchor_emb,
                positive_embeddings=positive_emb,
                negative_embeddings=negative_emb,
                negative_counts=negative_counts
            )

            loss = loss_dict['total_loss']

            # Compute similarities
            anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
            positive_norm = F.normalize(positive_emb, p=2, dim=1)
            pos_sim = (anchor_norm * positive_norm).sum(dim=1).mean().item()

            # Compute negative similarity (in-batch)
            batch_size = anchor_norm.size(0)
            if batch_size > 1:
                all_sims = torch.matmul(anchor_norm, positive_norm.T)
                mask = torch.eye(batch_size, device=device).bool()
                all_sims = all_sims.masked_fill(mask, float('-inf'))
                neg_sim = all_sims.max(dim=1)[0].mean().item()
            else:
                neg_sim = 0.0

            # Accumulate metrics
            total_loss += loss.item()
            total_softmax_loss += loss_dict['softmax_loss']
            total_hard_neg_loss += loss_dict['hard_negative_loss']
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            total_accuracy += loss_dict.get('accuracy', 0) * batch_size  # Weighted by batch size
            total_mean_rank += loss_dict.get('mean_rank', 0) * batch_size  # Weighted by batch size
            num_batches += batch_size

            # Log intermediate metrics to TensorBoard
            writer.add_scalar('Validation/Loss', loss.item(), global_step)
            writer.add_scalar('Validation/Positive_Similarity', pos_sim, global_step)
            writer.add_scalar('Validation/Negative_Similarity', neg_sim, global_step)
            writer.add_scalar('Validation/Accuracy', loss_dict.get('accuracy', 0), global_step)
            writer.add_scalar('Validation/Mean_Rank', loss_dict.get('mean_rank', 0), global_step)
            global_step += 1

    # Compute averaged metrics
    metrics = {
        'total_loss': total_loss / num_batches,
        'softmax_loss': total_softmax_loss / num_batches,
        'hard_negative_loss': total_hard_neg_loss / num_batches,
        'pos_sim': total_pos_sim / num_batches,
        'neg_sim': total_neg_sim / num_batches,
        'accuracy': total_accuracy / num_batches,  # Average accuracy
        'mean_rank': total_mean_rank / num_batches  # Average mean rank
    }

    return metrics['total_loss'], metrics, global_step


def validate(model, dataloader, criterion, device, epoch, writer, global_step):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_softmax_loss = 0
    total_hard_neg_loss = 0
    total_pos_sim = 0
    total_neg_sim = 0
    total_accuracy = 0
    total_mean_rank = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move tensors to device
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            query_audio_values = batch['query_audio_values'].to(device)

            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            positive_audio_values = batch['positive_audio_values'].to(device)

            # Handle optional negatives
            negative_input_ids = batch.get('negative_input_ids', None)
            negative_attention_mask = batch.get('negative_attention_mask', None)
            negative_audio_values = batch.get('negative_audio_values', None)
            negative_counts = batch.get('negative_counts', None)

            if negative_input_ids is not None:
                negative_input_ids = negative_input_ids.to(device)
                negative_attention_mask = negative_attention_mask.to(device)
                negative_audio_values = negative_audio_values.to(device)

            # Extract embeddings using pre-tokenized inputs
            anchor_output = model.forward_from_tensors(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                audio_values=query_audio_values
            )
            positive_output = model.forward_from_tensors(
                input_ids=positive_input_ids,
                attention_mask=positive_attention_mask,
                audio_values=positive_audio_values
            )

            # Extract embeddings from output dictionaries
            anchor_emb = anchor_output['embedding']
            positive_emb = positive_output['embedding']

            # Extract negative embeddings if provided
            negative_emb = None
            if negative_input_ids is not None and negative_input_ids.size(0) > 0:
                negative_output = model.forward_from_tensors(
                    input_ids=negative_input_ids,
                    attention_mask=negative_attention_mask,
                    audio_values=negative_audio_values
                )
                negative_emb = negative_output['embedding']

            # Convert negative_counts to tensor if provided
            if negative_counts is not None:
                negative_counts = torch.tensor(negative_counts, device=device)

            # Compute loss
            loss_dict = criterion(
                query_embeddings=anchor_emb,
                positive_embeddings=positive_emb,
                negative_embeddings=negative_emb,
                negative_counts=negative_counts
            )

            loss = loss_dict['total_loss']

            # Compute similarities
            anchor_norm = F.normalize(anchor_emb, p=2, dim=1)
            positive_norm = F.normalize(positive_emb, p=2, dim=1)
            pos_sim = (anchor_norm * positive_norm).sum(dim=1).mean().item()

            # Compute negative similarity (in-batch)
            batch_size = anchor_norm.size(0)
            if batch_size > 1:
                all_sims = torch.matmul(anchor_norm, positive_norm.T)
                mask = torch.eye(batch_size, device=device).bool()
                all_sims = all_sims.masked_fill(mask, float('-inf'))
                neg_sim = all_sims.max(dim=1)[0].mean().item()
            else:
                neg_sim = 0.0

            total_loss += loss.item()
            total_softmax_loss += loss_dict['softmax_loss']
            total_hard_neg_loss += loss_dict['hard_negative_loss']
            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            total_accuracy += loss_dict.get('accuracy', 0.0)
            total_mean_rank += loss_dict.get('mean_rank', 0.0)
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})

            # Log to TensorBoard
            writer.add_scalar('Validation/Loss', loss.item(), global_step)
            writer.add_scalar('Validation/Positive_Similarity', pos_sim, global_step)
            writer.add_scalar('Validation/Negative_Similarity', neg_sim, global_step)
            writer.add_scalar('Validation/Accuracy', loss_dict.get('accuracy', 0), global_step)
            writer.add_scalar('Validation/Mean_Rank', loss_dict.get('mean_rank', 0), global_step)
            global_step += 1

    metrics = {
        'total_loss': total_loss / num_batches,
        'softmax_loss': total_softmax_loss / num_batches,
        'hard_negative_loss': total_hard_neg_loss / num_batches,
        'pos_sim': total_pos_sim / num_batches,
        'neg_sim': total_neg_sim / num_batches,
        'accuracy': total_accuracy / num_batches,
        'mean_rank': total_mean_rank / num_batches
    }

    return metrics['total_loss'], metrics, global_step


def main():
    parser = argparse.ArgumentParser(description='Train Omni Embedding Model')

    # Data
    parser.add_argument('--train_data', type=str, required=True,
                       help='Training data JSON file')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Validation data JSON file')
    parser.add_argument('--audio_root', type=str, default='',
                       help='Root directory for audio files')

    # Model
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Omni-3B',
                       help='Pretrained Omni model name or path')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Output embedding dimension')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze the backbone model (ignored if use_lora=True)')
    parser.add_argument('--pooling_mode', type=str, default='mean',
                       choices=['mean', 'last'])

    # LoRA parameters
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA fine-tuning (recommended for memory efficiency)')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank (r parameter)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha scaling parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='InfoNCE loss temperature')
    parser.add_argument('--use_hard_negatives', action='store_true',
                       help='Use hard negatives from data')
    parser.add_argument('--use_expanded_negatives', action='store_true', default=True,
                       help='Use query-to-query and positive-to-positive as additional negatives (KaLM-style)')
    parser.add_argument('--save_steps', type=int, default=100,
                       help='Steps interval for saving checkpoints')

    # Dataset parameters
    parser.add_argument('--max_negatives', type=int, default=7,
                       help='Maximum number of hard negatives to use')
    parser.add_argument('--max_duration', type=float, default=30.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--min_duration', type=float, default=1.0,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--skip_prefilter', action='store_true', default=True,
                       help='Skip duration/existence pre-filtering')
    parser.add_argument('--use_audio_in_query', action='store_true', default=True,
                       help='Whether query includes audio')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for dataset')
    parser.add_argument('--path_mapping', type=str, action='append', default=None,
                       help='Path mapping in format "from_path::to_path" (e.g., "D:\\data\\::/workspace/data/"). '
                            'Can be specified multiple times. All backslashes will be converted to forward slashes.')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Checkpoint saving interval (epochs)')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from (contains LoRA weights only, base LLM loaded separately)')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lora_suffix = '_lora' if args.use_lora else ''
    output_dir = Path(args.output_dir) / f'omni_embedding{lora_suffix}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir)

    # Create datasets
    logger.info("Creating datasets...")
    
    # Parse path mappings from command-line arguments
    path_mappings = []
    if args.path_mapping:
        logger.info("Path mappings configured:")
        for mapping in args.path_mapping:
            if '::' in mapping:
                from_path, to_path = mapping.split('::', 1)
                path_mappings.append((from_path, to_path))
                logger.info(f"  '{from_path}' -> '{to_path}'")
            else:
                logger.warning(f"Invalid path mapping format (expected 'from::to'): {mapping}")
    
    train_dataset = OmniEmbeddingDataset(
        data_list_file=args.train_data,
        model_dir=args.model_path,
        use_hard_negatives=args.use_hard_negatives,
        max_negatives=args.max_negatives,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        skip_prefilter=args.skip_prefilter,
        use_audio_in_query=args.use_audio_in_query,
        random_seed=args.random_seed,
        path_mappings=path_mappings
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_omni,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = None
    if args.val_data:
        val_dataset = OmniEmbeddingDataset(
            data_list_file=args.val_data,
            model_dir=args.model_path,
            use_hard_negatives=args.use_hard_negatives,
            max_negatives=args.max_negatives,
            max_duration=args.max_duration,
            min_duration=args.min_duration,
            skip_prefilter=args.skip_prefilter,
            use_audio_in_query=args.use_audio_in_query,
            random_seed=args.random_seed,
            path_mappings=path_mappings
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_omni,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )

    # Create model with LoRA support
    model = OmniEmbeddingModel(
        model_path=args.model_path,
        pooling_mode=args.pooling_mode,
        embedding_dim=args.embedding_dim,
        freeze_backbone=args.freeze_backbone,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled")
    model = model.to(device)

    # Loss and optimizer - Use InfoNCELoss like train.py
    criterion = InfoNCELoss(
        temperature=args.temperature,
        use_hard_negatives=args.use_hard_negatives,
        use_expanded_negatives=args.use_expanded_negatives
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    logger.info(f"Loss configuration:")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Use hard negatives: {args.use_hard_negatives}")
    logger.info(f"  - Use expanded negatives: {args.use_expanded_negatives}")

    # Load checkpoint if resuming training
    start_epoch = 1
    global_step = 0
    best_val_loss = float('inf')

    if args.resume_checkpoint:
        logger.info(f"\n{'='*60}")
        logger.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
        logger.info(f"{'='*60}")
        logger.info("Note: Base LLM weights are already loaded. Loading additional LoRA/projection weights from checkpoint...")

        checkpoint_info = load_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        # Resume from next epoch
        start_epoch = checkpoint_info['epoch'] + 1
        global_step = checkpoint_info.get('global_step', 0) or 0
        best_val_loss = checkpoint_info.get('loss', float('inf'))

        # Verify LoRA setting matches
        if checkpoint_info.get('use_lora', False) != args.use_lora:
            raise RuntimeError(
                f"LoRA setting mismatch: checkpoint use_lora={checkpoint_info.get('use_lora')}, "
                f"but current args.use_lora={args.use_lora}. "
                f"Please use the same --use_lora setting as when the checkpoint was created."
            )

        logger.info(f"Resuming from epoch {start_epoch}, global_step {global_step}")
        logger.info(f"Best validation loss so far: {best_val_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train
        train_loss, train_metrics, global_step, best_val_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer, global_step,
            val_loader=val_loader, save_steps=args.save_steps, output_dir=output_dir,
            best_val_loss=best_val_loss, scheduler=scheduler, use_lora=args.use_lora
        )

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics:")
        logger.info(f"  - Softmax Loss: {train_metrics['softmax_loss']:.4f}")
        logger.info(f"  - Hard Negative Loss: {train_metrics['hard_negative_loss']:.4f}")
        logger.info(f"  - Positive Similarity: {train_metrics['pos_sim']:.3f}")
        logger.info(f"  - Negative Similarity: {train_metrics['neg_sim']:.3f}")

        # Validate
        if val_loader:
            val_loss, val_metrics, global_step = validate(
                model, val_loader, criterion, device, epoch, writer, global_step
            )
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics:")
            logger.info(f"  - Softmax Loss: {val_metrics['softmax_loss']:.4f}")
            logger.info(f"  - Hard Negative Loss: {val_metrics['hard_negative_loss']:.4f}")
            logger.info(f"  - Positive Similarity: {val_metrics['pos_sim']:.3f}")
            logger.info(f"  - Negative Similarity: {val_metrics['neg_sim']:.3f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / 'best_model.pt'
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    checkpoint_path, val_loss, val_metrics, args.use_lora
                )
                logger.info(f"Saved best model to {checkpoint_path}")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                checkpoint_path, train_loss, train_metrics, args.use_lora
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Update scheduler
        scheduler.step()
        logger.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    writer.close()
    logger.info(f"\nTraining complete! Models saved to {output_dir}")


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path,
                   loss, metrics, use_lora, global_step=None):
    """
    Save checkpoint with proper handling of LoRA parameters.

    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        checkpoint_path: Path to save checkpoint
        loss: Loss value
        metrics: Training metrics
        use_lora: Whether LoRA is being used
        global_step: Current global step (optional)
    """
    logger.info(f"Saving checkpoint to: {checkpoint_path}")

    try:
        # Prepare state dict for saving
        if use_lora:
            # Only save LoRA parameters (trainable parameters)
            logger.info("Extracting LoRA weights only (not saving frozen base model parameters)...")

            # Get the PEFT model state dict (includes only LoRA adapters)
            from peft import get_peft_model_state_dict

            # Extract LoRA state dict from the wrapped model
            lora_state_dict = get_peft_model_state_dict(model.model)

            # Create a clean state dict with proper key names
            clean_state_dict = {}
            for key, value in model.state_dict().items():
                if key.startswith('model.'):
                    # Only save LoRA weights (skip base_layer weights)
                    if 'lora_' in key:
                        # Remove 'base_model.model.' prefix from PEFT wrapper if present
                        clean_key = key.replace('model.base_model.model.', 'model.')
                        clean_state_dict[clean_key] = value
                else:
                    # Save non-model parts (like embedding_proj, layer_norm)
                    clean_state_dict[key] = value

            state_dict_to_save = clean_state_dict

            logger.info(f"@ Extracted {len(state_dict_to_save)} LoRA parameter tensors")
            logger.info(f"  (Frozen base model parameters not saved - reducing checkpoint size)")
        else:
            # Save full model state dict (standard training without LoRA)
            state_dict_to_save = model.state_dict()
            logger.info(f"Saving full model state dict with {len(state_dict_to_save)} parameters")

        # If you want to examine all parameters being saved (emb proj, layer norm, and all lora params), see below
        logger.debug("ALL PARAMETERS TO SAVE:")
        for param_name, param_tensor in state_dict_to_save.items():
            logger.debug(f"  - {param_name}: {param_tensor.size()}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'use_lora': use_lora,  # Store whether LoRA was used
            'global_step': global_step,  # Store global step if provided
            'config': {}  # Add config if needed
        }, checkpoint_path)

        logger.info(f"@ Checkpoint saved successfully: {checkpoint_path}")

        # Log checkpoint size
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Checkpoint size: {checkpoint_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"✗ Failed to save checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load checkpoint with proper handling of LoRA parameters.

    The checkpoint only contains LoRA weights and projection layers (not the full LLM).
    The base LLM is already loaded in the model, so we only need to load the additional
    trained weights on top.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model (with base LLM already loaded)
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        device: Device to load tensors to

    Returns:
        dict with 'epoch', 'global_step', 'loss', 'metrics', 'use_lora'
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get checkpoint metadata
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    loss = checkpoint.get('loss', float('inf'))
    metrics = checkpoint.get('metrics', {})
    use_lora = checkpoint.get('use_lora', False)

    logger.info(f"Checkpoint info: epoch={epoch}, global_step={global_step}, loss={loss:.4f}, use_lora={use_lora}")

    # Load model state dict
    checkpoint_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()

    logger.info(f"Checkpoint contains {len(checkpoint_state_dict)} parameter tensors")
    logger.info(f"Model has {len(model_state_dict)} parameter tensors")

    # Verify all checkpoint keys exist in model (strict matching for LoRA weights)
    missing_in_model = []
    shape_mismatches = []
    matched_keys = []

    for ckpt_key, ckpt_tensor in checkpoint_state_dict.items():
        if ckpt_key not in model_state_dict:
            missing_in_model.append(ckpt_key)
        elif model_state_dict[ckpt_key].shape != ckpt_tensor.shape:
            shape_mismatches.append(
                f"{ckpt_key}: checkpoint {ckpt_tensor.shape} vs model {model_state_dict[ckpt_key].shape}"
            )
        else:
            matched_keys.append(ckpt_key)

    # Report findings
    logger.info(f"Matched keys: {len(matched_keys)}")

    if missing_in_model:
        logger.error(f"ERROR: {len(missing_in_model)} keys in checkpoint not found in model:")
        for key in missing_in_model:
            logger.error(f"  - {key}")
        raise RuntimeError(
            f"Checkpoint contains {len(missing_in_model)} keys not found in model. "
            f"This likely means the model architecture has changed or the checkpoint is incompatible. "
            f"Missing keys: {missing_in_model[:5]}{'...' if len(missing_in_model) > 5 else ''}"
        )

    if shape_mismatches:
        logger.error(f"ERROR: {len(shape_mismatches)} keys have shape mismatches:")
        for mismatch in shape_mismatches:
            logger.error(f"  - {mismatch}")
        raise RuntimeError(
            f"Checkpoint contains {len(shape_mismatches)} keys with shape mismatches. "
            f"This likely means the model configuration (e.g., lora_rank, embedding_dim) has changed. "
            f"Mismatches: {shape_mismatches[:3]}{'...' if len(shape_mismatches) > 3 else ''}"
        )

    # All checks passed - load the state dict
    logger.info("All checkpoint keys matched successfully. Loading weights...")

    # Use strict=False because model has additional base LLM weights not in checkpoint
    # But we've already verified all checkpoint keys exist in model
    load_result = model.load_state_dict(checkpoint_state_dict, strict=False)

    # Log what was not loaded from model (expected: base LLM weights)
    if load_result.missing_keys:
        # These are expected - they are the frozen base model weights
        lora_missing = [k for k in load_result.missing_keys if 'lora_' in k]
        if lora_missing:
            logger.warning(f"WARNING: {len(lora_missing)} LoRA keys were not loaded:")
            for key in lora_missing[:10]:
                logger.warning(f"  - {key}")
            if len(lora_missing) > 10:
                logger.warning(f"  ... and {len(lora_missing) - 10} more")
        else:
            logger.info(f"Note: {len(load_result.missing_keys)} base model keys not in checkpoint (expected)")

    if load_result.unexpected_keys:
        logger.warning(f"WARNING: Unexpected keys in checkpoint: {load_result.unexpected_keys}")

    logger.info(f"Successfully loaded {len(matched_keys)} parameter tensors from checkpoint")

    # Optionally restore optimizer and scheduler states
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Restored optimizer state")
        except Exception as e:
            logger.warning(f"Could not restore optimizer state: {e}")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Restored scheduler state")
        except Exception as e:
            logger.warning(f"Could not restore scheduler state: {e}")

    return {
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        'metrics': metrics,
        'use_lora': use_lora
    }


if __name__ == '__main__':
    main()
