"""
Evaluate accuracy of trained embedding checkpoint on validation sets.

This script computes accuracy where each example is considered a True Positive (TP)
only when the positive sample has the highest similarity score compared to all
negative samples provided with that example.

Usage:
    python embedding/eval_accuracy.py \
        --checkpoint path/to/checkpoint-step2600.pt \
        --model_dir path/to/CosyVoice3-0.5B-2512 \
        --val_dir path/to/val \
        --batch_size 8 \
        --max_batches 32
"""

import sys
sys.path.append('.')

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from hyperpyyaml import load_hyperpyyaml

from embedding.model import CosyVoice3Embedding
from embedding.parquet_dataset import ParquetVoiceEmbeddingDataset, collate_fn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device) -> Dict:
    """
    Load checkpoint and return metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint metadata
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get checkpoint state dict
    if 'model_state_dict' in checkpoint:
        checkpoint_state = checkpoint['model_state_dict']
    else:
        checkpoint_state = checkpoint
    
    # Remap checkpoint keys to match current model structure
    remapped_state = {}
    unmapped_keys = []
    
    logger.info("Remapping checkpoint keys to match model structure...")
    
    for ckpt_key, ckpt_value in checkpoint_state.items():
        if ckpt_key in model.state_dict():
            remapped_state[ckpt_key] = ckpt_value
        elif ckpt_key.startswith('llm.llm.'):
            new_key = ckpt_key.replace('llm.llm.', 'llm.base_model.model.llm.')
            if new_key in model.state_dict():
                remapped_state[new_key] = ckpt_value
            else:
                unmapped_keys.append((ckpt_key, new_key))
        else:
            unmapped_keys.append((ckpt_key, None))
    
    logger.info(f"  Remapped {len(remapped_state)} parameters")
    if unmapped_keys:
        logger.warning(f"  Could not remap {len(unmapped_keys)} keys")
    
    checkpoint_state = remapped_state
    
    # Check alignment
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model.state_dict().keys())
    
    matched_keys = checkpoint_keys & model_keys
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    logger.info(f"\nCheckpoint alignment:")
    logger.info(f"  Matched: {len(matched_keys)}")
    logger.info(f"  Missing: {len(missing_keys)}")
    logger.info(f"  Unexpected: {len(unexpected_keys)}")
    
    # Check for missing LoRA weights
    missing_lora = [k for k in missing_keys if 'lora_' in k]
    if missing_lora:
        logger.error(f"❌ CRITICAL: {len(missing_lora)} LoRA weights missing!")
        raise ValueError(f"Cannot proceed: {len(missing_lora)} LoRA weights missing from checkpoint")
    
    # Load weights
    result = model.load_state_dict(checkpoint_state, strict=False)
    
    logger.info("✓ Checkpoint loaded successfully!")
    
    # Extract metadata
    metadata = {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
    }
    
    return metadata


def compute_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    max_batches: int = 32
) -> Dict[str, float]:
    """
    Compute accuracy where TP = positive has highest similarity among [positive + negatives].
    Also computes mean rank of the positive example.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run on
        dataset_name: Name of the dataset
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Dictionary with accuracy statistics and mean rank
    """
    model.eval()
    
    total_examples = 0
    correct_examples = 0
    
    # For detailed statistics
    num_negatives_list = []  # Track how many negatives each example has
    correct_per_num_negatives = {}  # accuracy by number of negatives
    ranks_list = []  # Track rank of positive example (1 = best, 2 = second best, etc.)
    
    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=min(len(dataloader), max_batches),
            desc=f"  Computing accuracy for {dataset_name}",
            leave=False
        )
        
        for batch_idx, batch in pbar:
            if batch_idx >= max_batches:
                break
            
            try:
                # Move batch to device
                query_text_token = batch['query_text_token'].to(device)
                query_text_token_len = batch['query_text_token_len'].to(device)
                query_speech_token = batch['query_speech_token'].to(device)
                query_speech_token_len = batch['query_speech_token_len'].to(device)
                
                positive_text_token = batch['positive_text_token'].to(device)
                positive_text_token_len = batch['positive_text_token_len'].to(device)
                positive_speech_token = batch['positive_speech_token'].to(device)
                positive_speech_token_len = batch['positive_speech_token_len'].to(device)
                
                # Get query embeddings
                query_output = model(query_text_token, query_text_token_len, 
                                    query_speech_token, query_speech_token_len)
                query_emb = query_output['embedding']  # [batch_size, emb_dim]
                
                # Get positive embeddings
                positive_output = model(positive_text_token, positive_text_token_len,
                                       positive_speech_token, positive_speech_token_len)
                positive_emb = positive_output['embedding']  # [batch_size, emb_dim]
                
                # Normalize embeddings for cosine similarity
                query_emb = F.normalize(query_emb, p=2, dim=1)
                positive_emb = F.normalize(positive_emb, p=2, dim=1)
                
                # Compute positive similarities
                # [batch_size]
                pos_similarities = (query_emb * positive_emb).sum(dim=1)
                
                # Get negative embeddings if present
                if 'negative_text_token' in batch and batch['negative_text_token'] is not None:
                    negative_text_token = batch['negative_text_token'].to(device)
                    negative_text_token_len = batch['negative_text_token_len'].to(device)
                    negative_speech_token = batch['negative_speech_token'].to(device)
                    negative_speech_token_len = batch['negative_speech_token_len'].to(device)
                    
                    # negative_text_token: [num_total_negatives, seq_len]
                    # We need to know which negatives belong to which batch item
                    negative_counts = batch.get('negative_counts', None)
                    
                    if negative_counts is not None:
                        # Get negative embeddings for all negatives at once
                        negative_output = model(negative_text_token, negative_text_token_len,
                                              negative_speech_token, negative_speech_token_len)
                        negative_emb = negative_output['embedding']  # [num_total_negatives, emb_dim]
                        negative_emb = F.normalize(negative_emb, p=2, dim=1)
                        
                        # Split negatives by batch item
                        batch_size = query_emb.size(0)
                        negative_start_idx = 0
                        
                        for i in range(batch_size):
                            num_negs = negative_counts[i]
                            
                            if num_negs > 0:
                                # Get negatives for this example
                                neg_embs = negative_emb[negative_start_idx:negative_start_idx + num_negs]
                                negative_start_idx += num_negs
                                
                                # Compute similarities: [num_negs]
                                neg_sims = (query_emb[i:i+1] @ neg_embs.T).squeeze(0)
                                
                                # Get positive similarity
                                pos_sim = pos_similarities[i]
                                
                                # Combine all similarities: [positive + negatives]
                                all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                                
                                # Sort in descending order and find rank of positive (index 0)
                                sorted_indices = torch.argsort(all_sims, descending=True)
                                pos_rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based rank
                                
                                # Check if positive has highest similarity (rank 1)
                                is_correct = (pos_rank == 1)
                                
                                # Update statistics
                                total_examples += 1
                                if is_correct:
                                    correct_examples += 1
                                
                                ranks_list.append(pos_rank)
                                
                                # Track by number of negatives
                                num_negatives_list.append(num_negs)
                                if num_negs not in correct_per_num_negatives:
                                    correct_per_num_negatives[num_negs] = {'correct': 0, 'total': 0, 'ranks': []}
                                correct_per_num_negatives[num_negs]['total'] += 1
                                correct_per_num_negatives[num_negs]['ranks'].append(pos_rank)
                                if is_correct:
                                    correct_per_num_negatives[num_negs]['correct'] += 1
                            else:
                                # No negatives for this example - skip
                                pass
                    else:
                        # No negative_counts - can't split negatives properly
                        logger.warning(f"Batch {batch_idx}: negative_counts not found, skipping")
                else:
                    # No negatives in this batch - skip
                    pass
                
                # Update progress bar
                if total_examples > 0:
                    current_acc = correct_examples / total_examples * 100
                    current_mean_rank = np.mean(ranks_list) if ranks_list else 0.0
                    pbar.set_postfix({
                        'accuracy': f'{current_acc:.2f}%',
                        'mean_rank': f'{current_mean_rank:.2f}',
                        'examples': total_examples
                    })
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate statistics
    if total_examples == 0:
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'mean_rank': 0.0,
            'avg_num_negatives': 0.0,
            'accuracy_by_num_negatives': {}
        }
    
    accuracy = correct_examples / total_examples * 100
    mean_rank = float(np.mean(ranks_list)) if ranks_list else 0.0
    
    # Calculate accuracy by number of negatives
    accuracy_by_num_negatives = {}
    for num_negs in sorted(correct_per_num_negatives.keys()):
        stats = correct_per_num_negatives[num_negs]
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
        mean_rank_for_num_negs = float(np.mean(stats['ranks'])) if stats['ranks'] else 0.0
        accuracy_by_num_negatives[int(num_negs)] = {
            'accuracy': acc,
            'correct': stats['correct'],
            'total': stats['total'],
            'mean_rank': mean_rank_for_num_negs
        }
    
    return {
        'accuracy': accuracy,
        'correct': correct_examples,
        'total': total_examples,
        'mean_rank': mean_rank,
        'avg_num_negatives': float(np.mean(num_negatives_list)) if num_negatives_list else 0.0,
        'accuracy_by_num_negatives': accuracy_by_num_negatives
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate embedding checkpoint accuracy on validation sets')
    
    # Checkpoint and model
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoint-step2600.pt)')
    parser.add_argument('--model_dir', type=str,
                        default='D:\\models\\cosyvoice_models\\CosyVoice3-0.5B-2512',
                        help='Path to CosyVoice3 model directory')
    parser.add_argument('--llm_path', type=str, default=None,
                       help='Path to LLM checkpoint (default: model_dir/llm.pt)')
    
    # Data
    parser.add_argument('--val_dir', type=str, 
                       default='D:\\data\\embedding_data\\OUTPUT-parquet\\val',
                       help='Path to validation data directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to test (default: all datasets in val_dir)')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--max_batches', type=int, default=32,
                       help='Maximum batches per dataset (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # LoRA settings
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Whether checkpoint uses LoRA')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout (default: 0.1)')
    
    # Output
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save results as JSON (optional)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup LLM path
    if args.llm_path is None:
        args.llm_path = os.path.join(args.model_dir, 'llm.pt')
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Configuration")
    logger.info(f"{'='*80}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"LLM checkpoint: {args.llm_path}")
    logger.info(f"Training checkpoint: {args.checkpoint}")
    logger.info(f"Use LoRA: {args.use_lora}")
    if args.use_lora:
        logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"{'='*80}\n")
    
    # Load config
    logger.info("Loading model configuration...")
    config_path = os.path.join(args.model_dir, 'cosyvoice3.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f)
    
    # Initialize model and load base LLM weights
    logger.info("\nInitializing model and loading base LLM weights...")
    model = CosyVoice3Embedding(
        llm_config=configs['llm'],
        speech_tokenizer_path=os.path.join(args.model_dir, 'speech_tokenizer_v3.onnx')
    )
    
    logger.info(f"Loading base LLM weights from: {args.llm_path}")
    model.load_llm(args.llm_path, strict=False)
    logger.info("@ Base LLM weights loaded successfully")
    
    # Apply LoRA (if checkpoint uses it)
    if args.use_lora:
        logger.info("\nApplying LoRA to model...")
        from peft import LoraConfig, get_peft_model
        
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        model.llm = get_peft_model(model.llm, lora_config)
        logger.info("@ LoRA applied successfully")
    
    # Move model to device BEFORE loading checkpoint
    model.to(device)
    
    # Load LoRA weights from training checkpoint
    logger.info("\nLoading LoRA weights from training checkpoint...")
    checkpoint_metadata = load_checkpoint(args.checkpoint, model, device)
    logger.info(f"Checkpoint step: {checkpoint_metadata['step']}, epoch: {checkpoint_metadata['epoch']}")
    
    # Discover validation datasets
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    dataset_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    
    if args.datasets:
        dataset_dirs = [d for d in dataset_dirs if d.name in args.datasets]
    
    if not dataset_dirs:
        raise ValueError(f"No validation datasets found in {val_dir}")
    
    dataset_dirs = sorted(dataset_dirs, key=lambda x: x.name)
    
    logger.info(f"\nFound {len(dataset_dirs)} validation dataset(s):")
    for dataset_dir in dataset_dirs:
        logger.info(f"  - {dataset_dir.name}")
    
    # Evaluate each dataset
    logger.info(f"\n{'='*80}")
    logger.info("Starting Accuracy Evaluation")
    logger.info(f"{'='*80}\n")
    
    results = {}
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        
        logger.info(f"\n{'-'*80}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'-'*80}")
        
        try:
            # Load dataset - must include negatives
            dataset = ParquetVoiceEmbeddingDataset(
                parquet_dir=str(val_dir),
                split=None,
                datasets=[dataset_name],
                use_hard_negatives=True,  # Must be True to get negatives
                max_negatives=7,
                shuffle_datasets=False,
                seed=42
            )
            
            logger.info(f"  Loaded {len(dataset)} samples")
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=False
            )
            
            # Compute accuracy
            stats = compute_accuracy(
                model=model,
                dataloader=dataloader,
                device=device,
                dataset_name=dataset_name,
                max_batches=args.max_batches
            )
            
            # Store results
            results[dataset_name] = stats
            
            # Print results
            logger.info(f"\n  Results for {dataset_name}:")
            logger.info(f"    Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
            logger.info(f"    Mean rank: {stats['mean_rank']:.2f}")
            logger.info(f"    Avg negatives per example: {stats['avg_num_negatives']:.1f}")
            
            if stats['accuracy_by_num_negatives']:
                logger.info(f"\n    Accuracy by number of negatives:")
                for num_negs, acc_stats in sorted(stats['accuracy_by_num_negatives'].items()):
                    logger.info(
                        f"      {num_negs} negatives: {acc_stats['accuracy']:.2f}% "
                        f"({acc_stats['correct']}/{acc_stats['total']}) "
                        f"Mean rank: {acc_stats['mean_rank']:.2f}"
                    )
            
        except Exception as e:
            logger.error(f"  Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {
                'error': str(e),
                'accuracy': 0.0,
                'correct': 0,
                'total': 0
            }
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("Summary")
    logger.info(f"{'='*80}\n")
    
    # Sort by accuracy (descending)
    valid_results = {k: v for k, v in results.items() if 'error' not in v and v['total'] > 0}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    logger.info(f"{'Dataset':<50} {'Accuracy':<15} {'Correct/Total':<20} {'Mean Rank':<15} {'Avg Negs'}")
    logger.info(f"{'-'*50} {'-'*15} {'-'*20} {'-'*15} {'-'*10}")
    
    for dataset_name, stats in sorted_results:
        logger.info(
            f"{dataset_name:<50} "
            f"{stats['accuracy']:>6.2f}%        "
            f"{stats['correct']}/{stats['total']:<15} "
            f"{stats['mean_rank']:>6.2f}        "
            f"{stats['avg_num_negatives']:>6.1f}"
        )
    
    # Calculate overall statistics
    if valid_results:
        all_correct = sum(v['correct'] for v in valid_results.values())
        all_total = sum(v['total'] for v in valid_results.values())
        overall_accuracy = all_correct / all_total * 100 if all_total > 0 else 0.0
        overall_mean_rank = np.mean([v['mean_rank'] for v in valid_results.values()]) if valid_results else 0.0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Overall (across all datasets):")
        logger.info(f"  Accuracy: {overall_accuracy:.2f}% ({all_correct}/{all_total})")
        logger.info(f"  Mean rank: {overall_mean_rank:.2f}")
        logger.info(f"  Best dataset: {sorted_results[0][0]} ({sorted_results[0][1]['accuracy']:.2f}%)")
        logger.info(f"  Worst dataset: {sorted_results[-1][0]} ({sorted_results[-1][1]['accuracy']:.2f}%)")
    
    # Save results to JSON if requested
    if args.output_json:
        output_data = {
            'checkpoint': args.checkpoint,
            'checkpoint_metadata': checkpoint_metadata,
            'evaluation_settings': {
                'batch_size': args.batch_size,
                'max_batches': args.max_batches,
            },
            'results': results,
        }
        
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResults saved to: {args.output_json}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Evaluation Complete!")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()