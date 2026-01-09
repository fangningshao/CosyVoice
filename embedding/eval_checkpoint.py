"""
Test a trained embedding checkpoint against validation sets.

Usage:
    python embedding/eval_checkpoint.py \
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from hyperpyyaml import load_hyperpyyaml

from embedding.model import CosyVoice3Embedding
from embedding.parquet_dataset import ParquetVoiceEmbeddingDataset, collate_fn
from embedding.loss import InfoNCELoss

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
    
    # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy scalars
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get checkpoint state dict
    if 'model_state_dict' in checkpoint:
        checkpoint_state = checkpoint['model_state_dict']
    else:
        checkpoint_state = checkpoint
    
    # Get model state dict
    model_state = model.state_dict()
    
    # ========================================================================
    # Remap checkpoint keys to match current model structure
    # ========================================================================
    # During training: llm.llm.model.model.layers.X.lora_A...
    # After get_peft_model(): llm.base_model.model.llm.model.model.layers.X.lora_A...
    
    remapped_state = {}
    unmapped_keys = []
    
    logger.info("Remapping checkpoint keys to match model structure...")
    
    for ckpt_key, ckpt_value in checkpoint_state.items():
        # Try direct match first
        if ckpt_key in model_state:
            remapped_state[ckpt_key] = ckpt_value
        # Remap LoRA keys: llm.llm.* -> llm.base_model.model.llm.*
        elif ckpt_key.startswith('llm.llm.'):
            new_key = ckpt_key.replace('llm.llm.', 'llm.base_model.model.llm.')
            if new_key in model_state:
                remapped_state[new_key] = ckpt_value
            else:
                unmapped_keys.append((ckpt_key, new_key))
        else:
            unmapped_keys.append((ckpt_key, None))
    
    logger.info(f"  Remapped {len(remapped_state)} parameters")
    if unmapped_keys:
        logger.warning(f"  Could not remap {len(unmapped_keys)} keys")
    
    # Use remapped checkpoint state
    checkpoint_state = remapped_state
    
    # Analyze parameter alignment
    logger.info("\n" + "="*80)
    logger.info("Checkpoint vs Model Parameter Alignment (After Remapping)")
    logger.info("="*80)
    
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())
    
    # Keys in checkpoint but not in model (unexpected)
    unexpected_keys = checkpoint_keys - model_keys
    # Keys in model but not in checkpoint (missing)
    missing_keys = model_keys - checkpoint_keys
    # Keys in both (matched)
    matched_keys = checkpoint_keys & model_keys
    
    logger.info(f"\nTotal checkpoint parameters: {len(checkpoint_keys)}")
    logger.info(f"Total model parameters: {len(model_keys)}")
    logger.info(f"Matched parameters: {len(matched_keys)}")
    logger.info(f"Missing in checkpoint: {len(missing_keys)}")
    logger.info(f"Unexpected in checkpoint: {len(unexpected_keys)}")
    
    # Categorize missing keys by type
    missing_base_weights = []
    missing_lora_weights = []
    missing_other = []
    
    for key in missing_keys:
        if 'lora_' in key:
            missing_lora_weights.append(key)
        elif 'base_layer' in key or 'llm.base_model.model.llm.model.model' in key or 'embed_tokens' in key or 'lm_head' in key:
            missing_base_weights.append(key)
        else:
            missing_other.append(key)
    
    # Show missing keys by category
    if missing_keys:
        logger.info(f"\n{'-'*80}")
        logger.info(f"MISSING KEYS (in model but not in checkpoint): {len(missing_keys)}")
        logger.info(f"{'-'*80}")
        
        if missing_base_weights:
            logger.info(f"\n  Base LLM weights (loaded from base checkpoint): {len(missing_base_weights)}")
            for i, key in enumerate(sorted(missing_base_weights)[:5]):
                logger.info(f"    {i+1}. {key}")
            if len(missing_base_weights) > 5:
                logger.info(f"    ... and {len(missing_base_weights) - 5} more")
        
        if missing_lora_weights:
            logger.error(f"\n  ❌ LoRA weights (RANDOMLY INITIALIZED - CRITICAL ERROR!): {len(missing_lora_weights)}")
            for i, key in enumerate(sorted(missing_lora_weights)[:20]):
                logger.error(f"    {i+1}. {key}")
            if len(missing_lora_weights) > 20:
                logger.error(f"    ... and {len(missing_lora_weights) - 20} more")
        
        if missing_other:
            logger.info(f"\n  Other parameters: {len(missing_other)}")
            for i, key in enumerate(sorted(missing_other)[:5]):
                logger.info(f"    {i+1}. {key}")
            if len(missing_other) > 5:
                logger.info(f"    ... and {len(missing_other) - 5} more")
    
    # Show unexpected keys
    if unexpected_keys:
        logger.info(f"\n{'-'*80}")
        logger.info(f"UNEXPECTED KEYS (in checkpoint but not in model): {len(unexpected_keys)}")
        logger.info(f"{'-'*80}")
        for i, key in enumerate(sorted(unexpected_keys)[:10]):
            param = checkpoint_state[key]
            logger.info(f"  {i+1}. {key}")
            logger.info(f"      Shape: {tuple(param.shape)}")
        if len(unexpected_keys) > 10:
            logger.info(f"  ... and {len(unexpected_keys) - 10} more")
    
    # Check for shape mismatches in matched keys
    shape_mismatches = []
    for key in matched_keys:
        ckpt_shape = checkpoint_state[key].shape
        model_shape = model_state[key].shape
        if ckpt_shape != model_shape:
            shape_mismatches.append((key, ckpt_shape, model_shape))
    
    if shape_mismatches:
        logger.error(f"\n{'-'*80}")
        logger.error(f"SHAPE MISMATCHES: {len(shape_mismatches)}")
        logger.error(f"{'-'*80}")
        for i, (key, ckpt_shape, model_shape) in enumerate(shape_mismatches):
            logger.error(f"  {i+1}. {key}")
            logger.error(f"      Checkpoint: {tuple(ckpt_shape)}")
            logger.error(f"      Model:      {tuple(model_shape)}")
    
    logger.info(f"\n{'='*80}\n")
    
    # CRITICAL: Fail if any LoRA weights are missing
    if missing_lora_weights:
        logger.error("\n" + "="*80)
        logger.error("CRITICAL ERROR: LoRA weights not found in checkpoint!")
        logger.error("="*80)
        logger.error(f"Found {len(missing_lora_weights)} LoRA parameters that would be randomly initialized.")
        logger.error("This means the model would not use the trained LoRA weights!")
        logger.error("\nPossible causes:")
        logger.error("  1. Checkpoint was not saved with LoRA weights")
        logger.error("  2. Key remapping failed (check prefix mismatch)")
        logger.error("  3. LoRA config mismatch (rank, alpha, target modules)")
        logger.error("="*80)
        raise ValueError(f"Cannot proceed: {len(missing_lora_weights)} LoRA weights missing from checkpoint")
    
    # Load with strict=False to allow missing base weights (loaded separately)
    result = model.load_state_dict(checkpoint_state, strict=False)
    
    # Verify no LoRA weights in missing_keys
    missing_lora_after_load = [k for k in result.missing_keys if 'lora_' in k]
    if missing_lora_after_load:
        logger.error(f"\n❌ CRITICAL: {len(missing_lora_after_load)} LoRA weights still missing after load!")
        raise ValueError(f"LoRA weights not properly loaded: {missing_lora_after_load[:5]}")
    
    # Show load_state_dict result summary
    logger.info("load_state_dict() result:")
    if result.missing_keys:
        missing_base = [k for k in result.missing_keys if 'lora_' not in k]
        logger.info(f"  Missing base weights: {len(missing_base)} (expected - loaded from base LLM)")
    if result.unexpected_keys:
        logger.info(f"  Unexpected keys: {len(result.unexpected_keys)}")
    
    logger.info("✓ All LoRA parameters loaded successfully!")
    
    # Extract metadata
    metadata = {
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'config': checkpoint.get('config', {}),
        'alignment_stats': {
            'matched': len(matched_keys),
            'missing': len(missing_keys),
            'unexpected': len(unexpected_keys),
            'shape_mismatches': len(shape_mismatches),
            'missing_lora': len(missing_lora_weights),
            'missing_base': len(missing_base_weights)
        }
    }
    
    logger.info(f"\nCheckpoint metadata:")
    logger.info(f"  Step: {metadata['step']}")
    logger.info(f"  Epoch: {metadata['epoch']}")
    if metadata['best_val_loss'] < float('inf'):
        logger.info(f"  Best validation loss: {metadata['best_val_loss']:.4f}")
    
    return metadata


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset_name: str,
    max_batches: int = 32
) -> Dict[str, float]:
    """
    Evaluate model on a single dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        criterion: Loss function
        device: Device to run on
        dataset_name: Name of the dataset
        max_batches: Maximum number of batches to evaluate
        
    Returns:
        Dictionary with loss statistics
    """
    model.eval()
    
    losses = []
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(
            enumerate(dataloader),
            total=min(len(dataloader), max_batches),
            desc=f"  Evaluating {dataset_name}",
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
                
                # Note: parquet dataset uses 'positive_*' not 'pos_*'
                positive_text_token = batch['positive_text_token'].to(device)
                positive_text_token_len = batch['positive_text_token_len'].to(device)
                positive_speech_token = batch['positive_speech_token'].to(device)
                positive_speech_token_len = batch['positive_speech_token_len'].to(device)
                
                # Handle negatives if present
                negative_text_token = None
                negative_speech_token = None
                if 'negative_text_token' in batch:
                    negative_text_token = batch['negative_text_token'].to(device)
                    negative_text_token_len = batch['negative_text_token_len'].to(device)
                    negative_speech_token = batch['negative_speech_token'].to(device)
                    negative_speech_token_len = batch['negative_speech_token_len'].to(device)
                
                # Forward pass
                query_output = model(query_text_token, query_text_token_len, 
                                    query_speech_token, query_speech_token_len)
                query_emb = query_output['embedding']
                
                positive_output = model(positive_text_token, positive_text_token_len,
                                       positive_speech_token, positive_speech_token_len)
                positive_emb = positive_output['embedding']
                
                # Extract negative embeddings if provided
                negative_emb = None
                if negative_speech_token is not None:
                    negative_output = model(negative_text_token, negative_text_token_len,
                                          negative_speech_token, negative_speech_token_len)
                    negative_emb = negative_output['embedding']
                
                # Calculate loss
                loss_dict = criterion(
                    query_emb,
                    positive_emb,
                    negative_emb,
                    batch.get('negative_counts', None)
                )
                loss = loss_dict['total_loss']
                
                losses.append(loss.item())
                num_samples += query_emb.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate statistics
    if not losses:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'num_batches': 0,
            'num_samples': 0
        }
    
    return {
        'mean': float(np.mean(losses)),
        'std': float(np.std(losses)),
        'min': float(np.min(losses)),
        'max': float(np.max(losses)),
        'num_batches': len(losses),
        'num_samples': num_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Test embedding checkpoint on validation sets')
    
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
                       help='Path to validation data directory (e.g., D:/data/embedding_data/OUTPUT-parquet/val)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to test (default: all datasets in val_dir)')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--max_batches', type=int, default=32,
                       help='Maximum batches per dataset (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    # Loss settings
    parser.add_argument('--loss_type', type=str, default='infonce',
                       choices=['infonce', 'ntxent'],
                       help='Loss type (default: infonce)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for loss (default: 0.1)')
    parser.add_argument('--batch_softmax_only', action='store_true',
                       help='Use only in-batch negatives (ignore hard negatives)')
    
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
    
    # ========================================================================
    # STEP 1: Initialize model and load base LLM weights
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Initialize model and load base LLM weights")
    logger.info("="*80)
    
    model = CosyVoice3Embedding(
        llm_config=configs['llm'],
        speech_tokenizer_path=os.path.join(args.model_dir, 'speech_tokenizer_v3.onnx')
    )
    
    logger.info(f"Loading base LLM weights from: {args.llm_path}")
    model.load_llm(args.llm_path, strict=False)
    logger.info("@ Base LLM weights loaded successfully")
    
    # ========================================================================
    # STEP 2: Apply LoRA (if checkpoint uses it)
    # ========================================================================
    if args.use_lora:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Apply LoRA to model")
        logger.info("="*80)
        logger.info(f"  LoRA rank (r): {args.lora_r}")
        logger.info(f"  LoRA alpha: {args.lora_alpha}")
        logger.info(f"  LoRA dropout: {args.lora_dropout}")
        
        from peft import LoraConfig, get_peft_model
        
        # Parse target modules
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # Apply LoRA to the LLM part of the model
        model.llm = get_peft_model(model.llm, lora_config)
        logger.info("@ LoRA applied successfully (LoRA weights randomly initialized)")
    
    # Move model to device BEFORE loading checkpoint
    model.to(device)
    
    # ========================================================================
    # STEP 3: Load LoRA weights from training checkpoint
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Load LoRA weights from training checkpoint")
    logger.info("="*80)
    
    checkpoint_metadata = load_checkpoint(args.checkpoint, model, device)
    
    # ========================================================================
    # STEP 4: Check for uninitialized parameters
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Verify all parameters are properly initialized")
    logger.info("="*80)
    
    # Get all parameter names and check which ones might be uninitialized
    # (LoRA weights from checkpoint should have overwritten the random init)
    model_state = model.state_dict()
    
    # Count parameters by category
    base_params = []
    lora_params = []
    other_params = []
    
    for name, param in model_state.items():
        if 'lora_' in name:
            lora_params.append(name)
        elif 'base_layer' in name or 'llm.model' in name or 'embed_tokens' in name or 'lm_head' in name:
            base_params.append(name)
        else:
            other_params.append(name)
    
    logger.info(f"\nParameter Summary:")
    logger.info(f"  Base LLM parameters: {len(base_params)}")
    logger.info(f"  LoRA parameters: {len(lora_params)}")
    logger.info(f"  Other parameters (llm_decoder, etc.): {len(other_params)}")
    logger.info(f"  Total parameters: {len(model_state)}")
    
    # Check if any parameters are still random (not loaded from checkpoint)
    # This is hard to detect automatically, so we rely on the missing_keys from load_checkpoint
    if checkpoint_metadata['alignment_stats']['missing'] > 0:
        logger.warning(f"\n#  WARNING: {checkpoint_metadata['alignment_stats']['missing']} parameters were not loaded from checkpoint!")
        logger.warning(f"   These parameters are using either:")
        logger.warning(f"   - Base weights from LLM checkpoint (for base_layer.*)")
        logger.warning(f"   - Random initialization (for new LoRA weights)")
    else:
        logger.info(f"\n@ All parameters successfully loaded!")
    
    logger.info(f"\n{'='*80}\n")
    
    # Setup loss function
    if args.loss_type == 'infonce':
        criterion = InfoNCELoss(
            temperature=args.temperature,
            use_hard_negatives=not args.batch_softmax_only
        )
    else:
        raise NotImplementedError(f"Loss type '{args.loss_type}' not implemented")
    
    logger.info(f"Loss function: {args.loss_type.upper()}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Use hard negatives: {not args.batch_softmax_only}")
    
    # Discover validation datasets
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    # Get list of dataset directories
    dataset_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    
    if args.datasets:
        # Filter to specified datasets
        dataset_dirs = [d for d in dataset_dirs if d.name in args.datasets]
    
    if not dataset_dirs:
        raise ValueError(f"No validation datasets found in {val_dir}")
    
    dataset_dirs = sorted(dataset_dirs, key=lambda x: x.name)
    
    logger.info(f"\nFound {len(dataset_dirs)} validation dataset(s):")
    for dataset_dir in dataset_dirs:
        logger.info(f"  - {dataset_dir.name}")
    
    # Evaluate each dataset
    logger.info(f"\n{'='*80}")
    logger.info("Starting Evaluation")
    logger.info(f"{'='*80}\n")
    
    results = {}
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        
        logger.info(f"\n{'-'*80}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'-'*80}")
        
        try:
            # Load dataset - parquet files are directly in dataset_dir, not in subdirectories
            # So we pass the parent dir and specify this dataset by name
            dataset = ParquetVoiceEmbeddingDataset(
                parquet_dir=str(val_dir),  # Parent directory (val/)
                split=None,  # Don't append train/val
                datasets=[dataset_name],  # Load only this specific dataset
                use_hard_negatives=not args.batch_softmax_only,
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
            
            # Evaluate
            stats = evaluate_dataset(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                device=device,
                dataset_name=dataset_name,
                max_batches=args.max_batches
            )
            
            # Store results
            results[dataset_name] = stats
            
            # Print results
            logger.info(f"\n  Results for {dataset_name}:")
            logger.info(f"    Loss (mean): {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(f"    Loss (min):  {stats['min']:.4f}")
            logger.info(f"    Loss (max):  {stats['max']:.4f}")
            logger.info(f"    Batches:     {stats['num_batches']}/{min(len(dataloader), args.max_batches)}")
            logger.info(f"    Samples:     {stats['num_samples']}")
            
        except Exception as e:
            logger.error(f"  Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {
                'error': str(e),
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'num_batches': 0,
                'num_samples': 0
            }
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("Summary")
    logger.info(f"{'='*80}\n")
    
    # Sort by loss (mean)
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['mean'])}
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mean'])
    
    logger.info(f"{'Dataset':<50} {'Loss (mean ± std)':<25} {'Batches':<10} {'Samples'}")
    logger.info(f"{'-'*50} {'-'*25} {'-'*10} {'-'*10}")
    
    for dataset_name, stats in sorted_results:
        logger.info(
            f"{dataset_name:<50} "
            f"{stats['mean']:>6.4f} ± {stats['std']:<6.4f}        "
            f"{stats['num_batches']:<10} "
            f"{stats['num_samples']}"
        )
    
    # Calculate overall statistics
    if valid_results:
        all_losses = [v['mean'] for v in valid_results.values()]
        overall_mean = np.mean(all_losses)
        overall_std = np.std(all_losses)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Overall (across datasets):")
        logger.info(f"  Mean loss: {overall_mean:.4f} ± {overall_std:.4f}")
        logger.info(f"  Best dataset: {sorted_results[0][0]} ({sorted_results[0][1]['mean']:.4f})")
        logger.info(f"  Worst dataset: {sorted_results[-1][0]} ({sorted_results[-1][1]['mean']:.4f})")
    
    # Save results to JSON if requested
    if args.output_json:
        output_data = {
            'checkpoint': args.checkpoint,
            'checkpoint_metadata': checkpoint_metadata,
            'evaluation_settings': {
                'batch_size': args.batch_size,
                'max_batches': args.max_batches,
                'loss_type': args.loss_type,
                'temperature': args.temperature,
                'batch_softmax_only': args.batch_softmax_only,
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