#!/usr/bin/env python3
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
Batch process multiple JSONL datasets to Parquet format.

This script automatically discovers and processes all JSONL files in a directory structure,
making it easy to preprocess entire datasets at once.

Usage:
    python batch_tokenize.py \
        --data_root data/ \
        --output_dir parquet/ \
        --model_dir pretrained_models/CosyVoice-300M \
        --rows_per_partition 1000 \
        --num_workers 8
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import subprocess

# Set offline mode
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'


def find_jsonl_files(data_root: Path) -> dict:
    """
    Find all JSONL files organized by train/val split.
    
    Expected structure:
        data_root/
            train/
                111.train.jsonl
                222.train.jsonl
            val/
                111.val.jsonl
                222.val.jsonl
    
    Or flat structure:
        data_root/
            111.train.jsonl
            222.train.jsonl
            111.val.jsonl
            222.val.jsonl
    
    Returns:
        Dict with 'train' and 'val' lists of file paths
    """
    files = {'train': [], 'val': []}
    
    # Check for organized structure
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    
    if train_dir.exists():
        files['train'].extend(sorted(train_dir.glob('*.jsonl')))
    
    if val_dir.exists():
        files['val'].extend(sorted(val_dir.glob('*.jsonl')))
    
    # Also check flat structure
    for jsonl_file in sorted(data_root.glob('*.jsonl')):
        if 'train' in jsonl_file.stem.lower():
            files['train'].append(jsonl_file)
        elif 'val' in jsonl_file.stem.lower():
            files['val'].append(jsonl_file)
        else:
            # Default to train if ambiguous
            files['train'].append(jsonl_file)
    
    return files


def main():
    parser = argparse.ArgumentParser(description='Batch tokenize JSONL files to Parquet')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for parquet files')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory for CosyVoice frontend')
    parser.add_argument('--rows_per_partition', type=int, default=1000,
                        help='Number of rows per partition file')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes per file')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip files that are already processed')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train_only', action='store_true',
                        help='Only process training files')
    parser.add_argument('--val_only', action='store_true',
                        help='Only process validation files')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = f"tokenize_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("BATCH TOKENIZATION TO PARQUET")
    logging.info("="*80)
    logging.info(f"Data root: {args.data_root}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Model dir: {args.model_dir}")
    logging.info(f"Workers: {args.num_workers}")
    logging.info(f"Rows per partition: {args.rows_per_partition}")
    logging.info("")
    
    # Find all JSONL files
    data_root = Path(args.data_root)
    if not data_root.exists():
        logging.error(f"Data root not found: {data_root}")
        return
    
    files = find_jsonl_files(data_root)
    
    # Filter by split if requested
    if args.train_only:
        files['val'] = []
    elif args.val_only:
        files['train'] = []
    
    total_files = len(files['train']) + len(files['val'])
    
    if total_files == 0:
        logging.error("No JSONL files found!")
        return
    
    logging.info(f"Found {len(files['train'])} training files")
    logging.info(f"Found {len(files['val'])} validation files")
    logging.info(f"Total: {total_files} files")
    logging.info("")
    
    # Get path to tokenize_to_parquet.py
    script_dir = Path(__file__).parent
    tokenize_script = script_dir / 'tokenize_to_parquet.py'
    
    if not tokenize_script.exists():
        logging.error(f"Tokenization script not found: {tokenize_script}")
        return
    
    # Process all files
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    all_files = [('val', f) for f in files['val']] + [('train', f) for f in files['train']]
    
    for idx, (split, jsonl_file) in enumerate(all_files, 1):
        logging.info("="*80)
        logging.info(f"Processing file {idx}/{total_files}: {jsonl_file.name}")
        logging.info("="*80)
        
        # Build command
        cmd = [
            sys.executable,
            str(tokenize_script),
            '--input', str(jsonl_file),
            '--output_dir', args.output_dir,
            '--model_dir', args.model_dir,
            '--rows_per_partition', str(args.rows_per_partition),
            '--num_workers', str(args.num_workers),
            '--seed', str(args.seed),
        ]
        
        if args.no_shuffle:
            cmd.append('--no_shuffle')
        
        if args.skip_existing:
            cmd.append('--skip_existing')
        
        logging.info(f"Command: {' '.join(cmd)}")
        logging.info("")
        
        try:
            # Run the command
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Let output stream to console
                text=True
            )
            
            success_count += 1
            logging.info(f"✅ Successfully processed: {jsonl_file.name}\n")
            
        except subprocess.CalledProcessError as e:
            fail_count += 1
            logging.error(f"❌ Failed to process: {jsonl_file.name}")
            logging.error(f"   Error: {e}\n")
            continue
        
        except KeyboardInterrupt:
            logging.warning("\n⚠️  Interrupted by user")
            break
    
    # Final summary
    logging.info("="*80)
    logging.info("BATCH PROCESSING COMPLETE")
    logging.info("="*80)
    logging.info(f"Total files: {total_files}")
    logging.info(f"✅ Success: {success_count}")
    logging.info(f"❌ Failed: {fail_count}")
    logging.info(f"⏭️  Skipped: {skip_count}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*80)


if __name__ == '__main__':
    main()
