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
Tokenize JSONL files to Parquet format for fast training.

Usage:
    python tokenize_to_parquet.py \
        --input data/train/111.train.jsonl \
        --output_dir parquet/train \
        --model_dir pretrained_models/CosyVoice-300M \
        --rows_per_partition 1000 \
        --num_workers 4

Output structure:
    parquet/train/111/
        part-00000.parquet
        part-00001.parquet
        ...
        _metadata.json  # Statistics and info
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import torch
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set offline mode BEFORE any CosyVoice imports
os.environ['MODELSCOPE_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from hyperpyyaml import load_hyperpyyaml


# Global frontend (initialized once per worker process)
_frontend = None
_model_dir = None
_timing_stats = {'text': 0, 'speech': 0, 'samples': 0}


def init_worker(model_dir: str):
    """Initialize worker process with frontend."""
    global _frontend, _model_dir, _timing_stats
    _model_dir = model_dir
    _timing_stats = {'text': 0, 'speech': 0, 'samples': 0}
    
    config_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    if not os.path.exists(config_path):
        raise ValueError(f"Config not found: {config_path}")
    
    logging.info(f"[Worker {os.getpid()}] Initializing frontend...")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f)
    
    _frontend = CosyVoiceFrontEnd(
        configs['get_tokenizer'],
        configs['feat_extractor'],
        os.path.join(model_dir, 'campplus.onnx'),
        os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
        os.path.join(model_dir, 'spk2info.pt'),
        configs['allowed_special']
    )
    
    logging.info(f"[Worker {os.getpid()}] Frontend ready")


def tokenize_text(text: str) -> tuple:
    """Tokenize text instruction."""
    text_normalized = _frontend.text_normalize(text, split=False, text_frontend=True)
    text_token = _frontend.tokenizer.encode(text_normalized, 
                                            allowed_special=_frontend.allowed_special)
    return text_token, len(text_token)


def tokenize_speech(audio_path: str) -> tuple:
    """Tokenize speech audio with error handling for memory issues."""
    try:
        speech_token, speech_token_len = _frontend._extract_speech_token(audio_path)
        
        # Convert to CPU and numpy
        if speech_token.is_cuda:
            speech_token = speech_token.cpu()
        if speech_token_len.is_cuda:
            speech_token_len = speech_token_len.cpu()
        
        # Flatten to 1D array
        speech_token = speech_token.flatten().numpy().astype(np.int32)
        speech_token_len = int(speech_token_len.item())
        
        return speech_token, speech_token_len
        
    except MemoryError as e:
        # Memory allocation failed - try to free memory and skip this file
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise Exception(f"Memory allocation failed for {audio_path}: {e}")
    except RuntimeError as e:
        # PyTorch runtime errors (CUDA OOM, etc.)
        if "out of memory" in str(e).lower() or "unable to allocate" in str(e).lower():
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise Exception(f"Out of memory for {audio_path}: {e}")
        raise Exception(f"Runtime error tokenizing {audio_path}: {e}")
    except Exception as e:
        raise Exception(f"Error tokenizing speech {audio_path}: {e}")


def tokenize_speech_batch(audio_paths: List[str]) -> List[tuple]:
    """
    Tokenize multiple speech audios in batch for better throughput.
    Falls back to single processing if batch processing fails.
    """
    results = []
    
    # Try batch processing (if supported by frontend)
    # Otherwise fall back to sequential
    for audio_path in audio_paths:
        try:
            speech_token, speech_token_len = tokenize_speech(audio_path)
            results.append((speech_token, speech_token_len))
        except Exception as e:
            logging.warning(f"Failed to tokenize {audio_path}: {e}")
            results.append((np.array([], dtype=np.int32), 0))
    
    return results


def extract_instruction_prefix(query_text: str) -> str:
    """Extract instruction prefix for text2speech tasks."""
    if "content:" in query_text.lower():
        return "Retrieve the voice with the following content"
    return query_text


def process_sample(item: Dict[str, Any], sample_idx: int, skip_negatives: bool = False) -> Optional[Dict[str, Any]]:
    """
    Process a single sample and return tokenized data.
    
    Args:
        item: Sample data dict
        sample_idx: Sample index
        skip_negatives: If True, skip processing hard negatives (8x faster)
    
    Returns:
        Dict with tokenized data, or None if sample failed
    """
    global _timing_stats
    sample_start = time.time()
    
    try:
        # Handle different data formats
        if 'query_text' in item and 'query_wav' in item:
            # Standard format
            query_instruction = item['query_text']
            query_audio = item['query_wav']
            positive_audio = item['pos_wav']
            negative_audios = item.get('neg_wavs', []) if not skip_negatives else []
            
        elif 'query' in item and 'pos' in item:
            # KaLM format
            query_instruction = item['query']
            query_audio = item.get('query_wav')
            positive_audio = item['pos'][0] if isinstance(item['pos'], list) else item['pos']
            negative_audios = item.get('neg', []) if not skip_negatives else []
            
        else:
            # Old format
            query_audio = item['query']
            query_instruction = item.get('query_instruction', 'Retrieve semantically similar voice')
            positive_audio = item['positive']
            negative_audios = item.get('negatives', []) if not skip_negatives else []
        
        # Extract instruction prefix
        pos_neg_instruction = extract_instruction_prefix(query_instruction)
        
        # Tokenize texts (fast)
        text_start = time.time()
        query_text_token, query_text_token_len = tokenize_text(query_instruction)
        positive_text_token, positive_text_token_len = tokenize_text(pos_neg_instruction)
        _timing_stats['text'] += time.time() - text_start
        
        # Collect all audio paths to tokenize
        audio_paths_to_process = []
        audio_path_types = []  # Track which is query/positive/negative
        
        if query_audio and query_audio != "null" and os.path.exists(query_audio):
            audio_paths_to_process.append(query_audio)
            audio_path_types.append('query')
        
        audio_paths_to_process.append(positive_audio)
        audio_path_types.append('positive')
        
        # Only add negatives if not skipping
        if not skip_negatives:
            for neg_audio in negative_audios:
                audio_paths_to_process.append(neg_audio)
                audio_path_types.append('negative')
        
        # Batch tokenize all audios at once
        speech_start = time.time()
        speech_results = tokenize_speech_batch(audio_paths_to_process)
        _timing_stats['speech'] += time.time() - speech_start
        
        # Parse results
        result_idx = 0
        
        # Query audio
        if audio_path_types and audio_path_types[0] == 'query':
            query_speech_token, query_speech_token_len = speech_results[result_idx]
            result_idx += 1
        else:
            query_speech_token = np.array([], dtype=np.int32)
            query_speech_token_len = 0
        
        # Positive audio
        positive_speech_token, positive_speech_token_len = speech_results[result_idx]
        result_idx += 1
        
        # Prepare result
        result = {
            'sample_idx': sample_idx,
            'query_text': query_instruction,
            'query_text_token': np.array(query_text_token, dtype=np.int32),
            'query_text_token_len': query_text_token_len,
            'query_speech_token': query_speech_token,
            'query_speech_token_len': query_speech_token_len,
            'positive_text_token': np.array(positive_text_token, dtype=np.int32),
            'positive_text_token_len': positive_text_token_len,
            'positive_speech_token': positive_speech_token,
            'positive_speech_token_len': positive_speech_token_len,
            'positive_audio_path': positive_audio,
        }
        
        # Add negatives only if not skipping
        if not skip_negatives and result_idx < len(speech_results):
            neg_speech_tokens = []
            neg_speech_token_lens = []
            valid_neg_paths = []
            
            for i in range(result_idx, len(speech_results)):
                neg_speech_token, neg_speech_token_len = speech_results[i]
                if neg_speech_token_len > 0:  # Valid result
                    neg_speech_tokens.append(neg_speech_token)
                    neg_speech_token_lens.append(neg_speech_token_len)
                    valid_neg_paths.append(negative_audios[i - result_idx])
            
            if neg_speech_tokens:
                result['negative_speech_tokens'] = neg_speech_tokens
                result['negative_speech_token_lens'] = np.array(neg_speech_token_lens, dtype=np.int32)
                result['negative_audio_paths'] = valid_neg_paths
        
        _timing_stats['samples'] += 1
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing sample {sample_idx}: {e}")
        if 'positive_audio' in locals():
            logging.error(f"  Positive audio: {positive_audio}")
        return None


def create_parquet_schema() -> pa.Schema:
    """Create PyArrow schema for the parquet files."""
    return pa.schema([
        ('sample_idx', pa.int64()),
        ('query_text', pa.string()),
        ('query_text_token', pa.list_(pa.int32())),
        ('query_text_token_len', pa.int32()),
        ('query_speech_token', pa.list_(pa.int32())),
        ('query_speech_token_len', pa.int32()),
        ('positive_text_token', pa.list_(pa.int32())),
        ('positive_text_token_len', pa.int32()),
        ('positive_speech_token', pa.list_(pa.int32())),
        ('positive_speech_token_len', pa.int32()),
        ('positive_audio_path', pa.string()),
        # Negatives stored as list of lists (variable length)
        ('negative_speech_tokens', pa.list_(pa.list_(pa.int32()))),
        ('negative_speech_token_lens', pa.list_(pa.int32())),
        ('negative_audio_paths', pa.list_(pa.string())),
    ])


def write_parquet_partition(data: List[Dict], output_path: str, schema: pa.Schema):
    """Write a partition of data to parquet file."""
    # Prepare columns
    columns = {field.name: [] for field in schema}
    
    for item in data:
        for field in schema:
            if field.name in item:
                columns[field.name].append(item[field.name])
            else:
                # Handle missing optional fields (negatives)
                if field.name == 'negative_speech_tokens':
                    columns[field.name].append([])
                elif field.name == 'negative_speech_token_lens':
                    columns[field.name].append([])
                elif field.name == 'negative_audio_paths':
                    columns[field.name].append([])
                else:
                    raise ValueError(f"Missing required field: {field.name}")
    
    # Create table
    table = pa.table(columns, schema=schema)
    
    # Write to parquet
    pq.write_table(table, output_path, compression='snappy')
    
    return len(data)


def process_jsonl_file(
    input_file: str,
    output_dir: str,
    model_dir: str,
    rows_per_partition: int = 1000,
    shuffle: bool = True,
    num_workers: int = 4,
    skip_existing: bool = False,
    skip_negatives: bool = False
) -> Dict[str, Any]:
    """
    Process a single JSONL file and convert to Parquet.
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Base output directory
        model_dir: Model directory for frontend
        rows_per_partition: Number of rows per partition file
        shuffle: Whether to shuffle samples
        num_workers: Number of worker processes
        skip_existing: Skip if output already exists
        skip_negatives: Skip tokenizing hard negatives (8x faster)
        
    Returns:
        Statistics dict
    """
    # Parse input filename to get dataset name
    input_path = Path(input_file)
    filename = input_path.stem  # e.g., "111.train" or "222.val"
    
    # Remove extension suffix (e.g., "111.train.jsonl" -> "111")
    dataset_name = filename.split('.')[0]
    
    # Determine split type (train/val)
    if '.train' in filename or 'train' in input_path.parent.name:
        split = 'train'
    elif '.val' in filename or 'val' in input_path.parent.name:
        split = 'val'
    else:
        split = 'data'
    
    # Create output directory: output_dir/split/dataset_name/
    output_path = Path(output_dir) / split / dataset_name
    
    # Check if already processed
    if skip_existing and output_path.exists():
        metadata_file = output_path / '_metadata.json'
        if metadata_file.exists():
            logging.info(f"⏭️  Skipping {input_file} (already processed)")
            with open(metadata_file, 'r') as f:
                return json.load(f)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"📂 Processing: {input_file}")
    logging.info(f"   Output: {output_path}")
    
    # Load all samples
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                samples.append(item)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse line: {line[:100]}")
                continue
    
    logging.info(f"   Loaded {len(samples)} samples")
    
    # Shuffle if requested
    if shuffle:
        np.random.shuffle(samples)
        logging.info(f"   ✓ Shuffled samples")
    
    # Process samples with multiprocessing
    if skip_negatives:
        logging.info(f"   Tokenizing with {num_workers} workers (skipping negatives for 8x speedup)...")
    else:
        logging.info(f"   Tokenizing with {num_workers} workers...")
    
    processed_samples = []
    failed_count = 0
    
    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(model_dir,)
        ) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_sample, sample, idx, skip_negatives): idx 
                for idx, sample in enumerate(samples)
            }
            
            # Collect results with progress bar
            with tqdm(total=len(samples), desc="   Tokenizing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        processed_samples.append(result)
                    else:
                        failed_count += 1
                    pbar.update(1)
    else:
        # Single process mode
        init_worker(model_dir)
        for idx, sample in enumerate(tqdm(samples, desc="   Tokenizing")):
            result = process_sample(sample, idx, skip_negatives)
            if result is not None:
                processed_samples.append(result)
            else:
                failed_count += 1
    
    logging.info(f"   ✓ Tokenized {len(processed_samples)} samples ({failed_count} failed)")
    
    if not processed_samples:
        logging.error(f"   ❌ No valid samples to write!")
        return {'error': 'No valid samples'}
    
    # Sort by sample_idx to maintain consistent ordering within partitions
    processed_samples.sort(key=lambda x: x['sample_idx'])
    
    # Write to parquet partitions
    schema = create_parquet_schema()
    partition_count = 0
    total_rows = 0
    
    logging.info(f"   Writing parquet files...")
    
    for i in range(0, len(processed_samples), rows_per_partition):
        partition_data = processed_samples[i:i + rows_per_partition]
        partition_file = output_path / f"part-{partition_count:05d}.parquet"
        
        rows_written = write_parquet_partition(partition_data, str(partition_file), schema)
        total_rows += rows_written
        partition_count += 1
    
    logging.info(f"   ✓ Wrote {partition_count} partition files")
    
    # Calculate statistics
    avg_query_text_len = np.mean([s['query_text_token_len'] for s in processed_samples])
    avg_query_speech_len = np.mean([s['query_speech_token_len'] for s in processed_samples])
    avg_positive_speech_len = np.mean([s['positive_speech_token_len'] for s in processed_samples])
    
    num_with_negatives = sum(1 for s in processed_samples if 'negative_speech_tokens' in s)
    avg_negatives_per_sample = np.mean([
        len(s.get('negative_speech_tokens', [])) for s in processed_samples
    ])
    
    # Write metadata
    metadata = {
        'input_file': str(input_file),
        'dataset_name': dataset_name,
        'split': split,
        'total_samples': len(samples),
        'valid_samples': len(processed_samples),
        'failed_samples': failed_count,
        'partition_count': partition_count,
        'rows_per_partition': rows_per_partition,
        'shuffled': shuffle,
        'statistics': {
            'avg_query_text_token_len': float(avg_query_text_len),
            'avg_query_speech_token_len': float(avg_query_speech_len),
            'avg_positive_speech_token_len': float(avg_positive_speech_len),
            'samples_with_negatives': int(num_with_negatives),
            'avg_negatives_per_sample': float(avg_negatives_per_sample),
        }
    }
    
    metadata_file = output_path / '_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logging.info(f"   ✓ Metadata saved to {metadata_file}")
    logging.info(f"   ✅ Complete: {len(processed_samples)} samples in {partition_count} partitions\n")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Tokenize JSONL files to Parquet format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file or directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for parquet files')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Model directory for CosyVoice frontend')
    parser.add_argument('--rows_per_partition', type=int, default=1000,
                        help='Number of rows per partition file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle samples')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip files that are already processed')
    parser.add_argument('--skip_negatives', action='store_true',
                        help='Skip tokenizing hard negatives (use in-batch negatives during training instead). 8x faster!')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        metadata = process_jsonl_file(
            str(input_path),
            args.output_dir,
            args.model_dir,
            args.rows_per_partition,
            not args.no_shuffle,
            args.num_workers,
            args.skip_existing,
            args.skip_negatives
        )
        
    elif input_path.is_dir():
        # Directory - process all JSONL files
        jsonl_files = sorted(input_path.glob('**/*.jsonl'))
        
        if not jsonl_files:
            logging.error(f"No JSONL files found in {input_path}")
            return
        
        logging.info(f"Found {len(jsonl_files)} JSONL files\n")
        
        all_metadata = []
        for jsonl_file in jsonl_files:
            metadata = process_jsonl_file(
                str(jsonl_file),
                args.output_dir,
                args.model_dir,
                args.rows_per_partition,
                not args.no_shuffle,
                args.num_workers,
                args.skip_existing,
                args.skip_negatives
            )
            all_metadata.append(metadata)
        
        # Write summary
        summary_file = Path(args.output_dir) / '_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(jsonl_files),
                'files': all_metadata
            }, f, indent=2, ensure_ascii=False)
        
        logging.info(f"\n✅ All files processed! Summary: {summary_file}")
        
    else:
        logging.error(f"Input path does not exist: {input_path}")


if __name__ == '__main__':
    main()
