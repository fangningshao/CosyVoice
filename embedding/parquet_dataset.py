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
Fast Parquet-based dataset for voice embedding training.

This dataset reads pre-tokenized data from Parquet files for maximum throughput.
No on-the-fly tokenization, no file I/O bottleneck.
"""

import os
import json
import torch
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset, IterableDataset
import logging


class ParquetVoiceEmbeddingDataset(Dataset):
    """
    Fast dataset that reads pre-tokenized data from Parquet files.
    
    Expected directory structure:
        parquet_dir/
            train/ or val/
                111/
                    part-00000.parquet
                    part-00001.parquet
                    _metadata.json
                222/
                    part-00000.parquet
                    _metadata.json
    """
    
    def __init__(self,
                 parquet_dir: str,
                 split: str = 'train',
                 datasets: Optional[List[str]] = None,
                 use_hard_negatives: bool = True,
                 max_negatives: int = 7,
                 shuffle_datasets: bool = True,
                 seed: int = 42):
        """
        Args:
            parquet_dir: Root directory containing parquet files
            split: 'train' or 'val'
            datasets: List of dataset names to load (e.g., ['111', '222']). If None, load all.
            use_hard_negatives: Whether to use hard negatives
            max_negatives: Maximum number of hard negatives per sample. -1 = use all available.
                          If > 0 and < available, randomly sample max_negatives from available (different each epoch).
            shuffle_datasets: Whether to shuffle across datasets
            seed: Random seed for shuffling
        """
        if split:
            self.parquet_dir = Path(parquet_dir) / split
        else:
            self.parquet_dir = Path(parquet_dir)
        self.split = split
        self.use_hard_negatives = use_hard_negatives
        self.max_negatives = max_negatives
        self.shuffle_datasets = shuffle_datasets
        self.seed = seed
        
        if not self.parquet_dir.exists():
            raise ValueError(f"Parquet directory not found: {self.parquet_dir}")
        
        # Discover datasets
        if datasets is None:
            # Auto-discover all dataset directories
            dataset_dirs = [d for d in self.parquet_dir.iterdir() if d.is_dir()]
        else:
            dataset_dirs = [self.parquet_dir / ds for ds in datasets]
        
        # Load all parquet files
        self.parquet_files = []
        self.dataset_info = {}
        
        for dataset_dir in sorted(dataset_dirs):
            if not dataset_dir.exists():
                logging.warning(f"Dataset directory not found: {dataset_dir}")
                continue
            
            dataset_name = dataset_dir.name
            
            # Load metadata
            metadata_file = dataset_dir / '_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.dataset_info[dataset_name] = metadata
            
            # Find all parquet files
            parquet_files = sorted(dataset_dir.glob('part-*.parquet'))
            self.parquet_files.extend(parquet_files)
            
            logging.info(f"  {dataset_name}: {len(parquet_files)} partitions")
        
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {self.parquet_dir}")
        
        # Shuffle parquet files for better mixing
        if self.shuffle_datasets:
            np.random.seed(self.seed)
            np.random.shuffle(self.parquet_files)
        
        # Build index: (file_idx, row_idx) for each sample
        logging.info(f"Building index for {len(self.parquet_files)} parquet files...")
        self.index = []
        
        for file_idx, parquet_file in enumerate(self.parquet_files):
            parquet_table = pq.read_table(parquet_file)
            num_rows = len(parquet_table)
            
            for row_idx in range(num_rows):
                self.index.append((file_idx, row_idx))
        
        logging.info(f"Loaded {len(self.index)} samples from {len(self.parquet_files)} files")
        logging.info(f"  Datasets: {list(self.dataset_info.keys())}")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index."""
        file_idx, row_idx = self.index[idx]
        parquet_file = self.parquet_files[file_idx]
        
        # Read the specific row (PyArrow is efficient for random access)
        table = pq.read_table(parquet_file)
        row = table.slice(row_idx, 1).to_pydict()
        
        # Convert to tensors
        result = {
            'query_text_token': torch.tensor(row['query_text_token'][0], dtype=torch.long),
            'query_text_token_len': torch.tensor(row['query_text_token_len'][0], dtype=torch.long),
            'query_speech_token': torch.tensor(row['query_speech_token'][0], dtype=torch.long),
            'query_speech_token_len': torch.tensor(row['query_speech_token_len'][0], dtype=torch.long),
            'positive_text_token': torch.tensor(row['positive_text_token'][0], dtype=torch.long),
            'positive_text_token_len': torch.tensor(row['positive_text_token_len'][0], dtype=torch.long),
            'positive_speech_token': torch.tensor(row['positive_speech_token'][0], dtype=torch.long),
            'positive_speech_token_len': torch.tensor(row['positive_speech_token_len'][0], dtype=torch.long),
        }
        
        # Handle negatives with optional random sampling
        if self.use_hard_negatives:
            neg_speech_tokens = row['negative_speech_tokens'][0]
            neg_speech_token_lens = row['negative_speech_token_lens'][0]
            
            if neg_speech_tokens and len(neg_speech_tokens) > 0:
                num_available = len(neg_speech_tokens)
                
                # Determine how many negatives to use
                if self.split == 'val':
                    # Validation: deterministic, use first max_negatives (for speed and consistency)
                    if self.max_negatives > 0:
                        num_negatives = min(num_available, self.max_negatives)
                        sampled_indices = list(range(num_negatives))
                    else:
                        # max_negatives = -1: use all
                        num_negatives = num_available
                        sampled_indices = list(range(num_negatives))
                else:
                    # Training: random sampling if max_negatives is set and less than available
                    if self.max_negatives > 0 and self.max_negatives < num_available:
                        # Randomly sample max_negatives from available (different each access)
                        num_negatives = self.max_negatives
                        sampled_indices = np.random.choice(num_available, num_negatives, replace=False).tolist()
                    else:
                        # Use all negatives (-1 or max_negatives >= num_available)
                        num_negatives = num_available
                        sampled_indices = list(range(num_negatives))
                
                # Note: positive_text_token is reused for all negatives (instruction prefix)
                result['negative_text_tokens'] = [result['positive_text_token'] for _ in range(num_negatives)]
                result['negative_text_token_lens'] = [result['positive_text_token_len'] for _ in range(num_negatives)]
                result['negative_speech_tokens'] = [
                    torch.tensor(neg_speech_tokens[i], dtype=torch.long) for i in sampled_indices
                ]
                result['negative_speech_token_lens'] = [
                    torch.tensor(neg_speech_token_lens[i], dtype=torch.long) for i in sampled_indices
                ]
        
        return result


class StreamingParquetDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that iterates through parquet files.
    Better for very large datasets that don't fit in memory.
    """
    
    def __init__(self,
                 parquet_dir: str,
                 split: str = 'train',
                 datasets: Optional[List[str]] = None,
                 use_hard_negatives: bool = True,
                 max_negatives: int = 7,
                 shuffle_files: bool = True,
                 buffer_size: int = 10000,
                 seed: int = 42):
        """
        Args:
            parquet_dir: Root directory containing parquet files
            split: 'train' or 'val'
            datasets: List of dataset names to load (e.g., ['111', '222']). If None, load all.
            use_hard_negatives: Whether to use hard negatives
            max_negatives: Maximum number of hard negatives per sample
            shuffle_files: Whether to shuffle file order
            buffer_size: Size of shuffle buffer (larger = better randomness, more memory)
            seed: Random seed
        """
        super().__init__()
        self.parquet_dir = Path(parquet_dir) / split
        self.split = split
        self.use_hard_negatives = use_hard_negatives
        self.max_negatives = max_negatives
        self.shuffle_files = shuffle_files
        self.buffer_size = buffer_size
        self.seed = seed
        
        if not self.parquet_dir.exists():
            raise ValueError(f"Parquet directory not found: {self.parquet_dir}")
        
        # Discover datasets
        if datasets is None:
            dataset_dirs = [d for d in self.parquet_dir.iterdir() if d.is_dir()]
        else:
            dataset_dirs = [self.parquet_dir / ds for ds in datasets]
        
        # Collect all parquet files
        self.parquet_files = []
        for dataset_dir in sorted(dataset_dirs):
            if not dataset_dir.exists():
                continue
            parquet_files = sorted(dataset_dir.glob('part-*.parquet'))
            self.parquet_files.extend(parquet_files)
        
        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {self.parquet_dir}")
        
        logging.info(f"Found {len(self.parquet_files)} parquet files for streaming")
    
    def __iter__(self):
        """Iterate through samples with optional shuffling."""
        # Shuffle files at start of each epoch
        file_order = list(range(len(self.parquet_files)))
        if self.shuffle_files:
            # Use worker-specific seed for better randomness
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed + (worker_info.id if worker_info else 0)
            rng = np.random.RandomState(seed)
            rng.shuffle(file_order)
        
        # Shuffle buffer for within-file shuffling
        buffer = []
        
        for file_idx in file_order:
            parquet_file = self.parquet_files[file_idx]
            table = pq.read_table(parquet_file)
            
            for row_idx in range(len(table)):
                row = table.slice(row_idx, 1).to_pydict()
                
                # Convert to tensors
                sample = {
                    'query_text_token': torch.tensor(row['query_text_token'][0], dtype=torch.long),
                    'query_text_token_len': torch.tensor(row['query_text_token_len'][0], dtype=torch.long),
                    'query_speech_token': torch.tensor(row['query_speech_token'][0], dtype=torch.long),
                    'query_speech_token_len': torch.tensor(row['query_speech_token_len'][0], dtype=torch.long),
                    'positive_text_token': torch.tensor(row['positive_text_token'][0], dtype=torch.long),
                    'positive_text_token_len': torch.tensor(row['positive_text_token_len'][0], dtype=torch.long),
                    'positive_speech_token': torch.tensor(row['positive_speech_token'][0], dtype=torch.long),
                    'positive_speech_token_len': torch.tensor(row['positive_speech_token_len'][0], dtype=torch.long),
                }
                
                # Handle negatives
                if self.use_hard_negatives:
                    neg_speech_tokens = row['negative_speech_tokens'][0]
                    neg_speech_token_lens = row['negative_speech_token_lens'][0]
                    
                    if neg_speech_tokens and len(neg_speech_tokens) > 0:
                        num_negatives = min(len(neg_speech_tokens), self.max_negatives)
                        sampled_indices = np.random.choice(len(neg_speech_tokens), num_negatives, replace=False)
                        
                        sample['negative_text_tokens'] = [sample['positive_text_token'] for _ in range(num_negatives)]
                        sample['negative_text_token_lens'] = [sample['positive_text_token_len'] for _ in range(num_negatives)]
                        sample['negative_speech_tokens'] = [
                            torch.tensor(neg_speech_tokens[i], dtype=torch.long) for i in sampled_indices
                        ]
                        sample['negative_speech_token_lens'] = [
                            torch.tensor(neg_speech_token_lens[i], dtype=torch.long) for i in sampled_indices
                        ]
                
                # Shuffle buffer
                buffer.append(sample)
                if len(buffer) >= self.buffer_size:
                    if self.shuffle_files:
                        np.random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []
        
        # Yield remaining items in buffer
        if buffer:
            if self.shuffle_files:
                np.random.shuffle(buffer)
            for item in buffer:
                yield item


# Keep the same collate_fn from the original dataset
def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function to pad sequences in a batch.
    """
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        logging.warning("All samples in batch failed to load!")
        return None
    
    batch_size = len(batch)
    
    # Simple keys to stack
    result = {
        'query_text_token_len': torch.stack([item['query_text_token_len'] for item in batch]),
        'query_speech_token_len': torch.stack([item['query_speech_token_len'].squeeze() for item in batch]),
        'positive_text_token_len': torch.stack([item['positive_text_token_len'] for item in batch]),
        'positive_speech_token_len': torch.stack([item['positive_speech_token_len'].squeeze() for item in batch]),
    }
    
    # Pad text tokens
    for key in ['query_text_token', 'positive_text_token']:
        tokens = [item[key] for item in batch]
        max_len = max(t.size(-1) for t in tokens) if tokens else 1
        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, t in enumerate(tokens):
            t_flat = t.flatten()
            if t_flat.size(0) > 0:
                padded[i, :t_flat.size(0)] = t_flat
        result[key] = padded
    
    # Pad speech tokens
    for key in ['query_speech_token', 'positive_speech_token']:
        tokens = [item[key] for item in batch]
        flattened_tokens = [t.flatten() for t in tokens]
        max_len = max((t.size(0) for t in flattened_tokens), default=1)
        
        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, t in enumerate(flattened_tokens):
            if t.size(0) > 0:
                padded[i, :t.size(0)] = t
        result[key] = padded
    
    # Handle negatives if present
    if 'negative_speech_tokens' in batch[0]:
        all_neg_text_tokens = []
        all_neg_text_lens = []
        all_neg_speech_tokens = []
        all_neg_speech_lens = []
        neg_counts = []
        
        for item in batch:
            neg_text_tokens = item.get('negative_text_tokens', [])
            neg_text_lens = item.get('negative_text_token_lens', [])
            neg_speech_tokens = item.get('negative_speech_tokens', [])
            neg_speech_lens = item.get('negative_speech_token_lens', [])
            
            neg_counts.append(len(neg_speech_tokens))
            all_neg_text_tokens.extend(neg_text_tokens)
            all_neg_text_lens.extend([l.squeeze() for l in neg_text_lens])
            all_neg_speech_tokens.extend(neg_speech_tokens)
            all_neg_speech_lens.extend([l.squeeze() for l in neg_speech_lens])
        
        if all_neg_speech_tokens:
            # Pad text tokens
            flattened_text = [t.flatten() for t in all_neg_text_tokens]
            max_text_len = max((t.size(0) for t in flattened_text), default=1)
            padded_text = torch.zeros(len(flattened_text), max_text_len, dtype=torch.long)
            for i, t in enumerate(flattened_text):
                if t.size(0) > 0:
                    padded_text[i, :t.size(0)] = t
            
            # Pad speech tokens
            flattened_speech = [t.flatten() for t in all_neg_speech_tokens]
            max_speech_len = max((t.size(0) for t in flattened_speech), default=1)
            padded_speech = torch.zeros(len(flattened_speech), max_speech_len, dtype=torch.long)
            for i, t in enumerate(flattened_speech):
                if t.size(0) > 0:
                    padded_speech[i, :t.size(0)] = t
            
            result['negative_text_token'] = padded_text
            result['negative_text_token_len'] = torch.stack(all_neg_text_lens)
            result['negative_speech_token'] = padded_speech
            result['negative_speech_token_len'] = torch.stack(all_neg_speech_lens)
            result['negative_counts'] = torch.tensor(neg_counts, dtype=torch.long)
    
    return result
