"""
Prepare semantic similarity data from KaLM embedding dataset.

This script processes KaLM parquet files and generates synthetic audio samples using CosyVoice3.

Input: KaLM parquet files in D:\data\embedding_data\KaLM-embedding-finetuning-data
  Each parquet has columns:
    - query: "Instruct: Retrieve semantically similar text.\n Query: <text>"
    - pos: List[str] of positive examples
    - neg: List[str] of negative examples

Process:
1. Extract text after "Query: " from query/pos/neg
2. Truncate to max 256 tokens
3. Sample random speakers from Emilia dataset (EN*/ZH* based on language)
4. Synthesize all audio using CosyVoice3 zero-shot cloning
5. Keep only audio <=30s
6. Generate train.jsonl (K=1000) and val.jsonl (K=50) per dataset

Output structure:
    kalm_dataset_name/
        synthetic/                        - All synthetic audio files
        kalm_dataset_name.train.jsonl    - Train samples (max K=1000)
        kalm_dataset_name.val.jsonl      - Val samples (max K=50)
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import os
import json
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import argh
from tqdm import tqdm
import torch
import torchaudio
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from cosyvoice.cli.cosyvoice import AutoModel

import logging

# Set logging level to INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class KaLMDataPreparation:
    """Prepare semantic similarity training data from KaLM parquet files."""
    
    def __init__(
        self,
        kalm_data_dir: str,
        emilia_dirs: List[str],
        cosyvoice_model_dir: str,
        output_base_dir: str,
        max_train_samples: int = 1000,
        max_val_samples: int = 50,
        max_negatives_per_sample: int = 4,
        checkpoint_file: str = "progress-kalm.json",
        skip_datasets: List[str] = []
    ):
        self.kalm_data_dir = Path(kalm_data_dir)
        self.emilia_dirs = [Path(d) for d in emilia_dirs]
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_negatives_per_sample = max_negatives_per_sample
        self.checkpoint_file = self.output_base_dir / checkpoint_file
        
        # Initialize CosyVoice3
        logging.info(f"Loading CosyVoice3 from {cosyvoice_model_dir}...")
        self.cosyvoice = AutoModel(model_dir=cosyvoice_model_dir, llm_ckpt=os.path.join(cosyvoice_model_dir, 'llm.rl.pt'))
        logging.info("CosyVoice3 loaded successfully")
        
        # Load Emilia speaker pool
        self.speaker_pool = self.load_emilia_speaker_pool()
        
        # Load checkpoint if exists
        self.processed_datasets = set()
        self.processed_datasets.update(skip_datasets)
        self.skip_datasets = skip_datasets if skip_datasets else []
        self.processed_samples = {}  # {dataset_name: {split_name: [original_indices]}}
        self.load_checkpoint()
        
        # Tokenizer for text truncation (simple whitespace-based for now)
        self.max_tokens = 256
    
    def load_checkpoint(self):
        """Load progress checkpoint."""
        if self.checkpoint_file.exists():
            logging.info(f"Loading checkpoint from {self.checkpoint_file}")
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.processed_datasets = set(checkpoint.get('processed_datasets', []))
                self.processed_datasets.update(self.skip_datasets)
                self.processed_samples = checkpoint.get('processed_samples', {})
            
            logging.info(f"Resumed: {len(self.processed_datasets)} datasets already processed")
            for dataset, splits in self.processed_samples.items():
                for split, samples in splits.items():
                    logging.info(f"  {dataset}/{split}: {len(samples)} samples already processed")
    
    def save_checkpoint(self):
        """Save progress checkpoint."""
        checkpoint = {
            'processed_datasets': list(self.processed_datasets),
            'processed_samples': self.processed_samples,
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_emilia_speaker_pool(self) -> Dict[str, List[Dict]]:
        """
        Load speaker pool from Emilia datasets.
        Each speaker keeps at most 1 audio sample.
        
        Optimized: Check filenames first before loading metadata.
        
        Returns:
            Dictionary mapping language ('en'/'zh') to list of speaker samples
        """
        speaker_pool = {'en': [], 'zh': []}
        speaker_seen = {'en': set(), 'zh': set()}  # Track which speakers we've seen
        
        logging.info("Loading Emilia speaker pool (1 audio per speaker)...")
        for emilia_dir in self.emilia_dirs:
            if not emilia_dir.exists():
                logging.warning(f"Emilia directory not found: {emilia_dir}")
                continue
            
            # Determine language from directory name
            dir_name = emilia_dir.name.upper()
            if 'EN' in dir_name:
                lang = 'en'
            elif 'ZH' in dir_name:
                lang = 'zh'
            else:
                logging.warning(f"Cannot determine language for {emilia_dir}")
                continue
            
            # Load all JSON metadata files
            json_files = list(emilia_dir.glob("*.json"))
            logging.info(f"  {emilia_dir.name}: Found {len(json_files)} files, filtering to unique speakers...")
            
            added_count = 0
            skipped_count = 0
            
            for json_file in json_files:
                # Extract speaker ID from filename FIRST (before loading metadata)
                filename = json_file.stem
                parts = filename.split('_')
                speaker_id = parts[1] if len(parts) >= 3 else filename
                
                # Skip if we already have this speaker - NO FILE I/O needed!
                if speaker_id in speaker_seen[lang]:
                    skipped_count += 1
                    continue
                
                # Only load metadata if this is a new speaker
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load {json_file}: {e}")
                    continue
                
                mp3_file = json_file.with_suffix('.mp3')
                if not mp3_file.exists():
                    continue
                
                sample = {
                    'speaker_id': speaker_id,
                    'audio_path': str(mp3_file),
                    'text': metadata.get('text', '').strip(),
                    'duration': metadata.get('duration', 0),
                    'language': metadata.get('language', lang)
                }
                
                # Filter out empty or very short text, or too long/short audio
                if len(sample['text']) >= 10 and 1.0 <= sample['duration'] <= 15.0:
                    speaker_pool[lang].append(sample)
                    speaker_seen[lang].add(speaker_id)
                    added_count += 1
            
            logging.info(f"    Added {added_count} unique speakers, skipped {skipped_count} duplicates")
        
        logging.info(f"Loaded speaker pool (unique speakers): EN={len(speaker_pool['en'])}, ZH={len(speaker_pool['zh'])}")
        return speaker_pool
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from text (simple heuristic).
        
        Args:
            text: Input text
            
        Returns:
            'zh' for Chinese, 'en' for English
        """
        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return 'zh' if chinese_chars > len(text) * 0.3 else 'en'
    
    def extract_query_text(self, full_text: str) -> str:
        """
        Extract text after 'Query: ' and truncate to max tokens.
        
        Args:
            full_text: Full text like "Instruct: ... Query: <text>"
            
        Returns:
            Extracted and truncated text
        """
        # Extract after "Query: "
        if "Query: " in full_text:
            text = full_text.split("Query: ", 1)[1].strip()
        else:
            text = full_text.strip()
        
        # Simple tokenization (split by whitespace and punctuation)
        # For Chinese, each character is roughly a token
        if self.detect_language(text) == 'zh':
            # For Chinese, truncate by characters
            if len(text) > self.max_tokens:
                text = text[:self.max_tokens]
        else:
            # For English, truncate by words
            words = text.split()
            if len(words) > self.max_tokens:
                text = ' '.join(words[:self.max_tokens])
        
        return text
    
    def extract_instruction(self, full_text: str) -> str:
        """
        Extract instruction and replace 'text' with 'voice'.
        
        Args:
            full_text: Full text like "Instruct: Retrieve semantically similar text.\n Query: ..."
            
        Returns:
            Modified instruction like "Retrieve semantically similar voice"
        """
        # Extract between "Instruct: " and "\n"
        match = re.search(r'Instruct:\s*(.+?)\s*\n', full_text)
        if match:
            instruction = match.group(1).strip()
            # Replace 'text' with 'voice'
            instruction = instruction.replace('text', 'voice')
            return instruction
        
        # Default instruction
        return "Retrieve semantically similar voice"
    
    def synthesize_voice(
        self,
        text: str,
        reference_sample: Dict,
        output_path: str
    ) -> Tuple[bool, float]:
        """
        Synthesize voice using CosyVoice3 zero-shot cloning.
        
        Args:
            text: Text to synthesize
            reference_sample: Reference speaker sample (dict with 'audio_path' and 'text')
            output_path: Output audio path
            
        Returns:
            (success, duration) tuple
        """
        try:
            # Use zero-shot inference
            prompt = "You are a helpful assistant."
            
            # Generate audio
            for i, result in enumerate(self.cosyvoice.inference_zero_shot(
                text,
                f"{prompt}<|endofprompt|>{reference_sample['text']}",
                reference_sample['audio_path'],
                stream=False
            )):
                # Save the synthesized audio
                torchaudio.save(
                    output_path,
                    result['tts_speech'],
                    self.cosyvoice.sample_rate
                )
                
                # Calculate duration
                duration = result['tts_speech'].shape[1] / self.cosyvoice.sample_rate
                
                return True, duration
            
            return False, 0.0
            
        except Exception as e:
            logging.error(f"Error synthesizing voice: {e}")
            return False, 0.0
    
    def process_parquet_file(
        self,
        parquet_file: Path,
        output_dir: Path,
        dataset_name: str,
        val_ratio: float = 0.05,
        seed: int = 42
    ):
        """
        Process a single parquet file.
        
        Args:
            parquet_file: Path to parquet file
            output_dir: Output directory for this dataset
            dataset_name: Name of the dataset
            val_ratio: Validation ratio (default 5%)
            seed: Random seed
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Processing: {dataset_name}")
        logging.info(f"{'='*80}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        synthetic_dir = output_dir / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        # Initialize processed samples tracking for this dataset
        if dataset_name not in self.processed_samples:
            self.processed_samples[dataset_name] = {'train': [], 'val': []}
        
        # Read parquet file
        logging.info(f"Reading parquet file: {parquet_file}")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Add original_index column BEFORE any shuffling
        df['original_index'] = df.index.tolist()
        
        logging.info(f"Total samples in parquet: {len(df)}")
        
        # Shuffle with fixed seed
        df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        # Split into train/val
        num_val = max(1, int(len(df_shuffled) * val_ratio))
        df_val = df_shuffled[:num_val]
        df_train = df_shuffled[num_val:]
        
        # Limit samples
        df_train = df_train[:self.max_train_samples]
        df_val = df_val[:self.max_val_samples]
        
        logging.info(f"Train samples: {len(df_train)} (max: {self.max_train_samples})")
        logging.info(f"Val samples: {len(df_val)} (max: {self.max_val_samples})")
        
        # Process train and val
        for split_name, df_split in [('train', df_train), ('val', df_val)]:
            output_jsonl = output_dir / f"{dataset_name}.{split_name}.jsonl"
            
            # Get already processed samples for this split
            processed_set = set(self.processed_samples[dataset_name][split_name])
            
            logging.info(f"\nProcessing {split_name} split...")
            logging.info(f"  Already processed: {len(processed_set)} samples")
            
            # Open in append mode if file exists and has content
            mode = 'a' if output_jsonl.exists() and output_jsonl.stat().st_size > 0 else 'w'
            
            with open(output_jsonl, mode, encoding='utf-8') as f_out:
                for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"  {split_name}"):
                    try:
                        # Get original index for resume tracking
                        original_idx = int(row['original_index'])
                        
                        # Skip if already processed
                        if original_idx in processed_set:
                            continue
                        
                        # Extract texts
                        query_full = row['query']
                        
                        # Convert pos to list of strings
                        pos_raw = row['pos']
                        if isinstance(pos_raw, np.ndarray):
                            pos_list = pos_raw.tolist()
                        elif isinstance(pos_raw, list):
                            pos_list = pos_raw
                        else:
                            pos_list = [str(pos_raw)]
                        
                        # Convert neg to list of strings
                        neg_raw = row['neg']
                        if isinstance(neg_raw, np.ndarray):
                            neg_list = neg_raw.tolist()
                        elif isinstance(neg_raw, list):
                            neg_list = neg_raw
                        else:
                            neg_list = [str(neg_raw)]
                        
                        # Limit negatives to max_negatives_per_sample with deterministic sampling
                        if len(neg_list) > self.max_negatives_per_sample:
                            # Use random seed based on original_idx for reproducibility
                            rng = random.Random(seed + original_idx)
                            neg_list = rng.sample(neg_list, self.max_negatives_per_sample)
                        
                        # Convert query to string if needed
                        if isinstance(query_full, np.ndarray):
                            query_full = str(query_full)
                        
                        # Extract query text and instruction
                        query_text = self.extract_query_text(query_full)
                        instruction = self.extract_instruction(query_full)
                        
                        # Detect language
                        lang = self.detect_language(query_text)
                        
                        # Check if we have speakers for this language
                        if not self.speaker_pool[lang]:
                            logging.warning(f"  No speakers available for language: {lang}")
                            continue
                        
                        # Sample random speakers (deterministic based on original_idx)
                        rng = random.Random(seed + original_idx)
                        query_speaker = rng.choice(self.speaker_pool[lang])
                        pos_speaker = rng.choice(self.speaker_pool[lang])
                        
                        # Generate query audio
                        query_filename = f"{dataset_name}_{split_name}_query_{original_idx}.wav"
                        query_path = synthetic_dir / query_filename
                        
                        if not query_path.exists():
                            success, duration = self.synthesize_voice(
                                query_text, query_speaker, str(query_path)
                            )
                            if not success or duration > 30.0:
                                if duration > 30.0:
                                    logging.warning(f"  Query audio too long: {duration:.1f}s")
                                    query_path.unlink(missing_ok=True)
                                continue
                        
                        # Generate positive audio (take first positive)
                        pos_text = self.extract_query_text(pos_list[0])
                        pos_filename = f"{dataset_name}_{split_name}_pos_{original_idx}.wav"
                        pos_path = synthetic_dir / pos_filename
                        
                        if not pos_path.exists():
                            success, duration = self.synthesize_voice(
                                pos_text, pos_speaker, str(pos_path)
                            )
                            if not success or duration > 30.0:
                                if duration > 30.0:
                                    logging.warning(f"  Positive audio too long: {duration:.1f}s")
                                    pos_path.unlink(missing_ok=True)
                                continue
                        
                        # Generate negative audios
                        neg_paths = []
                        for neg_idx, neg_full in enumerate(neg_list):
                            neg_text = self.extract_query_text(neg_full)
                            neg_filename = f"{dataset_name}_{split_name}_neg_{original_idx}_{neg_idx}.wav"
                            neg_path = synthetic_dir / neg_filename
                            
                            if not neg_path.exists():
                                neg_speaker = rng.choice(self.speaker_pool[lang])
                                success, duration = self.synthesize_voice(
                                    neg_text, neg_speaker, str(neg_path)
                                )
                                if not success or duration > 30.0:
                                    if duration > 30.0:
                                        logging.warning(f"  Negative audio too long: {duration:.1f}s")
                                        neg_path.unlink(missing_ok=True)
                                    continue
                            
                            neg_paths.append(str(neg_path))
                        
                        # Skip if not enough negatives
                        if len(neg_paths) == 0:
                            logging.warning(f"  No valid negatives for sample {original_idx}")
                            continue
                        
                        # Create training sample
                        training_sample = {
                            'query_text': instruction,
                            'query_wav': str(query_path),
                            'pos_wav': str(pos_path),
                            'neg_wavs': neg_paths,
                            'task_type': 'semantic_similarity',
                            'metadata': {
                                'dataset': dataset_name,
                                'language': lang,
                                'original_index': original_idx,  # Track original parquet index
                                'query_transcript': query_text,
                                'pos_transcript': pos_text,
                                'num_negatives': len(neg_paths)
                            }
                        }
                        
                        # Write to JSONL
                        f_out.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                        # Mark as processed and save checkpoint periodically
                        processed_set.add(original_idx)
                        self.processed_samples[dataset_name][split_name].append(original_idx)
                        
                        # Save checkpoint every 10 samples
                        if len(self.processed_samples[dataset_name][split_name]) % 10 == 0:
                            self.save_checkpoint()
                        
                    except Exception as e:
                        logging.error(f"  Error processing sample {idx} (original: {original_idx}): {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            logging.info(f"  Completed {split_name} split: {output_jsonl}")
        
        # Final checkpoint save for this dataset
        self.save_checkpoint()
        
        # Mark dataset as completed
        self.processed_datasets.add(dataset_name)
        self.save_checkpoint()
    
    def process_all_datasets(self, val_ratio: float = 0.05, seed: int = 42):
        """
        Process all parquet datasets in the KaLM data directory.
        
        Args:
            val_ratio: Validation ratio
            seed: Random seed
        """
        # Find all dataset directories
        dataset_dirs = [d for d in self.kalm_data_dir.iterdir() if d.is_dir()]
        
        logging.info(f"Found {len(dataset_dirs)} dataset directories")
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            
            # Skip if already processed
            if dataset_name in self.processed_datasets:
                logging.info(f"Skipping {dataset_name} (already processed)")
                continue
            
            # Find parquet files in this directory
            parquet_files = list(dataset_dir.glob("*.parquet"))
            
            if not parquet_files:
                logging.warning(f"No parquet files found in {dataset_dir}")
                continue
            
            # Use first parquet file (assuming one per dataset)
            parquet_file = parquet_files[0]
            
            # Create output directory for this dataset
            output_dir = self.output_base_dir / dataset_name
            
            # Process this dataset
            self.process_parquet_file(
                parquet_file,
                output_dir,
                dataset_name,
                val_ratio=val_ratio,
                seed=seed
            )
        
        logging.info(f"\n{'='*80}")
        logging.info("All datasets processed!")
        logging.info(f"{'='*80}")
        logging.info(f"Output directory: {self.output_base_dir}")


def prepare_kalm_data(
    kalm_data_dir: str = r"D:\data\embedding_data\KaLM-embedding-finetuning-data",
    emilia_en_dirs: str = r"D:\TTS-data\Emilia_Yodas\EN-B000000",
    emilia_zh_dirs: str = r"D:\TTS-data\Emilia_Yodas\ZH-B000000",
    cosyvoice_model_dir: str = r"D:\models\cosyvoice_models\CosyVoice3-0.5B-2512",
    output_dir: str = r"D:\data\embedding_data\OUTPUT-kalm",
    max_train_samples: int = 1000,
    max_val_samples: int = 50,
    max_negatives_per_sample: int = 4,
    val_ratio: float = 0.05,
    seed: int = 42,
    skip_datasets: str = ''
):
    """
    Prepare semantic similarity data from KaLM parquet files.
    
    Args:
        kalm_data_dir: Directory containing KaLM datasets
        emilia_en_dirs: Comma-separated English Emilia directories (supports glob patterns like EN-B*)
        emilia_zh_dirs: Comma-separated Chinese Emilia directories (supports glob patterns like ZH-B*)
        cosyvoice_model_dir: Path to CosyVoice3 model directory
        output_dir: Output directory
        max_train_samples: Maximum training samples per dataset
        max_val_samples: Maximum validation samples per dataset
        max_negatives_per_sample: Maximum number of negatives to keep per sample (default: 4)
        val_ratio: Validation ratio
        seed: Random seed
    """
    import glob as glob_module
    
    random.seed(seed)
    torch.manual_seed(seed)

    skip_datasets = skip_datasets.split(',') if skip_datasets else []
    
    # Parse Emilia directories with glob support
    emilia_dirs = []
    
    # Process EN directories
    for dir_pattern in emilia_en_dirs.split(','):
        dir_pattern = dir_pattern.strip()
        
        # Check if it's a glob pattern
        if '*' in dir_pattern or '?' in dir_pattern:
            # Expand glob pattern
            matched_dirs = glob_module.glob(dir_pattern)
            if matched_dirs:
                logging.info(f"Glob pattern '{dir_pattern}' matched {len(matched_dirs)} directories")
                emilia_dirs.extend(matched_dirs)
            else:
                logging.warning(f"Glob pattern '{dir_pattern}' matched no directories")
        else:
            # Direct path
            emilia_dirs.append(dir_pattern)
    
    # Process ZH directories
    for dir_pattern in emilia_zh_dirs.split(','):
        dir_pattern = dir_pattern.strip()
        
        # Check if it's a glob pattern
        if '*' in dir_pattern or '?' in dir_pattern:
            # Expand glob pattern
            matched_dirs = glob_module.glob(dir_pattern)
            if matched_dirs:
                logging.info(f"Glob pattern '{dir_pattern}' matched {len(matched_dirs)} directories")
                emilia_dirs.extend(matched_dirs)
            else:
                logging.warning(f"Glob pattern '{dir_pattern}' matched no directories")
        else:
            # Direct path
            emilia_dirs.append(dir_pattern)
    
    if not emilia_dirs:
        raise ValueError("No Emilia directories found! Check your --emilia-en-dirs and --emilia-zh-dirs paths.")
    
    logging.info(f"Total Emilia directories to process: {len(emilia_dirs)}")
    
    preparer = KaLMDataPreparation(
        kalm_data_dir=kalm_data_dir,
        emilia_dirs=emilia_dirs,
        cosyvoice_model_dir=cosyvoice_model_dir,
        output_base_dir=output_dir,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_negatives_per_sample=max_negatives_per_sample,
        skip_datasets=skip_datasets
    )
    
    preparer.process_all_datasets(val_ratio=val_ratio, seed=seed)


if __name__ == '__main__':
    argh.dispatch_command(prepare_kalm_data)
