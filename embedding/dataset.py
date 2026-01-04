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
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import logging

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from hyperpyyaml import load_hyperpyyaml


class VoiceEmbeddingDataset(Dataset):
    """
    Pickle-friendly dataset for voice embedding training.
    
    Expected data format (JSON lines):
    {
        "query": "path/to/query_audio.wav",
        "query_instruction": "Retrieve semantically similar voice",
        "positive": "path/to/positive_audio.wav",
        "positive_instruction": "Retrieve semantically similar voice",
        "negatives": ["path/to/neg1.wav", "path/to/neg2.wav", ...]  # Optional
    }
    
    Or from KaLM dataset format:
    {
        "query": "text instruction",
        "pos": ["audio1.wav"],
        "neg": ["audio2.wav", "audio3.wav"]
    }
    """
    
    def __init__(self,
                 data_list_file: str,
                 frontend: Optional[CosyVoiceFrontEnd] = None,
                 model_dir: str = None,
                 sample_rate: int = 22050,
                 use_hard_negatives: bool = True,
                 max_negatives: int = 7,
                 max_duration: float = 30.0,
                 min_duration: float = 1.0):
        """
        Args:
            data_list_file: Path to data list file (JSON lines)
            frontend: CosyVoice frontend for tokenization (not stored, recreated per worker)
            model_dir: Model directory (to initialize frontend if not provided)
            sample_rate: Audio sample rate
            use_hard_negatives: Whether to use hard negatives
            max_negatives: Maximum number of hard negatives to use
            max_duration: Maximum audio duration in seconds (default 30s for CosyVoice)
            min_duration: Minimum audio duration in seconds
        """
        self.data_list_file = data_list_file
        self.sample_rate = sample_rate
        self.use_hard_negatives = use_hard_negatives
        self.max_negatives = max_negatives
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Store model_dir for lazy initialization
        self.model_dir = model_dir
        
        # Don't store frontend directly (not pickle-friendly)
        # Store config path instead
        self.config_path = os.path.join(model_dir, 'cosyvoice3.yaml') if model_dir else None
        
        # These will be lazily initialized per worker
        self._frontend = None
        self._configs = None
        
        # Load data list
        self.data = []
        with open(data_list_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse line: {line}")
                    continue
        
        logging.info(f"Loaded {len(self.data)} samples from {data_list_file}")
        
        # Pre-filter invalid samples
        self.valid_indices = list(range(len(self.data)))
        self._prefilter_samples()
    
    def _check_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        Check audio duration. Returns duration if valid, None otherwise.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds if valid, None if invalid
        """
        if not os.path.exists(audio_path):
            return None
        
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            if duration < self.min_duration or duration > self.max_duration:
                return None
            
            return duration
        except Exception as e:
            logging.warning(f"Error checking audio duration for {audio_path}: {e}")
            return None
    
    def _prefilter_samples(self):
        """Pre-filter samples that are too long, too short, or have missing audio."""
        logging.info("Pre-filtering samples for duration and existence...")
        
        new_valid_indices = []
        skipped_missing = 0
        skipped_too_long = 0
        skipped_too_short = 0
        skipped_error = 0
        
        for idx in self.valid_indices:
            item = self.data[idx]
            
            # Determine audio paths based on format
            if 'query_text' in item and 'query_wav' in item:
                # Standard format
                query_audio = item.get('query_wav')
                positive_audio = item.get('pos_wav')
                negative_audios = item.get('neg_wavs', [])
            elif 'query' in item and 'pos' in item:
                # KaLM format
                query_audio = item.get('query_wav')
                positive_audio = item['pos'][0] if isinstance(item['pos'], list) else item['pos']
                negative_audios = item.get('neg', [])
            else:
                # Old format
                query_audio = item.get('query')
                positive_audio = item.get('positive')
                negative_audios = item.get('negatives', [])
            
            # Check query audio (if exists)
            if query_audio:
                duration = self._check_audio_duration(query_audio)
                if duration is None:
                    if not os.path.exists(query_audio):
                        skipped_missing += 1
                    else:
                        info = torchaudio.info(query_audio)
                        dur = info.num_frames / info.sample_rate
                        if dur > self.max_duration:
                            skipped_too_long += 1
                        elif dur < self.min_duration:
                            skipped_too_short += 1
                        else:
                            skipped_error += 1
                    continue
            
            # Check positive audio
            if not positive_audio:
                skipped_missing += 1
                continue
                
            duration = self._check_audio_duration(positive_audio)
            if duration is None:
                if not os.path.exists(positive_audio):
                    skipped_missing += 1
                else:
                    try:
                        info = torchaudio.info(positive_audio)
                        dur = info.num_frames / info.sample_rate
                        if dur > self.max_duration:
                            skipped_too_long += 1
                        elif dur < self.min_duration:
                            skipped_too_short += 1
                        else:
                            skipped_error += 1
                    except:
                        skipped_error += 1
                continue
            
            # Filter negative audios
            if negative_audios:
                valid_negatives = []
                for neg_audio in negative_audios:
                    duration = self._check_audio_duration(neg_audio)
                    if duration is not None:
                        valid_negatives.append(neg_audio)
                
                # Update the sample with valid negatives
                if 'neg_wavs' in item:
                    item['neg_wavs'] = valid_negatives
                elif 'neg' in item:
                    item['neg'] = valid_negatives
                elif 'negatives' in item:
                    item['negatives'] = valid_negatives
            
            # Sample is valid
            new_valid_indices.append(idx)
        
        self.valid_indices = new_valid_indices
        
        logging.info(f"Pre-filtering complete:")
        logging.info(f"  Valid samples: {len(self.valid_indices)}/{len(self.data)}")
        logging.info(f"  Skipped (missing audio): {skipped_missing}")
        logging.info(f"  Skipped (too long > {self.max_duration}s): {skipped_too_long}")
        logging.info(f"  Skipped (too short < {self.min_duration}s): {skipped_too_short}")
        logging.info(f"  Skipped (error): {skipped_error}")
    
    @property
    def frontend(self):
        """Lazy load frontend per worker (pickle-friendly)."""
        if self._frontend is None:
            if self.config_path is None or not os.path.exists(self.config_path):
                raise ValueError(f"Config path not found: {self.config_path}")
            
            # Load config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._configs = load_hyperpyyaml(f)
            
            # Initialize frontend
            self._frontend = CosyVoiceFrontEnd(
                self._configs['get_tokenizer'],
                self._configs['feat_extractor'],
                os.path.join(self.model_dir, 'campplus.onnx'),
                os.path.join(self.model_dir, 'speech_tokenizer_v3.onnx'),
                os.path.join(self.model_dir, 'spk2info.pt'),
                self._configs['allowed_special']
            )
            
            logging.info(f"Frontend initialized for worker process (PID: {os.getpid()})")
        
        return self._frontend
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getstate__(self):
        """Custom pickle support - exclude non-picklable objects."""
        state = self.__dict__.copy()
        # Remove non-picklable frontend and configs
        state['_frontend'] = None
        state['_configs'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickle support - restore state."""
        self.__dict__.update(state)
        # Frontend will be lazily re-initialized when accessed
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        speech, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            speech = torchaudio.functional.resample(speech, sr, self.sample_rate)
        
        # Convert to mono
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        
        return speech
    
    def tokenize_text(self, text: str) -> tuple:
        """Tokenize text instruction."""
        text_normalized = self.frontend.text_normalize(text, split=False, text_frontend=True)
        text_token = self.frontend.tokenizer.encode(text_normalized, 
                                                    allowed_special=self.frontend.allowed_special)
        text_token = torch.tensor(text_token, dtype=torch.long)  # Always CPU
        text_token_len = torch.tensor(len(text_token), dtype=torch.long)  # Always CPU
        
        return text_token, text_token_len

    def tokenize_speech(self, audio_path: str) -> tuple:
        """Tokenize speech audio with duration check."""
        try:
            # Use frontend's _extract_speech_token method which handles loading internally
            speech_token, speech_token_len = self.frontend._extract_speech_token(audio_path)
            
            # IMPORTANT: Move to CPU before returning (dataset should always return CPU tensors)
            if speech_token.is_cuda:
                speech_token = speech_token.cpu()
            if speech_token_len.is_cuda:
                speech_token_len = speech_token_len.cpu()
            
            return speech_token, speech_token_len
            
        except AssertionError as e:
            # Re-raise with more context about the audio file
            raise AssertionError(f"Audio duration error for {audio_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error tokenizing speech {audio_path}: {e}")
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a training sample.
        
        Returns:
            Dict with tokenized data, or None if sample cannot be processed
        """
        # Get valid sample index
        real_idx = self.valid_indices[idx]
        item = self.data[real_idx]
        
        try:
            # Handle different data formats
            if 'query_text' in item and 'query_wav' in item:
                # Standard format from data_format.py
                query_instruction = item['query_text']
                query_audio = item['query_wav']
                positive_audio = item['pos_wav']
                negative_audios = item.get('neg_wavs', [])
                
            elif 'query' in item and 'pos' in item:
                # KaLM format
                query_instruction = item['query']
                # Check if query is audio path or text
                if isinstance(item.get('query_wav'), str):
                    query_audio = item['query_wav']
                else:
                    query_audio = None  # Query is text-only
                
                positive_audio = item['pos'][0] if isinstance(item['pos'], list) else item['pos']
                negative_audios = item.get('neg', [])
                
            else:
                # Old format - for backward compatibility
                query_audio = item['query']
                query_instruction = item.get('query_instruction', 'Retrieve semantically similar voice')
                positive_audio = item['positive']
                negative_audios = item.get('negatives', [])
            
            # Tokenize query instruction text
            query_text_token, query_text_token_len = self.tokenize_text(query_instruction)
            
            # Tokenize query audio (if exists)
            if query_audio and os.path.exists(query_audio):
                query_speech_token, query_speech_token_len = self.tokenize_speech(query_audio)
            else:
                # No query audio, use empty tensor
                query_speech_token = torch.tensor([], dtype=torch.long)
                query_speech_token_len = torch.tensor(0, dtype=torch.long)
            
            # Tokenize positive audio
            positive_text_token, positive_text_token_len = self.tokenize_text(query_instruction)
            positive_speech_token, positive_speech_token_len = self.tokenize_speech(positive_audio)
            
            result = {
                'query_text_token': query_text_token,
                'query_text_token_len': query_text_token_len,
                'query_speech_token': query_speech_token,
                'query_speech_token_len': query_speech_token_len,
                'positive_text_token': positive_text_token,
                'positive_text_token_len': positive_text_token_len,
                'positive_speech_token': positive_speech_token,
                'positive_speech_token_len': positive_speech_token_len,
            }
            
            # Add hard negatives if available and enabled
            if self.use_hard_negatives and negative_audios:
                # Sample up to max_negatives
                num_negatives = min(len(negative_audios), self.max_negatives)
                sampled_negatives = np.random.choice(negative_audios, num_negatives, replace=False).tolist()
                
                neg_text_tokens = []
                neg_text_token_lens = []
                neg_speech_tokens = []
                neg_speech_token_lens = []
                
                for neg_audio in sampled_negatives:
                    try:
                        neg_text_token, neg_text_token_len = self.tokenize_text(query_instruction)
                        neg_speech_token, neg_speech_token_len = self.tokenize_speech(neg_audio)
                        
                        neg_text_tokens.append(neg_text_token)
                        neg_text_token_lens.append(neg_text_token_len)
                        neg_speech_tokens.append(neg_speech_token)
                        neg_speech_token_lens.append(neg_speech_token_len)
                    except Exception as e:
                        logging.warning(f"Failed to load negative {neg_audio}: {e}")
                        continue
                
                if neg_speech_tokens:
                    result['negative_text_tokens'] = neg_text_tokens  # List of tensors
                    result['negative_text_token_lens'] = neg_text_token_lens
                    result['negative_speech_tokens'] = neg_speech_tokens
                    result['negative_speech_token_lens'] = neg_speech_token_lens
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing sample {idx} (real_idx={real_idx}): {e}")
            if query_audio:
                logging.error(f"  Query audio: {query_audio}")
            logging.error(f"  Positive audio: {positive_audio}")
            
            # Return None to be filtered out by collate_fn
            return None


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function to pad sequences in a batch.
    Filters out None samples from failed processing.
    
    Args:
        batch: List of sample dicts (may contain None)
        
    Returns:
        Collated batch dict, or None if all samples failed
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        logging.warning("All samples in batch failed to load!")
        return None
    
    batch_size = len(batch)
    
    # Simple keys to stack
    result = {
        'query_text_token_len': torch.stack([item['query_text_token_len'] for item in batch]),
        'query_speech_token_len': torch.stack([item['query_speech_token_len'].squeeze() for item in batch]),  # Squeeze scalar tensors
        'positive_text_token_len': torch.stack([item['positive_text_token_len'] for item in batch]),
        'positive_speech_token_len': torch.stack([item['positive_speech_token_len'].squeeze() for item in batch]),
    }
    
    # Pad text tokens
    for key in ['query_text_token', 'positive_text_token']:
        tokens = [item[key] for item in batch]
        max_len = max(t.size(-1) for t in tokens) if tokens else 1  # Use size(-1) for last dimension
        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, t in enumerate(tokens):
            t_flat = t.flatten()
            if t_flat.size(0) > 0:
                padded[i, :t_flat.size(0)] = t_flat
        result[key] = padded
    
    # Pad speech tokens (handle 2D tensors from frontend)
    for key in ['query_speech_token', 'positive_speech_token']:
        tokens = [item[key] for item in batch]
        # Flatten each token and get max length
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
            all_neg_text_lens.extend([l.squeeze() for l in neg_text_lens])  # Squeeze scalars
            all_neg_speech_tokens.extend(neg_speech_tokens)
            all_neg_speech_lens.extend([l.squeeze() for l in neg_speech_lens])  # Squeeze scalars
        
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
