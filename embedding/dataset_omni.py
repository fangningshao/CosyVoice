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
import glob
import random
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import logging
import threading

from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class OmniEmbeddingDataset(Dataset):
    """
    Dataset for voice embedding training using Qwen2.5-Omni-3B.

    Expected data format (JSON lines):
    {
        "query_text": "Retrieve semantically similar voice",
        "query_wav": "path/to/query_audio.wav",  # Optional, can be null for text-only queries
        "pos_wav": "path/to/positive_audio.wav",
        "neg_wavs": ["path/to/neg1.wav", "path/to/neg2.wav", ...]  # Optional and variable length
    }

    Key differences from CosyVoice dataset:
    - Uses Qwen2_5OmniProcessor instead of CosyVoiceFrontEnd
    - Formats inputs as conversational prompts (system + user)
    - System prompt = query_text (instruction)
    - User prompt = audio input
    """

    # Class-level cache for processors: {(model_dir, pid): processor}
    _processor_cache = {}
    _processor_lock = {}  # Lock per PID to prevent concurrent initialization

    # System prompt for Qwen2.5-Omni (from official examples)
    SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    def __init__(self,
                 data_list_file: str,
                 model_dir: str,
                 use_hard_negatives: bool = True,
                 max_negatives: int = 7,
                 max_duration: float = 30.0,
                 min_duration: float = 1.0,
                 skip_prefilter: bool = True,
                 use_audio_in_query: bool = True,
                 random_seed: int = 42,
                 path_mappings: Optional[List[tuple]] = None):
        """
        Args:
            data_list_file: Path to data list file (JSON lines)
            model_dir: Qwen2.5-Omni model directory (for processor initialization)
            use_hard_negatives: Whether to use hard negatives
            max_negatives: Maximum number of hard negatives to use
            max_duration: Maximum audio duration in seconds (default 30s)
            min_duration: Minimum audio duration in seconds
            skip_prefilter: Skip duration/existence pre-filtering (assumes already filtered)
            use_audio_in_query: Whether query includes audio (set False for text-only queries)
            path_mappings: List of (from_path, to_path) tuples for path remapping
                          e.g., [("D:\\data\\", "/workspace/data/")]
        """
        self.data_list_file = data_list_file
        self.model_dir = model_dir
        self.use_hard_negatives = use_hard_negatives
        self.max_negatives = max_negatives
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.skip_prefilter = skip_prefilter
        self.use_audio_in_query = use_audio_in_query
        self.path_mappings = path_mappings or []

        # Log path mappings if configured
        if self.path_mappings:
            logging.info(f"Path mappings configured:")
            for from_path, to_path in self.path_mappings:
                logging.info(f"  '{from_path}' -> '{to_path}'")

        # Load data list
        self.data = []
        self.dataset_sources = []  # Track which file each sample came from
        for data_file in glob.glob(data_list_file):
            items = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        items.append(item)
                        self.dataset_sources.append(data_file)  # Store source file path
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to parse line: {line}")
                        continue

            logging.info(f"Loaded {len(items)} samples from {data_file}")
            self.data.extend(items)

        # Pre-filter invalid samples (skip if already filtered)
        self.valid_indices = list(range(len(self.data)))
        if not self.skip_prefilter:
            self._prefilter_samples()
        else:
            logging.info("Pre-filtering skipped (assuming files already filtered)")

        # Shuffle
        random.seed(random_seed)
        random.shuffle(self.valid_indices)

    def _remap_path(self, path: str) -> str:
        """
        Remap and normalize file path using configured mappings.
        
        Args:
            path: Original path (may be Windows or Linux format)
            
        Returns:
            Remapped path with forward slashes
        """
        if not path or path == "null":
            return path
            
        # Apply each mapping in order
        remapped_path = path
        for from_path, to_path in self.path_mappings:
            if remapped_path.startswith(from_path):
                remapped_path = remapped_path.replace(from_path, to_path, 1)
                break
        
        # Normalize all backslashes to forward slashes if mode specified
        if self.path_mappings:
            remapped_path = remapped_path.replace('\\', '/')
        
        return remapped_path

    def _check_audio_duration(self, audio_path: str) -> Optional[float]:
        """Check audio duration. Returns duration if valid, None otherwise."""
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

        for idx in self.valid_indices:
            item = self.data[idx]

            # Determine audio paths based on format
            if 'query_text' in item and 'pos_wav' in item:
                query_audio = item.get('query_wav')
                positive_audio = item.get('pos_wav')
                negative_audios = item.get('neg_wavs', [])
            elif 'query' in item and 'pos' in item:
                query_audio = item.get('query_wav')
                positive_audio = item['pos'][0] if isinstance(item['pos'], list) else item['pos']
                negative_audios = item.get('neg', [])
            else:
                query_audio = item.get('query')
                positive_audio = item.get('positive')
                negative_audios = item.get('negatives', [])

            # Check query audio (if exists and not null)
            if query_audio and query_audio != "null":
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
                    except:
                        pass
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
        logging.info(f"  Skipped (missing): {skipped_missing}")
        logging.info(f"  Skipped (too long): {skipped_too_long}")
        logging.info(f"  Skipped (too short): {skipped_too_short}")

    @property
    def processor(self):
        """Lazy load processor per process."""
        current_pid = os.getpid()
        cache_key = (self.model_dir, current_pid)

        # Check if processor already exists for this process
        if cache_key in self._processor_cache:
            return self._processor_cache[cache_key]

        # Create lock for this PID if doesn't exist
        if cache_key not in self._processor_lock:
            self._processor_lock[cache_key] = threading.Lock()

        # Acquire lock to prevent concurrent initialization
        with self._processor_lock[cache_key]:
            # Double-check after acquiring lock
            if cache_key in self._processor_cache:
                return self._processor_cache[cache_key]

            # Get worker info for logging
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else -1

            logging.info(f"Initializing Qwen2.5-Omni processor for worker {worker_id} (PID: {current_pid})")

            # Initialize processor
            processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                # Disable image/video processor for audio-only data
                # image_processor=None,
                # video_processor=None
            )

            # Cache the processor for this process
            self._processor_cache[cache_key] = processor

            logging.info(f"✓ Processor cached for PID {current_pid}")

        return self._processor_cache[cache_key]

    def __len__(self):
        return len(self.valid_indices)

    def __getstate__(self):
        """Custom pickle support - exclude non-picklable objects."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom unpickle support - restore state."""
        self.__dict__.update(state)

    def prepare_conversation(self, instruction: str, audio_path: Optional[str] = None) -> tuple:
        """
        Prepare conversation format for Qwen2.5-Omni.

        Args:
            instruction: Text instruction (used as system prompt)
            audio_path: Path to audio file (optional)

        Returns:
            Tuple of (conversation, audios, images, videos)
        """
        # Build conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        audios = []

        if audio_path:
            # Load audio data - processor expects actual audio waveform, not path
            try:
                waveform, sr = torchaudio.load(audio_path)

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Resample to 16kHz if needed (Qwen2.5-Omni expects 16kHz)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)

                # Convert to numpy array (processor expects numpy)
                audio_data = waveform.squeeze(0).numpy()

                # Add audio to conversation content
                conversation[1]["content"].append({"type": "audio", "audio": audio_path})
                audios.append(audio_data)

            except Exception as e:
                logging.warning(f"Failed to load audio {audio_path}: {e}")
                # If audio fails to load, just skip it
                pass

        # Always add some text content if empty
        if not conversation[1]["content"]:
            conversation[1]["content"].append({"type": "text", "text": "Analyze this."})

        return conversation, audios, None, None

    def extract_instruction_prefix(self, query_text: str) -> str:
        """
        Extract instruction prefix from query text for positive/negative alignment.

        For text2speech tasks, if query contains "content:" substring,
        return a homogeneous instruction prefix.
        """
        if "content:" in query_text.lower():
            return "Retrieve the voice with the following content"
        return query_text

    def process_sample(self, instruction: str, audio_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Process a single sample (instruction + audio) into model inputs.

        Args:
            instruction: Text instruction
            audio_path: Optional audio file path

        Returns:
            Dictionary containing:
                - input_ids: Token IDs [seq_len]
                - attention_mask: Attention mask [seq_len]
                - audio_values: Audio features (if audio provided)
        """
        # Build conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}]
            },
        ]

        audios = []

        if audio_path:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio",
                     "audio": audio_path}]
            })
        else:
            # Text-only query
            pass

        # Apply chat template
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # DEBUG - move later
        # print(f"Formatted prompt:\n{text}\n")

        # Process inputs - pass loaded audio data, not paths
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        # dict_keys(['input_ids', 'attention_mask', 'feature_attention_mask', 'input_features'])
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True)

        # Squeeze batch dimension (will be re-batched in collate_fn)
        result = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }

        # Add audio features if present
        # audios: e.g. list of (203520,) array
        if audios:
            # For single audio, take the first element and convert to tensor
            if len(audios) == 1:
                result['audio_values'] = torch.from_numpy(audios[0])  # [audio_len]
            else:
                # Multiple audios - stack them (if same length) or keep as list
                result['audio_values'] = [torch.from_numpy(a) for a in audios]

        return result


    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get a training sample.

        Returns:
            Dict with processed data, or None if sample cannot be processed
        """
        # Get valid sample index
        real_idx = self.valid_indices[idx]
        item = self.data[real_idx]

        try:
            # Handle different data formats
            if 'query_text' in item and 'pos_wav' in item:
                query_instruction = item['query_text']
                query_audio = self._remap_path(item.get('query_wav')) if item.get('query_wav') else None
                positive_audio = self._remap_path(item['pos_wav'])
                negative_audios = [self._remap_path(neg) for neg in item.get('neg_wavs', [])] if self.use_hard_negatives else []
            else:
                raise ValueError(f"Unrecognized data format in item: query_text, pos_wav must exist, got {item}")

            # Check file existence before processing
            if query_audio and self.use_audio_in_query and not os.path.exists(query_audio):
                logging.warning(f"Query audio not found: {query_audio}")
                return None

            if not os.path.exists(positive_audio):
                logging.warning(f"Positive audio not found: {positive_audio}")
                return None

            # Filter out non-existent negative audios
            if negative_audios:
                negative_audios = [neg for neg in negative_audios if os.path.exists(neg)]

            # Extract instruction prefix for positive/negative alignment
            pos_neg_instruction = self.extract_instruction_prefix(query_instruction)

            # Process query (system: instruction, user: audio)
            query_inputs = self.process_sample(query_instruction, query_audio if self.use_audio_in_query else None)

            # Process positive (system: instruction prefix, user: audio)
            positive_inputs = self.process_sample(pos_neg_instruction, positive_audio)

            result = {
                'query_input_ids': query_inputs['input_ids'],
                'query_attention_mask': query_inputs['attention_mask'],
                'positive_input_ids': positive_inputs['input_ids'],
                'positive_attention_mask': positive_inputs['attention_mask'],
                'dataset_path': self.dataset_sources[real_idx],  # Add dataset path to result
            }

            # Add query audio features if present
            if 'audio_values' in query_inputs:
                result['query_audio_values'] = query_inputs['audio_values']

            # Add positive audio features
            if 'audio_values' in positive_inputs:
                result['positive_audio_values'] = positive_inputs['audio_values']

            # Add hard negatives
            if self.use_hard_negatives and negative_audios:
                num_negatives = min(len(negative_audios), self.max_negatives)
                sampled_negatives = np.random.choice(negative_audios, num_negatives, replace=False).tolist()

                neg_input_ids = []
                neg_attention_masks = []
                neg_audio_values = []

                for neg_audio in sampled_negatives:
                    try:
                        neg_inputs = self.process_sample(pos_neg_instruction, neg_audio)
                        neg_input_ids.append(neg_inputs['input_ids'])
                        neg_attention_masks.append(neg_inputs['attention_mask'])
                        if 'audio_values' in neg_inputs:
                            neg_audio_values.append(neg_inputs['audio_values'])
                    except Exception as e:
                        logging.warning(f"Failed to load negative {neg_audio}: {e}")
                        continue

                if neg_input_ids:
                    result['negative_input_ids'] = neg_input_ids  # List of tensors
                    result['negative_attention_masks'] = neg_attention_masks
                    if neg_audio_values:
                        result['negative_audio_values'] = neg_audio_values

            return result

        except Exception as e:
            logging.error(f"Error processing sample {idx} (real_idx={real_idx}): {e}")
            if 'query_audio' in locals() and query_audio:
                logging.error(f"  Query audio: {query_audio}")
            if 'positive_audio' in locals():
                logging.error(f"  Positive audio: {positive_audio}")

            return None


def collate_fn_omni(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    Collate function for Qwen2.5-Omni batches.
    Pads sequences and handles variable-length negatives.

    Args:
        batch: List of sample dicts (may contain None)

    Returns:
        Collated batch dict, or None if all samples failed.
        Collated keys:
            dict_keys(['query_input_ids', 'query_attention_mask', 'positive_input_ids', 'positive_attention_mask', 'query_audio_values', 'positive_audio_values', 'negative_input_ids', 'negative_attention_mask', 'negative_counts', 'negative_audio_values', 'dataset_paths'])

    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        logging.warning("All samples in batch failed to load!")
        return None

    batch_size = len(batch)

    # Collect dataset paths
    dataset_paths = [item.get('dataset_path', '') for item in batch]

    # Pad query inputs
    query_input_ids = [item['query_input_ids'] for item in batch]
    query_attention_masks = [item['query_attention_mask'] for item in batch]
    max_query_len = max(ids.size(0) for ids in query_input_ids)

    padded_query_ids = torch.zeros(batch_size, max_query_len, dtype=torch.long)
    padded_query_masks = torch.zeros(batch_size, max_query_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(query_input_ids, query_attention_masks)):
        padded_query_ids[i, :ids.size(0)] = ids
        padded_query_masks[i, :mask.size(0)] = mask

    # Pad positive inputs
    positive_input_ids = [item['positive_input_ids'] for item in batch]
    positive_attention_masks = [item['positive_attention_mask'] for item in batch]
    max_positive_len = max(ids.size(0) for ids in positive_input_ids)

    padded_positive_ids = torch.zeros(batch_size, max_positive_len, dtype=torch.long)
    padded_positive_masks = torch.zeros(batch_size, max_positive_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(positive_input_ids, positive_attention_masks)):
        padded_positive_ids[i, :ids.size(0)] = ids
        padded_positive_masks[i, :mask.size(0)] = mask

    result = {
        'query_input_ids': padded_query_ids,
        'query_attention_mask': padded_query_masks,
        'positive_input_ids': padded_positive_ids,
        'positive_attention_mask': padded_positive_masks,
        'dataset_paths': dataset_paths,  # Add dataset paths to result
    }

    # Handle audio features - pad and stack into tensors
    if 'query_audio_values' in batch[0]:
        query_audios = [item['query_audio_values'] for item in batch if 'query_audio_values' in item]
        if query_audios:
            # Pad audio to same length
            max_audio_len = max(a.size(0) for a in query_audios)
            padded_query_audio = torch.zeros(len(query_audios), max_audio_len, dtype=query_audios[0].dtype)
            for i, audio in enumerate(query_audios):
                padded_query_audio[i, :audio.size(0)] = audio
            result['query_audio_values'] = padded_query_audio

    if 'positive_audio_values' in batch[0]:
        positive_audios = [item['positive_audio_values'] for item in batch]
        if positive_audios:
            # Pad audio to same length
            max_audio_len = max(a.size(0) for a in positive_audios)
            padded_positive_audio = torch.zeros(len(positive_audios), max_audio_len, dtype=positive_audios[0].dtype)
            for i, audio in enumerate(positive_audios):
                padded_positive_audio[i, :audio.size(0)] = audio
            result['positive_audio_values'] = padded_positive_audio

    # Handle negatives if present
    if 'negative_input_ids' in batch[0]:
        all_neg_ids = []
        all_neg_masks = []
        all_neg_audio = []
        neg_counts = []

        for item in batch:
            neg_ids = item.get('negative_input_ids', [])
            neg_masks = item.get('negative_attention_masks', [])
            neg_audio = item.get('negative_audio_values', [])

            neg_counts.append(len(neg_ids))
            all_neg_ids.extend(neg_ids)
            all_neg_masks.extend(neg_masks)
            all_neg_audio.extend(neg_audio)

        if all_neg_ids:
            # Pad negative inputs
            max_neg_len = max(ids.size(0) for ids in all_neg_ids)
            padded_neg_ids = torch.zeros(len(all_neg_ids), max_neg_len, dtype=torch.long)
            padded_neg_masks = torch.zeros(len(all_neg_ids), max_neg_len, dtype=torch.long)

            for i, (ids, mask) in enumerate(zip(all_neg_ids, all_neg_masks)):
                padded_neg_ids[i, :ids.size(0)] = ids
                padded_neg_masks[i, :mask.size(0)] = mask

            result['negative_input_ids'] = padded_neg_ids
            result['negative_attention_mask'] = padded_neg_masks
            result['negative_counts'] = torch.tensor(neg_counts, dtype=torch.long)

            if all_neg_audio:
                # Pad negative audio to same length
                max_neg_audio_len = max(a.size(0) for a in all_neg_audio)
                padded_neg_audio = torch.zeros(len(all_neg_audio), max_neg_audio_len, dtype=all_neg_audio[0].dtype)
                for i, audio in enumerate(all_neg_audio):
                    padded_neg_audio[i, :audio.size(0)] = audio
                result['negative_audio_values'] = padded_neg_audio

    return result