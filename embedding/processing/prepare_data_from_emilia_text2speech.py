"""
Prepare text-to-speech retrieval data from Emilia dataset.

This processor creates training data where the model learns to retrieve audio based on transcript content.

Task Types:
- 60%: Full transcript retrieval - "Retrieve the voice with the following content: [full text]"
- 20%: Partial transcript retrieval - "Retrieve the voice that contains the following content: [substring]"
- 20%: Start/End retrieval - "Retrieve the voice that starts with / ends with the following content: [substring]"

No GPT API calls or speech synthesis needed - pure text manipulation.

For each audio:
- Query: Text-based query (full/partial/start/end)
- Positive: The matching audio file
- Negatives: Random audios from different speakers

Filter: Only use speakers with >= 8 audio samples

Output structure:
    emilia_en_part0000/
        wavs/                                    - Shared original audio files (symbolic links)
        emilia_en_part0000-text2speech.train.jsonl - Text-to-speech retrieval task
        emilia_en_part0000-text2speech.val.jsonl
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional
import argh
from tqdm import tqdm
import torch

import logging

# Set logging level to INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Text2SpeechRetrievalDataPreparation:
    """Prepare text-to-speech retrieval training data."""
    
    def __init__(
        self,
        emilia_folder: str,
        output_dir: str = "emilia_data",
        task_name: str = "text2speech",
        checkpoint_file: str = "progress-text2speech.json"
    ):
        self.emilia_folder = Path(emilia_folder)
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.checkpoint_file = self.output_dir / checkpoint_file
        
        # Extract dataset name from emilia_folder (e.g., "EN-B000000")
        self.dataset_name = self.emilia_folder.name
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(exist_ok=True)
        
        # Load checkpoint if exists
        self.processed_samples = set()
        self.load_checkpoint()
        
        # Output file handles for streaming
        self.train_file = None
        self.val_file = None
    
    def load_checkpoint(self):
        """Load progress checkpoint."""
        if self.checkpoint_file.exists():
            print(f"Loading checkpoint from {self.checkpoint_file}")
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.processed_samples = set(checkpoint.get('processed_samples', []))
            print(f"Resumed: {len(self.processed_samples)} samples already processed")
    
    def save_checkpoint(self):
        """Save progress checkpoint."""
        checkpoint = {
            'processed_samples': list(self.processed_samples),
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_emilia_data(self, min_samples_per_speaker: int = 8, max_duration: float = 30.0) -> Dict[str, List[Dict]]:
        """
        Load Emilia data and filter speakers with enough samples.
        
        Args:
            min_samples_per_speaker: Minimum number of samples per speaker
            max_duration: Maximum audio duration in seconds (default: 30s)
            
        Returns:
            Dictionary mapping speaker_id to list of samples
        """
        speaker_samples = defaultdict(list)
        skipped_too_long = 0
        
        json_files = list(self.emilia_folder.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in tqdm(json_files, desc="Loading metadata"):
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract speaker ID from filename
            filename = json_file.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                speaker_id = parts[1]
            else:
                continue
            
            mp3_file = json_file.with_suffix('.mp3')
            if not mp3_file.exists():
                continue
            
            duration = metadata.get('duration', 0)
            
            # Filter out audio longer than max_duration
            if duration > max_duration:
                skipped_too_long += 1
                continue
            
            sample = {
                'speaker_id': speaker_id,
                'mp3_path': str(mp3_file),
                'text': metadata.get('text', '').strip(),
                'duration': duration,
                'language': metadata.get('language', 'en'),
                'id': metadata.get('_id', filename)
            }
            
            # Filter out empty or very short text
            if len(sample['text']) < 20:  # Need at least 20 chars for substring extraction
                continue
            
            speaker_samples[speaker_id].append(sample)
        
        # Filter speakers with enough samples
        filtered_speakers = {
            spk: samples for spk, samples in speaker_samples.items()
            if len(samples) >= min_samples_per_speaker
        }
        
        print(f"\nFiltered to {len(filtered_speakers)} speakers with >= {min_samples_per_speaker} samples")
        print(f"Total samples: {sum(len(samples) for samples in filtered_speakers.values())}")
        print(f"Skipped {skipped_too_long} samples longer than {max_duration}s")
        
        return filtered_speakers
    
    def copy_audio_if_needed(self, src_path: str, dst_dir: Path) -> str:
        """
        Copy audio file to destination directory if not already exists.
        
        Args:
            src_path: Source audio file path
            dst_dir: Destination directory
            
        Returns:
            Destination file path
        """
        src = Path(src_path)
        dst = dst_dir / src.name
        
        # Only copy if doesn't exist
        if not dst.exists():
            shutil.copy2(src, dst)
        
        return str(dst)
    
    def _is_word_based_language(self, text: str) -> bool:
        """
        Determine if text uses word-based spacing (English, Latin languages).
        
        Returns True if text has many spaces and is long enough to split by words.
        """
        # Count spaces
        num_spaces = text.count(' ')
        text_len = len(text)
        
        # If text has at least 5 spaces and reasonable length, consider it word-based
        # This covers English and most Latin languages
        if num_spaces >= 5 and text_len >= 30:
            return True
        return False
    
    def extract_substring(self, text: str, min_ratio: float = 0.3, max_ratio: float = 0.7) -> str:
        """
        Extract a random substring from text.
        For word-based languages (English, Latin), split by words.
        For character-based languages (Chinese, Japanese), split by characters.
        
        Args:
            text: Full text
            min_ratio: Minimum substring length as ratio of full text
            max_ratio: Maximum substring length as ratio of full text
            
        Returns:
            Extracted substring
        """
        if self._is_word_based_language(text):
            # Word-based extraction
            words = text.split()
            num_words = len(words)
            
            min_words = max(3, int(num_words * min_ratio))
            max_words = int(num_words * max_ratio)
            
            if min_words >= max_words:
                return text
            
            # Random number of words
            num_selected = random.randint(min_words, max_words)
            
            # Random start position
            max_start = num_words - num_selected
            if max_start <= 0:
                return text
            
            start = random.randint(0, max_start)
            return ' '.join(words[start:start + num_selected])
        else:
            # Character-based extraction (original logic)
            text_len = len(text)
            min_len = max(10, int(text_len * min_ratio))
            max_len = int(text_len * max_ratio)
            
            if min_len >= max_len:
                max_len = text_len
            
            substr_len = random.randint(min_len, max_len)
            max_start = text_len - substr_len
            if max_start <= 0:
                return text
            
            start = random.randint(0, max_start)
            return text[start:start + substr_len].strip()
    
    def extract_start_text(self, text: str, min_ratio: float = 0.3, max_ratio: float = 0.7) -> str:
        """
        Extract text from the start.
        For word-based languages, split by words.
        For character-based languages, split by characters.
        """
        if self._is_word_based_language(text):
            # Word-based extraction
            words = text.split()
            num_words = len(words)
            
            min_words = max(3, int(num_words * min_ratio))
            max_words = int(num_words * max_ratio)
            
            if min_words >= max_words:
                return text
            
            end_word = random.randint(min_words, max_words)
            return ' '.join(words[:end_word])
        else:
            # Character-based extraction
            text_len = len(text)
            min_len = max(10, int(text_len * min_ratio))
            max_len = int(text_len * max_ratio)
            
            if min_len >= max_len:
                return text
            
            end = random.randint(min_len, max_len)
            return text[:end].strip()
    
    def extract_end_text(self, text: str, min_ratio: float = 0.3, max_ratio: float = 0.7) -> str:
        """
        Extract text from the end.
        For word-based languages, split by words.
        For character-based languages, split by characters.
        """
        if self._is_word_based_language(text):
            # Word-based extraction
            words = text.split()
            num_words = len(words)
            
            min_words = max(3, int(num_words * min_ratio))
            max_words = int(num_words * max_ratio)
            
            if min_words >= max_words:
                return text
            
            start_word = num_words - random.randint(min_words, max_words)
            return ' '.join(words[start_word:])
        else:
            # Character-based extraction
            text_len = len(text)
            min_len = max(10, int(text_len * min_ratio))
            max_len = int(text_len * max_ratio)
            
            if min_len >= max_len:
                return text
            
            start = text_len - random.randint(min_len, max_len)
            return text[start:].strip()
    
    def generate_query_text(self, sample: Dict) -> tuple:
        """
        Generate query text based on task type distribution.
        
        Returns:
            (query_instruction, query_type)
        """
        rand = random.random()
        text = sample['text']
        
        if rand < 0.6:
            # Full transcript retrieval (60%)
            query_instruction = f"Retrieve the voice with the following content: {text}"
            query_type = "full_transcript"
        elif rand < 0.8:
            # Partial transcript retrieval (20%)
            substring = self.extract_substring(text)
            query_instruction = f"Retrieve the voice that contains the following content: {substring}"
            query_type = "partial_transcript"
        else:
            # Start or end retrieval (20%)
            if random.random() < 0.5:
                # Start
                start_text = self.extract_start_text(text)
                query_instruction = f"Retrieve the voice that starts with the following content: {start_text}"
                query_type = "start_with"
            else:
                # End
                end_text = self.extract_end_text(text)
                query_instruction = f"Retrieve the voice that ends with the following content: {end_text}"
                query_type = "end_with"
        
        return query_instruction, query_type
    
    def write_sample_to_file(self, sample: Dict, split: str):
        """
        Write sample to corresponding output file immediately.
        
        Args:
            sample: Training sample dictionary
            split: 'train' or 'val'
        """
        if split == 'train':
            if self.train_file is None:
                train_path = self.output_dir / f"{self.dataset_name}-{self.task_name}.train.jsonl"
                self.train_file = open(train_path, 'a', encoding='utf-8')
            self.train_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.train_file.flush()
        else:
            if self.val_file is None:
                val_path = self.output_dir / f"{self.dataset_name}-{self.task_name}.val.jsonl"
                self.val_file = open(val_path, 'a', encoding='utf-8')
            self.val_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.val_file.flush()
    
    def create_training_sample(
        self,
        positive_sample: Dict,
        all_samples: List[Dict],
        num_negatives: int = 7,
        split: str = 'train'
    ) -> Optional[Dict]:
        """
        Create a training sample for text-to-speech retrieval.
        
        Args:
            positive_sample: The positive sample
            all_samples: All available samples for negative selection
            num_negatives: Number of negative samples
            split: 'train' or 'val'
            
        Returns:
            Training sample dict or None if failed
        """
        sample_id = positive_sample['id']
        
        # Skip if already processed
        if sample_id in self.processed_samples:
            return None
        
        # Generate query text
        query_instruction, query_type = self.generate_query_text(positive_sample)
        
        # Copy positive audio
        pos_wav = self.copy_audio_if_needed(positive_sample['mp3_path'], self.wavs_dir)
        
        # Select negatives from different speakers
        available_negatives = [
            s for s in all_samples 
            if s['speaker_id'] != positive_sample['speaker_id'] and s['id'] != sample_id
        ]
        
        if len(available_negatives) < num_negatives:
            num_negatives = len(available_negatives)
        
        if num_negatives == 0:
            print(f"  Skipping: not enough negatives")
            return None
        
        negative_samples = random.sample(available_negatives, num_negatives)
        
        # Copy negative audios
        neg_paths = []
        for neg_sample in negative_samples:
            neg_dst = self.copy_audio_if_needed(neg_sample['mp3_path'], self.wavs_dir)
            neg_paths.append(neg_dst)
        
        # Create training sample
        training_sample = {
            'query_text': query_instruction,
            'query_wav': None,  # No query audio for text-to-speech retrieval
            'pos_wav': pos_wav,
            'neg_wavs': neg_paths,
            'task_type': 'text2speech_retrieval',
            'metadata': {
                'pos_speaker': positive_sample['speaker_id'],
                'neg_speakers': [s['speaker_id'] for s in negative_samples],
                'pos_transcript': positive_sample['text'],
                'query_type': query_type,
                'language': positive_sample['language'],
            }
        }
        
        # Write to file immediately
        self.write_sample_to_file(training_sample, split)
        
        # Mark as processed
        self.processed_samples.add(sample_id)
        
        return training_sample
    
    def prepare_data(
        self,
        max_samples_per_speaker: int = 10,
        num_negatives: int = 7,
        val_ratio: float = 0.05
    ):
        """
        Main preparation pipeline.
        
        Args:
            max_samples_per_speaker: Maximum samples per speaker
            num_negatives: Number of negative samples per query
            val_ratio: Validation speaker ratio
        """
        print("=" * 80)
        print("Text-to-Speech Retrieval Data Preparation")
        print("=" * 80)
        
        # Load data
        print("\nStep 1: Loading Emilia data")
        print("=" * 80)
        speaker_samples = self.load_emilia_data(min_samples_per_speaker=8)
        
        # Split speakers into train/val
        print("\nStep 2: Splitting speakers")
        print("=" * 80)
        all_speakers = list(speaker_samples.keys())
        random.shuffle(all_speakers)
        
        num_val_speakers = max(1, int(len(all_speakers) * val_ratio))
        val_speakers = set(all_speakers[:num_val_speakers])
        train_speakers = set(all_speakers[num_val_speakers:])
        
        print(f"Train speakers: {len(train_speakers)}")
        print(f"Val speakers: {len(val_speakers)}")
        
        # Flatten samples for negative selection
        train_samples_flat = []
        for spk in train_speakers:
            train_samples_flat.extend(speaker_samples[spk])
        
        val_samples_flat = []
        for spk in val_speakers:
            val_samples_flat.extend(speaker_samples[spk])
        
        # Process train speakers
        print("\nStep 3: Creating training samples")
        print("=" * 80)
        train_count = 0
        
        for speaker_id in tqdm(train_speakers, desc="Train speakers"):
            samples = speaker_samples[speaker_id]
            random.shuffle(samples)
            
            # Process up to max_samples_per_speaker
            num_to_process = min(len(samples), max_samples_per_speaker)
            
            for sample in samples[:num_to_process]:
                training_sample = self.create_training_sample(
                    sample,
                    train_samples_flat,
                    num_negatives=num_negatives,
                    split='train'
                )
                
                if training_sample:
                    train_count += 1
                    
                    # Save checkpoint every 100 samples
                    if train_count % 100 == 0:
                        self.save_checkpoint()
                        print(f"\n  Checkpoint saved: {train_count} samples")
        
        # Process val speakers
        print("\nStep 4: Creating validation samples")
        print("=" * 80)
        val_count = 0
        
        for speaker_id in tqdm(val_speakers, desc="Val speakers"):
            samples = speaker_samples[speaker_id]
            random.shuffle(samples)
            
            num_to_process = min(len(samples), max_samples_per_speaker)
            
            for sample in samples[:num_to_process]:
                training_sample = self.create_training_sample(
                    sample,
                    val_samples_flat,
                    num_negatives=num_negatives,
                    split='val'
                )
                
                if training_sample:
                    val_count += 1
                    
                    if val_count % 100 == 0:
                        self.save_checkpoint()
        
        # Close file handles
        if self.train_file:
            self.train_file.close()
        if self.val_file:
            self.val_file.close()
        
        # Final checkpoint
        self.save_checkpoint()
        
        print("\n" + "=" * 80)
        print("Data preparation completed!")
        print("=" * 80)
        print(f"\nOutput structure:")
        print(f"  {self.output_dir}/")
        print(f"    wavs/                                    - {len(list(self.wavs_dir.glob('*.mp3')))} original audio files (shared)")
        print(f"    {self.dataset_name}-{self.task_name}.train.jsonl    - {train_count} train samples")
        print(f"    {self.dataset_name}-{self.task_name}.val.jsonl      - {val_count} val samples")
        print(f"\nTask type distribution:")
        print(f"  Full transcript (20%): ~{int(train_count * 0.2)} samples")
        print(f"  Partial transcript (40%): ~{int(train_count * 0.4)} samples")
        print(f"  Start/End with (40%): ~{int(train_count * 0.4)} samples")
        print()


def prepare_text2speech_retrieval_data(
    emilia_folder: str,
    output_dir: str = None,
    max_samples_per_speaker: int = 50,
    num_negatives: int = 7,
    val_ratio: float = 0.05,
    seed: int = 42
):
    """
    Prepare text-to-speech retrieval data from Emilia dataset.
    
    Args:
        emilia_folder: Path to Emilia folder (e.g., D:\\data\\Emilia_Yodas\\EN-B000000)
        output_dir: Output directory (default: parent of emilia_folder with dataset name)
        max_samples_per_speaker: Maximum samples per speaker
        num_negatives: Number of negative samples
        val_ratio: Validation speaker ratio
        seed: Random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Auto-generate output_dir if not provided
    if output_dir is None:
        emilia_path = Path(emilia_folder)
        dataset_name = emilia_path.name  # e.g., "EN-B000000"
        output_dir = emilia_path.parent / f"{dataset_name}_processed"
    
    preparer = Text2SpeechRetrievalDataPreparation(
        emilia_folder=emilia_folder,
        output_dir=output_dir,
        task_name="text2speech"
    )
    
    preparer.prepare_data(
        max_samples_per_speaker=max_samples_per_speaker,
        num_negatives=num_negatives,
        val_ratio=val_ratio
    )


if __name__ == '__main__':
    argh.dispatch_command(prepare_text2speech_retrieval_data)
