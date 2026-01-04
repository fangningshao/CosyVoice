"""
Prepare semantic similarity data from Emilia dataset.

For each voice:
1. Use GPT-4.1-nano to paraphrase the text to semantically similar content
2. Use CosyVoice3 to clone the paraphrased text with a DIFFERENT speaker's voice
3. Use other voices from the SAME speaker as negatives

Task: "Retrieve voice that is semantically similar"
- Query: Original audio (50% chance to flip with positive)
- Positive: Synthetic audio with paraphrased text + different speaker voice
- Negatives: Other audios from the same speaker as positive

Filter: Only use speakers with >= 8 audio samples

Output structure:
    emilia_en_part0000/
        wavs/                                    - Shared original audio files
        synthetic/                               - Shared synthetic audio files
        paraphrases.jsonl                        - Cache of paraphrased texts
        emilia_en_part0000-speakerid.train.jsonl - Speaker similarity task
        emilia_en_part0000-semantic.train.jsonl  - Semantic similarity task
        emilia_en_part0000-speakerid.val.jsonl
        emilia_en_part0000-semantic.val.jsonl
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import argh
from tqdm import tqdm
import torch
import torchaudio
from openai import OpenAI

from cosyvoice.cli.cosyvoice import AutoModel

import logging

# Set logging level to INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Suppress DEBUG logs from OpenAI and httpx/httpcore
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


class SemanticSimilarityDataPreparation:
    """Prepare semantic similarity training data."""
    
    def __init__(
        self,
        emilia_folder: str,
        cosyvoice_model_dir: str,
        openai_api_key: str,
        output_dir: str = "emilia_data",
        task_name: str = "semantic",
        checkpoint_file: str = "progress-semantic.json"
    ):
        self.emilia_folder = Path(emilia_folder)
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.checkpoint_file = self.output_dir / checkpoint_file
        self.paraphrase_cache_file = self.output_dir / "paraphrases.jsonl"
        
        # Extract dataset name from emilia_folder (e.g., "emilia_en_part0000")
        self.dataset_name = self.emilia_folder.name
        
        # Create output directories - SHARED across tasks
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(exist_ok=True)
        self.synthetic_dir = self.output_dir / "synthetic"
        self.synthetic_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize CosyVoice3
        print(f"Loading CosyVoice3 from {cosyvoice_model_dir}...")
        self.cosyvoice = AutoModel(model_dir=cosyvoice_model_dir, llm_ckpt=os.path.join(cosyvoice_model_dir, 'llm.rl.pt'))
        print("CosyVoice3 loaded successfully")
        
        # Load paraphrase cache
        self.paraphrase_cache = {}
        self.load_paraphrase_cache()
        
        # Load checkpoint if exists
        self.processed_samples = set()
        self.all_samples = []
        self.load_checkpoint()
        
        # Output file handles for streaming
        self.train_file = None
        self.val_file = None
    
    def load_paraphrase_cache(self):
        """Load paraphrase cache to avoid repeated API calls."""
        if self.paraphrase_cache_file.exists():
            print(f"Loading paraphrase cache from {self.paraphrase_cache_file}")
            with open(self.paraphrase_cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    self.paraphrase_cache[entry['original']] = entry['paraphrased']
            print(f"Loaded {len(self.paraphrase_cache)} cached paraphrases")
    
    def save_paraphrase_to_cache(self, original: str, paraphrased: str):
        """Save paraphrase to cache file (append mode)."""
        with open(self.paraphrase_cache_file, 'a', encoding='utf-8') as f:
            entry = {'original': original, 'paraphrased': paraphrased}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.paraphrase_cache[original] = paraphrased
    
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
    
    def load_emilia_data(self, min_samples_per_speaker: int = 8) -> Dict[str, List[Dict]]:
        """
        Load Emilia data and filter speakers with enough samples.
        
        Args:
            min_samples_per_speaker: Minimum number of samples per speaker
            
        Returns:
            Dictionary mapping speaker_id to list of samples
        """
        speaker_samples = defaultdict(list)
        
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
            
            sample = {
                'speaker_id': speaker_id,
                'mp3_path': str(mp3_file),
                'text': metadata.get('text', '').strip(),
                'duration': metadata.get('duration', 0),
                'language': metadata.get('language', 'en'),
                'id': metadata.get('_id', filename)
            }
            
            # Filter out empty or very short text
            if len(sample['text']) < 10:
                continue
            
            speaker_samples[speaker_id].append(sample)
        
        # Filter speakers with enough samples
        filtered_speakers = {
            spk: samples for spk, samples in speaker_samples.items()
            if len(samples) >= min_samples_per_speaker
        }
        
        print(f"\nFiltered to {len(filtered_speakers)} speakers with >= {min_samples_per_speaker} samples")
        print(f"Total samples: {sum(len(samples) for samples in filtered_speakers.values())}")
        
        return filtered_speakers
    
    def paraphrase_text(self, original_text: str) -> Optional[str]:
        # Check cache first
        if original_text in self.paraphrase_cache:
            print(f"  Using cached paraphrase")
            return self.paraphrase_cache[original_text]
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that paraphrases text while preserving its semantic meaning. Keep the paraphrase natural and similar in length to the original. The paraphrased text should be of the same language. Only return the paraphrased text, nothing else."
                    },
                    {
                        "role": "user",
                        "content": f"Paraphrase the following text:\n\n{original_text}"
                    }
                ],
                temperature=1,
                max_completion_tokens=200
            )
            
            print("DEBUG: OpenAI response:", response)
            paraphrased = response.choices[0].message.content.strip()
            paraphrased = paraphrased.strip('"').strip("'")
            
            # # dont Save to cache here yet
            # self.save_paraphrase_to_cache(original_text, paraphrased)

            return paraphrased
            
        except Exception as e:
            print(f"Error paraphrasing text: {e}")
            return None
        
    def synthesize_voice(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: str,
        output_path: str
    ) -> bool:
        """
        Synthesize voice using CosyVoice3 zero-shot cloning.
        
        Args:
            text: Text to synthesize
            reference_audio_path: Reference audio for voice cloning
            reference_text: Text corresponding to reference audio
            output_path: Output audio path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use zero-shot inference
            prompt = "You are a helpful assistant."
            
            # Generate audio
            # Format: inference_zero_shot(tts_text, prompt_text, prompt_audio_path)
            # prompt_text should be: "{instruction}<|endofprompt|>{reference_text}"
            for i, result in enumerate(self.cosyvoice.inference_zero_shot(
                text,
                f"{prompt}<|endofprompt|>{reference_text}",
                reference_audio_path,
                stream=False
            )):
                # Save the synthesized audio
                torchaudio.save(
                    output_path,
                    result['tts_speech'],
                    self.cosyvoice.sample_rate
                )
                return True
            
            return False
            
        except Exception as e:
            print(f"Error synthesizing voice: {e}")
            return False
    
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
            self.train_file.flush()  # Ensure data is written
        else:
            if self.val_file is None:
                val_path = self.output_dir / f"{self.dataset_name}-{self.task_name}.val.jsonl"
                self.val_file = open(val_path, 'a', encoding='utf-8')
            self.val_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
            self.val_file.flush()
    
    def create_training_sample(
        self,
        query_sample: Dict,
        pos_speaker_samples: List[Dict],
        num_negatives: int = 7,
        split: str = 'train'
    ) -> Optional[Dict]:
        sample_id = query_sample['id']
        
        # Skip if already processed
        if sample_id in self.processed_samples:
            return None
        
        # Select a random reference speaker (for positive)
        pos_reference = random.choice(pos_speaker_samples)
        
        # Check negatives BEFORE paraphrasing (don't waste API call)
        available_negatives = [
            s for s in pos_speaker_samples 
            if s['id'] != pos_reference['id']
        ]
        
        if len(available_negatives) < num_negatives:
            num_negatives = len(available_negatives)
        
        if num_negatives == 0:
            print(f"  Skipping: not enough negatives")
            return None
        
        # NOW paraphrase (after we know we won't skip)
        print(f"  Paraphrasing: {query_sample['text'][:300]}...")
        paraphrased_text = self.paraphrase_text(query_sample['text'])
        if not paraphrased_text:
            print("WARN: Paraphrasing failed or returned empty text" + query_sample['text'])
            return None
        
        print(f"  Paraphrased: {paraphrased_text[:300]}...")
                
        # Select a random reference speaker (for positive)
        pos_reference = random.choice(pos_speaker_samples)
        
        # Synthesize positive audio with paraphrased text + different speaker voice
        # Store in synthetic/ subdirectory with unique name
        synthetic_filename = f"syn_{self.task_name}_{sample_id}.wav"
        synthetic_path = self.synthetic_dir / synthetic_filename
        
        # After synthesis succeeds
        if not synthetic_path.exists():
            print(f"  Synthesizing with speaker {pos_reference['speaker_id']}...")
            success = self.synthesize_voice(
                paraphrased_text,
                pos_reference['mp3_path'],
                pos_reference['text'],  # Pass reference text
                str(synthetic_path)
            )
            
            if not success:
                return None
            
            # NOW save paraphrase to cache (after synthesis success)
            self.save_paraphrase_to_cache(query_sample['text'], paraphrased_text)
        else:
            print(f"  Synthetic audio already exists: {synthetic_path.name}")
            # Also save to cache if using existing synthetic
            self.save_paraphrase_to_cache(query_sample['text'], paraphrased_text)
        
        # Copy query audio to wavs directory (skip if exists)
        query_dst = self.copy_audio_if_needed(query_sample['mp3_path'], self.wavs_dir)
        
        # Select negatives from the same speaker as positive (excluding the reference used)
        available_negatives = [
            s for s in pos_speaker_samples 
            if s['id'] != pos_reference['id']
        ]
        
        if len(available_negatives) < num_negatives:
            num_negatives = len(available_negatives)
        
        if num_negatives == 0:
            return None
        
        negative_samples = random.sample(available_negatives, num_negatives)
        
        # Copy negative audios (skip if exist)
        neg_paths = []
        for neg_sample in negative_samples:
            neg_dst = self.copy_audio_if_needed(neg_sample['mp3_path'], self.wavs_dir)
            neg_paths.append(neg_dst)
        
        # 50% chance to flip query and positive
        flip = random.random() < 0.5
        
        if flip:
            query_wav = str(synthetic_path)
            pos_wav = query_dst
            query_text_label = paraphrased_text
            pos_text_label = query_sample['text']
        else:
            query_wav = query_dst
            pos_wav = str(synthetic_path)
            query_text_label = query_sample['text']
            pos_text_label = paraphrased_text
        
        # Create training sample
        training_sample = {
            'query_text': 'Retrieve voice that is semantically similar',
            'query_wav': query_wav,
            'pos_wav': pos_wav,
            'neg_wavs': neg_paths,
            'task_type': 'semantic_similarity',
            'metadata': {
                'query_speaker': pos_reference['speaker_id'] if flip else query_sample['speaker_id'],
                'pos_speaker': query_sample['speaker_id'] if flip else pos_reference['speaker_id'],
                'neg_speakers': [pos_reference['speaker_id']] * len(neg_paths),
                'query_transcript': query_text_label,
                'pos_transcript': pos_text_label,
                'original_text': query_sample['text'],
                'paraphrased_text': paraphrased_text,
                'flipped': flip,
                'language': query_sample['language'],
            }
        }
        
        # Write to file immediately
        self.write_sample_to_file(training_sample, split)
        
        # Mark as processed
        self.processed_samples.add(sample_id)
        
        return training_sample
    
    def prepare_data(
        self,
        max_samples_per_speaker: int = 5,
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
        print("Semantic Similarity Data Preparation")
        print("=" * 80)
        
        # Load data
        print("\nStep 1: Loading Emilia data")
        print("=" * 80)
        speaker_samples = self.load_emilia_data(min_samples_per_speaker=8)
        
        # Split speakers
        print("\nStep 2: Splitting speakers")
        print("=" * 80)
        all_speakers = list(speaker_samples.keys())
        random.shuffle(all_speakers)
        
        num_val_speakers = max(1, int(len(all_speakers) * val_ratio))
        val_speakers = set(all_speakers[:num_val_speakers])
        train_speakers = set(all_speakers[num_val_speakers:])
        
        print(f"Train speakers: {len(train_speakers)}")
        print(f"Val speakers: {len(val_speakers)}")
        
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
                # Select a different speaker for positive
                other_speakers = [s for s in train_speakers if s != speaker_id]
                if not other_speakers:
                    continue
                
                pos_speaker = random.choice(list(other_speakers))
                pos_speaker_samples = speaker_samples[pos_speaker]
                
                print(f"\nProcessing {sample['id']} (speaker: {speaker_id})")
                training_sample = self.create_training_sample(
                    sample,
                    pos_speaker_samples,
                    num_negatives=num_negatives,
                    split='train'
                )
                
                if training_sample:
                    train_count += 1
                    
                    # Save checkpoint every 10 samples
                    if train_count % 10 == 0:
                        self.save_checkpoint()
                        print(f"  Checkpoint saved: {train_count} samples")
        
        # Process val speakers
        print("\nStep 4: Creating validation samples")
        print("=" * 80)
        val_count = 0
        
        for speaker_id in tqdm(val_speakers, desc="Val speakers"):
            samples = speaker_samples[speaker_id]
            random.shuffle(samples)
            
            num_to_process = min(len(samples), max_samples_per_speaker)
            
            for sample in samples[:num_to_process]:
                other_speakers = [s for s in val_speakers if s != speaker_id]
                if not other_speakers:
                    continue
                
                pos_speaker = random.choice(list(other_speakers))
                if random.random() < 0.1:
                    pos_speaker = speaker_id  # 10% chance to use same speaker for positive
                pos_speaker_samples = speaker_samples[pos_speaker]
                
                print(f"\nProcessing {sample['id']} (speaker: {speaker_id})")
                training_sample = self.create_training_sample(
                    sample,
                    pos_speaker_samples,
                    num_negatives=num_negatives,
                    split='val'
                )
                
                if training_sample:
                    val_count += 1
                    
                    if val_count % 10 == 0:
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
        print(f"    synthetic/                               - {len(list(self.synthetic_dir.glob('*.wav')))} synthetic audio files (shared)")
        print(f"    paraphrases.jsonl                        - {len(self.paraphrase_cache)} paraphrase cache entries")
        print(f"    {self.dataset_name}-{self.task_name}.train.jsonl    - {train_count} train samples")
        print(f"    {self.dataset_name}-{self.task_name}.val.jsonl      - {val_count} val samples")
        print()


def prepare_semantic_similarity_data(
    emilia_folder: str,
    cosyvoice_model_dir: str,
    openai_api_key: str,
    output_dir: str = None,
    max_samples_per_speaker: int = 5,
    num_negatives: int = 7,
    val_ratio: float = 0.05,
    seed: int = 42
):
    """
    Prepare semantic similarity data from Emilia dataset.
    
    Args:
        emilia_folder: Path to Emilia folder (e.g., D:\\data\\Emilia_Yodas\\EN-B000000)
        cosyvoice_model_dir: Path to CosyVoice3 model directory
        openai_api_key: OpenAI API key for GPT-4.1-nano
        output_dir: Output directory (default: parent of emilia_folder with dataset name)
        max_samples_per_speaker: Maximum samples per speaker
        num_negatives: Number of negative samples
        val_ratio: Validation speaker ratio
        seed: Random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Auto-generate output_dir if not provided
    if output_dir is None:
        emilia_path = Path(emilia_folder)
        dataset_name = emilia_path.name  # e.g., "EN-B000000"
        output_dir = emilia_path.parent / f"{dataset_name}_processed"
    
    preparer = SemanticSimilarityDataPreparation(
        emilia_folder=emilia_folder,
        cosyvoice_model_dir=cosyvoice_model_dir,
        openai_api_key=openai_api_key,
        output_dir=output_dir,
        task_name="semantic"
    )
    
    preparer.prepare_data(
        max_samples_per_speaker=max_samples_per_speaker,
        num_negatives=num_negatives,
        val_ratio=val_ratio
    )


if __name__ == '__main__':
    argh.dispatch_command(prepare_semantic_similarity_data)