"""
Training data format for CosyVoice3 voice embedding model.

This is a **task-conditioned** voice embedding model where the instruction text
defines what type of similarity to retrieve.

Each training sample should be a JSON line with the following structure:

{
    "query_text": "Retrieve voice that is semantically similar",
    "query_wav": "/path/to/query_audio.wav",
    "pos_wav": "/path/to/positive_audio.wav",
    "neg_wavs": ["/path/to/neg1.wav", "/path/to/neg2.wav", ...],
    "task_type": "semantic_similarity",  # optional, for analysis
    "metadata": {
        "query_speaker": "speaker_001",    # optional
        "pos_speaker": "speaker_002",      # can be different!
        "query_transcript": "Hello world",
        "pos_transcript": "Hi there",
        "language": "en"
    }
}

Supported Task Instructions:
1. "Retrieve voice that is semantically similar" 
   - Query: "The weather is nice"
   - Positive: "It's a beautiful day"
   - Negative: "I hate pizza"

2. "Retrieve voice that is emotionally similar"
   - Query: Happy voice saying anything
   - Positive: Another happy voice
   - Negative: Sad/angry voices

3. "Retrieve voice with the same speaker"
   - Query: Speaker A saying X
   - Positive: Speaker A saying Y
   - Negative: Speaker B, C, D

4. "Retrieve voice of the same dialect"
   - Query: British English
   - Positive: Another British English
   - Negative: American/Australian English

5. "Retrieve voice with the same speaker that is semantically similar"
   - Query: Speaker A saying "I'm happy"
   - Positive: Speaker A saying "I'm joyful" 
   - Negative: Speaker A saying "I'm sad" OR Speaker B saying "I'm happy"

6. "Retrieve voice that answers the question"
   - Query: "What's the capital of France?"
   - Positive: "Paris is the capital"
   - Negative: "London is nice"

7. Custom instructions:
   - "Retrieve voice with the same gender"
   - "Retrieve voice with similar pitch"
   - "Retrieve voice in the same language"
   - etc.

Training Objectives:
- Contrastive Learning: Pull query and positive close, push negatives away
- The model learns to follow the instruction text to define similarity
- Same audio can be positive or negative depending on the task instruction!

Data Requirements:
- Audio files: 16kHz or higher, resampled to 16kHz for speech tokens
- Duration: 2-10 seconds recommended
- 3-5 negative samples per query improves performance
- Mix different task types in training for a versatile model
"""

"""
Preprocess Emilia dataset for voice embedding training.

This script creates training data for the task: "Retrieve voice with the same speaker"
- For each speaker, create samples where positive is same speaker, negatives are different speakers
- Maximum 20 samples per speaker
- 7 random negatives per sample
- 80/20 train/val split
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import argh
from tqdm import tqdm


def load_emilia_data(emilia_folder: str) -> Dict[str, List[Dict]]:
    """
    Load all Emilia samples and group by speaker.
    
    Args:
        emilia_folder: Path to Emilia folder (e.g., D:\data\Emilia_Yodas\EN-B000000)
        
    Returns:
        Dictionary mapping speaker_id to list of sample metadata
    """
    emilia_path = Path(emilia_folder)
    
    # Group samples by speaker
    speaker_samples = defaultdict(list)
    
    # Find all JSON files
    json_files = list(emilia_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {emilia_folder}")
    
    for json_file in tqdm(json_files, desc="Loading metadata"):
        # Load JSON metadata
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract speaker ID from filename
        # Format: EN_ta_bpb8s-hA_W000000.json -> speaker is "ta_bpb8s-hA"
        filename = json_file.stem  # EN_ta_bpb8s-hA_W000000
        parts = filename.split('_')
        if len(parts) >= 3:
            speaker_id = parts[1]  # ta_bpb8s-hA
        else:
            print(f"Warning: Cannot extract speaker from {filename}, skipping")
            continue
        
        # Find corresponding MP3 file
        mp3_file = json_file.with_suffix('.mp3')
        if not mp3_file.exists():
            print(f"Warning: MP3 file not found for {json_file}, skipping")
            continue
        
        # Store sample info
        sample = {
            'speaker_id': speaker_id,
            'mp3_path': str(mp3_file),
            'text': metadata.get('text', ''),
            'duration': metadata.get('duration', 0),
            'language': metadata.get('language', 'en'),
            'dnsmos': metadata.get('dnsmos', 0),
            'id': metadata.get('_id', filename)
        }
        
        speaker_samples[speaker_id].append(sample)
    
    # Filter speakers with at least 2 samples (need at least query + positive)
    speaker_samples = {
        spk: samples for spk, samples in speaker_samples.items()
        if len(samples) >= 2
    }
    
    print(f"\nFound {len(speaker_samples)} speakers with at least 2 samples")
    print(f"Total samples: {sum(len(samples) for samples in speaker_samples.values())}")
    
    return speaker_samples


def create_training_samples(
    speaker_samples: Dict[str, List[Dict]],
    max_samples_per_speaker: int = 20,
    num_negatives: int = 7
) -> List[Dict]:
    """
    Create training samples with positive and negative pairs.
    
    Args:
        speaker_samples: Dictionary mapping speaker_id to samples
        max_samples_per_speaker: Maximum samples to create per speaker
        num_negatives: Number of negative samples per query
        
    Returns:
        List of training samples
    """
    all_speakers = list(speaker_samples.keys())
    training_samples = []
    
    for speaker_id, samples in tqdm(speaker_samples.items(), desc="Creating samples"):
        # Randomly shuffle samples for this speaker
        random.shuffle(samples)
        
        # Create at most max_samples_per_speaker pairs
        num_samples = min(len(samples) - 1, max_samples_per_speaker)
        
        for i in range(num_samples):
            # Query is sample i, positive is sample i+1 (same speaker)
            query_sample = samples[i]
            pos_sample = samples[i + 1]
            
            # Select negative speakers (different from current speaker)
            other_speakers = [spk for spk in all_speakers if spk != speaker_id]
            if len(other_speakers) < num_negatives:
                print(f"Warning: Not enough speakers for negatives, using all available")
                neg_speakers = other_speakers
            else:
                neg_speakers = random.sample(other_speakers, num_negatives)
            
            # Select one random sample from each negative speaker
            neg_samples = []
            for neg_speaker in neg_speakers:
                neg_speaker_samples = speaker_samples[neg_speaker]
                neg_sample = random.choice(neg_speaker_samples)
                neg_samples.append(neg_sample)
            
            # Create training sample
            training_sample = {
                'query_text': 'Retrieve voice with the same speaker',
                'query_wav': query_sample['mp3_path'],
                'pos_wav': pos_sample['mp3_path'],
                'neg_wavs': [neg['mp3_path'] for neg in neg_samples],
                'task_type': 'speaker_similarity',
                'metadata': {
                    'query_speaker': speaker_id,
                    'pos_speaker': speaker_id,
                    'neg_speakers': [neg['speaker_id'] for neg in neg_samples],
                    'query_transcript': query_sample['text'],
                    'pos_transcript': pos_sample['text'],
                    'query_duration': query_sample['duration'],
                    'pos_duration': pos_sample['duration'],
                    'language': query_sample['language'],
                }
            }
            
            training_samples.append(training_sample)
    
    print(f"\nCreated {len(training_samples)} training samples")
    return training_samples


def copy_audio_files(samples: List[Dict], output_wavs_dir: Path):
    """
    Copy all unique audio files to output directory.
    
    Args:
        samples: List of training samples
        output_wavs_dir: Output directory for audio files
    """
    output_wavs_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all unique audio file paths
    audio_paths = set()
    for sample in samples:
        audio_paths.add(sample['query_wav'])
        audio_paths.add(sample['pos_wav'])
        audio_paths.update(sample['neg_wavs'])
    
    print(f"\nCopying {len(audio_paths)} unique audio files...")
    
    # Copy files and update paths in samples
    path_mapping = {}
    for audio_path in tqdm(audio_paths, desc="Copying audio files"):
        src_path = Path(audio_path)
        dst_path = output_wavs_dir / src_path.name
        
        # Copy file if not already exists
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        
        # Store mapping
        path_mapping[audio_path] = str(dst_path)
    
    # Update paths in samples
    for sample in samples:
        sample['query_wav'] = path_mapping[sample['query_wav']]
        sample['pos_wav'] = path_mapping[sample['pos_wav']]
        sample['neg_wavs'] = [path_mapping[neg] for neg in sample['neg_wavs']]


def save_jsonl(samples: List[Dict], output_path: Path):
    """
    Save samples to JSONL file.
    
    Args:
        samples: List of training samples
        output_path: Output JSONL file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")


def split_train_val_by_speaker(
    speaker_samples: Dict[str, List[Dict]], 
    val_ratio: float = 0.05
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Split speakers into train and validation sets.
    Ensures validation speakers are completely unseen in training.
    
    Args:
        speaker_samples: Dictionary mapping speaker_id to samples
        val_ratio: Ratio of speakers for validation
        
    Returns:
        Tuple of (train_speaker_samples, val_speaker_samples)
    """
    all_speakers = list(speaker_samples.keys())
    random.shuffle(all_speakers)
    
    # Split speakers
    num_val_speakers = max(1, int(len(all_speakers) * val_ratio))
    val_speakers = set(all_speakers[:num_val_speakers])
    train_speakers = set(all_speakers[num_val_speakers:])
    
    # Split samples by speaker
    train_speaker_samples = {
        spk: samples for spk, samples in speaker_samples.items()
        if spk in train_speakers
    }
    val_speaker_samples = {
        spk: samples for spk, samples in speaker_samples.items()
        if spk in val_speakers
    }
    
    print(f"\nTrain speakers: {len(train_speaker_samples)}")
    print(f"Val speakers: {len(val_speaker_samples)}")
    
    return train_speaker_samples, val_speaker_samples


def prepare_emilia_data(
    emilia_folder: str,
    output_dir: str = "emilia_data",
    max_samples_per_speaker: int = 20,
    num_negatives: int = 7,
    val_ratio: float = 0.05,
    seed: int = 42
):
    """
    Prepare Emilia dataset for voice embedding training.
    
    Args:
        emilia_folder: Path to Emilia folder (e.g., D:\\data\\Emilia_Yodas\\EN-B000000)
        output_dir: Output directory for processed data
        max_samples_per_speaker: Maximum training samples per speaker
        num_negatives: Number of negative samples per query
        val_ratio: Validation set ratio (by number of speakers)
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("Emilia Dataset Preparation for Voice Embedding Training")
    print("=" * 80)
    print(f"\nInput folder: {emilia_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples per speaker: {max_samples_per_speaker}")
    print(f"Negatives per sample: {num_negatives}")
    print(f"Validation speaker ratio: {val_ratio}")
    print(f"Random seed: {seed}\n")
    
    # Set random seed
    random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Emilia data
    print("\n" + "=" * 80)
    print("Step 1: Loading Emilia metadata")
    print("=" * 80)
    speaker_samples = load_emilia_data(emilia_folder)
    
    # Split speakers into train/val BEFORE creating samples
    print("\n" + "=" * 80)
    print("Step 2: Splitting speakers into train/val")
    print("=" * 80)
    train_speaker_samples, val_speaker_samples = split_train_val_by_speaker(
        speaker_samples, val_ratio=val_ratio
    )
    
    # Create training samples for train speakers only
    print("\n" + "=" * 80)
    print("Step 3: Creating training samples (train speakers)")
    print("=" * 80)
    train_samples = create_training_samples(
        train_speaker_samples,
        max_samples_per_speaker=max_samples_per_speaker,
        num_negatives=num_negatives
    )
    
    # Create validation samples for val speakers only
    print("\n" + "=" * 80)
    print("Step 4: Creating validation samples (val speakers)")
    print("=" * 80)
    val_samples = create_training_samples(
        val_speaker_samples,
        max_samples_per_speaker=max_samples_per_speaker,
        num_negatives=num_negatives
    )
    
    # Copy audio files
    print("\n" + "=" * 80)
    print("Step 5: Copying audio files")
    print("=" * 80)
    wavs_dir = output_path / "wavs"
    all_samples_combined = train_samples + val_samples
    copy_audio_files(all_samples_combined, wavs_dir)
    
    # Save JSONL files
    print("\n" + "=" * 80)
    print("Step 6: Saving JSONL files")
    print("=" * 80)
    save_jsonl(train_samples, output_path / "train.jsonl")
    save_jsonl(val_samples, output_path / "val.jsonl")
    
    print("\n" + "=" * 80)
    print("Data preparation completed!")
    print("=" * 80)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    wavs/          - {len(list(wavs_dir.glob('*.mp3')))} MP3 files")
    print(f"    train.jsonl    - {len(train_samples)} samples from {len(train_speaker_samples)} speakers")
    print(f"    val.jsonl      - {len(val_samples)} samples from {len(val_speaker_samples)} speakers")
    print(f"\n  Train speakers and val speakers are completely disjoint!")
    print()


if __name__ == '__main__':
    argh.dispatch_command(prepare_emilia_data)
