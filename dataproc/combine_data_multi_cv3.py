import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random
import shutil
import subprocess
import sys

def setup_logger():
    """Setup basic logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)

def run_command(cmd: str, desc: str):
    """Run a command and log its output"""
    logger = logging.getLogger(__name__)
    logger.info(f"Step: {desc}")
    logger.info(f"Running command: {cmd}")
    
    try:
        process = subprocess.run(
            cmd,
            shell=True,
            text=True,
            check=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if process.stdout:
            logger.info(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        raise

def parse_dataset_mapping(mapping_str: str) -> List[Tuple[str, str]]:
    """Parse dataset mapping string
    
    Args:
        mapping_str: Format like "D:\\test1-output,bob;D:\\test2-output,alice"
        
    Returns:
        List of (dataset_path, speaker_name) tuples
    """
    mappings = []
    for item in mapping_str.split(';'):
        if ',' not in item:
            raise ValueError(f"Invalid mapping format: {item}. Expected 'path,speaker_name'")
        
        path, speaker = item.rsplit(',', 1)  # Split on last comma
        path = path.strip()
        speaker = speaker.strip()
        
        if not Path(path).exists():
            raise ValueError(f"Dataset path does not exist: {path}")
        
        mappings.append((path, speaker))
    
    return mappings

def load_kaldi_file(file_path: Path) -> Dict:
    """Load a Kaldi-style file into a dictionary"""
    data = {}
    if not file_path.exists():
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ', 1)
            if len(parts) == 2:
                key, value = parts
                # Handle spk2utt format (space-separated values)
                if ' ' in value:
                    data[key] = value.split()
                else:
                    data[key] = value
            else:
                # Single key line
                data[parts[0]] = ""
    
    return data

def save_kaldi_file(data: Dict, file_path: Path):
    """Save dictionary to Kaldi-style file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            if isinstance(value, list):
                f.write(f"{key} {' '.join(value)}\n")
            else:
                f.write(f"{key} {value}\n")

def combine_datasets(
    dataset_mappings: str,
    output_dir: str,
    cv_ratio: float = 0.05,
    random_seed: int = 42,
    cosyvoice_model_dir: str = "D:\\models\\cosyvoice_models\\CosyVoice3-0.5B-2512",
    num_utts_per_parquet: int = 1000,
    num_processes: int = 10,
    max_samples_per_speaker: int = 0,
    skip_train: bool = False
) -> None:
    """Combine multiple datasets with speaker remapping
    
    Args:
        dataset_mappings: Format like "D:\\test1-output,bob;D:\\test2-output,alice;D:\\test3-output,original"
        output_dir: Output directory for combined dataset
        cv_ratio: Ratio for cross-validation split
        random_seed: Random seed for shuffling
        cosyvoice_model_dir: Directory containing model files
        num_utts_per_parquet: Number of utterances per parquet file
        num_processes: Number of processes for parquet generation
        
    Note:
        Use "original" as speaker name to keep existing speaker mappings from that dataset
    """
    logger = setup_logger()
    
    # Parse dataset mappings
    mappings = parse_dataset_mapping(dataset_mappings)
    logger.info(f"Processing {len(mappings)} datasets:")
    for path, speaker in mappings:
        if speaker.lower() == "original":
            logger.info(f"  {path} -> keeping original speaker mappings")
        else:
            logger.info(f"  {path} -> {speaker}")
    
    # Setup output directories
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    cv_dir = output_dir / "cv"
    
    for d in [output_dir, train_dir, cv_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Combined data dictionaries
    combined_wav_scp = {}
    combined_text = {}
    combined_utt2spk = {}
    combined_spk2utt = {}
    
    # Process each dataset
    for dataset_path, new_speaker_name in mappings:
        dataset_path = Path(dataset_path)
        
        if new_speaker_name.lower() == "original":
            logger.info(f"Processing dataset: {dataset_path} -> keeping original speakers")
        else:
            logger.info(f"Processing dataset: {dataset_path} -> speaker: {new_speaker_name}")
        
        # Load Kaldi files
        wav_scp = load_kaldi_file(dataset_path / "wav.scp")
        text = load_kaldi_file(dataset_path / "text")
        utt2spk = load_kaldi_file(dataset_path / "utt2spk")
        spk2utt = load_kaldi_file(dataset_path / "spk2utt")
        
        logger.info(f"  Loaded {len(wav_scp)} utterances from {dataset_path}")
        
        # Handle speaker remapping
        if new_speaker_name.lower() == "original":
            # Keep original speaker mappings
            for old_utt_id in wav_scp.keys():
                original_speaker = utt2spk.get(old_utt_id, "unknown")
                
                # Create unique utterance ID to avoid conflicts between datasets
                # Prefix with a dataset identifier if needed
                dataset_name = dataset_path.name
                new_utt_id = f"{dataset_name}_{old_utt_id}"
                
                # Update dictionaries with original speaker names
                combined_wav_scp[new_utt_id] = wav_scp[old_utt_id]
                combined_text[new_utt_id] = text.get(old_utt_id, "")
                combined_utt2spk[new_utt_id] = original_speaker
                
                # Add to spk2utt with original speaker name
                if original_speaker not in combined_spk2utt:
                    combined_spk2utt[original_speaker] = []
                combined_spk2utt[original_speaker].append(new_utt_id)
        else:
            # Remap to new speaker name
            for old_utt_id in wav_scp.keys():
                # Create new utterance ID with new speaker name
                new_utt_id = f"{new_speaker_name}_{old_utt_id}"
                
                # Update all dictionaries with remapped data
                combined_wav_scp[new_utt_id] = wav_scp[old_utt_id]
                combined_text[new_utt_id] = text.get(old_utt_id, "")
                combined_utt2spk[new_utt_id] = new_speaker_name
                
                # Add to spk2utt
                if new_speaker_name not in combined_spk2utt:
                    combined_spk2utt[new_speaker_name] = []
                combined_spk2utt[new_speaker_name].append(new_utt_id)
    
    logger.info(f"Combined dataset contains {len(combined_wav_scp)} utterances from {len(combined_spk2utt)} speakers")
    for spk, utts in combined_spk2utt.items():
        logger.info(f"  Speaker {spk}: {len(utts)} utterances")

    # Apply max_samples_per_speaker limit if specified
    if max_samples_per_speaker > 0:
        logger.info(f"Applying max_samples_per_speaker limit: {max_samples_per_speaker}")
        random.seed(random_seed)  # Use same seed for reproducibility
        
        filtered_spk2utt = {}
        removed_utt_ids = set()
        
        for spk, utts in combined_spk2utt.items():
            if len(utts) > max_samples_per_speaker:
                # Randomly sample utterances for this speaker
                sampled_utts = random.sample(utts, max_samples_per_speaker)
                filtered_spk2utt[spk] = sampled_utts
                
                # Mark removed utterances
                removed_utts = set(utts) - set(sampled_utts)
                removed_utt_ids.update(removed_utts)
                
                logger.info(f"  Speaker {spk}: reduced from {len(utts)} to {len(sampled_utts)} utterances")
            else:
                filtered_spk2utt[spk] = utts
        
        # Update combined dictionaries by removing filtered utterances
        for utt_id in removed_utt_ids:
            combined_wav_scp.pop(utt_id, None)
            combined_text.pop(utt_id, None)
            combined_utt2spk.pop(utt_id, None)
        
        # Update spk2utt with filtered data
        combined_spk2utt = filtered_spk2utt
        
        logger.info(f"After filtering: {len(combined_wav_scp)} utterances from {len(combined_spk2utt)} speakers")
        for spk, utts in combined_spk2utt.items():
            logger.info(f"  Speaker {spk}: {len(utts)} utterances")
    
    # Shuffle and split data
    random.seed(random_seed)
    all_utt_ids = list(combined_wav_scp.keys())
    random.shuffle(all_utt_ids)
    
    cv_size = max(1, int(len(all_utt_ids) * cv_ratio))
    cv_utt_ids = set(all_utt_ids[:cv_size])
    train_utt_ids = set(all_utt_ids[cv_size:])
    
    logger.info(f"Split: {len(train_utt_ids)} train, {len(cv_utt_ids)} CV utterances")
    
    # Split data into train and CV
    def split_data(utt_ids_set):
        split_wav_scp = {uid: combined_wav_scp[uid] for uid in utt_ids_set if uid in combined_wav_scp}
        split_text = {uid: combined_text[uid] for uid in utt_ids_set if uid in combined_text}
        split_utt2spk = {uid: combined_utt2spk[uid] for uid in utt_ids_set if uid in combined_utt2spk}
        
        # Rebuild spk2utt for this split
        split_spk2utt = {}
        for uid, spk in split_utt2spk.items():
            if spk not in split_spk2utt:
                split_spk2utt[spk] = []
            split_spk2utt[spk].append(uid)
        
        return split_wav_scp, split_text, split_utt2spk, split_spk2utt
    
    # Save train split
    train_wav_scp, train_text, train_utt2spk, train_spk2utt = split_data(train_utt_ids)
    save_kaldi_file(train_wav_scp, train_dir / "wav.scp")
    save_kaldi_file(train_text, train_dir / "text")
    save_kaldi_file(train_utt2spk, train_dir / "utt2spk")
    save_kaldi_file(train_spk2utt, train_dir / "spk2utt")
    
    # Save CV split
    cv_wav_scp, cv_text, cv_utt2spk, cv_spk2utt = split_data(cv_utt_ids)
    save_kaldi_file(cv_wav_scp, cv_dir / "wav.scp")
    save_kaldi_file(cv_text, cv_dir / "text")
    save_kaldi_file(cv_utt2spk, cv_dir / "utt2spk")
    save_kaldi_file(cv_spk2utt, cv_dir / "spk2utt")
    
    logger.info("Kaldi files saved successfully")
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    tools_dir = script_dir.parent / "tools"

    # Process CV set first as it is smaller
    logger.info("Processing CV set...")
    
    # Extract embeddings for CV
    cmd = (f"python {tools_dir / 'extract_embedding.py'} "
           f"--dir {cv_dir} "
           f"--onnx_path {Path(cosyvoice_model_dir) / 'campplus.onnx'}")
    run_command(cmd, "CV voice embedding extraction")
    
    # Extract speech tokens for CV
    cmd = (f"python {tools_dir / 'extract_speech_token.py'} "
           f"--dir {cv_dir} "
           f"--onnx_path {Path(cosyvoice_model_dir) / 'speech_tokenizer_v3.onnx'}")
    run_command(cmd, "CV speech token extraction")
    
    # Generate CV parquet files
    cv_parquet_dir = cv_dir / "parquet"
    cv_parquet_dir.mkdir(exist_ok=True)
    cmd = (f"python {tools_dir / 'make_parquet_list.py'} "
           f"--num_utts_per_parquet {num_utts_per_parquet} "
           f"--num_processes {num_processes} "
           f"--src_dir {cv_dir} "
           f"--des_dir {cv_parquet_dir}")
    run_command(cmd, "CV parquet file generation")

    # Process train set
    logger.info("Processing train set...")
    
    # Extract embeddings for train
    cmd = (f"python {tools_dir / 'extract_embedding.py'} "
           f"--dir {train_dir} "
           f"--onnx_path {Path(cosyvoice_model_dir) / 'campplus.onnx'}")
    run_command(cmd, "Train voice embedding extraction")
    
    # Extract speech tokens for train
    cmd = (f"python {tools_dir / 'extract_speech_token.py'} "
           f"--dir {train_dir} "
           f"--onnx_path {Path(cosyvoice_model_dir) / 'speech_tokenizer_v3.onnx'}")
    run_command(cmd, "Train speech token extraction")
    
    # Generate train parquet files
    train_parquet_dir = train_dir / "parquet"
    train_parquet_dir.mkdir(exist_ok=True)
    cmd = (f"python {tools_dir / 'make_parquet_list.py'} "
           f"--num_utts_per_parquet {num_utts_per_parquet} "
           f"--num_processes {num_processes} "
           f"--src_dir {train_dir} "
           f"--des_dir {train_parquet_dir}")
    run_command(cmd, "Train parquet file generation")
    
    
    logger.info("Dataset combination completed successfully!")
    logger.info(f"Train data: {train_dir}")
    logger.info(f"CV data: {cv_dir}")


def main():
    """Usage:
    python dataproc\combine_data_multi.py ^
    D:\data\FreeTalk\output-cv-data,original;D:\data\yoyo_mix_en_zh\output-cv-data-train,yoyo;D:\data\SHC-Lulu-audio\output_cv_data-train,lulu ^
    D:\data\FreeTalk\combined-output-v1 ^
        --cv-ratio=0.05 ^
        --random-seed=42 ^
        --cosyvoice-model-dir=D:\models\cosyvoice_models\CosyVoice3-0.5B-2512 ^
        --num-utts-per-parquet=1000 ^
        --num-processes=10

2025-05-31 17:24:03,935 - INFO -   Speaker alicia: 674 utterances
2025-05-31 17:24:03,935 - INFO -   Speaker chao: 866 utterances
2025-05-31 17:24:03,935 - INFO -   Speaker fangning: 1021 utterances
2025-05-31 17:24:03,936 - INFO -   Speaker qing: 1997 utterances
2025-05-31 17:24:03,936 - INFO -   Speaker zhezong: 456 utterances
2025-05-31 17:24:03,936 - INFO -   Speaker yoyo: 14902 utterances
2025-05-31 17:24:03,936 - INFO -   Speaker lulu: 5889 utterances
Split: 24515 train, 1290 CV utterances

Train data: D:\data\FreeTalk\combined-output-v1\train
CV data: D:\data\FreeTalk\combined-output-v1\cv


## V2: use Chinese for yoyo; use Aoede original

python dataproc\combine_data_multi.py ^
    D:\data\FreeTalk\output-cv-data,original;D:\data\Yoyo_pure-mandarin_22k\output_cv_data-train,yoyo;D:\data\SHC-Lulu-audio\output_cv_data-train,lulu ^
    D:\data\FreeTalk\combined-output-v2 ^
        --cv-ratio=0.05 ^
        --random-seed=42 ^
        --cosyvoice-model-dir=D:\models\cosyvoice_models\CosyVoice3-0.5B-2512 ^
        --num-utts-per-parquet=1000 ^
        --num-processes=10

    """
    import argh
    argh.dispatch_command(combine_datasets)

if __name__ == "__main__":
    main()
