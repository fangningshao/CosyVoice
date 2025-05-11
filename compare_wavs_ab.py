import os
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

def setup_logger():
    """Setup basic logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)

def compare_and_zip_wavs(wavs_dir_a: str, wavs_dir_b: str, output_dir: str):
    """Compare and zip wav files from two directories
    
    Args:
        wavs_dir_a: First wav directory
        wavs_dir_b: Second wav directory
        output_dir: Output directory for combined files
    """
    logger = setup_logger()
    
    # Convert to Path objects
    dir_a = Path(wavs_dir_a)
    dir_b = Path(wavs_dir_b)
    out_dir = Path(output_dir)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get wav files from both directories
    wavs_a = set(f.stem for f in dir_a.glob("*.wav"))
    wavs_b = set(f.stem for f in dir_b.glob("*.wav"))
    
    # Find common files
    common_files = sorted(list(wavs_a.intersection(wavs_b)))
    
    if not common_files:
        logger.error("No matching wav files found between directories")
        return
    
    logger.info(f"Found {len(common_files)} matching wav files")
    
    # Copy and rename files
    for wav_name in tqdm(common_files, desc="Processing files"):
        # Source files
        wav_a = dir_a / f"{wav_name}.wav"
        wav_b = dir_b / f"{wav_name}.wav"
        
        # Destination files
        new_wav_a = out_dir / f"{wav_name}_a.wav"
        new_wav_b = out_dir / f"{wav_name}_b.wav"
        
        # Copy files with new names
        shutil.copy2(wav_a, new_wav_a)
        shutil.copy2(wav_b, new_wav_b)
    
    logger.info(f"Successfully processed {len(common_files)} file pairs")
    logger.info(f"Output directory: {out_dir}")

def main():
    """Command line interface"""
    import argh
    argh.dispatch_command(compare_and_zip_wavs)

if __name__ == "__main__":
    main()