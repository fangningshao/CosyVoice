import os
import glob
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, desc: str):
    """Run a command and log its output"""
    logger = logging.getLogger(__name__)
    logger.info(f"Step: {desc}")
    logger.info(f"Running command: {cmd}")
    
    # 使用 subprocess.run 替代 Popen，并设置 bufsize=1
    try:
        process = subprocess.run(
            cmd,
            shell=True,
            text=True,
            check=True,  # 如果命令返回非零状态码则抛出异常
            bufsize=1,   # 行缓冲
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # 打印输出
        if process.stdout:
            logger.info(process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        raise


def setup_logger():
    """Setup basic logging configuration"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)


def collect_wav_txt_data(wav_dir: str, txt_dir: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Collect data from wav and txt folders
    
    Args:
        wav_dir: Directory containing wav files
        txt_dir: Directory containing txt files
        
    Returns:
        Tuple of dictionaries (utt2wav, utt2text, utt2spk, spk2utt)
    """
    
    wav_dir = Path(wav_dir)
    txt_dir = Path(txt_dir)
    
    wavs = list(wav_dir.glob("*.wav"))
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    
    for wav_file in tqdm.tqdm(wavs, desc="Collecting files"):
        txt_file = txt_dir / f"{wav_file.stem}.txt"
        if not txt_file.exists():
            logging.warning(f'{txt_file} does not exist')
            continue
            
        with open(txt_file, encoding='utf-8') as f:
            content = ''.join(l.strip() for l in f.readlines())
            
        utt = wav_file.stem  # 使用文件名作为utterance ID
        spk = "speaker1"     # 如果需要，这里可以从文件名提取说话人ID
        
        utt2wav[utt] = str(wav_file)
        utt2text[utt] = content
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)
        
    return utt2wav, utt2text, utt2spk, spk2utt

def save_single_parquet(utt: str, wav_path: str, text: str, spk: str, parquet_dir: Path):
    """Save single utterance data as parquet file"""
    # Read audio data
    with open(wav_path, 'rb') as wf:
        audio_data = wf.read()
    
    # Create parquet file
    parquet_file = parquet_dir / f"{utt}.parquet"
    table = pa.Table.from_pydict({
        "audio_data": [audio_data],
        "text": [text],
        "spk": [spk],
        "utt": [utt]
    })
    pq.write_table(table, parquet_file)
    return parquet_file


def prepare_data_initial(
    wav_dir: str,
    txt_dir: str,
    des_dir: str,
    cv_ratio: float = 0.05
) -> None:
    """Convert wav and txt files to parquet files
    Args:
        src_dir: Source directory containing wav files and transcripts
        des_dir: Destination directory for parquet files
        cv_ratio: Ratio of validation set (default: 0.05)
    """
    logger = setup_logger()
    
    # Setup output directories
    des_dir = Path(des_dir)
    # train_dir = des_dir / "train"
    # cv_dir = des_dir / "dev"
    # train_parquet_dir = train_dir / "parquet_files"
    # cv_parquet_dir = cv_dir / "parquet_files"
    for d in [des_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Collect data
    logger.info("Collecting wav and txt files...")
    utt2wav, utt2text, utt2spk, spk2utt = collect_wav_txt_data(wav_dir, txt_dir)
    
    # # Split data  -- you need to do this manually now
    # all_utts = list(utt2wav.keys())
    # import random
    # random.seed(42)
    # random.shuffle(all_utts)
    
    # cv_size = max(1, int(len(all_utts) * cv_ratio))
    # cv_utts = all_utts[:cv_size]
    # train_utts = all_utts[cv_size:]
    
    
    # Save Kaldi-style files for reference
    def save_kaldi_file(data: Dict, filename: str):
        with open(des_dir / filename, 'w', encoding='utf-8') as f:
            for k, v in data.items():
                if isinstance(v, list):
                    f.write(f"{k} {' '.join(v)}\n")
                else:
                    f.write(f"{k} {v}\n")
    
    save_kaldi_file(utt2wav, "wav.scp")
    save_kaldi_file(utt2text, "text")
    save_kaldi_file(utt2spk, "utt2spk")
    save_kaldi_file(spk2utt, "spk2utt")
    
    logger.info("Done!")



def prepare_data(
    wav_dir: str,
    txt_dir: str,
    des_dir: str,
    cv_ratio: float = 0.05,
    cosyvoice_model_dir: str = "D:\\models\\cosyvoice_models\\CosyVoice2-0.5B",
    num_utts_per_parquet: int = 1000,
    num_processes: int = 10,
    start_from_step: int = 1
) -> None:
    """Convert wav and txt files to parquet files and run all processing steps
    
    Args:
        wav_dir: Directory containing wav files
        txt_dir: Directory containing txt files
        des_dir: Destination directory for processed files
        cv_ratio: Ratio of validation set
        cosyvoice_model_dir: Directory containing model files
        num_utts_per_parquet: Number of utterances per parquet file
        num_processes: Number of processes for parquet generation
    """
    logger = setup_logger()
    des_dir = Path(des_dir)
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    tools_dir = script_dir.parent / "tools"
    dataproc_dir = script_dir.parent / "dataproc"
    
    if start_from_step <= 1:
        # Step 1: Prepare initial data
        logger.info("Step 1: Preparing initial data structure...")
        prepare_data_initial(wav_dir, txt_dir, des_dir, cv_ratio)

    if start_from_step <= 2:
        # Step 2: Extract voice embedding
        logger.info("Step 2: Extracting voice embedding...")
        cmd = (f"python {tools_dir / 'extract_embedding.py'} "
            f"--dir {des_dir} "
            f"--onnx_path {Path(cosyvoice_model_dir) / 'campplus.onnx'}")
        run_command(cmd, "Voice embedding extraction")
    
    if start_from_step <= 3:
        # Step 3: Extract speech token
        logger.info("Step 3: Extracting speech token...")
        cmd = (f"python {tools_dir / 'extract_speech_token.py'} "
            f"--dir {des_dir} "
            f"--onnx_path {Path(cosyvoice_model_dir) / 'speech_tokenizer_v2.onnx'}")
        run_command(cmd, "Speech token extraction")

    if start_from_step <= 4:
        # Step 4: Generate parquet files
        logger.info("Step 4: Generating parquet files...")
        parquet_dir = des_dir / "parquet"
        parquet_dir.mkdir(exist_ok=True)
        
        cmd = (f"python {tools_dir / 'make_parquet_list.py'} "
            f"--num_utts_per_parquet {num_utts_per_parquet} "
            f"--num_processes {num_processes} "
            f"--src_dir {des_dir} "
            f"--des_dir {parquet_dir}")
        run_command(cmd, "Parquet file generation")
        
    logger.info("All steps completed successfully!")


def main():
    """Usage:
    python dataproc\prepare_data_full.py ^
        \PATH\TO\filtered-wavs ^
        \PATH\TO\filtered-txts ^
        \PATH\TO\output-cv-data ^
        --cosyvoice-model-dir=\PATH\TO\CosyVoice2-0.5B ^
        --num-utts-per-parquet=1000 ^
        --num-processes=10
    """
    import argh
    argh.dispatch_command(prepare_data)

if __name__ == "__main__":
    main()
