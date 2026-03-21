"""
DPO (Direct Preference Optimization) data preparation for CosyVoice3.

Input layout (all for a single speaker):
    wav/          - positive (chosen) wav files
    wav_reject/   - rejected wav files  (same filenames / utt IDs as wav/)
    txt/          - transcripts shared by both sets (one .txt per utt)

What this script does
---------------------
Step 1 – Build Kaldi-style files for both the positive dir and the _reject dir:
         wav.scp / text / utt2spk / spk2utt

Step 2 – Extract campplus speaker embeddings for the POSITIVE set only.
         (make_parquet_list --dpo does NOT need embeddings for the reject set)

Step 3 – Extract speech tokens for BOTH the positive AND the reject set.
         The reject speech tokens are saved to  des_dir_reject/utt2speech_token.pt
         which is exactly where make_parquet_list.py --dpo looks for them.

Step 4 – Build parquet files with --dpo flag.
         make_parquet_list.py will automatically read
         {src_dir}_reject/utt2speech_token.pt  and write a
         'reject_speech_token' column alongside the normal columns.

Usage
-----
python dataproc\\prepare_data_full_cv3_dpo.py ^
    D:\\data\\MySpk\\wav ^
    D:\\data\\MySpk\\wav_reject ^
    D:\\data\\MySpk\\txt ^
    D:\\data\\MySpk\\output-cv-dpo ^
    --cosyvoice-model-dir=D:\\models\\cosyvoice_models\\CosyVoice3-0.5B-2512 ^
    --num-utts-per-parquet=1000 ^
    --num-processes=10

To resume from a specific step (e.g. skip step 1 and 2):
    ... --start-from-step=3
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def run_command(cmd: str, desc: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Step: {desc}")
    logger.info(f"Running: {cmd}")
    try:
        process = subprocess.run(
            cmd,
            shell=True,
            text=True,
            check=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout:
            logger.info(process.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed (rc={e.returncode})")
        logger.error(f"stderr: {e.stderr}")
        raise


def collect_wav_data(wav_dir: Path, txt_dir: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """Scan wav_dir for *.wav and pair each with a transcript in txt_dir.

    Returns (utt2wav, utt2text, utt2spk, spk2utt).
    For single-speaker data the speaker ID is fixed to 'speaker1'.
    """
    wavs = sorted(wav_dir.glob("*.wav"))
    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    spk = "speaker1"

    for wav_file in tqdm.tqdm(wavs, desc=f"Scanning {wav_dir.name}"):
        txt_file = txt_dir / f"{wav_file.stem}.txt"
        if not txt_file.exists():
            logging.warning(f"No transcript for {wav_file.name}, skipping.")
            continue

        with open(txt_file, encoding="utf-8") as f:
            content = "".join(line.strip() for line in f)

        utt = wav_file.stem
        utt2wav[utt] = str(wav_file)
        utt2text[utt] = content
        utt2spk[utt] = spk
        spk2utt.setdefault(spk, []).append(utt)

    return utt2wav, utt2text, utt2spk, spk2utt


def collect_wav_only(wav_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Scan wav_dir for *.wav without needing transcripts.

    Used for the reject set where we only need wav.scp + utt2spk for
    speech-token extraction.  Returns (utt2wav, utt2spk, spk2utt).
    """
    wavs = sorted(wav_dir.glob("*.wav"))
    utt2wav, utt2spk, spk2utt = {}, {}, {}
    spk = "speaker1"

    for wav_file in tqdm.tqdm(wavs, desc=f"Scanning {wav_dir.name}"):
        utt = wav_file.stem
        utt2wav[utt] = str(wav_file)
        utt2spk[utt] = spk
        spk2utt.setdefault(spk, []).append(utt)

    return utt2wav, utt2spk, spk2utt


def save_kaldi_files(
    des_dir: Path,
    utt2wav: Dict,
    utt2spk: Dict,
    spk2utt: Dict,
    utt2text: Dict = None,
) -> None:
    """Write wav.scp / utt2spk / spk2utt (and optionally text) into des_dir."""

    def write(data: Dict, filename: str):
        with open(des_dir / filename, "w", encoding="utf-8") as f:
            for k, v in data.items():
                if isinstance(v, list):
                    f.write(f"{k} {' '.join(v)}\n")
                else:
                    f.write(f"{k} {v}\n")

    write(utt2wav, "wav.scp")
    write(utt2spk, "utt2spk")
    write(spk2utt, "spk2utt")
    if utt2text is not None:
        write(utt2text, "text")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_data_dpo(
    wav_dir: str,
    wav_reject_dir: str,
    txt_dir: str,
    des_dir: str,
    cosyvoice_model_dir: str = "D:\\models\\cosyvoice_models\\CosyVoice3-0.5B-2512",
    num_utts_per_parquet: int = 1000,
    num_processes: int = 10,
    start_from_step: int = 1,
) -> None:
    """Full DPO data preparation pipeline.

    Args:
        wav_dir:             Directory of POSITIVE (chosen) wav files.
        wav_reject_dir:      Directory of REJECTED wav files.
                             File stems must match those in wav_dir.
        txt_dir:             Directory of transcript .txt files (shared by both sets).
        des_dir:             Output root for the positive set kaldi/parquet data.
                             The reject set will be written to  des_dir + '_reject'
                             (e.g. 'output-cv-dpo_reject') because that is the
                             exact path that make_parquet_list.py --dpo expects.
        cosyvoice_model_dir: Path to CosyVoice3 model dir (needs campplus.onnx
                             and speech_tokenizer_v3.onnx).
        num_utts_per_parquet: Utterances packed into each parquet shard.
        num_processes:       Parallel workers for parquet generation.
        start_from_step:     Resume from this step (1–4).
    """
    logger = setup_logger()

    des_dir = Path(des_dir)
    # make_parquet_list.py --dpo hardcodes the reject dir as {src_dir}_reject
    reject_dir = Path(str(des_dir) + "_reject")

    script_dir = Path(__file__).parent
    tools_dir = script_dir.parent / "tools"
    model_dir = Path(cosyvoice_model_dir)

    # -----------------------------------------------------------------------
    # Step 1: Build Kaldi-style files for positive AND reject sets
    # -----------------------------------------------------------------------
    if start_from_step <= 1:
        logger.info("=" * 60)
        logger.info("Step 1: Building Kaldi-style data files")
        logger.info("=" * 60)

        # -- Positive set --
        des_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Positive dir: {des_dir}")
        utt2wav, utt2text, utt2spk, spk2utt = collect_wav_data(
            Path(wav_dir), Path(txt_dir)
        )
        save_kaldi_files(des_dir, utt2wav, utt2spk, spk2utt, utt2text)
        logger.info(f"  Positive utterances: {len(utt2wav)}")

        # -- Reject set --
        reject_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Reject dir:   {reject_dir}")
        r_utt2wav, r_utt2spk, r_spk2utt = collect_wav_only(Path(wav_reject_dir))
        # Verify alignment: every positive utt must have a reject counterpart
        missing = set(utt2wav.keys()) - set(r_utt2wav.keys())
        if missing:
            logger.warning(
                f"  {len(missing)} positive utts have NO reject counterpart "
                f"(first 5: {list(missing)[:5]}). They will be skipped by "
                f"make_parquet_list at parquet-build time."
            )
        extra = set(r_utt2wav.keys()) - set(utt2wav.keys())
        if extra:
            logger.warning(
                f"  {len(extra)} reject utts have no positive match "
                f"and will be ignored."
            )
        # For the reject set we only need wav.scp + utt2spk + spk2utt
        # (no 'text' file required – make_parquet_list reads text from positive side)
        save_kaldi_files(reject_dir, r_utt2wav, r_utt2spk, r_spk2utt)
        logger.info(f"  Reject utterances: {len(r_utt2wav)}")

    # -----------------------------------------------------------------------
    # Step 2: Extract campplus speaker embeddings  ← POSITIVE set only
    # -----------------------------------------------------------------------
    if start_from_step <= 2:
        logger.info("=" * 60)
        logger.info("Step 2: Extracting speaker embeddings (positive set only)")
        logger.info("=" * 60)
        cmd = (
            f"python {tools_dir / 'extract_embedding.py'} "
            f"--dir {des_dir} "
            f"--onnx_path {model_dir / 'campplus.onnx'}"
        )
        run_command(cmd, "Speaker embedding extraction")

    # -----------------------------------------------------------------------
    # Step 3: Extract speech tokens for BOTH positive and reject sets
    # -----------------------------------------------------------------------
    if start_from_step <= 3:
        logger.info("=" * 60)
        logger.info("Step 3: Extracting speech tokens (positive + reject)")
        logger.info("=" * 60)

        tokenizer_onnx = model_dir / "speech_tokenizer_v3.onnx"

        # Positive
        cmd = (
            f"python {tools_dir / 'extract_speech_token.py'} "
            f"--dir {des_dir} "
            f"--onnx_path {tokenizer_onnx}"
        )
        run_command(cmd, "Speech token extraction – positive set")

        # Reject  (produces reject_dir/utt2speech_token.pt)
        cmd = (
            f"python {tools_dir / 'extract_speech_token.py'} "
            f"--dir {reject_dir} "
            f"--onnx_path {tokenizer_onnx}"
        )
        run_command(cmd, "Speech token extraction – reject set")

    # -----------------------------------------------------------------------
    # Step 4: Build parquet shards with --dpo flag
    # make_parquet_list.py will automatically load
    #   {src_dir}_reject/utt2speech_token.pt  (i.e. reject_dir)
    # and write a 'reject_speech_token' column into every parquet shard.
    # -----------------------------------------------------------------------
    if start_from_step <= 4:
        logger.info("=" * 60)
        logger.info("Step 4: Building DPO parquet files")
        logger.info("=" * 60)

        parquet_dir = des_dir / "parquet"
        parquet_dir.mkdir(exist_ok=True)

        cmd = (
            f"python {tools_dir / 'make_parquet_list.py'} "
            f"--num_utts_per_parquet {num_utts_per_parquet} "
            f"--num_processes {num_processes} "
            f"--dpo "
            f"--src_dir {des_dir} "
            f"--des_dir {parquet_dir}"
        )
        run_command(cmd, "DPO parquet generation")

    logger.info("=" * 60)
    logger.info("All DPO preparation steps completed successfully!")
    logger.info(f"  Parquet list: {des_dir / 'parquet' / 'data.list'}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """
    Examples
    --------

    Full run from step 1:

        python dataproc\\prepare_data_full_cv3_dpo.py ^
            D:\\data\\MySpk\\wav ^
            D:\\data\\MySpk\\wav_reject ^
            D:\\data\\MySpk\\txt ^
            D:\\data\\MySpk\\output-cv-dpo ^
            --cosyvoice-model-dir=D:\\models\\cosyvoice_models\\CosyVoice3-0.5B-2512 ^
            --num-utts-per-parquet=1000 ^
            --num-processes=10

    Resume from step 3 (e.g. embeddings already done):

        python dataproc\\prepare_data_full_cv3_dpo.py ^
            D:\\data\\MySpk\\wav ^
            D:\\data\\MySpk\\wav_reject ^
            D:\\data\\MySpk\\txt ^
            D:\\data\\MySpk\\output-cv-dpo ^
            --cosyvoice-model-dir=D:\\models\\cosyvoice_models\\CosyVoice3-0.5B-2512 ^
            --start-from-step=3

    The resulting data.list is at:
        D:\\data\\MySpk\\output-cv-dpo\\parquet\\data.list

    Use it with  --train_data  and  --dpo  flag in train.py.
    """
    import argh
    argh.dispatch_command(prepare_data_dpo)


if __name__ == "__main__":
    main()
