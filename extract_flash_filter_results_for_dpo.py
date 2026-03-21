"""
Extract and organize passed samples from a flash-filter JSONL result file.

Steps
-----
1. Read the JSONL file and keep only records where  passed == true.
2. Copy the source wav  →  wav_filtered/<filename>
3. Write the transcript  →  txt_filtered/<stem>.txt          (with [laughter])
4. Write the transcript  →  txt_filtered_nolaughter/<stem>.txt  (without [laughter])

Usage
-----
python extract_flash_filter_results_for_dpo.py ^
    D:\\data\\20260307_laughter-full-output\\epoch4-flash-filter-laughter-2-v3.jsonl ^
    --wav-out-dir   D:\\data\\dpo_laughter\\wav_filtered ^
    --txt-out-dir   D:\\data\\dpo_laughter\\txt_filtered ^
    --txt-nolaughter-out-dir D:\\data\\dpo_laughter\\txt_filtered_nolaughter
"""

import json
import logging
import re
import shutil
from pathlib import Path

import argh


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def extract(
    jsonl_path: str,
    wav_out_dir: str = "wav_filtered",
    txt_out_dir: str = "txt_filtered",
    txt_nolaughter_out_dir: str = "txt_filtered_nolaughter",
) -> None:
    """Filter flash-filter JSONL results and organise outputs for DPO data prep.

    Args:
        jsonl_path:              Path to the input .jsonl file.
        wav_out_dir:             Destination directory for passing wav files.
        txt_out_dir:             Destination directory for transcripts
                                 (with [laughter] markers preserved).
        txt_nolaughter_out_dir:  Destination directory for transcripts
                                 with [laughter] markers removed.
    """
    logger = setup_logger()

    jsonl_path = Path(jsonl_path)
    wav_dir = Path(wav_out_dir)
    txt_dir = Path(txt_out_dir)
    txt_nl_dir = Path(txt_nolaughter_out_dir)

    for d in (wav_dir, txt_dir, txt_nl_dir):
        d.mkdir(parents=True, exist_ok=True)

    total = passed = skipped_missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)

            # Skip failed records
            if not record.get("passed", False):
                continue

            transcript = record.get("transcript") or ""
            if not transcript:
                logger.warning(f"Passed record has empty transcript, skipping: {record.get('filename')}")
                continue

            src_wav = Path(record["filepath"])
            stem = src_wav.stem          # e.g. "04855"
            filename = src_wav.name      # e.g. "04855.wav"

            # 1. Copy wav
            if not src_wav.exists():
                logger.warning(f"Source wav not found, skipping: {src_wav}")
                skipped_missing += 1
                continue

            shutil.copy2(src_wav, wav_dir / filename)

            # 2. Write transcript with [laughter]
            (txt_dir / f"{stem}.txt").write_text(transcript, encoding="utf-8")

            # 3. Write transcript without [laughter]
            clean = re.sub(r"\[laughter\]", "", transcript, flags=re.IGNORECASE)
            # Collapse multiple spaces that may result from removal
            clean = re.sub(r" {2,}", " ", clean).strip()
            (txt_nl_dir / f"{stem}.txt").write_text(clean, encoding="utf-8")

            passed += 1

    logger.info("=" * 60)
    logger.info(f"Total records  : {total}")
    logger.info(f"Passed         : {passed}")
    logger.info(f"Failed/skipped : {total - passed - skipped_missing}")
    logger.info(f"Missing wavs   : {skipped_missing}")
    logger.info(f"wav_filtered   → {wav_dir}  ({passed} files)")
    logger.info(f"txt_filtered   → {txt_dir}  ({passed} files)")
    logger.info(f"txt_nolaughter → {txt_nl_dir}  ({passed} files)")
    logger.info("=" * 60)


if __name__ == "__main__":
    argh.dispatch_command(extract)
