# Preparing training data

1. First, prepare aligned wavs and txts inside the directory.
2. Run the following commands to prepare files:

```bash
python dataproc\prepare_data_full.py ^
        D:\path\to\your\wavs-dev ^
        D:\path\to\your\txts-dev ^
        D:\path\to\your\output_cv_data-dev ^
        --cosyvoice-model-dir=\PATH\TO\models\cosyvoice_models\CosyVoice2-0.5B ^
        --num-utts-per-parquet=500 ^
        --num-processes=10

python dataproc\prepare_data_full.py ^
        D:\path\to\your\wavs-train ^
        D:\path\to\your\txts-train ^
        D:\path\to\your\output_cv_data-train ^
        --cosyvoice-model-dir=\PATH\TO\models\cosyvoice_models\CosyVoice2-0.5B ^
        --num-utts-per-parquet=500 ^
        --num-processes=10
```

(set --num-utts-per-parquet to 500 or 1000 depend on how much data you have.)
