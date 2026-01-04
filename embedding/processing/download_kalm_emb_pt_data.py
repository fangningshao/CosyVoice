from datasets import load_dataset

# Download and cache to specific directory
ds = load_dataset(
    "KaLM-Embedding/KaLM-embedding-pretrain-data",
    cache_dir="D:/data/embedding_data"
)

# Optionally save to disk in a specific format
ds.save_to_disk("D:\\data\\embedding_data\\KaLM-embedding-pretrain-data")

# Or manual download using git lfs (recommended):
# git clone https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-pretraining-data D:/data/embedding_data/KaLM-embedding-pretraining-data