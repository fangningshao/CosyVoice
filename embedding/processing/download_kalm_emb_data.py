from datasets import load_dataset

# Download and cache to specific directory
ds = load_dataset(
    "KaLM-Embedding/KaLM-embedding-finetuning-data",
    cache_dir="D:/data/embedding_data"
)

# Optionally save to disk in a specific format
ds.save_to_disk("D:\\data\\embedding_data\\KaLM-embedding-finetuning-data")

# Or manual download using git lfs:
# git clone https://huggingface.co/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data D:/data/embedding_data/KaLM-embedding-finetuning-data