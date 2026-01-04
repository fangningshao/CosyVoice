# encoding: utf-8
import sys
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import re
from multiprocessing import Pool, cpu_count

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr:
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import fasttext (suppress repeated messages)
FASTTEXT_AVAILABLE = False
FASTTEXT_MODEL = None

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    print("✗ fasttext not available")
    sys.exit(1)

def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def load_fasttext_model():
    """Load fasttext language identification model."""
    global FASTTEXT_MODEL
    
    if FASTTEXT_MODEL is not None:
        return FASTTEXT_MODEL
    
    model_path = Path.home() / '.fasttext' / 'lid.176.bin'
    
    if not model_path.exists():
        print("\nDownloading fasttext language identification model...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        import urllib.request
        url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
        
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Model downloaded to {model_path}")
        except Exception as e:
            print(f"✗ Failed to download model: {e}")
            return None
    
    try:
        # Load model (suppress warnings)
        import warnings
        warnings.filterwarnings('ignore')
        FASTTEXT_MODEL = fasttext.load_model(str(model_path))
        print(f"✓ Fasttext model loaded from {model_path}")
        return FASTTEXT_MODEL
    except Exception as e:
        print(f"✗ Failed to load fasttext model: {e}")
        return None

def extract_query_text(text):
    """
    Extract actual query text from formatted strings.
    Handles formats like:
    - "Instruct: ... Query: <actual text>"
    - Plain text
    """
    if not isinstance(text, str):
        return ""
    
    # Try to find "Query: " pattern
    query_match = re.search(r'Query:\s*(.+?)(?:\n|$)', text, re.DOTALL)
    if query_match:
        return query_match.group(1).strip()
    
    # Fallback to original text
    return text.strip()

def detect_language_fasttext(text, model):
    """Detect language using fasttext."""
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    
    try:
        # Extract actual query text if formatted
        text_clean = extract_query_text(text)
        
        # Further clean and limit length
        text_clean = ' '.join(text_clean.split())[:500]
        
        if not text_clean:
            return 'unknown'
        
        # Predict language
        predictions = model.predict(text_clean, k=1)
        lang_code = predictions[0][0].replace('__label__', '')
        
        return lang_code
    except Exception:
        return 'unknown'

def detect_language_batch(texts):
    """Detect language for a batch of texts (for multiprocessing)."""
    global FASTTEXT_MODEL
    
    # Load model in subprocess if needed
    if FASTTEXT_MODEL is None:
        model_path = Path.home() / '.fasttext' / 'lid.176.bin'
        if model_path.exists():
            try:
                import warnings
                warnings.filterwarnings('ignore')
                FASTTEXT_MODEL = fasttext.load_model(str(model_path))
            except Exception:
                return ['unknown'] * len(texts)
        else:
            return ['unknown'] * len(texts)
    
    return [detect_language_fasttext(text, FASTTEXT_MODEL) for text in texts]

def analyze_language_distribution(df, text_column='text', batch_size=5000, num_workers=None):
    """
    Analyze language distribution using fasttext with parallel processing.
    
    Args:
        df: DataFrame to analyze
        text_column: Column containing text data
        batch_size: Number of texts to process per batch
        num_workers: Number of parallel workers
    """
    if text_column not in df.columns:
        text_cols = [col for col in df.columns if 'text' in col.lower() or 'sentence' in col.lower() or 'query' in col.lower()]
        if text_cols:
            text_column = text_cols[0]
        else:
            return None
    
    # Get number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"    Detecting languages ({num_workers} workers)... ", end='', flush=True)
    
    # Extract texts
    texts = df[text_column].tolist()
    total_texts = len(texts)
    
    # For small datasets, process serially
    if total_texts < 1000:
        languages = detect_language_batch(texts)
        print("Done")
        return Counter(languages)
    
    # Split into batches
    batches = [texts[i:i + batch_size] for i in range(0, total_texts, batch_size)]
    
    # Process in parallel
    try:
        with Pool(processes=num_workers) as pool:
            results = pool.map(detect_language_batch, batches)
        
        # Flatten results
        languages = [lang for batch_result in results for lang in batch_result]
        
    except Exception as e:
        print(f"\n⚠ Parallel processing failed ({e}), using serial...")
        languages = detect_language_batch(texts)
    
    print("Done")
    return Counter(languages)

def safe_value_repr(val):
    """Safely represent a value for display."""
    try:
        if val is None:
            return "<None>"
        
        try:
            if pd.isna(val):
                return "<NA>"
        except (ValueError, TypeError):
            pass
        
        if isinstance(val, (list, tuple)):
            if len(val) > 0:
                first_elem = str(val[0])[:100]
                return f"<array: {type(val).__name__}, length: {len(val)}> | First: {first_elem}..."
            else:
                return f"<array: {type(val).__name__}, length: 0>"
        elif isinstance(val, dict):
            val_str = str(val)[:150]
            if len(str(val)) > 150:
                val_str += "..."
            return val_str
        elif isinstance(val, bytes):
            return f"<bytes: {len(val)} bytes>"
        elif hasattr(val, '__len__') and not isinstance(val, str):
            first_elem = str(val[0])[:100]
            return f"<array: {type(val).__name__}, length: {len(val)}> | First: {first_elem}..."
        else:
            val_str = str(val)[:150]
            if len(str(val)) > 150:
                val_str += "..."
            return val_str
    except Exception as e:
        return f"<error: {type(e).__name__}>"

def analyze_parquet_files(base_dir="D:/data/embedding_data/KaLM-embedding-finetuning-data", 
                          show_full_samples=False,
                          num_workers=None):
    """Analyze all parquet files in the dataset directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"✗ Directory not found: {base_dir}")
        return
    
    print("="*100)
    print(f"KaLM Embedding Dataset Analysis Report")
    print("="*100)
    print(f"Base Directory: {base_dir}")
    print(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load fasttext model (only once)
    print("\nLoading fasttext language detection model...")
    fasttext_model = load_fasttext_model()
    if not fasttext_model:
        print("✗ Failed to load fasttext model")
        return
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"✓ Using {num_workers} parallel workers for language detection")
    print("="*100)
    
    # Find all parquet files
    datasets = defaultdict(list)
    for parquet_file in base_path.rglob("*.parquet"):
        dataset_name = parquet_file.parent.name
        datasets[dataset_name].append(parquet_file)
    
    if not datasets:
        print("\n✗ No parquet files found!")
        return
    
    print(f"\nFound {len(datasets)} dataset(s) with parquet files\n")
    
    total_size = 0
    total_rows = 0
    dataset_summary = []
    
    for dataset_name in sorted(datasets.keys()):
        files = sorted(datasets[dataset_name])
        
        print("\n" + "="*100)
        print(f"DATASET: {dataset_name}")
        print("="*100)
        
        dataset_size = 0
        dataset_rows = 0
        overall_lang_counts = Counter()
        
        print(f"\n{'File Name':<50} {'Size':>15} {'Rows':>15}")
        print("-"*100)
        
        for file_path in files:
            file_size = file_path.stat().st_size
            dataset_size += file_size
            
            try:
                df = pd.read_parquet(file_path)
                num_rows = len(df)
                dataset_rows += num_rows
                
                lang_counts = analyze_language_distribution(df, num_workers=num_workers)
                if lang_counts:
                    overall_lang_counts.update(lang_counts)
                
                print(f"{file_path.name:<50} {format_size(file_size):>15} {num_rows:>15,}")
            except Exception as e:
                print(f"{file_path.name:<50} {format_size(file_size):>15} {'ERROR':>15}")
                print(f"  ⚠ Error: {e}")
        
        print("-"*100)
        print(f"{'DATASET TOTAL':<50} {format_size(dataset_size):>15} {dataset_rows:>15,}")
        
        total_size += dataset_size
        total_rows += dataset_rows
        
        # Language distribution
        if overall_lang_counts:
            print(f"\nLANGUAGE DISTRIBUTION:")
            print("-"*100)
            print(f"{'Language Code':<20} {'Count':>15} {'Percentage':>15}")
            print("-"*100)
            
            for lang, count in overall_lang_counts.most_common():
                percentage = (count / dataset_rows * 100) if dataset_rows > 0 else 0
                print(f"{lang:<20} {count:>15,} {percentage:>14.2f}%")
            
            print("-"*100)
            
            dominant_langs = [lang for lang, _ in overall_lang_counts.most_common(3)]
            dataset_summary.append({
                'name': dataset_name,
                'rows': dataset_rows,
                'size': dataset_size,
                'languages': dominant_langs,
                'lang_counts': dict(overall_lang_counts)
            })
        
        # Schema
        print(f"\nSCHEMA:")
        print("-"*100)
        try:
            first_file = files[0]
            df_sample = pd.read_parquet(first_file)
            
            print(f"{'Column Name':<30} {'Data Type':<20} {'Non-Null':>15}")
            print("-"*100)
            
            for col in df_sample.columns:
                dtype = str(df_sample[col].dtype)
                non_null = df_sample[col].notna().sum()
                print(f"{col:<30} {dtype:<20} {non_null:>15,}")
            
            # Column examples
            print(f"\nCOLUMN EXAMPLES:")
            print("-"*100)
            
            for col in df_sample.columns:
                print(f"\n  Column: '{col}' (type: {df_sample[col].dtype})")
                
                for i, val in enumerate(df_sample[col].head(3)):
                    val_str = safe_value_repr(val)
                    print(f"    [{i}] {val_str}")
            
            if show_full_samples:
                print(f"\nFULL SAMPLE DATA (First 5 Rows):")
                print("-"*100)
                sample_df = df_sample.head(5)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                print(sample_df.to_string(index=True))
            
        except Exception as e:
            print(f"⚠ Error reading schema: {e}")
        
        print("="*100)
    
    # Overall summary
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    print(f"Total Datasets:     {len(datasets)}")
    print(f"Total Files:        {sum(len(files) for files in datasets.values())}")
    print(f"Total Size:         {format_size(total_size)}")
    print(f"Total Rows:         {total_rows:,}")
    print("="*100)
    
    # Datasets ranked
    print("\n" + "="*100)
    print("DATASETS RANKED BY SIZE (Descending)")
    print("="*100)
    print(f"{'Rank':<6} {'Dataset Name':<40} {'Rows':>15} {'Size':>15} {'Top Languages':<30}")
    print("-"*100)
    
    dataset_summary.sort(key=lambda x: x['rows'], reverse=True)
    
    for rank, ds in enumerate(dataset_summary, 1):
        lang_str = ", ".join(ds['languages'][:3])
        print(f"{rank:<6} {ds['name']:<40} {ds['rows']:>15,} {format_size(ds['size']):>15} {lang_str:<30}")
    
    print("="*100)
    
    # Overall language distribution
    print("\n" + "="*100)
    print("OVERALL LANGUAGE DISTRIBUTION (All Datasets)")
    print("="*100)
    
    total_lang_counts = Counter()
    for ds in dataset_summary:
        total_lang_counts.update(ds['lang_counts'])
    
    print(f"{'Language Code':<25} {'Count':>15} {'Percentage':>15}")
    print("-"*100)
    
    for lang, count in total_lang_counts.most_common():
        percentage = (count / total_rows * 100) if total_rows > 0 else 0
        print(f"{lang:<25} {count:>15,} {percentage:>14.2f}%")
    
    print("="*100)

def save_report_to_file(base_dir="D:\\data\\embedding_data\\KaLM-embedding-finetuning-data", 
                        output_file="D:\\data\\embedding_data\\KaLM-embedding-finetuning-data-analysis.txt",
                        show_full_samples=False,
                        num_workers=None):
    """Save analysis report to a file."""
    original_stdout = sys.stdout
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            analyze_parquet_files(base_dir, show_full_samples, num_workers)
        
        sys.stdout = original_stdout
        print(f"✓ Report saved to: {output_file}")
        
    except Exception as e:
        sys.stdout = original_stdout
        print(f"✗ Error saving report: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze KaLM embedding dataset parquet files")
    parser.add_argument('--base_dir', type=str, 
                       default='D:/data/embedding_data/KaLM-embedding-finetuning-data',
                       help='Base directory containing dataset folders')
    parser.add_argument('--output', type=str, 
                        default="D:\\data\\embedding_data\\KaLM-embedding-finetuning-data-analysis.txt",
                        help='Output file path')
    parser.add_argument('--show_full_samples', action='store_true',
                        help='Show full sample data tables')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    
    args = parser.parse_args()
    
    if args.output:
        save_report_to_file(args.base_dir, args.output, args.show_full_samples, args.workers)
    else:
        analyze_parquet_files(args.base_dir, args.show_full_samples, args.workers)

if __name__ == "__main__":
    main()
