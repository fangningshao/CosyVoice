#!/usr/bin/env python3
"""
Filter JSONL files by audio existence and duration.
Removes samples with any audio file that is:
  - Missing (doesn't exist)
  - Too long (>30s by default)
  - Too short (<1s by default)
  - Has errors when reading

This script modifies JSONL files in-place, creating backups with .bak extension.

Usage:
    python filter_jsonl_by_audio.py --input data/embedding_data/OUTPUT --max_duration 30.0 --min_duration 1.0
    python filter_jsonl_by_audio.py --input data/file.train.jsonl --recursive
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import torchaudio
from tqdm import tqdm

# Global cache for audio duration checks
_audio_duration_cache: Dict[str, Tuple[bool, str]] = {}


def check_audio_duration(audio_path: str, min_duration: float, max_duration: float) -> Tuple[bool, str]:
    """
    Check if audio file exists and has valid duration (with caching).
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        (is_valid, reason) - True if valid, False otherwise with reason
    """
    # Check cache first
    cache_key = f"{audio_path}|{min_duration}|{max_duration}"
    if cache_key in _audio_duration_cache:
        return _audio_duration_cache[cache_key]
    
    # Perform actual check
    if not audio_path or audio_path == "null":
        result = (True, "no_audio")  # Null audio is valid (text-only query)
    elif not os.path.exists(audio_path):
        result = (False, "missing")
    else:
        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            if duration < min_duration:
                result = (False, f"too_short_{duration:.2f}s")
            elif duration > max_duration:
                result = (False, f"too_long_{duration:.2f}s")
            else:
                result = (True, "valid")
        except Exception as e:
            result = (False, f"error_{str(e)[:30]}")
    
    # Cache the result
    _audio_duration_cache[cache_key] = result
    return result


def validate_sample(sample: dict, min_duration: float, max_duration: float) -> Tuple[bool, dict, str]:
    """
    Validate a single JSONL sample and filter out invalid audio files.
    
    Args:
        sample: JSONL sample dictionary
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        
    Returns:
        (is_valid, filtered_sample, reason) - Whether sample is valid, filtered sample, and reason if invalid
    """
    # Determine data format and extract audio paths
    if 'query_text' in sample and 'query_wav' in sample:
        # Standard format
        query_audio = sample.get('query_wav')
        positive_audio = sample.get('pos_wav')
        negative_audios = sample.get('neg_wavs', [])
        
        # Check query audio (optional)
        if query_audio:
            is_valid, reason = check_audio_duration(query_audio, min_duration, max_duration)
            if not is_valid:
                return False, sample, f"query_{reason}"
        
        # Check positive audio (required)
        if not positive_audio:
            return False, sample, "missing_positive"
        
        is_valid, reason = check_audio_duration(positive_audio, min_duration, max_duration)
        if not is_valid:
            return False, sample, f"positive_{reason}"
        
        # Filter negative audios
        if negative_audios:
            valid_negatives = []
            for neg_audio in negative_audios:
                is_valid, _ = check_audio_duration(neg_audio, min_duration, max_duration)
                if is_valid:
                    valid_negatives.append(neg_audio)
            
            sample['neg_wavs'] = valid_negatives
        
        return True, sample, "valid"
        
    elif 'query' in sample and 'pos' in sample:
        # KaLM format
        query_audio = sample.get('query_wav')
        positive_audio = sample['pos'][0] if isinstance(sample['pos'], list) else sample['pos']
        negative_audios = sample.get('neg', [])
        
        # Check query audio (optional)
        if query_audio:
            is_valid, reason = check_audio_duration(query_audio, min_duration, max_duration)
            if not is_valid:
                return False, sample, f"query_{reason}"
        
        # Check positive audio (required)
        if not positive_audio:
            return False, sample, "missing_positive"
        
        is_valid, reason = check_audio_duration(positive_audio, min_duration, max_duration)
        if not is_valid:
            return False, sample, f"positive_{reason}"
        
        # Filter negative audios
        if negative_audios:
            valid_negatives = []
            for neg_audio in negative_audios:
                is_valid, _ = check_audio_duration(neg_audio, min_duration, max_duration)
                if is_valid:
                    valid_negatives.append(neg_audio)
            
            sample['neg'] = valid_negatives
        
        return True, sample, "valid"
        
    else:
        # Old format
        query_audio = sample.get('query')
        positive_audio = sample.get('positive')
        negative_audios = sample.get('negatives', [])
        
        # Check query audio (optional)
        if query_audio:
            is_valid, reason = check_audio_duration(query_audio, min_duration, max_duration)
            if not is_valid:
                return False, sample, f"query_{reason}"
        
        # Check positive audio (required)
        if not positive_audio:
            return False, sample, "missing_positive"
        
        is_valid, reason = check_audio_duration(positive_audio, min_duration, max_duration)
        if not is_valid:
            return False, sample, f"positive_{reason}"
        
        # Filter negative audios
        if negative_audios:
            valid_negatives = []
            for neg_audio in negative_audios:
                is_valid, _ = check_audio_duration(neg_audio, min_duration, max_duration)
                if is_valid:
                    valid_negatives.append(neg_audio)
            
            sample['negatives'] = valid_negatives
        
        return True, sample, "valid"


def filter_jsonl_file(jsonl_path: str, min_duration: float, max_duration: float, 
                     create_backup: bool = True, dry_run: bool = False) -> dict:
    """
    Filter a single JSONL file in-place.
    
    Args:
        jsonl_path: Path to JSONL file
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        create_backup: Whether to create .bak backup file
        dry_run: If True, only report statistics without modifying files
        
    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*80}")
    print(f"Processing: {jsonl_path}")
    print(f"{'='*80}")
    
    # Load samples
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(samples)} samples")
    
    # Validate and filter samples
    valid_samples = []
    invalid_reasons = {}
    
    for sample in tqdm(samples, desc="Validating", ncols=100):
        is_valid, filtered_sample, reason = validate_sample(sample, min_duration, max_duration)
        
        if is_valid:
            valid_samples.append(filtered_sample)
        else:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    
    # Statistics
    total_samples = len(samples)
    valid_count = len(valid_samples)
    removed_count = total_samples - valid_count
    
    stats = {
        'file': jsonl_path,
        'total': total_samples,
        'valid': valid_count,
        'removed': removed_count,
        'reasons': invalid_reasons
    }
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Total samples:    {total_samples}")
    print(f"  Valid samples:    {valid_count} ({valid_count/total_samples*100:.1f}%)")
    print(f"  Removed samples:  {removed_count} ({removed_count/total_samples*100:.1f}%)")
    
    if invalid_reasons:
        print(f"\n  Removal reasons:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {reason}: {count}")
    
    # Write filtered file
    if not dry_run and removed_count > 0:
        # Create backup
        if create_backup:
            backup_path = jsonl_path + '.bak'
            shutil.copy2(jsonl_path, backup_path)
            print(f"\n💾 Backup created: {backup_path}")
        
        # Write filtered samples
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sample in valid_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✅ File updated: {jsonl_path}")
    elif dry_run:
        print(f"\n🔍 [DRY RUN] No files modified")
    else:
        print(f"\n✨ No changes needed (all samples valid)")
    
    return stats


def find_jsonl_files(input_path: str, recursive: bool = False) -> List[str]:
    """
    Find all JSONL files in input path, excluding .bak files and files that already have backups.
    
    Args:
        input_path: File or directory path
        recursive: Whether to search recursively
        
    Returns:
        List of JSONL file paths (excluding already filtered files)
    """
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix == '.jsonl' and not path.name.endswith('.bak.jsonl'):
            # Check if this file already has a .bak backup (already filtered)
            if not Path(str(path) + '.bak').exists():
                return [str(path)]
        return []
    
    if path.is_dir():
        if recursive:
            all_files = [f for f in path.rglob('*.jsonl') if not f.name.endswith('.bak.jsonl')]
        else:
            all_files = [f for f in path.glob('*.jsonl') if not f.name.endswith('.bak.jsonl')]
        
        # Filter out files that already have .bak backups (already processed)
        filtered_files = []
        for f in all_files:
            backup_path = Path(str(f) + '.bak')
            if not backup_path.exists():
                filtered_files.append(str(f))
            else:
                # File already filtered, skip it
                pass
        
        return filtered_files
    
    return []


def main():
    parser = argparse.ArgumentParser(
        description='Filter JSONL files by audio existence and duration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter a single file
  python filter_jsonl_by_audio.py --input data.train.jsonl
  
  # Filter all JSONL files in a directory
  python filter_jsonl_by_audio.py --input data/embedding_data/OUTPUT
  
  # Recursive search with custom duration limits
  python filter_jsonl_by_audio.py --input data/ --recursive --max_duration 25.0 --min_duration 0.5
  
  # Dry run (preview without modifying)
  python filter_jsonl_by_audio.py --input data/ --dry_run
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSONL file or directory containing JSONL files')
    parser.add_argument('--max_duration', type=float, default=30.0,
                       help='Maximum audio duration in seconds (default: 30.0)')
    parser.add_argument('--min_duration', type=float, default=1.0,
                       help='Minimum audio duration in seconds (default: 1.0)')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for JSONL files recursively in subdirectories')
    parser.add_argument('--no_backup', action='store_true',
                       help='Do not create .bak backup files')
    parser.add_argument('--dry_run', action='store_true',
                       help='Preview changes without modifying files')
    
    args = parser.parse_args()
    
    # Find JSONL files
    jsonl_files = find_jsonl_files(args.input, args.recursive)
    
    if not jsonl_files:
        print(f"❌ No JSONL files found in: {args.input}")
        return
    
    print(f"\n🔍 Found {len(jsonl_files)} JSONL file(s)")
    print(f"📏 Duration limits: {args.min_duration}s - {args.max_duration}s")
    
    if args.dry_run:
        print(f"🔍 [DRY RUN MODE] No files will be modified")
    
    # Process each file
    all_stats = []
    for jsonl_file in jsonl_files:
        try:
            stats = filter_jsonl_file(
                jsonl_file,
                args.min_duration,
                args.max_duration,
                create_backup=not args.no_backup,
                dry_run=args.dry_run
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error processing {jsonl_file}: {e}")
            continue
    
    # Print overall summary
    if len(all_stats) > 1:
        print(f"\n{'='*80}")
        print(f"📊 OVERALL SUMMARY")
        print(f"{'='*80}")
        
        total_all = sum(s['total'] for s in all_stats)
        valid_all = sum(s['valid'] for s in all_stats)
        removed_all = sum(s['removed'] for s in all_stats)
        
        print(f"Files processed:     {len(all_stats)}")
        print(f"Total samples:       {total_all}")
        print(f"Valid samples:       {valid_all} ({valid_all/total_all*100:.1f}%)")
        print(f"Removed samples:     {removed_all} ({removed_all/total_all*100:.1f}%)")
        
        # Aggregate reasons
        all_reasons = {}
        for stats in all_stats:
            for reason, count in stats['reasons'].items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        
        if all_reasons:
            print(f"\nAggregated removal reasons:")
            for reason, count in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
    
    print(f"\n✨ Done!")


if __name__ == '__main__':
    main()
