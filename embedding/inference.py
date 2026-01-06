# Copyright (c) 2026 (authors: Fangning Shao)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.append('.')
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.llm.llm import CosyVoice3LM
from embedding.model import CosyVoice3Embedding
from cosyvoice.utils.file_utils import logging


class CosyVoice3EmbeddingInference:
    """
    Inference wrapper for CosyVoice3Embedding model.
    
    Provides convenient methods to:
    - Encode text instructions with voice samples
    - Generate embeddings for voice retrieval
    - Batch processing of voice samples
    """
    
    def __init__(self, 
                model_dir: str, 
                device: str = 'cuda', 
                qwen_path: str = None,
                checkpoint_path: str = None):
        """
        Args:
            model_dir: Directory containing CosyVoice3 model files
            device: Device to run inference on ('cuda' or 'cpu')
            qwen_path: Path to Qwen model (optional override)
            checkpoint_path: Path to custom trained checkpoint (optional)
                        e.g., "exp/embedding/checkpoints/checkpoint_epoch0_step160.pt"
                        If None, uses the base LLM checkpoint from model_dir
        """
        self.model_dir = model_dir
        self.device = device
        self.qwen_path = qwen_path
        self.checkpoint_path = checkpoint_path
        
        # Load config using hyperpyyaml (same as CosyVoice)
        config_path = os.path.join(model_dir, 'cosyvoice3.yaml')
        
        if not os.path.exists(config_path):
            raise ValueError(f'Config file not found in {model_dir}!')
        
        logging.info(f"Loading config from {config_path}")
        
        # Load config with override
        overrides = {}
        if qwen_path:
            overrides['qwen_pretrain_path'] = qwen_path
            
        with open(config_path, 'r') as f:
            self.configs = load_hyperpyyaml(f, overrides=overrides)
        
        # Get sample rate from config
        self.sample_rate = self.configs.get('sample_rate', 22050)
        
        # Initialize frontend (reuse CosyVoice's frontend)
        logging.info("Initializing frontend...")
        self.frontend = CosyVoiceFrontEnd(
            self.configs['get_tokenizer'],
            self.configs['feat_extractor'],
            os.path.join(model_dir, 'campplus.onnx'),
            os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
            os.path.join(model_dir, 'spk2info.pt'),
            self.configs['allowed_special'],
        )
        
        # Get LLM from config (it's already initialized by hyperpyyaml)
        logging.info("Getting LLM from config...")
        self.llm = self.configs['llm']
        
        # Initialize embedding model (wraps the LLM)
        logging.info("Initializing embedding model...")
        self.model = CosyVoice3Embedding(
            self.llm,
            speech_tokenizer_path=os.path.join(model_dir, 'speech_tokenizer_v3.onnx')
        )
        
        # ====================================================================
        # LOADING PROCEDURE (same as eval_checkpoint.py):
        # 1. Load base LLM weights from llm.pt
        # 2. Apply LoRA (if training checkpoint has LoRA)
        # 3. Load LoRA weights from training checkpoint
        # ====================================================================
        
        # STEP 1: Always load base LLM weights first
        llm_path = self._find_llm_checkpoint()
        logging.info("="*80)
        logging.info("STEP 1: Loading base LLM weights")
        logging.info("="*80)
        logging.info(f"Loading from: {llm_path}")
        self.model.load_llm(llm_path, strict=False)
        logging.info("✓ Base LLM weights loaded successfully")
        
        # STEP 2 & 3: If training checkpoint provided, check if it has LoRA
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self._is_training_checkpoint(checkpoint_path):
                logging.info("\n" + "="*80)
                logging.info("STEP 2 & 3: Applying LoRA and loading trained weights")
                logging.info("="*80)
                logging.info(f"Loading trained checkpoint from: {checkpoint_path}")
                self._load_trained_checkpoint(checkpoint_path)
            else:
                # It's a base LLM checkpoint, but we already loaded base LLM above
                logging.warning(f"Checkpoint {checkpoint_path} is a base LLM checkpoint, already loaded from {llm_path}")
        else:
            logging.info("\n" + "="*80)
            logging.info("No training checkpoint provided - using base LLM only")
            logging.info("="*80)
        
        self.model.to(device)
        self.model.eval()
        
        logging.info(f"\n✓ CosyVoice3EmbeddingInference initialized on {device}")

    def _is_training_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Check if checkpoint is a training checkpoint (has model_state_dict, optimizer, etc.)
        or a base LLM checkpoint (just model weights).
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if training checkpoint, False if base LLM checkpoint
        """
        try:
            # PyTorch 2.6+ requires weights_only=False for numpy types in checkpoints
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Training checkpoints have these keys
            training_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']
            
            # Check if it has training checkpoint structure
            has_training_keys = any(key in checkpoint for key in training_keys)
            
            if has_training_keys:
                logging.info("Detected training checkpoint format")
                return True
            else:
                logging.info("Detected base LLM checkpoint format")
                return False
                
        except Exception as e:
            logging.warning(f"Error checking checkpoint format: {e}")
            # Default to base LLM checkpoint
            return False

    def _find_llm_checkpoint(self) -> str:
        """Find LLM checkpoint in model directory."""
        # Try different checkpoint names
        candidates = [
            'llm.rl.pt',
            'llm.pt',
            'checkpoint.pt'
        ]
        
        for name in candidates:
            path = os.path.join(self.model_dir, name)
            if os.path.exists(path):
                return path
        
        raise ValueError(f"LLM checkpoint not found in {self.model_dir}")

    def _load_trained_checkpoint(self, checkpoint_path: str):
        """
        Load weights from a trained checkpoint.
        
        The checkpoint contains:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state (not needed for inference)
        - scheduler_state_dict: Scheduler state (not needed for inference)
        - epoch: Training epoch
        - step: Training step
        - val_loss: Validation loss
        - config: Training config including LoRA parameters
        """
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint (PyTorch 2.6+ requires weights_only=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Log checkpoint info
        if 'epoch' in checkpoint:
            logging.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            logging.info(f"  Checkpoint step: {checkpoint['step']}")
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            logging.info(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        
        # Extract model state dict
        if 'model_state_dict' not in checkpoint:
            raise ValueError(
                f"No 'model_state_dict' found in checkpoint {checkpoint_path}\n"
                f"This appears to be a base LLM checkpoint, not a training checkpoint.\n"
                f"Available keys: {list(checkpoint.keys())}"
            )
        
        state_dict = checkpoint['model_state_dict']
        
        # Handle DDP wrapped models (keys start with 'module.')
        if list(state_dict.keys())[0].startswith('module.'):
            logging.info("Removing 'module.' prefix from DDP checkpoint")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Check for LoRA weights in checkpoint
        lora_keys = [k for k in state_dict.keys() if 'lora_A' in k or 'lora_B' in k]
        use_lora = len(lora_keys) > 0
        
        if use_lora:
            logging.info("="*80)
            logging.info("✓ Detected LoRA checkpoint")
            logging.info("="*80)
            
            # Get LoRA config from checkpoint (preferred) or infer from weights
            if 'config' in checkpoint and checkpoint['config']:
                config = checkpoint['config']
                lora_rank = config.get('lora_r', config.get('lora_rank', 8))
                lora_alpha = config.get('lora_alpha', lora_rank * 2)
                lora_dropout = config.get('lora_dropout', 0.1)
                target_modules = config.get('target_modules', 
                    ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                logging.info("✓ Using LoRA config from checkpoint")
            else:
                # Infer from first lora_A weight
                first_lora_a = [k for k in lora_keys if 'lora_A' in k][0]
                lora_rank = state_dict[first_lora_a].shape[0]
                lora_alpha = lora_rank * 2
                lora_dropout = 0.1
                target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                logging.warning("⚠️  No config found in checkpoint, inferring LoRA parameters")
            
            logging.info(f"  LoRA rank: {lora_rank}")
            logging.info(f"  LoRA alpha: {lora_alpha}")
            logging.info(f"  LoRA dropout: {lora_dropout}")
            logging.info(f"  Target modules: {target_modules}")
            
            # Apply LoRA to the LLM
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            
            self.model.llm = get_peft_model(self.model.llm, lora_config)
            logging.info("✓ LoRA applied to model")
            
            # ================================================================
            # CRITICAL: Remap checkpoint keys to match PEFT structure
            # ================================================================
            # Training format:   llm.llm.model.model.layers.X.lora_A...
            # After get_peft_model(): llm.base_model.model.llm.model.model.layers.X.lora_A...
            
            logging.info("\nRemapping checkpoint keys to match PEFT structure...")
            
            model_state = self.model.state_dict()
            remapped_state = {}
            unmapped_keys = []
            
            for ckpt_key, ckpt_value in state_dict.items():
                # Try direct match first
                if ckpt_key in model_state:
                    remapped_state[ckpt_key] = ckpt_value
                # Remap LoRA keys: llm.llm.* -> llm.base_model.model.llm.*
                elif ckpt_key.startswith('llm.llm.'):
                    new_key = ckpt_key.replace('llm.llm.', 'llm.base_model.model.llm.')
                    if new_key in model_state:
                        remapped_state[new_key] = ckpt_value
                    else:
                        unmapped_keys.append((ckpt_key, new_key))
                else:
                    # Non-LLM keys (llm_decoder, etc.) - keep as is
                    if ckpt_key in model_state:
                        remapped_state[ckpt_key] = ckpt_value
                    else:
                        unmapped_keys.append((ckpt_key, None))
            
            logging.info(f"  Remapped {len(remapped_state)} parameters")
            if unmapped_keys:
                logging.warning(f"  Could not remap {len(unmapped_keys)} keys")
                for ckpt_key, new_key in unmapped_keys[:5]:
                    logging.warning(f"    - {ckpt_key}")
            
            # Use remapped state
            state_dict = remapped_state
            
            # ================================================================
            # CRITICAL: Verify LoRA weights will be loaded
            # ================================================================
            checkpoint_keys = set(state_dict.keys())
            model_keys = set(model_state.keys())
            
            matched_keys = checkpoint_keys & model_keys
            missing_keys = model_keys - checkpoint_keys
            
            # Check for missing LoRA weights
            missing_lora = [k for k in missing_keys if 'lora_' in k]
            matched_lora = [k for k in matched_keys if 'lora_' in k]
            
            logging.info(f"\nLoRA Parameter Alignment:")
            logging.info(f"  Total LoRA params in checkpoint: {len(lora_keys)}")
            logging.info(f"  After remapping - matched: {len(matched_lora)}")
            logging.info(f"  After remapping - missing: {len(missing_lora)}")
            
            if missing_lora:
                logging.error("="*80)
                logging.error("CRITICAL ERROR: LoRA weights missing after remapping!")
                logging.error("="*80)
                logging.error(f"Found {len(missing_lora)} LoRA parameters that would be randomly initialized:")
                for i, key in enumerate(missing_lora[:10]):
                    logging.error(f"  {i+1}. {key}")
                if len(missing_lora) > 10:
                    logging.error(f"  ... and {len(missing_lora) - 10} more")
                logging.error("\nThis means inference would use UNTRAINED LoRA weights!")
                logging.error("="*80)
                raise ValueError(f"Cannot proceed: {len(missing_lora)} LoRA weights missing from checkpoint")
            
            # Load state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Verify no LoRA weights in missing_keys after load
            missing_lora_after = [k for k in missing_keys if 'lora_' in k]
            if missing_lora_after:
                logging.error(f"❌ CRITICAL: {len(missing_lora_after)} LoRA weights still missing after load!")
                raise ValueError(f"LoRA weights not properly loaded: {missing_lora_after[:5]}")
            
            # Count loaded LoRA parameters
            loaded_lora = len(matched_lora)
            logging.info(f"\n✓ Successfully loaded {loaded_lora} LoRA parameters")
            
            # Missing base weights are expected (loaded from base LLM)
            missing_base = [k for k in missing_keys if 'lora_' not in k]
            if missing_base:
                logging.info(f"  {len(missing_base)} base weights missing (expected - using base LLM)")
            
            if unexpected_keys:
                logging.warning(f"  {len(unexpected_keys)} unexpected keys in checkpoint")
            
            logging.info("="*80)
            logging.info("✓ LoRA checkpoint loaded successfully!")
            logging.info("="*80)
        else:
            # No LoRA - regular checkpoint
            logging.info("Loading non-LoRA checkpoint...")
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys in checkpoint: {missing_keys[:10]}")
                if len(missing_keys) > 10:
                    logging.warning(f"  ... and {len(missing_keys) - 10} more")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:10]}")
                if len(unexpected_keys) > 10:
                    logging.warning(f"  ... and {len(unexpected_keys) - 10} more")
            
            logging.info("✓ Checkpoint loaded successfully")
                    
    def encode_voice(self, 
                    instruction_text: str,
                    voice_audio_path: str,
                    normalize: bool = True) -> np.ndarray:
        """
        Encode a voice sample with an instruction text.
        
        Args:
            instruction_text: Text instruction (e.g., "Retrieve semantically similar voice")
            voice_audio_path: Path to voice audio file
            normalize: Whether to normalize the embedding (for cosine similarity)
            
        Returns:
            embedding: numpy array of shape [896]
        """
        # Normalize instruction text
        instruction_text_normalized = self.frontend.text_normalize(
            instruction_text, 
            split=False, 
            text_frontend=True
        )
        
        # Tokenize text using the tokenizer
        text_token = self.frontend.tokenizer.encode(
            instruction_text_normalized, 
            allowed_special=self.frontend.allowed_special
        )
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        
        # Extract speech token using frontend's method (it handles loading internally)
        # _extract_speech_token loads at 16000 Hz and returns tokens
        speech_token, speech_token_len = self.frontend._extract_speech_token(voice_audio_path)
        
        # Move to device and add batch dimension if needed
        if speech_token.dim() == 1:
            speech_token = speech_token.unsqueeze(0)
        if speech_token_len.dim() == 0:
            speech_token_len = speech_token_len.unsqueeze(0)
        
        speech_token = speech_token.to(self.device)
        speech_token_len = speech_token_len.to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            output = self.model(text_token, text_token_len, speech_token, speech_token_len)
            embedding = output['embedding']
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]
        
        # Normalize if requested
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def encode_batch(self,
                     instruction_texts: list,
                     voice_audio_paths: list,
                     batch_size: int = 8,
                     normalize: bool = True) -> np.ndarray:
        """
        Encode multiple voice samples in batches.
        
        Args:
            instruction_texts: List of instruction texts
            voice_audio_paths: List of voice audio file paths
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            embeddings: numpy array of shape [num_samples, 896]
        """
        assert len(instruction_texts) == len(voice_audio_paths), \
            "Number of texts and audio files must match"
        
        embeddings = []
        
        for i in range(0, len(instruction_texts), batch_size):
            batch_texts = instruction_texts[i:i+batch_size]
            batch_audios = voice_audio_paths[i:i+batch_size]
            
            batch_embeddings = []
            for text, audio in zip(batch_texts, batch_audios):
                try:
                    emb = self.encode_voice(text, audio, normalize=normalize)
                    batch_embeddings.append(emb)
                except Exception as e:
                    logging.error(f"Failed to encode {audio}: {e}")
                    # Add zero embedding for failed samples
                    batch_embeddings.append(np.zeros(896))
            
            embeddings.extend(batch_embeddings)
            
            logging.info(f"Processed {min(i+batch_size, len(instruction_texts))}/{len(instruction_texts)} samples")
        
        return np.array(embeddings)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding [896]
            emb2: Second embedding [896]
            
        Returns:
            similarity: Cosine similarity score [-1, 1]
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    
    def retrieve_similar_voices(self,
                                query_audio_path: str,
                                database_audio_paths: list,
                                instruction_text: str = "Retrieve semantically similar voice",
                                top_k: int = 5) -> list:
        """
        Retrieve top-k most similar voices from a database.
        
        Args:
            query_audio_path: Path to query voice audio
            database_audio_paths: List of database voice audio paths
            instruction_text: Instruction for encoding
            top_k: Number of top results to return
            
        Returns:
            results: List of tuples (audio_path, similarity_score)
        """
        # Encode query
        logging.info("Encoding query voice...")
        query_emb = self.encode_voice(instruction_text, query_audio_path, normalize=True)
        
        # Encode database
        logging.info(f"Encoding {len(database_audio_paths)} database voices...")
        db_texts = [instruction_text] * len(database_audio_paths)
        db_embeddings = self.encode_batch(db_texts, database_audio_paths, normalize=True)
        
        # Compute similarities
        similarities = []
        for i, db_emb in enumerate(db_embeddings):
            sim = self.compute_similarity(query_emb, db_emb)
            similarities.append((database_audio_paths[i], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 Embedding Inference")
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to CosyVoice3 model directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained checkpoint (e.g., checkpoint_epoch0_step160.pt)')
    parser.add_argument('--query_audio', type=str, required=True,
                       help='Path to query voice audio')
    parser.add_argument('--database_dir', type=str, required=True,
                       help='Directory containing database voice audios')
    parser.add_argument('--instruction', type=str, 
                       default='Retrieve semantically similar voice',
                       help='Instruction text for encoding')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top results to return')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--output', type=str, default='retrieval_results.txt',
                       help='Output file for results')
    parser.add_argument('--qwen_path', type=str, default=None,
                       help='Path to Qwen model (optional override)')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = CosyVoice3EmbeddingInference(
        args.model_dir, 
        args.device,
        qwen_path=args.qwen_path,
        checkpoint_path=args.checkpoint
    )
    
    # Get database files
    database_dir = Path(args.database_dir)
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    database_files = []
    for ext in audio_extensions:
        database_files.extend(database_dir.glob(ext))
    
    if len(database_files) == 0:
        logging.error(f"No audio files found in {args.database_dir}")
        return
    
    logging.info(f"Found {len(database_files)} audio files in database")
    
    # Retrieve similar voices
    results = inference.retrieve_similar_voices(
        args.query_audio,
        [str(f) for f in database_files],
        args.instruction,
        args.top_k
    )
    
    # Print and save results
    print(f"\nTop {args.top_k} similar voices:")
    print("=" * 80)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"Query: {args.query_audio}\n")
        f.write(f"Instruction: {args.instruction}\n")
        if args.checkpoint:
            f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write("\n")
        f.write(f"Top {args.top_k} Results:\n")
        f.write("=" * 80 + "\n")
        
        for i, (audio_path, similarity) in enumerate(results, 1):
            result_str = f"{i}. {Path(audio_path).name:<50} Similarity: {similarity:.4f}"
            print(result_str)
            f.write(result_str + "\n")
    
    print("=" * 80)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
