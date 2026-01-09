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
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
import logging
from transformers import Qwen2_5OmniThinkerForConditionalGeneration


class OmniEmbeddingModel(nn.Module):
    """
    Voice embedding model based on Qwen2.5-Omni Thinker.

    This model:
    - Uses Qwen2.5-Omni Thinker as the base encoder
    - Supports both audio and text inputs in a conversational format
    - Outputs embeddings using either mean pooling or last token pooling
    - Embedding dimension: 2048 for Qwen2.5-Omni-3B (hidden_size from text_config)
    - Supports LoRA fine-tuning for efficient training

    Input format:
        System prompt: "Retrieve voice that is semantically similar."
        User prompt: <audio> (from conversation format)

    Output: Normalized embedding vector
    """

    def __init__(self,
                 model_path: str,
                 pooling_mode: Literal["mean", "last"] = "mean",
                 embedding_dim: int = 256,
                 freeze_backbone: bool = False,
                 use_lora: bool = False,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Optional[list] = None):
        """
        Args:
            model_path: Path to Qwen2.5-Omni model directory (will load thinker subfolder)
            pooling_mode: Pooling strategy - "mean" for mean pooling, "last" for last token
            embedding_dim: Output embedding dimension
            freeze_backbone: Whether to freeze the backbone model (ignored if use_lora=True)
            use_lora: Whether to use LoRA fine-tuning
            lora_rank: LoRA rank (r parameter)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: LoRA dropout rate
            lora_target_modules: Target modules for LoRA (default: query, key, value projections)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.pooling_mode = pooling_mode
        self.use_lora = use_lora

        # Load Qwen2.5-Omni Thinker model
        logging.info(f"Loading Qwen2.5-Omni Thinker from {model_path}")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=False
        )

        # Drop visual components to save memory (we don't use visual data)
        logging.info("Dropping visual components to save memory...")
        if hasattr(self.model, 'visual'):
            del self.model.visual
            logging.info("✓ Visual tower deleted")

        # Freeze ALL base model parameters first (audio_tower + LLM)
        logging.info("Freezing all base model parameters (audio + LLM)...")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        logging.info("✓ All base model parameters frozen")

        # Apply LoRA ONLY to LLM layers (model.model.*) if requested
        if use_lora:
            logging.info("Applying LoRA ONLY to LLM layers (model.model.*)...")
            self._apply_lora(
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules
            )

        # Get embedding dimension from the model's text config
        model_hidden_size = self.model.config.text_config.hidden_size
        logging.info(f"Model hidden size: {model_hidden_size}")

        # Projection layer if embedding_dim differs from hidden_size
        if model_hidden_size != self.embedding_dim:
            self.embedding_proj = nn.Linear(model_hidden_size, self.embedding_dim)
            logging.info(f"Added projection layer: {model_hidden_size} -> {self.embedding_dim}")
        else:
            self.embedding_proj = nn.Identity()

        # Layer norm for stable embeddings
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        logging.info(f"Qwen2.5-Omni Thinker Embedding initialized:")
        logging.info(f"  - Model path: {model_path}")
        logging.info(f"  - Model hidden size: {model_hidden_size}")
        logging.info(f"  - Output embedding dim: {self.embedding_dim}")
        logging.info(f"  - Pooling mode: {pooling_mode}")
        logging.info(f"  - Using LoRA: {use_lora}")
        if use_lora:
            logging.info(f"  - LoRA rank: {lora_rank}")
            logging.info(f"  - LoRA alpha: {lora_alpha}")
            logging.info(f"  - LoRA dropout: {lora_dropout}")
            logging.info(f"  - LoRA applied to: LLM layers only (model.model.*)")
        logging.info(f"  - Visual components: Dropped")
        logging.info(f"  - All base parameters: Frozen")

    def _apply_lora(self, lora_rank: int, lora_alpha: int, lora_dropout: float,
                    lora_target_modules: Optional[list] = None):
        """Apply LoRA to the LLM layers only (model.model.*), excluding audio_tower and visual."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA training. "
                "Install it with: pip install peft"
            )

        # We need to be very specific about which modules to target
        # Instead of using generic patterns like "q_proj", we'll use full module paths
        # that only match LLM layers

        logging.info("Building explicit target module list for LLM layers ONLY...")

        # Scan the model to find all LLM layer modules
        llm_target_modules = []

        # Patterns we want to match (within LLM layers only)
        target_patterns = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

        for name, module in self.model.named_modules():
            # Only include modules that are in model.model.layers.* (LLM)
            # Explicitly exclude audio_tower and visual
            # if 'model.model.layers' in name and not any(x in name for x in ['audio_tower', 'visual']):
            if not any(x in name for x in ['audio_tower', 'visual']):
                # Check if this module matches our target patterns
                module_name = name.split('.')[-1]  # Get the last part (e.g., "q_proj")
                if module_name in target_patterns:
                    llm_target_modules.append(name)

        if not llm_target_modules:
            raise RuntimeError(
                "No LLM target modules found! This likely means the model structure "
                "has changed. Expected to find modules like 'model.model.layers.*.q_proj'"
            )

        logging.info(f"Found {len(llm_target_modules)} LLM modules to apply LoRA to")
        logging.info(f"Sample LLM target modules (first 5):")
        for module_name in llm_target_modules[:5]:
            logging.info(f"  - {module_name}")

        # Create LoRA config with explicit module paths
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=llm_target_modules,  # Use full module paths, not patterns
            bias="none",
            inference_mode=False,
        )

        # Apply LoRA to model
        logging.info("Applying LoRA with explicit module targets...")
        self.model = get_peft_model(self.model, lora_config)

        # Verify that LoRA was only applied to LLM layers
        lora_in_llm = 0
        lora_in_audio = 0
        lora_in_visual = 0
        lora_params = []

        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append(name)
                if 'model.model.layers' in name:
                    lora_in_llm += 1
                elif 'audio_tower' in name:
                    lora_in_audio += 1
                elif 'visual' in name:
                    lora_in_visual += 1

        logging.info(f"LoRA parameters after application:")
        logging.info(f"  - LoRA in LLM (model.model.layers.*): {lora_in_llm}")
        logging.info(f"  - LoRA in audio_tower: {lora_in_audio}")
        logging.info(f"  - LoRA in visual: {lora_in_visual}")

        if lora_in_audio > 0:
            logging.error(f"ERROR: LoRA was incorrectly applied to audio_tower!")
            logging.error(f"Audio LoRA parameters:")
            for name in lora_params:
                if 'audio_tower' in name:
                    logging.error(f"  - {name}")
            raise RuntimeError(
                f"LoRA was applied to {lora_in_audio} audio_tower parameters! "
                "This should not happen. Check model structure."
            )

        if lora_in_visual > 0:
            logging.error(f"ERROR: LoRA was incorrectly applied to visual!")
            logging.error(f"Visual LoRA parameters:")
            for name in lora_params:
                if 'visual' in name:
                    logging.error(f"  - {name}")
            raise RuntimeError(
                f"LoRA was applied to {lora_in_visual} visual parameters! "
                "This should not happen. Visual should have been deleted."
            )

        if lora_in_llm == 0:
            logging.error(f"ERROR: No LoRA parameters found in LLM!")
            raise RuntimeError(
                "No LoRA parameters were added to the LLM. Check model structure and target modules."
            )

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logging.info(f"✓ LoRA applied successfully to LLM ONLY:")
        logging.info(f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logging.info(f"  - Total parameters: {total_params:,}")
        logging.info(f"  - Memory reduction: {100 * (1 - trainable_params / total_params):.2f}%")

        # Log sample trainable parameter names to verify they're in the LLM
        logging.info("Sample trainable LoRA parameters (first 10):")
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                logging.info(f"  - {name}: {param.shape}")
                count += 1
                if count >= 10:
                    break

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                input_features: Optional[torch.Tensor] = None,
                feature_attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate embeddings.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            input_features: Audio mel-spectrogram features. Can be:
                           - [batch, audio_len] (1D flattened, will be reshaped)
                           - [batch, audio_dim, audio_seq] (2D, Qwen expected format)
            feature_attention_mask: Audio feature attention mask [batch, audio_seq] (optional)
            return_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing:
                - 'embedding': [batch, embedding_dim] embedding vectors
                - 'hidden_states': [batch, seq_len, hidden_dim] (if return_hidden_states=True)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Prepare model inputs
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True,
            'return_dict': True,
            'use_cache': False,  # Disable KV cache to save memory
        }

        # Add audio features if provided
        if input_features is not None:
            # Qwen2.5-Omni expects input_features with shape [batch, audio_dim, audio_seq]
            # where audio_dim is typically 128 (mel bins)
            audio_dim = 128  # Qwen2.5-Omni uses 128 mel bins

            if input_features.dim() == 2:
                # Input is [batch, audio_len] - 1D flattened mel-spectrogram
                # Reshape to [batch, audio_dim, audio_seq]
                audio_len = input_features.size(1)
                audio_seq = audio_len // audio_dim

                # Truncate if not perfectly divisible
                if audio_len % audio_dim != 0:
                    truncated_len = audio_seq * audio_dim
                    input_features = input_features[:, :truncated_len]

                # Reshape to [batch, audio_seq, audio_dim] then permute to [batch, audio_dim, audio_seq]
                input_features = input_features.view(batch_size, audio_seq, audio_dim)
                input_features = input_features.permute(0, 2, 1)  # [batch, audio_dim, audio_seq]

            elif input_features.dim() == 3:
                # Check if already in correct format [batch, audio_dim, audio_seq]
                # or if it's [batch, audio_seq, audio_dim] and needs permutation
                if input_features.size(1) != audio_dim and input_features.size(2) == audio_dim:
                    # Input is [batch, audio_seq, audio_dim], permute to [batch, audio_dim, audio_seq]
                    input_features = input_features.permute(0, 2, 1)
                # else: assume already [batch, audio_dim, audio_seq]
            else:
                raise ValueError(f"Unexpected input_features shape: {input_features.shape}. "
                               f"Expected 2D [batch, audio_len] or 3D [batch, audio_dim, audio_seq]")

            model_inputs['input_features'] = input_features

            # Create feature_attention_mask if not provided
            # Shape should be [batch, audio_seq]
            audio_seq_len = input_features.size(2)  # After permutation, audio_seq is dim 2

            if feature_attention_mask is None:
                # All audio features are valid (no padding)
                feature_attention_mask = torch.ones(
                    (batch_size, audio_seq_len),
                    dtype=torch.long,
                    device=device
                )

            model_inputs['feature_attention_mask'] = feature_attention_mask

        # Forward through Qwen2.5-Omni Thinker
        outputs = self.model(**model_inputs)

        # Get last layer hidden states
        # Shape: [batch, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]

        # Apply pooling based on mode
        if self.pooling_mode == "mean":
            # Mean pooling over all tokens (excluding padding)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

            # Sum embeddings and divide by number of non-padding tokens
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # [batch, hidden_dim]

        elif self.pooling_mode == "last":
            # Last token pooling (use last non-padding token)
            sequence_lengths = attention_mask.sum(dim=1) - 1
            sequence_lengths = sequence_lengths.clamp(min=0)

            batch_indices = torch.arange(batch_size, device=device)
            embeddings = hidden_states[batch_indices, sequence_lengths, :]  # [batch, hidden_dim]
        else:
            raise ValueError(f"Invalid pooling mode: {self.pooling_mode}. Must be 'mean' or 'last'")

        # Project to target dimension
        embeddings = self.embedding_proj(embeddings)  # [batch, embedding_dim]
        embeddings = self.layer_norm(embeddings)

        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # [batch, embedding_dim]

        result = {'embedding': embeddings}

        if return_hidden_states:
            result['hidden_states'] = hidden_states

        return result

    def encode(self,
               input_ids: torch.Tensor,
               attention_mask: torch.Tensor,
               input_features: Optional[torch.Tensor] = None,
               feature_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method to directly get embeddings.

        Returns:
            embeddings: [batch, embedding_dim] normalized embedding vectors
        """
        with torch.no_grad():
            output = self.forward(
                input_ids,
                attention_mask,
                input_features,
                feature_attention_mask
            )
        return output['embedding']

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory savings."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled for Qwen2.5-Omni Thinker")
        else:
            logging.warning("Gradient checkpointing not available for this model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def forward_from_tensors(self,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor,
                             audio_values: Optional[torch.Tensor] = None,
                             return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass using pre-tokenized tensors from collate function.

        This method provides a convenient interface that matches the output format
        of the training data collate function.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            audio_values: Audio features [batch, audio_seq, audio_dim] (optional)
                         Maps to input_features in the base forward method
            return_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing:
                - 'embedding': [batch, embedding_dim] embedding vectors
                - 'hidden_states': [batch, seq_len, hidden_dim] (if return_hidden_states=True)
        """
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=audio_values,
            feature_attention_mask=None,
            return_hidden_states=return_hidden_states
        )
