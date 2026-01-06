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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging


class CosyVoice3Embedding(nn.Module):
    """
    Voice embedding model based on CosyVoice3 LLM.
    
    This model:
    - Reuses the speech tokenizer from CosyVoice3
    - Initializes from CosyVoice3 LLM (llm.rl.pt)
    - Outputs 896-dimensional embeddings using the last token representation
    - Does not include flow/hifigan modules (not a TTS model)
    
    Input: Free-form combination of text and audio tokens
    Output: 896-dimensional embedding vector
    """
    
    def __init__(self, llm_config, speech_tokenizer_path=None):
        """
        Args:
            llm_config: Configuration for the LLM model (or the LLM model itself)
            speech_tokenizer_path: Path to speech tokenizer ONNX model
        """
        super().__init__()
        
        # Initialize LLM from CosyVoice3
        self.llm = llm_config
        self.embedding_dim = 896  # CosyVoice3 LLM hidden size
        
        # Projection layer to ensure output is 896-d
        # (in case we want to add layer norm or other transformations)
        self.embedding_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Layer norm for stable embeddings
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Speech tokenizer path (will be loaded separately)
        self.speech_tokenizer_path = speech_tokenizer_path
        
        logging.info(f"CosyVoice3Embedding initialized with {self.embedding_dim}-d embeddings")
    
    def load_llm(self, llm_checkpoint_path: str, strict: bool = False):
        """
        Load LLM weights from CosyVoice3 checkpoint.
        
        Args:
            llm_checkpoint_path: Path to llm.rl.pt
            strict: Whether to strictly match state dict keys
        """
        logging.info(f"Loading LLM from {llm_checkpoint_path}")
        
        # PyTorch 2.6+ requires weights_only=False for numpy types in checkpoints
        checkpoint = torch.load(llm_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Filter out non-LLM keys (epoch, step, optimizer, etc.)
        if isinstance(checkpoint, dict):
            state_dict = {k: v for k, v in checkpoint.items() 
                         if k not in ['epoch', 'step', 'optimizer', 'scheduler', 'save_time']}
        else:
            state_dict = checkpoint
        
        # Load into LLM
        missing_keys, unexpected_keys = self.llm.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logging.warning(f"Missing keys when loading LLM: {missing_keys[:5]}...")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading LLM: {unexpected_keys[:5]}...")
        
        logging.info("LLM loaded successfully")
        
        # Free memory after loading
        del checkpoint
        torch.cuda.empty_cache()
   
    def encode(self, 
               text_token: torch.Tensor,
               text_token_len: torch.Tensor,
               speech_token: Optional[torch.Tensor] = None,
               speech_token_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convenience method to directly get embeddings.
        
        Returns:
            embeddings: [batch, 896] normalized embedding vectors
        """
        with torch.no_grad():
            output = self.forward(text_token, text_token_len, speech_token, speech_token_len)
        return output['embedding']

    def forward(self, 
                text_token: torch.Tensor,
                text_token_len: torch.Tensor,
                speech_token: Optional[torch.Tensor] = None,
                speech_token_len: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate embeddings.
        
        Args:
            text_token: Text tokens [batch, text_len]
            text_token_len: Text token lengths [batch]
            speech_token: Speech tokens [batch, speech_len] (optional)
            speech_token_len: Speech token lengths [batch] (optional)
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary containing:
                - 'embedding': [batch, 896] embedding vectors
                - 'hidden_states': [batch, seq_len, 896] (if return_hidden_states=True)
        """
        batch_size = text_token.size(0)
        device = text_token.device
        
        # Concatenate text and speech tokens if speech is provided
        if speech_token is not None and speech_token.numel() > 0:
            # Format: [text_instruction] [speech_tokens]
            input_token = torch.cat([text_token, speech_token], dim=1)
            total_len = text_token_len + speech_token_len
        else:
            input_token = text_token
            total_len = text_token_len
        
        # Access the Qwen2Model: self.llm.llm.model
        qwen_model = self.llm.llm.model
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        max_len = input_token.shape[1]
        attention_mask = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
        for i in range(batch_size):
            attention_mask[i, total_len[i]:] = 0
        
        # Call Qwen2Model directly with memory optimization
        # Only output hidden states if needed (saves memory)
        outputs = qwen_model(
            input_ids=input_token,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=False,  # Disable KV cache to save memory
            output_attentions=False,  # Don't compute attention weights
            output_hidden_states=True,  # We need this for embeddings
            return_dict=True,
        )
        
        # Get last layer hidden states
        # Shape: [batch, seq_len, 896]
        hidden_states = outputs.hidden_states[-1]
        
        # Extract last token representation for each sequence
        # Use advanced indexing to avoid loop (more memory efficient)
        batch_indices = torch.arange(batch_size, device=device)
        last_token_indices = (total_len - 1).clamp(min=0)  # Prevent negative indices
        embeddings = hidden_states[batch_indices, last_token_indices, :]  # [batch, 896]
        
        # Project and normalize
        embeddings = self.embedding_proj(embeddings)  # [batch, 896]
        embeddings = self.layer_norm(embeddings)  # [batch, 896]
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # [batch, 896]
        
        result = {'embedding': embeddings}
        
        if return_hidden_states:
            result['hidden_states'] = hidden_states
        else:
            # Explicitly delete hidden_states if not needed to free memory
            del hidden_states
            del outputs
        
        return result
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory savings."""
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled for LLM")
        elif hasattr(self.llm, 'llm') and hasattr(self.llm.llm, 'gradient_checkpointing_enable'):
            self.llm.llm.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled for LLM")
        else:
            logging.warning("Gradient checkpointing not available for this model")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.llm, 'gradient_checkpointing_disable'):
            self.llm.gradient_checkpointing_disable()
        elif hasattr(self.llm, 'llm') and hasattr(self.llm.llm, 'gradient_checkpointing_disable'):
            self.llm.llm.gradient_checkpointing_disable()
