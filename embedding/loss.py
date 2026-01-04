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


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with in-batch negatives and optional hard negatives.
    
    This loss encourages similar voice samples to have close embeddings
    while pushing dissimilar samples apart.
    
    Loss = -log( exp(sim(query, positive) / tau) / 
                 (exp(sim(query, positive) / tau) + sum(exp(sim(query, negative_i) / tau))) )
    """
    
    def __init__(self, temperature: float = 0.07, use_hard_negatives: bool = True):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            use_hard_negatives: Whether to use hard negatives from data
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
    
    def forward(self, 
                query_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None,
                negative_counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss.
        
        Args:
            query_embeddings: Query embeddings [batch, dim]
            positive_embeddings: Positive sample embeddings [batch, dim]
            negative_embeddings: Hard negative embeddings [num_total_negatives, dim] (optional)
            negative_counts: Number of negatives per sample [batch] (optional)
            
        Returns:
            Dictionary containing:
                - 'total_loss': Total loss
                - 'softmax_loss': Loss from in-batch negatives
                - 'hard_negative_loss': Loss from hard negatives (if provided)
        """
        batch_size = query_embeddings.size(0)
        device = query_embeddings.device
        
        # Normalize embeddings (already normalized in model, but ensure it)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        
        # Compute similarity between query and positive
        # [batch, batch] - diagonal contains query-positive similarities
        pos_sim = torch.matmul(query_embeddings, positive_embeddings.T) / self.temperature
        
        # Create labels (diagonal elements are positives)
        labels = torch.arange(batch_size, device=device)
        
        # In-batch negatives: all other samples in the batch
        logits = pos_sim
        
        # Compute softmax loss (in-batch negatives only)
        softmax_loss = F.cross_entropy(logits, labels)
        
        # Initialize hard negative loss
        hard_negative_loss = torch.tensor(0.0, device=device)
        
        # Add hard negatives if provided
        if negative_embeddings is not None and self.use_hard_negatives and negative_embeddings.size(0) > 0:
            # negative_embeddings: [num_total_negatives, dim]
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
            
            if negative_counts is not None:
                # Variable number of negatives per sample
                # Split negatives by counts and compute similarities
                neg_sims = []
                start_idx = 0
                
                for i in range(batch_size):
                    count = negative_counts[i].item()
                    if count > 0:
                        # Get negatives for this sample
                        sample_negs = negative_embeddings[start_idx:start_idx + count]  # [count, dim]
                        # Compute similarity with query
                        sample_neg_sim = torch.matmul(
                            query_embeddings[i:i+1], sample_negs.T
                        ) / self.temperature  # [1, count]
                        neg_sims.append(sample_neg_sim.squeeze(0))  # [count]
                        start_idx += count
                    else:
                        # No negatives for this sample, add empty tensor
                        neg_sims.append(torch.tensor([], device=device))
                
                # Pad negative similarities to same length
                max_neg_count = max(len(ns) for ns in neg_sims) if neg_sims else 0
                if max_neg_count > 0:
                    padded_neg_sims = torch.full(
                        (batch_size, max_neg_count), 
                        float('-inf'), 
                        device=device
                    )
                    for i, ns in enumerate(neg_sims):
                        if len(ns) > 0:
                            padded_neg_sims[i, :len(ns)] = ns
                    
                    # Concatenate with in-batch similarities for combined loss
                    combined_logits = torch.cat([logits, padded_neg_sims], dim=1)
                    
                    # Compute combined loss (in-batch + hard negatives)
                    total_loss = F.cross_entropy(combined_logits, labels)
                    
                    # Hard negative loss = total - softmax
                    hard_negative_loss = total_loss - softmax_loss
                else:
                    total_loss = softmax_loss
            else:
                # All negatives are for all samples (old format)
                # Compute similarity with hard negatives
                # [batch, num_total_negatives]
                neg_sim = torch.matmul(query_embeddings, negative_embeddings.T) / self.temperature
                
                # Concatenate with in-batch similarities
                combined_logits = torch.cat([logits, neg_sim], dim=1)
                
                # Compute combined loss
                total_loss = F.cross_entropy(combined_logits, labels)
                
                # Hard negative loss = total - softmax
                hard_negative_loss = total_loss - softmax_loss
        else:
            total_loss = softmax_loss
        
        return {
            'total_loss': total_loss,
            'softmax_loss': softmax_loss.item(),
            'hard_negative_loss': hard_negative_loss.item() if isinstance(hard_negative_loss, torch.Tensor) else hard_negative_loss
        }


class MultiTaskContrastiveLoss(nn.Module):
    """
    Multi-task contrastive loss that supports different task types.
    
    Supports task-specific weighting and logging of per-task losses.
    """
    
    def __init__(self, 
                 temperature: float = 0.07, 
                 use_hard_negatives: bool = True,
                 task_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            use_hard_negatives: Whether to use hard negatives from data
            task_weights: Optional dictionary of task-specific weights
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        
        # Default task weights
        self.task_weights = task_weights or {
            'semantic_similarity': 1.0,
            'speaker_verification': 1.0,
            'emotion_recognition': 1.0,
            'prosody_matching': 1.0,
        }
        
        # Base InfoNCE loss
        self.infonce = InfoNCELoss(temperature=temperature, use_hard_negatives=use_hard_negatives)
    
    def forward(self, model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task contrastive loss.
        
        Args:
            model: The embedding model
            batch: Dictionary containing:
                - query_text_token: [batch, text_len]
                - query_text_token_len: [batch]
                - query_speech_token: [batch, speech_len]
                - query_speech_token_len: [batch]
                - positive_text_token: [batch, text_len]
                - positive_text_token_len: [batch]
                - positive_speech_token: [batch, speech_len]
                - positive_speech_token_len: [batch]
                - negative_text_token: [num_negatives, text_len] (optional)
                - negative_text_token_len: [num_negatives] (optional)
                - negative_speech_token: [num_negatives, speech_len] (optional)
                - negative_speech_token_len: [num_negatives] (optional)
                - negative_counts: [batch] (optional)
                - task_type: [batch] (optional)
        
        Returns:
            Dictionary containing:
                - 'total_loss': Total weighted loss
                - 'softmax_loss': Average softmax loss
                - 'hard_negative_loss': Average hard negative loss
                - 'task_losses': Dictionary of per-task losses
        """
        # Extract query embeddings
        query_output = model(
            batch['query_text_token'],
            batch['query_text_token_len'],
            batch['query_speech_token'],
            batch['query_speech_token_len']
        )
        query_embeddings = query_output['embedding']
        
        # Extract positive embeddings
        positive_output = model(
            batch['positive_text_token'],
            batch['positive_text_token_len'],
            batch['positive_speech_token'],
            batch['positive_speech_token_len']
        )
        positive_embeddings = positive_output['embedding']
        
        # Extract negative embeddings if provided
        negative_embeddings = None
        if 'negative_speech_token' in batch:
            negative_output = model(
                batch['negative_text_token'],
                batch['negative_text_token_len'],
                batch['negative_speech_token'],
                batch['negative_speech_token_len']
            )
            negative_embeddings = negative_output['embedding']
        
        # Get negative counts
        negative_counts = batch.get('negative_counts', None)
        
        # Compute InfoNCE loss
        loss_dict = self.infonce(
            query_embeddings,
            positive_embeddings,
            negative_embeddings,
            negative_counts
        )
        
        # If task types are provided, compute per-task losses
        task_losses = {}
        if 'task_type' in batch:
            # Get unique task types in this batch
            # For now, assume all samples are same task (can extend later)
            # task_losses would be computed by grouping by task_type
            pass
        
        loss_dict['task_losses'] = task_losses
        
        # Free memory
        del query_output, positive_output
        if negative_embeddings is not None:
            del negative_output
        
        return loss_dict


class TripletLoss(nn.Module):
    """
    Triplet loss as an alternative to InfoNCE.
    
    Loss = max(0, margin + sim(query, negative) - sim(query, positive))
    """
    
    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self,
                query_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None,
                negative_counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute triplet loss.
        
        Args:
            query_embeddings: Query embeddings [batch, dim]
            positive_embeddings: Positive embeddings [batch, dim]
            negative_embeddings: Negative embeddings [num_total_negatives, dim] (optional)
            negative_counts: Number of negatives per sample [batch] (optional)
            
        Returns:
            Dictionary containing loss values
        """
        if negative_embeddings is None:
            # Use in-batch negatives only
            # For each query, all other positives are negatives
            batch_size = query_embeddings.size(0)
            
            # Normalize
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
            
            # Compute all pairwise similarities
            all_sims = torch.matmul(query_embeddings, positive_embeddings.T)  # [batch, batch]
            
            # Positive similarities (diagonal)
            pos_sim = all_sims.diag()  # [batch]
            
            # Negative similarities (off-diagonal, take hardest)
            mask = torch.eye(batch_size, device=query_embeddings.device).bool()
            all_sims.masked_fill_(mask, float('-inf'))
            neg_sim = all_sims.max(dim=1)[0]  # [batch] - hardest in-batch negative
            
            # Triplet loss
            loss = F.relu(self.margin + neg_sim - pos_sim).mean()
            
            return {
                'total_loss': loss,
                'softmax_loss': loss.item(),
                'hard_negative_loss': 0.0,
                'task_losses': {}
            }
        
        # Normalize
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
        
        # Compute positive similarities
        pos_sim = (query_embeddings * positive_embeddings).sum(dim=-1)  # [batch]
        
        batch_size = query_embeddings.size(0)
        
        if negative_counts is not None:
            # Variable number of negatives per sample
            losses = []
            start_idx = 0
            
            for i in range(batch_size):
                count = negative_counts[i].item()
                if count > 0:
                    # Get negatives for this sample
                    sample_negs = negative_embeddings[start_idx:start_idx + count]  # [count, dim]
                    # Compute similarities
                    neg_sims = torch.matmul(query_embeddings[i:i+1], sample_negs.T).squeeze(0)  # [count]
                    # Use hardest negative
                    hardest_neg_sim = neg_sims.max()
                    # Triplet loss for this sample
                    sample_loss = F.relu(self.margin + hardest_neg_sim - pos_sim[i])
                    losses.append(sample_loss)
                    start_idx += count
                else:
                    # No hard negatives, skip or use in-batch negative
                    losses.append(torch.tensor(0.0, device=query_embeddings.device))
            
            loss = torch.stack(losses).mean()
        else:
            # All negatives are for all samples (old format)
            # negative_embeddings: [num_negatives, dim]
            neg_sim = torch.matmul(query_embeddings, negative_embeddings.T)  # [batch, num_negatives]
            # Use hardest negative for each query
            neg_sim = neg_sim.max(dim=-1)[0]  # [batch]
            
            # Triplet loss
            loss = F.relu(self.margin + neg_sim - pos_sim).mean()
        
        return {
            'total_loss': loss,
            'softmax_loss': 0.0,
            'hard_negative_loss': loss.item(),
            'task_losses': {}
        }
