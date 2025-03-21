"""
Attention analysis module for the Language Model Evaluation Framework.

This module provides functionality for analyzing attention patterns in language models.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np

from ..core.model_loader import ModelWrapper


class AttentionAnalyzer:
    """Analyzer for attention patterns in language models."""

    def __init__(
        self,
        model: ModelWrapper,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize the attention analyzer.
        
        Args:
            model: The model to analyze
            layer_indices: Indices of layers to analyze (if None, analyze all layers)
            head_indices: Indices of attention heads to analyze (if None, analyze all heads)
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.layer_indices = layer_indices
        self.head_indices = head_indices
        self.kwargs = kwargs

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze attention patterns for a text input.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Get attention weights from the model
        attention_weights = self.model.get_attention(text)
        
        # Get tokenized input
        tokens = self.model.tokenizer.tokenize(text)
        if not tokens:
            # Some tokenizers might not return the expected output, so we'll create a fallback
            tokens = [t for t in text.split()]
        
        # Create a safe copy of attention weights for analysis
        processed_weights = []
        for layer in attention_weights:
            if isinstance(layer, torch.Tensor):
                # Convert to numpy for easier processing
                layer_np = layer.detach().cpu().numpy()
                processed_weights.append(layer_np)
            else:
                processed_weights.append(layer)
        
        # Filter layers and heads if specified
        if self.layer_indices is not None:
            processed_weights = [processed_weights[i] for i in self.layer_indices if i < len(processed_weights)]
        
        if self.head_indices is not None and processed_weights:
            processed_weights = [layer[:, self.head_indices, :, :] for layer in processed_weights]
            
        # Calculate attention statistics (safely)
        statistics = self._calculate_statistics(processed_weights)
        
        # Identify attention patterns (safely)
        patterns = self._identify_patterns(processed_weights, tokens)
        
        # Format and return results
        return {
            "attention_weights": "Tensor data (too large to serialize)",  # Don't return raw tensors
            "tokens": tokens,
            "statistics": statistics,
            "patterns": patterns
        }

    def _calculate_statistics(self, attention_weights) -> Dict[str, Any]:
        """
        Calculate statistical measures of attention.
        
        Args:
            attention_weights: Attention weight matrices
            
        Returns:
            Dictionary of statistical measures
        """
        # Return default statistics if no weights
        if not attention_weights:
            return {
                "mean_attention": 0.0,
                "mean_entropy": 0.0,
                "mean_sparsity": 0.0
            }
        
        # Convert to numpy if needed
        weights_np = []
        for layer in attention_weights:
            if isinstance(layer, torch.Tensor):
                weights_np.append(layer.detach().cpu().numpy())
            else:
                weights_np.append(layer)
        
        # Calculate mean attention across all layers and heads
        layer_means = []
        for layer in weights_np:
            # Handle different tensor dimensions
            if len(layer.shape) == 4:  # [batch, head, seq, seq]
                layer_means.append(np.mean(layer))
            elif len(layer.shape) == 3:  # [head, seq, seq]
                layer_means.append(np.mean(layer))
            elif len(layer.shape) == 2:  # [seq, seq]
                layer_means.append(np.mean(layer))
            else:
                # Skip layers with unexpected dimensions
                continue
        
        mean_attn = np.mean(layer_means) if layer_means else 0.0
        
        # Calculate attention entropy (measure of focus)
        entropies = []
        for layer in weights_np:
            try:
                if len(layer.shape) == 4:  # [batch, head, seq, seq]
                    for batch_idx in range(layer.shape[0]):
                        for head_idx in range(layer.shape[1]):
                            for seq_idx in range(layer.shape[2]):
                                attn_dist = layer[batch_idx, head_idx, seq_idx]
                                # Normalize distribution
                                attn_sum = np.sum(attn_dist)
                                if attn_sum > 0:
                                    attn_dist = attn_dist / attn_sum
                                    # Calculate entropy
                                    entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-10))
                                    entropies.append(entropy)
                elif len(layer.shape) == 3:  # [head, seq, seq]
                    for head_idx in range(layer.shape[0]):
                        for seq_idx in range(layer.shape[1]):
                            attn_dist = layer[head_idx, seq_idx]
                            # Normalize distribution
                            attn_sum = np.sum(attn_dist)
                            if attn_sum > 0:
                                attn_dist = attn_dist / attn_sum
                                # Calculate entropy
                                entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-10))
                                entropies.append(entropy)
                elif len(layer.shape) == 2:  # [seq, seq]
                    for seq_idx in range(layer.shape[0]):
                        attn_dist = layer[seq_idx]
                        # Normalize distribution
                        attn_sum = np.sum(attn_dist)
                        if attn_sum > 0:
                            attn_dist = attn_dist / attn_sum
                            # Calculate entropy
                            entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-10))
                            entropies.append(entropy)
            except Exception as e:
                # Skip this layer if there's an error
                continue
        
        mean_entropy = np.mean(entropies) if entropies else 0.0
        
        # Calculate sparsity (percentage of near-zero attention weights)
        threshold = 0.01  # Consider weights below 1% as near-zero
        sparsities = []
        for layer in weights_np:
            try:
                sparsity = np.mean(layer < threshold)
                sparsities.append(sparsity)
            except Exception as e:
                # Skip this layer if there's an error
                continue
        
        mean_sparsity = np.mean(sparsities) if sparsities else 0.0
        
        return {
            "mean_attention": float(mean_attn),
            "mean_entropy": float(mean_entropy),
            "mean_sparsity": float(mean_sparsity)
        }

    def _identify_patterns(self, attention_weights, tokens) -> Dict[str, Any]:
        """
        Identify common attention patterns.
        
        Args:
            attention_weights: Attention weight matrices
            tokens: List of tokens
            
        Returns:
            Dictionary of identified patterns
        """
        # Return empty patterns if no weights
        if not attention_weights:
            return {
                "diagonal_attention": [],
                "previous_token_attention": [],
                "special_token_attention": []
            }
        
        # Convert to numpy if needed
        weights_np = []
        for layer in attention_weights:
            if isinstance(layer, torch.Tensor):
                weights_np.append(layer.detach().cpu().numpy())
            else:
                weights_np.append(layer)
                
        patterns = {}
        
        # Identify diagonal attention (token attending to itself)
        diagonal_scores = []
        for layer_idx, layer in enumerate(weights_np):
            try:
                # Handle different tensor dimensions
                if len(layer.shape) == 4:  # [batch, head, seq, seq]
                    for batch_idx in range(layer.shape[0]):
                        for head_idx in range(layer.shape[1]):
                            attn_matrix = layer[batch_idx, head_idx]
                            if min(attn_matrix.shape) > 1:
                                diag_mean = np.mean(np.diag(attn_matrix))
                                diagonal_scores.append({
                                    "layer": layer_idx,
                                    "head": head_idx,
                                    "score": float(diag_mean)
                                })
                elif len(layer.shape) == 3:  # [head, seq, seq]
                    for head_idx in range(layer.shape[0]):
                        attn_matrix = layer[head_idx]
                        if min(attn_matrix.shape) > 1:
                            diag_mean = np.mean(np.diag(attn_matrix))
                            diagonal_scores.append({
                                "layer": layer_idx,
                                "head": head_idx,
                                "score": float(diag_mean)
                            })
                elif len(layer.shape) == 2:  # [seq, seq]
                    attn_matrix = layer
                    if min(attn_matrix.shape) > 1:
                        diag_mean = np.mean(np.diag(attn_matrix))
                        diagonal_scores.append({
                            "layer": layer_idx,
                            "head": 0,
                            "score": float(diag_mean)
                        })
            except Exception as e:
                # Skip this layer if there's an error
                continue
        
        # Sort by score and take top 3
        diagonal_scores.sort(key=lambda x: x["score"], reverse=True)
        patterns["diagonal_attention"] = diagonal_scores[:3] if diagonal_scores else []
        
        # Identify previous token attention
        prev_token_scores = []
        for layer_idx, layer in enumerate(weights_np):
            try:
                # Handle different tensor dimensions
                if len(layer.shape) == 4:  # [batch, head, seq, seq]
                    for batch_idx in range(layer.shape[0]):
                        for head_idx in range(layer.shape[1]):
                            attn_matrix = layer[batch_idx, head_idx]
                            if min(attn_matrix.shape) > 1:
                                subdiag_mean = np.mean(np.diag(attn_matrix, k=-1))
                                prev_token_scores.append({
                                    "layer": layer_idx,
                                    "head": head_idx,
                                    "score": float(subdiag_mean)
                                })
                elif len(layer.shape) == 3:  # [head, seq, seq]
                    for head_idx in range(layer.shape[0]):
                        attn_matrix = layer[head_idx]
                        if min(attn_matrix.shape) > 1:
                            subdiag_mean = np.mean(np.diag(attn_matrix, k=-1))
                            prev_token_scores.append({
                                "layer": layer_idx,
                                "head": head_idx,
                                "score": float(subdiag_mean)
                            })
                elif len(layer.shape) == 2:  # [seq, seq]
                    attn_matrix = layer
                    if min(attn_matrix.shape) > 1:
                        subdiag_mean = np.mean(np.diag(attn_matrix, k=-1))
                        prev_token_scores.append({
                            "layer": layer_idx,
                            "head": 0,
                            "score": float(subdiag_mean)
                        })
            except Exception as e:
                # Skip this layer if there's an error
                continue
        
        # Sort by score and take top 3
        prev_token_scores.sort(key=lambda x: x["score"], reverse=True)
        patterns["previous_token_attention"] = prev_token_scores[:3] if prev_token_scores else []
        
        # Identify "special token" attention if available
        special_token_scores = []
        if tokens and ('[CLS]' in tokens or '<s>' in tokens or '[BOS]' in tokens):
            special_idx = -1
            if '[CLS]' in tokens:
                special_idx = tokens.index('[CLS]')
            elif '<s>' in tokens:
                special_idx = tokens.index('<s>')
            elif '[BOS]' in tokens:
                special_idx = tokens.index('[BOS]')
                
            if special_idx >= 0:
                for layer_idx, layer in enumerate(weights_np):
                    try:
                        # Handle different tensor dimensions
                        if len(layer.shape) == 4:  # [batch, head, seq, seq]
                            for batch_idx in range(layer.shape[0]):
                                for head_idx in range(layer.shape[1]):
                                    attn_matrix = layer[batch_idx, head_idx]
                                    if special_idx < attn_matrix.shape[1]:
                                        special_mean = np.mean(attn_matrix[:, special_idx])
                                        special_token_scores.append({
                                            "layer": layer_idx,
                                            "head": head_idx,
                                            "score": float(special_mean)
                                        })
                        elif len(layer.shape) == 3:  # [head, seq, seq]
                            for head_idx in range(layer.shape[0]):
                                attn_matrix = layer[head_idx]
                                if special_idx < attn_matrix.shape[1]:
                                    special_mean = np.mean(attn_matrix[:, special_idx])
                                    special_token_scores.append({
                                        "layer": layer_idx,
                                        "head": head_idx,
                                        "score": float(special_mean)
                                    })
                        elif len(layer.shape) == 2:  # [seq, seq]
                            attn_matrix = layer
                            if special_idx < attn_matrix.shape[1]:
                                special_mean = np.mean(attn_matrix[:, special_idx])
                                special_token_scores.append({
                                    "layer": layer_idx,
                                    "head": 0,
                                    "score": float(special_mean)
                                })
                    except Exception as e:
                        # Skip this layer if there's an error
                        continue
        
        # Sort by score and take top 3
        special_token_scores.sort(key=lambda x: x["score"], reverse=True)
        patterns["special_token_attention"] = special_token_scores[:3] if special_token_scores else []
        
        return patterns 