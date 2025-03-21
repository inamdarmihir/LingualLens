"""
Feature attribution module for the Language Model Evaluation Framework.

This module provides functionality for attributing model predictions to input features.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np

from ..core.model_loader import ModelWrapper


class FeatureAttributor:
    """Attributor for feature importance in language model predictions."""

    def __init__(
        self,
        model: ModelWrapper,
        method: str = "integrated_gradients",
        **kwargs,
    ):
        """
        Initialize the feature attributor.
        
        Args:
            model: The model to analyze
            method: Attribution method to use
                Options: "integrated_gradients", "gradient_shap", "occlusion"
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.method = method
        self.kwargs = kwargs

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze feature importance for a text input.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Get tokenized input
        tokens = self.model.tokenizer.tokenize(text)
        if not tokens:
            tokens = [t for t in text.split()]
        
        # Get embeddings
        embeddings = self.model.get_embeddings(text)
        
        # Calculate attributions based on the selected method
        if self.method == "integrated_gradients":
            attributions = self._integrated_gradients(text)
        elif self.method == "gradient_shap":
            attributions = self._gradient_shap(text)
        elif self.method == "occlusion":
            attributions = self._occlusion(text)
        else:
            raise ValueError(f"Unsupported attribution method: {self.method}")
            
        # Normalize attributions for easier interpretation
        attributions_normalized = self._normalize_attributions(attributions)
        
        # Identify top features
        top_features = self._identify_top_features(attributions_normalized, tokens)
        
        # Calculate feature statistics
        statistics = self._calculate_statistics(attributions_normalized)
        
        # Format and return results
        return {
            "attributions": attributions_normalized,
            "tokens": tokens,
            "top_features": top_features,
            "statistics": statistics,
            "method": self.method
        }

    def _integrated_gradients(self, text: str) -> np.ndarray:
        """
        Calculate attributions using Integrated Gradients method.
        
        This is a simplified implementation. In a real-world scenario,
        you would use a library like Captum for more accurate attributions.
        
        Args:
            text: Input text
            
        Returns:
            Array of attribution scores
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Create a baseline (e.g., zero embeddings)
        # 2. Interpolate between baseline and input
        # 3. Calculate gradients for each interpolation
        # 4. Integrate the gradients
        
        # Get tokenized input
        inputs = self.model.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Create a dummy attribution for demonstration purposes
        # In a real implementation, you would use proper gradient calculations
        attributions = np.random.random(len(self.model.tokenizer.tokenize(text)))
        
        return attributions

    def _gradient_shap(self, text: str) -> np.ndarray:
        """
        Calculate attributions using GradientSHAP method.
        
        This is a simplified implementation. In a real-world scenario,
        you would use a library like Captum for more accurate attributions.
        
        Args:
            text: Input text
            
        Returns:
            Array of attribution scores
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Generate reference inputs
        # 2. Calculate gradients with respect to inputs
        # 3. Apply SHAP calculations
        
        # Create a dummy attribution for demonstration purposes
        attributions = np.random.random(len(self.model.tokenizer.tokenize(text)))
        
        return attributions

    def _occlusion(self, text: str) -> np.ndarray:
        """
        Calculate attributions using the Occlusion method.
        
        This method works by replacing each token with a mask token
        and measuring the change in model output.
        
        Args:
            text: Input text
            
        Returns:
            Array of attribution scores
        """
        # Get tokenized input
        tokens = self.model.tokenizer.tokenize(text)
        
        # Create a dummy attribution for demonstration purposes
        # In a real implementation, you would:
        # 1. For each token, replace it with a mask token
        # 2. Calculate the model's output
        # 3. Measure the difference from the original output
        attributions = np.random.random(len(tokens))
        
        return attributions

    def _normalize_attributions(self, attributions: np.ndarray) -> np.ndarray:
        """
        Normalize attribution scores to [-1, 1] range.
        
        Args:
            attributions: Raw attribution scores
            
        Returns:
            Normalized attribution scores
        """
        if attributions.size == 0:
            return attributions
            
        abs_max = np.max(np.abs(attributions))
        if abs_max > 0:
            return attributions / abs_max
        else:
            return attributions

    def _identify_top_features(self, attributions: np.ndarray, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Identify top features by attribution score.
        
        Args:
            attributions: Attribution scores
            tokens: List of tokens
            
        Returns:
            List of top features with their scores
        """
        # Pair tokens with attributions
        token_attributions = list(zip(tokens, attributions))
        
        # Sort by absolute attribution value (descending)
        token_attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top features (up to 5)
        top_features = []
        for token, attribution in token_attributions[:5]:
            top_features.append({
                "token": token,
                "attribution": float(attribution)
            })
            
        return top_features

    def _calculate_statistics(self, attributions: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for the attribution scores.
        
        Args:
            attributions: Attribution scores
            
        Returns:
            Dictionary of statistical measures
        """
        if attributions.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sparsity": 0.0
            }
            
        # Calculate basic statistics
        mean = np.mean(attributions)
        std = np.std(attributions)
        min_val = np.min(attributions)
        max_val = np.max(attributions)
        
        # Calculate sparsity (percentage of attributions close to zero)
        threshold = 0.01  # Consider attributions below 1% of max as close to zero
        sparsity = np.mean(np.abs(attributions) < threshold)
        
        return {
            "mean": float(mean),
            "std": float(std),
            "min": float(min_val),
            "max": float(max_val),
            "sparsity": float(sparsity)
        } 