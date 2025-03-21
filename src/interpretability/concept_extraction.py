"""
Concept extraction module for the Language Model Evaluation Framework.

This module provides functionality for extracting interpretable concepts from
model representations.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from collections import defaultdict

from ..core.model_loader import ModelWrapper


class ConceptExtractor:
    """Extractor for interpretable concepts from model representations."""

    def __init__(
        self,
        model: ModelWrapper,
        method: str = "clustering",
        num_concepts: int = 10,
        **kwargs,
    ):
        """
        Initialize the concept extractor.
        
        Args:
            model: The model to analyze
            method: Method to use for concept extraction
                Options: "clustering", "pca", "dictionaries"
            num_concepts: Number of concepts to extract
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.method = method
        self.num_concepts = num_concepts
        self.kwargs = kwargs
        
        # Dictionary to cache concepts for reuse
        self.concept_cache = {}

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Extract concepts from a text input.
        
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
        
        # Extract concepts based on the selected method
        if self.method == "clustering":
            concepts, activations = self._extract_concepts_clustering(embeddings, tokens)
        elif self.method == "pca":
            concepts, activations = self._extract_concepts_pca(embeddings, tokens)
        elif self.method == "dictionaries":
            concepts, activations = self._extract_concepts_dictionaries(embeddings, tokens)
        else:
            raise ValueError(f"Unsupported concept extraction method: {self.method}")
            
        # Identify top concepts
        top_concepts = self._identify_top_concepts(concepts, activations)
        
        # Calculate concept statistics
        statistics = self._calculate_statistics(activations)
        
        # Format and return results
        return {
            "concepts": concepts,
            "activations": activations,
            "tokens": tokens,
            "top_concepts": top_concepts,
            "statistics": statistics,
            "method": self.method
        }

    def extract_concepts(self, text: str) -> Dict[str, Any]:
        """
        Extract concepts from a text input (alias for analyze).
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        return self.analyze(text)

    def _extract_concepts_clustering(
        self, embeddings: torch.Tensor, tokens: List[str]
    ) -> tuple:
        """
        Extract concepts using clustering of token embeddings.
        
        This is a simplified implementation. In a real-world scenario,
        you would use more sophisticated clustering techniques.
        
        Args:
            embeddings: Token embeddings
            tokens: List of tokens
            
        Returns:
            Tuple of (concepts, activations)
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Apply clustering to the embeddings
        # 2. Extract concept vectors from cluster centroids
        # 3. Calculate activations as similarity to concept vectors
        
        # Create dummy concepts and activations for demonstration purposes
        concepts = []
        num_concepts = min(self.num_concepts, max(1, len(tokens)))
        for i in range(num_concepts):
            # Each concept is represented by a name and a description
            concepts.append({
                "id": i,
                "name": f"Concept_{i}",
                "description": f"Automatically extracted concept {i}",
                "related_words": self._get_related_words(i, tokens)
            })
            
        # Create dummy activations (concept x token)
        activations = np.random.random((len(concepts), max(1, len(tokens))))
        
        return concepts, activations

    def _extract_concepts_pca(
        self, embeddings: torch.Tensor, tokens: List[str]
    ) -> tuple:
        """
        Extract concepts using PCA on token embeddings.
        
        Args:
            embeddings: Token embeddings
            tokens: List of tokens
            
        Returns:
            Tuple of (concepts, activations)
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Apply PCA to the embeddings
        # 2. Extract concept vectors from principal components
        # 3. Calculate activations as projections onto principal components
        
        # Create dummy concepts and activations for demonstration purposes
        concepts = []
        num_concepts = min(self.num_concepts, max(1, len(tokens)))
        for i in range(num_concepts):
            concepts.append({
                "id": i,
                "name": f"Component_{i}",
                "description": f"PCA component {i}",
                "related_words": self._get_related_words(i, tokens)
            })
            
        # Create dummy activations (concept x token)
        activations = np.random.random((len(concepts), max(1, len(tokens))))
        
        return concepts, activations

    def _extract_concepts_dictionaries(
        self, embeddings: torch.Tensor, tokens: List[str]
    ) -> tuple:
        """
        Extract concepts using predefined dictionaries of related words.
        
        Args:
            embeddings: Token embeddings
            tokens: List of tokens
            
        Returns:
            Tuple of (concepts, activations)
        """
        # Predefined concept dictionaries
        concept_dicts = {
            "sentiment": {
                "name": "Sentiment",
                "description": "Positive vs negative sentiment",
                "positive": ["good", "great", "excellent", "positive", "nice", "wonderful", "amazing"],
                "negative": ["bad", "terrible", "awful", "negative", "poor", "disappointing"]
            },
            "size": {
                "name": "Size",
                "description": "Large vs small size",
                "positive": ["large", "big", "huge", "enormous", "gigantic", "massive"],
                "negative": ["small", "tiny", "little", "miniature", "compact", "microscopic"]
            },
            "speed": {
                "name": "Speed",
                "description": "Fast vs slow speed",
                "positive": ["fast", "quick", "rapid", "swift", "speedy", "nimble"],
                "negative": ["slow", "sluggish", "crawling", "plodding", "lethargic"]
            }
        }
        
        # Token similarity function
        def token_similarity(token, word_list):
            if token.lower() in word_list:
                return 1.0
            # Simple string similarity (in a real system, you would use embeddings)
            return 0.0
        
        # Extract concepts from dictionaries
        concepts = []
        activations_dict = {}
        
        for concept_id, (concept_key, concept_dict) in enumerate(concept_dicts.items()):
            concept = {
                "id": concept_id,
                "name": concept_dict["name"],
                "description": concept_dict["description"],
                "related_words": concept_dict["positive"][:3] + concept_dict["negative"][:3]
            }
            concepts.append(concept)
            
            # Calculate token activations for this concept
            concept_activations = []
            for token in tokens:
                positive_sim = max([token_similarity(token, word) for word in concept_dict["positive"]])
                negative_sim = max([token_similarity(token, word) for word in concept_dict["negative"]])
                activation = positive_sim - negative_sim
                concept_activations.append(activation)
                
            activations_dict[concept_id] = concept_activations
            
        # Format activations as a matrix
        num_tokens = max(1, len(tokens))
        num_concepts = len(concepts)
        activations = np.zeros((num_concepts, num_tokens))
        
        for concept_id, concept_activations in activations_dict.items():
            if concept_id < num_concepts and len(concept_activations) > 0:
                for i, activation in enumerate(concept_activations):
                    if i < num_tokens:
                        activations[concept_id, i] = activation
        
        return concepts, activations

    def _get_related_words(
        self, concept_idx: int, tokens: List[str], activations: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Get words related to a concept.
        
        Args:
            concept_idx: Index of the concept
            tokens: List of tokens
            activations: Optional activations for the concept
            
        Returns:
            List of related words
        """
        # If activations are provided, use them to identify related words
        if activations is not None:
            # Pair tokens with activations
            token_activations = list(zip(tokens, activations))
            
            # Sort by activation value (descending)
            token_activations.sort(key=lambda x: x[1], reverse=True)
            
            # Return top related words (up to 5)
            return [token for token, _ in token_activations[:5]]
        
        # Otherwise, return a random subset of tokens
        else:
            # Shuffle tokens
            shuffled_tokens = tokens.copy()
            np.random.shuffle(shuffled_tokens)
            
            # Return a subset of tokens (up to 5)
            return shuffled_tokens[:min(5, len(shuffled_tokens))]

    def _identify_top_concepts(self, concepts: List[Dict[str, Any]], activations: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify top concepts by average activation.
        
        Args:
            concepts: List of concepts
            activations: Concept activations
            
        Returns:
            List of top concepts with their scores
        """
        # Calculate average activation for each concept
        avg_activations = np.mean(activations, axis=1)
        
        # Pair concepts with average activations
        concept_activations = list(zip(concepts, avg_activations))
        
        # Sort by average activation (descending)
        concept_activations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top concepts (up to 5)
        top_concepts = []
        for concept, avg_activation in concept_activations[:5]:
            concept_copy = concept.copy()
            concept_copy["average_activation"] = float(avg_activation)
            top_concepts.append(concept_copy)
            
        return top_concepts

    def _calculate_statistics(self, activations: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for the concept activations.
        
        Args:
            activations: Concept activations
            
        Returns:
            Dictionary of statistical measures
        """
        if activations.size == 0:
            return {
                "mean_activation": 0.0,
                "max_activation": 0.0,
                "concept_sparsity": 0.0,
                "token_coverage": 0.0
            }
            
        # Calculate basic statistics
        mean_activation = np.mean(activations)
        max_activation = np.max(activations)
        
        # Calculate concept sparsity (percentage of activations close to zero)
        threshold = 0.1  # Consider activations below 10% as close to zero
        concept_sparsity = np.mean(activations < threshold)
        
        # Calculate token coverage (percentage of tokens with at least one significant concept)
        token_coverage = np.mean(np.max(activations, axis=0) >= threshold)
        
        return {
            "mean_activation": float(mean_activation),
            "max_activation": float(max_activation),
            "concept_sparsity": float(concept_sparsity),
            "token_coverage": float(token_coverage)
        } 