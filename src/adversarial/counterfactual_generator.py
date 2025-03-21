"""
Counterfactual generator module for the Language Model Evaluation Framework.

This module provides functionality for generating counterfactual examples to test
model robustness and decision boundaries.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
import random
import re
from collections import defaultdict

from ..core.model_loader import ModelWrapper


class CounterfactualGenerator:
    """Generator for counterfactual examples that test model decision boundaries."""

    def __init__(
        self,
        model: ModelWrapper,
        method: str = "rule_based",
        **kwargs,
    ):
        """
        Initialize the counterfactual generator.
        
        Args:
            model: The model to test
            method: Method to use for counterfactual generation
                (options: "rule_based", "perturbation", "model_guided")
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.method = method
        self.kwargs = kwargs
        
        # Load resources for counterfactual generation
        self._load_resources()
        
    def _load_resources(self):
        """Load necessary resources for counterfactual generation."""
        # Define opposites for rule-based counterfactual generation
        self.opposites = {
            "good": "bad",
            "bad": "good",
            "happy": "sad",
            "sad": "happy",
            "positive": "negative",
            "negative": "positive",
            "agree": "disagree",
            "disagree": "agree",
            "like": "dislike",
            "dislike": "like",
            "love": "hate",
            "hate": "love",
            "smart": "stupid",
            "stupid": "smart",
            "true": "false",
            "false": "true",
            "yes": "no",
            "no": "yes",
        }
        
        # Define negation words
        self.negations = ["not", "no", "never", "none", "nobody", "nothing", "nowhere"]
        
        # Define common sentiment words
        self.sentiment_words = {
            "positive": [
                "good", "great", "excellent", "amazing", "wonderful", "fantastic",
                "terrific", "outstanding", "superb", "awesome", "brilliant", "perfect"
            ],
            "negative": [
                "bad", "terrible", "awful", "horrible", "poor", "disappointing",
                "unfortunate", "unpleasant", "dreadful", "inferior", "subpar", "atrocious"
            ]
        }

    def generate(
        self, text: str, target_class: Optional[str] = None, num_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual examples for the input text.
        
        Args:
            text: The input text to generate counterfactuals from
            target_class: Target class for classification models
            num_examples: Number of counterfactuals to generate
            
        Returns:
            List of dictionaries containing counterfactual results
        """
        # Select the appropriate generation method
        if self.method == "rule_based":
            counterfactuals = self._rule_based_generation(text, num_examples)
        elif self.method == "perturbation":
            counterfactuals = self._perturbation_based_generation(text, num_examples)
        elif self.method == "model_guided":
            counterfactuals = self._model_guided_generation(text, target_class, num_examples)
        else:
            raise ValueError(f"Unsupported counterfactual generation method: {self.method}")
            
        results = []
        
        # Get original model output
        original_output = None
        if self.model.task == "text-generation":
            original_output = self.model.generate(text)
        elif self.model.task == "classification":
            original_output = self.model.classify(text)
        else:
            # Use embeddings for general purpose models
            original_output = self.model.get_embeddings(text).cpu().numpy()
            
        # Process each counterfactual
        for cf_text in counterfactuals:
            # Skip if no changes were made
            if cf_text == text:
                continue
                
            # Get model output for counterfactual
            cf_output = None
            if self.model.task == "text-generation":
                cf_output = self.model.generate(cf_text)
            elif self.model.task == "classification":
                cf_output = self.model.classify(cf_text)
            else:
                # Use embeddings for general purpose models
                cf_output = self.model.get_embeddings(cf_text).cpu().numpy()
                
            # Determine if counterfactual was successful
            success = self._is_counterfactual_successful(original_output, cf_output, target_class)
            
            # Add to results
            results.append({
                "original_text": text,
                "counterfactual_text": cf_text,
                "original_output": original_output,
                "counterfactual_output": cf_output,
                "success": success,
                "difference": self._calculate_difference(original_output, cf_output),
                "method": self.method
            })
            
            # Limit to num_examples
            if len(results) >= num_examples:
                break
                
        return results

    def _rule_based_generation(self, text: str, num_examples: int) -> List[str]:
        """
        Generate counterfactuals using rule-based methods.
        
        Args:
            text: Input text
            num_examples: Number of examples to generate
            
        Returns:
            List of counterfactual texts
        """
        results = []
        
        # 1. Replace with opposites
        words = text.split()
        for i in range(len(words)):
            word = words[i].lower().strip(",.!?;:")
            if word in self.opposites:
                new_words = words.copy()
                new_words[i] = self.opposites[word]
                results.append(" ".join(new_words))
                
        # 2. Add/remove negations
        has_negation = any(neg in text.lower() for neg in self.negations)
        words = text.split()
        
        if has_negation:
            # Remove negation
            for i in range(len(words)):
                if words[i].lower() in self.negations:
                    new_words = words.copy()
                    new_words.pop(i)
                    results.append(" ".join(new_words))
        else:
            # Add negation before important words
            for i in range(len(words)):
                word = words[i].lower().strip(",.!?;:")
                if word in self.opposites or word in self.sentiment_words["positive"] or word in self.sentiment_words["negative"]:
                    new_words = words.copy()
                    new_words.insert(i, "not")
                    results.append(" ".join(new_words))
        
        # 3. Change sentiment polarity
        positive_sentiment = any(word in text.lower() for word in self.sentiment_words["positive"])
        negative_sentiment = any(word in text.lower() for word in self.sentiment_words["negative"])
        
        if positive_sentiment:
            for pos_word in self.sentiment_words["positive"]:
                if pos_word in text.lower():
                    # Replace with negative sentiment
                    for neg_word in self.sentiment_words["negative"]:
                        results.append(text.lower().replace(pos_word, neg_word))
        
        if negative_sentiment:
            for neg_word in self.sentiment_words["negative"]:
                if neg_word in text.lower():
                    # Replace with positive sentiment
                    for pos_word in self.sentiment_words["positive"]:
                        results.append(text.lower().replace(neg_word, pos_word))
        
        # Shuffle and return limited results
        random.shuffle(results)
        return results[:num_examples]

    def _perturbation_based_generation(self, text: str, num_examples: int) -> List[str]:
        """
        Generate counterfactuals using perturbation methods.
        
        Args:
            text: Input text
            num_examples: Number of examples to generate
            
        Returns:
            List of counterfactual texts
        """
        results = []
        
        # 1. Word removal
        words = text.split()
        for i in range(len(words)):
            new_words = words.copy()
            new_words.pop(i)
            results.append(" ".join(new_words))
        
        # 2. Word addition
        words = text.split()
        for i in range(len(words) + 1):
            # Add positive words
            for word in random.sample(self.sentiment_words["positive"], min(3, len(self.sentiment_words["positive"]))):
                new_words = words.copy()
                new_words.insert(i, word)
                results.append(" ".join(new_words))
                
            # Add negative words
            for word in random.sample(self.sentiment_words["negative"], min(3, len(self.sentiment_words["negative"]))):
                new_words = words.copy()
                new_words.insert(i, word)
                results.append(" ".join(new_words))
        
        # 3. Word replacement
        words = text.split()
        for i in range(len(words)):
            word = words[i].lower().strip(",.!?;:")
            
            # Replace with positive words
            for new_word in random.sample(self.sentiment_words["positive"], min(2, len(self.sentiment_words["positive"]))):
                if word != new_word:
                    new_words = words.copy()
                    new_words[i] = new_word
                    results.append(" ".join(new_words))
            
            # Replace with negative words
            for new_word in random.sample(self.sentiment_words["negative"], min(2, len(self.sentiment_words["negative"]))):
                if word != new_word:
                    new_words = words.copy()
                    new_words[i] = new_word
                    results.append(" ".join(new_words))
        
        # Shuffle and return limited results
        random.shuffle(results)
        return results[:num_examples]

    def _model_guided_generation(
        self, text: str, target_class: Optional[str], num_examples: int
    ) -> List[str]:
        """
        Generate counterfactuals using model guidance.
        
        This is a simplified implementation. In a real-world scenario,
        you would use more sophisticated methods to explore the model's decision boundary.
        
        Args:
            text: Input text
            target_class: Target class for counterfactuals
            num_examples: Number of examples to generate
            
        Returns:
            List of counterfactual texts
        """
        # Start with candidates from rule-based and perturbation methods
        candidates = self._rule_based_generation(text, num_examples * 2)
        candidates.extend(self._perturbation_based_generation(text, num_examples * 2))
        
        results = []
        
        # Get original model output
        original_output = None
        if self.model.task == "classification":
            original_output = self.model.classify(text)
            
            # Determine the source class
            source_class = max(original_output.items(), key=lambda x: x[1])[0]
            
            # If target_class is not specified, use any class different from source
            if target_class is None:
                target_classes = [c for c in original_output.keys() if c != source_class]
                if not target_classes:
                    return []  # No other classes available
                target_class = random.choice(target_classes)
                
            # Evaluate candidates and select those that change the prediction
            for candidate in candidates:
                cf_output = self.model.classify(candidate)
                cf_class = max(cf_output.items(), key=lambda x: x[1])[0]
                
                # Check if prediction changed to target class
                if cf_class == target_class:
                    results.append(candidate)
                    
                    # Limit to num_examples
                    if len(results) >= num_examples:
                        break
        
        else:
            # For non-classification models, use embedding distance as a proxy
            original_embedding = self.model.get_embeddings(text).cpu().numpy()
            
            # Calculate embedding for each candidate
            candidate_distances = []
            for candidate in candidates:
                cf_embedding = self.model.get_embeddings(candidate).cpu().numpy()
                
                # Calculate cosine distance
                cosine_sim = np.dot(original_embedding, cf_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(cf_embedding)
                )
                distance = 1 - cosine_sim
                
                candidate_distances.append((candidate, distance))
            
            # Sort by distance (descending) to find most different candidates
            candidate_distances.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top candidates
            results = [c[0] for c in candidate_distances[:num_examples]]
        
        return results

    def _is_counterfactual_successful(
        self, original_output: Any, cf_output: Any, target_class: Optional[str]
    ) -> bool:
        """
        Determine if a counterfactual example was successful.
        
        Args:
            original_output: Original model output
            cf_output: Counterfactual model output
            target_class: Target class (for classification)
            
        Returns:
            True if successful, False otherwise
        """
        # For classification models
        if isinstance(original_output, dict) and isinstance(cf_output, dict):
            original_class = max(original_output.items(), key=lambda x: x[1])[0]
            cf_class = max(cf_output.items(), key=lambda x: x[1])[0]
            
            # Check if prediction changed to target class if specified
            if target_class is not None:
                return cf_class == target_class
            # Otherwise, any change in prediction is considered successful
            else:
                return original_class != cf_class
        
        # For text generation models
        elif isinstance(original_output, str) and isinstance(cf_output, str):
            # Simple difference check
            return original_output != cf_output
        
        # For embedding models
        elif isinstance(original_output, np.ndarray) and isinstance(cf_output, np.ndarray):
            # Check if embeddings are significantly different
            cosine_sim = np.dot(original_output, cf_output) / (
                np.linalg.norm(original_output) * np.linalg.norm(cf_output)
            )
            return cosine_sim < 0.8  # Consider different if similarity below threshold
        
        return False

    def _calculate_difference(self, original_output: Any, cf_output: Any) -> float:
        """
        Calculate difference between original and counterfactual outputs.
        
        Args:
            original_output: Original model output
            cf_output: Counterfactual model output
            
        Returns:
            Difference metric (0-1 scale)
        """
        # For classification models
        if isinstance(original_output, dict) and isinstance(cf_output, dict):
            diff = 0
            for key in set(original_output.keys()) | set(cf_output.keys()):
                orig_val = original_output.get(key, 0)
                cf_val = cf_output.get(key, 0)
                diff += abs(orig_val - cf_val)
            return min(1.0, diff / 2.0)  # Normalize to 0-1
        
        # For text generation models
        elif isinstance(original_output, str) and isinstance(cf_output, str):
            # Use string similarity
            from difflib import SequenceMatcher
            return 1 - SequenceMatcher(None, original_output, cf_output).ratio()
        
        # For embedding models
        elif isinstance(original_output, np.ndarray) and isinstance(cf_output, np.ndarray):
            # Use cosine distance
            cosine_sim = np.dot(original_output, cf_output) / (
                np.linalg.norm(original_output) * np.linalg.norm(cf_output)
            )
            return 1 - max(0, cosine_sim)  # Convert similarity to distance
        
        return 0.0 