"""
Attack generator module for the Language Model Evaluation Framework.

This module provides functionality for generating adversarial attacks on language models.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
import random
import re
from collections import defaultdict

from ..core.model_loader import ModelWrapper


class AttackGenerator:
    """Generator for various adversarial attacks on language models."""

    def __init__(
        self,
        model: ModelWrapper,
        attack_types: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the attack generator.
        
        Args:
            model: The model to attack
            attack_types: Types of attacks to use
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.attack_types = attack_types or ["typo", "synonym", "insertion", "deletion"]
        self.kwargs = kwargs
        
        # Load resources if needed
        self._load_resources()

    def _load_resources(self):
        """Load necessary resources for attacks."""
        # Example synonyms dictionary (in a real implementation, this would be more extensive)
        self.synonyms = {
            "good": ["great", "excellent", "fine", "positive"],
            "bad": ["poor", "terrible", "awful", "negative"],
            "happy": ["glad", "pleased", "delighted", "joyful"],
            "sad": ["unhappy", "depressed", "melancholy", "gloomy"],
            "big": ["large", "huge", "enormous", "gigantic"],
            "small": ["tiny", "little", "miniature", "diminutive"],
        }
        
        # Common character-level substitutions
        self.char_subs = {
            'a': ['@', '4', 'á', 'à', 'â', 'ä'],
            'b': ['8', '6', 'ß', 'þ'],
            'c': ['(', '<', '¢', 'ç'],
            'e': ['3', '€', 'é', 'è', 'ê', 'ë'],
            'i': ['1', '!', '|', 'í', 'ì', 'î', 'ï'],
            'l': ['1', '|', '/'],
            'o': ['0', '()', 'ó', 'ò', 'ô', 'ö'],
            's': ['5', '$', 'š'],
            't': ['7', '+', 'ţ'],
            'u': ['ú', 'ù', 'û', 'ü'],
        }

    def generate(
        self, text: str, level: str = "word", num_attacks: int = 5, targeted: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate adversarial examples based on the input text.
        
        Args:
            text: The input text to generate attacks from
            level: Level of attacks (character, word, sentence)
            num_attacks: Number of attacks to generate
            targeted: Whether to target specific outputs
            
        Returns:
            List of dictionaries containing attack results
        """
        if level not in ["character", "word", "sentence"]:
            raise ValueError(f"Unsupported attack level: {level}")
            
        results = []
        
        # Get original model output for comparison
        original_output = None
        if self.model.task == "text-generation":
            original_output = self.model.generate(text)
        elif self.model.task == "classification":
            original_output = self.model.classify(text)
        else:
            # Use embeddings for general purpose models
            original_output = self.model.get_embeddings(text).cpu().numpy()
            
        # Generate different types of attacks based on level
        for _ in range(num_attacks):
            for attack_type in self.attack_types:
                # Apply attack
                attacked_text = self._apply_attack(text, level, attack_type)
                
                # Skip if no changes were made
                if attacked_text == text:
                    continue
                    
                # Get model output for attacked text
                attacked_output = None
                if self.model.task == "text-generation":
                    attacked_output = self.model.generate(attacked_text)
                elif self.model.task == "classification":
                    attacked_output = self.model.classify(attacked_text)
                else:
                    # Use embeddings for general purpose models
                    attacked_output = self.model.get_embeddings(attacked_text).cpu().numpy()
                    
                # Calculate success metrics
                success = self._calculate_success(original_output, attacked_output)
                
                # Add to results
                results.append({
                    "original_text": text,
                    "attacked_text": attacked_text,
                    "attack_type": attack_type,
                    "attack_level": level,
                    "original_output": original_output,
                    "attacked_output": attacked_output,
                    "success": success,
                    "difference": self._calculate_difference(original_output, attacked_output)
                })
                
                # Limit to num_attacks
                if len(results) >= num_attacks:
                    return results
                    
        return results

    def _apply_attack(self, text: str, level: str, attack_type: str) -> str:
        """
        Apply a specific attack to the text.
        
        Args:
            text: Input text
            level: Attack level
            attack_type: Type of attack
            
        Returns:
            Attacked text
        """
        if level == "character":
            return self._character_level_attack(text, attack_type)
        elif level == "word":
            return self._word_level_attack(text, attack_type)
        elif level == "sentence":
            return self._sentence_level_attack(text, attack_type)
        else:
            return text

    def _character_level_attack(self, text: str, attack_type: str) -> str:
        """Apply character-level attacks."""
        if attack_type == "typo":
            # Randomly swap, delete, or substitute characters
            chars = list(text)
            for _ in range(max(1, len(text) // 10)):  # Modify ~10% of characters
                idx = random.randint(0, len(chars) - 1)
                op = random.choice(["swap", "delete", "substitute"])
                
                if op == "swap" and idx < len(chars) - 1:
                    chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
                elif op == "delete":
                    chars[idx] = ""
                elif op == "substitute" and chars[idx].lower() in self.char_subs:
                    chars[idx] = random.choice(self.char_subs[chars[idx].lower()])
                    
            return "".join(chars)
        
        elif attack_type == "insertion":
            # Insert random characters
            chars = list(text)
            for _ in range(max(1, len(text) // 10)):
                idx = random.randint(0, len(chars))
                chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz "))
            return "".join(chars)
        
        else:
            return text

    def _word_level_attack(self, text: str, attack_type: str) -> str:
        """Apply word-level attacks."""
        words = text.split()
        
        if attack_type == "synonym":
            # Replace words with synonyms
            for i in range(len(words)):
                if words[i].lower() in self.synonyms and random.random() < 0.3:
                    words[i] = random.choice(self.synonyms[words[i].lower()])
            return " ".join(words)
        
        elif attack_type == "deletion":
            # Delete random words
            indices = list(range(len(words)))
            random.shuffle(indices)
            deletion_count = max(1, len(words) // 10)
            for idx in indices[:deletion_count]:
                words[idx] = ""
            return " ".join(word for word in words if word)
        
        elif attack_type == "insertion":
            # Insert random words
            insertion_count = max(1, len(words) // 10)
            common_words = ["the", "a", "an", "and", "or", "but", "so", "very", "quite", "really"]
            
            for _ in range(insertion_count):
                idx = random.randint(0, len(words))
                words.insert(idx, random.choice(common_words))
            
            return " ".join(words)
        
        else:
            return text

    def _sentence_level_attack(self, text: str, attack_type: str) -> str:
        """Apply sentence-level attacks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if attack_type == "deletion" and len(sentences) > 1:
            # Delete a random sentence
            idx = random.randint(0, len(sentences) - 1)
            sentences.pop(idx)
            return " ".join(sentences)
        
        elif attack_type == "insertion":
            # Insert a generated or template sentence
            templates = [
                "This is very interesting.",
                "I'm not sure about that.",
                "Let me think about it.",
                "That's a good point.",
                "I have some doubts about this."
            ]
            idx = random.randint(0, len(sentences))
            sentences.insert(idx, random.choice(templates))
            return " ".join(sentences)
        
        else:
            return text

    def _calculate_success(self, original_output: Any, attacked_output: Any) -> bool:
        """
        Calculate if the attack was successful.
        
        Args:
            original_output: Original model output
            attacked_output: Output after attack
            
        Returns:
            True if attack succeeded, False otherwise
        """
        # For text generation, simple string comparison
        if isinstance(original_output, str) and isinstance(attacked_output, str):
            return original_output != attacked_output
        
        # For classification, check if prediction changed
        elif isinstance(original_output, dict) and isinstance(attacked_output, dict):
            original_class = max(original_output.items(), key=lambda x: x[1])[0]
            attacked_class = max(attacked_output.items(), key=lambda x: x[1])[0]
            return original_class != attacked_class
        
        # For embeddings, check cosine distance
        elif isinstance(original_output, np.ndarray) and isinstance(attacked_output, np.ndarray):
            cosine_sim = np.dot(original_output, attacked_output) / (
                np.linalg.norm(original_output) * np.linalg.norm(attacked_output)
            )
            # Different if similarity is below threshold
            return cosine_sim < 0.9
        
        return False

    def _calculate_difference(self, original_output: Any, attacked_output: Any) -> float:
        """
        Calculate difference between original and attacked outputs.
        
        Args:
            original_output: Original model output
            attacked_output: Output after attack
            
        Returns:
            Difference metric (0-1 scale)
        """
        # For text generation, use string difference metrics
        if isinstance(original_output, str) and isinstance(attacked_output, str):
            # Levenshtein distance normalized by max length
            from difflib import SequenceMatcher
            return 1 - SequenceMatcher(None, original_output, attacked_output).ratio()
        
        # For classification, calculate probability difference
        elif isinstance(original_output, dict) and isinstance(attacked_output, dict):
            diff = 0
            for key in set(original_output.keys()) | set(attacked_output.keys()):
                orig_val = original_output.get(key, 0)
                attack_val = attacked_output.get(key, 0)
                diff += abs(orig_val - attack_val)
            return min(1.0, diff / 2.0)  # Normalize to 0-1
        
        # For embeddings, use cosine distance
        elif isinstance(original_output, np.ndarray) and isinstance(attacked_output, np.ndarray):
            cosine_sim = np.dot(original_output, attacked_output) / (
                np.linalg.norm(original_output) * np.linalg.norm(attacked_output)
            )
            return 1 - max(0, cosine_sim)  # Convert similarity to distance
        
        return 0.0 