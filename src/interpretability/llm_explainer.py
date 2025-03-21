"""
LLM Explainer module for the LingualLens framework.

This module provides techniques for explaining the behavior of black box LLMs
across different model providers including Hugging Face, OpenAI, Google, and Anthropic.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from transformers import AutoTokenizer

from src.core.model_loader import ModelWrapper, ModelProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMExplainer:
    """Class for explaining black box LLM behavior and outputs across different providers."""
    
    def __init__(self, model: ModelWrapper, tokenizer=None):
        """
        Initialize the LLM explainer.
        
        Args:
            model: The model to explain (ModelWrapper instance)
            tokenizer: Optional tokenizer. If None, will use the model's tokenizer
        """
        self.model = model
        
        # Handle tokenizer initialization based on model provider
        if tokenizer:
            self.tokenizer = tokenizer
        elif model.provider == ModelProvider.HUGGINGFACE and model.tokenizer:
            self.tokenizer = model.tokenizer
        elif model.provider == ModelProvider.HUGGINGFACE:
            self.tokenizer = AutoTokenizer.from_pretrained(model.model_name)
        else:
            # For API-based models, use a default tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                logger.info(f"Using default GPT-2 tokenizer for {model.provider} model")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {str(e)}")
                self.tokenizer = None
                
        # Provider-specific explainability capabilities
        self.provider_capabilities = {
            ModelProvider.HUGGINGFACE: {
                "attention_analysis": True,
                "embedding_access": True,
                "token_importance": True,
                "internal_representations": True
            },
            ModelProvider.OPENAI: {
                "attention_analysis": False,
                "embedding_access": True,
                "token_importance": True,
                "internal_representations": False
            },
            ModelProvider.GOOGLE: {
                "attention_analysis": False,
                "embedding_access": True,
                "token_importance": True,
                "internal_representations": False
            },
            ModelProvider.ANTHROPIC: {
                "attention_analysis": False,
                "embedding_access": False,
                "token_importance": True,
                "internal_representations": False
            },
            ModelProvider.AZURE: {
                "attention_analysis": False,
                "embedding_access": True,
                "token_importance": True,
                "internal_representations": False
            },
            ModelProvider.CUSTOM: {
                "attention_analysis": False,
                "embedding_access": False,
                "token_importance": True,
                "internal_representations": False
            }
        }
        
    def output_sensitivity(self, text: str, n_samples: int = 5) -> Optional[float]:
        """
        Measure the sensitivity of the model's output to perturbations in the input.
        
        Args:
            text: The input text to analyze
            n_samples: Number of perturbation samples
            
        Returns:
            Sensitivity score (higher means more sensitive)
        """
        try:
            # Get original output
            original_output = self.model.generate(text, max_length=50)
            
            # List to store differences
            differences = []
            
            # Apply small perturbations
            for _ in range(n_samples):
                perturbed_text = self._create_perturbation(text)
                perturbed_output = self.model.generate(perturbed_text, max_length=50)
                
                difference = self._semantic_difference(original_output, perturbed_output)
                differences.append(difference)
                
            # Overall sensitivity score
            sensitivity = sum(differences) / len(differences)
            return sensitivity
            
        except Exception as e:
            logger.error(f"Error in output_sensitivity: {str(e)}")
            return None
            
    def _create_perturbation(self, text: str) -> str:
        """Create a small perturbation of the input text."""
        perturbation_types = [
            self._add_typo,
            self._change_word_order,
            self._remove_word,
            self._add_filler_word
        ]
        
        # Select a random perturbation type
        perturbation_func = random.choice(perturbation_types)
        return perturbation_func(text)
    
    def _add_typo(self, text: str) -> str:
        """Add a typo to a random word."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        
        if len(word) <= 1:
            return text
            
        # Add a typo
        char_idx = random.randint(0, len(word) - 1)
        typo_type = random.choice(['swap', 'remove', 'duplicate'])
        
        if typo_type == 'swap' and char_idx < len(word) - 1:
            word = word[:char_idx] + word[char_idx+1] + word[char_idx] + word[char_idx+2:]
        elif typo_type == 'remove':
            word = word[:char_idx] + word[char_idx+1:]
        elif typo_type == 'duplicate':
            word = word[:char_idx] + word[char_idx] + word[char_idx:]
            
        words[idx] = word
        return ' '.join(words)
    
    def _change_word_order(self, text: str) -> str:
        """Change the order of two adjacent words."""
        words = text.split()
        if len(words) <= 2:
            return text
            
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx+1] = words[idx+1], words[idx]
        return ' '.join(words)
    
    def _remove_word(self, text: str) -> str:
        """Remove a random word."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
        return ' '.join(words)
    
    def _add_filler_word(self, text: str) -> str:
        """Add a filler word."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        fillers = ["actually", "basically", "like", "sort of", "kind of", "you know"]
        filler = random.choice(fillers)
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, filler)
        return ' '.join(words)
    
    def _semantic_difference(self, text1: str, text2: str) -> float:
        """
        Calculate semantic difference between two texts.
        
        For simplicity, using word overlap difference as a proxy.
        A more sophisticated implementation would use embeddings.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
        return 1.0 - jaccard
        
    def generate_counterfactuals(self, text: str, n_samples: int = 3) -> List[str]:
        """
        Generate counterfactual examples by systematically modifying the input.
        
        Args:
            text: The input text
            n_samples: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual examples
        """
        counterfactuals = []
        
        # Different counterfactual generation strategies
        strategies = [
            self._negate_statement,
            self._change_subject,
            self._change_sentiment,
            self._change_specificity
        ]
        
        # Try each strategy
        for strategy in strategies:
            if len(counterfactuals) >= n_samples:
                break
                
            try:
                cf = strategy(text)
                if cf and cf != text:
                    counterfactuals.append(cf)
            except Exception as e:
                logger.warning(f"Error in counterfactual strategy {strategy.__name__}: {str(e)}")
                
        # If still need more samples, use model-based approach
        if len(counterfactuals) < n_samples and hasattr(self.model, 'generate'):
            try:
                prompt = f"Generate a statement that is similar to but meaningfully different from: '{text}'\n\n"
                for _ in range(n_samples - len(counterfactuals)):
                    response = self.model.generate(prompt, max_length=100)
                    cf = self._extract_counterfactual(response, text)
                    if cf and cf != text and cf not in counterfactuals:
                        counterfactuals.append(cf)
            except Exception as e:
                logger.warning(f"Error in model-based counterfactual generation: {str(e)}")
                
        return counterfactuals[:n_samples]
    
    def _negate_statement(self, text: str) -> str:
        """Negate the main statement in the text."""
        negation_patterns = [
            (r'\b(is|are|was|were)\b', r'\1 not'),
            (r'\b(do|does|did)\b', r'\1 not'),
            (r'\b(can|could|should|would|will)\b', r'\1 not'),
            (r'\bnot\b', '')  # Remove existing negation
        ]
        
        result = text
        for pattern, replacement in negation_patterns:
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result, count=1)
                return result
                
        # If no pattern matched, try adding "not" before a key verb
        words = result.split()
        verbs = ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'have', 'has', 'had']
        for i, word in enumerate(words):
            if word.lower() in verbs and i < len(words) - 1:
                words.insert(i + 1, 'not')
                return ' '.join(words)
                
        return result
    
    def _change_subject(self, text: str) -> str:
        """Change the subject of the statement."""
        # Simple implementation - replace first pronoun or name
        pronouns = {
            'I': 'They', 'i': 'they',
            'We': 'They', 'we': 'they',
            'You': 'I', 'you': 'I',
            'He': 'She', 'he': 'she',
            'She': 'He', 'she': 'he',
            'They': 'We', 'they': 'we'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word in pronouns:
                words[i] = pronouns[word]
                return ' '.join(words)
                
        # If no pronouns found, try to substitute a common name
        names = ['John', 'Mary', 'The company', 'The team', 'The group', 'The organization']
        if len(words) > 2:
            words[0] = random.choice(names)
            return ' '.join(words)
            
        return text
    
    def _change_sentiment(self, text: str) -> str:
        """Change the sentiment of the statement."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'terrific', 'positive']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'negative', 'disappointing']
        
        # Replace positive words with negative words and vice versa
        words = text.split()
        for i, word in enumerate(words):
            lower_word = word.lower()
            if lower_word in positive_words:
                words[i] = random.choice(negative_words)
                return ' '.join(words)
            elif lower_word in negative_words:
                words[i] = random.choice(positive_words)
                return ' '.join(words)
                
        # If no sentiment words found, try adding a sentiment modifier
        if len(words) > 2:
            sentiment = random.choice(['Unfortunately', 'Sadly', 'Luckily', 'Fortunately', 'Surprisingly'])
            words.insert(0, sentiment + ',')
            return ' '.join(words)
            
        return text
    
    def _change_specificity(self, text: str) -> str:
        """Change specificity by adding or removing details."""
        words = text.split()
        
        if len(words) < 5:
            # Short text - add details
            details = [
                'specifically', 'particularly', 'especially', 'in detail', 
                'precisely', 'exactly', 'notably', 'in particular'
            ]
            idx = min(1, len(words) - 1)
            words.insert(idx, random.choice(details))
            
        else:
            # Longer text - remove some words to reduce specificity
            # Remove 20% of words, skipping first and last
            remove_count = max(1, len(words) // 5)
            indices = sorted(random.sample(range(1, len(words) - 1), remove_count))
            words = [w for i, w in enumerate(words) if i not in indices]
            
        return ' '.join(words)
    
    def _extract_counterfactual(self, response: str, original: str) -> Optional[str]:
        """Extract a counterfactual from model response."""
        # Try to find the counterfactual in the response
        if isinstance(response, dict) and 'text' in response:
            response = response['text']
            
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines, the original text, and lines that are too similar
            if (line and line != original and
                self._semantic_difference(line, original) > 0.3):
                return line
                
        return None
    
    def token_importance(self, text: str) -> Dict[str, float]:
        """
        Calculate the importance of each token in the input.
        
        Args:
            text: The input text
            
        Returns:
            Dictionary mapping tokens to importance scores
        """
        # Handle different model providers
        if self.model.provider == ModelProvider.HUGGINGFACE:
            return self._token_importance_huggingface(text)
        else:
            return self._token_importance_api(text)
    
    def _token_importance_huggingface(self, text: str) -> Dict[str, float]:
        """Token importance calculation for Hugging Face models."""
        if not self.tokenizer:
            logger.warning("Tokenizer not available for token importance analysis")
            return {}
            
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return {}
            
        importance_scores = {}
        original_output = self.model.generate(text, max_length=50)
        
        # Measure importance by ablating one token at a time
        for i, token in enumerate(tokens):
            try:
                ablated_tokens = tokens.copy()
                ablated_tokens[i] = "[MASK]"
                ablated_text = self.tokenizer.convert_tokens_to_string(ablated_tokens)
                
                ablated_output = self.model.generate(ablated_text, max_length=50)
                importance = self._semantic_difference(original_output, ablated_output)
                importance_scores[token] = importance
            except Exception as e:
                logger.warning(f"Error calculating importance for token {token}: {str(e)}")
                importance_scores[token] = 0.0
                
        return importance_scores
    
    def _token_importance_api(self, text: str) -> Dict[str, float]:
        """Token importance calculation for API-based models."""
        # For API models, use a simpler approach with word-level importance
        words = text.split()
        importance_scores = {}
        
        original_output = self.model.generate(text, max_length=50)
        
        # Measure importance by removing one word at a time
        for i, word in enumerate(words):
            try:
                ablated_words = words.copy()
                ablated_words[i] = "[...]"
                ablated_text = ' '.join(ablated_words)
                
                ablated_output = self.model.generate(ablated_text, max_length=50)
                importance = self._semantic_difference(original_output, ablated_output)
                importance_scores[word] = importance
            except Exception as e:
                logger.warning(f"Error calculating importance for word {word}: {str(e)}")
                importance_scores[word] = 0.0
                
        return importance_scores
    
    def visualize_token_importance(self, text: str, importance_scores: Optional[Dict[str, float]] = None, ax=None):
        """
        Visualize the importance of each token in the input.
        
        Args:
            text: The input text
            importance_scores: Optional pre-computed importance scores
            ax: Optional matplotlib axis for plotting
            
        Returns:
            Matplotlib axis object
        """
        # Calculate importance scores if not provided
        if importance_scores is None:
            importance_scores = self.token_importance(text)
            
        if not importance_scores:
            logger.warning("No importance scores available for visualization")
            return None
            
        tokens = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            
        # Color based on importance scores
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(scores)))
        sorted_indices = np.argsort(scores)
        
        ax.bar(range(len(tokens)), [scores[i] for i in sorted_indices], color=colors)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([tokens[i] for i in sorted_indices], rotation=45, ha="right")
        ax.set_ylabel("Importance Score")
        ax.set_title(f"Token Importance for {self.model.provider} Model")
        
        plt.tight_layout()
        return ax
    
    def analyze_internal_representations(self, text: str, layer_indices: List[int] = None) -> Optional[Dict]:
        """
        Analyze internal representations of the model for the given input.
        
        Args:
            text: The input text
            layer_indices: Indices of layers to analyze. If None, analyze all layers.
            
        Returns:
            Dictionary with analysis results
        """
        # Only supported for Hugging Face models
        if self.model.provider != ModelProvider.HUGGINGFACE:
            logger.warning(f"Internal representation analysis not supported for {self.model.provider} models")
            return None
            
        # Check if attention analysis is supported
        if not self.provider_capabilities[self.model.provider]["internal_representations"]:
            logger.warning(f"Internal representation analysis not supported for {self.model.provider} models")
            return None
            
        try:
            # Get attention patterns
            attention = self.model.get_attention(text)
            if not attention:
                logger.warning("No attention patterns available for analysis")
                return None
                
            # If layer_indices not specified, use all layers
            if layer_indices is None:
                layer_indices = list(range(len(attention)))
                
            results = {
                "attention_entropy": {},
                "attention_focus": {}
            }
            
            # Calculate attention entropy and focus for specified layers
            tokens = self.tokenizer.tokenize(text) if self.tokenizer else text.split()
            for layer_idx in layer_indices:
                if layer_idx >= len(attention):
                    continue
                    
                layer_attention = attention[layer_idx]
                
                # Attention entropy (higher means more uniform attention)
                entropy = -np.sum(layer_attention * np.log2(layer_attention + 1e-10), axis=-1).mean()
                results["attention_entropy"][f"layer_{layer_idx}"] = float(entropy)
                
                # Attention focus (position with highest attention weight)
                focus_positions = layer_attention.argmax(axis=-1)
                if len(focus_positions) > 0 and len(tokens) > 0:
                    focus_tokens = [tokens[pos] if pos < len(tokens) else "[PAD]" for pos in focus_positions]
                    results["attention_focus"][f"layer_{layer_idx}"] = focus_tokens
                
            return results
        except Exception as e:
            logger.error(f"Error in internal representation analysis: {str(e)}")
            return None
    
    def visualize_attention_patterns(self, text: str, layer_idx: int = 0, head_idx: int = 0, ax=None):
        """
        Visualize attention patterns for a specific layer and head.
        
        Args:
            text: The input text
            layer_idx: Layer index to visualize
            head_idx: Attention head index to visualize
            ax: Optional matplotlib axis for plotting
            
        Returns:
            Matplotlib axis object or None if not supported
        """
        # Only supported for Hugging Face models
        if self.model.provider != ModelProvider.HUGGINGFACE:
            logger.warning(f"Attention visualization not supported for {self.model.provider} models")
            return None
            
        # Check if attention analysis is supported
        if not self.provider_capabilities[self.model.provider]["attention_analysis"]:
            logger.warning(f"Attention visualization not supported for {self.model.provider} models")
            return None
            
        try:
            # Get attention patterns
            attention = self.model.get_attention(text)
            if not attention or layer_idx >= len(attention):
                logger.warning("No attention patterns available for visualization")
                return None
                
            # Extract attention weights for the specified layer and head
            layer_attention = attention[layer_idx]
            if head_idx >= layer_attention.shape[1]:
                logger.warning(f"Head index {head_idx} out of range")
                return None
                
            # Get attention weights for the specified head
            head_attention = layer_attention[0, head_idx].cpu().numpy()
            
            # Create a figure if not provided
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
                
            # Get tokens
            tokens = self.tokenizer.tokenize(text) if self.tokenizer else text.split()
            
            # Create attention heatmap
            sns.heatmap(
                head_attention[:len(tokens), :len(tokens)],
                annot=True,
                fmt=".2f",
                cmap="YlOrRd",
                xticklabels=tokens,
                yticklabels=tokens,
                ax=ax
            )
            
            ax.set_title(f"Attention Weights - Layer {layer_idx}, Head {head_idx}")
            plt.tight_layout()
            
            return ax
        except Exception as e:
            logger.error(f"Error in attention visualization: {str(e)}")
            return None
    
    def explain_generation(self, text: str, max_length: int = 50) -> Optional[str]:
        """
        Provide a comprehensive explanation of the model's generation process.
        
        Args:
            text: The input text
            max_length: Maximum length for generated output
            
        Returns:
            Explanation of the generation process
        """
        try:
            # Generate output
            output = self.model.generate(text, max_length=max_length)
            
            # Provider-specific explanations
            if self.model.provider == ModelProvider.HUGGINGFACE:
                explanation = self._explain_huggingface(text, output)
            else:
                explanation = self._explain_api_model(text, output)
                
            return explanation
        except Exception as e:
            logger.error(f"Error in generation explanation: {str(e)}")
            return f"Could not generate explanation due to error: {str(e)}"
    
    def _explain_huggingface(self, text: str, output: str) -> str:
        """Generate explanation for Hugging Face models."""
        # Create a structured explanation
        explanation_parts = []
        
        try:
            # Get token importance
            importance_scores = self.token_importance(text)
            
            if importance_scores:
                # Identify most important tokens
                top_tokens = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                important_tokens = ", ".join([f"'{t[0]}' (score: {t[1]:.2f})" for t in top_tokens])
                explanation_parts.append(f"Most influential input tokens: {important_tokens}")
                
            # Get attention analysis
            attention_analysis = self.analyze_internal_representations(text)
            if attention_analysis and "attention_entropy" in attention_analysis:
                # Identify layers with highest and lowest entropy
                entropies = attention_analysis["attention_entropy"]
                if entropies:
                    layers = sorted(entropies.items(), key=lambda x: x[1])
                    most_focused = layers[0][0] if layers else "None"
                    most_diffuse = layers[-1][0] if layers else "None"
                    explanation_parts.append(f"Most focused layer: {most_focused}, Most diffuse layer: {most_diffuse}")
                    
            # Add general explanation
            explanation_parts.append(f"Input text: \"{text}\"")
            explanation_parts.append(f"Model output: \"{output}\"")
            explanation_parts.append(f"Output length: {len(output.split())} words")
            
            # If no analysis was successful, provide a generic explanation
            if len(explanation_parts) <= 3:
                explanation_parts.append("The model processed the input using its trained parameters to generate the response.")
                
        except Exception as e:
            explanation_parts.append(f"Error during detailed analysis: {str(e)}")
            explanation_parts.append("The model generated a response based on patterns it learned during training.")
            
        return "\n".join(explanation_parts)
    
    def _explain_api_model(self, text: str, output: str) -> str:
        """Generate explanation for API-based models."""
        explanation_parts = []
        
        try:
            # Add general explanation
            explanation_parts.append(f"Input text: \"{text}\"")
            explanation_parts.append(f"Model output: \"{output}\"")
            explanation_parts.append(f"Output length: {len(output.split())} words")
            
            # Try to get token importance if available
            try:
                importance_scores = self.token_importance(text)
                if importance_scores:
                    top_tokens = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    important_tokens = ", ".join([f"'{t[0]}' (score: {t[1]:.2f})" for t in top_tokens])
                    explanation_parts.append(f"Most influential input words: {important_tokens}")
            except Exception:
                pass
                
            provider_name = self.model.provider.value.capitalize()
            model_name = self.model.model_name
            
            explanation_parts.append(f"This response was generated by {provider_name}'s {model_name} model.")
            explanation_parts.append("The model processed your input and generated a response based on its training data.")
            
            # Add provider-specific notes
            if self.model.provider == ModelProvider.OPENAI:
                explanation_parts.append("OpenAI models like GPT-3.5 and GPT-4 use a transformer architecture with a large number of parameters.")
            elif self.model.provider == ModelProvider.GOOGLE:
                explanation_parts.append("Google's Gemini models use a multimodal architecture trained on a diverse dataset.")
            elif self.model.provider == ModelProvider.ANTHROPIC:
                explanation_parts.append("Anthropic's Claude models are designed with Constitutional AI principles focusing on helpfulness and harmlessness.")
                
        except Exception as e:
            explanation_parts.append(f"Error during analysis: {str(e)}")
            explanation_parts.append(f"The {self.model.provider.value} model generated a response based on its training.")
            
        return "\n".join(explanation_parts) 