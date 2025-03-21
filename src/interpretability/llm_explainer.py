"""
LLM Explainer module for the LingualLens framework.

This module provides techniques for explaining the behavior of black box LLMs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import torch
from transformers import AutoTokenizer

class LLMExplainer:
    """Class for explaining black box LLM behavior and outputs."""
    
    def __init__(self, model, tokenizer=None):
        """
        Initialize the LLM explainer.
        
        Args:
            model: The model to explain (ModelWrapper instance)
            tokenizer: Optional tokenizer. If None, will use the model's tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(model.model_name)
        
    def output_sensitivity(self, prompt: str, target_tokens: List[str], 
                          perturbation_strength: float = 0.1, 
                          num_samples: int = 10) -> Dict:
        """
        Measure the sensitivity of the model's output to perturbations in the input.
        
        Args:
            prompt: The input prompt to analyze
            target_tokens: List of tokens to measure sensitivity for
            perturbation_strength: How much to perturb the embeddings
            num_samples: Number of perturbation samples
            
        Returns:
            Dictionary with sensitivity scores for each token
        """
        # Tokenize the prompt and identify target token positions
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        token_positions = {}
        
        for target in target_tokens:
            target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
            positions = (token_ids == target_id).nonzero()
            if len(positions) > 0:
                token_positions[target] = positions[0].item()
        
        # Get original model output
        original_output = self.model.generate(prompt, max_length=50)
        
        # Measure sensitivity by perturbing embeddings
        sensitivities = {}
        for target, position in token_positions.items():
            # Get embeddings
            embeddings = self.model.get_embeddings(prompt)
            
            # Create perturbed versions
            outputs = []
            for _ in range(num_samples):
                # Apply perturbation to the specific token
                noise = torch.randn_like(embeddings[0][position]) * perturbation_strength
                perturbed_prompt = self._apply_embedding_perturbation(prompt, position, noise)
                outputs.append(self.model.generate(perturbed_prompt, max_length=50))
                
            # Calculate output differences
            differences = [self._output_difference(original_output, output) for output in outputs]
            sensitivities[target] = sum(differences) / len(differences)
            
        return sensitivities
    
    def _apply_embedding_perturbation(self, prompt, position, noise):
        """Apply perturbation to input at embedding level and return modified prompt."""
        # Note: In a real implementation, this would modify the embeddings
        # For demonstration, we use a simple word replacement as a proxy
        tokens = prompt.split()
        if position < len(tokens):
            tokens[position] = tokens[position] + " [perturbed]"
        return " ".join(tokens)
    
    def _output_difference(self, output1, output2):
        """Calculate difference between two outputs."""
        # Using simple string difference for demonstration
        # In practice, use semantic similarity or token-level difference
        return 1 - len(set(output1.split()) & set(output2.split())) / len(set(output1.split()) | set(output2.split()))
    
    def generate_counterfactuals(self, prompt: str, num_counterfactuals: int = 3) -> List[Dict]:
        """
        Generate counterfactual examples by systematically modifying the input.
        
        Args:
            prompt: The input prompt
            num_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            List of dictionaries containing counterfactuals and explanations
        """
        original_output = self.model.generate(prompt, max_length=50)
        
        # Identify most important tokens
        importance_scores = self.token_importance(prompt)
        
        # Generate counterfactuals by modifying key tokens
        counterfactuals = []
        top_tokens = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (token, _) in enumerate(top_tokens[:num_counterfactuals]):
            modified_prompt = self._modify_token(prompt, token)
            counterfactual_output = self.model.generate(modified_prompt, max_length=50)
            
            # Only add if output is substantially different
            if self._output_difference(original_output, counterfactual_output) > 0.3:
                counterfactuals.append({
                    "original_prompt": prompt,
                    "modified_prompt": modified_prompt,
                    "original_output": original_output,
                    "counterfactual_output": counterfactual_output,
                    "modified_token": token,
                    "explanation": f"Changing '{token}' significantly altered the output."
                })
                
        return counterfactuals
    
    def _modify_token(self, prompt, token):
        """Replace a token with alternatives."""
        # This is a simplified implementation
        # A complete implementation would use synonyms, antonyms, etc.
        replacements = {
            "good": "bad",
            "bad": "good",
            "positive": "negative",
            "negative": "positive",
            "excellent": "terrible",
            "terrible": "excellent",
            "love": "hate",
            "hate": "love"
        }
        
        if token.lower() in replacements:
            replacement = replacements[token.lower()]
            return prompt.replace(token, replacement)
        else:
            # Try to negate or modify the token
            return prompt.replace(token, "not " + token)
    
    def token_importance(self, prompt: str) -> Dict[str, float]:
        """
        Calculate the importance of each token in the input.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Dictionary mapping tokens to importance scores
        """
        tokens = self.tokenizer.tokenize(prompt)
        importance_scores = {}
        
        original_output = self.model.generate(prompt, max_length=50)
        
        # Measure importance by ablating one token at a time
        for i, token in enumerate(tokens):
            ablated_tokens = tokens.copy()
            ablated_tokens[i] = "[MASK]"
            ablated_prompt = self.tokenizer.convert_tokens_to_string(ablated_tokens)
            
            ablated_output = self.model.generate(ablated_prompt, max_length=50)
            importance = self._output_difference(original_output, ablated_output)
            importance_scores[token] = importance
            
        return importance_scores
    
    def visualize_token_importance(self, prompt: str, ax=None):
        """
        Visualize the importance of each token in the input.
        
        Args:
            prompt: The input prompt
            ax: Optional matplotlib axis for plotting
            
        Returns:
            Matplotlib axis object
        """
        importance_scores = self.token_importance(prompt)
        tokens = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
            
        colors = ["#ff9999" if score > 0.5 else "#66b3ff" for score in scores]
        ax.bar(range(len(tokens)), scores, color=colors)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_ylabel("Importance Score")
        ax.set_title("Token Importance for Model Output")
        
        return ax
    
    def analyze_internal_representations(self, prompt: str, layer_indices: List[int] = None) -> Dict:
        """
        Analyze internal representations of the model for the given input.
        
        Args:
            prompt: The input prompt
            layer_indices: Indices of layers to analyze. If None, analyze all layers.
            
        Returns:
            Dictionary with analysis results
        """
        # Get attention patterns
        attention = self.model.get_attention(prompt)
        
        # If layer_indices not specified, use all layers
        if layer_indices is None:
            layer_indices = list(range(len(attention)))
            
        results = {
            "attention_entropy": {},
            "attention_focus": {}
        }
        
        # Calculate attention entropy and focus for specified layers
        tokens = self.tokenizer.tokenize(prompt)
        for layer_idx in layer_indices:
            layer_attention = attention[layer_idx]
            
            # Attention entropy (higher means more uniform attention)
            entropy = -np.sum(layer_attention * np.log2(layer_attention + 1e-10), axis=-1).mean()
            results["attention_entropy"][f"layer_{layer_idx}"] = entropy
            
            # Attention focus (position with highest attention weight)
            focus_positions = layer_attention.argmax(axis=-1)
            if len(focus_positions) > 0 and len(tokens) > 0:
                focus_tokens = [tokens[pos] if pos < len(tokens) else "[PAD]" for pos in focus_positions]
                results["attention_focus"][f"layer_{layer_idx}"] = focus_tokens
            
        return results
    
    def visualize_attention_patterns(self, prompt: str, layer_idx: int = 0, head_idx: int = 0):
        """
        Visualize attention patterns for a specific layer and head.
        
        Args:
            prompt: The input prompt
            layer_idx: Layer index to visualize
            head_idx: Attention head index to visualize
            
        Returns:
            Matplotlib figure
        """
        attention = self.model.get_attention(prompt)
        tokens = self.tokenizer.tokenize(prompt)
        
        if len(attention) <= layer_idx:
            raise ValueError(f"Layer index {layer_idx} out of range (max: {len(attention)-1})")
            
        layer_attention = attention[layer_idx]
        if head_idx is not None and head_idx < layer_attention.shape[0]:
            attn_matrix = layer_attention[head_idx]
        else:
            # Average over all heads
            attn_matrix = layer_attention.mean(axis=0)
            
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attn_matrix[:len(tokens), :len(tokens)], 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap="viridis",
                   ax=ax)
        
        ax.set_title(f"Attention Pattern (Layer {layer_idx}, {'Average' if head_idx is None else f'Head {head_idx}'})")
        plt.tight_layout()
        
        return fig
    
    def explain_generation(self, prompt: str, max_length: int = 50) -> Dict:
        """
        Provide a comprehensive explanation of the model's generation process.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of generated output
            
        Returns:
            Dictionary with explanations
        """
        # Generate output
        output = self.model.generate(prompt, max_length=max_length)
        
        # Analyze token importance
        importance_scores = self.token_importance(prompt)
        top_tokens = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Analyze internal representations
        internal_analysis = self.analyze_internal_representations(prompt)
        
        # Generate counterfactuals
        counterfactuals = self.generate_counterfactuals(prompt, num_counterfactuals=2)
        
        # Compile explanation
        explanation = {
            "input": prompt,
            "output": output,
            "key_input_tokens": top_tokens,
            "attention_analysis": internal_analysis,
            "counterfactual_examples": counterfactuals,
            "summary": f"The model generated this output primarily focusing on {[t[0] for t in top_tokens]}. "
                      f"Changing these tokens would likely result in different outputs as shown in "
                      f"the counterfactual examples."
        }
        
        return explanation 