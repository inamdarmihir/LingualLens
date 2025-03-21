"""
Model comparison module for the LingualLens framework.

This module provides tools for comparing behavior and outputs of different language models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .model_loader import ModelWrapper

class ModelComparator:
    """Compare behavior and outputs of different language models."""
    
    def __init__(self, models: List[ModelWrapper]):
        """
        Initialize model comparator.
        
        Args:
            models: List of ModelWrapper instances to compare
        """
        self.models = models
        self.model_names = [model.model_name for model in models]
    
    def compare_outputs(self, texts: List[str], max_length: int = 50) -> Dict:
        """
        Compare model outputs on the same input texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum generation length
            
        Returns:
            Dictionary with output comparisons
        """
        results = []
        
        for text in texts:
            # Get outputs from all models
            outputs = []
            for model in self.models:
                output = model.generate(text, max_length=max_length)
                outputs.append(output)
            
            # Calculate pairwise similarity
            similarities = np.zeros((len(self.models), len(self.models)))
            for i in range(len(self.models)):
                for j in range(len(self.models)):
                    if i == j:
                        similarities[i, j] = 1.0
                    else:
                        # Simple Jaccard similarity
                        set_i = set(outputs[i].lower().split())
                        set_j = set(outputs[j].lower().split())
                        
                        intersection = len(set_i.intersection(set_j))
                        union = len(set_i.union(set_j))
                        
                        similarities[i, j] = intersection / union if union > 0 else 0.0
            
            results.append({
                "input": text,
                "outputs": outputs,
                "similarities": similarities.tolist()
            })
        
        return {
            "model_names": self.model_names,
            "results": results
        }
    
    def compare_embeddings(self, texts: List[str]) -> Dict:
        """
        Compare embeddings from different models.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with embedding comparisons
        """
        results = []
        
        for text in texts:
            # Get embeddings from all models
            embeddings = []
            for model in self.models:
                embedding = model.get_embeddings(text)
                embeddings.append(embedding)
            
            # Calculate pairwise cosine similarity
            similarities = np.zeros((len(self.models), len(self.models)))
            for i in range(len(self.models)):
                for j in range(len(self.models)):
                    if i == j:
                        similarities[i, j] = 1.0
                    else:
                        # Cosine similarity
                        dot_product = np.dot(embeddings[i], embeddings[j])
                        norm_i = np.linalg.norm(embeddings[i])
                        norm_j = np.linalg.norm(embeddings[j])
                        
                        similarities[i, j] = dot_product / (norm_i * norm_j) if norm_i * norm_j > 0 else 0.0
            
            results.append({
                "input": text,
                "similarities": similarities.tolist()
            })
        
        return {
            "model_names": self.model_names,
            "results": results
        }
    
    def compare_robustness(self, texts: List[str], perturbation_types: List[str] = None) -> Dict:
        """
        Compare model robustness to various perturbations.
        
        Args:
            texts: List of input texts
            perturbation_types: Types of perturbations to apply
            
        Returns:
            Dictionary with robustness comparisons
        """
        if perturbation_types is None:
            perturbation_types = ['typos', 'synonyms', 'paraphrase']
        
        results = []
        
        for text in texts:
            # Generate perturbed texts
            perturbed_texts = self._generate_perturbations(text, perturbation_types)
            
            # Compare model outputs on original vs perturbed
            model_results = []
            
            for model_idx, model in enumerate(self.models):
                # Get original output
                original_output = model.generate(text)
                
                # Get outputs for perturbed texts
                perturbed_outputs = {}
                similarity_scores = {}
                
                for perturb_type, perturbed_text in perturbed_texts.items():
                    perturbed_output = model.generate(perturbed_text)
                    perturbed_outputs[perturb_type] = perturbed_output
                    
                    # Calculate similarity between original and perturbed output
                    set_orig = set(original_output.lower().split())
                    set_perturb = set(perturbed_output.lower().split())
                    
                    intersection = len(set_orig.intersection(set_perturb))
                    union = len(set_orig.union(set_perturb))
                    
                    similarity = intersection / union if union > 0 else 0.0
                    similarity_scores[perturb_type] = similarity
                
                model_results.append({
                    "model_name": self.model_names[model_idx],
                    "original_output": original_output,
                    "perturbed_outputs": perturbed_outputs,
                    "similarity_scores": similarity_scores,
                    "average_robustness": np.mean(list(similarity_scores.values()))
                })
            
            results.append({
                "input": text,
                "perturbed_texts": perturbed_texts,
                "model_results": model_results
            })
        
        return {
            "perturbation_types": perturbation_types,
            "model_names": self.model_names,
            "results": results
        }
    
    def visualize_output_similarity(self, comparison_results: Dict):
        """
        Visualize output similarity between models.
        
        Args:
            comparison_results: Results from compare_outputs
            
        Returns:
            Matplotlib figure
        """
        n_models = len(self.model_names)
        
        # Average similarities across all inputs
        avg_similarities = np.zeros((n_models, n_models))
        n_inputs = len(comparison_results["results"])
        
        for result in comparison_results["results"]:
            avg_similarities += np.array(result["similarities"])
        
        avg_similarities /= n_inputs
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            avg_similarities, 
            annot=True, 
            cmap="YlGnBu", 
            xticklabels=self.model_names,
            yticklabels=self.model_names,
            ax=ax
        )
        ax.set_title("Average Output Similarity Between Models")
        plt.tight_layout()
        
        return fig
    
    def visualize_embedding_similarity(self, comparison_results: Dict):
        """
        Visualize embedding similarity between models.
        
        Args:
            comparison_results: Results from compare_embeddings
            
        Returns:
            Matplotlib figure
        """
        n_models = len(self.model_names)
        
        # Average similarities across all inputs
        avg_similarities = np.zeros((n_models, n_models))
        n_inputs = len(comparison_results["results"])
        
        for result in comparison_results["results"]:
            avg_similarities += np.array(result["similarities"])
        
        avg_similarities /= n_inputs
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            avg_similarities, 
            annot=True, 
            cmap="YlGnBu", 
            xticklabels=self.model_names,
            yticklabels=self.model_names,
            ax=ax
        )
        ax.set_title("Average Embedding Similarity Between Models")
        plt.tight_layout()
        
        return fig
    
    def visualize_robustness_comparison(self, comparison_results: Dict):
        """
        Visualize robustness comparison between models.
        
        Args:
            comparison_results: Results from compare_robustness
            
        Returns:
            Matplotlib figure
        """
        model_names = comparison_results["model_names"]
        perturbation_types = comparison_results["perturbation_types"]
        
        # Extract average robustness scores per model and perturbation type
        robustness_scores = {model_name: {p_type: [] for p_type in perturbation_types} 
                           for model_name in model_names}
        avg_scores = {model_name: 0.0 for model_name in model_names}
        
        for result in comparison_results["results"]:
            for model_result in result["model_results"]:
                model_name = model_result["model_name"]
                
                for p_type, score in model_result["similarity_scores"].items():
                    robustness_scores[model_name][p_type].append(score)
                
                avg_scores[model_name] += model_result["average_robustness"]
        
        # Average scores across all inputs
        n_inputs = len(comparison_results["results"])
        for model_name in avg_scores:
            avg_scores[model_name] /= n_inputs
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot average scores
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, p_type in enumerate(perturbation_types):
            values = [np.mean(robustness_scores[model_name][p_type]) for model_name in model_names]
            ax.bar(x + (i - len(perturbation_types)/2 + 0.5) * width, values, 
                  width, label=p_type)
        
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Robustness Score (higher is better)")
        ax.set_title("Robustness Comparison Across Models")
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        return fig
    
    def _generate_perturbations(self, text: str, perturbation_types: List[str]) -> Dict:
        """Generate perturbed versions of the input text."""
        perturbed_texts = {}
        
        for perturb_type in perturbation_types:
            if perturb_type == 'typos':
                # Add random typos
                words = text.split()
                if words:
                    idx = np.random.randint(0, len(words))
                    word = words[idx]
                    if len(word) > 2:
                        # Swap two adjacent characters
                        char_idx = np.random.randint(0, len(word) - 1)
                        new_word = word[:char_idx] + word[char_idx + 1] + word[char_idx] + word[char_idx + 2:]
                        words[idx] = new_word
                        perturbed_texts['typos'] = ' '.join(words)
                    else:
                        perturbed_texts['typos'] = text  # No change
                else:
                    perturbed_texts['typos'] = text  # No change
                    
            elif perturb_type == 'synonyms':
                # Replace a word with synonym
                words = text.split()
                if words:
                    idx = np.random.randint(0, len(words))
                    word = words[idx]
                    
                    # Very simple synonym mapping (would use WordNet in practice)
                    synonyms = {
                        'good': ['great', 'excellent', 'fine', 'nice'],
                        'bad': ['poor', 'terrible', 'awful', 'unpleasant'],
                        'big': ['large', 'huge', 'enormous', 'massive'],
                        'small': ['tiny', 'little', 'miniature', 'compact'],
                        'happy': ['glad', 'joyful', 'pleased', 'delighted'],
                        'sad': ['unhappy', 'sorrowful', 'depressed', 'gloomy'],
                        'beautiful': ['pretty', 'lovely', 'attractive', 'gorgeous'],
                        'ugly': ['unattractive', 'hideous', 'unsightly', 'plain'],
                    }
                    
                    if word.lower() in synonyms:
                        new_word = np.random.choice(synonyms[word.lower()])
                        words[idx] = new_word
                        perturbed_texts['synonyms'] = ' '.join(words)
                    else:
                        perturbed_texts['synonyms'] = text  # No change
                else:
                    perturbed_texts['synonyms'] = text  # No change
                    
            elif perturb_type == 'paraphrase':
                # Very simple rule-based paraphrasing
                paraphrase_patterns = [
                    (r'I am', 'I\'m'),
                    (r'You are', 'You\'re'),
                    (r'They are', 'They\'re'),
                    (r'We are', 'We\'re'),
                    (r'It is', 'It\'s'),
                    (r'do not', 'don\'t'),
                    (r'does not', 'doesn\'t'),
                    (r'did not', 'didn\'t'),
                    (r'is not', 'isn\'t'),
                    (r'are not', 'aren\'t'),
                ]
                
                perturbed_text = text
                for pattern, replacement in paraphrase_patterns:
                    if pattern in perturbed_text:
                        perturbed_text = perturbed_text.replace(pattern, replacement)
                        break
                    elif replacement in perturbed_text:
                        perturbed_text = perturbed_text.replace(replacement, pattern)
                        break
                
                if perturbed_text != text:
                    perturbed_texts['paraphrase'] = perturbed_text
                else:
                    # Add a filler word if no patterns matched
                    words = text.split()
                    filler_words = ['actually', 'basically', 'certainly', 'definitely', 'possibly']
                    if words:
                        filler = np.random.choice(filler_words)
                        insert_idx = np.random.randint(0, len(words))
                        words.insert(insert_idx, filler)
                        perturbed_texts['paraphrase'] = ' '.join(words)
                    else:
                        perturbed_texts['paraphrase'] = text  # No change
        
        return perturbed_texts 