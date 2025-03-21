"""
Model loader module for the Language Model Evaluation Framework.

This module provides functionality for loading and interfacing with various
language models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification


class ModelWrapper:
    """
    Wrapper class for language models that provides a unified interface.
    
    This class wraps various types of models (transformers, spaCy, etc.)
    to provide a consistent interface for the evaluation framework.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        model_type: str = "transformer",
        task: str = "base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name_or_path: Model name (from HuggingFace) or path to model
            model_type: Type of model to load (transformer, spacy, etc.)
            task: Task the model is designed for (text-generation, classification, base)
            device: Device to load the model on (cuda, cpu)
            **kwargs: Additional keyword arguments for model loading
        """
        self.model_name = model_name_or_path
        self.model_type = model_type
        self.task = task
        self.device = device
        self.kwargs = kwargs
        
        # Load the appropriate model and tokenizer
        if model_type == "transformer":
            self._load_transformer_model(model_name_or_path, task, device, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _load_transformer_model(
        self, model_name_or_path: str, task: str, device: str, **kwargs
    ):
        """
        Load a transformer model from HuggingFace.
        
        Args:
            model_name_or_path: Model name or path
            task: Model task
            device: Device to load on
            **kwargs: Additional loading arguments
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        
        # Load appropriate model based on task
        if task == "text-generation":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **kwargs
            ).to(device)
        elif task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, **kwargs
            ).to(device)
        else:
            # For general purpose, load base model
            self.model = AutoModel.from_pretrained(
                model_name_or_path, **kwargs
            ).to(device)
            
        # Set model to evaluation mode
        self.model.eval()

    def predict(self, text: str) -> Any:
        """
        Make a prediction based on the model's task.
        
        Args:
            text: The input text
            
        Returns:
            Model prediction output (varies by task)
        """
        if self.task == "text-generation":
            return self.generate(text)
        elif self.task == "classification":
            return self.classify(text)
        else:
            # Base task just returns embeddings
            return self.get_embeddings(text)

    def generate(
        self, prompt: str, max_length: int = 50, temperature: float = 1.0, **kwargs
    ) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        if self.task != "text-generation":
            raise ValueError("This model is not configured for text generation")
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
            
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def classify(self, text: str) -> Dict[str, float]:
        """
        Classify text into predefined categories.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        if self.task != "classification":
            raise ValueError("This model is not configured for classification")
            
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Convert to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Map to class names if available, otherwise use indices
        if hasattr(self.model.config, "id2label"):
            return {
                self.model.config.id2label[i]: p.item()
                for i, p in enumerate(probs)
            }
        else:
            return {str(i): p.item() for i, p in enumerate(probs)}

    def get_embeddings(
        self, text: str, layer: int = -1, pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get embeddings for input text.
        
        Args:
            text: Input text
            layer: Which layer to extract embeddings from (-1 for last)
            pooling: How to pool embeddings (mean, cls, etc.)
            
        Returns:
            Tensor containing embeddings
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Get the specified layer
        hidden_states = outputs.hidden_states[layer]
        
        # Apply pooling
        if pooling == "mean":
            # Mean of all token embeddings
            return hidden_states[0].mean(dim=0)
        elif pooling == "cls":
            # Use the [CLS] token embedding
            return hidden_states[0][0]
        else:
            # Return all token embeddings
            return hidden_states[0]

    def get_attention(self, text: str) -> List[torch.Tensor]:
        """
        Get attention maps for input text.
        
        Args:
            text: Input text
            
        Returns:
            List of attention tensors for each layer
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Return attention maps
        return outputs.attentions 

    def get_attention_weights(self, text: str) -> List[torch.Tensor]:
        """
        Get attention weights for input text (alias for get_attention).
        
        Args:
            text: Input text
            
        Returns:
            List of attention tensors for each layer
        """
        return self.get_attention(text) 