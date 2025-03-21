"""
Model Loader module for the LingualLens framework.

This module provides a unified interface for loading and interacting with 
different types of language models.
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

class ModelWrapper:
    """A wrapper class for different types of language models."""
    
    def __init__(self, model_name, task="base", device=None):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name or path of the model to load
            task: The task the model is intended for ("base", "causal_lm", etc.)
            device: Device to run the model on ("cpu", "cuda", etc.)
        """
        self.model_name = model_name
        self.task = task
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load appropriate model based on task
        if task == "base":
            self.model = AutoModel.from_pretrained(model_name)
        elif task == "causal_lm":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task: {task}")
            
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_embeddings(self, text):
        """
        Get embeddings for the input text.
        
        Args:
            text: Input text to get embeddings for
            
        Returns:
            Embeddings tensor
        """
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract embeddings (last hidden state for the [CLS] token)
        embeddings = outputs.last_hidden_state[0, 0].cpu().numpy()
        
        return embeddings
    
    def get_attention(self, text):
        """
        Get attention patterns for the input text.
        
        Args:
            text: Input text to get attention patterns for
            
        Returns:
            Attention patterns tensor or None if not available
        """
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Extract attention patterns if available
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            # Convert to numpy for easier processing
            attentions = [layer_attention.cpu().numpy() for layer_attention in outputs.attentions]
            return attentions
        else:
            return None
            
    def predict(self, text):
        """
        Make a prediction for the input text.
        
        Args:
            text: Input text to make prediction for
            
        Returns:
            Prediction result (depends on model type)
        """
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Return appropriate prediction based on model type
        if self.task == "causal_lm":
            # For language models, return the predicted token probabilities
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            return next_token_probs.cpu().numpy()
        else:
            # For other models, return the pooled output if available
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output.cpu().numpy()
            else:
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
    def generate(self, prompt, max_length=50, num_return_sequences=1, **kwargs):
        """
        Generate text from a prompt (for language models).
        
        Args:
            prompt: Input prompt to generate text from
            max_length: Maximum length of generated sequence
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional arguments for the generate method
            
        Returns:
            Generated text
        """
        if self.task != "causal_lm" and not hasattr(self.model, "generate"):
            # For non-generative models, simulate generation with simple prediction
            return prompt + " [Model output not available - not a generative model]"
            
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
                
            # Decode and return generated text
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            # Fall back to simple prediction if generation fails
            return f"[Generation failed: {str(e)}] {prompt}"
    
    def __repr__(self):
        """String representation of the model wrapper."""
        return f"ModelWrapper(model_name='{self.model_name}', task='{self.task}', device='{self.device}')" 