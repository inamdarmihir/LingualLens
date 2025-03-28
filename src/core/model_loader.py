"""
Model Loader module for the LingualLens framework.

This module provides a unified interface for loading and interacting with 
different types of language models from various providers including Hugging Face,
OpenAI, Google, and others.
"""

import torch
import numpy as np
import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    """Enum for supported model providers."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    AZURE = "azure_openai"
    CUSTOM = "custom"

class ModelWrapper:
    """A wrapper class for different types of language models."""
    
    def __init__(self, model_name, task="base", device=None, provider=ModelProvider.HUGGINGFACE, 
                 api_key=None, api_version=None, api_base=None, options=None):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name or path of the model to load
            task: The task the model is intended for ("base", "causal_lm", "seq2seq", etc.)
            device: Device to run the model on ("cpu", "cuda", etc.)
            provider: Provider of the model (huggingface, openai, google, etc.)
            api_key: API key for cloud-based models (optional)
            api_version: API version for cloud-based models (optional)
            api_base: API base URL for cloud-based models (optional)
            options: Additional provider-specific options (optional)
        """
        self.model_name = model_name
        self.task = task
        self.provider = provider if isinstance(provider, ModelProvider) else ModelProvider(provider)
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.options = options or {}
        
        # Client for API-based models
        self._client = None
        self._model = None
        self._tokenizer = None
        
        # Load API keys from environment variables if not provided
        if self.provider in [ModelProvider.OPENAI, ModelProvider.AZURE]:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("No OpenAI API key provided, set OPENAI_API_KEY environment variable")
        elif self.provider == ModelProvider.GOOGLE:
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                logger.warning("No Google API key provided, set GOOGLE_API_KEY environment variable")
        elif self.provider == ModelProvider.ANTHROPIC:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                logger.warning("No Anthropic API key provided, set ANTHROPIC_API_KEY environment variable")
                
        self.api_version = api_version
        self.api_base = api_base
        
        # Initialize the appropriate model based on provider
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model based on the provider and task."""
        if self.provider == ModelProvider.HUGGINGFACE:
            self._initialize_huggingface_model()
        elif self.provider == ModelProvider.OPENAI:
            self._initialize_openai_client()
        elif self.provider == ModelProvider.GOOGLE:
            self._initialize_google_client()
        elif self.provider == ModelProvider.ANTHROPIC:
            self._initialize_anthropic_client()
        elif self.provider == ModelProvider.AZURE:
            self._initialize_azure_openai_client()
        elif self.provider == ModelProvider.CUSTOM:
            self._initialize_custom_model()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _initialize_huggingface_model(self):
        """Initialize a Hugging Face model."""
        try:
            # Load appropriate model based on task
            if self.task == "base":
                self._model = AutoModel.from_pretrained(self.model_name, **self.options)
            elif self.task == "causal_lm":
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.options)
            elif self.task == "seq2seq":
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **self.options)
            else:
                raise ValueError(f"Unsupported task for Hugging Face: {self.task}")
                
            self._model.to(self.device)
            self._model.eval()  # Set to evaluation mode
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Make sure padding token is set for generation
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Error initializing Hugging Face model: {str(e)}")
            raise
    
    def _initialize_openai_client(self):
        """Initialize an OpenAI client."""
        try:
            import openai
            if self.api_key:
                # Set default API key
                openai.api_key = self.api_key
                
                # Set API base if provided
                if self.api_base:
                    openai.api_base = self.api_base
                    
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
            else:
                logger.warning("OpenAI API key not provided, client initialization skipped")
                
        except ImportError:
            logger.error("OpenAI Python package not installed. Install with 'pip install openai'")
            raise
    
    def _initialize_azure_openai_client(self):
        """Initialize an Azure OpenAI client."""
        try:
            import openai
            if self.api_key and self.api_base:
                # Set up Azure-specific configuration
                openai.api_key = self.api_key
                openai.api_base = self.api_base
                openai.api_type = "azure"
                openai.api_version = self.api_version or "2023-05-15"
                
                self._client = openai.AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.api_base,
                    api_version=self.api_version or "2023-05-15"
                )
            else:
                logger.warning("Azure OpenAI API key or endpoint not provided, client initialization skipped")
                
        except ImportError:
            logger.error("OpenAI Python package not installed. Install with 'pip install openai'")
            raise
    
    def _initialize_google_client(self):
        """Initialize a Google client."""
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._client = genai
            else:
                logger.warning("Google API key not provided, client initialization skipped")
                
        except ImportError:
            logger.error("Google Generative AI package not installed. Install with 'pip install google-generativeai'")
            raise
    
    def _initialize_anthropic_client(self):
        """Initialize an Anthropic client."""
        try:
            import anthropic
            if self.api_key:
                self._client = anthropic.Anthropic(api_key=self.api_key)
            else:
                logger.warning("Anthropic API key not provided, client initialization skipped")
                
        except ImportError:
            logger.error("Anthropic Python package not installed. Install with 'pip install anthropic'")
            raise
    
    def _initialize_custom_model(self):
        """Initialize a custom model."""
        # Implement custom model initialization logic
        logger.info("Custom model initialization - implement specific logic for your model")
        
        # Example: load a custom model from a specified file path
        if "model_path" in self.options and "tokenizer_path" in self.options:
            try:
                # Load model
                if self.task == "causal_lm":
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.options["model_path"], 
                        device_map=self.device
                    )
                else:
                    self._model = AutoModel.from_pretrained(
                        self.options["model_path"], 
                        device_map=self.device
                    )
                    
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.options["tokenizer_path"]
                )
            except Exception as e:
                logger.error(f"Error loading custom model: {str(e)}")
                raise
    
    @property
    def model(self) -> Optional[PreTrainedModel]:
        """Get the underlying model (for Hugging Face models)."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Get the underlying tokenizer (for Hugging Face models)."""
        return self._tokenizer
        
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings for the input text.
        
        Args:
            text: Input text to get embeddings for
            
        Returns:
            Embeddings as numpy array
        """
        if self.provider == ModelProvider.HUGGINGFACE:
            # Local model embedding generation
            if not self._model or not self._tokenizer:
                raise ValueError("Model or tokenizer not initialized")
                
            # Tokenize text
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self._model(**inputs)
                
            # Extract embeddings (last hidden state for the first token)
            embeddings = outputs.last_hidden_state[0, 0].cpu().numpy()
            return embeddings
            
        elif self.provider == ModelProvider.OPENAI:
            # OpenAI API embedding generation
            if not self._client:
                raise ValueError("OpenAI client not initialized")
                
            try:
                response = self._client.embeddings.create(
                    model="text-embedding-ada-002",  # Can be parameterized
                    input=text
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error getting OpenAI embeddings: {str(e)}")
                raise
                
        elif self.provider == ModelProvider.GOOGLE:
            # Google API embedding generation
            if not self._client:
                raise ValueError("Google client not initialized")
                
            try:
                embedding_model = self._client.get_model("embedding-001")  # Can be parameterized
                result = embedding_model.embed_content(content=text)
                return np.array(result["embedding"])
            except Exception as e:
                logger.error(f"Error getting Google embeddings: {str(e)}")
                raise
                
        else:
            logger.warning(f"Embeddings not supported for provider {self.provider}")
            # Return a dummy embedding for unsupported providers
            return np.zeros(768)  # Standard size for many embedding models
    
    def get_attention(self, text: str) -> Optional[List[np.ndarray]]:
        """
        Get attention patterns for the input text.
        
        Args:
            text: Input text to get attention patterns for
            
        Returns:
            Attention patterns as list of numpy arrays or None if not available
        """
        if self.provider != ModelProvider.HUGGINGFACE:
            logger.warning(f"Attention patterns not available for provider {self.provider}")
            return None
            
        if not self._model or not self._tokenizer:
            raise ValueError("Model or tokenizer not initialized")
            
        # Tokenize text
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
            
        # Extract attention patterns if available
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            # Convert to numpy for easier processing
            attentions = [layer_attention.cpu().numpy() for layer_attention in outputs.attentions]
            return attentions
        else:
            return None
            
    def predict(self, text: str) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Make a prediction for the input text.
        
        Args:
            text: Input text to make prediction for
            
        Returns:
            Prediction result (depends on model type and provider)
        """
        if self.provider == ModelProvider.HUGGINGFACE:
            if not self._model or not self._tokenizer:
                raise ValueError("Model or tokenizer not initialized")
                
            # Tokenize text
            inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self._model(**inputs)
                
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
                    
        elif self.provider == ModelProvider.OPENAI:
            if not self._client:
                raise ValueError("OpenAI client not initialized")
                
            try:
                response = self._client.completions.create(
                    model=self.model_name,
                    prompt=text,
                    max_tokens=1,
                    logprobs=5,
                    echo=False
                )
                return {
                    "text": response.choices[0].text,
                    "logprobs": response.choices[0].logprobs
                }
            except Exception as e:
                logger.error(f"Error getting OpenAI prediction: {str(e)}")
                return {"error": str(e)}
                
        elif self.provider == ModelProvider.GOOGLE:
            if not self._client:
                raise ValueError("Google client not initialized")
                
            try:
                model = self._client.GenerativeModel(self.model_name)
                response = model.generate_content(text)
                return {
                    "text": response.text,
                    "candidates": [c.content.text for c in response.candidates] if hasattr(response, "candidates") else []
                }
            except Exception as e:
                logger.error(f"Error getting Google prediction: {str(e)}")
                return {"error": str(e)}
                
        elif self.provider == ModelProvider.ANTHROPIC:
            if not self._client:
                raise ValueError("Anthropic client not initialized")
                
            try:
                response = self._client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=1
                )
                return {
                    "text": response.content[0].text,
                    "model": response.model,
                    "stop_reason": response.stop_reason
                }
            except Exception as e:
                logger.error(f"Error getting Anthropic prediction: {str(e)}")
                return {"error": str(e)}
                
        else:
            logger.warning(f"Prediction not fully supported for provider {self.provider}")
            return {"warning": f"Prediction not fully supported for provider {self.provider}"}
                
    def generate(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1, 
                 temperature: float = 1.0, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt to generate text from
            max_length: Maximum length of generated sequence
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature (higher = more random)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text or dict with generation results
        """
        if self.provider == ModelProvider.HUGGINGFACE:
            if not self._model or not self._tokenizer:
                raise ValueError("Model or tokenizer not initialized")
                
            if self.task != "causal_lm" and self.task != "seq2seq" and not hasattr(self._model, "generate"):
                # For non-generative models, simulate generation with simple prediction
                return prompt + " [Model output not available - not a generative model]"
                
            # Tokenize prompt
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            try:
                with torch.no_grad():
                    outputs = self._model.generate(
                        inputs.input_ids,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,
                        temperature=temperature,
                        pad_token_id=self._tokenizer.eos_token_id,
                        **kwargs
                    )
                    
                # Decode and return generated text
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Error generating text with Hugging Face model: {str(e)}")
                return f"[Generation failed: {str(e)}] {prompt}"
                
        elif self.provider == ModelProvider.OPENAI:
            if not self._client:
                raise ValueError("OpenAI client not initialized")
                
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    n=num_return_sequences,
                    temperature=temperature,
                    **kwargs
                )
                if num_return_sequences == 1:
                    return response.choices[0].message.content
                else:
                    return {
                        "choices": [choice.message.content for choice in response.choices],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
            except Exception as e:
                logger.error(f"Error generating text with OpenAI model: {str(e)}")
                return f"[Generation failed: {str(e)}]"
                
        elif self.provider == ModelProvider.GOOGLE:
            if not self._client:
                raise ValueError("Google client not initialized")
                
            try:
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_length,
                    "candidate_count": num_return_sequences
                }
                
                # Add any additional configuration from kwargs
                for key, value in kwargs.items():
                    if key not in generation_config:
                        generation_config[key] = value
                
                model = self._client.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
                
                response = model.generate_content(prompt)
                
                if num_return_sequences == 1:
                    return response.text
                else:
                    return {
                        "choices": [candidate.content.text for candidate in response.candidates],
                        "prompt_feedback": response.prompt_feedback if hasattr(response, "prompt_feedback") else None
                    }
            except Exception as e:
                logger.error(f"Error generating text with Google model: {str(e)}")
                return f"[Generation failed: {str(e)}]"
                
        elif self.provider == ModelProvider.ANTHROPIC:
            if not self._client:
                raise ValueError("Anthropic client not initialized")
                
            try:
                response = self._client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=temperature,
                    **kwargs
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Error generating text with Anthropic model: {str(e)}")
                return f"[Generation failed: {str(e)}]"
                
        else:
            logger.warning(f"Generation not fully supported for provider {self.provider}")
            return f"[Generation not supported for provider {self.provider}]"
    
    def __repr__(self):
        """String representation of the model wrapper."""
        return f"ModelWrapper(model_name='{self.model_name}', provider='{self.provider}', task='{self.task}', device='{self.device}')"
