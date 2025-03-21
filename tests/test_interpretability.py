"""
Test script for the interpretability module.

This script verifies the basic functionality of the interpretability module.
"""

import sys
import os
import unittest
import torch
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import load_model
from src.interpretability import InterpretabilityModule
from src.interpretability.attention_analysis import AttentionAnalyzer
from src.interpretability.feature_attribution import FeatureAttributor
from src.interpretability.concept_extraction import ConceptExtractor


class MockModelWrapper:
    """Mock model wrapper for testing."""
    
    def __init__(self):
        self.model_name = "mock_model"
        self.device = "cpu"
    
    def get_attention_weights(self, text):
        # Create a dummy attention tensor with shape [layers, heads, seq_len, seq_len]
        # 2 layers, 2 heads, sequence length based on text length
        seq_len = max(len(text.split()), 2)
        return [
            torch.rand(2, seq_len, seq_len),
            torch.rand(2, seq_len, seq_len)
        ]
    
    def get_embeddings(self, text):
        # Create a dummy embeddings tensor with shape [seq_len, hidden_size]
        seq_len = max(len(text.split()), 2)
        return torch.rand(seq_len, 128)
    
    def tokenize(self, text):
        # Mock tokenization by simply splitting on whitespace
        return text.split()
    
    def generate(self, text, *args, **kwargs):
        # Mock text generation
        return text + " [mock generated text]"
    
    @property
    def tokenizer(self):
        # Mock tokenizer that simply returns this object (since we implemented tokenize)
        return self


class TestInterpretabilityModule(unittest.TestCase):
    """Test case for the InterpretabilityModule."""
    
    @classmethod
    def setUpClass(cls):
        # Set up a mock model for testing
        cls.model = MockModelWrapper()
    
    def test_initialization(self):
        """Test initialization of InterpretabilityModule."""
        # Initialize with default parameters
        module = InterpretabilityModule(self.model)
        self.assertEqual(module.model, self.model)
        self.assertListEqual(module.techniques, ["attention", "feature_attribution", "concept_extraction"])
        
        # Initialize with custom parameters
        module = InterpretabilityModule(self.model, techniques=["attention"])
        self.assertListEqual(module.techniques, ["attention"])
        self.assertIn("attention", module.analyzers)
        self.assertNotIn("feature_attribution", module.analyzers)
    
    def test_analyze(self):
        """Test analyze method."""
        # Initialize with a single technique for simplicity
        module = InterpretabilityModule(self.model, techniques=["attention"])
        
        # Test with a simple input
        text = "This is a test sentence"
        result = module.analyze(text)
        
        # Check that the result contains the expected keys
        self.assertIn("attention", result)
        self.assertIn("attention_weights", result["attention"])
        self.assertIn("tokens", result["attention"])
    
    def test_evaluate(self):
        """Test evaluate method."""
        # Initialize with a single technique for simplicity
        module = InterpretabilityModule(self.model, techniques=["attention"])
        
        # Test with a list of texts
        texts = ["This is a test", "Another example sentence"]
        result = module.evaluate(texts=texts)
        
        # Check that the result has the expected structure
        self.assertEqual(result.component_name, "interpretability")
        self.assertIsInstance(result.metrics, dict)
        self.assertIsInstance(result.examples, list)
        self.assertEqual(len(result.examples), len(texts))


class TestAttentionAnalyzer(unittest.TestCase):
    """Test case for the AttentionAnalyzer."""
    
    @classmethod
    def setUpClass(cls):
        # Set up a mock model for testing
        cls.model = MockModelWrapper()
    
    def test_initialization(self):
        """Test initialization of AttentionAnalyzer."""
        analyzer = AttentionAnalyzer(self.model)
        self.assertEqual(analyzer.model, self.model)
    
    def test_analyze(self):
        """Test analyze method."""
        analyzer = AttentionAnalyzer(self.model)
        
        # Test with a simple input
        text = "This is a test sentence"
        result = analyzer.analyze(text)
        
        # Check that the result contains the expected keys
        self.assertIn("attention_weights", result)
        self.assertIn("tokens", result)
        self.assertIn("statistics", result)
        self.assertIn("patterns", result)


class TestFeatureAttributor(unittest.TestCase):
    """Test case for the FeatureAttributor."""
    
    @classmethod
    def setUpClass(cls):
        # Set up a mock model for testing
        cls.model = MockModelWrapper()
    
    def test_initialization(self):
        """Test initialization of FeatureAttributor."""
        attributor = FeatureAttributor(self.model)
        self.assertEqual(attributor.model, self.model)
        self.assertEqual(attributor.method, "integrated_gradients")
    
    def test_analyze(self):
        """Test analyze method."""
        attributor = FeatureAttributor(self.model)
        
        # Test with a simple input
        text = "This is a test sentence"
        result = attributor.analyze(text)
        
        # Check that the result contains the expected keys
        self.assertIn("attributions", result)
        self.assertIn("tokens", result)
        self.assertIn("top_features", result)
        self.assertIn("statistics", result)


class TestConceptExtractor(unittest.TestCase):
    """Test case for the ConceptExtractor."""
    
    @classmethod
    def setUpClass(cls):
        # Set up a mock model for testing
        cls.model = MockModelWrapper()
    
    def test_initialization(self):
        """Test initialization of ConceptExtractor."""
        extractor = ConceptExtractor(self.model)
        self.assertEqual(extractor.model, self.model)
        self.assertEqual(extractor.method, "clustering")
        self.assertEqual(extractor.num_concepts, 10)
    
    def test_analyze(self):
        """Test analyze method."""
        extractor = ConceptExtractor(self.model)
        
        # Test with a simple input
        text = "This is a test sentence"
        result = extractor.analyze(text)
        
        # Check that the result contains the expected keys
        self.assertIn("concepts", result)
        self.assertIn("activations", result)
        self.assertIn("tokens", result)
        self.assertIn("top_concepts", result)
        self.assertIn("statistics", result)


if __name__ == "__main__":
    unittest.main() 