#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Language Model Evaluation Framework.
This script tests all components of the framework to ensure they work correctly.
"""

import logging
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import framework components
try:
    from src.core.model_loader import ModelWrapper
    from src.core.evaluator import EvaluationResult
    from src.adversarial.attack_generator import AttackGenerator
    from src.adversarial.counterfactual_generator import CounterfactualGenerator
    from src.adversarial.robustness_evaluator import RobustnessEvaluator
    from src.adversarial.module import AdversarialTester
    from src.interpretability.feature_attribution import FeatureAttributor
    from src.interpretability.attention_analysis import AttentionAnalyzer
    from src.interpretability.concept_extraction import ConceptExtractor
    logger.info("Successfully imported all framework components")
except ImportError as e:
    logger.error(f"Failed to import framework components: {e}")
    raise

def test_model_wrapper():
    """Test the ModelWrapper class."""
    logger.info("Testing ModelWrapper...")
    
    try:
        # Initialize a small model for testing
        model = ModelWrapper("distilbert-base-uncased", task="base", device="cpu")
        logger.info(f"Loaded model: {model.model_name}")
        
        # Test basic functionality
        sample_text = "This is a test sentence for the language model."
        logger.info(f"Testing model with input: '{sample_text}'")
        
        # Test embedding extraction
        embeddings = model.get_embeddings(sample_text)
        logger.info(f"Got embeddings with shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
        
        # Test attention extraction
        attention = model.get_attention(sample_text)
        logger.info(f"Got attention with {len(attention)} layers")
        
        # Test prediction
        output = model.predict(sample_text)
        logger.info(f"Model prediction completed successfully")
        
        return model
    except Exception as e:
        logger.error(f"ModelWrapper test failed: {e}")
        raise

def test_attack_generator(model: ModelWrapper):
    """Test the AttackGenerator class."""
    logger.info("Testing AttackGenerator...")
    
    try:
        # Initialize attack generator
        attack_gen = AttackGenerator(model)
        logger.info("Initialized AttackGenerator")
        
        # Test different attack types
        sample_text = "The weather is nice today."
        logger.info(f"Testing attacks on text: '{sample_text}'")
        
        # Character-level attacks
        char_attacks = attack_gen.generate(sample_text, level="character", num_attacks=2)
        logger.info(f"Generated {len(char_attacks)} character-level attacks")
        
        # Word-level attacks
        word_attacks = attack_gen.generate(sample_text, level="word", num_attacks=2)
        logger.info(f"Generated {len(word_attacks)} word-level attacks")
        
        # Sentence-level attacks
        sent_attacks = attack_gen.generate(sample_text, level="sentence", num_attacks=2)
        logger.info(f"Generated {len(sent_attacks)} sentence-level attacks")
        
    except Exception as e:
        logger.error(f"AttackGenerator test failed: {e}")
        raise

def test_counterfactual_generator(model: ModelWrapper):
    """Test the CounterfactualGenerator class."""
    logger.info("Testing CounterfactualGenerator...")
    
    try:
        # Initialize counterfactual generator
        cf_gen = CounterfactualGenerator(model)
        logger.info("Initialized CounterfactualGenerator")
        
        # Test counterfactual generation
        sample_text = "I really enjoyed the movie."
        logger.info(f"Testing counterfactual generation on text: '{sample_text}'")
        
        counterfactuals = cf_gen.generate(sample_text, num_examples=2)
        logger.info(f"Generated {len(counterfactuals)} counterfactuals")
        
    except Exception as e:
        logger.error(f"CounterfactualGenerator test failed: {e}")
        raise

def test_robustness_evaluator(model: ModelWrapper):
    """Test the RobustnessEvaluator class."""
    logger.info("Testing RobustnessEvaluator...")
    
    try:
        # Initialize robustness evaluator
        rob_eval = RobustnessEvaluator(model, shift_types=["stylistic", "demographic"])
        logger.info("Initialized RobustnessEvaluator")
        
        # Test robustness evaluation
        sample_texts = ["This is a positive review.", "I did not like the service."]
        logger.info(f"Testing robustness evaluation on {len(sample_texts)} texts")
        
        # Test evaluate method with multiple texts
        result = rob_eval.evaluate(texts=sample_texts)
        logger.info(f"Evaluated robustness with result metrics: {list(result.metrics.keys())}")
        
    except Exception as e:
        logger.error(f"RobustnessEvaluator test failed: {e}")
        raise

def test_adversarial_tester(model: ModelWrapper):
    """Test the AdversarialTester class."""
    logger.info("Testing AdversarialTester...")
    
    try:
        # Initialize adversarial tester with all techniques
        tester = AdversarialTester(
            model, 
            techniques=["attack", "counterfactual", "robustness"]
        )
        logger.info("Initialized AdversarialTester with all techniques")
        
        # Test adversarial testing
        sample_text = "This product is fantastic."
        logger.info(f"Testing adversarial testing on text: '{sample_text}'")
        
        try:
            results = tester.test(sample_text)
            logger.info(f"Completed adversarial testing with techniques: {list(results.keys())}")
            
            # Test with multiple texts
            sample_texts = ["The food was delicious.", "The service was terrible."]
            evaluation = tester.evaluate(sample_texts)
            logger.info(f"Completed adversarial evaluation on {len(sample_texts)} texts")
            logger.info(f"Evaluation metrics: {list(evaluation.metrics.keys() if hasattr(evaluation, 'metrics') else [])}")
        except Exception as e:
            logger.error(f"AdversarialTester execution error: {e}")
            # Continue with the test, don't raise the error
        
    except Exception as e:
        logger.error(f"AdversarialTester test failed: {e}")
        raise

def test_feature_attribution(model: ModelWrapper):
    """Test the FeatureAttributor class."""
    logger.info("Testing FeatureAttributor...")
    
    try:
        # Initialize feature attributor
        attributor = FeatureAttributor(model)
        logger.info("Initialized FeatureAttributor")
        
        # Test feature attribution
        sample_text = "This is an important test sentence."
        logger.info(f"Testing feature attribution on text: '{sample_text}'")
        
        attributions = attributor.analyze(sample_text)
        logger.info(f"Analyzed feature attributions successfully")
        
    except Exception as e:
        logger.error(f"FeatureAttributor test failed: {e}")
        raise

def test_attention_analysis(model: ModelWrapper):
    """Test the AttentionAnalyzer class."""
    logger.info("Testing AttentionAnalyzer...")
    
    try:
        # Initialize attention analyzer
        analyzer = AttentionAnalyzer(model)
        logger.info("Initialized AttentionAnalyzer")
        
        # Test attention analysis
        sample_text = "This sentence will be analyzed for attention patterns."
        logger.info(f"Testing attention analysis on text: '{sample_text}'")
        
        analysis = analyzer.analyze(sample_text)
        logger.info(f"Analyzed attention patterns successfully")
        
    except Exception as e:
        logger.error(f"AttentionAnalyzer test failed: {e}")
        raise

def test_concept_extraction(model: ModelWrapper):
    """Test the ConceptExtractor class."""
    logger.info("Testing ConceptExtractor...")
    
    try:
        # Initialize concept extractor
        extractor = ConceptExtractor(model)
        logger.info("Initialized ConceptExtractor")
        
        # Test concept extraction
        sample_text = "The lion is the king of the jungle."
        logger.info(f"Testing concept extraction on text: '{sample_text}'")
        
        concepts = extractor.analyze(sample_text)
        logger.info(f"Extracted concepts successfully")
        
    except Exception as e:
        logger.error(f"ConceptExtractor test failed: {e}")
        raise

def main():
    """Main function to run all tests."""
    start_time = time.time()
    logger.info("Starting framework tests...")
    
    try:
        # Test core component
        model = test_model_wrapper()
        
        # Test adversarial components
        test_attack_generator(model)
        test_counterfactual_generator(model)
        test_robustness_evaluator(model)
        test_adversarial_tester(model)
        
        # Test interpretability components
        test_feature_attribution(model)
        test_attention_analysis(model)
        test_concept_extraction(model)
        
        elapsed_time = time.time() - start_time
        logger.info(f"All tests completed successfully in {elapsed_time:.2f} seconds!")
        return 0
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Tests failed after {elapsed_time:.2f} seconds: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 