"""
Test script for the Language Model Evaluation Framework.

This script tests all components of the framework to ensure they're working properly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# Import core components
from src.core.model_loader import ModelWrapper
from src.core.evaluator import EvaluationResult

# Import adversarial components
from src.adversarial.attack_generator import AttackGenerator
from src.adversarial.counterfactual_generator import CounterfactualGenerator
from src.adversarial.robustness_evaluator import RobustnessEvaluator
from src.adversarial.module import AdversarialTester

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_wrapper():
    """Test the ModelWrapper class with a small pre-trained model."""
    logger.info("Testing ModelWrapper...")
    
    try:
        # Load a small model for testing
        model_name = "distilbert-base-uncased"
        model = ModelWrapper(model_name, model_type="transformer", task="classification")
        logger.info(f"Successfully loaded model: {model_name}")
        
        # Test embedding extraction
        text = "This is a test sentence."
        embeddings = model.get_embeddings(text)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Test attention extraction
        try:
            attention = model.get_attention(text)
            logger.info(f"Generated attention maps: {len(attention)} layers")
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"ModelWrapper test failed: {e}")
        return False

def test_attack_generator(model):
    """Test the AttackGenerator class."""
    logger.info("Testing AttackGenerator...")
    
    try:
        # Create attack generator
        attack_generator = AttackGenerator(model)
        
        # Test with sample text
        text = "This restaurant serves excellent food and the service is great."
        
        # Generate attacks at different levels
        for level in ["character", "word", "sentence"]:
            logger.info(f"Generating {level}-level attacks...")
            attacks = attack_generator.generate(text, level=level, num_attacks=2)
            
            # Log results
            for i, attack in enumerate(attacks):
                logger.info(f"Attack {i+1}: '{attack['attacked_text']}'")
                logger.info(f"  Success: {attack['success']}")
                logger.info(f"  Difference: {attack['difference']:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"AttackGenerator test failed: {e}")
        return False

def test_counterfactual_generator(model):
    """Test the CounterfactualGenerator class."""
    logger.info("Testing CounterfactualGenerator...")
    
    try:
        # Create counterfactual generator
        cf_generator = CounterfactualGenerator(model, method="rule_based")
        
        # Test with sample text
        text = "I'm happy with the product. It works great and I would recommend it to others."
        
        # Generate counterfactuals
        counterfactuals = cf_generator.generate(text, num_examples=2)
        
        # Log results
        for i, cf in enumerate(counterfactuals):
            logger.info(f"Counterfactual {i+1}: '{cf['counterfactual_text']}'")
            logger.info(f"  Success: {cf['success']}")
            logger.info(f"  Difference: {cf['difference']:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"CounterfactualGenerator test failed: {e}")
        return False

def test_robustness_evaluator(model):
    """Test the RobustnessEvaluator class."""
    logger.info("Testing RobustnessEvaluator...")
    
    try:
        # Create robustness evaluator
        robustness_evaluator = RobustnessEvaluator(model, shift_types=["stylistic", "demographic"])
        
        # Test with sample texts
        texts = [
            "The service at this restaurant was excellent and the food was delicious.",
            "I'm very disappointed with the product quality and customer service."
        ]
        
        # Evaluate robustness
        result = robustness_evaluator.evaluate(texts=texts)
        
        # Log results
        for shift_type, metrics in result.metrics.items():
            if shift_type != "overall":
                logger.info(f"{shift_type.capitalize()} Shifts:")
                logger.info(f"  Success Rate: {metrics['success_rate']:.2f}")
                logger.info(f"  Average Difference: {metrics['average_difference']:.2f}")
        
        logger.info("Overall Robustness:")
        logger.info(f"  Success Rate: {result.metrics['overall']['success_rate']:.2f}")
        logger.info(f"  Average Difference: {result.metrics['overall']['average_difference']:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"RobustnessEvaluator test failed: {e}")
        return False

def test_adversarial_tester(model):
    """Test the AdversarialTester class."""
    logger.info("Testing AdversarialTester...")
    
    try:
        # Create adversarial tester
        adversarial_tester = AdversarialTester(
            model, 
            techniques=["character", "word", "counterfactual"]
        )
        
        # Test with sample text
        text = "This is the best product I've ever purchased. It exceeded all my expectations."
        
        # Run tests
        test_results = adversarial_tester.test(text)
        
        # Log results
        for technique, results in test_results.items():
            logger.info(f"\n{technique.capitalize()} Testing:")
            
            if technique == "attack":
                for attack_level, attacks in results.items():
                    logger.info(f"  {attack_level.capitalize()} Level:")
                    logger.info(f"    Number of attacks: {len(attacks)}")
                    success_rate = sum(a['success'] for a in attacks) / len(attacks) if attacks else 0
                    logger.info(f"    Success rate: {success_rate:.2f}")
            elif technique == "counterfactual":
                logger.info(f"  Number of counterfactuals: {len(results)}")
                success_rate = sum(c['success'] for c in results) / len(results) if results else 0
                logger.info(f"  Success rate: {success_rate:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"AdversarialTester test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting framework evaluation...")
    
    # Test ModelWrapper
    if not test_model_wrapper():
        logger.error("ModelWrapper test failed. Exiting...")
        return
    
    # Load model for remaining tests
    try:
        model = ModelWrapper("distilbert-base-uncased", model_type="transformer", task="classification")
    except Exception as e:
        logger.error(f"Failed to load model for tests: {e}")
        return
    
    # Run component tests
    test_attack_generator(model)
    test_counterfactual_generator(model)
    test_robustness_evaluator(model)
    test_adversarial_tester(model)
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main() 