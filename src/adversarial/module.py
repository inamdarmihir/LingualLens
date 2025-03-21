"""
Main adversarial testing module for the Language Model Evaluation Framework.

This module integrates different adversarial testing techniques to provide
a comprehensive evaluation of model robustness.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np

from ..core.model_loader import ModelWrapper
from ..core.evaluator import Evaluator, EvaluationResult


class AdversarialTester(Evaluator):
    """
    Module for testing language model robustness against adversarial inputs.
    
    This module combines different adversarial testing techniques to provide
    a comprehensive analysis of model robustness.
    """

    def __init__(
        self,
        model: ModelWrapper,
        techniques: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the adversarial tester.
        
        Args:
            model: The model to test
            techniques: List of adversarial testing techniques to use
                (options: "character", "word", "sentence", "counterfactual", "distribution_shift")
            **kwargs: Additional keyword arguments for specific techniques
        """
        super().__init__(model, **kwargs)
        self.techniques = techniques or ["character", "word", "sentence", "counterfactual"]
        
        # Initialize testers based on techniques
        self.testers = {}
        if "character" in self.techniques or "word" in self.techniques or "sentence" in self.techniques:
            from .attack_generator import AttackGenerator
            self.testers["attack"] = AttackGenerator(model, **kwargs)
        
        if "counterfactual" in self.techniques:
            from .counterfactual_generator import CounterfactualGenerator
            self.testers["counterfactual"] = CounterfactualGenerator(model, **kwargs)
        
        if "distribution_shift" in self.techniques:
            from .robustness_evaluator import RobustnessEvaluator
            self.testers["distribution_shift"] = RobustnessEvaluator(model, **kwargs)

    def test(self, text: str) -> Dict[str, Any]:
        """
        Test a model's robustness with adversarial examples based on a text input.
        
        Args:
            text: The input text to generate adversarial examples from
            
        Returns:
            Dictionary containing test results from different techniques
        """
        results = {}
        
        if "attack" in self.testers:
            attack_results = {}
            if "character" in self.techniques:
                character_attacks = self.testers["attack"].generate(text, level="character")
                attack_results["character"] = character_attacks
            if "word" in self.techniques:
                word_attacks = self.testers["attack"].generate(text, level="word")
                attack_results["word"] = word_attacks
            if "sentence" in self.techniques:
                sentence_attacks = self.testers["attack"].generate(text, level="sentence")
                attack_results["sentence"] = sentence_attacks
            results["attack"] = attack_results
            
        if "counterfactual" in self.testers and "counterfactual" in self.techniques:
            counterfactuals = self.testers["counterfactual"].generate(text)
            results["counterfactual"] = counterfactuals
            
        if "distribution_shift" in self.testers and "distribution_shift" in self.techniques:
            distribution_results = self.testers["distribution_shift"].evaluate(text=text)
            results["distribution_shift"] = distribution_results
            
        return results

    def evaluate(
        self,
        dataset: Optional[Union[str, List[Dict[str, Any]]]] = None,
        texts: Optional[List[str]] = None,
        target_labels: Optional[List[Any]] = None,
        num_examples: int = 10,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate model robustness on a dataset or list of texts.
        
        Args:
            dataset: Dataset name or list of examples
            texts: List of input texts (if dataset is not provided)
            target_labels: Optional target labels (for classification tasks)
            num_examples: Number of examples to include in the result
            **kwargs: Additional keyword arguments
            
        Returns:
            EvaluationResult containing robustness metrics and examples
        """
        if dataset is None and texts is None:
            raise ValueError("Either dataset or texts must be provided")
            
        # Load dataset if provided
        if dataset is not None:
            if isinstance(dataset, str):
                # Load dataset from file or HuggingFace datasets
                try:
                    from datasets import load_dataset
                    dataset = load_dataset(dataset)
                    texts = [example["text"] for example in dataset["train"][:num_examples]]
                    if "label" in dataset["train"].features:
                        target_labels = [example["label"] for example in dataset["train"][:num_examples]]
                except Exception as e:
                    raise ValueError(f"Failed to load dataset: {e}")
            else:
                # Use provided list of examples
                texts = [example["text"] for example in dataset[:num_examples]]
                if all("label" in example for example in dataset[:num_examples]):
                    target_labels = [example["label"] for example in dataset[:num_examples]]
        
        # Ensure we don't evaluate too many examples
        texts = texts[:num_examples]
        if target_labels:
            target_labels = target_labels[:num_examples]
        
        # Calculate robustness metrics
        success_rates = {}
        semantic_similarity = []
        perturbation_size = []
        examples = []
        
        for i, text in enumerate(texts):
            # Get target label if available
            target_label = target_labels[i] if target_labels and i < len(target_labels) else None
            
            # Test model robustness
            test_result = self.test(text)
            example = {"text": text, "robustness": {}}
            
            # Calculate metrics
            if "attack" in test_result:
                try:
                    attack_success = self._calculate_attack_success(test_result["attack"], target_label)
                    for attack_type, success in attack_success.items():
                        success_rates[attack_type] = success_rates.get(attack_type, []) + [success]
                    example["robustness"]["attack_success"] = attack_success
                    
                    # Calculate semantic similarity for successful attacks
                    for attack_type, attack_results in test_result["attack"].items():
                        if isinstance(attack_results, list):
                            for attack_result in attack_results:
                                if isinstance(attack_result, dict) and attack_result.get("success", False):
                                    similarity = self._calculate_semantic_similarity(text, attack_result.get("attacked_text", ""))
                                    semantic_similarity.append(similarity)
                                    perturbation_size.append(attack_result.get("perturbation_size", 0))
                except Exception as e:
                    # Log the error but continue processing
                    print(f"Error processing attack results: {e}")
                    example["robustness"]["attack_error"] = str(e)
                
            if "counterfactual" in test_result:
                counterfactual_results = test_result["counterfactual"]
                example["robustness"]["counterfactual"] = {
                    "count": len(counterfactual_results),
                    "examples": counterfactual_results[:3]  # Include only a few examples
                }
                
                # Calculate semantic similarity for counterfactuals
                for counterfactual in counterfactual_results:
                    similarity = self._calculate_semantic_similarity(text, counterfactual["text"])
                    semantic_similarity.append(similarity)
                    
            if "distribution_shift" in test_result:
                example["robustness"]["distribution_shift"] = test_result["distribution_shift"]
                
            # Add full test results for a limited number of examples
            if len(examples) < min(5, num_examples):
                example["full_results"] = test_result
                
            examples.append(example)
            
        # Compile metrics
        metrics = {}
        
        # Calculate average success rate for each attack type
        for attack_type, rates in success_rates.items():
            metrics[f"{attack_type}_success_rate"] = np.mean(rates)
        
        # Calculate overall metrics
        if semantic_similarity:
            metrics["avg_semantic_similarity"] = np.mean(semantic_similarity)
        if perturbation_size:
            metrics["avg_perturbation_size"] = np.mean(perturbation_size)
            
        # Calculate overall robustness score (lower is better)
        if success_rates:
            # Average success rate across all attack types (lower is better for robustness)
            metrics["overall_attack_success_rate"] = np.mean([np.mean(rates) for rates in success_rates.values()])
            # Robustness score (higher is better)
            metrics["robustness_score"] = 1.0 - metrics["overall_attack_success_rate"]
            
        return EvaluationResult(
            component_name="adversarial",
            metrics=metrics,
            examples=examples,
            metadata={
                "model_name": self.model.model_name,
                "techniques": self.techniques,
            }
        )

    def _calculate_attack_success(self, attack_results: Dict[str, List[Dict[str, Any]]], target_label: Any = None) -> Dict[str, float]:
        """
        Calculate success rate for different attack types.
        
        Args:
            attack_results: Results from attack generator
            target_label: Target label (for classification tasks)
            
        Returns:
            Dictionary mapping attack types to success rates
        """
        success_rates = {}
        
        for attack_type, results in attack_results.items():
            if not isinstance(results, list):
                # Skip if results is not a list
                continue
                
            success_count = sum(1 for result in results if isinstance(result, dict) and result.get("success", False))
            success_rates[attack_type] = success_count / max(1, len(results))
            
        return success_rates

    def _calculate_semantic_similarity(self, original_text: str, adversarial_text: str) -> float:
        """
        Calculate semantic similarity between original and adversarial texts.
        
        Args:
            original_text: Original input text
            adversarial_text: Adversarial version of the text
            
        Returns:
            Similarity score between 0 and 1
        """
        # This is a simplified implementation
        # In a real implementation, you would use a proper semantic similarity measure
        # such as BERT embeddings similarity or BLEU score
        
        # Simple character-level similarity
        original_chars = set(original_text)
        adversarial_chars = set(adversarial_text)
        
        if not original_chars and not adversarial_chars:
            return 1.0
        
        intersection = len(original_chars.intersection(adversarial_chars))
        union = len(original_chars.union(adversarial_chars))
        
        return intersection / union 