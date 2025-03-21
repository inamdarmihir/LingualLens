"""
Robustness evaluator module for the Language Model Evaluation Framework.

This module provides functionality for evaluating the robustness of language models
against distribution shifts and perturbations.
"""

from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
import random
from collections import defaultdict

from ..core.model_loader import ModelWrapper
from ..core.evaluator import Evaluator, EvaluationResult


class RobustnessEvaluator(Evaluator):
    """Evaluator for testing model robustness against distribution shifts."""

    def __init__(
        self,
        model: ModelWrapper,
        shift_types: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the robustness evaluator.
        
        Args:
            model: The model to evaluate
            shift_types: Types of distribution shifts to test
                (options: "stylistic", "demographic", "domain", "temporal")
            **kwargs: Additional keyword arguments
        """
        super().__init__(model, **kwargs)
        self.shift_types = shift_types or ["stylistic", "demographic", "domain"]
        
        # Initialize resources
        self._load_resources()

    def _load_resources(self):
        """Load necessary resources for distribution shift simulation."""
        # Define style transformations
        self.style_transforms = {
            "formal": self._transform_to_formal,
            "casual": self._transform_to_casual,
            "technical": self._transform_to_technical,
            "simplified": self._transform_to_simplified
        }
        
        # Define demographic transforms
        self.demographic_patterns = {
            "age_young": {
                "words": ["cool", "awesome", "lit", "vibe", "flex", "lowkey", "highkey", "tbh", "sus", "fam"],
                "patterns": [
                    (r"\bis\b", "is"),
                    (r"\bvery\b", "so"),
                    (r"\bgood\b", "fire"),
                    (r"\bfriend\b", "bestie"),
                    (r"\blike\b", "love")
                ]
            },
            "age_older": {
                "words": ["indeed", "perhaps", "certainly", "rather", "quite", "nevertheless", "however"],
                "patterns": [
                    (r"\bis\b", "is"),
                    (r"\bvery\b", "quite"),
                    (r"\bgood\b", "excellent"),
                    (r"\bfriend\b", "colleague"),
                    (r"\blike\b", "appreciate")
                ]
            }
        }
        
        # Define domain transforms
        self.domain_terms = {
            "general": ["thing", "item", "person", "place", "time", "way", "problem", "solution"],
            "technical": ["algorithm", "function", "parameter", "variable", "module", "system", "interface", "API"],
            "medical": ["patient", "diagnosis", "treatment", "symptom", "condition", "prognosis", "medication", "test"],
            "legal": ["plaintiff", "defendant", "evidence", "testimony", "counsel", "statute", "ruling", "precedent"],
            "finance": ["asset", "liability", "investment", "portfolio", "derivative", "dividend", "equity", "bond"]
        }

    def evaluate(
        self,
        text: str = None,
        dataset: Optional[Union[str, List[Dict[str, Any]]]] = None,
        texts: Optional[List[str]] = None,
        num_examples: int = 10,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate model robustness against distribution shifts.
        
        Args:
            text: Single text to evaluate
            dataset: Dataset name or list of examples
            texts: List of texts to evaluate
            num_examples: Number of examples to include in results
            **kwargs: Additional keyword arguments
            
        Returns:
            Evaluation result
        """
        # Create evaluation result
        result = self._create_result("robustness")
        
        # Process inputs
        if text is not None:
            texts = [text]
        elif dataset is not None:
            if isinstance(dataset, str):
                # Load dataset from file or HuggingFace datasets
                try:
                    from datasets import load_dataset
                    dataset = load_dataset(dataset)
                    texts = [example["text"] for example in dataset["train"][:num_examples]]
                except Exception as e:
                    raise ValueError(f"Failed to load dataset: {e}")
            else:
                # Use provided list of examples
                texts = [example["text"] for example in dataset[:num_examples]]
        
        if not texts:
            raise ValueError("No texts provided for evaluation")
        
        # Limit to num_examples
        texts = texts[:num_examples]
        
        # Evaluate robustness for each text
        all_results = []
        for text in texts:
            text_result = self._evaluate_text_robustness(text)
            all_results.append(text_result)
            
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(all_results)
        result.metrics = metrics
        
        # Add examples
        for i, (text, text_result) in enumerate(zip(texts, all_results)):
            if i >= num_examples:
                break
                
            example = {
                "text": text,
                "shifts": {}
            }
            
            # Add results for each shift type
            for shift_type, shift_results in text_result.items():
                example["shifts"][shift_type] = {
                    "transformed_texts": [r["transformed_text"] for r in shift_results],
                    "success_rate": sum(r["success"] for r in shift_results) / len(shift_results) if shift_results else 0,
                    "average_difference": sum(r["difference"] for r in shift_results) / len(shift_results) if shift_results else 0
                }
                
            result.examples.append(example)
        
        return result

    def _evaluate_text_robustness(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate robustness of a single text.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Dictionary with results for each shift type
        """
        results = {}
        
        # Get original model output
        original_output = None
        if self.model.task == "text-generation":
            original_output = self.model.generate(text)
        elif self.model.task == "classification":
            original_output = self.model.classify(text)
        else:
            # Use embeddings for general purpose models
            original_output = self.model.get_embeddings(text).cpu().numpy()
        
        # Evaluate each shift type
        for shift_type in self.shift_types:
            if shift_type == "stylistic":
                results[shift_type] = self._evaluate_stylistic_shifts(text, original_output)
            elif shift_type == "demographic":
                results[shift_type] = self._evaluate_demographic_shifts(text, original_output)
            elif shift_type == "domain":
                results[shift_type] = self._evaluate_domain_shifts(text, original_output)
            elif shift_type == "temporal":
                results[shift_type] = self._evaluate_temporal_shifts(text, original_output)
        
        return results

    def _evaluate_stylistic_shifts(
        self, text: str, original_output: Any
    ) -> List[Dict[str, Any]]:
        """
        Evaluate robustness against stylistic shifts.
        
        Args:
            text: Input text
            original_output: Original model output
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Apply each style transformation
        for style, transform_func in self.style_transforms.items():
            transformed_text = transform_func(text)
            
            # Skip if no changes were made
            if transformed_text == text:
                continue
                
            # Get model output for transformed text
            transformed_output = None
            if self.model.task == "text-generation":
                transformed_output = self.model.generate(transformed_text)
            elif self.model.task == "classification":
                transformed_output = self.model.classify(transformed_text)
            else:
                # Use embeddings for general purpose models
                transformed_output = self.model.get_embeddings(transformed_text).cpu().numpy()
                
            # Calculate success and difference
            success = self._is_shift_successful(original_output, transformed_output)
            difference = self._calculate_difference(original_output, transformed_output)
            
            # Add result
            results.append({
                "shift_type": "stylistic",
                "style": style,
                "original_text": text,
                "transformed_text": transformed_text,
                "success": success,
                "difference": difference
            })
            
        return results

    def _evaluate_demographic_shifts(
        self, text: str, original_output: Any
    ) -> List[Dict[str, Any]]:
        """
        Evaluate robustness against demographic shifts.
        
        Args:
            text: Input text
            original_output: Original model output
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Apply each demographic transformation
        for demographic, patterns in self.demographic_patterns.items():
            transformed_text = self._transform_demographic(text, patterns)
            
            # Skip if no changes were made
            if transformed_text == text:
                continue
                
            # Get model output for transformed text
            transformed_output = None
            if self.model.task == "text-generation":
                transformed_output = self.model.generate(transformed_text)
            elif self.model.task == "classification":
                transformed_output = self.model.classify(transformed_text)
            else:
                # Use embeddings for general purpose models
                transformed_output = self.model.get_embeddings(transformed_text).cpu().numpy()
                
            # Calculate success and difference
            success = self._is_shift_successful(original_output, transformed_output)
            difference = self._calculate_difference(original_output, transformed_output)
            
            # Add result
            results.append({
                "shift_type": "demographic",
                "demographic": demographic,
                "original_text": text,
                "transformed_text": transformed_text,
                "success": success,
                "difference": difference
            })
            
        return results

    def _evaluate_domain_shifts(
        self, text: str, original_output: Any
    ) -> List[Dict[str, Any]]:
        """
        Evaluate robustness against domain shifts.
        
        Args:
            text: Input text
            original_output: Original model output
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Apply domain shifts
        for source_domain, target_domain in [
            ("general", "technical"),
            ("general", "medical"),
            ("general", "legal"),
            ("general", "finance")
        ]:
            if source_domain not in self.domain_terms or target_domain not in self.domain_terms:
                continue
                
            transformed_text = self._transform_domain(
                text, self.domain_terms[source_domain], self.domain_terms[target_domain]
            )
            
            # Skip if no changes were made
            if transformed_text == text:
                continue
                
            # Get model output for transformed text
            transformed_output = None
            if self.model.task == "text-generation":
                transformed_output = self.model.generate(transformed_text)
            elif self.model.task == "classification":
                transformed_output = self.model.classify(transformed_text)
            else:
                # Use embeddings for general purpose models
                transformed_output = self.model.get_embeddings(transformed_text).cpu().numpy()
                
            # Calculate success and difference
            success = self._is_shift_successful(original_output, transformed_output)
            difference = self._calculate_difference(original_output, transformed_output)
            
            # Add result
            results.append({
                "shift_type": "domain",
                "source_domain": source_domain,
                "target_domain": target_domain,
                "original_text": text,
                "transformed_text": transformed_text,
                "success": success,
                "difference": difference
            })
            
        return results

    def _evaluate_temporal_shifts(
        self, text: str, original_output: Any
    ) -> List[Dict[str, Any]]:
        """
        Evaluate robustness against temporal shifts.
        
        This is a simplified implementation. In a real-world scenario,
        you would use more sophisticated methods to simulate temporal shifts.
        
        Args:
            text: Input text
            original_output: Original model output
            
        Returns:
            List of evaluation results
        """
        # Note: This is a placeholder implementation
        # A real implementation would use time-specific corpora or language models
        return []

    def _transform_to_formal(self, text: str) -> str:
        """Transform text to a more formal style."""
        # Simple rule-based transformations
        transformations = [
            (r"\bI'm\b", "I am"),
            (r"\bdon't\b", "do not"),
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\bhadn't\b", "had not"),
            (r"\bhaven't\b", "have not"),
            (r"\bhasn't\b", "has not"),
            (r"\bdidn't\b", "did not"),
            (r"\bwouldn't\b", "would not"),
            (r"\bcouldn't\b", "could not"),
            (r"\bshouldn't\b", "should not"),
            (r"\bye\b", "yes"),
            (r"\bgonna\b", "going to"),
            (r"\bwanna\b", "want to"),
            (r"\bgotta\b", "got to"),
            (r"\bkinda\b", "kind of"),
            (r"\bsorta\b", "sort of"),
            (r"\bcourse\b", "of course"),
            (r"\bcuz\b", "because"),
            (r"\blol\b", ""),
            (r"\bwtf\b", ""),
            (r"\bomg\b", ""),
            (r"\bidk\b", "I do not know"),
            (r"\bfyi\b", "for your information"),
        ]
        
        # Apply transformations
        import re
        result = text
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result

    def _transform_to_casual(self, text: str) -> str:
        """Transform text to a more casual style."""
        # Simple rule-based transformations
        transformations = [
            (r"\bI am\b", "I'm"),
            (r"\bdo not\b", "don't"),
            (r"\bcannot\b", "can't"),
            (r"\bwill not\b", "won't"),
            (r"\bhad not\b", "hadn't"),
            (r"\bhave not\b", "haven't"),
            (r"\bhas not\b", "hasn't"),
            (r"\bdid not\b", "didn't"),
            (r"\bwould not\b", "wouldn't"),
            (r"\bcould not\b", "couldn't"),
            (r"\bshould not\b", "shouldn't"),
            (r"\bgoing to\b", "gonna"),
            (r"\bwant to\b", "wanna"),
            (r"\bgot to\b", "gotta"),
            (r"\bkind of\b", "kinda"),
            (r"\bsort of\b", "sorta"),
            (r"\bof course\b", "course"),
            (r"\bbecause\b", "cuz"),
            (r"\bI do not know\b", "idk"),
            (r"\bfor your information\b", "fyi"),
        ]
        
        # Apply transformations
        import re
        result = text
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result

    def _transform_to_technical(self, text: str) -> str:
        """Transform text to a more technical style."""
        # Simple rule-based transformations
        transformations = [
            (r"\buse\b", "utilize"),
            (r"\bstart\b", "initialize"),
            (r"\bend\b", "terminate"),
            (r"\bmake\b", "implement"),
            (r"\bcheck\b", "verify"),
            (r"\bfix\b", "resolve"),
            (r"\bchange\b", "modify"),
            (r"\bproblem\b", "issue"),
            (r"\bthing\b", "component"),
            (r"\btalk\b", "communicate"),
            (r"\bsee\b", "observe"),
            (r"\blearn\b", "acquire knowledge"),
            (r"\bget\b", "obtain"),
            (r"\bshow\b", "demonstrate"),
            (r"\bgive\b", "provide"),
            (r"\bask\b", "inquire"),
            (r"\banswer\b", "respond"),
            (r"\bhelp\b", "assist"),
        ]
        
        # Apply transformations
        import re
        result = text
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result

    def _transform_to_simplified(self, text: str) -> str:
        """Transform text to a more simplified style."""
        # Simple rule-based transformations
        transformations = [
            (r"\butilize\b", "use"),
            (r"\binitialize\b", "start"),
            (r"\bterminate\b", "end"),
            (r"\bimplement\b", "make"),
            (r"\bverify\b", "check"),
            (r"\bresolve\b", "fix"),
            (r"\bmodify\b", "change"),
            (r"\bissue\b", "problem"),
            (r"\bcomponent\b", "thing"),
            (r"\bcommunicate\b", "talk"),
            (r"\bobserve\b", "see"),
            (r"\bacquire knowledge\b", "learn"),
            (r"\bobtain\b", "get"),
            (r"\bdemonstrate\b", "show"),
            (r"\bprovide\b", "give"),
            (r"\binquire\b", "ask"),
            (r"\brespond\b", "answer"),
            (r"\bassist\b", "help"),
        ]
        
        # Apply transformations
        import re
        result = text
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result

    def _transform_demographic(self, text: str, patterns: Dict[str, Any]) -> str:
        """
        Transform text based on demographic patterns.
        
        Args:
            text: Input text
            patterns: Demographic patterns dictionary
            
        Returns:
            Transformed text
        """
        import re
        result = text
        
        # Apply word replacements
        words = patterns.get("words", [])
        if words:
            # Replace random words in the text
            text_words = result.split()
            for i in range(len(text_words)):
                if random.random() < 0.2:  # 20% chance to replace a word
                    text_words[i] = random.choice(words)
            result = " ".join(text_words)
            
        # Apply pattern replacements
        pattern_list = patterns.get("patterns", [])
        for pattern, replacement in pattern_list:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            
        return result

    def _transform_domain(
        self, text: str, source_terms: List[str], target_terms: List[str]
    ) -> str:
        """
        Transform text from one domain to another.
        
        Args:
            text: Input text
            source_terms: Terms from source domain
            target_terms: Terms from target domain
            
        Returns:
            Transformed text
        """
        result = text
        
        # Replace terms from source domain with terms from target domain
        for source_term in source_terms:
            if source_term in text.lower():
                # Randomly select a term from target domain
                target_term = random.choice(target_terms)
                result = result.replace(source_term, target_term)
                
        return result

    def _is_shift_successful(self, original_output: Any, transformed_output: Any) -> bool:
        """
        Determine if a distribution shift was successful.
        
        Args:
            original_output: Original model output
            transformed_output: Output after shift
            
        Returns:
            True if shift succeeded, False otherwise
        """
        # For text generation, check if outputs are different
        if isinstance(original_output, str) and isinstance(transformed_output, str):
            return original_output != transformed_output
        
        # For classification, check if prediction changed
        elif isinstance(original_output, dict) and isinstance(transformed_output, dict):
            original_class = max(original_output.items(), key=lambda x: x[1])[0]
            transformed_class = max(transformed_output.items(), key=lambda x: x[1])[0]
            return original_class != transformed_class
        
        # For embeddings, check cosine distance
        elif isinstance(original_output, np.ndarray) and isinstance(transformed_output, np.ndarray):
            cosine_sim = np.dot(original_output, transformed_output) / (
                np.linalg.norm(original_output) * np.linalg.norm(transformed_output)
            )
            # Different if similarity is below threshold
            return cosine_sim < 0.9
        
        return False

    def _calculate_difference(self, original_output: Any, transformed_output: Any) -> float:
        """
        Calculate difference between original and transformed outputs.
        
        Args:
            original_output: Original model output
            transformed_output: Output after transformation
            
        Returns:
            Difference metric (0-1 scale)
        """
        # For text generation, use string difference metrics
        if isinstance(original_output, str) and isinstance(transformed_output, str):
            # Levenshtein distance normalized by max length
            from difflib import SequenceMatcher
            return 1 - SequenceMatcher(None, original_output, transformed_output).ratio()
        
        # For classification, calculate probability difference
        elif isinstance(original_output, dict) and isinstance(transformed_output, dict):
            diff = 0
            for key in set(original_output.keys()) | set(transformed_output.keys()):
                orig_val = original_output.get(key, 0)
                trans_val = transformed_output.get(key, 0)
                diff += abs(orig_val - trans_val)
            return min(1.0, diff / 2.0)  # Normalize to 0-1
        
        # For embeddings, use cosine distance
        elif isinstance(original_output, np.ndarray) and isinstance(transformed_output, np.ndarray):
            cosine_sim = np.dot(original_output, transformed_output) / (
                np.linalg.norm(original_output) * np.linalg.norm(transformed_output)
            )
            return 1 - max(0, cosine_sim)  # Convert similarity to distance
        
        return 0.0

    def _calculate_aggregate_metrics(
        self, results: List[Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all texts and shifts.
        
        Args:
            results: List of evaluation results for each text
            
        Returns:
            Dictionary of aggregate metrics
        """
        metrics = {
            "overall": {
                "success_rate": 0,
                "average_difference": 0
            }
        }
        
        # Counters for calculating averages
        total_shifts = 0
        total_success = 0
        total_difference = 0
        
        # Process each shift type
        for shift_type in self.shift_types:
            metrics[shift_type] = {
                "success_rate": 0,
                "average_difference": 0
            }
            
            shift_count = 0
            shift_success = 0
            shift_difference = 0
            
            # Accumulate metrics for this shift type
            for text_result in results:
                if shift_type in text_result:
                    shift_results = text_result[shift_type]
                    shift_count += len(shift_results)
                    shift_success += sum(r["success"] for r in shift_results)
                    shift_difference += sum(r["difference"] for r in shift_results)
            
            # Calculate averages for this shift type
            if shift_count > 0:
                metrics[shift_type]["success_rate"] = shift_success / shift_count
                metrics[shift_type]["average_difference"] = shift_difference / shift_count
            
            # Update overall counters
            total_shifts += shift_count
            total_success += shift_success
            total_difference += shift_difference
        
        # Calculate overall averages
        if total_shifts > 0:
            metrics["overall"]["success_rate"] = total_success / total_shifts
            metrics["overall"]["average_difference"] = total_difference / total_shifts
        
        return metrics 