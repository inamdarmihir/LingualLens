"""
Evaluator module for the Language Model Evaluation Framework.

This module provides base classes for model evaluation.
"""

from typing import Dict, List, Any, Optional, Union
import json
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model_loader import ModelWrapper


@dataclass
class EvaluationResult:
    """
    Dataclass for storing evaluation results.
    
    This class provides a standardized format for evaluation results
    and includes methods for visualization and export.
    """
    
    # Basic information
    model_name: str
    evaluation_type: str
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Examples
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "model_name": self.model_name,
            "evaluation_type": self.evaluation_type,
            "metrics": self.metrics,
            "examples": self.examples,
            "metadata": self.metadata
        }
    
    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Convert the result to JSON.
        
        Args:
            filepath: If provided, save to this file
            
        Returns:
            JSON string if filepath is None, otherwise None
        """
        result_dict = self.to_dict()
        
        # Replace non-serializable objects
        def process_item(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, (np.int64, np.int32, np.float64, np.float32)):
                return item.item()
            elif isinstance(item, dict):
                return {k: process_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [process_item(i) for i in item]
            else:
                return item
        
        # Process all items to ensure they're JSON serializable
        result_dict = process_item(result_dict)
        
        # Convert to JSON
        json_str = json.dumps(result_dict, indent=2)
        
        # Save if filepath is provided
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        
        return json_str
    
    def visualize(self, metric_name: Optional[str] = None, figsize: tuple = (10, 6)):
        """
        Visualize the evaluation results.
        
        Args:
            metric_name: Name of metric to visualize (if None, visualize all)
            figsize: Figure size
        """
        if metric_name is not None:
            if metric_name not in self.metrics:
                raise ValueError(f"Metric '{metric_name}' not found in results")
            
            # Visualize specific metric
            metric_value = self.metrics[metric_name]
            self._visualize_metric(metric_name, metric_value, figsize)
        else:
            # Visualize all metrics
            for name, value in self.metrics.items():
                self._visualize_metric(name, value, figsize)
    
    def _visualize_metric(self, name: str, value: Any, figsize: tuple):
        """
        Visualize a specific metric.
        
        Args:
            name: Metric name
            value: Metric value
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        if isinstance(value, (int, float)):
            # Single numeric value
            plt.bar(['Value'], [value])
            plt.title(f"Metric: {name}")
            plt.ylabel("Value")
        
        elif isinstance(value, dict):
            # Dictionary of values
            plt.bar(list(value.keys()), list(value.values()))
            plt.title(f"Metric: {name}")
            plt.xticks(rotation=45)
            plt.ylabel("Value")
        
        elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            # List of numeric values
            plt.plot(value)
            plt.title(f"Metric: {name}")
            plt.xlabel("Index")
            plt.ylabel("Value")
        
        else:
            plt.text(0.5, 0.5, f"Cannot visualize {name} (type: {type(value).__name__})",
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()


class Evaluator:
    """
    Base class for model evaluators.
    
    This class provides common functionality for various evaluation methods.
    Specific evaluators should inherit from this class.
    """
    
    def __init__(self, model: ModelWrapper, **kwargs):
        """
        Initialize the evaluator.
        
        Args:
            model: The model to evaluate
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.kwargs = kwargs
    
    def evaluate(self, **kwargs) -> EvaluationResult:
        """
        Evaluate the model.
        
        Args:
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation result
            
        Raises:
            NotImplementedError: When not implemented by the subclass
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def _create_result(self, evaluation_type: str) -> EvaluationResult:
        """
        Create a new evaluation result object.
        
        Args:
            evaluation_type: Type of evaluation
            
        Returns:
            New evaluation result object
        """
        return EvaluationResult(
            model_name=self.model.model_name,
            evaluation_type=evaluation_type
        ) 