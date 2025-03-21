"""
Base evaluation module for the LingualLens framework.

This module provides base classes for evaluation tasks and result handling.
"""

from typing import Dict, List, Union, Any, Optional
import json
import numpy as np
from datetime import datetime

class EvaluationResult:
    """Base class for storing and processing evaluation results."""
    
    def __init__(self, 
                metrics: Dict[str, float] = None, 
                details: Dict[str, Any] = None,
                metadata: Dict[str, Any] = None):
        """
        Initialize evaluation result.
        
        Args:
            metrics: Dictionary mapping metric names to values
            details: Dictionary with detailed result information
            metadata: Dictionary with metadata (e.g., timestamp, model info)
        """
        self.metrics = metrics or {}
        self.details = details or {}
        self.metadata = metadata or {
            "timestamp": datetime.now().isoformat(),
        }
    
    def add_metric(self, name: str, value: float):
        """Add a metric to the result."""
        self.metrics[name] = value
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add multiple metrics to the result."""
        self.metrics.update(metrics)
    
    def add_detail(self, name: str, value: Any):
        """Add detailed information to the result."""
        self.details[name] = value
    
    def add_details(self, details: Dict[str, Any]):
        """Add multiple details to the result."""
        self.details.update(details)
    
    def add_metadata(self, name: str, value: Any):
        """Add metadata to the result."""
        self.metadata[name] = value
    
    def get_metric(self, name: str, default: float = None) -> Optional[float]:
        """Get a metric by name."""
        return self.metrics.get(name, default)
    
    def get_detail(self, name: str, default: Any = None) -> Any:
        """Get detail by name."""
        return self.details.get(name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "metrics": self.metrics,
            "details": self.details,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        # Convert numpy arrays to lists
        result_dict = self.to_dict()
        
        def convert_arrays(obj):
            if isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        result_dict = convert_arrays(result_dict)
        return json.dumps(result_dict, indent=indent)
    
    def __repr__(self) -> str:
        """String representation of the result."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EvaluationResult(metrics: {metrics_str})"


class Evaluator:
    """Base class for evaluators in the framework."""
    
    def __init__(self, model=None):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
        """
        self.model = model
    
    def evaluate(self, *args, **kwargs) -> EvaluationResult:
        """
        Evaluate the model.
        
        This is a base method to be implemented by subclasses.
        
        Returns:
            EvaluationResult object containing evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def _create_result(self, 
                      metrics: Dict[str, float] = None, 
                      details: Dict[str, Any] = None) -> EvaluationResult:
        """
        Create an evaluation result with standardized metadata.
        
        Args:
            metrics: Dictionary of evaluation metrics
            details: Dictionary of detailed results
            
        Returns:
            EvaluationResult instance
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "evaluator": self.__class__.__name__,
        }
        
        if self.model is not None and hasattr(self.model, "model_name"):
            metadata["model_name"] = self.model.model_name
        
        result = EvaluationResult(metrics, details, metadata)
        return result 