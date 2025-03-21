"""
LingualLens: A framework for black box LLM explainability and adversarial testing.
"""

__version__ = "0.1.0"

# Core components
from linguallens.core.model_loader import ModelWrapper
from linguallens.core.evaluator import Evaluator, EvaluationResult

# Interpretability components
from linguallens.interpretability.llm_explainer import LLMExplainer
from linguallens.interpretability.feature_attributor import FeatureAttributor
from linguallens.interpretability.attention_analyzer import AttentionAnalyzer
from linguallens.interpretability.concept_extractor import ConceptExtractor
from linguallens.interpretability.neuron_analyzer import NeuronAnalyzer
from linguallens.interpretability.advanced_attribution import AdvancedAttributor

# Adversarial components
from linguallens.adversarial.attack_generator import AttackGenerator
from linguallens.adversarial.counterfactual_generator import CounterfactualGenerator
from linguallens.adversarial.robustness_evaluator import RobustnessEvaluator
from linguallens.adversarial.module import AdversarialTester
from linguallens.adversarial.advanced_attacks import AdvancedAttackGenerator

# Model comparison
from linguallens.core.model_comparison import ModelComparator 