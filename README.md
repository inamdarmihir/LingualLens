# LingualLens

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive framework for demystifying the black box nature of Large Language Models through advanced interpretability techniques and adversarial robustness evaluation.

## Overview

LingualLens provides researchers and practitioners with a unified toolkit to:

1. **Demystify Black Box Models**: Open up LLMs by providing insights into their internal workings and decision processes
2. **Understand Model Behavior**: Visualize and interpret how language models process and transform information
3. **Test Model Robustness**: Systematically evaluate model vulnerability to adversarial attacks
4. **Improve Model Reliability**: Identify weaknesses and guide improvements in model performance

The framework is designed to reveal the inner workings of complex language models, supporting various architectures from smaller transformer models to large-scale LLMs.

## Features

### Explainability & Interpretability

- **Concept Extraction**: Discover latent concepts and knowledge representations within models, making implicit knowledge explicit
- **Feature Attribution**: Quantify which input tokens most influence model predictions, providing clear causal relationships
- **Attention Analysis**: Visualize attention patterns to understand information flow and focus areas in previously opaque transformer models
- **Decision Pathway Tracking**: Map how information propagates through model layers to form final outputs

### Adversarial Testing

- **Multi-level Attack Generation**:
  - Character-level attacks (typos, character substitutions)
  - Word-level attacks (synonym replacements, word insertions/deletions)
  - Sentence-level attacks (paraphrasing, structural modifications)
- **Counterfactual Generation**: Create minimal examples that change model predictions, revealing decision boundaries
- **Robustness Evaluation**: Test performance across different distribution shifts:
  - Stylistic shifts (formal/informal, technical/simple)
  - Demographic shifts (cultural references, dialect variations)
  - Domain shifts (generalization to new topics)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LingualLens.git
cd LingualLens

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LingualLens.git
cd LingualLens

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt  # if exists
```

## Quick Start

### Loading a Model

```python
from src.core.model_loader import ModelWrapper

# Load a pre-trained language model
model = ModelWrapper(
    "distilbert-base-uncased",  
    task="base",               
    device="cpu"               
)

# Get embeddings for a text input
text = "The framework provides tools for interpretability and adversarial testing."
embeddings = model.get_embeddings(text)
print(f"Embedding shape: {embeddings.shape}")

# Get attention patterns
attention = model.get_attention(text)
print(f"Attention shape: {attention.shape if attention is not None else 'Not available'}")
```

### Explainability Analysis

```python
from src.interpretability.feature_attributor import FeatureAttributor
from src.interpretability.attention_analyzer import AttentionAnalyzer
from src.interpretability.concept_extractor import ConceptExtractor

# Feature attribution - reveals which parts of the input drive the model's decision
attributor = FeatureAttributor(model)
text = "This movie was fantastic and I would highly recommend it to everyone."
attributions = attributor.attribute(text, method="integrated_gradients")
attributor.visualize(text, attributions)

# Attention analysis - shows where the model is focusing its computational resources
analyzer = AttentionAnalyzer(model)
attention_result = analyzer.analyze("The quick brown fox jumps over the lazy dog.")
analyzer.visualize(text, attention_result)

# Concept extraction - extracts high-level concepts the model has learned
extractor = ConceptExtractor(model)
concepts = extractor.extract([
    "The stock market crashed yesterday, causing significant losses for investors.",
    "The company reported strong quarterly earnings, exceeding analyst expectations.",
    "The central bank announced new interest rate policies to combat inflation."
])
```

### Adversarial Testing

```python
from src.adversarial.attack_generator import AttackGenerator
from src.adversarial.counterfactual_generator import CounterfactualGenerator
from src.adversarial.robustness_evaluator import RobustnessEvaluator
from src.adversarial.module import AdversarialTester

# Generate adversarial attacks - reveals model vulnerabilities
attack_gen = AttackGenerator(model)
attacks = attack_gen.generate(
    "This is a positive review of a great product.",
    level="word",
    num_attacks=3
)

# Generate counterfactuals - exposes decision boundaries
cf_gen = CounterfactualGenerator(model)
counterfactuals = cf_gen.generate(
    "I absolutely loved the movie, it was amazing.",
    num_examples=2
)

# Evaluate robustness - quantifies sensitivity to distribution shifts
rob_eval = RobustnessEvaluator(model)
result = rob_eval.evaluate(texts=[
    "The restaurant's food was delicious and I highly recommend it.",
    "This movie is probably the worst I've seen all year."
])

# Comprehensive adversarial testing - provides holistic understanding
adv_tester = AdversarialTester(model, techniques=["attack", "counterfactual", "robustness"])
test_result = adv_tester.test("This product has significantly improved my workflow efficiency.")
```

## Project Structure

```
LingualLens/
├── src/
│   ├── core/                  # Core functionality
│   │   ├── model_loader.py    # Model loading and interface
│   │   └── evaluator.py       # Base evaluation classes
│   ├── interpretability/      # Interpretability tools
│   │   ├── concept_extractor.py
│   │   ├── feature_attributor.py
│   │   └── attention_analyzer.py
│   └── adversarial/           # Adversarial testing tools
│       ├── attack_generator.py
│       ├── counterfactual_generator.py
│       ├── robustness_evaluator.py
│       └── module.py          # Combined adversarial tester
├── notebooks/                 # Demo notebooks
│   ├── demo.ipynb             # General framework demo
│   └── adversarial_testing.ipynb  # Deep dive into adversarial testing
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT License
└── README.md                  # Project documentation
```

## Notebooks & Examples

The repository includes interactive Jupyter notebooks to demonstrate the framework's capabilities:

- **demo.ipynb**: Provides a comprehensive overview of all framework features
- **adversarial_testing.ipynb**: In-depth exploration of adversarial testing techniques
- **black_box_explainability.ipynb**: Step-by-step guide to exploring and explaining LLM outputs

## Why LLM Explainability Matters

Large Language Models are increasingly being deployed in critical applications, yet their decision-making processes remain largely opaque. LingualLens addresses this challenge by:

- **Revealing Internal Mechanisms**: Providing visibility into how information flows through model components
- **Mapping Input-Output Relationships**: Tracing how specific inputs influence generated outputs
- **Identifying Failure Modes**: Discovering when and why models produce incorrect or biased results
- **Supporting Responsible AI**: Enabling the necessary transparency for ethical deployment of LLMs
- **Facilitating Model Improvement**: Guiding targeted enhancements based on explainable weaknesses

## Requirements

- **Python**: 3.8+
- **Deep Learning**: PyTorch 1.9+
- **NLP**: Transformers 4.10+
- **Data Science**: NumPy 1.20+, Pandas 1.3+, Matplotlib 3.4+
- **ML**: Scikit-learn 1.0+
- **Datasets**: HuggingFace Datasets 1.10+

For the complete list of dependencies, see [requirements.txt](requirements.txt).

## Contributing

We welcome contributions from the community! See our [contributing guidelines](CONTRIBUTING.md) for details on how to get started.

Areas where contributions are particularly welcome:
- Additional explainability techniques for LLMs
- New adversarial attack methods
- Support for more model architectures
- Improved visualizations
- Documentation and tutorials

## Citing

If you use LingualLens in your research, please cite it using the following BibTeX entry:

```bibtex
@software{linguallens2023,
  author = {LingualLens Contributors},
  title = {LingualLens: A Framework for Black Box Language Model Explainability and Adversarial Testing},
  year = {2023},
  url = {https://github.com/yourusername/LingualLens}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 