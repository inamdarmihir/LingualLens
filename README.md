# LingualLens

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Multi-Provider](https://img.shields.io/badge/multi--provider-OpenAI%20%7C%20Google%20%7C%20Anthropic%20%7C%20HF-blueviolet)]()

A comprehensive framework for demystifying the black box nature of Large Language Models through advanced interpretability techniques and adversarial robustness evaluation.

## 🆕 Multi-Provider Support (New!)

LingualLens now supports a wide range of model providers through a unified interface:

- **Open Source Models** (Hugging Face): Access hundreds of models like BERT, GPT-2, Llama, and more
- **OpenAI Models**: Analyze GPT-3.5, GPT-4, and other OpenAI models through their API
- **Google Models**: Work with Gemini and other Google AI models
- **Anthropic Models**: Interpret Claude models with the same tools
- **Azure OpenAI Models**: Leverage Microsoft-hosted OpenAI models
- **Custom Models**: Extensible architecture for adding support for other providers

The framework provides consistent interfaces across all providers for:

- **Model loading and inference**
- **Embedding extraction**
- **Token importance analysis**
- **Counterfactual generation**
- **Output sensitivity measurement**
- **Black box explainability**

See the `multi_provider_demo.ipynb` notebook for examples of using different model providers.

## Overview

LingualLens provides researchers and practitioners with a unified toolkit to:

1. **Demystify Black Box Models**: Open up LLMs by providing insights into their internal workings and decision processes
2. **Understand Model Behavior**: Visualize and interpret how language models process and transform information
3. **Test Model Robustness**: Systematically evaluate model vulnerability to adversarial attacks
4. **Improve Model Reliability**: Identify weaknesses and guide improvements in model performance
5. **Work Across Model Providers**: Use a unified interface for models from Hugging Face, OpenAI, Google, Anthropic, and more

The framework is designed to reveal the inner workings of complex language models, supporting various architectures from smaller transformer models to large-scale LLMs across different providers.

## Features

### Multi-Provider Support

- **Unified Interface**: Consistent API for models from:
  - **Hugging Face** (open-source models like BERT, GPT-2, Llama, etc.)
  - **OpenAI** (GPT-3.5, GPT-4, etc.)
  - **Google** (Gemini models)
  - **Anthropic** (Claude models)
  - **Azure OpenAI** (Microsoft-hosted OpenAI models)
  - **Custom Models** (Extensible to other providers)
- **Seamless Integration**: Easily switch between local and API-based models
- **Consistent Outputs**: Standardized format for results across different model types

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

### Loading Models from Different Providers

```python
from src.core.model_loader import ModelWrapper, ModelProvider

# Load a Hugging Face model (open source)
hf_model = ModelWrapper(
    model_name="gpt2",
    task="causal_lm",
    provider=ModelProvider.HUGGINGFACE,
    device="cuda"
)

# Load an OpenAI model (requires API key)
openai_model = ModelWrapper(
    model_name="gpt-3.5-turbo",
    provider=ModelProvider.OPENAI,
    api_key="your-api-key"  # or set OPENAI_API_KEY environment variable
)

# Load a Google model (requires API key)
google_model = ModelWrapper(
    model_name="gemini-pro",
    provider=ModelProvider.GOOGLE,
    api_key="your-api-key"  # or set GOOGLE_API_KEY environment variable
)

# Load an Anthropic model (requires API key)
anthropic_model = ModelWrapper(
    model_name="claude-3-sonnet-20240229",
    provider=ModelProvider.ANTHROPIC,
    api_key="your-api-key"  # or set ANTHROPIC_API_KEY environment variable
)
```

### Text Generation

```python
# Generate text with any model using the same interface
prompt = "Explain artificial intelligence in simple terms:"

# Generate with Hugging Face model
hf_response = hf_model.generate(prompt, max_length=100)
print(f"Hugging Face response: {hf_response}")

# Generate with OpenAI model
openai_response = openai_model.generate(prompt, max_length=100)
print(f"OpenAI response: {openai_response}")
```

### Explainability Analysis

```python
from src.interpretability.llm_explainer import LLMExplainer

# Create an explainer for any model
explainer = LLMExplainer(hf_model)  # Works with any model provider

# Analyze token importance
text = "This movie was fantastic and I would highly recommend it to everyone."
token_importance = explainer.token_importance(text)
explainer.visualize_token_importance(text, token_importance)

# Generate comprehensive explanation
explanation = explainer.explain_generation(text)
print(explanation)
```

### Adversarial Testing

```python
from src.adversarial.attack_generator import AttackGenerator
from src.adversarial.module import AdversarialTester

# Generate adversarial attacks for any model
attack_gen = AttackGenerator(openai_model)  # Works with any model provider
attacks = attack_gen.generate(
    "This is a positive review of a great product.",
    level="word",
    num_attacks=3
)

# Comprehensive testing across providers
adv_tester = AdversarialTester(google_model, techniques=["attack", "counterfactual"])
test_result = adv_tester.test("This product has significantly improved my workflow efficiency.")
```

## Project Structure

```
LingualLens/
├── src/
│   ├── core/                  # Core functionality
│   │   ├── model_loader.py    # Multi-provider model loading and interface
│   │   └── evaluator.py       # Base evaluation classes
│   ├── interpretability/      # Interpretability tools
│   │   ├── concept_extractor.py
│   │   ├── feature_attributor.py
│   │   ├── attention_analyzer.py
│   │   └── llm_explainer.py   # Unified explainability for all models
│   └── adversarial/           # Adversarial testing tools
│       ├── attack_generator.py
│       ├── counterfactual_generator.py
│       ├── robustness_evaluator.py
│       └── module.py          # Combined adversarial tester
├── notebooks/                 # Demo notebooks
│   ├── demo.ipynb             # General framework demo
│   ├── adversarial_testing.ipynb  # Deep dive into adversarial testing
│   ├── black_box_explainability.ipynb  # LLM explainability guide
│   └── multi_provider_demo.ipynb  # Demo for multiple model providers
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
- **multi_provider_demo.ipynb**: Tutorial on using different model providers with a unified interface

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
- **API Integration**: OpenAI 1.0+, Google Generative AI 0.3+, Anthropic 0.5+
- **Datasets**: HuggingFace Datasets 1.10+

For the complete list of dependencies, see [requirements.txt](requirements.txt).

## Contributing

We welcome contributions from the community! See our [contributing guidelines](CONTRIBUTING.md) for details on how to get started.

Areas where contributions are particularly welcome:
- Additional explainability techniques for LLMs
- New adversarial attack methods
- Support for more model providers and architectures
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