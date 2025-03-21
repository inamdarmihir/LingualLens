# Contributing to Language Model Evaluation Framework

Thank you for your interest in contributing to our project! This document provides guidelines for contributions.

## Getting Started

1. **Fork the repository** to your own GitHub account
2. **Clone your fork** to your local machine
3. **Create a new branch** for your changes
4. **Make your changes** following the coding guidelines
5. **Submit a pull request** from your branch to our main branch

## Development Environment

1. Install Python 3.8 or higher
2. Install development dependencies:
   ```
   pip install -r requirements.txt
   pip install -r dev-requirements.txt  # If it exists
   ```
3. Install pre-commit hooks:
   ```
   pre-commit install
   ```

## Coding Guidelines

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Document functions, classes, and modules using docstrings
- Keep functions small and focused on a single task
- Write descriptive variable names

### Testing

- Write unit tests for new features using pytest
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage for new code

### Git Workflow

- Keep pull requests focused on a single feature or bug fix
- Write clear commit messages that explain the purpose of the change
- Reference issue numbers in commit messages when applicable
- Rebase your branch on the latest main before submitting a PR

## Adding New Features

When adding new features to the framework, please ensure:

1. **Documentation**: Every new feature must include documentation
2. **Tests**: Include tests demonstrating the feature works as expected
3. **Examples**: Add example usage to relevant notebooks
4. **Backward Compatibility**: Ensure new features don't break existing functionality

## Components You Can Contribute To

- **Core Module**: Improvements to `ModelWrapper` or `Evaluator` classes
- **Interpretability Module**: New techniques for model interpretation
- **Adversarial Testing**: New attack types or counterfactual generation methods
- **Visualization**: Better ways to visualize model behavior and evaluation results
- **Documentation**: Improved notebooks, guides, or tutorials
- **Testing**: Enhanced test coverage or test infrastructure

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add your name to CONTRIBUTORS.md (if it exists)
4. Submit the pull request with a clear description of the changes
5. Address any feedback from code reviewers

## Code of Conduct

- Be respectful of other contributors
- Provide constructive feedback
- Help maintain a positive, inclusive community

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

If you have questions about contributing, please open an issue with the "question" label. 