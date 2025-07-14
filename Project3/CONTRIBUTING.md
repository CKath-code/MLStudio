# Contributing to Used Car Price Prediction MLOps

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/used-car-price-prediction-mlops.git
   cd used-car-price-prediction-mlops
   ```

2. **Environment Setup**
   ```bash
   conda env create -f env/conda.yml
   conda activate sklearn-env
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Maintain test coverage above 80%

## Testing

Run tests before submitting:
```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, descriptive commits
3. Add tests for new functionality
4. Update documentation as needed
5. Submit pull request with detailed description

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## Questions?

Open an issue or start a discussion for any questions about contributing.
