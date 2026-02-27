# Contributing to ML Financial Forecaster

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/ARASH3280ARASH/ml-financial-forecaster.git
cd ml-financial-forecaster
pip install -e ".[all]"
```

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ -v --cov=src --cov-report=html
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for all public classes and methods (Google style)
- Keep functions focused and under 50 lines where possible

## Adding a New Model

1. Create a new file in `src/models/`
2. Inherit from `BaseModel` and implement `_fit()` and `_predict()`
3. Add tests in `tests/test_models.py`
4. Register in `src/models/__init__.py`

## Adding New Features

1. Create an engine class in `src/features/`
2. Implement a `compute_all(df)` method returning a DataFrame
3. Add tests in `tests/test_features.py`
4. Document the feature set in `docs/architecture.md`

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a PR with a clear description of changes

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if bug)
- Expected vs actual behaviour
- Python version and OS
