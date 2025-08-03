# Contributing to Data Quality Management Application

Thank you for your interest in contributing to our project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Write tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/litedatum/validatelite.git
cd validatelite
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
pre-commit install
```
> `requirements-dev.txt` includes all production and development dependencies (testing, linting, type checking, etc.).

## Code Style

We use several tools to maintain code quality:

- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all checks before submitting:
```bash
black .
isort .
flake8
mypy .
```

## Testing

Write tests for all new features and bug fixes. Run tests with:
```bash
pytest
```

## Documentation

- Update documentation for any new features or changes
- Follow the existing documentation style
- Use clear and concise language

## Pull Request Process

1. Ensure your code passes all tests and checks
2. Update documentation as needed
3. Provide a clear description of your changes
4. Reference any related issues
5. Wait for review and address any feedback

## Issue Reporting

When reporting issues:
- Use the provided issue template
- Provide detailed steps to reproduce
- Include relevant error messages
- Specify your environment details

## Feature Requests

For feature requests:
- Explain the problem you're trying to solve
- Provide use cases and examples
- Suggest potential solutions

## Questions and Discussion

For questions and discussions:
- Use the project's issue tracker
- Join our community chat
- Check the documentation first

## Branch Naming Convention

- Use `feature/<short-description>` for new features
- Use `fix/<short-description>` for bug fixes
- Use `docs/<short-description>` for documentation updates
- Use `chore/<short-description>` for maintenance

## Commit Message Guidelines

- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
- Example: `feat: add new rule parser for custom checks`
- Example: `fix: correct SQL dialect detection logic`

## Pull Request Requirements

- PRs must pass all CI checks and code review before merging
- Reference related issues in the PR description
- Use clear, descriptive titles
- Update CHANGELOG.md for user-facing changes

## Communication

- For questions, use GitHub Discussions or open an issue
- For urgent matters, contact maintainers at datapebble@gmail.com

Thank you for contributing!
