# Contributing to Anomaly Detection

First off, thank you for considering contributing to our project!

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include details about your configuration and environment
* Attach relevant configuration files if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in the template
2. Follow the styleguides
3. After you submit your pull request, verify that all status checks are passing

## Development Process

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/ali-izhar/anomaly_detection.git
cd anomaly_detection

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements.txt
```

### Project Structure

When adding new features, please maintain the existing project structure:

```
anomaly_detection/
├── config/                           # experiment configuration setup
├── src/
│   ├── change_point/                 # Martingale-based change point detection algorithms
│   ├── graph/                        # Graph and synthetic data generators
│   ├── models/                       # Model implementations
│   └── utils/                        # Preprocessing and visualization utilities
```

### Code Style

#### Python Style Guide

* Follow PEP 8
* Use type hints
* Write docstrings for all public methods
* Keep functions focused and small
* Use meaningful variable names

Example:
```python
from typing import List, Optional

def process_data(data: List[float], threshold: Optional[float] = None) -> List[float]:
    """Process input data with optional threshold.
    
    Args:
        data: List of float values to process
        threshold: Optional cutoff value
        
    Returns:
        List of processed float values
    """
    if not data:
        return []
    
    if threshold is None:
        threshold = sum(data) / len(data)
        
    return [x for x in data if x > threshold]
```

#### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Testing

* Write tests for all new features
* Maintain or improve test coverage
* Run the test suite before submitting PR:
```bash
pytest tests/
```

### Documentation

* Update README.md if needed
* Add docstrings to all new functions/classes
* Update configuration templates if needed
* Add example usage for new features

## Additional Notes

### Issue and Pull Request Labels

* `bug`: Something isn't working
* `enhancement`: New feature or request
* `documentation`: Documentation only changes
* `good first issue`: Good for newcomers
* `help wanted`: Extra attention is needed

### Communication

* Use GitHub issues for bug reports and feature requests
* Use pull requests for code review discussions
* Tag maintainers when needed

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## Questions?

Feel free to contact the maintainers if you have any questions.

Thank you for contributing to anomaly detection!

