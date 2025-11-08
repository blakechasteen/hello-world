# Contributing to Promptly

Thank you for your interest in contributing to Promptly! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation](#documentation)
9. [Community](#community)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in Promptly a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior**:
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations can be reported to the project team at [conduct@promptly.dev](mailto:conduct@promptly.dev). All complaints will be reviewed and investigated promptly and fairly.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- OpenAI API key (or other LLM provider)
- Basic understanding of prompt engineering (optional)

### First-Time Contributors

**Great first issues**:
1. Documentation improvements
2. Example workflows
3. Test coverage
4. Bug fixes with clear reproduction steps

**Labels to look for**:
- `good first issue` - Beginner-friendly tasks
- `help wanted` - Community contributions welcome
- `documentation` - Documentation improvements
- `bug` - Bug fixes needed

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/promptly.git
cd promptly

# Add upstream remote
git remote add upstream https://github.com/promptly/promptly.git
```

### 2. Create Virtual Environment

```bash
# Create virtualenv
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install Promptly in editable mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install DSPy
pip install dspy-ai

# Optional: Install HoloLoom dependencies
pip install torch numpy gymnasium matplotlib
pip install spacy sentence-transformers scipy networkx
python -m spacy download en_core_web_sm
```

### 4. Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
export OPENAI_API_KEY="your-key-here"
```

### 5. Verify Setup

```bash
# Run verification script
python verify_dspy_installation.py

# Run tests
pytest HoloLoom/tests/integration/test_dspy_integration.py -v
```

### 6. Create Feature Branch

```bash
# Create branch for your work
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

---

## How to Contribute

### Types of Contributions

**1. Bug Reports**

When reporting bugs, include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, dependencies)
- Error messages/stack traces
- Minimal code example

**Template**:
```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: Windows 11 / macOS 14 / Ubuntu 22.04
- Python: 3.10.5
- Promptly: v1.0.0
- DSPy: v2.4.0

**Error Message**
```
Paste error message here
```

**Minimal Example**
```python
# Minimal code to reproduce
```
```

**2. Feature Requests**

When requesting features, include:
- Clear use case and problem statement
- Proposed solution
- Alternative solutions considered
- Willingness to implement (if applicable)

**Template**:
```markdown
**Problem Statement**
As a [user type], I want [feature] so that [benefit]

**Proposed Solution**
Describe your proposed solution

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information

**Implementation**
[ ] I'm willing to implement this feature
[ ] I need help implementing this
```

**3. Code Contributions**

**Areas needing contributions**:
- New problem solvers (beyond the core 6)
- Additional LLM provider integrations
- Performance optimizations
- Test coverage improvements
- Documentation examples
- Integration with other tools

**4. Documentation Contributions**

**Types of documentation**:
- Tutorials and guides
- API reference improvements
- Example workflows
- Architecture diagrams
- Video demonstrations
- Translations

**5. Community Support**

- Answer questions on GitHub Discussions
- Help newcomers get started
- Review pull requests
- Share your use cases and workflows

---

## Pull Request Process

### 1. Before You Start

- Search existing issues/PRs to avoid duplicates
- Open an issue first for significant changes
- Get feedback on your approach before implementing

### 2. Development Workflow

```bash
# Keep your fork up to date
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ... edit files ...

# Run tests
pytest HoloLoom/tests/ -v

# Run linters
black HoloLoom/promptly/
flake8 HoloLoom/promptly/
mypy HoloLoom/promptly/

# Commit changes
git add .
git commit -m "feat: Add feature description"

# Push to your fork
git push origin feature/your-feature
```

### 3. Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

**Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests
- `chore`: Build process, dependencies

**Examples**:
```bash
feat(schema): Add nested object support to schema builder

Implement recursive schema parsing for nested objects.
Adds validation for circular references.

Closes #123
```

```bash
fix(surgical): Preserve whitespace in surgical edits

Fixed issue where leading/trailing whitespace was
stripped during surgical edits.

Fixes #456
```

### 4. Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issue
Closes #issue_number

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings introduced
- [ ] Tests added for new functionality
- [ ] All tests passing

## Screenshots (if applicable)
Add screenshots for UI changes
```

### 5. Review Process

**What to expect**:
1. Automated checks run (tests, linters)
2. Maintainer reviews code (usually within 48 hours)
3. Feedback provided via review comments
4. You address feedback, push updates
5. Once approved, maintainer merges PR

**Review criteria**:
- Code quality and clarity
- Test coverage
- Documentation completeness
- Backward compatibility
- Performance impact
- Security considerations

---

## Coding Standards

### Python Style Guide

**Follow PEP 8** with these additions:

**Imports**:
```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import dspy
import numpy as np

# Local
from HoloLoom.promptly import DSPyHoloLoom
from HoloLoom.config import Config
```

**Type Hints**:
```python
# Always use type hints
def optimize_prompt(
    signature: DSPySignature,
    examples: List[Dict[str, Any]],
    config: Optional[Config] = None
) -> DSPyProgram:
    """Optimize prompt using examples."""
    ...
```

**Docstrings** (Google style):
```python
def create_signature(
    instruction: str,
    inputs: List[str],
    outputs: List[str]
) -> DSPySignature:
    """Create a DSPy signature from instruction and fields.

    Args:
        instruction: Natural language instruction describing the task
        inputs: List of input field names
        outputs: List of output field names

    Returns:
        DSPySignature ready to use with DSPy programs

    Raises:
        ValueError: If inputs or outputs are empty

    Example:
        >>> sig = create_signature(
        ...     "Answer questions",
        ...     inputs=["question"],
        ...     outputs=["answer"]
        ... )
    """
    ...
```

**Error Handling**:
```python
# Be specific with exceptions
try:
    result = optimize_from_memory(signature, query)
except ValueError as e:
    logger.error(f"Invalid signature: {e}")
    raise
except Exception as e:
    logger.error(f"Optimization failed: {e}")
    raise OptimizationError(f"Failed to optimize: {e}") from e
```

**Logging**:
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debugging information")
logger.info("General informational message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error")
```

### Code Organization

**File structure**:
```
HoloLoom/promptly/
├── solvers/              # Problem solvers (core logic)
│   ├── schema/
│   ├── surgical/
│   └── ...
├── core/                 # Core primitives
├── state/                # State management
├── execution/            # Execution engines
├── orchestration/        # Workflow orchestration
├── interfaces/           # User interfaces
└── tests/                # Tests mirror source structure
```

**Module structure**:
```python
"""Module docstring describing purpose."""

# Imports
import ...

# Constants
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3

# Type definitions
class MyType:
    ...

# Functions (public first, private last)
def public_function():
    ...

def _private_helper():
    ...
```

---

## Testing Guidelines

### Test Organization

```
HoloLoom/promptly/tests/
├── unit/               # Fast isolated tests (<5s)
├── integration/        # Multi-component tests (<30s)
└── e2e/                # Full pipeline tests (<2min)
```

### Writing Tests

**Use pytest**:
```python
import pytest
from HoloLoom.promptly import create_signature

def test_create_signature_basic():
    """Test basic signature creation."""
    sig = create_signature(
        "Answer questions",
        inputs=["question"],
        outputs=["answer"]
    )

    assert sig.instruction == "Answer questions"
    assert "question" in sig.input_fields
    assert "answer" in sig.output_fields


def test_create_signature_invalid():
    """Test signature creation with invalid inputs."""
    with pytest.raises(ValueError, match="inputs cannot be empty"):
        create_signature("Test", inputs=[], outputs=["out"])


@pytest.mark.asyncio
async def test_optimize_from_memory():
    """Test memory-based optimization."""
    # Setup
    config = Config.fast()
    bridge = DSPyHoloLoom(config=config)

    # Execute
    result = await bridge.optimize_from_memory(...)

    # Assert
    assert result.success
    assert result.optimized_program is not None
```

**Test naming**:
- `test_<function>_<scenario>()` for unit tests
- `test_<feature>_integration()` for integration tests
- `test_<workflow>_e2e()` for end-to-end tests

**Fixtures**:
```python
import pytest
from HoloLoom.config import Config

@pytest.fixture
def config():
    """Provide test configuration."""
    return Config.fast()

@pytest.fixture
def sample_shards():
    """Provide sample memory shards."""
    return [
        MemoryShard(content="Test content 1", ...),
        MemoryShard(content="Test content 2", ...),
    ]

def test_with_fixtures(config, sample_shards):
    """Test using fixtures."""
    assert config.mode == ExecutionMode.FAST
    assert len(sample_shards) == 2
```

### Running Tests

```bash
# All tests
pytest HoloLoom/promptly/tests/ -v

# Specific test file
pytest HoloLoom/promptly/tests/unit/test_schema.py -v

# Specific test function
pytest HoloLoom/promptly/tests/unit/test_schema.py::test_create_signature -v

# With coverage
pytest HoloLoom/promptly/tests/ --cov=HoloLoom.promptly --cov-report=html

# Fast tests only (unit)
pytest HoloLoom/promptly/tests/unit/ -v

# Mark slow tests
@pytest.mark.slow
def test_expensive_operation():
    ...

# Skip slow tests
pytest -m "not slow"
```

### Coverage Requirements

- **Minimum**: 80% overall coverage
- **New code**: 90%+ coverage required
- **Critical paths**: 100% coverage (authentication, payment, data loss)

---

## Documentation

### Types of Documentation

**1. Code Documentation**

```python
# Module docstrings
"""Schema builder for structured output generation.

This module implements the schema-first approach to solving
the Projection Trap. It provides tools for building JSON schemas
and generating prompts that enforce those schemas.

Example:
    >>> builder = SchemaBuilder()
    >>> schema = builder.add_field("name", FieldType.STRING, required=True)
    >>> prompt = builder.generate_prompt("Extract person info")
"""

# Class docstrings
class SchemaBuilder:
    """Builder for JSON schemas with validation rules.

    Provides a fluent interface for constructing schemas and
    generating schema-constrained prompts.

    Attributes:
        fields: List of schema fields
        validation_rules: Dict of field-specific validation rules

    Example:
        >>> builder = SchemaBuilder()
        >>> builder.add_field("age", FieldType.NUMBER, required=True)
        >>> schema = builder.build()
    """

# Function docstrings (see Google style above)
```

**2. User Documentation**

- **Tutorials**: Step-by-step guides for common tasks
- **How-to guides**: Solutions to specific problems
- **Explanations**: Conceptual deep dives
- **Reference**: API documentation

**3. Architecture Documentation**

- System diagrams
- Component interactions
- Design decisions (ADRs)
- Performance characteristics

### Documentation Style

**Be clear and concise**:
```markdown
# Bad
This functionality enables users to engage with the system via the CLI.

# Good
Use the CLI to interact with Promptly.
```

**Use examples**:
```markdown
# Bad
The schema builder supports nested objects.

# Good
Create nested schemas:
```python
schema = builder.add_field("address", FieldType.OBJECT)
schema.add_nested_field("address", "street", FieldType.STRING)
schema.add_nested_field("address", "city", FieldType.STRING)
```
```

**Keep it updated**:
- Update docs in the same PR as code changes
- Mark deprecated features with `[DEPRECATED]`
- Add migration guides for breaking changes

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Q&A, general discussion
- **Discord** (coming soon): Real-time chat
- **Twitter** (coming soon): Updates and announcements

### Getting Help

**Before asking**:
1. Search existing issues/discussions
2. Check documentation
3. Review examples
4. Try debugging yourself

**When asking**:
- Be specific and detailed
- Provide minimal reproducible example
- Show what you've tried
- Be patient and respectful

### Recognition

**Contributors are recognized via**:
- GitHub contributor graph
- CONTRIBUTORS.md file
- Release notes
- Social media shoutouts (with permission)

**Top contributors may receive**:
- Maintainer status
- Early access to features
- Promptly swag
- Recognition in talks/blog posts

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Cycle

- **Patch releases**: As needed (bug fixes)
- **Minor releases**: Monthly (new features)
- **Major releases**: Annually (breaking changes)

---

## License

By contributing to Promptly, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

- Open a [GitHub Discussion](https://github.com/promptly/promptly/discussions)
- Email us at [hello@promptly.dev](mailto:hello@promptly.dev)
- Join our [Discord](https://discord.gg/promptly) (coming soon)

---

**Thank you for contributing to Promptly!**

We're building the future of AI reliability together.
