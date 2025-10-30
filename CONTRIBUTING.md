# Contributing to HoloLoom

**Thank you for considering contributing to HoloLoom!** 🎉

We welcome contributions from everyone. Whether you're fixing bugs, adding features, improving documentation, or benchmarking performance, your help is appreciated.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [How to Contribute](#how-to-contribute)
5. [Contribution Guidelines](#contribution-guidelines)
6. [Testing](#testing)
7. [Code Style](#code-style)
8. [Commit Messages](#commit-messages)
9. [Pull Request Process](#pull-request-process)
10. [Community](#community)

---

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [blakechasteen@users.noreply.github.com].

---

## Getting Started

### Ways to Contribute

- 🐛 **Bug fixes** - Fix issues, improve stability
- ✨ **New features** - Add capabilities (after discussion)
- 📊 **Benchmarking** - Validate claims, compare approaches
- 📚 **Documentation** - Improve guides, examples, tutorials
- 🎨 **Visualizations** - Enhance dashboards, add charts
- 🧪 **Testing** - Increase coverage, add integration tests
- 🚀 **Performance** - Optimize speed, reduce memory

### Where to Start

1. **Good first issues**: Look for [`good first issue`](https://github.com/blakechasteen/mythRL/labels/good%20first%20issue) label
2. **Documentation**: Improve README, add examples, fix typos
3. **Tests**: Add missing tests (see [V1_REFINEMENT_PASSES.md](V1_REFINEMENT_PASSES.md) for gaps)
4. **Benchmarks**: Validate claims (e.g., "10-20% better after 100 queries")

---

## Development Setup

### Prerequisites

- Python 3.10+ (3.10, 3.11, 3.12 tested)
- Git
- Virtual environment tool (venv, conda, virtualenv)

### Setup Instructions

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/mythRL.git
cd mythRL

# 3. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. Install in development mode
pip install -e ".[dev]"  # Includes dev dependencies

# 5. Install optional dependencies (if needed)
pip install -e ".[nlp]"        # For Phase 5 Universal Grammar
pip install -e ".[production]" # For Neo4j + Qdrant
pip install -e ".[all]"        # Everything

# 6. Download spaCy model (if using NLP features)
python -m spacy download en_core_web_sm

# 7. Run tests to verify setup
pytest HoloLoom/tests/ -v
```

### Verify Installation

```bash
# Quick test
python test_v1_simplification.py

# Expected output: ✅ ALL TESTS PASSED
```

---

## How to Contribute

### 1. Pick an Issue

- Browse [open issues](https://github.com/blakechasteen/mythRL/issues)
- Comment: "I'd like to work on this"
- Wait for maintainer response (usually <48 hours)

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/issue-123-bug-description
```

**Branch naming**:
- `feature/multi-scale-benchmarks` - New features
- `fix/thompson-sampling-bug` - Bug fixes
- `docs/improve-quickstart` - Documentation
- `test/recursive-learning-integration` - Tests
- `perf/optimize-embeddings` - Performance

### 3. Make Changes

- Write code following [style guidelines](#code-style)
- Add tests for new functionality
- Update documentation if needed
- Run tests locally before committing

### 4. Commit Changes

```bash
git add .
git commit -m "feat: Add multi-scale benchmark suite"
```

See [Commit Messages](#commit-messages) for format.

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

---

## Contribution Guidelines

### Philosophy

HoloLoom follows these principles (from [V1_SIMPLIFICATION_COMPLETE.md](V1_SIMPLIFICATION_COMPLETE.md)):

1. **Ship simple, iterate based on data, benchmark always**
2. **Simplicity over features**
3. **Proven over speculative**
4. **Maintainable over clever**

### Decision Framework

**When to add a feature** (ALL must be true):
- ✅ Benchmark shows >10% improvement
- ✅ Complexity justified by benefit
- ✅ User demand (multiple requests)
- ✅ Fits architectural philosophy
- ✅ Maintainable long-term

**When to reject a feature** (ANY can disqualify):
- ❌ Benchmark shows <10% improvement
- ❌ Adds complexity without clear benefit
- ❌ No user demand
- ❌ Violates architectural principles
- ❌ Unsustainable to maintain

**Default stance**: No, unless proven necessary.

### Feature Proposals

**Before implementing a feature**:

1. Open a GitHub Issue with label `feature-request`
2. Include:
   - Problem statement (what does it solve?)
   - Expected benefit (quantify if possible)
   - Benchmark data (if available)
   - How it fits HoloLoom philosophy
3. Discuss with maintainers (may take a few days)
4. Wait for approval before implementing

**Example**: See [FUTURE_WORK.md](FUTURE_WORK.md) for feature proposal template.

---

## Testing

### Running Tests

```bash
# All tests
pytest HoloLoom/tests/ -v

# Specific test file
pytest HoloLoom/tests/unit/test_unified_policy.py -v

# With coverage
pytest HoloLoom/tests/ --cov=HoloLoom --cov-report=html

# v1.0 simplification tests
python test_v1_simplification.py
```

### Writing Tests

**Test structure**:
```
HoloLoom/tests/
├── unit/           # Fast (<5s), isolated component tests
├── integration/    # Medium (<30s), multi-component tests
└── e2e/            # Slow (<2min), full pipeline tests
```

**Test requirements**:
- ✅ All new code must have tests
- ✅ Tests must pass before PR is merged
- ✅ Aim for 85%+ code coverage
- ✅ Use pytest + pytest-asyncio
- ✅ Mock external dependencies (no API calls in tests)

**Example test**:
```python
# HoloLoom/tests/unit/test_embeddings.py

import pytest
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

def test_single_scale_embeddings():
    """Test that v1.0 uses single-scale [768]."""
    emb = MatryoshkaEmbeddings()

    assert emb.sizes == [768], "Should use single-scale"
    assert emb.base_dim == 768, "Base dimension should be 768"

    # Encode test
    texts = ["Test sentence"]
    result = emb.encode(texts)

    assert result.shape == (1, 768), "Should produce 768d embeddings"
```

---

## Code Style

### Python Style

**Follow PEP 8** with these specifics:

- **Line length**: 100 characters (not 80)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes `"` for strings
- **Imports**: Grouped (stdlib, third-party, local)
- **Type hints**: Use for public APIs
- **Docstrings**: Google style

**Example**:
```python
from typing import List, Optional
import numpy as np

from HoloLoom.documentation.types import Vector


def encode_texts(
    texts: List[str],
    normalize: bool = True
) -> np.ndarray:
    """
    Encode texts to vectors.

    Args:
        texts: List of text strings to encode
        normalize: Whether to L2-normalize embeddings

    Returns:
        Matrix of embeddings (n_texts × embedding_dim)

    Example:
        >>> texts = ["Hello world"]
        >>> embeddings = encode_texts(texts)
        >>> embeddings.shape
        (1, 768)
    """
    # Implementation
    pass
```

### Formatting Tools

**Use Black** (official HoloLoom formatter):
```bash
# Format code
black HoloLoom/

# Check formatting
black --check HoloLoom/
```

**Use Ruff** (linter):
```bash
# Lint code
ruff check HoloLoom/

# Auto-fix
ruff check --fix HoloLoom/
```

### Type Checking

**Use mypy** (optional but recommended):
```bash
mypy HoloLoom/ --ignore-missing-imports
```

---

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `perf`: Performance improvements
- `refactor`: Code refactoring
- `style`: Code style changes (formatting)
- `chore`: Maintenance tasks

### Examples

**Good commit messages**:
```
feat(embeddings): Add Nomic v1.5 support

Upgrade default embedding model from all-MiniLM-L12-v2 (2021)
to nomic-ai/nomic-embed-text-v1.5 (2024).

Results:
- +10-15% MTEB score improvement
- 32x longer context (8192 vs 256 tokens)
- Modern 2024 architecture

Closes #42
```

```
fix(thompson-sampling): Correct bandit update logic

Previously, bandit was updating statistics for the wrong tool
(predicted tool instead of actually selected tool).

Now correctly updates α/β for the tool that was executed.

Fixes #123
```

**Bad commit messages**:
```
fix stuff
update code
wip
changes
```

### Co-Authorship

If collaborating:
```
feat: Add multi-scale benchmarks

Co-authored-by: Name <email@example.com>
```

---

## Pull Request Process

### Before Submitting

✅ Checklist:
- [ ] Code follows style guidelines (Black, Ruff)
- [ ] Tests added for new functionality
- [ ] All tests pass locally (`pytest`)
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow format
- [ ] Branch is up-to-date with master

### PR Title Format

```
<type>: <Short description>
```

Examples:
- `feat: Add multi-scale embedding benchmarks`
- `fix: Correct Thompson Sampling bandit updates`
- `docs: Improve quickstart guide with examples`

### PR Description Template

```markdown
## Summary
Brief description of changes (1-3 sentences).

## Motivation
Why is this change necessary? What problem does it solve?

## Changes
- Bullet list of specific changes
- Each change on separate line

## Testing
- How was this tested?
- What edge cases were considered?

## Screenshots (if applicable)
Add screenshots/gifs for UI changes

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Follows code style
- [ ] Backward compatible (or migration guide added)

## Related Issues
Closes #123
Relates to #456
```

### Review Process

1. **Automated checks** (GitHub Actions):
   - Tests must pass
   - Code style must pass (Black, Ruff)
   - Coverage must not decrease

2. **Maintainer review** (usually 1-3 days):
   - Code quality
   - Architectural fit
   - Test coverage
   - Documentation

3. **Feedback addressed**:
   - Make requested changes
   - Push updates to same branch
   - Re-request review

4. **Merge**:
   - Maintainer merges (squash or rebase)
   - Branch deleted
   - Closes related issues

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, show-and-tell
- **Pull Requests**: Code contributions

### Getting Help

**Questions**:
- Check [README.md](README.md) and [documentation](docs/)
- Search [existing issues](https://github.com/blakechasteen/mythRL/issues)
- Open a [GitHub Discussion](https://github.com/blakechasteen/mythRL/discussions)

**Bugs**:
- Search [existing issues](https://github.com/blakechasteen/mythRL/issues)
- Open new issue with template

**Feature requests**:
- Check [FUTURE_WORK.md](FUTURE_WORK.md)
- Open GitHub Issue with `feature-request` label

### Recognition

**Contributors** are recognized in:
- GitHub contributors graph
- Release notes
- `CONTRIBUTORS.md` (coming in v1.1)

**Significant contributions** may receive:
- Co-authorship in commit messages
- Mention in release announcements
- Invitation to maintainer team (for sustained contributions)

---

## Roadmap

See [FUTURE_WORK.md](FUTURE_WORK.md) for planned features and priorities.

**High-priority areas** (v1.1):
1. **Benchmarking**: Multi-scale, quality trajectories, long-term learning
2. **Testing**: Integration tests for recursive learning, Thompson Sampling
3. **Documentation**: Example projects, tutorials, video walkthroughs
4. **Packaging**: Docker container, PyPI package
5. **Web UI**: Real-time learning dashboard

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## Questions?

Feel free to reach out:
- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions and ideas
- **Email**: blakechasteen@users.noreply.github.com

**Thank you for contributing to HoloLoom!** 🚀

---

**Built with care by developers who believe AI should learn from you, not just respond to you.**