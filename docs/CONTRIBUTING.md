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
8. [Pre-Commit Hooks](#pre-commit-hooks)
9. [Commit Messages](#commit-messages)
10. [Pull Request Process](#pull-request-process)
11. [SaaS Toolkit for Ecosystem Developers](#saas-toolkit-for-ecosystem-developers)
12. [Safety Research](#safety-research)
13. [Community](#community)

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
- 🏗️ **Ecosystem Apps** - Build apps using the SaaS toolkit
- 🔒 **Safety Research** - Improve alignment framework

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

## Pre-Commit Hooks

### What is Pre-Commit?

Pre-commit is an automated code quality framework that runs checks **before every commit**. It ensures:
- Code formatting consistency (Black, isort)
- Syntax errors are caught early
- Security issues are detected
- Commit messages follow conventions
- No accidental commits to main/master

### Setup

**Install pre-commit** (one-time setup):

```bash
# 1. Install pre-commit
pip install pre-commit

# 2. Install git hooks from .pre-commit-config.yaml
pre-commit install
pre-commit install --hook-type commit-msg  # For commit message validation

# 3. (Optional) Run on all files to fix existing issues
pre-commit run --all-files
```

### Available Hooks

**Automatic Fixes** (pre-commit will fix these for you):
- `trailing-whitespace` - Remove trailing spaces
- `end-of-file-fixer` - Ensure files end with newline
- `black` - Code formatting
- `isort` - Import sorting
- `ruff --fix` - Auto-fix linting issues
- `markdownlint --fix` - Fix markdown formatting
- `mixed-line-ending` - Convert to consistent line endings

**Manual Review Required** (pre-commit will show errors, you must fix them):
- `check-yaml` - YAML syntax errors
- `check-json` - JSON syntax errors
- `check-ast` - Python syntax errors
- `detect-private-key` - Accidentally committed secrets
- `check-merge-conflict` - Unresolved merge conflicts
- `conventional-pre-commit` - Commit message format

**Slow Checks** (run manually or in CI, skipped on commit):
- `mypy` - Type checking (slow)
- `bandit` - Security analysis (slow)

### Common Workflows

**Fix code before committing**:
```bash
# Pre-commit runs automatically on git commit
# If hooks fail, they auto-fix what they can, then:
git add .
git commit -m "feat: Add new feature"
```

**Run specific hooks manually**:
```bash
# Run all hooks on changed files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files

# Run slow type checking (not in pre-commit by default)
pre-commit run mypy --hook-stage manual --all-files

# Run slow security checks
pre-commit run bandit --hook-stage manual --all-files
```

**Skip pre-commit for an emergency fix** (use sparingly):
```bash
git commit --no-verify -m "hotfix: Critical bug"  # ⚠️ Not recommended
```

**Update pre-commit hooks to latest versions**:
```bash
pre-commit autoupdate
```

### Configuration

The main configuration is in `.pre-commit-config.yaml`. Key settings:

- **Python version**: `python3.8` (minimum supported)
- **Line length**: 100 characters (not 80)
- **Black profile for isort**: Ensures compatibility
- **Fail fast disabled**: Shows all issues, not just first

Additional tool configs (Black, isort, ruff, mypy) can go in:
- `pyproject.toml` (recommended - TOML format)
- `.flake8` (Ruff config)
- `.isort.cfg` (isort config)
- `mypy.ini` (mypy config)

Example `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ["py38"]

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
target-version = "py38"
select = ["F", "E", "W", "I", "C"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
ignore_missing_imports = true
```

### Troubleshooting

**Pre-commit rejects my code**:
1. Read the error message carefully
2. Most errors are auto-fixed (just `git add` and commit again)
3. For syntax errors, fix manually and retry

**Pre-commit is too slow**:
```bash
# Skip pre-commit (only when needed)
git commit --no-verify

# Or run only essential hooks
pre-commit run trailing-whitespace end-of-file-fixer check-yaml
```

**Hooks not running**:
```bash
# Verify hooks are installed
cat .git/hooks/pre-commit

# Reinstall if missing
pre-commit install
```

**Modified commit message format**:
```bash
# Pre-commit validates commit messages against conventional-commit
# Valid format: <type>(<scope>): <description>
# Example: feat(embeddings): Add Nomic v1.5 support

# Invalid: fix stuff
# Valid: fix(orchestrator): Correct bandit update logic
```

### For CI/CD Pipelines

Add to GitHub Actions (`.github/workflows/code-quality.yml`):

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - uses: pre-commit/action@v3
        with:
          extra_dependencies: ["safety"]  # Optional security check
```

This runs **all hooks** (including slow manual checks) on every PR.

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

## SaaS Toolkit for Ecosystem Developers

HoloLoom's SaaS Toolkit (`HoloLoom/saas/`) is designed for ecosystem developers who want to build their own applications on top of HoloLoom. The toolkit is modular - use only what you need.

### What's Included

The SaaS Toolkit provides production-ready components:

| Component | Purpose | Optional? |
|-----------|---------|-----------|
| **Authentication** | API keys, customer management | Core |
| **Usage Tracking** | Query counts, token usage | Optional |
| **Billing** | Stripe integration, subscriptions | Optional |
| **Audit Logging** | Event tracking, compliance | Optional |

### Quick Start

**1. Auth Only** (simplest):
```python
from HoloLoom.saas import SaaSConfig, create_saas_backend

config = SaaSConfig(
    sqlite_path="./data/myapp.db",
    fallback_to_sqlite=True,
    enable_usage_tracking=False,
    enable_billing=False,
)
backend = create_saas_backend(config)
```

**2. Auth + Usage Tracking**:
```python
config = SaaSConfig(
    sqlite_path="./data/myapp.db",
    enable_usage_tracking=True,  # Track queries/tokens
    enable_billing=False,
)
```

**3. Full Billing**:
```python
config = SaaSConfig(
    host="localhost",
    port=5432,
    database="myapp",
    enable_usage_tracking=True,
    enable_billing=True,
    stripe_api_key=os.getenv("STRIPE_API_KEY"),
)
```

### Example Applications

See `HoloLoom/saas/examples/` for complete working examples:

1. **auth_only_app.py** - Minimal authentication (SQLite, no dependencies)
2. **usage_tracking_app.py** - Auth + usage analytics (no billing)
3. **full_billing_app.py** - Complete Stripe billing integration

Run any example:
```bash
PYTHONPATH=. uvicorn HoloLoom.saas.examples.auth_only_app:app --reload
```

### Building Your Own App

**Step 1**: Create your FastAPI app with SaaS routes:
```python
from fastapi import FastAPI, Depends
from HoloLoom.saas import SaaSConfig, create_saas_backend
from HoloLoom.saas.auth import validate_api_key, AuthContext
from HoloLoom.saas.routes import customers_router, api_keys_router

app = FastAPI(title="My HoloLoom App")

# Mount SaaS routes
app.include_router(customers_router, tags=["customers"])
app.include_router(api_keys_router, tags=["api-keys"])

# Your protected endpoints
@app.post("/api/v1/query")
async def query(auth: AuthContext = Depends(validate_api_key)):
    # auth.customer_id, auth.plan available
    return {"customer": auth.customer_id}
```

**Step 2**: Add usage tracking (optional):
```python
@app.post("/api/v1/query")
async def query(data: dict, auth: AuthContext = Depends(validate_api_key)):
    # Track usage
    await backend.record_usage(
        customer_id=auth.customer_id,
        queries_delta=1,
        tokens_delta=len(str(data)) * 4
    )
    # Your logic here
    return {"status": "success"}
```

### Contributing to the Toolkit

We welcome contributions to the SaaS toolkit:

1. **New integrations** - Payment providers, usage exporters
2. **Bug fixes** - Security issues, edge cases
3. **Documentation** - Examples, tutorials
4. **Testing** - Integration tests, edge cases

**Before contributing**:
- Read the existing code in `HoloLoom/saas/`
- Check `HoloLoom/saas/README.md` for architecture
- Open an issue to discuss significant changes

---

## Safety Research

HoloLoom's Alignment Framework is open source because **safety mechanisms must be open for inspection**.

### What's Included

The alignment framework (`HoloLoom/alignment/`) provides:

| Component | Purpose | Overhead |
|-----------|---------|----------|
| **Safety Guardrails** | Risk-based action gating | 0.039 ms |
| **Deception Detection** | Goal transparency tracking | 0.034 ms |
| **Instrumental Convergence** | Power-seeking prevention | 0.015 ms |
| **Audit Trail** | Complete decision provenance | 0.015 ms |

**Total overhead: 0.103 ms** - Safety should not compromise performance.

### How to Contribute

**1. Use the Framework**:
```python
from HoloLoom.alignment import SafetyGuardrails, AuditTrail

guardrails = SafetyGuardrails(enable_human_in_loop=True)
audit_trail = AuditTrail()

# Gate actions through safety checks
result = await guardrails.gate_action(action, context)
await audit_trail.log_decision(query, action, outcome)
```

**2. Report Gaps**:
- Found a pattern the guardrails miss? Open an issue
- Discovered an edge case? Submit a test case
- Have a better heuristic? Propose it with benchmarks

**3. Improve Detection**:
- Add new adversarial patterns to `safety_guardrails.py`
- Improve deception detection heuristics
- Extend the audit trail format

**4. Research and Publish**:
- Use HoloLoom as a platform for safety research
- Publish findings on what works
- Share datasets of adversarial patterns

### Key Files

- `HoloLoom/alignment/safety_guardrails.py` - Risk classification and gating
- `HoloLoom/alignment/deception_detection.py` - Goal transparency
- `HoloLoom/alignment/audit_trail.py` - Decision provenance
- `HoloLoom/alignment/README.md` - Complete API reference

### Our Commitment

1. The alignment framework will always be open source (MIT licensed)
2. Safety features will never be paywalled
3. We will publish our research findings

See [SAFETY.md](SAFETY.md) for our complete AI safety philosophy.

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