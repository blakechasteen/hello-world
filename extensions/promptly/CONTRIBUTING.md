# Contributing to Promptly

Thanks for your interest in contributing to Promptly! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Install for Development

```bash
# Clone the repository
git clone https://github.com/promptly-cli/promptly.git
cd promptly

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with all extras
pip install -e ".[dev,all]"
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=promptly --cov-report=term-missing

# Specific test file
pytest tests/test_core.py -v
```

### Code Style

We use:
- **Black** for formatting (line length 100)
- **isort** for import sorting
- **mypy** for type checking

```bash
# Format
black promptly/ tests/

# Check types
mypy promptly/

# Sort imports
isort promptly/ tests/
```

## Making Changes

### Branch Naming

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `test/description` - Test additions

### Commit Messages

Use conventional commits:

```
feat: add prompt tagging support
fix: correct branch checkout when prompt doesn't exist
docs: update CLI command reference
test: add chain execution tests
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests
5. Run the full test suite
6. Submit a PR with a clear description

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black --check promptly/`)
- [ ] Types check (`mypy promptly/`)
- [ ] New features have tests
- [ ] Documentation updated if needed

## Project Structure

```
promptly/
├── core/           # Core API and database
│   ├── promptly.py # Main Promptly class
│   ├── database.py # SQLite storage layer
│   └── analytics.py# Analytics engine
├── cli/            # Click-based CLI
│   ├── main.py     # Primary commands
│   ├── analytics.py# Analytics commands
│   └── mrf.py      # MRF commands
├── judge/          # LLM evaluation
│   └── llm_judge.py
├── analytics/      # Analytics and dashboards
├── mcp/            # MCP server for Claude Desktop
├── integrations/   # Optional integrations
├── demos/          # Demo scripts
├── vscode/         # VS Code extension (TypeScript)
└── docs/           # Documentation
```

## Key Design Principles

1. **Local-first** - Data stays on the user's machine
2. **Graceful degradation** - Optional features fail silently with helpful messages
3. **Minimal dependencies** - Core needs only `click` and `pyyaml`
4. **Git-like UX** - Familiar mental model for developers

## Reporting Issues

When reporting bugs, please include:

1. Python version (`python --version`)
2. OS and version
3. Steps to reproduce
4. Expected vs actual behavior
5. Error messages (full traceback)

## Questions?

Open a [GitHub Discussion](https://github.com/promptly-cli/promptly/discussions) for questions or ideas.
