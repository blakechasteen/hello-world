# HoloLoom Makefile Guide

A comprehensive Makefile for HoloLoom that centralizes all common development tasks. Run `make help` to see all available commands.

## Quick Start

```bash
# First time setup
make install-dev

# Test your changes
make test-fast

# Format and check code quality
make check

# Run full validation before commit
make validate
```

## Command Categories

### Testing (Fast Feedback Loop)

| Command | Purpose | Speed |
|---------|---------|-------|
| `make test` | Run all tests | ~30s |
| `make test-unit` | Unit tests only | <500ms |
| `make test-integration` | Integration tests only | <2s |
| `make test-e2e` | End-to-end tests only | <30s |
| `make test-fast` | Unit + integration (fastest full coverage) | ~2.5s |
| `make test-watch` | Watch mode - auto-rerun on file changes | Continuous |
| `make coverage` | Generate coverage report (text) | ~35s |
| `make coverage-html` | Generate HTML coverage report | ~35s |

**Usage tips:**
```bash
# While developing a feature
make test-watch

# Check coverage for a specific module
make coverage | grep HoloLoom/policy

# View HTML report in browser
make coverage-html  # Then open htmlcov/index.html
```

### Code Quality

| Command | Purpose | Impact |
|---------|---------|--------|
| `make lint` | Check code style with ruff | Informational |
| `make format` | Auto-format code with black | Modifies files |
| `make format-check` | Check if formatting needed | Dry-run |
| `make typecheck` | Static type checking with mypy | Informational |
| `make check` | All quality checks | Dry-run |

**Usage tips:**
```bash
# Before committing
make check

# Fix formatting issues
make format

# Fix linting issues (many are auto-fixable)
make lint --fix  # (or use ruff directly: ruff check --fix)

# Check type safety
make typecheck
```

### Development Environment

| Command | Purpose |
|---------|---------|
| `make install` | Install package in editable mode |
| `make install-dev` | Install with dev dependencies (testing, linting, etc.) |
| `make install-all` | Install with ALL dependencies (production + dev + visualization) |
| `make clean` | Remove build artifacts and caches |
| `make clean-all` | Remove everything including .venv |

**Typical workflow:**
```bash
# Fresh setup
make clean-all
make install-dev

# After deps change
pip install -e .

# Before major cleanup
make clean
```

### Servers & Services

| Command | Purpose | Port |
|---------|---------|------|
| `make server` | Start API server | 8000 |
| `make server-dev` | API server with auto-reload | 8000 |
| `make server-workflow` | Workflow executor | 8001 |
| `make mcp-memory` | Memory MCP server | - |
| `make mcp-search` | Search MCP server | - |

**Usage tips:**
```bash
# Development with auto-reload
make server-dev
# Now visit http://localhost:8000/docs for API docs

# Run multiple servers
make server-dev &  # Start API in background
make server-workflow    # Start workflow in foreground
```

### Validation & Benchmarking

| Command | Purpose | When |
|---------|---------|------|
| `make validate` | Complete validation pipeline | Before commits/PRs |
| `make experiments` | Run experiment suite | Benchmarking new features |
| `make benchmark` | Performance benchmarks | Optimizing code |

**Example:**
```bash
# Before submitting PR
make validate

# After optimizing a component
make benchmark
```

### Docker Management

| Command | Purpose |
|---------|---------|
| `make docker-up` | Start Neo4j + Qdrant containers |
| `make docker-down` | Stop containers |
| `make docker-logs` | View container logs |
| `make docker-clean` | Remove containers and volumes |

**Setup persistent storage:**
```bash
make docker-up
# Now Neo4j is available at http://localhost:7687
# And Qdrant is available at http://localhost:6333
```

### Documentation

| Command | Purpose |
|---------|---------|
| `make docs` | Build documentation |
| `make serve-docs` | Serve documentation locally |

### Utilities

| Command | Purpose |
|---------|---------|
| `make help` | Show all available commands |
| `make version` | Show version and environment info |

## Common Workflows

### Development Workflow

```bash
# Initial setup
make install-dev

# Continuous development
make test-watch &          # Run tests in background
# Edit files...tests automatically re-run

# Code review before committing
make check
make test-fast
```

### Feature Development with Servers

```bash
# Terminal 1: Run server with auto-reload
make server-dev

# Terminal 2: Run tests in watch mode
make test-watch

# Terminal 3: Edit code and see changes live
vim HoloLoom/some_module.py
```

### Before Pull Request

```bash
# Complete validation
make validate

# Generate coverage report
make coverage-html

# Verify formatting
make check

# Run full test suite
make test
```

### Production Deployment

```bash
# Verify everything works
make validate
make test
make experiments

# Clean build artifacts
make clean

# Install production dependencies
make install

# Build and deploy
# (deployment commands not in Makefile - deployment-specific)
```

### Performance Investigation

```bash
# Establish baseline
make benchmark > baseline.txt

# Make optimizations...

# Compare performance
make benchmark > optimized.txt
diff baseline.txt optimized.txt
```

## Customization

### Add New Test Marks

Edit `.pytest.ini` or `conftest.py` to add new pytest marks:

```python
pytest.mark.slow
pytest.mark.gpu_required
```

Then add to Makefile:

```makefile
test-slow:  ## Run slow tests
    $(PYTEST) $(TESTS_DIR) -v -m "slow"
```

### Change Test Directories

Edit the test path variables at the top of the Makefile:

```makefile
UNIT_TESTS := $(TESTS_DIR)/unit
INTEGRATION_TESTS := $(TESTS_DIR)/integration
E2E_TESTS := $(TESTS_DIR)/e2e
```

### Adjust Line Length for Formatting

```makefile
# Change from 100 to 120
$(BLACK) $(SRC_DIR) --line-length=120
```

## Troubleshooting

### Command not found errors

If you see errors like `black not found`, ensure you have dev dependencies:

```bash
make install-dev
```

### Tests fail with import errors

Ensure PYTHONPATH is set:

```bash
export PYTHONPATH=.
make test
```

Or use the server targets which set this automatically:

```bash
make server-dev  # PYTHONPATH set internally
```

### Docker container issues

If Neo4j/Qdrant don't start:

```bash
# Check logs
make docker-logs

# Clean and restart
make docker-clean
make docker-up
```

### Coverage report missing

Make sure pytest-cov is installed:

```bash
pip install pytest-cov
make coverage-html
```

## Performance Notes

**Test speeds** (approximate, on modern hardware):
- Unit tests: <500ms (isolated, fast)
- Integration tests: <2s (multiple components)
- E2E tests: <30s (full pipeline)
- Full suite: ~35s (all together)

**Optimization tips:**
- Use `make test-fast` during development (2.5s)
- Use `make test-watch` for continuous feedback
- Only run `make test-e2e` for critical changes
- Run full `make test` before submitting PR

## Additional Resources

- [HoloLoom README](README.md)
- [CLAUDE.md](CLAUDE.md) - Comprehensive documentation
- [Contributing Guide](CONTRIBUTING.md)
- [Developer Tools Report](DEVELOPER_TOOLS_REPORT.md)

## Getting Help

```bash
# Show all available commands
make help

# Show version and environment info
make version

# Run specific target with verbose output
make test-unit -d
```

## Tips & Tricks

### Parallel testing (requires pytest-xdist)

```bash
pip install pytest-xdist

# Then modify Makefile to add:
# $(PYTEST) $(TESTS_DIR) -v -n auto
```

### Coverage report with specific module

```bash
make coverage | grep -A 50 "Name.*Statements"
```

### Run tests matching a pattern

```bash
pytest HoloLoom/tests -k "test_weaving" -v
```

### Generate slower tests report

```bash
pytest HoloLoom/tests --durations=10
```

### View test dependencies

```bash
pytest --collect-only HoloLoom/tests
```
