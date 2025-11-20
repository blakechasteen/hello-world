# HoloLoom Makefile - Usage Examples

This document provides practical examples of common Makefile usage patterns.

## Basic Operations

### First-Time Setup

```bash
# Install the package in development mode
make install-dev

# Verify installation
make version
```

### Cleaning Up

```bash
# Remove build artifacts and cache files
make clean

# Deep clean including Python virtual environment
make clean-all

# Remove Python caches across entire project
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

## Testing Workflows

### During Development

```bash
# Watch mode - auto-rerun tests whenever files change
make test-watch

# This is great for:
# - TDD (test-driven development)
# - Rapid iteration
# - Catching regressions immediately
```

### Quick Validation

```bash
# Run only fast tests (unit + integration)
make test-fast

# Expected time: ~2.5 seconds
# Good for: Quick validation between edits
```

### Full Test Suite

```bash
# Run ALL tests (unit, integration, e2e)
make test

# Expected time: ~35 seconds
# Good for: Before committing/pushing
```

### Specific Test Tiers

```bash
# Just unit tests (isolated components)
make test-unit

# Just integration tests (multiple components)
make test-integration

# Just end-to-end tests (full pipeline)
make test-e2e
```

### Coverage Reports

```bash
# Generate text coverage report
make coverage

# This shows:
# - Which files are tested
# - Coverage percentage per file
# - Lines not covered by tests

# Generate HTML report for visual browsing
make coverage-html

# This creates:
# - htmlcov/index.html (open in browser)
# - Interactive coverage visualization
# - Line-by-line coverage highlighting
```

## Code Quality

### Format Check (No Changes)

```bash
# Check if code needs formatting
make format-check

# This tells you WHAT needs fixing
# But doesn't modify files
```

### Auto-Format Code

```bash
# Auto-format all code to style guide
make format

# This:
# - Reformats files in-place
# - Follows black's opinionated style
# - Line length: 100 characters
```

### Linting

```bash
# Check for style and logic issues
make lint

# Reports:
# - Style violations (import order, etc)
# - Potential bugs
# - Code quality issues
```

### Type Checking

```bash
# Static type analysis with mypy
make typecheck

# Checks:
# - Type annotations correctness
# - Potential type errors
# - Missing type hints

# Requires type hints in code
```

### All Checks Combined

```bash
# Run full quality pipeline
make check

# This runs:
# 1. lint   (style checking)
# 2. format-check (formatting validation)
# 3. typecheck (type analysis)

# Good for: Pre-commit verification
```

## Server Management

### API Server

```bash
# Start server (no auto-reload)
make server

# Server starts at http://localhost:8000
# Access API docs at http://localhost:8000/docs
# Press Ctrl+C to stop

# For development with auto-reload
make server-dev

# Now changes to Python files auto-reload the server
# Great for development, API testing
```

### Multiple Servers

```bash
# Terminal 1: Start API server
make server-dev &

# Terminal 2: Start workflow executor
make server-workflow &

# Terminal 3: Start tests in watch mode
make test-watch

# All three running concurrently
# Kill with: pkill -f "make server" or Ctrl+C
```

## Validation & Deployment

### Pre-Commit Validation

```bash
# Run complete validation pipeline
make validate

# This runs:
# 1. All tests
# 2. Code linting
# 3. Format checking
# 4. Type checking

# Expected time: ~1-2 minutes
# Must pass before committing
```

### Performance Benchmarking

```bash
# Establish baseline
make benchmark > baseline.txt

# Make some optimizations...

# Compare performance
make benchmark > optimized.txt
diff baseline.txt optimized.txt

# This shows:
# - What got faster
# - What got slower
# - Performance regressions
```

### Experiments

```bash
# Run automated experiment suite
make experiments

# This:
# - Runs multiple configurations
# - Measures performance metrics
# - Generates comparison reports
# - Outputs to experiments/results/

# Great for:
# - Evaluating architectural changes
# - Benchmarking different approaches
# - Creating performance reports
```

## Docker & Services

### Setup Local Services

```bash
# Start Neo4j + Qdrant containers
make docker-up

# Now available:
# - Neo4j at http://localhost:7687
# - Qdrant at http://localhost:6333

# Work with services...
# make server, make tests, etc.

# Stop when done
make docker-down
```

### View Logs

```bash
# Watch container logs in real-time
make docker-logs

# Shows output from Neo4j, Qdrant, etc.
# Press Ctrl+C to stop
```

### Clean Docker State

```bash
# Remove containers and volumes
make docker-clean

# This:
# - Stops all containers
# - Removes them
# - Removes volumes (data is lost)

# Use when:
# - Starting fresh
# - Cleaning up after experiments
# - Resolving database corruption
```

## Documentation

### Build Docs

```bash
# Build documentation (if docs/ exists)
make docs

# Output in docs/_build/html/
```

### Serve Docs Locally

```bash
# Start local server for docs
make serve-docs

# Opens at http://localhost:8888
# Serve documentation as you edit it
```

## Real-World Scenarios

### Scenario 1: Rapid Development Iteration

```bash
# Terminal 1: Run tests in watch mode
make test-watch

# Terminal 2: Start server with reload
make server-dev

# Terminal 3: Edit code
vim HoloLoom/some_module.py

# Automatic flow:
# 1. Save file
# 2. Tests auto-rerun (green/red feedback)
# 3. Server auto-reloads
# 4. Browser refresh shows changes
```

### Scenario 2: Feature Implementation

```bash
# Start
make install-dev

# Develop with auto-testing
make test-watch &

# When ready to commit
make validate

# All tests pass? ✓
# Code formatted? ✓
# Types correct? ✓
# Then commit!
```

### Scenario 3: Performance Optimization

```bash
# Baseline before optimization
make benchmark > before.txt

# Implement optimization...

# Measure after
make benchmark > after.txt

# Compare
diff before.txt after.txt

# Successful if:
# - Performance improved
# - No regressions elsewhere
# - Tests still pass: make test
```

### Scenario 4: CI/CD Integration

```bash
# In GitHub Actions / GitLab CI:
make install-dev
make validate          # All checks
make test              # All tests
make benchmark         # Performance baseline

# If all pass, proceed with deployment
```

### Scenario 5: Production Preparation

```bash
# Complete validation
make validate

# Full test suite (all tiers)
make test

# Performance baseline
make benchmark > production_baseline.txt

# Experiments
make experiments

# Clean build
make clean
make install

# Ready to deploy!
```

## Tips & Tricks

### Speed up test runs

```bash
# Just unit tests (fastest)
make test-unit

# Watch mode for continuous feedback
make test-watch

# Run tests in parallel (requires pytest-xdist)
pytest HoloLoom/tests -n auto
```

### Debug a failing test

```bash
# Run with verbose output
pytest HoloLoom/tests/test_specific.py -vv --tb=long

# Run single test
pytest HoloLoom/tests/test_file.py::TestClass::test_method -vv

# Show print statements
pytest HoloLoom/tests -vv -s
```

### Format specific files

```bash
# Format specific module
black HoloLoom/policy/unified.py

# Format specific directory
black HoloLoom/tests/
```

### Lint specific files

```bash
# Check specific module
ruff check HoloLoom/policy/unified.py

# Auto-fix linting issues
ruff check --fix HoloLoom/
```

### Type check specific module

```bash
# Type check single module
mypy HoloLoom/policy/unified.py

# Type check with strict mode
mypy --strict HoloLoom/policy/
```

## Troubleshooting

### "Command not found" errors

```bash
# Ensure dev dependencies are installed
make install-dev

# Verify tools are available
which pytest
which black
which ruff
which mypy
```

### Docker connection errors

```bash
# Check Docker is running
docker ps

# Restart Docker services
make docker-down
make docker-up
```

### Tests failing with import errors

```bash
# Set PYTHONPATH
export PYTHONPATH=.

# Or use one of the server targets which sets it:
make server-dev  # PYTHONPATH automatically set
```

### Coverage report missing

```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Generate report
make coverage-html
```

## Common Command Sequences

### Commit Workflow

```bash
# Make changes...

# Validate before commit
make check          # Quality checks
make test-fast      # Quick test
make validate       # Full validation

# If all pass, commit
git add .
git commit -m "Feature: ..."
```

### Pull Request Workflow

```bash
# Development with auto-testing
make test-watch &

# Make changes...

# Pre-PR validation
make validate
make coverage-html

# Push changes
git push

# Create PR
# (CI system runs: make validate && make test && make experiments)
```

### Release Workflow

```bash
# Verify everything works
make validate       # All checks
make test           # All tests
make experiments    # Performance baseline
make benchmark      # Baseline metrics

# Clean for release
make clean
make install

# Version bump and release
```

## Performance Baselines

**Expected execution times** (modern hardware):

| Command | Time | Notes |
|---------|------|-------|
| `make test-unit` | <500ms | Fastest, isolated tests |
| `make test-fast` | ~2.5s | Unit + integration (recommended) |
| `make test-integration` | <2s | Multi-component tests |
| `make test-e2e` | <30s | Full pipeline tests |
| `make test` | ~35s | All tests combined |
| `make check` | ~2s | Quality checks |
| `make validate` | ~1-2min | Full validation pipeline |
| `make coverage` | ~35s | With coverage analysis |
| `make benchmark` | ~5-10s | Performance testing |

---

For more information, see:
- [MAKEFILE_GUIDE.md](MAKEFILE_GUIDE.md) - Complete reference guide
- [MAKEFILE_SUMMARY.md](MAKEFILE_SUMMARY.md) - Architecture overview
