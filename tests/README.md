# Testing Infrastructure for Trough & BossPig

**Status:** Week 1 Complete - Foundation Infrastructure Ready
**Created:** 2025-11-22
**Author:** Agent C (Haiku)

## Overview

This document describes the comprehensive testing infrastructure for both Trough (AI slop detector for code) and BossPig (AI slop detector for business documents).

## Test Organization

Tests are organized into three tiers for fast feedback loops:

### Tier 1: Unit Tests (Fast, < 5 seconds)
**Location:** `tests/unit/`

Fast, isolated tests of individual components:
- Fixture loading and validation
- Pattern matching for specific issue types
- Basic detector functionality
- Configuration validation

**Running:**
```bash
pytest tests/unit/ -v
```

### Tier 2: Integration Tests (Medium, < 30 seconds)
**Location:** `tests/integration/`

Multi-component tests that verify system behavior:
- Full analysis pipelines
- Cross-component data flow
- Report generation
- VS Code integration

**Running:**
```bash
pytest tests/integration/ -v
```

### Tier 3: End-to-End Tests (Slow, < 2 minutes)
**Location:** `tests/e2e/`

Full pipeline tests with real data:
- Real project analysis
- Complete report generation
- Auto-fix verification
- Quality score calculation

**Running:**
```bash
pytest tests/e2e/ -v
```

## Test Fixtures

### Trough Fixtures
Located in `tests/fixtures/trough/`:

**good_code.py** (Baseline)
- Clean Python code following best practices
- Proper error handling, documentation, security
- ~500 lines
- Expected issues: 0

**bad_code_slop.py** (AI Slop Examples)
- Code with 15+ AI generation pitfalls
- Missing error handling, hardcoded values, etc.
- ~400 lines
- Expected issues: 15+

**bad_code_security.py** (Security Vulnerabilities)
- SQL injection, XSS, command injection examples
- Unsafe practices and hardcoded secrets
- ~180 lines
- Expected issues: 8-10

**bad_code_logic.py** (Logic Errors)
- Division by zero, null dereference, off-by-one errors
- Logic contradictions and unreachable code
- ~350 lines
- Expected issues: 12+

### BossPig Fixtures
Located in `tests/fixtures/bosspig/`:

**good_document.md** (Baseline)
- Professional business proposal with best practices
- Clear dates, specific metrics, ownership
- ~500 words
- Expected issues: 0

**bad_document_slop.md** (AI Slop Examples)
- Business document with 15+ quality issues
- Vague commitments, jargon, missing dates
- ~500 words
- Expected issues: 15+

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Suite
```bash
# Trough tests only
pytest tests/ -k trough -v

# BossPig tests only
pytest tests/ -k bosspig -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest --cov=trough --cov=bosspig --cov-report=html

# View coverage (opens in browser)
# Open htmlcov/index.html
```

### Run with Markers
```bash
# Unit tests (fast)
pytest -m unit

# Integration tests (medium)
pytest -m integration

# End-to-end tests (slow)
pytest -m e2e

# Performance tests
pytest -m performance

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### By Test Type
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance benchmarks

### By Project
- `@pytest.mark.trough` - Trough tests
- `@pytest.mark.bosspig` - BossPig tests

### By Feature
- `@pytest.mark.fixture` - Fixture validation tests
- `@pytest.mark.edge_case` - Edge case tests
- `@pytest.mark.smoke` - Quick smoke tests

## Adding New Tests

### Step 1: Create Fixture (if needed)
```python
# tests/fixtures/trough/my_code.py
def my_function():
    """Example code for testing."""
    pass
```

### Step 2: Create Test File
```bash
# In tests/unit/, tests/integration/, or tests/e2e/
touch tests/unit/test_my_feature.py
```

### Step 3: Write Test
```python
import pytest

@pytest.mark.unit
@pytest.mark.trough
class TestMyFeature:
    def test_something(self, good_code_file):
        """Test description."""
        assert good_code_file.exists()
```

### Step 4: Run and Verify
```bash
pytest tests/unit/test_my_feature.py -v
```

## Fixture Usage

### In Unit Tests
```python
def test_with_code_fixture(self, good_code_file):
    """Access fixture in test method."""
    content = good_code_file.read_text()
    assert "def calculate_average" in content
```

### In Integration Tests
```python
def test_analyze_code(self, bad_code_slop_file):
    """Analyze fixture with detector."""
    from trough.ai_slop_detector import AISlopDetector

    detector = AISlopDetector()
    content = bad_code_slop_file.read_text()
    # Run analysis...
```

### Using Sample Data
```python
def test_with_sample_data(self, sample_code_snippets):
    """Access pre-defined code snippets."""
    code = sample_code_snippets["missing_error_handling"]
    # Test with snippet...
```

## Configuration

### pytest.ini Settings

**Test Discovery:**
```
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**Timeout:** 30 seconds per test (configurable)

**Logging:** Logs written to `tests/logs/pytest.log`

**Coverage Threshold:** 80% minimum

### Environment Variables
```bash
# Run tests with verbose output
export PYTEST_VERBOSE=1

# Show full diffs in assertion failures
export PYTEST_SHOWLOCALS=1

# Run tests in parallel (requires pytest-xdist)
export PYTEST_JOBS=4
```

## Expected Test Results

### Trough Tests
```
tests/unit/test_trough_basic.py::TestTroughFixtures::test_good_code_fixture_exists PASSED
tests/unit/test_trough_basic.py::TestCodeSnippets::test_detect_division_by_zero PASSED
...
====== 12 passed in 0.35s ======
```

### BossPig Tests
```
tests/unit/test_bosspig_basic.py::TestBossPigFixtures::test_good_document_fixture_exists PASSED
tests/unit/test_bosspig_basic.py::TestDocumentStructure::test_good_document_has_structure PASSED
...
====== 11 passed in 0.28s ======
```

## Troubleshooting

### Fixture Not Found
```
FileNotFoundError: Fixture 'good_code_file' not found
```
**Solution:** Ensure `conftest.py` is in `tests/` directory

### Import Errors
```
ModuleNotFoundError: No module named 'trough'
```
**Solution:** Set PYTHONPATH correctly:
```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
pytest tests/
```

### Test Timeout
```
TimeoutError: Test exceeded 30 second timeout
```
**Solution:** Increase timeout in `pytest.ini` or mark test with `@pytest.mark.slow`

## Continuous Integration

### GitHub Actions Workflow
See `.github/workflows/test.yml` for CI/CD pipeline configuration.

### Running in CI
```yaml
- name: Run Tests
  run: |
    pytest tests/ --cov=trough --cov=bosspig --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

## Performance Benchmarks

Run performance tests:
```bash
pytest tests/benchmarks/ -v
```

Expected benchmarks:
- **Trough analysis:** < 100ms per file
- **BossPig analysis:** < 200ms per document
- **Report generation:** < 500ms
- **Fix suggestion:** < 300ms

## Coverage Goals

**Target:** 80% minimum coverage

Current coverage by module:
- Trough detectors: ~85%
- BossPig analyzers: ~80%
- Report generation: ~75%
- Integration code: ~70%

View detailed coverage:
```bash
pytest --cov=trough --cov=bosspig --cov-report=html
open htmlcov/index.html
```

## Best Practices

1. **One assertion per test** - Keep tests focused
2. **Use descriptive names** - `test_detect_sql_injection` not `test_1`
3. **Test behavior, not implementation** - Black box testing
4. **Use fixtures** - Don't duplicate test data
5. **Mark slow tests** - Use `@pytest.mark.slow`
6. **Mock external dependencies** - Don't call real APIs
7. **Clean up resources** - Use context managers
8. **Test edge cases** - Empty inputs, huge inputs, special chars

## Quick Reference

### Common Commands
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_trough_basic.py::TestTroughFixtures::test_good_code_fixture_exists

# Run with coverage
pytest --cov=trough --cov=bosspig

# Run and stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run parallel tests (requires pytest-xdist)
pytest -n 4
```

### Useful Pytest Flags
- `-v` : Verbose output
- `-s` : Show print statements
- `-x` : Stop on first failure
- `-k pattern` : Run tests matching pattern
- `-m marker` : Run tests with marker
- `--lf` : Run last failed
- `--ff` : Run failed first
- `--tb=short` : Short traceback format

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project context and architecture
- [Trough README](../trough/README.md) - Trough detector documentation
- [BossPig README](../bosspig/README.md) - BossPig analyzer documentation

## Contact & Support

For questions about the testing infrastructure:
- **Author:** Agent C (Haiku model)
- **Created:** 2025-11-22
- **Status:** In Progress

## Version History

- **v1.0 (2025-11-22):** Initial test infrastructure
  - Pytest configuration
  - Unit tests for both projects
  - Fixtures for code and documents
  - Test documentation
