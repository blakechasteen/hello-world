# EdWIN Test Suite

Comprehensive test coverage for EdWIN Phase 2 and Phase 3 features.

## Test Organization

```
tests/
├── __init__.py               # Test suite initialization
├── test_multimodal.py        # Multimodal tutor tests (Phase 2)
├── test_analytics.py         # Learning analytics tests (Phase 2)
├── test_api.py               # FastAPI server tests (Phase 2)
└── test_teacher_dashboard.py # Teacher dashboard tests (Phase 3)
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest EduVerse/edwin/tests/ -v

# Run with coverage
pytest EduVerse/edwin/tests/ --cov=EduVerse.edwin --cov-report=html
```

### Specific Test Files

```bash
# Multimodal tests
pytest EduVerse/edwin/tests/test_multimodal.py -v

# Analytics tests
pytest EduVerse/edwin/tests/test_analytics.py -v

# API tests (unit tests only, skip integration)
pytest EduVerse/edwin/tests/test_api.py -v -m "not integration"

# API tests (integration tests - requires running server)
pytest EduVerse/edwin/tests/test_api.py -v -m "integration"

# Teacher dashboard tests
pytest EduVerse/edwin/tests/test_teacher_dashboard.py -v
```

### Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.asyncio` - Async tests (require `pytest-asyncio`)
- `@pytest.mark.integration` - Integration tests (require running servers)

Skip integration tests (fast unit tests only):
```bash
pytest EduVerse/edwin/tests/ -v -m "not integration"
```

## Test Coverage

### Phase 2 Tests

**Multimodal Tutor** (`test_multimodal.py`)
- ✓ Tutor initialization
- ✓ Visual Q&A functionality
- ✓ Video lesson ingestion
- ✓ Photo retrieval
- ✓ Graceful degradation without dependencies
- ✓ Resource type management

**Learning Analytics** (`test_analytics.py`)
- ✓ Learning velocity calculation
- ✓ Knowledge gap analysis
- ✓ Engagement metrics
- ✓ Intervention detection
- ✓ Progress reporting
- ✓ Interaction tracking
- ✓ Trend detection (accelerating/steady/decelerating)
- ✓ Risk level classification

**FastAPI Server** (`test_api.py`)
- ✓ Request/response model validation
- ✓ Input validation (grade levels, questions, counts)
- ✓ Student management endpoints
- ✓ Tutoring endpoints
- ✓ Progress tracking
- ✓ Health checks

### Phase 3 Tests

**Teacher Dashboard** (`test_teacher_dashboard.py`)
- ✓ Connection manager (WebSocket)
- ✓ Student management
- ✓ Activity logging
- ✓ Classroom snapshot generation
- ✓ Student overview creation
- ✓ Detailed student view
- ✓ Event dataclasses

## Dependencies

Required for testing:

```bash
pip install pytest pytest-asyncio pytest-cov
```

Optional (for full multimodal tests):

```bash
pip install Pillow torch transformers
```

## Integration Tests

Some tests require running servers:

### API Integration Tests

```bash
# Terminal 1: Start API server
PYTHONPATH=. uvicorn EduVerse.edwin.api:app --reload --port 8000

# Terminal 2: Run integration tests
pytest EduVerse/edwin/tests/test_api.py -v -m integration
```

### Dashboard Integration Tests

```bash
# Terminal 1: Start dashboard server
PYTHONPATH=. python -m EduVerse.edwin.teacher_dashboard

# Terminal 2: Run dashboard tests
pytest EduVerse/edwin/tests/test_teacher_dashboard.py -v
```

## Expected Results

**Unit Tests**: 35+ tests passing
**Integration Tests**: Requires running servers
**Coverage**: >80% for core modules

## Continuous Integration

Add to CI/CD pipeline:

```yaml
# .github/workflows/test.yml
- name: Run EdWIN tests
  run: |
    pip install pytest pytest-asyncio pytest-cov
    pytest EduVerse/edwin/tests/ -v -m "not integration" --cov=EduVerse.edwin
```

## Troubleshooting

### Import Errors

Make sure `PYTHONPATH` is set:

```bash
PYTHONPATH=. pytest EduVerse/edwin/tests/ -v
```

### Async Test Failures

Install `pytest-asyncio`:

```bash
pip install pytest-asyncio
```

### Multimodal Test Skips

Optional dependencies not installed. Tests will skip gracefully:

```bash
pip install Pillow torch transformers
```

## Writing New Tests

### Template

```python
import pytest
from EduVerse.edwin.your_module import YourClass

class TestYourFeature:
    """Tests for your feature"""

    @pytest.fixture
    def your_fixture(self):
        """Create test fixture"""
        return YourClass()

    def test_your_functionality(self, your_fixture):
        """Test your functionality"""
        result = your_fixture.do_something()
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_functionality(self, your_fixture):
        """Test async functionality"""
        result = await your_fixture.do_async_something()
        assert result is not None
```

### Best Practices

1. **One concept per test** - Test one thing at a time
2. **Arrange-Act-Assert** - Clear test structure
3. **Descriptive names** - `test_what_when_then` format
4. **Use fixtures** - Reusable test setup
5. **Test edge cases** - Not just happy paths
6. **Mock external deps** - Don't rely on external services
7. **Fast unit tests** - Integration tests separate

## Coverage Reports

Generate HTML coverage report:

```bash
pytest EduVerse/edwin/tests/ --cov=EduVerse.edwin --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Run tests locally
3. Ensure >80% coverage
4. Add integration tests if needed
5. Update this README

## Questions?

See main EdWIN documentation:
- [EdWIN Technical Specification](../../EDWIN_TECHNICAL_SPECIFICATION.md)
- [EdWIN Quick Start](../../EDWIN_QUICK_START.md)
- [Phase 3 Documentation](../../PHASE_3_DOCUMENTATION.md)
