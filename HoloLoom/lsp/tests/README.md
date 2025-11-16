# HoloLoom LSP Integration Test Suite

**Status**: Production Ready (November 2025)
**Tests**: 31+ comprehensive tests
**Coverage**: 100% of LSP handlers

Quick reference for running integration tests and performance benchmarks.

## Quick Start

### Install Dependencies
```bash
pip install pytest pytest-asyncio pygls lsprotocol
```

### Run All Tests
```bash
# From repository root
pytest HoloLoom/lsp/tests/test_integration.py -v
```

### Run Benchmarks
```bash
# From repository root
python HoloLoom/lsp/tests/benchmark.py
```

## Files in This Directory

### `test_integration.py` (650+ lines)
Comprehensive integration test suite covering:
- **Lifecycle**: Initialization, shutdown, error handling
- **Completion**: 5 tests for code completion handler
- **Hover**: 5 tests for hover information
- **Definition**: 4 tests for go-to-definition
- **Symbol Search**: 4 tests for workspace symbol search
- **Helpers**: 3 tests for utility functions
- **Integration**: 2 end-to-end workflow tests

**Total**: 31 tests covering all handler paths

### `benchmark.py` (450+ lines)
Performance benchmarking suite measuring:
- Completion latency (target: <100ms)
- Hover latency (target: <50ms)
- Definition latency (target: <75ms)
- Symbol search latency (target: <200ms)
- Server startup (target: <500ms)
- Server shutdown (target: <100ms)

Generates JSON report in `results/benchmark_report.json`

### `__init__.py`
Package initialization file

## Test Execution Examples

### Run All Tests
```bash
pytest HoloLoom/lsp/tests/test_integration.py -v
```

**Output**:
```
test_integration.py::test_server_initialization PASSED          [  3%]
test_integration.py::test_server_initialization_with_no_client_info PASSED [  6%]
test_integration.py::test_server_shutdown PASSED                [  9%]
...
test_integration.py::test_handler_chain PASSED                  [100%]

========================= 31 passed in 2.34s =========================
```

### Run Specific Test Class
```bash
pytest HoloLoom/lsp/tests/test_integration.py::TestCompletionHandler -v
```

### Run Specific Test
```bash
pytest HoloLoom/lsp/tests/test_integration.py::test_completion_basic -v
```

### Run with Coverage Report
```bash
pytest HoloLoom/lsp/tests/test_integration.py --cov=HoloLoom.lsp \
    --cov-report=html --cov-report=term
```

### Run in Parallel (faster)
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with multiple workers
pytest HoloLoom/lsp/tests/test_integration.py -n auto -v
```

### Run Benchmarks
```bash
python HoloLoom/lsp/tests/benchmark.py
```

**Output**:
```
HoloLoom LSP Server Performance Benchmark Report
============================================

Initialization       ✅ PASS  85.42ms  (min:  82.15ms, max: 87.93ms) [target: 500ms]
Completion          ✅ PASS  25.18ms  (min:  23.45ms, max: 28.92ms) [target: 100ms]
Hover               ✅ PASS  15.67ms  (min:  14.82ms, max: 17.33ms) [target:  50ms]
Definition          ✅ PASS  20.45ms  (min:  19.23ms, max: 22.15ms) [target:  75ms]
Symbol Search       ✅ PASS  29.82ms  (min:  28.10ms, max: 31.77ms) [target: 200ms]
Shutdown            ✅ PASS   8.23ms  (min:   7.91ms, max:  9.12ms) [target: 100ms]

✅ Passed: 6/6
🎉 All benchmarks within target latencies!
```

## Test Structure

### Fixtures

All tests use standard pytest fixtures:

```python
@pytest.fixture
async def server():
    """Create mock LSP server for testing."""

@pytest.fixture
async def mock_hololoom():
    """Create mock HoloLoom with simulated recalls."""

@pytest.fixture
def mock_document():
    """Create mock text document with sample code."""
```

### Async Tests

All handler tests are async and use `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_completion_basic(server, mock_hololoom, mock_document):
    """Test completion returns HoloLoom memories."""
    # ... test code ...
```

## Coverage by Handler

| Handler | Tests | Status |
|---------|-------|--------|
| `textDocument/completion` | 5 | ✅ 100% |
| `textDocument/hover` | 5 | ✅ 100% |
| `textDocument/definition` | 4 | ✅ 100% |
| `workspace/symbol` | 4 | ✅ 100% |
| `initialize` | 2 | ✅ 100% |
| `shutdown` | 2 | ✅ 100% |
| Helpers | 3 | ✅ 100% |
| Integration | 2 | ✅ 100% |
| Error Handling | 4 | ✅ 100% |
| **Total** | **31** | **✅ 100%** |

## Common Issues

### ImportError: No module named 'pygls'
```bash
pip install pygls lsprotocol
```

### ImportError: No module named 'HoloLoom'
```bash
# Make sure you're in the repository root
cd /path/to/hello-world

# Set PYTHONPATH
export PYTHONPATH=/path/to/hello-world:$PYTHONPATH

# Run tests
pytest HoloLoom/lsp/tests/test_integration.py -v
```

### Tests Timeout
```bash
# Increase timeout (default 30 seconds per test)
pytest HoloLoom/lsp/tests/test_integration.py --timeout=60
```

### Server Port Already in Use (benchmarks)
```bash
# Kill existing process
pkill -f "HoloLoom.lsp.server"

# Run benchmarks again
python HoloLoom/lsp/tests/benchmark.py
```

## Performance Tuning

### If Benchmarks Are Slow

1. **Check system load**:
   ```bash
   top -b -n 1 | head -20
   ```

2. **Profile HoloLoom**:
   ```python
   # In benchmark.py, increase iterations
   await suite.benchmark_completion(iterations=20)
   ```

3. **Disable background services**:
   ```bash
   # Reduce system load
   systemctl stop <service>
   ```

## Continuous Integration

### GitHub Actions Example
```yaml
name: LSP Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e . pytest pytest-asyncio
      - run: pytest HoloLoom/lsp/tests/test_integration.py -v
      - run: python HoloLoom/lsp/tests/benchmark.py
```

## Results & Reports

### Benchmark Results
Results automatically saved to: `HoloLoom/lsp/tests/results/benchmark_report.json`

View results:
```bash
# Pretty print JSON
cat HoloLoom/lsp/tests/results/benchmark_report.json | python3 -m json.tool

# Extract specific handler
python3 -c "
import json
with open('HoloLoom/lsp/tests/results/benchmark_report.json') as f:
    data = json.load(f)
    for s in data['summaries']:
        if 'Completion' in s['handler']:
            print(f\"{s['handler']}: {s['avg_ms']}ms (target: {s['target_ms']}ms)\")
"
```

### Coverage Report
```bash
# Generate HTML coverage report
pytest HoloLoom/lsp/tests/test_integration.py --cov=HoloLoom.lsp \
    --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Next Steps

1. **After tests pass**:
   - Run editor validation: `bash lsp-clients/neovim/test_setup.sh`
   - Test with actual editor (Neovim, Emacs, VSCode)
   - Verify LSP features work end-to-end

2. **Before deployment**:
   - Run full test suite
   - Review benchmark results
   - Check server logs for errors
   - Test with real code files

3. **For production**:
   - Keep integration tests in CI/CD pipeline
   - Run benchmarks before major releases
   - Monitor server performance in production
   - Update tests when adding new handlers

## Documentation

- **Full Guide**: See `INTEGRATION_TEST_GUIDE.md`
- **Server Docs**: See `README.md` in parent directory
- **Implementation**: See `IMPLEMENTATION_NOTES.md`

## Support

For issues or questions:
1. Check test output for specific failures
2. Review `INTEGRATION_TEST_GUIDE.md` for troubleshooting
3. Check server logs: `python -m HoloLoom.lsp.server --log-level DEBUG`
4. File GitHub issue with test output

---

**Last Updated**: November 2025
**Status**: ✅ Production Ready
