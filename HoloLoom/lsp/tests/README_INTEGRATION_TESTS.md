# HoloLoom LSP Integration Tests with Real Data

**Status**: Production Ready (November 2025)
**Location**: `HoloLoom/lsp/tests/`
**Version**: 0.1.0

## Overview

Comprehensive integration test suite for the HoloLoom Language Server Protocol (LSP) server with real knowledge graph data. Tests validate end-to-end workflows from workspace indexing through LSP queries.

### What Gets Tested

```
Workspace Files          Knowledge Graph          LSP Handlers
     ↓                         ↓                       ↓
 sample.py      ┐           Entities          → Completion
 another.py     ├→ Index →   Classes           → Hover
 utils.ts       ┘           Methods            → Definition
                            Relationships  →  → Symbol Search
```

## Test Structure

### 1. Test Fixtures (`test_workspace/`)

**Sample Python File** (`sample.py`):
- Calculator class with multiple methods
- BanditArm class for Thompson Sampling
- Type hints and docstrings
- Comments (NOTE, TODO, FIXME)
- Helper functions

**Another Module** (`another.py`):
- Cross-file imports from sample.py
- Class inheritance (AdvancedCalculator)
- Bandit experiment integration
- Tests module relationships

**TypeScript File** (`utils.ts`):
- Multi-language support validation
- TypeScript interfaces and classes
- Function definitions with types
- Comments in TypeScript format

### 2. Integration Tests (`test_lsp_with_real_data.py`)

**Test Categories**:

#### A. Workspace Indexing + LSP Completion
```python
test_workspace_indexing_basic()
  ✓ Workspace scanner processes files
  ✓ MemoryShards created
  ✓ Stats collected (files, entities, edges)

test_completion_with_indexed_data()
  ✓ HoloLoom.recall returns indexed entities
  ✓ Completion items have proper labels/kinds
  ✓ Latency <100ms
```

#### B. Hover with Knowledge Graph Data
```python
test_hover_with_indexed_entity()
  ✓ Hover retrieves entity from KG
  ✓ Shows docstring/metadata
  ✓ Latency <50ms

test_hover_shows_docstring()
  ✓ Displays function/class docstrings
  ✓ Includes metadata
```

#### C. Go-to-Definition via Knowledge Graph
```python
test_definition_navigates_to_entity()
  ✓ Definition returns file location
  ✓ Returns correct line number
  ✓ Latency <75ms

test_definition_for_unknown_entity()
  ✓ Returns None gracefully
  ✓ No crashes on missing entities
```

#### D. Symbol Search Across Workspace
```python
test_workspace_symbol_finds_indexed_entities()
  ✓ Search finds indexed classes/functions
  ✓ Returns SymbolInformation objects
  ✓ Semantic search works

test_workspace_symbol_finds_methods()
  ✓ Finds methods like 'add', 'multiply'
  ✓ Proper symbol kind detection
```

#### E. Empty Knowledge Graph Handling
```python
test_empty_kg_completion_no_crash()
  ✓ Returns empty list gracefully
  ✓ No exceptions thrown

test_empty_kg_hover_graceful()
  ✓ Returns None or empty
  ✓ Proper degradation

test_empty_kg_symbol_search()
  ✓ Returns empty list
  ✓ Stable behavior
```

#### F. Incremental Updates
```python
test_incremental_indexing()
  ✓ Initial indexing creates KG entries
  ✓ Re-indexing finds changes
  ✓ LSP queries reflect updates
  ✓ Cache invalidation works
```

#### G. Cross-File Relationships
```python
test_cross_file_imports()
  ✓ Another.py imports from sample.py
  ✓ Search finds imported entities
  ✓ KG tracks relationships

test_imported_class_completion()
  ✓ Completion works with imported classes
  ✓ Cross-file context respected
```

#### H. Multi-Language Support
```python
test_typescript_file_indexing()
  ✓ TypeScript files indexed
  ✓ TS entities searchable
  ✓ Multi-language support validated
```

### 3. Performance Tests (`test_lsp_performance.py`)

**Latency Benchmarks**:
```
test_completion_latency_single()
  Target: <100ms
  Measures: Single request warmup latency

test_hover_latency_single()
  Target: <50ms
  Measures: Hover request latency

test_definition_latency_single()
  Target: <75ms
  Measures: Definition lookup latency

test_symbol_search_latency_single()
  Target: <200ms
  Measures: Workspace symbol search latency
```

**Repeated Request Performance**:
```
test_completion_latency_repeated()
  ✓ Mean <100ms
  ✓ P95 <150ms
  ✓ Stable over 20 iterations

test_hover_latency_repeated()
  ✓ Mean <75ms
  ✓ Consistent performance
```

**Throughput Tests**:
```
test_completion_throughput()
  Target: >10 req/s
  Measures: Sustained throughput

test_concurrent_requests_performance()
  ✓ Handles 10 concurrent requests
  ✓ Measures aggregate throughput
```

**Scaling Tests**:
```
test_workspace_indexing_performance()
  Target: <10s (test workspace)
  Measures: Index time vs file count

test_memory_stability()
  ✓ Memory doesn't grow unboundedly
  ✓ Proper cleanup validated
```

## Running the Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-benchmark psutil

# Install HoloLoom and LSP dependencies
pip install pygls lsprotocol
```

### Run All Integration Tests

```bash
# From repository root
cd /home/user/hello-world

# Run all LSP tests
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v

# With output capture disabled (see print statements)
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v -s

# Run specific test
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py::test_workspace_indexing_basic -v
```

### Run Performance Tests

```bash
# Run only performance tests
pytest HoloLoom/lsp/tests/test_lsp_performance.py -v -m performance

# With profiling
pytest HoloLoom/lsp/tests/test_lsp_performance.py -v -s --durations=10
```

### Run Specific Test Category

```bash
# Completion tests only
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k completion -v

# Hover tests only
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k hover -v

# Definition tests only
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k definition -v

# Symbol search tests only
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k symbol -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest HoloLoom/lsp/tests/ --cov=HoloLoom.lsp --cov-report=html

# View report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

## Expected Output

### Successful Integration Test Run

```
HoloLoom/lsp/tests/test_lsp_with_real_data.py

test_workspace_indexing_basic PASSED
✅ Workspace Indexing Stats:
   Files: 3
   Entities: 15
   Edges: 12

test_completion_with_indexed_data PASSED
✅ Completion Test (Latency: 45.2ms):
   Items returned: 5
   - Calculator (Class)
   - add (Function)
   - multiply (Function)

test_hover_with_indexed_entity PASSED
✅ Hover Test (Latency: 28.1ms):
   Content available: Yes
   Content preview: A simple calculator...

...

========== 25 passed in 15.23s ==========
```

### Performance Test Output

```
test_lsp_performance.py

test_completion_latency_single PASSED
  Completion:
    Count:  2
    Min:    42.51ms
    Max:    48.73ms
    Mean:   45.62ms
    Median: 45.62ms
    StDev:  3.11ms
    P95:    48.73ms
    P99:    48.73ms

test_completion_throughput PASSED
  Completion Throughput:
    Requests: 50
    Time: 2.35s
    Throughput: 21.3 req/s

...

========== Performance Test Summary ==========
Performance Targets:
  Completion:     <100ms  ✓ PASS (45.62ms)
  Hover:          <50ms   ✓ PASS (28.15ms)
  Definition:     <75ms   ✓ PASS (52.34ms)
  Symbol Search:  <200ms  ✓ PASS (125.78ms)
  Throughput:     >5 req/s ✓ PASS (21.3 req/s)
```

## Test Data Summary

### File Count
- **Python**: 2 files (sample.py, another.py)
- **TypeScript**: 1 file (utils.ts)
- **Total**: 3 files

### Entity Count
- **Classes**: 4
  - Calculator (sample.py)
  - BanditArm (sample.py)
  - AdvancedCalculator (another.py)
  - BanditExperiment (another.py)
  - MathUtil (utils.ts)
  - BanditArm (utils.ts)
- **Methods**: 8+
- **Functions**: 5+
- **Imports**: 10+

### Relationships
- **Imports**: another.py → sample.py
- **Inheritance**: AdvancedCalculator → Calculator
- **Usage**: BanditExperiment uses BanditArm

## Validation Checklist

Each test validates:

- [ ] **Indexing Works**
  - Workspace scanner processes all files
  - AST/regex parsing extracts entities correctly
  - MemoryShards created with proper metadata

- [ ] **Knowledge Graph Populated**
  - Entities stored in KG
  - Relationships tracked
  - Metadata available

- [ ] **LSP Handlers Query KG**
  - Completion queries HoloLoom.recall()
  - Hover retrieves entity info
  - Definition extracts file/line metadata
  - Symbol search finds matches

- [ ] **Performance Meets Targets**
  - Completion: <100ms
  - Hover: <50ms
  - Definition: <75ms
  - Symbol search: <200ms

- [ ] **Graceful Degradation**
  - Empty KG returns empty results (no crash)
  - Invalid queries handled safely
  - Missing files handled gracefully

- [ ] **Cross-File Resolution**
  - Imports detected
  - Referenced entities found
  - Relationships tracked

- [ ] **Incremental Updates**
  - Changed files detected
  - KG updated incrementally
  - Cache invalidated properly

## Troubleshooting

### Import Errors

```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=.
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py

# Or run from repo root
cd /home/user/hello-world
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py
```

### HoloLoom Not Available

```
ImportError: No module named 'HoloLoom'
```

**Solution**:
```bash
pip install -e .  # Install package in development mode
# or
export PYTHONPATH=/home/user/hello-world:$PYTHONPATH
```

### Async Event Loop Issues

```
RuntimeError: Event loop is closed
```

**Solution**: Tests already include event loop fixture. If running manually:
```python
import asyncio

async def test():
    # Test code
    pass

asyncio.run(test())
```

### Slow Performance Tests

If tests timeout:
```bash
# Increase timeout
pytest HoloLoom/lsp/tests/test_lsp_performance.py --timeout=600
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: LSP Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio pytest-benchmark psutil

      - name: Run integration tests
        run: |
          pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v

      - name: Run performance tests
        run: |
          pytest HoloLoom/lsp/tests/test_lsp_performance.py -v -m performance
```

## Next Steps

### Phase 5 Wave 2 Roadmap

1. **Real Client Testing** (Week 2)
   - Connect VS Code extension client
   - Test with actual editor interactions
   - Validate feature behavior

2. **Extended Workspace** (Week 3)
   - Test with larger codebases (1000+ files)
   - Benchmark scaling characteristics
   - Optimize hot paths

3. **Continuous Indexing** (Week 4)
   - Background file watcher integration
   - Incremental indexing on file changes
   - Real-time symbol updates

4. **Advanced Features** (Week 5+)
   - Semantic refactoring support
   - Find all references
   - Rename symbol across workspace
   - Code lens integration

## References

- **LSP Specification**: https://microsoft.github.io/language-server-protocol/
- **pygls Documentation**: https://github.com/openlang/pygls
- **HoloLoom Documentation**: See `/home/user/hello-world/CLAUDE.md`
- **Test Fixtures**: `/home/user/hello-world/HoloLoom/lsp/tests/fixtures/test_workspace/`

## Support

For issues or questions:

1. Check test output for detailed error messages
2. Enable debug logging: `setup_logging("DEBUG")`
3. Review test fixtures for expected structure
4. Check HoloLoom integration in `test_lsp_with_real_data.py`

---

**Last Updated**: November 17, 2025
**Maintainer**: Claude Code (Phase 5 Wave 2)
**Status**: Production Ready
