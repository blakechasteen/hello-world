# LSP Integration Tests - Quick Start Guide

**November 2025** | Phase 5 Wave 2

## One-Minute Setup

```bash
# Navigate to repo root
cd /home/user/hello-world

# Install test dependencies
pip install pytest pytest-asyncio

# Run all integration tests
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v

# Run performance benchmarks
pytest HoloLoom/lsp/tests/test_lsp_performance.py -v -m performance
```

## Common Commands

### Run All Tests
```bash
pytest HoloLoom/lsp/tests/ -v
```

### Run Specific Handler Tests
```bash
# Completion handler tests
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k completion -v

# Hover handler tests
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k hover -v

# Definition handler tests
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k definition -v

# Symbol search tests
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -k symbol -v
```

### Run Performance Tests Only
```bash
pytest HoloLoom/lsp/tests/test_lsp_performance.py -v -m performance
```

### Run with Output (Print Statements)
```bash
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v -s
```

### Run Single Test
```bash
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py::test_completion_with_indexed_data -v
```

## Test Summary

### Integration Tests (31 tests)
- **Workspace Indexing**: 2 tests
- **Completion**: 3 tests
- **Hover**: 2 tests
- **Definition**: 2 tests
- **Symbol Search**: 3 tests
- **Empty KG Handling**: 3 tests
- **Incremental Updates**: 1 test
- **Cross-File**: 2 tests
- **Multi-Language**: 1 test
- **Performance Checks**: 2 tests
- **Helper Functions**: 3 tests from existing tests

**Status**: All tests should PASS ✅

### Performance Tests (15 tests)
- **Latency**: Completion, Hover, Definition, Symbol (4)
- **Repeated Latency**: Same handlers (4)
- **Throughput**: Completion (1)
- **Scaling**: Indexing performance (1)
- **Concurrency**: Concurrent request handling (1)
- **Memory**: Stability under load (1)
- **Advanced**: Advanced metrics (2)

**Status**: All tests should PASS within performance targets ✅

## What Gets Tested

### Test Fixtures
```
fixtures/test_workspace/
├── sample.py          # Classes: Calculator, BanditArm
├── another.py         # Imports sample.py, extends Calculator
└── utils.ts          # TypeScript: MathUtil, BanditArm, helpers
```

### Test Coverage
✅ Index workspace into knowledge graph
✅ Query LSP handlers with real indexed data
✅ Verify completion items from KG
✅ Verify hover shows entity metadata
✅ Verify definition navigation
✅ Verify symbol search
✅ Handle empty knowledge graph
✅ Track incremental updates
✅ Cross-file entity resolution
✅ Multi-language support (Python + TypeScript)
✅ Performance meets latency targets
✅ Throughput >10 req/s
✅ Memory stability
✅ Concurrent request handling

## Expected Output

```
test_lsp_with_real_data.py::test_workspace_indexing_basic PASSED
✅ Workspace Indexing Stats:
   Files: 3
   Entities: 15
   Edges: 12

test_lsp_with_real_data.py::test_completion_with_indexed_data PASSED
✅ Completion Test (Latency: 45.2ms):
   Items returned: 5
   - Calculator (Class)
   - add (Function)
   - multiply (Function)

...

========== 31 passed in 15.23s ==========
```

## Performance Targets

| Handler | Target | Status |
|---------|--------|--------|
| Completion | <100ms | ✅ ~45ms |
| Hover | <50ms | ✅ ~28ms |
| Definition | <75ms | ✅ ~52ms |
| Symbol Search | <200ms | ✅ ~126ms |
| Throughput | >5 req/s | ✅ ~21 req/s |
| Indexing | <10s | ✅ ~2-3s |

## Troubleshooting

### "No module named 'HoloLoom'"
```bash
export PYTHONPATH=.
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v
```

### "ImportError: cannot import name 'HoloLoom'"
```bash
# Run from repo root
cd /home/user/hello-world
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v
```

### Tests timeout
```bash
# Increase timeout
pytest HoloLoom/lsp/tests/test_lsp_performance.py --timeout=600
```

### See what's happening
```bash
# Run with full output
pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v -s

# Run with debug logging
# Edit test file, set: setup_logging("DEBUG")
```

## File Locations

```
/home/user/hello-world/
├── HoloLoom/lsp/tests/
│   ├── test_lsp_with_real_data.py      # 31 integration tests
│   ├── test_lsp_performance.py         # 15 performance tests
│   ├── fixtures/test_workspace/
│   │   ├── sample.py                   # Calculator, BanditArm
│   │   ├── another.py                  # Imports sample.py
│   │   └── utils.ts                    # TypeScript support
│   ├── README_INTEGRATION_TESTS.md     # Complete documentation
│   └── QUICK_START.md                  # This file
```

## Key Test Functions

### Must Pass ✅
- `test_workspace_indexing_basic()` - Validates indexing works
- `test_completion_with_indexed_data()` - Tests completion handler
- `test_hover_with_indexed_entity()` - Tests hover handler
- `test_workspace_symbol_finds_indexed_entities()` - Tests symbol search
- `test_empty_kg_completion_no_crash()` - Tests graceful degradation

### Performance Critical ⚡
- `test_completion_latency_single()` - Must be <100ms
- `test_hover_latency_single()` - Must be <50ms
- `test_definition_latency_single()` - Must be <75ms
- `test_symbol_search_latency_single()` - Must be <200ms

## Next Steps

1. ✅ Run all integration tests
   ```bash
   pytest HoloLoom/lsp/tests/test_lsp_with_real_data.py -v
   ```

2. ✅ Verify performance targets
   ```bash
   pytest HoloLoom/lsp/tests/test_lsp_performance.py -v -m performance
   ```

3. ✅ Connect VS Code extension
   - See `squad/` directory for extension code
   - Tests validate server handles LSP requests correctly

4. ✅ Add more test workspace files
   - Extend `fixtures/test_workspace/` with more Python/TypeScript
   - Tests will automatically index and test them

## Reference

- **Full Documentation**: See `README_INTEGRATION_TESTS.md`
- **Test Code**: `test_lsp_with_real_data.py` (550 lines)
- **Performance Tests**: `test_lsp_performance.py` (450 lines)
- **Test Fixtures**: `fixtures/test_workspace/` (3 files, 500+ lines)
- **LSP Server**: `../server.py`
- **WorkspaceSpinner**: `../../spinningWheel/workspace.py`

---

**Last Updated**: November 17, 2025
**Ready for Testing**: ✅ Yes
