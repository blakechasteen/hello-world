# HoloLoom Test Suite

Comprehensive testing infrastructure for HoloLoom Phase 2 components.

## Test Structure

```
tests/
├── integration/          # End-to-end integration tests
│   └── test_phase2_integration.py
├── performance/          # Performance benchmarks
│   └── test_phase2_performance.py
├── test_orchestrator.py  # Orchestrator unit tests
├── test_time_bucket.py   # Time bucketing tests
└── run_all_tests.sh      # Convenience runner
```

## Quick Start

### Run All Tests
```bash
./tests/run_all_tests.sh
```

### Run Integration Tests Only
```bash
pytest HoloLoom/tests/integration/ -v
```

### Run Performance Tests Only
```bash
pytest HoloLoom/tests/performance/ -v -s
```

### Run Specific Test File
```bash
pytest HoloLoom/tests/integration/test_phase2_integration.py -v
```

## Integration Tests

**File**: `integration/test_phase2_integration.py`

**Coverage**:
- File ingestion → chunking → storage pipeline
- Clustering → topic discovery
- MCP tool registration and execution
- Hybrid search (graph + vector)
- End-to-end complete pipeline
- Concurrent operations
- Error handling and graceful degradation
- Smart chunking logic

**Key Test Cases**:
1. `test_file_to_storage_pipeline` - Validates file processing → embedding → storage
2. `test_clustering_pipeline` - Tests clustering on synthetic data with 3 clusters
3. `test_mcp_tool_execution` - Validates MCP tool registry and execution
4. `test_hybrid_search_pipeline` - Tests graph + vector hybrid retrieval
5. `test_end_to_end_pipeline` - Complete E2E: ingest → store → cluster → search
6. `test_concurrent_operations` - 10 concurrent tool executions
7. `test_error_handling` - Invalid tools, missing params, nonexistent files
8. `test_smart_chunker` - Chunking with overlap and sentence boundaries

**Fixtures**:
- `unified_store` - Neo4j + Qdrant hybrid store (mock mode if DBs unavailable)
- `cluster_engine` - K-means clustering with 3 clusters
- `mcp_server` - MCP server with test echo/sum tools
- `file_processor` - Multi-format file processor with smart chunking

## Performance Tests

**File**: `performance/test_phase2_performance.py`

**Benchmarks**:

### Storage Performance
- `test_write_throughput_1k` - 1,000 items write throughput (target: >50 items/sec)
- `test_write_throughput_10k` - 10,000 items batched writes (target: >100 items/sec)
- `test_search_latency` - Hybrid search latency (target: <100ms avg, <200ms P95)

### Clustering Performance
- `test_clustering_1k_vectors` - 1,000 vectors, 384D (target: <2s)
- `test_clustering_10k_vectors` - 10,000 vectors, 384D (target: <10s)
- `test_clustering_high_dim` - 1,000 vectors, 1536D (target: <5s)

### MCP Performance
- `test_concurrent_execution_100` - 100 concurrent fast tools (target: >500 tools/sec)
- `test_concurrent_slow_tools` - 50 concurrent 50ms tools (target: <1s wall time)

### File Processing Performance
- `test_batch_processing_10_files` - 10 files in parallel (target: >5 files/sec)
- `test_large_file_processing` - 1MB+ file (target: <5s)

### End-to-End Performance
- `test_full_pipeline_latency` - Complete pipeline latency (target: <5s total)

**Performance Targets Summary**:
```
Component          Metric                    Target
─────────────────────────────────────────────────────
Storage            Write (1K)                >50/sec
Storage            Write (10K batched)       >100/sec
Storage            Search (avg)              <100ms
Storage            Search (P95)              <200ms
Clustering         1K vectors                <2s
Clustering         10K vectors               <10s
Clustering         High-dim (1536D)          <5s
MCP                Concurrent (100 fast)     >500/sec
MCP                Concurrent (50 slow)      <1s
Files              Batch (10 files)          >5/sec
Files              Large file (1MB+)         <5s
E2E Pipeline       Full latency              <5s
```

## Test Dependencies

**Required**:
```bash
pip install pytest pytest-asyncio numpy
```

**Phase 2 Dependencies** (from `requirements-phase2.txt`):
```bash
pip install neo4j qdrant-client hdbscan scikit-learn fastapi pypdf python-docx
```

**Note**: Tests use mock modes when external services (Neo4j, Qdrant) are unavailable.

## Running Tests in CI/CD

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements-phase2.txt
          pip install pytest pytest-asyncio
      - name: Run integration tests
        run: pytest HoloLoom/tests/integration/ -v
      - name: Run performance tests
        run: pytest HoloLoom/tests/performance/ -v -s
```

### Docker Test Runner
```dockerfile
FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install -r requirements-phase2.txt
RUN pip install pytest pytest-asyncio

CMD ["pytest", "HoloLoom/tests/", "-v"]
```

## Mock Mode vs. Real Services

Tests are designed to run in **mock mode** without external services:

- **Neo4j**: Uses in-memory dict when `neo4j` package unavailable or connection fails
- **Qdrant**: Uses in-memory list when `qdrant-client` unavailable or connection fails
- **Embeddings**: Uses random vectors for testing (real embedder not required)

To test with **real services**, ensure:
1. Neo4j running at `bolt://localhost:7687`
2. Qdrant running at `localhost:6333`
3. Set environment variables if needed

## Continuous Performance Monitoring

Track performance over time using pytest-benchmark:

```bash
pip install pytest-benchmark
pytest HoloLoom/tests/performance/ --benchmark-only
```

## Debugging Failed Tests

### Verbose Output
```bash
pytest HoloLoom/tests/ -vv -s
```

### Stop on First Failure
```bash
pytest HoloLoom/tests/ -x
```

### Run Specific Test
```bash
pytest HoloLoom/tests/integration/test_phase2_integration.py::TestPhase2Integration::test_end_to_end_pipeline -v
```

### Show Full Tracebacks
```bash
pytest HoloLoom/tests/ --tb=long
```

## Test Coverage

Generate coverage report:

```bash
pip install pytest-cov
pytest HoloLoom/tests/ --cov=HoloLoom --cov-report=html
```

View coverage: `open htmlcov/index.html`

## Contributing Tests

When adding new Phase 2 features:

1. Add integration test to `integration/test_phase2_integration.py`
2. Add performance benchmark to `performance/test_phase2_performance.py`
3. Ensure mock mode works (no external dependencies required)
4. Document performance targets
5. Run full test suite before committing

## Known Issues

1. **Timezone warnings**: Some tests may show timezone warnings from datetime.now() usage
2. **Async fixtures**: Requires `pytest-asyncio` with `--asyncio-mode=auto`
3. **Random seeds**: Some tests use random data; may need `np.random.seed()` for reproducibility

## Support

For test failures or questions:
- Check test output for specific error messages
- Verify all dependencies installed: `pip list | grep -E "neo4j|qdrant|hdbscan"`
- Review logs in `HoloLoom/logs/` if available
- Open issue with full test output
