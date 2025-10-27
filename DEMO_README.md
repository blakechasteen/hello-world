# SpinningWheel Demo

Interactive demonstration of the HoloLoom SpinningWheel module - a multi-modal data ingestion system.

## Quick Start

```bash
# From repository root
PYTHONPATH=. python demo_spinningwheel_simple.py
```

## What It Demonstrates

### 1. TextSpinner - Plain Text Processing
- Processes raw text documents
- Extracts entities automatically
- Generates structured MemoryShards
- **Performance**: Instant (< 1ms per document)

### 2. WebsiteSpinner - Web Content Extraction
- Scrapes web pages with metadata
- Automatic content chunking
- Domain and URL tracking
- Tag support for categorization
- **Performance**: Instant processing

### 3. YouTubeSpinner - Video Transcript Extraction
- Processes YouTube video transcripts
- Preserves timestamp information
- Supports multi-segment videos
- **Performance**: ~2.6s per video

### 4. Batch Processing
- Multiple document processing
- Parallel execution support
- Progress tracking
- **Performance**: INSTANT throughput for small batches

### 5. Performance Benchmarks
- Throughput measurement
- Latency tracking
- Scalability testing
- **Baseline**: 20,000+ items/sec (from integration tests)

### 6. Memory Integration
- Shard → Memory conversion
- HoloLoom orchestrator compatibility
- Ready for production use
- **Status**: ✅ SUCCESS

## Expected Output

```
======================================================================
                SpinningWheel Demo - Production Ready
======================================================================

1. TextSpinner - Plain Text Processing
[OK] Processed in 0.000s
  Shards generated: 1
  Text length: 330
  Entities extracted: 3
  Metadata: {'source': 'beekeeping_intro.txt', 'format': 'text', ...}

2. WebsiteSpinner - Web Content Extraction
[OK] Processed in 0.000s
  Shards generated: 3
  URL: https://example.com/winter-beekeeping
  Domain: example.com
  Tags: ['web:example.com', 'beekeeping', 'winter', 'guide']

[... continues through all 6 demos ...]

Demo Complete
```

## Features Highlighted

- ✅ **Multi-Modal Ingestion**: Text, web, video, code, PDF support
- ✅ **High Performance**: 20,000+ items/sec throughput
- ✅ **Entity Extraction**: Automatic entity identification
- ✅ **Memory Integration**: Seamless orchestrator compatibility
- ✅ **Batch Processing**: Parallel document handling
- ✅ **Production Ready**: Comprehensive test coverage (26/26 passing)

## Next Steps

After running the demo, explore:

1. **Integration Tests**:
   ```bash
   PYTHONPATH=. python HoloLoom/spinningWheel/tests/test_integration.py
   ```

2. **Unit Tests**:
   ```bash
   PYTHONPATH=. python HoloLoom/spinningWheel/tests/run_tests.py
   ```

3. **Batch Utilities**:
   ```bash
   # See batch ingestion examples in:
   HoloLoom/spinningWheel/batch_utils.py
   ```

4. **Individual Spinners**:
   ```bash
   # Explore examples:
   HoloLoom/spinningWheel/examples/
   ```

## Technical Details

### Spinners Demonstrated

| Spinner | Purpose | Input Format | Output |
|---------|---------|--------------|--------|
| TextSpinner | Plain text | String + metadata | MemoryShards |
| WebsiteSpinner | Web content | URL + content | MemoryShards with web metadata |
| YouTubeSpinner | Video transcripts | Video ID + transcript | MemoryShards with timestamps |
| PDFSpinner* | PDF documents | Path + pages | MemoryShards per page |
| CodeSpinner* | Source code | Code + language | MemoryShards with code metadata |

*Not included in simple demo but available in module

### Performance Metrics

From integration test suite:

- **TextSpinner**: 20,000 items/sec
- **CodeSpinner**: 33,000 items/sec
- **WebsiteSpinner**: 7,600 items/sec
- **Shard Generation**: 100,000 items/sec
- **Shard Conversion**: 16,600 items/sec

### Test Coverage

- **Unit Tests**: 17/17 passing (100%)
- **Integration Tests**: 9/9 passing (100%)
- **Total**: 26/26 tests passing (100%)

## Architecture

```
Raw Data → Spinner → MemoryShards → Memory System → Orchestrator
```

### MemoryShard Structure

```python
@dataclass
class MemoryShard:
    id: str                      # Unique identifier
    text: str                    # Content
    episode: str                 # Session/episode ID
    entities: List[str]          # Extracted entities
    motifs: List[str]            # Detected patterns
    metadata: Dict[str, Any]     # Format, source, timestamps, etc.
```

## Troubleshooting

### Import Errors

Ensure `PYTHONPATH=.` is set when running from repository root:

```bash
# Correct
PYTHONPATH=. python demo_spinningwheel_simple.py

# Incorrect
python demo_spinningwheel_simple.py  # Will fail with ModuleNotFoundError
```

### Unicode Errors (Windows)

The demo uses ASCII-safe characters to avoid Windows cp1252 encoding issues. If you see encoding errors, ensure your terminal supports UTF-8.

### Missing Dependencies

Optional dependencies (spaCy, sentence-transformers) will gracefully degrade:

```
pip install spacy sentence-transformers
python -m spacy download en_core_web_sm
```

## Development

Built with:
- Python 3.12+
- asyncio for async processing
- HoloLoom architecture (Warp, Loom, Chrono, Resonance)
- Test-driven development (100% passing)

## License

Part of the HoloLoom project. See repository LICENSE for details.

## Support

- GitHub Issues: https://github.com/blakechasteen/hello-world/issues
- Documentation: See HoloLoom/spinningWheel/README.md
- Tests: HoloLoom/spinningWheel/tests/
