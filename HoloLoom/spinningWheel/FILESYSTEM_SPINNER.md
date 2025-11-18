# FilesystemSpinner - Local File Ingestion

**Status**: Production Ready (November 2025)
**Location**: `HoloLoom/spinningWheel/filesystem_spinner.py`
**Total Code**: 600+ lines

Convert local files into MemoryShards for HoloLoom's RAG system.

---

## Overview

FilesystemSpinner scans directories on your filesystem, reads text files, chunks them intelligently, and converts them into `MemoryShard` objects that integrate seamlessly with HoloLoom's memory and retrieval systems.

**Key Features**:
- Directory scanning with glob patterns (allow/deny)
- Smart chunking with overlap for context preservation
- Incremental ingestion (only new/modified files)
- Importance scoring based on file properties
- Graceful handling of binary files
- Zero external dependencies

---

## Quick Start

### Programmatic Usage

```python
from HoloLoom.spinningWheel.filesystem_spinner import FilesystemSpinner

# Create spinner
spinner = FilesystemSpinner(
    chunk_size=1000,
    chunk_overlap=200,
    allow_patterns=["*.md", "*.txt"],
    deny_patterns=["*.log", ".git/**", "node_modules/**"]
)

# Ingest directory
result = await spinner.spin("/path/to/docs")

print(f"Created {result.shard_count} shards")
print(f"Average importance: {result.avg_importance:.2f}")
```

### CLI Usage

```bash
# Basic ingestion
python -m HoloLoom.ingestion.filesystem /path/to/docs

# Custom patterns
python -m HoloLoom.ingestion.filesystem /path/to/docs \
    --allow "*.md" "*.rst" \
    --deny "build/**" "*.log"

# Incremental (only new/modified files)
python -m HoloLoom.ingestion.filesystem /path/to/docs --incremental

# Integrate with HoloLoom memory
python -m HoloLoom.ingestion.filesystem /path/to/docs --integrate

# Preview without ingesting
python -m HoloLoom.ingestion.filesystem /path/to/docs --dry-run
```

---

## Configuration

### FilesystemSpinnerConfig

```python
from HoloLoom.spinningWheel.filesystem_spinner import FilesystemSpinnerConfig

config = FilesystemSpinnerConfig(
    # Chunking
    chunk_size=1000,           # Characters per chunk
    chunk_overlap=200,         # Overlap between chunks (for context)
    min_chunk_size=100,        # Discard chunks smaller than this

    # File filtering
    allow_patterns=["*.md", "*.txt"],
    deny_patterns=[
        "*.log",
        ".git/**",
        "node_modules/**",
        ".venv/**",
        "__pycache__/**"
    ],

    # File size limits
    max_file_size=10 * 1024 * 1024,  # 10MB default
    skip_binary=True,                 # Skip binary files

    # Importance scoring
    boost_markdown=1.2,     # Boost importance for Markdown
    boost_recent=1.1,       # Boost recently modified files

    # Incremental ingestion
    track_mtimes=True,      # Track modification times
    checkpoint_enabled=True # Enable checkpointing
)

spinner = FilesystemSpinner(config=config)
```

---

## Features

### 1. Smart Chunking

Files are split into chunks with configurable overlap:

```python
spinner = FilesystemSpinner(
    chunk_size=1000,    # Max 1000 characters per chunk
    chunk_overlap=200   # 200 character overlap for context
)
```

**Intelligent break points**:
- Prefers paragraph breaks (`\n\n`)
- Falls back to sentence breaks (`. ` or `.\n`)
- Avoids breaking mid-word

### 2. Incremental Ingestion

Only process new or modified files:

```python
# First run: processes all files
result1 = await spinner.spin_incremental("/path/to/docs")

# Second run: skips unchanged files
result2 = await spinner.spin_incremental("/path/to/docs")
# → result2.shard_count == 0

# Modify a file...
# Third run: processes only modified file
result3 = await spinner.spin_incremental("/path/to/docs")
```

**Checkpoint mechanism**:
- Tracks `path → mtime` mapping
- Stored in `~/.hololoom/checkpoints/` by default
- Automatic checkpoint save after each run

### 3. Importance Scoring

Each shard gets an importance score (0.0-1.0) based on:

| Signal | Weight | Description |
|--------|--------|-------------|
| **Length** | 0.15 | Longer files = more substantive |
| **Technical** | 0.20 | Markdown > plain text |
| **Recency** | 0.10 | Recently modified files |
| **Structural** | 0.10 | READMEs, docs, guides boosted |
| **Authority** | 0.20 | Source credibility (future) |
| **Engagement** | 0.15 | Reactions, shares (future) |
| **Reference** | 0.10 | Citations, backlinks (future) |

**Usage**:
```python
# Filter low-importance shards
spinner = FilesystemSpinner(importance_threshold=0.5)

result = await spinner.spin("/path/to/docs")
# Only shards with importance ≥ 0.5 are included
```

### 4. Glob Pattern Filtering

**Allow patterns** (inclusive):
```python
allow_patterns=[
    "*.md",          # All Markdown files
    "*.txt",         # All text files
    "*.rst",         # reStructuredText
    "docs/**/*.md"   # Markdown in docs/ subdirectories
]
```

**Deny patterns** (exclusive):
```python
deny_patterns=[
    "*.log",              # Log files
    ".git/**",            # Git directory
    "node_modules/**",    # Node.js dependencies
    ".venv/**",           # Python virtual environment
    "__pycache__/**",     # Python cache
    "build/**",           # Build artifacts
]
```

**Priority**: Deny patterns override allow patterns.

### 5. Metadata Preservation

Each shard includes rich metadata:

```python
{
    "file_path": "/path/to/docs/README.md",
    "file_name": "README.md",
    "file_size": 2048,
    "mtime": 1731931200.0,
    "media_type": "text/markdown",
    "chunk_index": 0,
    "total_chunks": 3,
    "importance_score": 0.85,
    "importance_reason": "high technical content + authoritative source",
    "importance_signals": {
        "length_score": 0.7,
        "technical_score": 0.8,
        "recency_score": 1.0,
        "structural_score": 1.0,
        ...
    }
}
```

---

## Integration with HoloLoom

### Add to Memory

```python
from HoloLoom import HoloLoom
from HoloLoom.config import Config

# Ingest files
spinner = FilesystemSpinner()
result = await spinner.spin("/path/to/docs")

# Add to HoloLoom memory
config = Config.fast()
async with HoloLoom(cfg=config) as loom:
    for shard in result.shards:
        await loom.experience(shard.text)

    # Query
    memories = await loom.recall("What did I learn about embeddings?")
```

### Use with SimpleRAG

```python
from HoloLoom.rag import SimpleRAG

# Ingest files
spinner = FilesystemSpinner()
result = await spinner.spin("/path/to/docs")

# Add to RAG
async with SimpleRAG() as rag:
    for shard in result.shards:
        await rag.ingest(shard.text)

    # Query
    result = await rag.query("Explain Thompson Sampling")
    print(result.response)
```

---

## CLI Reference

```bash
python -m HoloLoom.ingestion.filesystem [OPTIONS] DIRECTORY

Arguments:
  DIRECTORY                 Directory to ingest

Options:
  --allow PATTERN [...]     Glob patterns to include (default: *.txt *.md)
  --deny PATTERN [...]      Glob patterns to exclude
  --no-recursive            Don't scan subdirectories

  --chunk-size SIZE         Characters per chunk (default: 1000)
  --chunk-overlap SIZE      Overlap between chunks (default: 200)

  --incremental             Only process new/modified files
  --checkpoint-dir DIR      Directory for checkpoints

  --importance-threshold N  Minimum importance (0.0-1.0, default: 0.0)

  --integrate               Add shards to HoloLoom memory
  --dry-run                 Preview without ingesting
  --output {text,json}      Output format (default: text)
```

### Examples

```bash
# Ingest documentation
python -m HoloLoom.ingestion.filesystem docs/ \
    --allow "*.md" "*.rst" \
    --incremental

# Large knowledge base (high threshold)
python -m HoloLoom.ingestion.filesystem /path/to/kb \
    --importance-threshold 0.6 \
    --chunk-size 2000

# JSON output for scripting
python -m HoloLoom.ingestion.filesystem docs/ \
    --output json > ingestion_stats.json
```

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **File scanning** | ~1-5ms per file | OS-dependent |
| **Text reading** | ~5-10ms per file | For <10MB files |
| **Chunking** | ~2-5ms per file | Smart break detection |
| **Importance scoring** | <1ms per file | Lightweight heuristics |
| **Total throughput** | ~50-100 files/sec | Text files <100KB |

**Memory usage**: ~1-2MB per 1000 shards (before embedding)

---

## Supported File Types

| Format | Status | Media Type | Notes |
|--------|--------|------------|-------|
| **Text** | ✅ | `text/plain` | UTF-8 or Latin-1 |
| **Markdown** | ✅ | `text/markdown` | GitHub-flavored Markdown |
| **reStructuredText** | 🟡 | `text/x-rst` | Basic support |
| **PDF** | ❌ | - | v2.0 (planned) |
| **DOCX** | ❌ | - | v2.0 (planned) |
| **HTML** | ❌ | - | v2.0 (planned) |

---

## Roadmap

**v1.0** (Current):
- ✅ Text and Markdown support
- ✅ Incremental ingestion
- ✅ Importance scoring
- ✅ CLI interface

**v1.1** (Q1 2026):
- Streaming ingestion for large directories
- File-watch integration (real-time updates)
- Custom importance scoring functions

**v2.0** (Q2 2026):
- PDF support (via pypdf)
- DOCX support (via python-docx)
- HTML support (via BeautifulSoup)
- Advanced entity extraction

---

## Testing

```bash
# Run all tests
pytest HoloLoom/tests/integration/test_filesystem_spinner.py -v

# Run specific test
pytest HoloLoom/tests/integration/test_filesystem_spinner.py::test_incremental_ingestion -v

# Run demo
PYTHONPATH=. python demos/demo_filesystem_rag.py
```

---

## Troubleshooting

### Issue: "Binary file" errors

**Cause**: File contains null bytes (binary data)
**Solution**: Set `skip_binary=False` or exclude pattern

```python
spinner = FilesystemSpinner(
    deny_patterns=["*.pdf", "*.png", "*.jpg"]
)
```

### Issue: "UnicodeDecodeError"

**Cause**: Non-UTF-8 encoding
**Solution**: FilesystemSpinner automatically falls back to Latin-1

### Issue: No shards created

**Cause**: All files excluded by deny patterns
**Solution**: Check `--dry-run` output to see what's being excluded

```bash
python -m HoloLoom.ingestion.filesystem /path/to/docs --dry-run
```

### Issue: Low importance scores

**Cause**: Files are small or recently created
**Solution**: Adjust importance threshold or scoring config

```python
config = FilesystemSpinnerConfig(
    boost_markdown=1.5,  # Higher boost for Markdown
    boost_recent=1.2     # Higher boost for recent files
)
```

---

## See Also

- **SpinningWheel README**: `HoloLoom/spinningWheel/README.md`
- **RAG System**: `HoloLoom/rag/README.md`
- **SimpleRAG API**: `HoloLoom/rag/simple_rag.py`
- **Demo Script**: `demos/demo_filesystem_rag.py`
- **Tests**: `HoloLoom/tests/integration/test_filesystem_spinner.py`

---

**Author**: Claude Code
**Date**: November 2025
**License**: Same as HoloLoom
