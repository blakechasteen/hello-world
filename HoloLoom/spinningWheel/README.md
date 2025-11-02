# HoloLoom SpinningWheel

**Universal data ingestion for HoloLoom memory system**

Version: 2.0
Philosophy: *"If you need to configure it, we failed."*

---

## Overview

**SpinningWheel** transforms **any input** into **queryable MemoryShards** through a standardized, protocol-based architecture.

```python
from HoloLoom.spinningWheel import spin

# Ingest anything into memory
memory = await spin("My thoughts on Thompson Sampling...")
memory = await spin("/path/to/document.pdf")
memory = await spin("https://example.com/article")
memory = await spin([text, image, audio])  # Multi-modal
```

---

## Quick Links

| Document | Description |
|----------|-------------|
| **[PROTOCOL_GUIDE.md](PROTOCOL_GUIDE.md)** | Complete guide to building spinners |
| **[PIPELINE.md](PIPELINE.md)** | Data flow architecture (input → shard) |
| **[protocol.py](protocol.py)** | SpinnerProtocol interface definition |
| **[importance.py](importance.py)** | Importance scoring framework |
| **[utils.py](utils.py)** | Checkpointing, streaming, batch processing |

---

## Available Spinners

### Production Spinners

| Spinner | Input | Output | Status |
|---------|-------|--------|--------|
| **MultiModalSpinner** | Text, image, audio, structured data | MemoryShards with modality metadata | ✅ Active |
| **ChatHistorySpinner** | Conversation history | Conversation turns with importance scoring | ✅ Active |

### Planned Spinners (Tier 1)

| Spinner | Input | Importance | Priority |
|---------|-------|------------|----------|
| **GitSpinner** | Git repository commits | High | 1 |
| **EmailSpinner** | Email archives (IMAP, mbox) | High | 2 |
| **PDFSpinner** | PDF documents | High | 3 |
| **CodebaseSpinner** | Source code (AST, call graphs) | High | 4 |
| **SlackSpinner** | Slack/Discord/Teams messages | Medium | 5 |

See [Spinner Expansion Review](../SPINNER_EXPANSION_REVIEW.md) for complete roadmap (15 spinners).

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    SPINNING WHEEL                            │
└─────────────────────────────────────────────────────────────┘

Input (Anything) → InputRouter → Processor → MemoryShard(s)
                        │             │
                        ├─ TEXT → TextProcessor
                        ├─ IMAGE → ImageProcessor
                        ├─ AUDIO → AudioProcessor
                        └─ STRUCTURED → StructuredDataProcessor

All Spinners Implement SpinnerProtocol:
    - get_name() → str
    - get_capabilities() → SpinnerCapabilities
    - is_available() → bool
    - spin(source) → SpinResult
```

### Protocol Hierarchy

```python
SpinnerProtocol (Interface)
    ↓
BaseSpinner (Abstract Implementation)
    ├─ Error handling
    ├─ Checkpointing
    ├─ Importance filtering
    └─ Performance tracking
    ↓
Your Spinner (Concrete Implementation)
    └─ _spin_impl(source) → List[MemoryShard]
```

---

## Usage

### Simple Ingestion

```python
from HoloLoom.spinningWheel import spin

# Text
memory = await spin("Thompson Sampling balances exploration and exploitation")

# File
memory = await spin("/path/to/research_paper.pdf")

# URL
memory = await spin("https://wikipedia.org/wiki/Thompson_Sampling")

# Multi-modal
memory = await spin([text, image, audio])
```

### Batch Processing

```python
from HoloLoom.spinningWheel import spin_batch

sources = [
    "Text content",
    "/path/to/image.png",
    "https://example.com/article",
    "/path/to/audio.mp3"
]

memory = await spin_batch(sources, max_concurrent=5)
```

### Directory Ingestion

```python
from HoloLoom.spinningWheel import spin_directory

# Recursive ingestion
memory = await spin_directory(
    "/path/to/research",
    pattern="*.pdf",
    recursive=True
)
```

### Streaming (Large Sources)

```python
from HoloLoom.spinningWheel import MultiModalSpinner

spinner = MultiModalSpinner()

async for shard in spinner.spin_stream(large_source):
    await memory.add_shard(shard)
    print(f"Processed: {shard.id}")
```

---

## Building Your Own Spinner

### 3-Step Quick Start

```python
from HoloLoom.spinningWheel.protocol import BaseSpinner, SpinnerCapabilities
from HoloLoom.documentation.types import MemoryShard

class MySpinner(BaseSpinner):
    def __init__(self):
        super().__init__(name="my_spinner")

    # Step 1: Define capabilities
    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            supported_formats=['txt']
        )

    # Step 2: Check availability
    def is_available(self) -> bool:
        try:
            import required_library
            return True
        except ImportError:
            return False

    # Step 3: Implement core logic
    async def _spin_impl(self, source, **kwargs):
        # Convert source → MemoryShards
        shard = self._create_shard(
            id_suffix="001",
            text=str(source),
            episode="my_episode",
            entities=["Entity1"],
            motifs=["topic1"],
            metadata={'confidence': 0.9}
        )
        return [shard]
```

**See [PROTOCOL_GUIDE.md](PROTOCOL_GUIDE.md) for complete documentation.**

---

## Importance Scoring

All spinners can score importance to filter noise:

```python
from HoloLoom.spinningWheel.importance import ImportanceScorer

scorer = ImportanceScorer()

importance = scorer.score(
    text="Thompson Sampling is a Bayesian approach...",
    source="research_paper.pdf",
    timestamp=time.time(),
    engagement={'likes': 10, 'shares': 5}
)

print(f"Importance: {importance.score:.2f}")
print(f"Reason: {importance.reason}")

# Output:
# Importance: 0.78
# Reason: substantive length + high technical content + authoritative source
```

### Preset Configurations

```python
from HoloLoom.spinningWheel.importance import (
    create_chat_scorer,
    create_git_scorer,
    create_email_scorer,
    create_document_scorer
)

# Use domain-specific scorer
spinner.scorer = create_git_scorer()
```

---

## Features

### ✅ Implemented

- **Protocol-Based Architecture**: All spinners implement `SpinnerProtocol`
- **Graceful Degradation**: Missing dependencies handled automatically
- **Importance Scoring**: 7 standardized signals (length, technical, authority, etc.)
- **Checkpointing**: Resume long operations
- **Streaming**: Memory-efficient processing
- **Batch Processing**: Concurrent ingestion with limits
- **Error Handling**: Automatic error recovery
- **Progress Tracking**: Visual feedback for long operations
- **Deduplication**: Filter duplicate shards
- **Multi-Modal Support**: Text, image, audio, structured data

### 🚧 In Progress

- GitSpinner (Tier 1 priority)
- EmailSpinner (Tier 1 priority)
- PDFSpinner (Tier 1 priority)

---

## Performance

### Latency by Modality

| Input Type | Latency | Bottleneck |
|------------|---------|------------|
| Text (simple) | ~10ms | Embedding + NER |
| Image (CLIP + OCR) | ~183ms | CLIP inference |
| Audio (Whisper) | ~350ms | STT transcription |
| Multi-modal (3 inputs, attention) | ~576ms | Parallel processing + fusion |

### Throughput

- **Sequential**: N × latency
- **Concurrent (max=10)**: ~10× speedup for I/O-bound tasks

See [PIPELINE.md](PIPELINE.md) for detailed performance analysis.

---

## Testing

Run tests:

```bash
# All spinner tests
pytest HoloLoom/tests/unit/test_spinner_protocol.py -v

# Specific test
pytest HoloLoom/tests/unit/test_spinner_protocol.py::test_importance_scorer -v
```

Test coverage:
- ✅ SpinnerProtocol interface (5 tests)
- ✅ BaseSpinner implementation (6 tests)
- ✅ ImportanceScorer (12 tests)
- ✅ Checkpointing (4 tests)
- ✅ Streaming (2 tests)
- ✅ Batch processing (3 tests)
- ✅ Utilities (8 tests)

**Total: 40+ tests**

---

## File Structure

```
HoloLoom/spinningWheel/
├── __init__.py                      # Public API exports
├── README.md                        # This file
├── PIPELINE.md                      # Data flow architecture (1000+ lines)
├── PROTOCOL_GUIDE.md                # Complete spinner guide (1200+ lines)
│
├── protocol.py                      # SpinnerProtocol interface (600+ lines)
├── importance.py                    # Importance scoring framework (500+ lines)
├── utils.py                         # Utilities (checkpointing, streaming) (400+ lines)
│
├── auto.py                          # Convenience functions (spin, spin_batch) (430+ lines)
├── multimodal_spinner.py            # Multi-modal spinner (380+ lines)
├── chat_history.py                  # Chat history spinner (590+ lines)
│
└── tests/
    └── test_spinner_protocol.py     # Comprehensive tests (700+ lines)
```

**Total Code**: ~5,800 lines

---

## Dependencies

### Required
- `HoloLoom.documentation.types` (MemoryShard)
- `HoloLoom.input.*` (InputRouter, processors)

### Optional (Graceful Degradation)
- `spaCy` - Advanced NER
- `sentence-transformers` - Semantic embeddings
- `CLIP` - Image embeddings
- `Whisper` - Audio transcription
- `pandas` - Structured data processing

---

## Contributing

### Adding a New Spinner

1. **Inherit from BaseSpinner**:
   ```python
   class MySpinner(BaseSpinner):
       def __init__(self):
           super().__init__(name="my_spinner")
   ```

2. **Implement required methods**:
   - `get_capabilities()`
   - `is_available()`
   - `_spin_impl()`

3. **Add tests**:
   ```python
   class TestMySpinner:
       def test_spin(self):
           spinner = MySpinner()
           result = await spinner.spin(source)
           assert result.success
   ```

4. **Document**:
   - Add to this README
   - Add usage example to PROTOCOL_GUIDE.md

5. **Submit PR** with tests passing

---

## Spinner Roadmap

### Phase 1: Core Data Spinners (Q4 2025) ✅ COMPLETE
- ✅ MultiModalSpinner (text, image, audio, structured)
- ✅ ChatHistorySpinner (conversation history)
- ✅ GitSpinner (repository commits) - 19/19 tests
- ✅ MatrixSpinner (Matrix.org chat) - 17/17 tests
- 🚧 PDFSpinner (document extraction) - IN PROGRESS
- 🚧 EmailSpinner (IMAP/mbox) - IN PROGRESS
- 🚧 CodebaseSpinner (AST parsing) - IN PROGRESS

### Phase 2: Developer Tools (Q1 2026)
- SlackSpinner (Slack/Teams/Discord patterns)
- NotionSpinner (Notion database export)
- JupyterSpinner (notebook cells)
- StackOverflowSpinner (Q&A threads)

### Phase 3: Knowledge Sources (Q2 2026)
- BrowserHistorySpinner (Chrome/Firefox/Edge)
- CalendarSpinner (Google/Outlook events)
- RSSSpinner (feed aggregation)
- TwitterSpinner (tweet archives)
- YouTubeSpinner (video transcripts) - ALREADY EXISTS!

### Phase 4: Specialized Domains (Q3 2026)
- ResearchSpinner (ArXiv, PubMed papers)
- CodeReviewSpinner (GitHub PRs)
- DatabaseSpinner (SQL schema + data)
- LogSpinner (application logs)

---

## Examples

### Example 1: Chat History

```python
from HoloLoom.spinningWheel.chat_history import ChatHistorySpinner
from HoloLoom.web_dashboard.conversation_manager import ConversationManager

conv_mgr = ConversationManager("./conversations.db")
spinner = ChatHistorySpinner(conv_mgr, importance_threshold=0.3)

# Spin all conversations
shards = await spinner.spin_all()

# Spin recent (last 24 hours)
shards = await spinner.spin_recent(hours=24)

# Spin by tag
shards = await spinner.spin_by_tag("alignment")
```

### Example 2: Multi-Modal

```python
from HoloLoom.spinningWheel import spin

# Text + Image fusion
inputs = [
    "This diagram shows neural network architecture",
    "/path/to/nn_diagram.png"
]

memory = await spin(inputs)

# Automatically creates:
# - Text shard
# - Image shard
# - Fused shard (attention-based fusion)
```

### Example 3: Batch Directory

```python
from HoloLoom.spinningWheel import spin_directory

# Ingest all PDFs recursively
memory = await spin_directory(
    "/path/to/research_papers",
    pattern="*.pdf",
    recursive=True
)

# Result: N shards (one per PDF)
```

---

## Philosophy

> **"If you need to configure it, we failed."**

SpinningWheel embodies ruthless elegance:
- **Zero configuration**: `spin(anything)` just works
- **Graceful degradation**: Missing dependencies → fallback, not crash
- **Protocol-based**: Consistent API across all data sources
- **Performance-aware**: Streaming, checkpointing, batch processing
- **Quality-first**: Importance scoring filters noise

Everything you've ever experienced becomes queryable through a unified memory interface.

---

## License

Part of HoloLoom project. See root LICENSE.

---

## Contact

- Issues: [GitHub Issues](https://github.com/anthropics/HoloLoom/issues)
- Docs: [PROTOCOL_GUIDE.md](PROTOCOL_GUIDE.md)
- Tests: [test_spinner_protocol.py](../tests/unit/test_spinner_protocol.py)
