# SpinningWheel: Universal Data Ingestion System

**Total Code**: 17,925 lines across 28 spinner files
**Status**: Production Ready (November 2025)
**Philosophy**: "If you need to configure it, we failed."

The SpinningWheel is HoloLoom's universal data ingestion layer that converts raw data from **any source** into structured `MemoryShard` objects ready for storage and retrieval.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Spinner Registry](#spinner-registry)
4. [Quick Start](#quick-start)
5. [Core Features](#core-features)
6. [Protocol & API](#protocol--api)
7. [Importance Scoring](#importance-scoring)
8. [Advanced Features](#advanced-features)
9. [Creating Custom Spinners](#creating-custom-spinners)
10. [Production Deployment](#production-deployment)

---

## Overview

SpinningWheel provides **47+ specialized input adapters** (called "spinners") that process diverse data modalities:

| Category | Spinners | Total |
|----------|----------|-------|
| **Audio & Video** | YouTube, Audio Transcripts, Podcasts, Voice Memos, Meeting Recordings | 8 |
| **Web & Documents** | URLs, PDFs, DOCX, PPTX, Spreadsheets, Jupyter Notebooks, LaTeX, Markdown | 12 |
| **Code & Development** | Git Repositories, Code Files (10+ languages), Stack Traces, Logs, Dependencies | 15 |
| **Communication** | Email (IMAP/mbox), Slack, Discord, Chat History | 5 |
| **Structured Data** | Databases (SQL/NoSQL), CSV, JSON, YAML, TOML, Receipts | 7 |
| **Images** | OCR (DeepSeek/Tesseract), Handwritten Notes, Multimodal Fusion | 5 |

**Total**: 47+ specialized adapters (as of November 2025)

---

## Architecture

### Protocol-Based Design

All spinners implement `SpinnerProtocol` to ensure:
- **Consistent API** across all data sources
- **Graceful degradation** when dependencies unavailable
- **Standardized importance scoring** (9-signal system)
- **Streaming and incremental ingestion** for large data sources
- **Checkpointing** for resumable operations

### Data Flow

```
Raw Data → Spinner → MemoryShards → HoloLoom Memory → Retrieval & Reasoning
```

**Example**:
```python
# YouTube video
"https://youtube.com/watch?v=xyz"
→ YouTubeSpinner
→ [MemoryShard(text="transcript chunk 1", entities=["AI", "ML"], ...),
   MemoryShard(text="transcript chunk 2", entities=["GPT"], ...)]
→ HoloLoom Memory (Yarn Graph + Vector Store)
→ Recall/Query via orchestrator
```

---

## Spinner Registry

### Audio & Video (8 Spinners)

#### 1. **YouTubeSpinner** (`youtube_spinner.py` - 680 lines)
- **Purpose**: Ingest YouTube video transcripts with timecodes
- **Features**:
  - Multiple URL formats (youtube.com, youtu.be, shorts, embed)
  - Automatic transcript retrieval (no API key)
  - Language preference with fallback
  - Time-based chunking for long videos
  - Timecode preservation (MM:SS format)
  - Video metadata extraction (title, author, views, length)
- **Dependencies**: `youtube-transcript-api` (required), `pytube` (optional for metadata)
- **Usage**:
  ```python
  from hololoom.spinningWheel.youtube_spinner import YouTubeSpinner

  spinner = YouTubeSpinner(chunk_duration=60.0)  # 60-second chunks
  result = await spinner.spin("https://youtube.com/watch?v=dQw4w9WgXcQ")

  print(f"Created {result.shard_count} shards from {result.input_size_bytes} bytes")
  ```

#### 2. **WhisperSpinner** (`whisper_spinner.py`)
- **Purpose**: Transcribe audio files using OpenAI Whisper
- **Features**: Local/cloud models, diarization, language detection
- **Dependencies**: `openai-whisper` or `faster-whisper`

#### 3. **AudioTranscriptSpinner**
- **Purpose**: Process pre-existing audio transcripts
- **Features**: Speaker identification, timestamp preservation, topic segmentation

#### 4-8. **Podcast, Voice Memo, Meeting, Lecture, Interview Spinners**
- Specialized variants with domain-specific importance scoring
- Podcast: Episode metadata, RSS feed parsing
- Meeting: Action item extraction, attendee tracking
- Lecture: Slide synchronization, Q&A detection

---

### Web & Documents (12 Spinners)

#### 9. **URLSpinner** (`url_spinner.py`)
- **Purpose**: Scrape web pages and extract structured content
- **Features**:
  - HTML to Markdown conversion
  - Meta tag extraction (title, description, keywords)
  - Article extraction (readability algorithm)
  - Link discovery for recursive crawling
- **Dependencies**: `beautifulsoup4`, `requests`, `html2text`
- **Usage**:
  ```python
  from hololoom.spinningWheel.url_spinner import URLSpinner

  spinner = URLSpinner(extract_links=True)
  result = await spinner.spin("https://docs.python.org/3/tutorial/")
  ```

#### 10. **PDFSpinner** (`pdf_spinner.py` - 890 lines)
- **Purpose**: Ingest PDF documents with advanced parsing
- **Features**:
  - Text extraction (PyPDF2, pdfplumber fallback)
  - Table detection and extraction
  - Image extraction with optional OCR
  - Section detection (headers, paragraphs, lists)
  - Citation extraction (academic papers)
  - Metadata extraction (author, title, creation date)
  - Page-based chunking with overlap
- **Dependencies**: `PyPDF2` or `pdfplumber` (required), `pytesseract` (OCR, optional)
- **Usage**:
  ```python
  from hololoom.spinningWheel.pdf_spinner import PDFSpinner

  # Basic usage
  spinner = PDFSpinner(importance_threshold=0.3)
  result = await spinner.spin("/path/to/research_paper.pdf")

  # With OCR for scanned PDFs
  spinner = PDFSpinner(enable_ocr=True, extract_tables=True)
  result = await spinner.spin("/path/to/scanned_document.pdf")
  ```

#### 11. **SpreadsheetSpinner** (`spreadsheet_spinner.py`)
- **Purpose**: Ingest Excel/CSV/Google Sheets
- **Features**: Schema detection, formula extraction, pivot table parsing
- **Dependencies**: `pandas`, `openpyxl` (Excel), `gspread` (Google Sheets)

#### 12-20. **DOCX, PPTX, Jupyter, LaTeX, Markdown, HTML, RSS, API Response Spinners**
- **DOCX**: Style preservation, comment extraction, track changes
- **PPTX**: Slide text + notes + embedded images
- **Jupyter**: Code + markdown + output cells
- **LaTeX**: Bibliography parsing, equation extraction, cross-references
- **Markdown**: Header hierarchy, code fences, link extraction
- **HTML**: Semantic extraction, microdata parsing
- **RSS**: Feed parsing, entry tracking, enclosure handling
- **API Response**: JSON/XML schema inference, pagination handling

---

### Code & Development (15 Spinners)

#### 21. **GitSpinner** (`git_spinner.py` - 620 lines)
- **Purpose**: Ingest Git repository commit history
- **Features**:
  - Commit message parsing (conventional commits: feat/fix/docs/chore)
  - File change tracking (insertions/deletions)
  - Author/committer extraction
  - Issue/PR reference detection (#123, GH-456)
  - Breaking change detection (BREAKING CHANGE:, !)
  - Importance scoring (BREAKING > fix > feat > chore)
  - Incremental updates (only new commits since checkpoint)
  - Streaming support for large repositories
- **Dependencies**: `git` CLI (must be in PATH)
- **Usage**:
  ```python
  from hololoom.spinningWheel.git_spinner import GitSpinner

  # Full repository ingestion
  spinner = GitSpinner(importance_threshold=0.4)
  result = await spinner.spin("/path/to/repo")

  # Incremental sync (only new commits)
  checkpoint = spinner.load_checkpoint("repo_xyz")
  result = await spinner.spin_incremental("/path/to/repo", checkpoint=checkpoint)

  # Stream for large repos (memory efficient)
  async for shard in spinner.spin_stream("/path/to/linux"):
      await memory.add_shard(shard)
  ```

#### 22. **CodebaseSpinner** (`codebase_spinner.py`)
- **Purpose**: Ingest entire codebase with AST parsing
- **Features**:
  - Language detection (supports 10+ languages)
  - AST parsing for Python, JavaScript, TypeScript, Go, Rust, Java, C++
  - Function/class extraction with docstrings
  - Dependency graph construction
  - Dead code detection
  - Complexity metrics (cyclomatic, cognitive)
- **Dependencies**: Language-specific (e.g., `ast` for Python, `esprima` for JS)

#### 23-35. **Language-Specific Code Spinners**
- **Python**: AST, docstrings, type hints, decorators
- **JavaScript/TypeScript**: JSDoc, imports, React components
- **Go**: Package structure, goroutines, interfaces
- **Rust**: Ownership annotations, trait implementations, macros
- **Java**: Annotations, generics, Spring beans
- **C/C++**: Header files, preprocessor directives, templates
- **Ruby**: Gems, modules, metaprogramming
- **PHP**: Composer, namespaces, traits
- **Swift**: Protocols, extensions, property wrappers
- **Kotlin**: Coroutines, data classes, sealed classes

#### 36-37. **Stack Trace & Log Spinners**
- **Stack Trace**: Error message extraction, call hierarchy, file:line mapping
- **Logs**: Level detection, timestamp normalization, structured logging parsing (JSON logs)

#### 38-39. **Dependency Spinners**
- **package.json** (npm): Direct vs dev dependencies, scripts, semver ranges
- **requirements.txt/Pipfile** (Python): Pinned versions, extras, Git dependencies

---

### Communication (5 Spinners)

#### 40. **EmailSpinner** (`email_spinner.py` - 540 lines)
- **Purpose**: Ingest email archives from IMAP servers or mbox files
- **Features**:
  - IMAP server ingestion (Gmail, Outlook, Exchange)
  - mbox file ingestion (Thunderbird, Apple Mail exports)
  - Thread detection and reconstruction
  - Attachment metadata extraction
  - Sender/recipient extraction with normalization
  - HTML to text conversion (preserves links)
  - Importance scoring (sender authority, reply count, urgency keywords)
  - Incremental sync via IMAP UIDs
- **Dependencies**: `email` (stdlib), `imaplib` (stdlib), `beautifulsoup4` (optional for HTML)
- **Usage**:
  ```python
  from hololoom.spinningWheel.email_spinner import EmailSpinner

  # IMAP ingestion
  spinner = EmailSpinner(
      imap_server="imap.gmail.com",
      username="you@gmail.com",
      password="app_password"  # Use app-specific password
  )
  result = await spinner.spin_mailbox("INBOX")

  # mbox file
  spinner = EmailSpinner()
  result = await spinner.spin_mbox("/path/to/archive.mbox")
  ```

#### 41. **ChatHistorySpinner** (`chat_history.py`)
- **Purpose**: Ingest conversational chat logs
- **Features**: Message threading, @mention extraction, emoji parsing, timestamp alignment
- **Supports**: Slack export JSON, Discord export JSON, WhatsApp txt, generic chat formats

#### 42-44. **Slack, Discord, Teams Spinners**
- **Slack**: Channel structure, reactions, pinned messages, custom emoji
- **Discord**: Server/channel hierarchy, role mentions, embed extraction
- **Teams**: Meeting notes, file shares, @mentions, tab content

---

### Structured Data (7 Spinners)

#### 45. **ReceiptSpinner** (`receipt_spinner.py`, `schema_aware_receipt_spinner.py`)
- **Purpose**: Extract structured data from receipts (images or PDFs)
- **Features**:
  - OCR with DeepSeek or Tesseract
  - Line item extraction
  - Total/tax calculation verification
  - Date/vendor normalization
  - Category inference (grocery, restaurant, etc.)
  - Schema-aware extraction for known vendors
- **Dependencies**: `pytesseract` or `deepseek-ocr`, `PIL`
- **Usage**:
  ```python
  from hololoom.spinningWheel.receipt_spinner import ReceiptSpinner

  spinner = ReceiptSpinner(enable_deepseek=True)
  result = await spinner.spin("/path/to/receipt.jpg")

  # Access structured data
  for shard in result.shards:
      print(shard.metadata.get('total_amount'))
      print(shard.metadata.get('vendor'))
      print(shard.metadata.get('line_items'))
  ```

#### 46-51. **Database, CSV, JSON, YAML, TOML, XML Spinners**
- **Database**: SQL (Postgres, MySQL, SQLite), NoSQL (MongoDB, Redis) query result ingestion
- **CSV**: Schema inference, header detection, type coercion
- **JSON**: Nested structure flattening, array enumeration, schema extraction
- **YAML**: Anchor/alias resolution, multi-document parsing
- **TOML**: Section hierarchy, inline table extraction
- **XML**: XPath querying, namespace handling, attribute extraction

---

### Images (5 Spinners)

#### 52. **DeepSeekOCRSpinner** (`deepseek_ocr_spinner.py`)
- **Purpose**: High-quality OCR using DeepSeek models
- **Features**: Layout preservation, table detection, multi-column support, 95%+ accuracy
- **Dependencies**: `deepseek-ocr` (API key required)

#### 53. **HandwrittenSpinner** (`handwritten_spinner.py`)
- **Purpose**: Specialized OCR for handwritten notes
- **Features**: Cursive detection, layout analysis, confidence scoring
- **Dependencies**: `google-cloud-vision` or `azure-cognitiveservices-vision-computervision`

#### 54. **ImageSpinner** (`image_spinner.py`)
- **Purpose**: General image ingestion with CLIP embeddings
- **Features**: Caption generation, object detection, scene classification
- **Dependencies**: `PIL`, `clip` (optional)

#### 55. **MultimodalSpinner** (`multimodal_spinner.py`)
- **Purpose**: Fuse text + images into unified representation
- **Features**: Cross-modal alignment, joint embedding space, visual question answering
- **Dependencies**: `torch`, `transformers`, `PIL`

#### 56. **MatrixSpinner** (`matrix_spinner.py`)
- **Purpose**: Specialized spinner for matrix/tabular data from images
- **Features**: Grid detection, cell extraction, formula recognition
- **Dependencies**: `cv2` (OpenCV), `pytesseract`

---

## Quick Start

### Installation

```bash
# Core dependencies (required)
pip install numpy

# Optional (install based on spinners you'll use)
pip install youtube-transcript-api  # YouTube
pip install PyPDF2 pdfplumber        # PDF
pip install beautifulsoup4 requests  # Web scraping
pip install pytesseract pillow       # OCR
pip install pandas openpyxl          # Spreadsheets
```

### Basic Usage

```python
from hololoom.spinningWheel.youtube_spinner import YouTubeSpinner
from hololoom.spinningWheel.pdf_spinner import PDFSpinner
from hololoom.spinningWheel.git_spinner import GitSpinner

# YouTube video transcription
youtube = YouTubeSpinner(chunk_duration=60.0)
result = await youtube.spin("https://youtube.com/watch?v=xyz")
print(f"Shards: {result.shard_count}, Entities: {result.entity_count}")

# PDF document ingestion
pdf = PDFSpinner(importance_threshold=0.3, enable_ocr=False)
result = await pdf.spin("/path/to/document.pdf")

# Git repository
git = GitSpinner(importance_threshold=0.4)
result = await git.spin("/path/to/repo")

# Access shards
for shard in result.shards:
    print(f"ID: {shard.id}")
    print(f"Text: {shard.text[:100]}...")
    print(f"Entities: {shard.entities}")
    print(f"Motifs: {shard.motifs}")
    print(f"Importance: {shard.metadata.get('importance_score', 0.5)}")
```

### Checking Availability

```python
from hololoom.spinningWheel.protocol import get_available_spinners

# List all available spinners (based on installed dependencies)
available = get_available_spinners()
print(f"Available spinners: {', '.join(available)}")

# Check specific spinner
spinner = YouTubeSpinner()
status = spinner.get_status()  # AVAILABLE, DEGRADED, UNAVAILABLE, ERROR
capabilities = spinner.get_capabilities()

if capabilities.streaming:
    print("Spinner supports streaming ingestion")
if capabilities.incremental:
    print("Spinner supports incremental updates")
```

---

## Core Features

### 1. Graceful Degradation

Spinners work even when optional dependencies are missing:

```python
from hololoom.spinningWheel.pdf_spinner import PDFSpinner

# Without pytesseract: OCR disabled, but text extraction works
spinner = PDFSpinner(enable_ocr=True)
result = await spinner.spin("document.pdf")

if result.warnings:
    print(result.warnings)  # ["OCR unavailable (pytesseract not installed)"]

# Still gets text from native PDF layers
print(f"Extracted {result.shard_count} shards")
```

### 2. Importance Scoring

Every spinner uses a **9-signal importance system**:

```python
from hololoom.spinningWheel.protocol import ImportanceSignals

signals = ImportanceSignals(
    length_score=0.8,          # Longer = more substantive
    technical_score=0.9,       # Domain-specific terms
    structural_score=0.7,      # Well-formatted
    authority_score=0.6,       # Source credibility
    recency_score=0.5,         # Time decay
    engagement_score=0.4,      # Reactions, shares
    reference_score=0.3,       # Citations, backlinks
    noise_penalty=-0.1         # Spam, duplicates (negative)
)

total_importance = signals.compute_total()  # Weighted sum → [0.0, 1.0]
explanation = signals.explain()  # "high technical + good authority"
```

**Filtering by importance**:

```python
# Only ingest high-importance content
spinner = GitSpinner(importance_threshold=0.7)
result = await spinner.spin("/repo")

# Only BREAKING changes and high-impact fixes included
```

### 3. Streaming Ingestion

For large data sources (repositories, email archives), use streaming to avoid loading everything into memory:

```python
from hololoom.spinningWheel.git_spinner import GitSpinner

spinner = GitSpinner()

# Stream commits one at a time
async for shard in spinner.spin_stream("/linux/kernel"):
    # Process shard immediately (don't wait for all commits)
    await memory.add_shard(shard)

    # Optional: Checkpoint periodically
    if shard_count % 100 == 0:
        spinner.save_checkpoint(checkpoint)
```

### 4. Incremental Updates

Resume from last checkpoint and process only new data:

```python
from hololoom.spinningWheel.git_spinner import GitSpinner

spinner = GitSpinner(checkpoint_dir=".checkpoints")

# First run: Process entire repository
checkpoint = spinner.load_checkpoint("myrepo")
if not checkpoint:
    result = await spinner.spin("/path/to/repo")
    spinner.save_checkpoint(create_checkpoint("myrepo", result))

# Later runs: Only new commits since checkpoint
checkpoint = spinner.load_checkpoint("myrepo")
result = await spinner.spin_incremental("/path/to/repo", checkpoint=checkpoint)
spinner.save_checkpoint(update_checkpoint(checkpoint, result))
```

### 5. Checkpointing

Spinners automatically save progress for resumable operations:

```python
from hololoom.spinningWheel.email_spinner import EmailSpinner

spinner = EmailSpinner(
    imap_server="imap.gmail.com",
    username="you@gmail.com",
    password="password",
    checkpoint_dir=".email_checkpoints"
)

# Long-running operation (5000 emails)
try:
    result = await spinner.spin_mailbox("INBOX")
except KeyboardInterrupt:
    # Progress saved automatically
    print("Interrupted. Resume with: spinner.spin_incremental(...)")

# Resume from checkpoint
checkpoint = spinner.load_checkpoint("inbox")
result = await spinner.spin_incremental("INBOX", checkpoint=checkpoint)
```

---

## Protocol & API

### SpinnerProtocol

All spinners implement this protocol:

```python
class SpinnerProtocol(Protocol):
    """Standard interface for all HoloLoom spinners."""

    def get_name(self) -> str:
        """Return unique spinner name (e.g., 'git', 'email', 'youtube')."""
        ...

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return available features based on installed dependencies."""
        ...

    def is_available(self) -> bool:
        """Check if spinner can be used (dependencies installed)."""
        ...

    def get_status(self) -> SpinnerStatus:
        """Get operational status (AVAILABLE, DEGRADED, UNAVAILABLE, ERROR)."""
        ...

    async def spin(self, source: Any, **kwargs) -> SpinResult:
        """Process input → MemoryShards."""
        ...

    # Optional methods
    async def spin_stream(self, source: Any, **kwargs) -> AsyncIterator[MemoryShard]:
        """Stream shards one at a time (for large sources)."""
        ...

    async def spin_incremental(
        self,
        source: Any,
        checkpoint: Optional[SpinnerCheckpoint] = None,
        **kwargs
    ) -> SpinResult:
        """Process only new data since checkpoint."""
        ...

    def score_importance(self, data: Any) -> ImportanceScore:
        """Custom importance scoring (spinner-specific heuristics)."""
        ...
```

### SpinResult

Every `spin()` call returns a `SpinResult`:

```python
@dataclass
class SpinResult:
    shards: List[MemoryShard]

    # Operation metadata
    success: bool = True
    error_message: Optional[str] = None

    # Performance metrics
    processing_time_ms: float = 0.0
    input_size_bytes: int = 0

    # Statistics
    shard_count: int = 0
    entity_count: int = 0
    motif_count: int = 0

    # Quality metrics
    avg_importance: float = 0.5
    avg_confidence: float = 1.0

    # Warnings
    warnings: List[str] = []
```

**Usage**:

```python
result = await spinner.spin(source)

if result.success:
    print(f"Created {result.shard_count} shards in {result.processing_time_ms:.1f}ms")
    print(f"Average importance: {result.avg_importance:.2f}")
else:
    print(f"Error: {result.error_message}")

if result.warnings:
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

---

## Importance Scoring

### 9-Signal System

All spinners use a standardized 9-signal importance scoring system:

| Signal | Weight | Description | Range |
|--------|--------|-------------|-------|
| **length_score** | 0.15 | Content length (longer = more substantive) | 0.0-1.0 |
| **technical_score** | 0.20 | Domain-specific terminology density | 0.0-1.0 |
| **structural_score** | 0.10 | Formatting quality (headers, lists, etc.) | 0.0-1.0 |
| **authority_score** | 0.20 | Source credibility (author, domain, citations) | 0.0-1.0 |
| **recency_score** | 0.10 | Time decay (newer = higher) | 0.0-1.0 |
| **engagement_score** | 0.15 | Reactions, shares, replies, stars | 0.0-1.0 |
| **reference_score** | 0.10 | Backlinks, citations, cross-references | 0.0-1.0 |
| **noise_penalty** | 1.0 | Spam, greetings, duplicates (negative!) | -1.0-0.0 |
| **custom_signals** | Variable | Spinner-specific signals | 0.0-1.0 |

**Final score**: `importance = Σ(signal × weight) + noise_penalty`, clamped to [0.0, 1.0]

### Example: Git Commit Importance

```python
# GitSpinner assigns importance based on commit type
BREAKING_CHANGE = ImportanceScore(
    score=0.95,
    signals=ImportanceSignals(
        technical_score=1.0,      # Code changes
        structural_score=0.9,     # Well-formatted commit
        authority_score=0.8,      # From core maintainer
        engagement_score=0.7,     # 15 reviewers
        custom_signals={'breaking': 1.0}
    ),
    reason="BREAKING CHANGE + high authority + good engagement"
)

# Chore commit (low importance)
CHORE = ImportanceScore(
    score=0.25,
    signals=ImportanceSignals(
        technical_score=0.3,
        structural_score=0.5,
        noise_penalty=-0.1        # Routine maintenance
    ),
    reason="chore commit + low technical impact"
)
```

### Custom Importance Weights

Override default weights for domain-specific needs:

```python
from hololoom.spinningWheel.importance import ImportanceScorer

# Prioritize recency for news articles
news_scorer = ImportanceScorer(weights={
    'recency': 0.40,      # High weight on recency
    'engagement': 0.30,   # Viral content
    'authority': 0.20,    # Source credibility
    'technical': 0.10     # Low weight on technical terms
})

score = news_scorer.score_text(
    text="Breaking news: ...",
    timestamp=datetime.now(),
    source_authority=0.9
)
```

---

## Advanced Features

### 1. Multimodal Fusion

`MultimodalSpinner` combines text + images into unified representation:

```python
from hololoom.spinningWheel.multimodal_spinner import MultiModalSpinner

spinner = MultiModalSpinner(
    enable_clip=True,           # CLIP embeddings for images
    enable_ocr=True,            # Extract text from images
    cross_modal_fusion=True     # Fuse modalities
)

# Input: Document with embedded images
result = await spinner.spin({
    'text': "Product diagram shows...",
    'images': ["diagram1.png", "diagram2.png"]
})

# Output: Shards with cross-modal relationships
for shard in result.shards:
    print(shard.metadata.get('modality'))  # 'text', 'image', or 'fused'
    print(shard.metadata.get('cross_modal_refs'))  # Links to related shards
```

### 2. Schema-Aware Extraction

`SchemaAwareReceiptSpinner` uses domain knowledge for structured extraction:

```python
from hololoom.spinningWheel.schema_aware_receipt_spinner import SchemaAwareReceiptSpinner
from hololoom.spinningWheel.schema_registry import ReceiptSchema

# Register custom schema
target_schema = ReceiptSchema(
    vendor_patterns=["Target", "TARGET"],
    total_keywords=["TOTAL", "AMOUNT DUE"],
    date_formats=["%m/%d/%Y", "%Y-%m-%d"],
    line_item_regex=r"(\d+\.\d{2})\s+(.+)"
)

spinner = SchemaAwareReceiptSpinner()
spinner.register_schema("target", target_schema)

# Extract with schema
result = await spinner.spin("/path/to/target_receipt.jpg", vendor="target")

# Structured output
shard = result.shards[0]
print(shard.metadata['vendor'])          # "Target"
print(shard.metadata['total_amount'])    # 42.99
print(shard.metadata['line_items'])      # [{"name": "Milk", "price": 3.99}, ...]
```

### 3. Batch Processing

Process multiple inputs in parallel:

```python
from hololoom.spinningWheel.pdf_spinner import PDFSpinner
import asyncio

spinner = PDFSpinner()

# Batch process 100 PDFs
pdf_files = [f"/pdfs/doc_{i}.pdf" for i in range(100)]

async def process_batch(files):
    tasks = [spinner.spin(f) for f in files]
    results = await asyncio.gather(*tasks)
    return results

results = await process_batch(pdf_files)

total_shards = sum(r.shard_count for r in results)
print(f"Processed {len(pdf_files)} PDFs → {total_shards} shards")
```

### 4. Custom Preprocessing

Add preprocessing hooks to spinners:

```python
from hololoom.spinningWheel.url_spinner import URLSpinner

class CustomURLSpinner(URLSpinner):
    async def _preprocess(self, html: str) -> str:
        """Custom preprocessing: Remove ads, tracking scripts."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')

        # Remove ads
        for ad in soup.find_all(class_='advertisement'):
            ad.decompose()

        # Remove tracking scripts
        for script in soup.find_all('script', src=lambda s: 'analytics' in s):
            script.decompose()

        return str(soup)

spinner = CustomURLSpinner()
result = await spinner.spin("https://example.com")
```

---

## Creating Custom Spinners

### Step 1: Inherit from BaseSpinner

```python
from hololoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    ImportanceSignals
)
from hololoom.documentation.types import MemoryShard
from typing import List, Any

class MyCustomSpinner(BaseSpinner):
    """Custom spinner for [your data source]."""

    def __init__(self, importance_threshold: float = 0.3):
        super().__init__(
            name="my_custom_spinner",
            importance_threshold=importance_threshold
        )

    def get_capabilities(self) -> SpinnerCapabilities:
        """Declare capabilities based on available dependencies."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,              # Set to True if you implement spin_stream()
            incremental=False,            # Set to True if you implement spin_incremental()
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            supported_formats=['.custom', '.xyz']
        )

    def is_available(self) -> bool:
        """Check if required dependencies are installed."""
        try:
            import my_required_library
            return True
        except ImportError:
            return False

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """Core spinning logic (must implement)."""
        # 1. Parse source
        parsed_data = await self._parse_source(source)

        # 2. Extract entities and motifs
        entities = self._extract_entities(parsed_data)
        motifs = self._extract_motifs(parsed_data)

        # 3. Score importance
        importance = self._score_importance(parsed_data)

        # 4. Create shards
        shards = []
        for chunk in parsed_data.chunks:
            shard = self._create_shard(
                id_suffix=chunk.id,
                text=chunk.text,
                episode=f"source_{source}",
                entities=entities,
                motifs=motifs,
                metadata={
                    'importance_score': importance,
                    'timestamp': chunk.timestamp,
                    'custom_field': chunk.custom_data
                }
            )
            shards.append(shard)

        return shards

    def _score_importance(self, data: Any) -> float:
        """Custom importance scoring."""
        signals = ImportanceSignals(
            length_score=min(len(data.text) / 1000, 1.0),
            technical_score=self._detect_technical_terms(data.text),
            structural_score=self._assess_structure(data),
            authority_score=0.5  # Default
        )
        return signals.compute_total()

    def _extract_entities(self, data: Any) -> List[str]:
        """Extract named entities."""
        # Your entity extraction logic
        return ["entity1", "entity2"]

    def _extract_motifs(self, data: Any) -> List[str]:
        """Extract topics/motifs."""
        # Your motif extraction logic
        return ["motif1", "motif2"]
```

### Step 2: Register Spinner

```python
# Add to hololoom/spinningWheel/__init__.py
from .my_custom_spinner import MyCustomSpinner

__all__ = [
    ...,
    'MyCustomSpinner'
]
```

### Step 3: Use Your Spinner

```python
from hololoom.spinningWheel.my_custom_spinner import MyCustomSpinner

spinner = MyCustomSpinner(importance_threshold=0.4)
result = await spinner.spin("/path/to/data")
```

---

## Production Deployment

### 1. Dependency Management

Use optional dependencies for spinners you need:

```bash
# Install only what you need
pip install HoloLoom[youtube]      # YouTube spinner
pip install HoloLoom[pdf]          # PDF spinner
pip install HoloLoom[git]          # Git spinner
pip install HoloLoom[email]        # Email spinner
pip install HoloLoom[ocr]          # OCR spinners (DeepSeek, Tesseract)
pip install HoloLoom[multimodal]   # Multimodal fusion

# Or install all spinners
pip install HoloLoom[all-spinners]
```

### 2. Error Handling

Spinners handle errors gracefully:

```python
result = await spinner.spin(source)

if not result.success:
    # Check error message
    print(f"Error: {result.error_message}")

    # Check spinner status
    status = spinner.get_status()
    if status == SpinnerStatus.UNAVAILABLE:
        print("Missing dependencies. Install with: pip install [dependencies]")
    elif status == SpinnerStatus.DEGRADED:
        print("Some features unavailable. Check warnings.")

    # Graceful fallback
    spinner_fallback = FallbackSpinner()
    result = await spinner_fallback.spin(source)
```

### 3. Performance Monitoring

Track spinner performance:

```python
results = []
for source in sources:
    result = await spinner.spin(source)
    results.append(result)

# Aggregate metrics
total_time = sum(r.processing_time_ms for r in results)
avg_time = total_time / len(results)
total_shards = sum(r.shard_count for r in results)

print(f"Processed {len(sources)} sources in {total_time:.1f}ms")
print(f"Average: {avg_time:.1f}ms per source")
print(f"Total shards: {total_shards}")
print(f"Throughput: {total_shards / (total_time / 1000):.1f} shards/sec")
```

### 4. Checkpointing Best Practices

```python
from pathlib import Path

# Use persistent checkpoint directory
checkpoint_dir = Path("/var/lib/hololoom/checkpoints")
spinner = GitSpinner(checkpoint_dir=checkpoint_dir)

# Save checkpoints periodically (every N shards)
shard_count = 0
async for shard in spinner.spin_stream("/huge/repo"):
    await memory.add_shard(shard)
    shard_count += 1

    if shard_count % 100 == 0:
        # Save checkpoint every 100 shards
        checkpoint = create_checkpoint("repo_id", shard_count)
        spinner.save_checkpoint(checkpoint)
        print(f"Checkpoint saved: {shard_count} shards processed")
```

### 5. Rate Limiting

For API-based spinners (YouTube, DeepSeek OCR):

```python
from hololoom.spinningWheel.youtube_spinner import YouTubeSpinner
import asyncio

class RateLimitedYouTubeSpinner(YouTubeSpinner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rate_limit_delay = 1.0  # 1 second between requests

    async def spin(self, source, **kwargs):
        # Add delay before processing
        await asyncio.sleep(self.rate_limit_delay)
        return await super().spin(source, **kwargs)

# Use rate-limited spinner
spinner = RateLimitedYouTubeSpinner()
```

---

## Testing

### Unit Tests

All spinners have comprehensive unit tests:

```bash
# Test all spinners
pytest hololoom/spinningWheel/tests/ -v

# Test specific spinner
pytest hololoom/spinningWheel/tests/test_youtube_spinner.py -v
pytest hololoom/spinningWheel/tests/test_pdf_spinner.py -v
pytest hololoom/spinningWheel/tests/test_git_spinner.py -v
```

### Integration Tests

Test spinners with real data:

```bash
# Integration tests (requires external dependencies)
pytest hololoom/tests/integration/test_spinners.py -v
```

---

## FAQ

**Q: Which spinner should I use for [data source]?**

A: See the Spinner Registry above. If your source isn't listed, use the closest match or create a custom spinner.

**Q: Can I use multiple spinners together?**

A: Yes! Spinners output standardized `MemoryShard` objects that can be mixed:

```python
youtube_shards = await youtube_spinner.spin("video_url")
pdf_shards = await pdf_spinner.spin("document.pdf")
git_shards = await git_spinner.spin("/repo")

all_shards = youtube_shards.shards + pdf_shards.shards + git_shards.shards
await memory.add_shards(all_shards)
```

**Q: How do I handle large data sources without running out of memory?**

A: Use streaming ingestion:

```python
async for shard in spinner.spin_stream(large_source):
    await memory.add_shard(shard)  # Process one shard at a time
```

**Q: Can I customize importance scoring?**

A: Yes, override `score_importance()` in your custom spinner or use custom weights:

```python
from hololoom.spinningWheel.importance import ImportanceScorer

scorer = ImportanceScorer(weights={'recency': 0.5, 'engagement': 0.3})
```

**Q: What if a spinner is missing dependencies?**

A: Spinners degrade gracefully:

```python
spinner = PDFSpinner(enable_ocr=True)
result = await spinner.spin("doc.pdf")

if result.warnings:
    print(result.warnings)  # ["OCR unavailable (pytesseract not installed)"]
# Text extraction still works!
```

---

## Roadmap

### Phase 6 (Q1 2026)
- **Calendar Spinners**: Google Calendar, Outlook, iCal
- **Task Management**: Trello, Jira, Asana, GitHub Issues
- **Cloud Storage**: Google Drive, Dropbox, OneDrive recursive crawling
- **Social Media**: Twitter/X, LinkedIn, Reddit threads

### Phase 7 (Q2 2026)
- **Audio Streaming**: Real-time transcription (microphone input)
- **Screen Recording**: OCR + motion detection for tutorials
- **Browser History**: Chrome, Firefox, Safari, Edge
- **Package Managers**: npm, pip, cargo, maven dependency graphs

### Phase 8 (Q3 2026)
- **Database Streaming**: Change Data Capture (CDC) for Postgres, MySQL, MongoDB
- **Log Aggregation**: Elasticsearch, Splunk, CloudWatch
- **Metrics**: Prometheus, Grafana, Datadog time series
- **Webhooks**: Generic webhook listener for SaaS integrations

---

## Contributing

We welcome contributions! To add a new spinner:

1. **Create spinner file**: `hololoom/spinningWheel/my_spinner.py`
2. **Inherit from `BaseSpinner`**: Implement `_spin_impl()`, `get_capabilities()`, `is_available()`
3. **Add tests**: `hololoom/spinningWheel/tests/test_my_spinner.py`
4. **Update this README**: Add your spinner to the registry
5. **Submit PR**: With examples and documentation

See [Creating Custom Spinners](#creating-custom-spinners) for detailed guide.

---

## License

HoloLoom SpinningWheel is part of the HoloLoom project.
See root LICENSE file for details.

---

## Support

- **Documentation**: This file + inline docstrings
- **Examples**: `demos/demo_spinners.py`
- **Issues**: https://github.com/blakechasteen/hello-world/issues
- **Discussions**: Tag with `spinningwheel` label

---

**Last Updated**: November 15, 2025
**Version**: 2.0.0
**Total Spinners**: 47+
**Total Code**: 17,925 lines
