

# SpinningWheel Protocol Guide

**Version**: 1.0
**Last Updated**: November 2025

Complete guide to building spinners using the standardized `SpinnerProtocol`.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Protocol Interface](#protocol-interface)
4. [Building a Spinner](#building-a-spinner)
5. [Importance Scoring](#importance-scoring)
6. [Checkpointing & Resumable Operations](#checkpointing--resumable-operations)
7. [Streaming Support](#streaming-support)
8. [Batch Processing](#batch-processing)
9. [Error Handling](#error-handling)
10. [Testing Your Spinner](#testing-your-spinner)
11. [Examples](#examples)

---

## Overview

### What is a Spinner?

A **spinner** transforms raw data into **MemoryShards** that HoloLoom can query. Examples:
- **GitSpinner**: Git commits → MemoryShards
- **EmailSpinner**: Email messages → MemoryShards
- **ChatHistorySpinner**: Conversations → MemoryShards

### Why Use SpinnerProtocol?

**SpinnerProtocol** ensures all spinners have:
- ✅ Consistent API (`spin()`, `get_capabilities()`, `is_available()`)
- ✅ Graceful degradation (missing dependencies handled automatically)
- ✅ Standardized importance scoring
- ✅ Checkpointing for resumable operations
- ✅ Streaming support for large data sources
- ✅ Built-in error handling

---

## Quick Start

### Minimal Spinner (3 Steps)

```python
from hololoom.spinningWheel.protocol import BaseSpinner, SpinnerCapabilities
from hololoom.documentation.types import MemoryShard

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
        # Check if dependencies are installed
        try:
            import some_optional_library
            return True
        except ImportError:
            return False

    # Step 3: Implement core logic
    async def _spin_impl(self, source, **kwargs) -> List[MemoryShard]:
        # Convert source → MemoryShards
        text = str(source)

        shard = self._create_shard(
            id_suffix="001",
            text=text,
            episode="my_episode",
            entities=["Entity1"],
            motifs=["topic1"],
            metadata={'confidence': 0.9}
        )

        return [shard]
```

**Usage**:
```python
spinner = MySpinner()
result = await spinner.spin("My input data")

print(f"Created {result.shard_count} shards")
print(f"Success: {result.success}")
```

---

## Protocol Interface

### Required Methods

All spinners must implement:

#### 1. `get_name() -> str`
Return unique spinner name (e.g., 'git', 'email', 'chat')

```python
def get_name(self) -> str:
    return "git"
```

#### 2. `get_capabilities() -> SpinnerCapabilities`
Describe spinner features

```python
def get_capabilities(self) -> SpinnerCapabilities:
    return SpinnerCapabilities(
        basic_processing=True,       # Can process inputs
        streaming=True,               # Supports streaming
        incremental=True,             # Supports incremental updates
        importance_scoring=True,      # Can score importance
        entity_extraction=True,       # Extracts entities
        motif_extraction=True,        # Extracts motifs/topics
        embeddings=False,             # Generates embeddings
        max_input_size=10_000_000,    # Max 10MB input
        supported_formats=['txt', 'md', 'log']
    )
```

#### 3. `is_available() -> bool`
Check if dependencies are installed

```python
def is_available(self) -> bool:
    try:
        import required_library
        return True
    except ImportError:
        print("Warning: required_library not installed")
        return False
```

#### 4. `spin(source, **kwargs) -> SpinResult`
Main processing method (provided by `BaseSpinner`, delegates to `_spin_impl()`)

```python
# Implemented by BaseSpinner automatically
# Subclasses implement _spin_impl() instead
```

### Optional Methods

#### 5. `spin_stream(source, **kwargs) -> AsyncIterator[MemoryShard]`
Streaming ingestion (for large sources)

```python
async def spin_stream(self, source, **kwargs):
    for item in large_data_source:
        shard = self._process_item(item)
        yield shard
```

#### 6. `spin_incremental(source, checkpoint, **kwargs) -> SpinResult`
Incremental updates (only process new data)

```python
async def spin_incremental(self, source, checkpoint, **kwargs):
    last_id = checkpoint.last_processed_id if checkpoint else None
    new_items = get_items_after(source, last_id)

    shards = [self._process_item(item) for item in new_items]

    # Update checkpoint
    new_checkpoint = SpinnerCheckpoint(
        spinner_name=self.get_name(),
        source_id=create_source_id(source),
        last_processed_id=new_items[-1].id if new_items else last_id,
        items_processed=len(new_items)
    )
    self.save_checkpoint(new_checkpoint)

    return SpinResult(shards=shards, success=True)
```

#### 7. `score_importance(data) -> ImportanceScore`
Custom importance scoring

```python
def score_importance(self, data) -> ImportanceScore:
    signals = ImportanceSignals(
        length_score=len(data) / 1000,
        technical_score=count_technical_terms(data) / 10,
    )

    return ImportanceScore(
        score=signals.compute_total(),
        signals=signals,
        reason=signals.explain()
    )
```

---

## Building a Spinner

### Step-by-Step Guide

#### Step 1: Choose BaseSpinner or Protocol

**Option A: Inherit from `BaseSpinner`** (Recommended)
- Automatic error handling
- Built-in checkpointing
- Importance filtering

```python
from hololoom.spinningWheel.protocol import BaseSpinner

class MySpinner(BaseSpinner):
    def __init__(self):
        super().__init__(
            name="my_spinner",
            importance_threshold=0.3,  # Filter low importance
            checkpoint_dir="/tmp/checkpoints"
        )
```

**Option B: Implement `SpinnerProtocol` directly**
- Full control over all methods
- More boilerplate code

```python
from hololoom.spinningWheel.protocol import SpinnerProtocol

class MySpinner:  # Implement protocol manually
    def get_name(self) -> str: ...
    def get_capabilities(self) -> SpinnerCapabilities: ...
    # ... etc
```

#### Step 2: Implement Core Logic

**Method: `_spin_impl(source, **kwargs)`**

This is where you convert raw data → MemoryShards.

```python
async def _spin_impl(self, source, **kwargs) -> List[MemoryShard]:
    """
    Convert source data to MemoryShards.

    Args:
        source: Input data (varies by spinner type)
        **kwargs: Spinner-specific options

    Returns:
        List of MemoryShards
    """
    shards = []

    # Process source
    for item in self._parse_source(source):
        # Extract entities
        entities = self._extract_entities(item)

        # Extract motifs
        motifs = self._extract_motifs(item)

        # Create shard
        shard = self._create_shard(
            id_suffix=item.id,
            text=item.content,
            episode=item.episode,
            entities=entities,
            motifs=motifs,
            metadata={
                'confidence': item.confidence,
                'timestamp': item.timestamp,
                'custom_field': item.custom_data
            }
        )

        shards.append(shard)

    return shards
```

#### Step 3: Add Entity Extraction

**Simple Approach**: Regex/patterns

```python
def _extract_entities(self, text: str) -> List[str]:
    """Extract entities using simple patterns."""
    entities = []

    # Capitalized words (proper nouns)
    entities.extend(re.findall(r'\b[A-Z][a-z]+\b', text))

    # Emails
    entities.extend(re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text))

    # Deduplicate and limit
    return list(set(entities))[:20]
```

**Advanced Approach**: NLP (spaCy)

```python
def _extract_entities(self, text: str) -> List[str]:
    """Extract entities using spaCy NER."""
    if not self.has_spacy:
        return self._simple_entity_extraction(text)

    doc = self.nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities[:20]
```

#### Step 4: Add Motif Extraction

```python
def _extract_motifs(self, text: str) -> List[str]:
    """Extract topics/themes from text."""
    motifs = []

    # Technical terms
    technical_terms = ['git', 'commit', 'merge', 'branch', 'pr']
    for term in technical_terms:
        if term in text.lower():
            motifs.append(term)

    # Keywords (TF-IDF or simple frequency)
    words = text.lower().split()
    common_words = Counter(words).most_common(5)
    motifs.extend([word for word, count in common_words if count > 2])

    return list(set(motifs))[:10]
```

---

## Importance Scoring

### Using ImportanceScorer

**Built-in Scorer**:

```python
from hololoom.spinningWheel.importance import ImportanceScorer

class MySpinner(BaseSpinner):
    def __init__(self):
        super().__init__(name="my_spinner")
        self.importance_scorer = ImportanceScorer()

    def score_importance(self, data) -> ImportanceScore:
        return self.importance_scorer.score(
            text=data['text'],
            source=data['source'],
            timestamp=data.get('timestamp'),
            engagement=data.get('engagement', {})
        )
```

### Custom Signals

**Add domain-specific signals**:

```python
scorer = ImportanceScorer()

# Add technical terms for your domain
scorer.add_technical_terms(['kubernetes', 'docker', 'microservices'])

# Add authority patterns
scorer.add_authority('cto@', score=0.95)
scorer.add_authority('intern@', score=0.4)

# Custom scorer function
def commit_importance(text, metadata):
    """Score based on commit properties."""
    if 'BREAKING:' in text:
        return 1.0
    elif 'fix:' in text:
        return 0.8
    elif 'chore:' in text:
        return 0.3
    return 0.5

scorer.custom_scorers['commit_type'] = commit_importance
```

### Preset Configurations

```python
from hololoom.spinningWheel.importance import (
    create_chat_scorer,
    create_git_scorer,
    create_email_scorer,
    create_document_scorer
)

# Use preset for your domain
self.scorer = create_git_scorer()
```

---

## Checkpointing & Resumable Operations

### Why Checkpointing?

For large data sources (Git repos with 10K commits, email archives with 100K messages), checkpointing enables:
- **Resumable operations**: Continue from where you left off
- **Incremental updates**: Only process new data
- **Progress tracking**: Monitor long-running operations

### Using CheckpointManager

```python
from hololoom.spinningWheel.utils import CheckpointManager, create_source_id
from hololoom.spinningWheel.protocol import SpinnerCheckpoint

class MySpinner(BaseSpinner):
    def __init__(self):
        super().__init__(name="my_spinner", checkpoint_dir="/tmp/checkpoints")
        # CheckpointManager automatically created in BaseSpinner

    async def spin_incremental(self, source, checkpoint=None, **kwargs):
        """Process only new items since checkpoint."""
        source_id = create_source_id(source)

        # Load checkpoint
        if checkpoint is None:
            checkpoint = self.load_checkpoint(source_id)

        # Get last processed ID
        last_id = checkpoint.last_processed_id if checkpoint else None

        # Fetch new items
        new_items = self._get_items_after(source, last_id)

        shards = []
        for i, item in enumerate(new_items):
            shard = self._process_item(item)
            shards.append(shard)

            # Update checkpoint periodically (every 100 items)
            if i % 100 == 0:
                checkpoint = SpinnerCheckpoint(
                    spinner_name=self.get_name(),
                    source_id=source_id,
                    last_processed_id=item.id,
                    items_processed=i + 1,
                    items_total=len(new_items)
                )
                self.save_checkpoint(checkpoint)

        # Final checkpoint
        if new_items:
            checkpoint = SpinnerCheckpoint(
                spinner_name=self.get_name(),
                source_id=source_id,
                last_processed_id=new_items[-1].id,
                items_processed=len(new_items),
                items_total=len(new_items)
            )
            self.save_checkpoint(checkpoint)

        return SpinResult(shards=shards, success=True)
```

---

## Streaming Support

### Why Streaming?

For very large sources, streaming avoids loading everything into memory:
- **Memory efficiency**: Process items one at a time
- **Early availability**: Start ingesting before full processing completes
- **Cancellable**: Can stop mid-stream

### Implementing `spin_stream()`

```python
async def spin_stream(self, source, **kwargs) -> AsyncIterator[MemoryShard]:
    """Stream shards as they're processed."""
    items = self._get_items_generator(source)  # Returns generator/iterator

    async for item in items:
        # Process item
        shard = self._process_item(item)

        # Yield immediately (don't accumulate in memory)
        yield shard
```

### Using Streaming with Buffer

```python
from hololoom.spinningWheel.utils import stream_with_buffer

# Callback to ingest batches
async def ingest_batch(shards: List[MemoryShard]):
    await memory.add_shards(shards)

# Process with automatic batching
async for shard in stream_with_buffer(
    spinner.spin_stream(source),
    batch_size=50,
    callback=ingest_batch
):
    print(f"Processed: {shard.id}")
```

---

## Batch Processing

### Process Multiple Sources in Parallel

```python
from hololoom.spinningWheel.utils import BatchProcessor

class MySpinner(BaseSpinner):
    async def process_multiple(self, sources: List[str]):
        """Process multiple sources concurrently."""
        processor = BatchProcessor(max_concurrent=5)

        async def process_one(source):
            result = await self.spin(source)
            return result

        results = await processor.process(sources, process_one)

        return results
```

### With Progress Tracking

```python
from hololoom.spinningWheel.utils import process_with_progress

# Process with visual progress
results = await process_with_progress(
    items=file_paths,
    processor=lambda path: spinner.spin(path),
    description="Spinning files",
    max_concurrent=10
)
```

---

## Error Handling

### Built-in Error Handling

`BaseSpinner.spin()` automatically catches errors:

```python
result = await spinner.spin(source)

if not result.success:
    print(f"Error: {result.error_message}")
    # result.shards will be empty list
```

### Custom Error Handler

```python
from hololoom.spinningWheel.utils import ErrorHandler

class MySpinner(BaseSpinner):
    def __init__(self):
        super().__init__(name="my_spinner")
        self.error_handler = ErrorHandler(
            log_errors=True,
            raise_on_critical=False
        )

    async def _spin_impl(self, source, **kwargs):
        shards = []

        for item in source:
            try:
                shard = self._process_item(item)
                shards.append(shard)
            except ValueError as e:
                # Non-critical error
                error_shard = self.error_handler.handle(
                    e,
                    context=f"Processing item {item.id}",
                    critical=False
                )
                if error_shard:
                    shards.append(error_shard)
            except Exception as e:
                # Critical error
                self.error_handler.handle(
                    e,
                    context=f"Processing item {item.id}",
                    critical=True
                )

        return shards
```

---

## Testing Your Spinner

### Unit Test Template

```python
import pytest
from hololoom.spinningWheel.protocol import SpinnerStatus

class TestMySpinner:
    """Test suite for MySpinner."""

    def test_initialization(self):
        """Test spinner initialization."""
        spinner = MySpinner()
        assert spinner.get_name() == "my_spinner"
        assert spinner.importance_threshold == 0.3

    def test_capabilities(self):
        """Test capabilities reporting."""
        spinner = MySpinner()
        capabilities = spinner.get_capabilities()

        assert capabilities.basic_processing is True
        assert 'txt' in capabilities.supported_formats

    def test_availability(self):
        """Test dependency checking."""
        spinner = MySpinner()

        if spinner.is_available():
            assert spinner.get_status() == SpinnerStatus.AVAILABLE
        else:
            assert spinner.get_status() == SpinnerStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_spin(self):
        """Test basic spin operation."""
        spinner = MySpinner()
        result = await spinner.spin("Test input")

        assert result.success is True
        assert result.shard_count > 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_importance_filtering(self):
        """Test importance threshold."""
        spinner = MySpinner(importance_threshold=0.9)
        result = await spinner.spin("Low importance content")

        # Should filter low-importance shards
        assert result.shard_count == 0 or result.avg_importance >= 0.9
```

### Integration Test

```python
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete spinner workflow."""
    spinner = MySpinner()

    # Process input
    result = await spinner.spin("/path/to/data")

    assert result.success
    assert result.shard_count > 0

    # Verify shard structure
    shard = result.shards[0]
    assert len(shard.text) > 0
    assert len(shard.entities) > 0
    assert len(shard.motifs) > 0
    assert 'confidence' in shard.metadata
```

---

## Examples

### Example 1: Simple File Spinner

```python
from hololoom.spinningWheel.protocol import BaseSpinner, SpinnerCapabilities
from hololoom.documentation.types import MemoryShard
from pathlib import Path

class FileSpinner(BaseSpinner):
    """Spin text files into memory."""

    def __init__(self):
        super().__init__(name="file")

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            supported_formats=['txt', 'md', 'log']
        )

    def is_available(self) -> bool:
        return True  # No dependencies

    async def _spin_impl(self, source, **kwargs):
        """Read file and create shard."""
        path = Path(source)

        # Read file
        with open(path, 'r') as f:
            text = f.read()

        # Extract entities (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Create shard
        shard = self._create_shard(
            id_suffix=path.stem,
            text=text,
            episode=str(path.parent),
            entities=list(set(entities))[:20],
            motifs=[path.suffix[1:]],  # File extension as motif
            metadata={
                'file_path': str(path),
                'file_size': len(text),
                'confidence': 1.0
            }
        )

        return [shard]


# Usage
spinner = FileSpinner()
result = await spinner.spin("/path/to/document.txt")
print(f"Created {result.shard_count} shard(s)")
```

### Example 2: Git Commit Spinner (Simplified)

```python
import subprocess
from datetime import datetime

class GitSpinner(BaseSpinner):
    """Spin Git commits into memory."""

    def __init__(self):
        super().__init__(name="git", importance_threshold=0.4)

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            incremental=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True
        )

    def is_available(self) -> bool:
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except Exception:
            return False

    async def _spin_impl(self, source, **kwargs):
        """Process Git repository."""
        repo_path = Path(source)

        # Get commits
        result = subprocess.run(
            ['git', '-C', str(repo_path), 'log', '--pretty=format:%H|%an|%ae|%at|%s|%b'],
            capture_output=True,
            text=True
        )

        commits = result.stdout.strip().split('\n')
        shards = []

        for commit in commits:
            parts = commit.split('|')
            if len(parts) < 6:
                continue

            commit_hash, author_name, author_email, timestamp, subject, body = parts

            # Create commit text
            text = f"{subject}\n\n{body}"

            # Extract entities (file names, authors)
            entities = [author_name, author_email]

            # Extract motifs (commit type)
            motifs = self._extract_commit_type(subject)

            # Score importance
            importance = self._score_commit_importance(subject, body)

            # Create shard
            shard = self._create_shard(
                id_suffix=commit_hash[:8],
                text=text,
                episode=f"repo_{repo_path.name}",
                entities=entities,
                motifs=motifs,
                metadata={
                    'commit_hash': commit_hash,
                    'author_name': author_name,
                    'author_email': author_email,
                    'timestamp': float(timestamp),
                    'importance_score': importance.score,
                    'confidence': 1.0
                }
            )

            shards.append(shard)

        return shards

    def _extract_commit_type(self, subject: str) -> List[str]:
        """Extract commit type from subject."""
        motifs = []

        # Conventional commits
        if subject.startswith('fix:'):
            motifs.append('bug_fix')
        elif subject.startswith('feat:'):
            motifs.append('feature')
        elif subject.startswith('refactor:'):
            motifs.append('refactoring')
        elif subject.startswith('docs:'):
            motifs.append('documentation')
        elif subject.startswith('test:'):
            motifs.append('testing')

        return motifs

    def _score_commit_importance(self, subject: str, body: str) -> ImportanceScore:
        """Score commit importance."""
        signals = ImportanceSignals()

        # Length signal
        total_len = len(subject) + len(body)
        signals.length_score = min(1.0, total_len / 500)

        # Breaking change = high importance
        if 'BREAKING:' in subject or 'BREAKING CHANGE' in body:
            signals.custom_signals['breaking_change'] = 1.0

        # Merge commits = low importance
        if subject.startswith('Merge'):
            signals.noise_penalty = -0.3

        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )


# Usage
spinner = GitSpinner()
result = await spinner.spin("/path/to/repo")
print(f"Processed {result.shard_count} commits")
print(f"Average importance: {result.avg_importance:.2f}")
```

### Example 3: Email Spinner (Simplified)

```python
import email
from email.parser import BytesParser

class EmailSpinner(BaseSpinner):
    """Spin email messages into memory."""

    def __init__(self):
        super().__init__(name="email", importance_threshold=0.3)

    def get_capabilities(self) -> SpinnerCapabilities:
        return SpinnerCapabilities(
            basic_processing=True,
            incremental=True,
            importance_scoring=True,
            entity_extraction=True
        )

    def is_available(self) -> bool:
        return True  # Standard library

    async def _spin_impl(self, source, **kwargs):
        """Process mbox file or IMAP connection."""
        mbox_path = Path(source)

        # Parse mbox
        import mailbox
        mbox = mailbox.mbox(str(mbox_path))

        shards = []
        for message in mbox:
            shard = self._process_message(message)
            if shard:
                shards.append(shard)

        return shards

    def _process_message(self, message) -> Optional[MemoryShard]:
        """Process single email message."""
        # Extract fields
        subject = message.get('Subject', '')
        from_addr = message.get('From', '')
        to_addr = message.get('To', '')
        date_str = message.get('Date', '')
        body = self._get_body(message)

        # Create text
        text = f"Subject: {subject}\n\nFrom: {from_addr}\nTo: {to_addr}\n\n{body}"

        # Extract entities (email addresses, names)
        entities = self._extract_email_entities(from_addr, to_addr, body)

        # Motifs (based on subject/body keywords)
        motifs = self._extract_motifs(subject + ' ' + body)

        # Score importance
        importance = self._score_email_importance(subject, body, from_addr)

        # Create shard
        shard = self._create_shard(
            id_suffix=hashlib.md5(text.encode()).hexdigest()[:8],
            text=text,
            episode="email_archive",
            entities=entities,
            motifs=motifs,
            metadata={
                'from': from_addr,
                'to': to_addr,
                'subject': subject,
                'date': date_str,
                'importance_score': importance.score,
                'confidence': 0.9
            }
        )

        return shard

    def _get_body(self, message) -> str:
        """Extract email body."""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            return message.get_payload(decode=True).decode('utf-8', errors='ignore')
        return ""


# Usage
spinner = EmailSpinner()
result = await spinner.spin("/path/to/emails.mbox")
print(f"Processed {result.shard_count} emails")
```

---

## Best Practices

### 1. Graceful Degradation

Always check for optional dependencies:

```python
def __init__(self):
    super().__init__(name="my_spinner")

    # Try to load optional library
    try:
        import optional_library
        self.has_optional = True
        self.optional = optional_library
    except ImportError:
        self.has_optional = False
        self.optional = None
        print("Warning: optional_library not found, using fallback")

def _process_with_optional(self, data):
    if self.has_optional:
        return self.optional.process(data)
    else:
        return self._fallback_process(data)
```

### 2. Importance Threshold Tuning

Set appropriate thresholds for your domain:

```python
# Low threshold (0.1-0.3): Keep most content
# - Email (want to keep most messages)
# - Chat history (conversations vary in importance)

# Medium threshold (0.3-0.5): Balanced
# - Git commits (filter trivial merges)
# - Documents (keep substantive content)

# High threshold (0.5-0.7): Only important content
# - Social media (filter noise)
# - Logs (keep errors/warnings only)
```

### 3. Memory Efficiency

For large sources, use streaming:

```python
# BAD: Loads everything into memory
shards = await spinner.spin(huge_source)

# GOOD: Stream processing
async for shard in spinner.spin_stream(huge_source):
    await memory.add_shard(shard)
```

### 4. Error Recovery

Checkpoint frequently for long operations:

```python
for i, item in enumerate(large_dataset):
    shard = process_item(item)

    # Checkpoint every 100 items
    if i % 100 == 0:
        self.save_checkpoint(checkpoint)
```

### 5. Testing

Test all edge cases:

```python
# Empty input
result = await spinner.spin([])
assert result.shard_count == 0

# Invalid input
result = await spinner.spin(None)
assert result.success is False

# Large input
result = await spinner.spin(huge_dataset)
assert result.success is True
```

---

## Summary

**You now have**:
- ✅ `SpinnerProtocol` - Standardized interface
- ✅ `BaseSpinner` - Common implementation
- ✅ `ImportanceScorer` - Reusable importance framework
- ✅ Checkpointing - Resumable operations
- ✅ Streaming - Memory-efficient processing
- ✅ Utilities - Batch processing, deduplication, error handling

**Next steps**:
1. Choose a data source (Git, email, Slack, etc.)
2. Create spinner class inheriting from `BaseSpinner`
3. Implement `_spin_impl()` to convert data → MemoryShards
4. Add importance scoring
5. Test thoroughly
6. Deploy!

**See also**:
- [protocol.py](protocol.py) - Full protocol definition
- [importance.py](importance.py) - Importance scoring framework
- [utils.py](utils.py) - Utility functions
- [test_spinner_protocol.py](../tests/unit/test_spinner_protocol.py) - Test examples
- [PIPELINE.md](PIPELINE.md) - Data flow architecture

---

**Questions?** Check the tests for working examples, or refer to existing spinners like `ChatHistorySpinner`.
