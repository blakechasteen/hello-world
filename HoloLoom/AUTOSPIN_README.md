# AutoSpin: Automatic Text Spinning for HoloLoom

**Make HoloLoom as easy to use as ChatGPT!**

AutoSpin automatically converts your text into MemoryShards without manual spinner setup. Just provide text and start asking questions.

## Quick Start

```python
from HoloLoom.autospin import auto_loom_from_text
from HoloLoom.Documentation.types import Query

# Create orchestrator from text (auto-spins in the background)
orch = await auto_loom_from_text("Your knowledge base here...")

# Ask questions
response = await orch.process(Query(text="What is HoloLoom?"))
```

That's it! No manual shard creation, no complex setup.

## Features

### 1. From Text

Create an orchestrator directly from a string:

```python
from HoloLoom.autospin import AutoSpinOrchestrator
from HoloLoom.config import Config

knowledge = """
HoloLoom is a neural decision-making system...
"""

orchestrator = await AutoSpinOrchestrator.from_text(
    text=knowledge,
    config=Config.fast(),
    chunk_by='paragraph',
    chunk_size=500
)

response = await orchestrator.process(Query(text="Question?"))
```

### 2. From File

Load a text file automatically:

```python
orchestrator = await AutoSpinOrchestrator.from_file(
    file_path="documentation.md",
    config=Config.fused()
)
```

### 3. From Multiple Documents

Combine multiple documents:

```python
docs = [
    {'text': 'First document...', 'source': 'doc1.md'},
    {'text': 'Second document...', 'source': 'doc2.md'},
    {'text': 'Third document...', 'source': 'doc3.md'}
]

orchestrator = await AutoSpinOrchestrator.from_documents(docs)
```

### 4. Dynamic Addition

Add more content after creation:

```python
# Start with initial content
orch = await AutoSpinOrchestrator.from_text("Initial knowledge...")

# Add more later
await orch.add_text("Additional content...", source="update_1")
await orch.add_text("Even more...", source="update_2")

# Now queries can access all content
response = await orch.process(Query(text="Question?"))
```

## Before vs After

### BEFORE AutoSpin (Manual Process)

```python
# Step 1: Create spinner config
spinner_config = TextSpinnerConfig(
    chunk_by='paragraph',
    chunk_size=500,
    extract_entities=True
)

# Step 2: Create spinner
spinner = TextSpinner(spinner_config)

# Step 3: Manually spin text
shards = await spinner.spin({
    'text': knowledge_base,
    'source': 'kb.txt'
})

# Step 4: Create orchestrator with shards
config = Config.fused()
orchestrator = HoloLoomOrchestrator(cfg=config, shards=shards)

# Step 5: Process query
response = await orchestrator.process(Query(text="Question?"))
```

### AFTER AutoSpin (One Line!)

```python
# Everything in one line!
orch = await auto_loom_from_text(knowledge_base)
response = await orch.process(Query(text="Question?"))
```

## Configuration Options

### Chunking Modes

```python
# Paragraph chunking (default)
orch = await auto_loom_from_text(
    text=content,
    chunk_by='paragraph',
    chunk_size=500
)

# Sentence chunking
orch = await auto_loom_from_text(
    text=content,
    chunk_by='sentence',
    chunk_size=300
)

# No chunking (single shard)
orch = await auto_loom_from_text(
    text=content,
    chunk_by=None
)
```

### Execution Modes

```python
from HoloLoom.config import Config

# BARE: Fastest, minimal features
orch = await auto_loom_from_text(content, config=Config.bare())

# FAST: Balanced (default)
orch = await auto_loom_from_text(content, config=Config.fast())

# FUSED: Full features, slower
orch = await auto_loom_from_text(content, config=Config.fused())
```

### Advanced Config

```python
from HoloLoom.spinningWheel import TextSpinnerConfig

spinner_config = TextSpinnerConfig(
    chunk_by='paragraph',
    chunk_size=500,
    min_chunk_size=50,
    extract_entities=True,
    preserve_structure=True
)

orch = await AutoSpinOrchestrator.from_text(
    text=content,
    config=Config.fast(),
    spinner_config=spinner_config
)
```

## Utility Methods

### Get Shard Count

```python
count = orch.get_shard_count()
print(f"Memory contains {count} shards")
```

### Preview Shards

```python
preview = orch.get_shard_preview(limit=5)
for shard_info in preview:
    print(f"{shard_info['id']}: {shard_info['char_count']} chars")
    print(f"  Source: {shard_info['source']}")
    print(f"  Entities: {shard_info['entities']}")
```

## API Reference

### AutoSpinOrchestrator

#### Class Methods

- **`from_text(text, config, spinner_config, source)`** - Create from plain text
- **`from_file(file_path, config, spinner_config, encoding)`** - Create from file
- **`from_documents(documents, config, spinner_config)`** - Create from multiple docs

#### Instance Methods

- **`process(query)`** - Process a query (returns response dict)
- **`add_text(text, source)`** - Add content dynamically
- **`get_shard_count()`** - Get number of shards
- **`get_shard_preview(limit)`** - Preview shard metadata

### Convenience Functions

- **`auto_loom_from_text(text, config, chunk_by, chunk_size)`** - Quick orchestrator from text
- **`auto_loom_from_file(file_path, config)`** - Quick orchestrator from file

## Examples

See:
- `example_autospin.py` - Full examples showing all features
- `test_autospin_concept.py` - Concept demonstration

## Architecture

```
User Text
    ↓
AutoSpinOrchestrator.from_text()
    ↓
TextSpinner (automatic)
    ↓
MemoryShards
    ↓
HoloLoomOrchestrator
    ↓
Query Processing
    ↓
Response
```

The AutoSpinOrchestrator is a thin wrapper that:
1. Takes your text
2. Automatically creates a TextSpinner
3. Spins text into MemoryShards
4. Initializes HoloLoomOrchestrator with shards
5. Exposes the same `process(query)` interface

## Why AutoSpin?

**Before:** Users had to understand MemoryShards, SpinningWheel, TextSpinner, configs, etc.

**After:** Just provide text and ask questions!

This makes HoloLoom accessible to users who just want to:
- Load documents and query them
- Build knowledge bases quickly
- Prototype without understanding internal architecture

## Integration with SpinningWheel

AutoSpin uses the TextSpinner under the hood:
- Respects all TextSpinnerConfig options
- Supports all chunking modes (paragraph, sentence, character)
- Entity extraction works automatically
- Can enable optional enrichment (Ollama, Neo4j, mem0)

## Next Steps

Want more control? Use the SpinningWheel directly:
- `HoloLoom.spinningWheel.TextSpinner` - For text
- `HoloLoom.spinningWheel.AudioSpinner` - For audio transcripts
- `HoloLoom.spinningWheel.YouTubeSpinner` - For video captions

See `HoloLoom/spinningWheel/README.md` for details.