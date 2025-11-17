# 📚 HoloLoom Quick Reference

**Fast lookup guide for HoloLoom concepts and code patterns**

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Basic Patterns](#basic-patterns)
- [Configuration Modes](#configuration-modes)
- [Memory Shards](#memory-shards)
- [Queries](#queries)
- [Common Operations](#common-operations)
- [Troubleshooting](#troubleshooting)

---

## Core Concepts

### What is HoloLoom?

An AI system with persistent memory that:
- ✅ **Remembers everything** across sessions
- 🧠 **Gets smarter** with use
- 🔍 **Shows its work** with complete provenance
- ⚡ **Makes intelligent decisions** using Thompson Sampling

### Key Components

| Component | Purpose | Analogy |
|-----------|---------|---------|
| **MemoryShard** | One fact/piece of knowledge | Flashcard |
| **Knowledge Graph** | How memories connect | Web of ideas |
| **WeavingOrchestrator** | The brain that processes queries | Librarian |
| **Config** | How smart/fast to run | Performance mode |
| **Query** | Question you ask | Library search |
| **Spacetime** | Response with full context | Search results + provenance |

---

## Basic Patterns

### The 5-Step Pattern

Every HoloLoom interaction follows this:

```python
# 1. Import tools
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard

# 2. Create memory shards (facts to teach)
shards = [
    MemoryShard(text="Python is a programming language", source="basics"),
    MemoryShard(text="HoloLoom uses neural networks", source="tech")
]

# 3. Configure HoloLoom
config = Config.fast()  # Balanced speed/quality

# 4. Ask a question
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    spacetime = await shuttle.weave(Query(text="What is Python?"))

# 5. Get answer
print(spacetime.response)
```

---

## Configuration Modes

### Three Modes

| Mode | Speed | Use Case | Code |
|------|-------|----------|------|
| **BARE** | ~50ms | Simple lookups | `Config.bare()` |
| **FAST** | ~150ms | Most queries (recommended) | `Config.fast()` |
| **FUSED** | ~300ms | Complex reasoning | `Config.fused()` |

### When to Use Each

- **BARE**: "What's the capital of France?"
- **FAST**: "Explain how Thompson Sampling works"
- **FUSED**: "Compare three different RL algorithms"

### Customization

```python
config = Config.fast()
config.max_retrieval = 5  # Retrieve top 5 memories
config.temperature = 0.7  # Exploration vs exploitation
```

---

## Memory Shards

### Creating Shards

**Basic:**
```python
shard = MemoryShard(
    text="The fact you want to remember",
    source="where it came from"
)
```

**Multiple shards:**
```python
shards = [
    MemoryShard(text="Fact 1", source="book"),
    MemoryShard(text="Fact 2", source="article"),
    MemoryShard(text="Fact 3", source="experience")
]
```

**From file:**
```python
with open("knowledge.txt", "r") as f:
    lines = f.readlines()
    shards = [MemoryShard(text=line.strip(), source="file")
              for line in lines if line.strip()]
```

### Best Practices

✅ **DO:**
- Keep each shard focused on one concept
- Include source for context
- Use clear, complete sentences

❌ **DON'T:**
- Mix multiple unrelated facts in one shard
- Use fragments or incomplete sentences
- Leave source blank (makes debugging hard)

---

## Queries

### Basic Query

```python
query = Query(text="What is Thompson Sampling?")
```

### Query with Context

```python
query = Query(
    text="How does this relate to my project?",
    context={
        "project": "building a recommendation system",
        "goal": "improve user engagement"
    }
)
```

### Multiple Queries (Batch)

```python
queries = [
    Query(text="What is X?"),
    Query(text="How does Y work?"),
    Query(text="Compare X and Y")
]

for query in queries:
    result = await shuttle.weave(query)
    print(f"{query.text}: {result.response}\n")
```

---

## Common Operations

### Check Progress

```python
from core.progress import ProgressTracker

tracker = ProgressTracker()
print(f"Level: {tracker.progress.level}")
print(f"XP: {tracker.progress.total_xp}")
print(f"Lessons: {len(tracker.progress.completed_lessons)}")
```

### Load a Lesson

```python
from core.lesson import LessonManager
from pathlib import Path

content_dir = Path("edwin_tutor/content")
manager = LessonManager(content_dir)

lesson = manager.get_lesson("beginner_01")
print(lesson.title)
print(lesson.content)
```

### Mark Lesson Complete

```python
result = tracker.mark_lesson_complete(
    lesson_id="beginner_01",
    xp_earned=50,
    score=100
)

if result['leveled_up']:
    print(f"LEVEL UP! Now level {result['new_level']}")
```

### Use Hints

```python
tracker.use_hint()  # Records that you used a hint
```

### Complete Challenge

```python
tracker.mark_challenge_complete(
    challenge_id="beginner_02_challenge_1",
    points=20
)
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'HoloLoom'`

**Solution:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "HoloLoom"))
```

### Async Errors

**Problem:** `RuntimeError: asyncio.run() cannot be called from a running event loop`

**Solution:** Use `await` instead of `asyncio.run()` when in notebook/async context:
```python
# In Jupyter notebooks
result = await shuttle.weave(query)

# In scripts
import asyncio
asyncio.run(main())
```

### Progress Not Saving

**Problem:** Progress resets after restart

**Solution:** Check that `.edwin_progress.json` exists and has write permissions:
```bash
ls -la .edwin_progress.json
chmod 644 .edwin_progress.json
```

### Memory Issues

**Problem:** Too many shards, system slow

**Solution:** Limit number of shards or use batching:
```python
# Limit to most recent/relevant
shards = recent_shards[:100]

# Or use BARE mode for speed
config = Config.bare()
```

---

## Cheat Sheet

### Imports
```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard
from core.lesson import LessonManager
from core.progress import ProgressTracker
```

### Quick Setup
```python
shards = [MemoryShard(text="...", source="...")]
config = Config.fast()
tracker = ProgressTracker()
```

### Query
```python
async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
    result = await shuttle.weave(Query(text="..."))
    print(result.response)
```

### Progress
```python
tracker.start_lesson("lesson_id")
tracker.mark_lesson_complete("lesson_id", xp=50, score=100)
tracker.use_hint()
```

---

## Next Steps

- **Practice:** Try the [terminal UI](../terminal/edwin.py)
- **Visual:** Use the [web interface](../web/server.py)
- **Hands-on:** Code along with [Jupyter notebooks](../notebooks/)
- **Deep dive:** Read [detailed guides](02_Advanced_Topics.md)

---

**Last Updated:** November 2025
**Version:** 1.0
**Part of:** EdWIN Tutor Multi-Modal Learning Platform
