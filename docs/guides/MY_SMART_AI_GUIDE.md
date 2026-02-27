# Your Smart AI - Complete Usage Guide

## 🚀 Quick Start (5 minutes)

### 1. Test the System
```bash
PYTHONPATH=. python quickstart_test.py
```

### 2. Try the Demo
```bash
PYTHONPATH=. python my_smart_ai.py
```

### 3. Load Your Creative Writing
```bash
PYTHONPATH=. python ingest_my_writing.py
```

---

## 📁 What Data Can I Use?

### Text Files
```python
# Any text content
await rag.ingest("Your text here...")

# From files
content = Path("my_notes.txt").read_text()
await rag.ingest(content)
```

### Multiple Files at Once
```python
files = [
    "notes.txt",
    "research.md",
    "diary.txt",
]

for file in files:
    content = Path(file).read_text()
    await rag.ingest(content)
```

### Folders (Recursive)
```python
from pathlib import Path

folder = Path("my_documents")
for file in folder.rglob("*.txt"):
    content = file.read_text()
    await rag.ingest(content)
```

---

## 🎮 Reasoning Modes (Pick Your Power Level)

### DIRECT - Fast & Simple (~150ms)
```python
result = await rag.query("What is X?", mode="direct")
```
**Use for**: Quick factual questions

### VERIFY - Quality Answers (~600ms)
```python
result = await rag.query("Explain X", mode="verify")
```
**Use for**: Claims you want verified ✓ **Recommended default**

### RESEARCH - Deep Exploration (~900ms)
```python
result = await rag.query("What are the tradeoffs of X?", mode="research")
```
**Use for**: Open-ended research, analysis, creative questions

### PLAN_EXECUTE - Multi-Step Tasks (~750ms)
```python
result = await rag.query("How do I build X?", mode="plan_execute")
```
**Use for**: Complex tasks needing steps

---

## 💡 Real-World Use Cases

### 1. Personal Knowledge Base
```python
# Ingest all your notes
await rag.ingest(Path("notes.txt").read_text())

# Query anytime
result = await rag.query("What did I learn about Python?")
```

### 2. Research Assistant
```python
# Ingest research papers
papers = ["paper1.txt", "paper2.txt", "paper3.txt"]
for paper in papers:
    await rag.ingest(Path(paper).read_text())

# Ask research questions
result = await rag.query(
    "What are the main findings across these papers?",
    mode="research"
)
```

### 3. Creative Writing Assistant (Your Use Case!)
```python
# Ingest your chapters
chapters = Path("SpeakForMe").glob("chapter*")
for chapter in chapters:
    await rag.ingest(chapter.read_text())

# Analyze your writing
queries = [
    "What are the main character arcs?",
    "What themes appear most often?",
    "What happens in chapter 5?",
    "How does my writing style evolve?",
]
```

### 4. Code Documentation
```python
# Ingest your codebase
code_files = Path("src").rglob("*.py")
for file in code_files:
    await rag.ingest(file.read_text())

# Ask about your code
result = await rag.query("How does authentication work?")
```

### 5. Meeting Notes & Tasks
```python
# Ingest meeting notes
await rag.ingest(Path("meetings.txt").read_text())

# Extract tasks
result = await rag.query(
    "What action items do I have?",
    mode="plan_execute"
)
```

---

## 🎛️ Configuration Options

### Speed vs Quality Tradeoff

```python
from hololoom.config import Config

# BARE - Fastest (~50ms), good for simple queries
config = Config.bare()

# FAST - Balanced (~150ms) ✓ **Recommended**
config = Config.fast()

# FUSED - Highest quality (~300ms), best for complex queries
config = Config.fused()

rag = SimpleRAG(config=config)
```

### Caching (100x Speedup!)
```python
# Enable caching (default: ON)
rag = SimpleRAG(enable_caching=True)

# First query: ~150ms
result1 = await rag.query("What is Thompson Sampling?")

# Repeat query: <1ms (100x faster!)
result2 = await rag.query("What is Thompson Sampling?")
```

---

## 🧠 Advanced: Full HoloLoom API

For maximum control, use the full HoloLoom API:

```python
from hololoom import hololoom

async with HoloLoom() as loom:
    # Experience (form memories)
    mem = await loom.experience("Your content")

    # Recall (retrieve)
    memories = await loom.recall("Your query")

    # Reflect (learn from feedback)
    await loom.reflect(memories, feedback={"helpful": True})

    # Get awareness metrics
    metrics = loom.get_metrics()
    print(f"Active memories: {metrics['activation']['active_nodes']}")
    print(f"Coherence: {metrics['coherence']['global_coherence']:.2f}")
```

---

## 🔧 Troubleshooting

### "Module not found"
```bash
# Make sure PYTHONPATH is set
PYTHONPATH=. python my_script.py
```

### "Slow first query"
```bash
# First run downloads embeddings (~137MB)
# Cached at: ~/.cache/huggingface/
# Subsequent runs are fast
```

### "Low confidence answers"
```python
# Solution 1: Use VERIFY or RESEARCH mode
result = await rag.query(query, mode="research")

# Solution 2: Ingest more context
await rag.ingest("More relevant content...")

# Solution 3: Use FUSED config for better quality
config = Config.fused()
rag = SimpleRAG(config=config)
```

### "Out of memory"
```python
# Use smaller config
config = Config.bare()  # Minimal memory usage
rag = SimpleRAG(config=config)
```

---

## 📊 What You Get

Every query returns:

```python
result = await rag.query("Your question")

result.response         # The answer (string)
result.confidence       # 0.0 - 1.0 (how confident)
result.sources          # Retrieved source texts
result.reasoning_mode   # Which mode was used
result.metadata         # Extra info (timing, etc.)
```

---

## 🎯 Next Steps

### Level 1: Basic (You are here!)
- ✓ Ingest text files
- ✓ Query and get answers
- ✓ Use different reasoning modes

### Level 2: Multimodal
- Add images (photos, diagrams)
- Visual question answering
- See: `demos/demo_multimodal_rag.py`

### Level 3: Production
- Enable persistent storage (Neo4j + Qdrant)
- Deploy FastAPI server
- See: `DOCKER_MEMORY_SETUP.md`

### Level 4: Custom Workflows
- Visual workflow builder
- Multi-agent pipelines
- See: `WORKFLOW_BUILDER_COMPLETE.md`

---

## 💬 Support

- **Documentation**: `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
- **Quick Start**: `VISUAL_QUICK_START.md`
- **Demos**: `demos/` folder
- **Tests**: `pytest hololoom/rag/tests/ -v`

---

## 🎉 You're Ready!

Your smart AI is operational. Start with:

```bash
PYTHONPATH=. python my_smart_ai.py
```

Happy querying! 🚀