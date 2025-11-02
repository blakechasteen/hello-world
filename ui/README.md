# 🧠 Consciousness Stack Web UI

Interactive Gradio interface for exploring the complete mythRL consciousness infrastructure.

## Quick Start

### Option 1: PowerShell Script (Recommended)
```powershell
.\launch_ui.ps1
```

### Option 2: Direct Python
```powershell
$env:PYTHONPATH = "."; python ui/consciousness_ui.py
```

Then open your browser to: **http://localhost:7860**

## Features

### 🔍 **Pipeline Visualization**
Watch your query flow through all 5 consciousness stages:

1. **Awareness Analysis** - Confidence scoring, pattern recognition, structural analysis
2. **Memory Fusion** - Multipass graph crawling with depth tracking
3. **Context Packing** - Token optimization with importance weighting
4. **LLM Context** - See the exact formatted context sent to LLM
5. **Dual-Stream Generation** - Internal reasoning + external response

### ⚙️ **Interactive Controls**

- **Complexity Level**: LITE → FAST → FULL → RESEARCH
  - Controls multipass depth and gating thresholds
  - LITE: 1 pass, <50ms
  - RESEARCH: 4 passes, deep exploration

- **Memory Fusion**: Toggle multipass graph crawling on/off
  - See the difference between single retrieval vs. recursive exploration
  
- **Max Memories**: 5-20 items
  - Control how many knowledge items to retrieve
  
- **Token Budget**: 1000-8000 tokens
  - Set available context window size

### 📊 **Real-Time Metrics**

- Cache hit rates and speedup factors
- Retrieval depth and composite scores
- Token usage and compression ratios
- Sub-millisecond timing breakdowns
- Quality metrics (importance, confidence)

### 💡 **Example Queries**

The UI includes pre-loaded examples:
- "What are the applications of quantum computing?"
- "Explain quantum entanglement"
- "How does quantum teleportation work?"
- "What are the challenges in building quantum computers?"

## UI Components

### Left Panel: Configuration
- Query input textbox
- Complexity selector (radio buttons)
- Memory fusion toggle
- Max memories slider
- Token budget slider
- Process button
- JSON results viewer

### Right Panel: Results
- Tabbed interface for each pipeline stage
- Markdown formatting for readability
- Code blocks for LLM context
- Performance summary tab

## Architecture

```
Query Input
    ↓
[Awareness Analysis]  ← CompositionalAwarenessLayer
    ↓
[Memory Fusion]       ← MemoryFusion (multipass graph crawling)
    ↓
[Context Packing]     ← SmartContextPacker (importance weighting)
    ↓
[LLM Context]         ← Formatted markdown with metadata
    ↓
[Generation]          ← DualStreamGenerator (internal + external)
    ↓
Final Response
```

## Demo Memory Backend

The UI includes a **UIMemoryBackend** with a small quantum computing knowledge graph:
- 5 interconnected knowledge nodes
- Entity relationships for graph traversal
- Relevance scores (0.82-0.95)
- Timestamps for temporal weighting

This demonstrates the full consciousness stack without requiring Neo4j/Qdrant.

## Performance Targets

- **LITE**: <50ms total, 1 retrieval pass
- **FAST**: <150ms total, 2 retrieval passes
- **FULL**: <300ms total, 3 retrieval passes
- **RESEARCH**: No limit, 4+ retrieval passes

*Actual times shown in Performance tab*

## What You'll See

### Awareness Tab
```
🔍 Awareness Analysis
Confidence: 87.3%
Uncertainty: 12.7%
Cache Status: MISS
Knowledge Gap: No

Structure:
- Type: question
- Is Question: True
- Expected Response: factual_answer

Pattern Recognition:
- Domain: science/quantum_physics
- Seen Count: 0×
- Pattern Confidence: 85.0%
```

### Memory Fusion Tab
```
🕷️ Memory Fusion
Retrieved: 10 memories
Max Depth: 2
Avg Score: 0.886
Passes: 3

Top Memories:
1. [Depth 0, Score 0.950] Quantum entanglement is a physical phenomenon...
2. [Depth 1, Score 0.920] Quantum computing uses superposition and entanglement...
3. [Depth 1, Score 0.880] Quantum teleportation transfers quantum states...
```

### Context Packing Tab
```
📦 Context Packing
Total Tokens: 487/2700
Elements: 12 included, 3 compressed, 0 excluded
Avg Importance: 74%
Min Importance: 68%
Packing Time: 0.83ms

Compression Stats: {
  "included_uncompressed": 9,
  "included_compressed": 3,
  "excluded": 0
}
```

### Performance Tab
```
⚡ Performance Summary
Total Time: 4.73ms
- Awareness: <1ms
- Memory Fusion: <2ms
- Context Packing: 0.83ms
- Generation: 3.90ms

Efficiency:
- Token Usage: 18.0%
- Quality: 74% importance
- Compression: 25% compressed
```

## Customization

### Add Your Own Memory Backend

Replace `UIMemoryBackend` with your own:

```python
from HoloLoom.memory.protocol import MemoryBackend

# Use your backend
memory_backend = YourCustomBackend()
```

### Enable Real LLM Generation

Modify the generation call:

```python
response = await generator.generate(
    query, 
    show_internal=True, 
    use_llm=True  # Enable real LLM
)
```

### Adjust Theme and Layout

Gradio supports custom themes:

```python
demo = gr.Blocks(
    title="Custom Title",
    theme=gr.themes.Base()  # or Soft(), Glass(), etc.
)
```

## Troubleshooting

### "Module 'gradio' not found"
```powershell
pip install gradio
```

### "PYTHONPATH issues"
Always run from repo root with `$env:PYTHONPATH = "."`

### "Port 7860 already in use"
Change port in `consciousness_ui.py`:
```python
demo.launch(server_port=7861)
```

### Slow performance
- Reduce token budget
- Use LITE/FAST complexity
- Disable memory fusion for testing

## Next Steps

1. **Connect Real Memory**: Swap UIMemoryBackend for Neo4j/Qdrant
2. **Enable LLM**: Set `use_llm=True` and configure API keys
3. **Add Visualizations**: Graph network diagrams, token usage charts
4. **Deploy**: Use `share=True` for public Gradio link
5. **Extend**: Add chat history, multi-turn conversations, file uploads

## Production Deployment

```bash
# Install in production environment
pip install gradio uvicorn

# Run with gunicorn for production
gunicorn ui.consciousness_ui:app -w 4 -k uvicorn.workers.UvicornWorker
```

Or use the Gradio Cloud deployment:
```python
demo.launch(share=True)  # Creates public link
```

## Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [mythRL Architecture](../COMPLETE_CONSCIOUSNESS_STACK.md)
- [Memory Fusion Details](../demos/demo_memory_fusion.py)
- [Context Packer Demo](../demos/demo_context_packer.py)

---

**Built with**: Gradio 4.x, Python 3.8+, mythRL Consciousness Stack

**License**: MIT
