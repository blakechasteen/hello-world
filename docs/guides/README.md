# HoloLoom - Neural Decision-Making Through Weaving

A Python-based system that implements computation as a **weaving process** - transforming queries into fabric through coordinated multi-modal intelligence.

---

## Quick Start

```python
from HoloLoom import HoloLoom

# Create instance
loom = await HoloLoom.create()

# Query
response = await loom.query("What is HoloLoom?")
print(response.response)

# Chat
response = await loom.chat("Tell me more")

# Ingest data
await loom.ingest_text("Knowledge base content...")
await loom.ingest_web("https://example.com")
```

**Run Demo:**
```bash
export PYTHONPATH=.
python HoloLoom/unified_api.py
```

---

## What is HoloLoom?

HoloLoom combines cutting-edge ML techniques through a unique **weaving metaphor architecture**:

- **Multi-scale Embeddings**: Matryoshka representations (96d, 192d, 384d)
- **Knowledge Graph Memory**: NetworkX-based symbolic memory with spectral features
- **Thompson Sampling**: Bayesian exploration/exploitation balance
- **Synthesis Pipeline**: Automatic entity extraction and pattern mining
- **Multi-Modal Ingestion**: Text, web, YouTube, audio processing

---

## The Weaving Architecture

HoloLoom treats computation as **literal weaving** through 7 stages:

1. **LoomCommand** - Selects execution pattern (BARE/FAST/FUSED)
2. **ChronoTrigger** - Temporal control and thread activation
3. **ResonanceShed** - Multi-modal feature extraction and interference
4. **SynthesisBridge** - Pattern enrichment (entities, reasoning, topics)
5. **WarpSpace** - Tensions threads into continuous tensor field
6. **ConvergenceEngine** - Collapses probabilities to discrete decisions
7. **Spacetime** - Woven fabric output with complete provenance

**Queries literally get woven into fabric with full computational lineage!**

---

## Repository Structure

**Clean Root (Phase 1+2 Cleanup - Oct 2025):**

```
mythRL/
├── HoloLoom/               # Core system modules
│   ├── weaving_orchestrator.py   # Main weaving cycle coordinator
│   ├── weaving_shuttle.py        # Async context manager entry point
│   ├── unified_api.py             # Unified API (HoloLoom class)
│   ├── config.py                  # Configuration (BARE/FAST/FUSED)
│   ├── protocols/                 # Protocol definitions (canonical)
│   ├── policy/                    # Neural decision-making
│   ├── embedding/                 # Multi-scale embeddings
│   ├── memory/                    # Knowledge graph & vector storage
│   ├── motif/                     # Pattern detection
│   ├── spinningWheel/             # Multi-modal data ingestion
│   ├── synthesis/                 # Entity extraction & pattern mining
│   ├── loom/                      # Pattern card system
│   ├── chrono/                    # Temporal control
│   ├── resonance/                 # Feature interference
│   ├── warp/                      # Tensor operations
│   ├── convergence/               # Decision collapse
│   ├── fabric/                    # Spacetime trace generation
│   ├── tests/                     # 3-tier test structure
│   │   ├── unit/                  # Fast isolated tests (<5s)
│   │   ├── integration/           # Multi-component tests (<30s)
│   │   └── e2e/                   # Full pipeline tests (<2min)
│   └── tools/                     # Developer utilities
│       ├── bootstrap_system.py
│       ├── validate_pipeline.py
│       └── archive/               # Archived code (safety net)
│
├── demos/                  # Usage examples
│   ├── 01_quickstart.py
│   ├── 02_web_to_memory.py
│   ├── 03_conversational.py
│   └── 04_mcp_integration.py
│
├── config/                 # Configuration files
├── docs/                   # Documentation
│   ├── sessions/           # Development session logs
│   └── guides/             # Feature guides
│
├── CLAUDE.md              # Complete developer guide
└── README.md              # This file
```

---

## Key Features

**Unified API:**
- `HoloLoom.query()` - One-shot queries with full trace
- `HoloLoom.chat()` - Conversational interface with context
- `HoloLoom.ingest_*()` - Multi-modal data ingestion

**Weaving Cycle:**
- 7-stage processing pipeline (9-12ms execution)
- Complete Spacetime traces with full provenance
- Entity extraction and reasoning detection
- Pattern mining infrastructure

**Memory Systems:**
- Simple, Neo4j, Qdrant, or hybrid backends
- Graph-based symbolic memory
- Vector-based semantic search
- Spectral graph features

**Data Ingestion:**
- Text processing via TextSpinner
- Web scraping via WebsiteSpinner
- YouTube transcription via YouTubeSpinner
- Audio processing via AudioSpinner

---

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy gymnasium matplotlib

# Optional: Full features
pip install spacy sentence-transformers scipy networkx ollama
python -m spacy download en_core_web_sm
```

---

## Usage Examples

### Basic Query
```python
from HoloLoom import HoloLoom

loom = await HoloLoom.create(pattern="fast")
result = await loom.query("What is Thompson Sampling?")

print(result.response)
print(f"Confidence: {result.confidence:.1%}")
print(f"Entities: {result.trace.synthesis_result['entities']}")
```

### Conversational Chat
```python
loom = await HoloLoom.create()

await loom.chat("What is HoloLoom?")
await loom.chat("Tell me about the weaving metaphor")
await loom.chat("How does synthesis work?")

# View conversation history
for turn in loom.conversation_history:
    print(f"User: {turn['user']}")
    print(f"Assistant: {turn['assistant']}")
```

### Data Ingestion
```python
# Ingest text
count = await loom.ingest_text("""
HoloLoom is a neural decision-making system...
""")

# Scrape website
count = await loom.ingest_web("https://example.com/docs")

# Process YouTube video
count = await loom.ingest_youtube("VIDEO_ID", languages=['en'])
```

### Pattern Selection
```python
# Fast mode (default)
result = await loom.query("Quick question", pattern="fast")

# High quality mode
result = await loom.query("Complex analysis", pattern="fused")

# Minimal mode
result = await loom.query("Simple lookup", pattern="bare")
```

---

## Performance

- **Weaving Cycle:** 9-12ms per query
- **Synthesis Overhead:** 0-2.5ms
- **Entity Extraction:** Real-time
- **Pattern Detection:** Working
- **Thompson Sampling:** Operational

---

## Documentation

**Getting Started:**
- [README.md](README.md) - This file
- [CLAUDE.md](CLAUDE.md) - Complete developer guide
- [demos/](demos/) - Working examples

**Architecture:**
- [HoloLoom/weaving_orchestrator.py](HoloLoom/weaving_orchestrator.py) - Weaving cycle
- [HoloLoom/synthesis_bridge.py](HoloLoom/synthesis_bridge.py) - Synthesis integration
- [HoloLoom/unified_api.py](HoloLoom/unified_api.py) - Unified API

**Session Logs:**
- [docs/sessions/](docs/sessions/) - Development session documentation
- [docs/guides/](docs/guides/) - Feature guides and tutorials

---

## Testing

**3-Tier Test Structure** (Phase 1+2 Cleanup - Oct 2025):

```bash
# Unit Tests (Fast - <5s) - Isolated component testing
pytest HoloLoom/tests/unit/ -v

# Integration Tests (Medium - <30s) - Multi-component testing
pytest HoloLoom/tests/integration/ -v

# End-to-End Tests (Slow - <2min) - Full pipeline testing
pytest HoloLoom/tests/e2e/ -v

# Run all tests
pytest HoloLoom/tests/ -v

# Run demo
PYTHONPATH=. python HoloLoom/unified_api.py

# Run specific demo
PYTHONPATH=. python demos/01_quickstart.py
```

---

## Architecture Highlights

**Symbolic ↔ Continuous:**
- Seamless transitions between discrete (Yarn Graph) and continuous (Warp Space)
- Tension/detension lifecycle for tensor operations

**Temporal Control:**
- Fine-grained timing via ChronoTrigger
- Thread decay and evolution mechanisms
- Temporal windows for context selection

**Multi-Modal Fusion:**
- Interference patterns from motifs, embeddings, spectral features
- ResonanceShed creates "DotPlasma" (flowing features)

**Complete Provenance:**
- Every Spacetime fabric includes full computational trace
- Pattern card, temporal window, motifs, entities, reasoning type, confidence

**Adaptive Learning:**
- Reflection Buffer stores outcomes
- Thompson Sampling balances exploration/exploitation
- System evolves through experience

---

## Use Cases

- Intelligent query processing with multi-modal understanding
- Conversational AI with memory and context
- Knowledge graph reasoning with spectral analysis
- Multi-modal data ingestion and synthesis
- Pattern extraction from conversations
- Training data generation for LLMs

---

## Development

See [CLAUDE.md](CLAUDE.md) for:
- Complete architecture documentation
- Development commands
- Module structure
- Testing strategy
- Common workflows
- Import path requirements

---

## Status

**Current Version:** v1.0 - Integration Sprint Complete

**What Works:**
- Complete 7-stage weaving cycle
- Full synthesis integration
- Unified API (query, chat, ingest)
- Entity extraction
- Reasoning detection
- Pattern mining infrastructure
- Multi-backend memory support
- Thompson Sampling exploration

**Production Ready:** Yes

---

**Where queries become fabric.**

Created with [Claude Code](https://claude.com/claude-code)
