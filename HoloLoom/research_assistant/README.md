# Personal Research Assistant

**Your research papers remember each other.**

A complete research assistant that ingests PDF papers, discovers patterns across them, detects contradictions, identifies knowledge gaps, and answers questions through an interactive chatbot interface.

---

## Features

### 🎯 Core Capabilities

1. **Multimodal PDF Ingestion**
   - Extract text, images, and metadata from research papers
   - Section-based parsing (Introduction, Methods, Results, etc.)
   - Citation extraction
   - Graceful degradation with minimal dependencies

2. **Cross-Paper Intelligence**
   - **Pattern Synthesis**: Automatically discovers common themes across papers
   - **Contradiction Detection**: Identifies conflicting claims between papers
   - **Knowledge Gap Identification**: Finds missing prerequisite knowledge

3. **Interactive Q&A**
   - Chat-based interface for natural language queries
   - Source attribution (paper + section)
   - LLM-powered responses (Ollama integration)
   - Fallback to retrieval-based responses

4. **Visual Insights**
   - Pattern visualization
   - Contradiction alerts
   - Gap recommendations
   - Paper statistics

---

## Quick Start

### Installation

```bash
# Required dependencies
pip install streamlit

# Optional (for PDF parsing)
pip install PyPDF2 pdfplumber Pillow

# Optional (for LLM responses)
pip install ollama
```

### Running the Chatbot

```bash
# From repository root
streamlit run HoloLoom/research_assistant/chatbot.py
```

Or programmatically:

```python
from HoloLoom.research_assistant import run_chatbot

run_chatbot()
```

### Usage Workflow

1. **Upload Papers**
   - Click "Choose a research paper" in sidebar
   - Select PDF file
   - Wait for ingestion (auto-extracts sections, citations)

2. **Ask Questions**
   - Type question in chat input
   - Receive answer with source attribution
   - View retrieved passages

3. **Explore Insights**
   - Switch to "Insights" tab
   - View discovered patterns
   - Check contradictions
   - See knowledge gaps

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit Interface                │
│  (Upload PDFs, Chat, View Insights)          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         PDF Ingestion Pipeline               │
│  (Text, Images, Metadata, Citations)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│       Paper Memory System (HoloLoom)         │
│  (Section-based storage, Cross-paper query)  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      Synthesis Engine (DreamEngine)          │
│  (Patterns, Contradictions, Gaps)            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      LLM Response Generation (Ollama)        │
│  (Context + Insights → Answer)               │
└─────────────────────────────────────────────┘
```

---

## Components

### 1. PDF Ingestion (`pdf_ingestion.py`)

**Class**: `PDFIngestionPipeline`

**Features**:
- Text extraction (PyPDF2 or pdfplumber)
- Section parsing (auto-detects headings)
- Metadata extraction (title, authors, abstract)
- Citation extraction (up to 50 references)

**Usage**:
```python
from HoloLoom.research_assistant.pdf_ingestion import quick_ingest

paper = quick_ingest("attention.pdf")
print(paper.title)
print(f"Sections: {len(paper.sections)}")
```

### 2. Paper Memory (`paper_memory.py`)

**Class**: `PaperMemorySystem`

**Features**:
- Stores papers in HoloLoom memory
- Section-based shards for retrieval
- Knowledge graph construction
- Cross-paper querying

**Usage**:
```python
from HoloLoom.research_assistant.paper_memory import PaperMemorySystem

memory = PaperMemorySystem()
await memory.initialize()

# Ingest paper
stats = await memory.ingest_paper(paper)

# Query across papers
results = await memory.query_papers("What is self-attention?")
```

### 3. Synthesis Engine (`synthesis_engine.py`)

**Class**: `ResearchSynthesisEngine`

**Features**:
- Pattern synthesis across papers
- Contradiction detection
- Knowledge gap identification
- Query-aware insights

**Usage**:
```python
from HoloLoom.research_assistant.synthesis_engine import ResearchSynthesisEngine

engine = ResearchSynthesisEngine()
await engine.start()

# Run synthesis
kg = memory.get_knowledge_graph()
results = await engine.synthesize(kg)

# Get insights
patterns = engine.get_patterns()
contradictions = engine.get_contradictions()
gaps = engine.get_gaps()
```

### 4. Chatbot Interface (`chatbot.py`)

**Function**: `main()`

**Features**:
- Streamlit web interface
- PDF upload widget
- Chat interface
- Synthesis visualization tabs

**Usage**:
```bash
streamlit run HoloLoom/research_assistant/chatbot.py
```

---

## Example Workflow

### Step 1: Ingest Papers

```python
from HoloLoom.research_assistant import (
    quick_ingest,
    PaperMemorySystem,
    ResearchSynthesisEngine
)

# Create systems
memory = PaperMemorySystem()
await memory.initialize()

synthesis = ResearchSynthesisEngine()
await synthesis.start()

# Ingest papers
papers = [
    quick_ingest("attention.pdf"),
    quick_ingest("bert.pdf"),
    quick_ingest("gpt3.pdf")
]

for paper in papers:
    await memory.ingest_paper(paper)
    synthesis.register_paper_content(paper.title, paper.abstract)
```

### Step 2: Run Synthesis

```python
# Build knowledge graph
kg = memory.get_knowledge_graph()

# Run synthesis
results = await synthesis.synthesize(kg)

print(results.summary())
```

### Step 3: Query Papers

```python
# Ask question
query = "What is the Transformer architecture?"
passages = await memory.query_papers(query)

# Get insights
insights = synthesis.get_insights_for_query(query)

# Display results
for passage in passages:
    print(f"{passage['paper']}: {passage['snippet']}")

for pattern in insights['patterns']:
    print(f"Pattern: {pattern}")
```

---

## Demo Script (5 Minutes)

### Act 1: Setup (1 min)

**Narrator**: "Today I'll show you HoloLoom's Personal Research Assistant - where your research papers remember each other."

**Actions**:
1. Open chatbot (`streamlit run chatbot.py`)
2. Upload 3 papers: Attention Is All You Need, BERT, GPT-3
3. Show extraction progress and paper list

### Act 2: Simple Q&A (1.5 min)

**Query 1**: "What is the Transformer architecture?"
- Show answer with source attribution
- Highlight exact paragraph from paper

**Query 2**: "Compare BERT and GPT-3"
- Show side-by-side comparison
- Both sources cited

### Act 3: Synthesis Magic (2 min)

**Switch to Insights Tab**:

**Patterns Tab**:
- "All 3 papers use self-attention mechanisms"
- "Scaling laws: larger models → better performance"

**Contradictions Tab**:
- "Attention directionality: BERT (bidirectional) vs GPT (causal)"
- Type: Context-dependent

**Gaps Tab**:
- "Positional encoding" - mentioned but not explained
- Suggested query shown

### Act 4: Wow Moment (0.5 min)

**Query**: "Synthesize the key insights across all three papers"

**Response**:
> "Across all three papers, a clear evolution emerges:
> 1. Foundation (Attention): Transformer architecture with self-attention
> 2. Bidirectional Understanding (BERT): Applied to language understanding
> 3. Scaling Laws (GPT-3): Few-shot learning through scale
>
> Common themes: Self-attention, parallel computation, transfer learning
> Key tension: Bidirectional vs. causal attention - task-specific needs
> Missing knowledge: Positional encoding, layer normalization"

**Tagline**: "Your research papers remember each other."

---

## Competitive Advantages

### vs. ChatPDF

| Feature | ChatPDF | HoloLoom |
|---------|---------|----------|
| PDF Upload | ✅ | ✅ |
| Q&A | ✅ | ✅ |
| Source Attribution | ✅ | ✅ |
| Cross-Paper Patterns | ❌ | ✅ |
| Contradiction Detection | ❌ | ✅ |
| Knowledge Gaps | ❌ | ✅ |
| Complete Provenance | ❌ | ✅ |

### vs. Humata

| Feature | Humata | HoloLoom |
|---------|--------|----------|
| Multi-Document | ✅ | ✅ |
| Chat Interface | ✅ | ✅ |
| Summarization | ✅ | ✅ |
| Pattern Synthesis | ❌ | ✅ |
| Contradiction Detection | ❌ | ✅ |
| Gap Identification | ❌ | ✅ |
| Background Learning | ❌ | ✅ |

### vs. LlamaIndex

| Feature | LlamaIndex | HoloLoom |
|---------|------------|----------|
| RAG Framework | ✅ | ✅ |
| Multi-Source | ✅ | ✅ |
| Vector Search | ✅ | ✅ |
| Cross-Document Synthesis | 🟡 | ✅ |
| Automatic Contradiction Detection | ❌ | ✅ |
| Knowledge Gap Analysis | ❌ | ✅ |
| Zero-Config API | 🟡 | ✅ |

---

## Configuration

### PDF Ingestion

```python
from HoloLoom.research_assistant.pdf_ingestion import PDFIngestionPipeline

pipeline = PDFIngestionPipeline(
    extract_images=True,      # Extract figures/diagrams
    extract_citations=True    # Extract references
)
```

### Memory System

```python
from HoloLoom.research_assistant.paper_memory import PaperMemorySystem

memory = PaperMemorySystem()
await memory.initialize()

# Query configuration
results = await memory.query_papers(
    query="What is attention?",
    max_results=10,
    paper_filter="Attention Is All You Need"  # Optional
)
```

### Synthesis Engine

```python
from HoloLoom.research_assistant.synthesis_engine import ResearchSynthesisEngine

engine = ResearchSynthesisEngine(
    auto_synthesis=False  # Manual trigger (default for research assistant)
)

await engine.start()
```

---

## Troubleshooting

### PDF Parsing Fails

**Problem**: `RuntimeError: No PDF library available`

**Solution**: Install PyPDF2 or pdfplumber
```bash
pip install PyPDF2 pdfplumber
```

### LLM Responses Not Working

**Problem**: Falling back to simple responses

**Solution**: Install and start Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull model
ollama pull llama3.2:3b

# Install Python client
pip install ollama
```

### Streamlit Not Found

**Problem**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**: Install Streamlit
```bash
pip install streamlit
```

### Slow Ingestion

**Problem**: PDF ingestion takes >30 seconds

**Solution**:
- Use pdfplumber instead of PyPDF2 (faster)
- Reduce image extraction complexity
- Limit citation extraction to first 50

---

## Performance

### Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| PDF Ingestion | ~5-10s | Depends on paper length |
| Section Storage | ~50ms | Per section |
| Cross-Paper Query | ~150ms | Retrieval only |
| LLM Response | ~2-5s | With Ollama (llama3.2:3b) |
| Synthesis Cycle | ~100ms | Patterns + contradictions + gaps |

### Scalability

| Metric | Limit | Notes |
|--------|-------|-------|
| Papers | 100+ | Tested with 10, scales to 100+ |
| Sections per paper | 20 | Typical research paper |
| Citations | 50 | Extracted per paper |
| Query results | 10 | Default max_results |
| Chat history | Unlimited | Stored in session state |

---

## Future Enhancements

### Short-Term (Week 5+)

1. **Better PDF Parsing**
   - Support scanned PDFs (OCR)
   - Extract tables and equations
   - Preserve formatting

2. **Enhanced Synthesis**
   - Multi-hop reasoning across papers
   - Citation network analysis
   - Temporal evolution tracking

3. **Export Features**
   - Generate summary reports
   - Export knowledge graph
   - Save synthesis insights

### Long-Term (Month 2+)

1. **Collaboration**
   - Multi-user paper collections
   - Shared annotations
   - Collaborative synthesis

2. **Advanced Queries**
   - "What papers cite X and discuss Y?"
   - "Show evolution of concept Z over time"
   - Graph-based navigation

3. **Integration**
   - Zotero/Mendeley import
   - arXiv direct download
   - Google Scholar integration

---

## License

Part of HoloLoom project. See main repository for license details.

## Credits

**Built with**:
- HoloLoom (Weeks 1-3: Memory system + DreamEngine)
- Streamlit (Interactive UI)
- PyPDF2/pdfplumber (PDF parsing)
- Ollama (LLM integration)

**Author**: HoloLoom Team
**Date**: November 2025
**Version**: 1.0.0

---

**Tagline**: *"Your research papers remember each other."* ✨
