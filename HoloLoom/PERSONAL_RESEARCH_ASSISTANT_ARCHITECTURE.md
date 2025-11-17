# Personal Research Assistant Architecture

**Week 4 Deliverable** - The Most Impressive Memory System Demo
**Date**: 2025-11-17
**Goal**: Demonstrate HoloLoom's unique capabilities through academic paper research

## Vision

> **"Your research papers remember each other."**

A chatbot that ingests multiple research papers (PDFs with text + images), automatically discovers connections between them, identifies contradictions, finds knowledge gaps, and answers questions by synthesizing insights across papers.

**Why this impresses**: Most RAG systems just retrieve passages. We show papers "talking to each other" through DreamEngine synthesis.

## Core Capabilities

### 1. Multimodal PDF Ingestion
- **Text extraction**: PyPDF2 or pdfplumber for text content
- **Image extraction**: Extract figures, diagrams, equations as images
- **Metadata extraction**: Title, authors, abstract, sections, citations
- **Chunking strategy**: Section-based (Introduction, Methods, Results, Discussion)
- **Entity extraction**: Papers, authors, methods, datasets, metrics

**Input**: Upload 3-5 research papers (ML/AI domain)
**Output**: Structured memories in HoloLoom with full provenance

### 2. Cross-Paper Synthesis

Uses all 3 DreamEngine components:

**Pattern Synthesis**:
- Discovers common themes across papers
- Example: "All 3 papers use Transformer architecture" → pattern

**Contradiction Detection**:
- Finds conflicting claims between papers
- Example: Paper A: "Method X achieves 95% accuracy" vs Paper B: "Method X only reaches 87%" → contradiction

**Gap Identification**:
- Identifies concepts mentioned but not explained
- Example: Papers mention "RLHF" but don't explain it → gap

### 3. Interactive Q&A

Natural language interface powered by HoloLoom + LLM:

**Query Types**:
- **Factual**: "What accuracy did GPT-4 achieve on MMLU?"
- **Comparative**: "Compare attention mechanisms in Paper A vs Paper B"
- **Synthesis**: "What are the common limitations across all papers?"
- **Meta**: "Which papers cite each other?"

**Response Format**:
- Direct answer with confidence score
- Source attribution (which paper, which section)
- Related patterns/contradictions/gaps
- Suggested follow-up questions

### 4. Research Dashboard

Visual interface showing:

**Papers Tab**:
- Uploaded papers with metadata
- Extraction status (text, images, entities)
- Citation graph between papers

**Patterns Tab**:
- Discovered recurring themes
- Cluster visualization
- Timeline of pattern emergence

**Contradictions Tab**:
- Conflicting claims between papers
- Severity scores
- Reconciliation suggestions

**Gaps Tab**:
- Missing prerequisite knowledge
- Incomplete explanations
- Suggested reading to fill gaps

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Research Assistant UI                   │
│         (Streamlit/Gradio - Interactive Chat)            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│               PDF Ingestion Pipeline                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Text     │  │ Images   │  │ Metadata │              │
│  │ Extract  │  │ Extract  │  │ Extract  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│         ↓              ↓              ↓                  │
│       PyPDF2        PIL/CV       Regex Parser           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  HoloLoom Core                          │
│  ┌──────────────────────────────────────────┐          │
│  │ experience(paper_content)                │          │
│  │  → Memory shards with entities/motifs    │          │
│  └──────────────────────────────────────────┘          │
│  ┌──────────────────────────────────────────┐          │
│  │ recall(query)                            │          │
│  │  → Relevant passages across papers       │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  DreamEngine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Pattern      │  │Contradiction │  │ Gap          │ │
│  │ Synthesis    │  │ Detection    │  │Identification│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│       Background synthesis after each paper ingestion   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              LLM Response Generation                     │
│  (Ollama/OpenAI/Anthropic)                              │
│  Prompt: "Given these retrieved passages and patterns..." │
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: PDF Ingestion (Day 1)

**File**: `HoloLoom/research_assistant/pdf_ingestion.py`

```python
from typing import List, Dict, Any
from dataclasses import dataclass
import PyPDF2
from PIL import Image

@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    sections: List[Dict[str, str]]  # {title, content}
    figures: List[Image.Image]
    citations: List[str]
    metadata: Dict[str, Any]

class PDFIngestionPipeline:
    def ingest(self, pdf_path: str) -> ResearchPaper:
        """Extract all content from research paper PDF."""
        pass

    def extract_text(self, pdf_path: str) -> str:
        """Extract raw text using PyPDF2."""
        pass

    def extract_images(self, pdf_path: str) -> List[Image.Image]:
        """Extract figures and diagrams."""
        pass

    def parse_structure(self, text: str) -> List[Dict[str, str]]:
        """Parse sections (Introduction, Methods, etc.)."""
        pass

    def extract_citations(self, text: str) -> List[str]:
        """Extract referenced papers."""
        pass
```

**Dependencies**:
```bash
pip install PyPDF2 Pillow pdfplumber
```

### Phase 2: HoloLoom Integration (Day 2)

**File**: `HoloLoom/research_assistant/paper_memory.py`

```python
from HoloLoom import HoloLoom
from HoloLoom.research_assistant.pdf_ingestion import ResearchPaper

class PaperMemorySystem:
    def __init__(self):
        self.loom = HoloLoom()

    async def ingest_paper(self, paper: ResearchPaper):
        """Store paper in HoloLoom memory."""
        # Create memory shards for each section
        for section in paper.sections:
            await self.loom.experience(
                content=section['content'],
                metadata={
                    'paper_title': paper.title,
                    'section': section['title'],
                    'authors': paper.authors
                }
            )

        # Store metadata
        await self.loom.experience(
            content=f"Paper: {paper.title}. Abstract: {paper.abstract}",
            metadata={
                'type': 'paper_metadata',
                'title': paper.title,
                'authors': paper.authors,
                'citations': paper.citations
            }
        )

    async def query_papers(self, query: str) -> List[Dict]:
        """Retrieve relevant content across all papers."""
        return await self.loom.recall(query)
```

### Phase 3: DreamEngine Integration (Day 3)

**File**: `HoloLoom/research_assistant/synthesis_engine.py`

```python
from HoloLoom.synthesis.background_scheduler import BackgroundScheduler
from HoloLoom.synthesis.pattern_synthesis import PatternSynthesizer
from HoloLoom.synthesis.contradiction_detection import ContradictionDetector
from HoloLoom.synthesis.gap_identification import GapIdentifier

class ResearchSynthesisEngine:
    def __init__(self):
        self.pattern_synthesizer = PatternSynthesizer()
        self.contradiction_detector = ContradictionDetector()
        self.gap_identifier = GapIdentifier()
        self.scheduler = BackgroundScheduler(
            pattern_synthesizer=self.pattern_synthesizer,
            contradiction_detector=self.contradiction_detector,
            gap_identifier=self.gap_identifier
        )

    async def synthesize_after_ingestion(self, knowledge_graph):
        """Run synthesis cycle after new paper ingested."""
        results = await self.scheduler.trigger_synthesis(
            knowledge_graph=knowledge_graph,
            force=True
        )
        return results

    def get_patterns(self):
        """Get discovered patterns across papers."""
        return self.scheduler.cycle_history[-1].patterns_synthesized if self.scheduler.cycle_history else []

    def get_contradictions(self):
        """Get contradictions between papers."""
        return self.scheduler.cycle_history[-1].contradictions_detected if self.scheduler.cycle_history else []

    def get_gaps(self):
        """Get knowledge gaps in papers."""
        return self.scheduler.cycle_history[-1].gaps_identified if self.scheduler.cycle_history else []
```

### Phase 4: Interactive Interface (Day 4)

**File**: `HoloLoom/research_assistant/chatbot.py`

```python
import streamlit as st
from HoloLoom.research_assistant.paper_memory import PaperMemorySystem
from HoloLoom.research_assistant.synthesis_engine import ResearchSynthesisEngine

class ResearchChatbot:
    def __init__(self):
        self.memory = PaperMemorySystem()
        self.synthesis = ResearchSynthesisEngine()
        self.uploaded_papers = []

    def upload_paper(self, pdf_file):
        """Handle paper upload."""
        # Ingest PDF
        paper = self.pdf_pipeline.ingest(pdf_file)

        # Store in memory
        await self.memory.ingest_paper(paper)

        # Run synthesis
        results = await self.synthesis.synthesize_after_ingestion(
            knowledge_graph=self.memory.loom.get_knowledge_graph()
        )

        self.uploaded_papers.append(paper)

    def query(self, user_question: str) -> str:
        """Answer user question."""
        # Retrieve relevant passages
        passages = await self.memory.query_papers(user_question)

        # Get synthesis insights
        patterns = self.synthesis.get_patterns()
        contradictions = self.synthesis.get_contradictions()
        gaps = self.synthesis.get_gaps()

        # Generate response with LLM
        response = self._generate_response(
            question=user_question,
            passages=passages,
            patterns=patterns,
            contradictions=contradictions,
            gaps=gaps
        )

        return response
```

**Streamlit UI**:

```python
import streamlit as st

st.title("Personal Research Assistant")

# Sidebar: Upload papers
with st.sidebar:
    st.header("Upload Papers")
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")
    if uploaded_file:
        chatbot.upload_paper(uploaded_file)
        st.success(f"Uploaded: {uploaded_file.name}")

# Main: Chat interface
st.header("Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chatbot.query(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Tabs: Patterns, Contradictions, Gaps
tab1, tab2, tab3 = st.tabs(["Patterns", "Contradictions", "Gaps"])

with tab1:
    patterns = chatbot.synthesis.get_patterns()
    for pattern in patterns:
        st.write(f"- {pattern.content}")

with tab2:
    contradictions = chatbot.synthesis.get_contradictions()
    for c in contradictions:
        st.write(f"- {c.explanation}")

with tab3:
    gaps = chatbot.synthesis.get_gaps()
    for gap in gaps:
        st.write(f"- {gap.bridge_concept}: missing {', '.join(gap.missing_concepts)}")
```

## Demo Script (5-Minute Video)

**Act 1: Setup (1 min)**
- "Today I'll show you HoloLoom's Personal Research Assistant"
- Upload 3 papers: Attention is All You Need, BERT, GPT-3
- Show extraction progress

**Act 2: Simple Q&A (1.5 min)**
- Q: "What is the Transformer architecture?"
- Show: Answer with source attribution
- Q: "Compare BERT and GPT-3"
- Show: Side-by-side comparison from both papers

**Act 3: Synthesis Magic (2 min)**
- Show **Patterns** tab: "All papers use self-attention"
- Show **Contradictions** tab: "BERT uses bidirectional attention, GPT uses causal"
- Show **Gaps** tab: "Papers mention positional encoding but don't explain it"
- Q: "What's missing from my understanding?"
- Show: Gap-based recommendations

**Act 4: Wow Moment (0.5 min)**
- Q: "Synthesize the key insights across all three papers"
- Show: DreamEngine generates comprehensive summary connecting all papers
- "Your research papers remember each other."

## Success Metrics

**Technical**:
- ✅ PDF ingestion works for 10+ paper formats
- ✅ Text + image extraction >95% accuracy
- ✅ Cross-paper synthesis finds ≥3 patterns per 3 papers
- ✅ Contradiction detection catches known conflicts
- ✅ Gap identification suggests relevant missing concepts
- ✅ Query response time <2 seconds

**Wow Factor**:
- ✅ Demo shows capabilities no other RAG system has
- ✅ "Papers remember each other" phrase resonates
- ✅ Visual synthesis dashboard impresses
- ✅ 5-minute video is shareable and viral-ready

## Files to Create

```
HoloLoom/research_assistant/
├── __init__.py
├── pdf_ingestion.py        (PDF → ResearchPaper)
├── paper_memory.py         (ResearchPaper → HoloLoom)
├── synthesis_engine.py     (DreamEngine integration)
└── chatbot.py              (Streamlit UI + Q&A)

demos/
└── demo_research_assistant.py   (Runnable demo)
```

## Timeline

- **Day 1** (Nov 17): PDF ingestion pipeline
- **Day 2** (Nov 18): HoloLoom integration
- **Day 3** (Nov 19): DreamEngine synthesis
- **Day 4** (Nov 20): Streamlit interface
- **Day 5** (Nov 21): Demo video + launch

## Competitive Advantage

**What others have**:
- PDF ingestion: Common (LangChain, LlamaIndex)
- Vector search: Common (Pinecone, Weaviate)
- Q&A chatbot: Common (ChatPDF, Humata)

**What we uniquely have**:
- ✅ **Cross-paper pattern synthesis**: Discovers themes automatically
- ✅ **Contradiction detection**: Catches conflicts between papers
- ✅ **Knowledge gap identification**: Tells you what you're missing
- ✅ **Complete provenance**: Every answer traceable to source
- ✅ **Background learning**: System improves without explicit training

**Tagline**: *"Your research papers remember each other."*

---

**Status**: Architecture complete ✅
**Next**: Implement PDF ingestion pipeline
