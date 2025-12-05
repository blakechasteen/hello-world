# Nonfiction Writing Toolkit

A comprehensive writing system for research-based nonfiction, built on Zero-G and HoloLoom.

## Overview

The Nonfiction Writing Toolkit supports the complete writing workflow from research to publication:

```
Research → Outline → Draft → Revise → Verify → Publish
```

Each phase is powered by HoloLoom's advanced capabilities:
- **Research**: RAG system (RESEARCH mode) + 47 document adapters
- **Outline**: Agentic reasoning (PLAN_EXECUTE mode)
- **Draft**: Context-aware generation with automatic citations
- **Revision**: Recursive refinement (ELEGANCE/VERIFY strategies)
- **Verification**: Fact-checking (agentic VERIFY mode)

## Quick Start

```python
import asyncio
from zero-g.backend.apps.nonfiction_writer import (
    CitationManager,
    ResearchPhase,
    OutlineGenerator,
    DraftGenerator,
    RevisionEngine,
    VerificationPhase,
)

async def write_article():
    # 1. Research
    citation_manager = CitationManager()
    research = ResearchPhase("Climate Adaptation", citation_manager)

    await research.ingest_pdf("paper1.pdf")
    await research.ingest_url("https://example.com/article")
    corpus = await research.build_corpus()

    # 2. Outline
    outline_gen = OutlineGenerator(corpus)
    outline = await outline_gen.generate(
        thesis="Adaptation requires multi-scale strategies",
        target_sections=5,
        target_word_count=3000
    )

    # 3. Draft
    draft_gen = DraftGenerator(outline, corpus, citation_manager)
    draft = await draft_gen.generate()

    # 4. Revise
    revision_engine = RevisionEngine()
    result = await revision_engine.revise(draft, max_iterations=3)
    draft = result.revised_draft

    # 5. Verify
    verifier = VerificationPhase(corpus)
    report = await verifier.verify(draft)

    # 6. Export
    print(draft.get_full_text())
    print(draft.bibliography)
    print(report.get_summary())

asyncio.run(write_article())
```

## Components

### CitationManager

Handles bibliographic citations with support for multiple formats:

```python
from zero-g.backend.apps.nonfiction_writer import CitationManager, CitationStyle

manager = CitationManager()

# Extract from PDF
citation = manager.extract_from_pdf_metadata(pdf_text, metadata)
manager.add_citation(citation)

# Format in different styles
apa = manager.format_citation(citation.id, CitationStyle.APA)
mla = manager.format_citation(citation.id, CitationStyle.MLA)
chicago = manager.format_citation(citation.id, CitationStyle.CHICAGO)

# Generate bibliography
bibliography = manager.generate_bibliography(CitationStyle.APA)
```

**Features**:
- 4 citation styles (APA, MLA, Chicago, IEEE)
- Extract from PDFs, web pages
- Inline citation generation
- Source credibility scoring
- Paragraph-level citation linking

### ResearchPhase

Wrapper around HoloLoom's RAG system for research gathering:

```python
from zero-g.backend.apps.nonfiction_writer import ResearchPhase

research = ResearchPhase("Your Topic")

# Ingest documents
await research.ingest_pdf("research.pdf")
await research.ingest_url("https://example.com")

# Perform research queries
result = await research.research_query("What are the main findings?")

# Build corpus
corpus = await research.build_corpus([
    "Question 1?",
    "Question 2?",
])

# Access organized research
peer_reviewed = corpus.get_peer_reviewed_sources()
primary = corpus.get_primary_sources()
```

**Features**:
- Multiple input formats (PDF, web, docs)
- Automatic citation extraction
- Knowledge graph building
- Theme identification
- Contradiction detection
- Gap analysis

### OutlineGenerator

Hierarchical outline generation using agentic reasoning:

```python
from zero-g.backend.apps.nonfiction_writer import OutlineGenerator, OutlineStyle

generator = OutlineGenerator(corpus)

# Generate outline
outline = await generator.generate(
    thesis="Your thesis statement",
    target_sections=5,
    max_depth=3,
    target_word_count=3000,
    style=OutlineStyle.ROMAN  # I, II, III → A, B, C → 1, 2, 3
)

# Validate coverage
validation = await generator.validate(outline)

# Refine based on feedback
outline = await generator.refine(outline, "Add more detail on section II")

# Export
markdown = outline.to_markdown()
json = outline.to_json()
```

**Features**:
- 3 numbering styles (Roman, Decimal, Alphanumeric)
- Hierarchical structure (up to 4 levels)
- Automatic validation
- Iterative refinement
- Multiple export formats

### DraftGenerator

Draft generation with automatic citation insertion:

```python
from zero-g.backend.apps.nonfiction_writer import DraftGenerator, CitationStyle

generator = DraftGenerator(outline, corpus, citation_manager)

# Generate draft
draft = await generator.generate(
    citation_style=CitationStyle.APA,
    target_density='moderate'  # sparse/moderate/dense
)

# Access draft components
full_text = draft.get_full_text()
section_text = draft.get_section_text("section_0001")
bibliography = draft.bibliography

# Statistics
print(f"Word count: {draft.word_count:,}")
print(f"Citations: {draft.get_citation_count()}")
print(f"Source coverage: {draft.get_source_coverage():.1%}")
```

**Features**:
- Context-aware paragraph generation
- Automatic citation insertion
- Source-to-paragraph linking
- Multiple citation densities
- Bibliography generation

### RevisionEngine

Multi-pass refinement using recursive learning:

```python
from zero-g.backend.apps.nonfiction_writer import RevisionEngine, RefinementStrategy

engine = RevisionEngine()

# Auto-select strategy
result = await engine.revise(draft, max_iterations=3)

# Specific strategy
result = await engine.revise(
    draft,
    strategy=RefinementStrategy.ELEGANCE,  # Clarity → Simplicity → Beauty
    quality_threshold=0.9
)

# Custom focus
result = await engine.revise_with_focus(
    draft,
    focus="Make it more accessible to non-experts"
)

# Review improvements
print(result.get_summary())
draft = result.revised_draft
```

**Strategies**:
- **ELEGANCE**: Clarity → Simplicity → Beauty (3 passes)
- **VERIFY**: Accuracy → Completeness → Consistency (3 passes)
- **CRITIQUE**: Self-improvement loops

**Features**:
- Quality trajectory tracking
- Automatic improvement detection
- Pass-by-pass reporting
- Configurable quality thresholds

### VerificationPhase

Fact-checking using agentic VERIFY mode:

```python
from zero-g.backend.apps.nonfiction_writer import VerificationPhase

verifier = VerificationPhase(corpus)

# Verify entire draft
report = await verifier.verify(draft)

# Check specific claim
status, confidence = await verifier.verify_claim(
    "Climate change causes sea level rise"
)

# Review report
print(report.get_summary())
print(f"Verification rate: {report.get_verification_rate():.1%}")

# Access details
for claim in report.claims:
    if claim.status == ClaimStatus.UNSUPPORTED:
        print(f"⚠ Unsupported: {claim.text}")
```

**Features**:
- Automatic claim extraction
- Source-based verification
- Contradiction detection
- Confidence scoring
- Issue identification
- Actionable recommendations

## Demo

Run the complete end-to-end demo:

```bash
cd zero-g/backend/apps/nonfiction_writer
PYTHONPATH=../../../../ python demo_writing_workflow.py
```

This demonstrates:
1. Research ingestion (3 sources)
2. Outline generation (5 sections, 3 levels)
3. Draft creation (with citations)
4. Revision (ELEGANCE strategy, 3 passes)
5. Verification (fact-checking)
6. Export (Markdown, JSON, bibliography)

## Integration with HoloLoom

The toolkit integrates with HoloLoom's advanced capabilities:

**RAG System** (`HoloLoom/rag/`):
- Simple RAG for research queries
- Multimodal RAG for images + text
- 4 reasoning modes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)

**SpinningWheel** (`HoloLoom/spinningWheel/`):
- 47 document adapters
- Advanced PDF extraction
- Web scraping
- Citation extraction

**Recursive Learning** (`HoloLoom/recursive/`):
- Multi-pass refinement
- Quality tracking
- Strategy adaptation

**Agentic Reasoning** (`HoloLoom/agentic/`):
- PLAN_EXECUTE for outlining
- VERIFY for fact-checking
- Multi-step reasoning

**Memory Systems**:
- YarnGraph for citation networks
- Vector memory for semantic search
- Query cache for performance

## Citation Styles

All citation styles follow the latest editions:

- **APA**: American Psychological Association (7th edition)
- **MLA**: Modern Language Association (9th edition)
- **Chicago**: Chicago Manual of Style (17th edition)
- **IEEE**: Institute of Electrical and Electronics Engineers

Example citations:

**APA**:
```
Smith, J., & Jones, M. (2023). Climate adaptation strategies. *Nature Climate Change*, 13(4), 234-245. https://doi.org/10.1038/example
```

**MLA**:
```
Smith, John, et al. "Climate Adaptation Strategies." *Nature Climate Change*, vol. 13, no. 4, 2023, pp. 234-245.
```

**Chicago**:
```
Smith, John et al. "Climate Adaptation Strategies." *Nature Climate Change* 13, no. 4 (2023): 234-245.
```

## Output Formats

Export your work in multiple formats:

**Markdown**:
```python
markdown = draft.get_full_text()
Path("output.md").write_text(markdown)
```

**JSON**:
```python
json_data = outline.to_json()
Path("output.json").write_text(json_data)
```

**Bibliography**:
```python
bibliography = citation_manager.generate_bibliography(CitationStyle.APA)
Path("bibliography.txt").write_text(bibliography)
```

## Best Practices

### 1. Research Phase
- Ingest high-quality, peer-reviewed sources
- Use diverse source types (journals, books, reports)
- Verify source credibility before including
- Build comprehensive corpus (10+ sources recommended)

### 2. Outline Phase
- Start with a clear thesis statement
- Aim for 3-5 main sections
- Keep depth to 3 levels maximum
- Validate outline before drafting

### 3. Draft Phase
- Use moderate citation density (2-3 per paragraph)
- Ensure all key points have source support
- Link citations to specific claims
- Generate bibliography early

### 4. Revision Phase
- Use ELEGANCE for general improvement
- Use VERIFY for technical accuracy
- Run multiple passes (3 recommended)
- Track quality improvements

### 5. Verification Phase
- Verify before final publication
- Address all unsupported claims
- Resolve contradictions
- Aim for >80% verification rate

## Roadmap

**Phase 1** (Current): Core workflow ✅
- Research, Outline, Draft, Revise, Verify

**Phase 2** (Week 3-4): Zero-G Integration
- App docking protocol
- Lifecycle management (Preflight → Orbit)
- Safety checks

**Phase 3** (Week 5+): Advanced Features
- Source credibility scoring
- Collaborative editing
- Export formats (PDF, DOCX, LaTeX)
- Additional style guides
- Plagiarism detection

## Contributing

To extend the toolkit:

1. **Add new citation style**: Extend `CitationManager._format_*` methods
2. **Add document adapter**: Integrate with SpinningWheel
3. **Add revision strategy**: Extend `RevisionEngine._improve_*` methods
4. **Add export format**: Extend `Draft.to_*` methods

## License

Part of the mythRL/HoloLoom project.

## Support

For issues or questions, see the main HoloLoom documentation or open an issue on GitHub.
