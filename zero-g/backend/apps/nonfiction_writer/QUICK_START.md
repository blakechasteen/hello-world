# Quick Start Guide - Nonfiction Writing Toolkit

Get started writing in **5 minutes**! ⏱️

## Installation

No installation needed! The toolkit uses Python standard library + optional HoloLoom integration.

**Optional (for full features)**:
```bash
pip install torch numpy  # For HoloLoom integration
```

## Your First Article (3 minutes)

Create `my_article.py`:

```python
import asyncio
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from zero-g.backend.apps.nonfiction_writer import (
    CitationManager,
    ResearchPhase,
    OutlineGenerator,
    DraftGenerator,
    CitationStyle,
)

async def write_article():
    # 1. Research (30 seconds)
    print("📚 Gathering research...")
    citation_manager = CitationManager()
    research = ResearchPhase("Climate Change Adaptation", citation_manager)

    # Ingest a web article (or use ingest_pdf for PDFs)
    await research.ingest_url("https://example.com/climate-article")
    corpus = await research.build_corpus()

    print(f"   ✓ Sources: {len(corpus.sources)}")

    # 2. Outline (30 seconds)
    print("\n📝 Creating outline...")
    outline_gen = OutlineGenerator(corpus)
    outline = await outline_gen.generate(
        thesis="Effective climate adaptation requires multi-scale strategies",
        target_sections=3,
        target_word_count=1500
    )

    print(f"   ✓ Sections: {outline.total_sections}")

    # 3. Draft (1 minute)
    print("\n✍️  Writing draft...")
    draft_gen = DraftGenerator(outline, corpus, citation_manager)
    draft = await draft_gen.generate(citation_style=CitationStyle.APA)

    print(f"   ✓ Words: {draft.word_count:,}")
    print(f"   ✓ Citations: {draft.get_citation_count()}")

    # 4. Export (instant)
    print("\n💾 Exporting...")
    Path("my_article.md").write_text(draft.get_full_text())
    Path("bibliography.txt").write_text(draft.bibliography)

    print("\n✨ Done! Check my_article.md and bibliography.txt")

# Run it
asyncio.run(write_article())
```

**Run**:
```bash
python my_article.py
```

**Output**:
- `my_article.md` - Your complete article with citations
- `bibliography.txt` - Formatted bibliography

---

## Full Workflow (5 minutes)

For a complete research paper with revision and verification:

```python
import asyncio
from zero-g.backend.apps.nonfiction_writer import *

async def write_paper():
    # Setup
    citation_manager = CitationManager()
    research = ResearchPhase("Your Topic", citation_manager)

    # 1. Research (Ingest multiple sources)
    await research.ingest_pdf("paper1.pdf")
    await research.ingest_pdf("paper2.pdf")
    await research.ingest_url("https://example.com/article")
    corpus = await research.build_corpus()

    # 2. Outline
    outline_gen = OutlineGenerator(corpus)
    outline = await outline_gen.generate(
        thesis="Your thesis statement",
        target_sections=5,
        target_word_count=3000
    )

    # 3. Draft
    draft_gen = DraftGenerator(outline, corpus, citation_manager)
    draft = await draft_gen.generate()

    # 4. Revise (ELEGANCE strategy - 3 passes)
    revision_engine = RevisionEngine()
    result = await revision_engine.revise(draft, max_iterations=3)
    draft = result.revised_draft

    # 5. Verify
    verifier = VerificationPhase(corpus)
    report = await verifier.verify(draft)

    # 6. Export
    Path("paper.md").write_text(draft.get_full_text())
    Path("bibliography.txt").write_text(draft.bibliography)
    Path("verification_report.txt").write_text(report.get_summary())

    print("✨ Complete! Check paper.md")

asyncio.run(write_paper())
```

---

## Run the Demo

See the complete workflow in action:

```bash
cd zero-g/backend/apps/nonfiction_writer
PYTHONPATH=../../../../ python demo_writing_workflow.py
```

This demonstrates:
- ✅ Research gathering (3 sources)
- ✅ Outline generation (5 sections, 3 levels)
- ✅ Draft writing (with citations)
- ✅ Revision (ELEGANCE strategy, 3 passes)
- ✅ Verification (fact-checking)
- ✅ Export (Markdown, JSON, bibliography)

**Output files**:
```
demo_output_outline.md          # Outline in Markdown
demo_output_outline.json        # Outline structure
demo_output_draft.md            # Complete draft
demo_output_bibliography.txt    # Formatted bibliography
demo_output_verification.txt    # Verification report
```

---

## Common Use Cases

### Blog Post (Quick)

```python
async def blog_post():
    citation_manager = CitationManager()
    research = ResearchPhase("AI Safety", citation_manager)

    # Single source
    await research.ingest_url("https://example.com/ai-safety")
    corpus = await research.build_corpus()

    # Simple outline
    outline_gen = OutlineGenerator(corpus)
    outline = await outline_gen.generate(
        thesis="AI safety is crucial",
        target_sections=3,
        target_word_count=1000
    )

    # Draft
    draft_gen = DraftGenerator(outline, corpus, citation_manager)
    draft = await draft_gen.generate()

    print(draft.get_full_text())
```

### Academic Paper (Rigorous)

```python
async def academic_paper():
    citation_manager = CitationManager()
    research = ResearchPhase("Neural Architecture Search", citation_manager)

    # Multiple PDFs
    for pdf in ["paper1.pdf", "paper2.pdf", "paper3.pdf"]:
        await research.ingest_pdf(pdf)

    corpus = await research.build_corpus()

    # Detailed outline
    outline_gen = OutlineGenerator(corpus)
    outline = await outline_gen.generate(
        thesis="NAS enables automated model design",
        target_sections=6,
        max_depth=3,
        target_word_count=5000
    )

    # Validate
    validation = await outline_gen.validate(outline)

    # Draft with dense citations
    draft_gen = DraftGenerator(outline, corpus, citation_manager)
    draft = await draft_gen.generate(
        citation_style=CitationStyle.APA,
        target_density='dense'
    )

    # Multiple revision passes
    revision_engine = RevisionEngine()
    result = await revision_engine.revise(
        draft,
        strategy=RefinementStrategy.ELEGANCE,
        max_iterations=3
    )
    draft = result.revised_draft

    # Verify
    verifier = VerificationPhase(corpus)
    report = await verifier.verify(draft)

    # Must have >80% verification rate
    assert report.get_verification_rate() > 0.8

    Path("paper.md").write_text(draft.get_full_text())
```

### Magazine Article (Balanced)

```python
async def magazine_article():
    citation_manager = CitationManager()
    research = ResearchPhase("Sustainable Agriculture", citation_manager)

    # Mix of sources
    await research.ingest_pdf("study.pdf")
    await research.ingest_url("https://example.com/farming")
    await research.ingest_url("https://example.com/sustainability")

    corpus = await research.build_corpus()

    # Moderate outline
    outline_gen = OutlineGenerator(corpus)
    outline = await outline_gen.generate(
        thesis="Sustainable farming can feed the world",
        target_sections=4,
        target_word_count=2500
    )

    # Draft with moderate citations
    draft_gen = DraftGenerator(outline, corpus, citation_manager)
    draft = await draft_gen.generate(
        citation_style=CitationStyle.MLA,
        target_density='moderate'
    )

    # Single ELEGANCE pass
    revision_engine = RevisionEngine()
    result = await revision_engine.revise(draft, max_iterations=1)

    Path("article.md").write_text(result.revised_draft.get_full_text())
```

---

## Citation Styles

Change citation style easily:

```python
# APA (default)
draft = await draft_gen.generate(citation_style=CitationStyle.APA)
# Output: (Smith, 2023)

# MLA
draft = await draft_gen.generate(citation_style=CitationStyle.MLA)
# Output: (Smith 45)

# Chicago
draft = await draft_gen.generate(citation_style=CitationStyle.CHICAGO)
# Output: (Smith 2023)

# IEEE
draft = await draft_gen.generate(citation_style=CitationStyle.IEEE)
# Output: [1]
```

---

## Revision Strategies

Choose the right strategy for your needs:

```python
from zero-g.backend.apps.nonfiction_writer.revision_phase import RefinementStrategy

# ELEGANCE: Clarity → Simplicity → Beauty
result = await revision_engine.revise(
    draft,
    strategy=RefinementStrategy.ELEGANCE,
    max_iterations=3
)

# VERIFY: Accuracy → Completeness → Consistency
result = await revision_engine.revise(
    draft,
    strategy=RefinementStrategy.VERIFY,
    max_iterations=3
)

# CRITIQUE: Self-improvement
result = await revision_engine.revise(
    draft,
    strategy=RefinementStrategy.CRITIQUE,
    max_iterations=2
)

# Auto-select (let the engine choose)
result = await revision_engine.revise(draft, max_iterations=3)
```

---

## Verification

Check your work before publishing:

```python
verifier = VerificationPhase(corpus)
report = await verifier.verify(draft)

# Check overall quality
print(f"Verified: {report.get_verification_rate():.1%}")
print(f"Confidence: {report.overall_confidence:.2f}")

# Review issues
if report.issues:
    print("\n⚠ Issues:")
    for issue in report.issues:
        print(f"  - {issue}")

# Get recommendations
if report.recommendations:
    print("\n→ Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

# Check specific claims
for claim in report.claims:
    if claim.status == ClaimStatus.UNSUPPORTED:
        print(f"⚠ Unsupported: {claim.text}")
    elif claim.status == ClaimStatus.CONTRADICTED:
        print(f"❌ Contradicted: {claim.text}")
```

---

## Export Formats

Multiple export options:

```python
# Markdown (default)
markdown = draft.get_full_text()
Path("article.md").write_text(markdown)

# JSON (for programmatic use)
json_data = outline.to_json()
Path("outline.json").write_text(json_data)

# Bibliography only
bib = draft.bibliography
Path("bibliography.txt").write_text(bib)

# Specific section
section_text = draft.get_section_text("section_0001")
```

---

## Tips & Tricks

### 1. Start with Good Research
```python
# Ingest 5-10 high-quality sources
# Prioritize peer-reviewed articles
peer_reviewed = corpus.get_peer_reviewed_sources()
```

### 2. Validate Your Outline
```python
validation = await outline_gen.validate(outline)
if not validation['valid']:
    # Fix issues before drafting
    outline = await outline_gen.refine(outline, feedback)
```

### 3. Use Appropriate Citation Density
```python
# Academic papers: 'dense'
# Magazine articles: 'moderate'
# Blog posts: 'sparse'
draft = await draft_gen.generate(target_density='moderate')
```

### 4. Always Verify
```python
report = await verifier.verify(draft)
# Aim for >80% verification rate
assert report.get_verification_rate() > 0.8
```

### 5. Track Quality Improvements
```python
result = await revision_engine.revise(draft)
print(result.get_summary())
# Shows: passes, changes, quality improvement
```

---

## Next Steps

1. **Read the full README**: `README.md` for complete documentation
2. **Run the demo**: See all features in action
3. **Try the examples**: Blog post, academic paper, magazine article
4. **Explore components**: Deep dive into each module
5. **Integrate with HoloLoom**: Enable advanced features (RAG, agentic reasoning)

---

## Need Help?

- **Documentation**: See `README.md` for full reference
- **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
- **Examples**: Check `demo_writing_workflow.py`

---

## Phase 3: Advanced Features ⚡

**NEW!** Phase 3 brings professional publishing capabilities:

### Source Credibility Scoring

Evaluate research quality automatically:

```python
from zero_g.backend.apps.nonfiction_writer.source_credibility import create_credibility_scorer

scorer = create_credibility_scorer()

# Score a source
metrics = scorer.score_source(
    title="Deep Learning in Medical Imaging",
    authors=["Smith, John A.", "Jones, Mary B."],
    journal="Nature Medicine",
    year=2024,
    doi="10.1038/nm.2024.123"
)

print(f"Overall Score: {metrics.overall_score:.2f}")
print(f"Level: {metrics.credibility_level.value}")
print(f"Peer Reviewed: {metrics.is_peer_reviewed}")
```

### Zinsser Scientific Editing

Apply William Zinsser's clarity principles:

```python
from zero_g.backend.apps.nonfiction_writer.zinsser_editor import create_zinsser_editor

editor = create_zinsser_editor()

# Improve clarity and brevity
improved_text = editor.improve_all(original_text)

# Get detailed analysis
report = editor.analyze(original_text)
print(f"Clarity Score: {report.clarity_score:.2f}")
print(f"Suggestions: {len(report.all_suggestions)}")
```

### PDF Export

Professional publication-ready PDFs:

```python
from zero_g.backend.apps.nonfiction_writer.pdf_exporter import (
    create_pdf_exporter,
    PDFTemplate,
    PDFMetadata
)

exporter = create_pdf_exporter()
metadata = PDFMetadata(
    author="Your Name",
    institution="Your Institution",
    abstract="Your abstract here"
)

# Export with template
exporter.export_draft(
    draft,
    "paper.pdf",
    template=PDFTemplate.ACADEMIC,  # or MAGAZINE, BOOK, REPORT
    metadata=metadata
)
```

### Advanced Bibliography

DOI enrichment and citation networks:

```python
from zero_g.backend.apps.nonfiction_writer.advanced_bibliography import create_advanced_bibliography_manager

bib_manager = create_advanced_bibliography_manager()

# Enrich from DOI
citation = await bib_manager.enrich_from_doi("10.1038/nature12345")

# Export formats
bib_manager.export_bibtex(citations, "references.bib")
bib_manager.export_endnote(citations, "references.xml")

# Analyze citation network
network = bib_manager.analyze_citation_network(citations)
print(f"Most cited: {network['most_cited']}")
```

### Collaboration Engine

Multi-user editing with conflict resolution:

```python
from zero_g.backend.apps.nonfiction_writer.collaboration_engine import (
    create_collaboration_engine,
    User,
    ChangeType
)

collab = create_collaboration_engine()

# Create session
session = await collab.create_session("document_id")

# Users join
await collab.join_session(session.session_id, user1)
await collab.join_session(session.session_id, user2)

# Apply changes
change = await collab.apply_change(
    session.session_id,
    user1.id,
    ChangeType.INSERT,
    position=10,
    content="new text"
)

# Add comments
comment = await collab.add_comment(
    session.session_id,
    user2.id,
    position=50,
    length=10,
    text="This needs clarification"
)

# Suggest edits
suggestion = await collab.suggest_edit(
    session.session_id,
    user1.id,
    position=20,
    length=5,
    suggested_text="better wording",
    rationale="More concise"
)
```

### Run Phase 3 Demo

See all advanced features in action:

```bash
PYTHONPATH=../../../../ python demo_phase3_features.py
```

---

## What's Next?

**Phase 2** (Week 3-4): Zero-G Integration
- Lifecycle management (Preflight → Orbit)
- App docking protocol
- Safety checks

**Phase 3** ✅ **COMPLETE!** Advanced Features
- ✅ Source credibility scoring
- ✅ Zinsser scientific editing
- ✅ PDF export (Academic, Magazine, Book, Report)
- ✅ Advanced bibliography (DOI, BibTeX, EndNote)
- ✅ Collaboration engine (multi-user, conflict resolution)

---

🎉 **Start writing!** You're ready to create professional nonfiction content with automatic citations, multi-pass refinement, fact-checking, and publication-ready export.
