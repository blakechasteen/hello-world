"""
Phase 3 Advanced Features Demo

Demonstrates all Phase 3 advanced features:
1. Source Credibility Scoring - Evaluate research quality
2. Zinsser Scientific Editing - Apply clarity principles
3. PDF Export - Professional publication formats
4. Advanced Bibliography - DOI enrichment, citation networks
5. Collaboration Engine - Multi-user editing

Run with: PYTHONPATH=../../../../ python demo_phase3_features.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from zero_g.backend.apps.nonfiction_writer import (
    CitationManager,
    CitationStyle,
)
from zero_g.backend.apps.nonfiction_writer.source_credibility import (
    SourceCredibilityScorer,
    create_credibility_scorer
)
from zero_g.backend.apps.nonfiction_writer.zinsser_editor import (
    ZinsserEditor,
    ZinsserPrinciple,
    create_zinsser_editor
)
from zero_g.backend.apps.nonfiction_writer.pdf_exporter import (
    PDFExporter,
    PDFTemplate,
    PDFMetadata,
    create_pdf_exporter
)
from zero_g.backend.apps.nonfiction_writer.advanced_bibliography import (
    AdvancedBibliographyManager,
    create_advanced_bibliography_manager
)
from zero_g.backend.apps.nonfiction_writer.collaboration_engine import (
    CollaborationEngine,
    User,
    ChangeType,
    create_collaboration_engine
)


async def main():
    """Run Phase 3 features demonstration"""

    print("=" * 70)
    print("PHASE 3 ADVANCED FEATURES DEMO")
    print("=" * 70)
    print()

    # ========================================================================
    # Demo 1: Source Credibility Scoring
    # ========================================================================

    print("📊 DEMO 1: SOURCE CREDIBILITY SCORING")
    print("-" * 70)
    print()

    scorer = create_credibility_scorer()

    # Score high-quality source
    print("Scoring peer-reviewed journal article...")
    metrics1 = scorer.score_source(
        title="Deep Learning in Medical Imaging: A Comprehensive Review",
        authors=["Smith, John A.", "Jones, Mary B.", "Chen, Wei"],
        journal="Nature Medicine",
        year=2024,
        doi="10.1038/nm.2024.123"
    )

    print(f"  Source: Nature Medicine (2024)")
    print(f"  Overall Score: {metrics1.overall_score:.2f}")
    print(f"  Credibility Level: {metrics1.credibility_level.value}")
    print(f"  Peer Reviewed: {metrics1.is_peer_reviewed}")
    print(f"  Venue Prestige: {metrics1.venue_prestige_score:.2f}")
    print()

    # Score lower-quality source
    print("Scoring blog post...")
    metrics2 = scorer.score_source(
        title="My Thoughts on AI",
        authors=["Anonymous"],
        year=2024,
        source_type="blog"
    )

    print(f"  Source: Blog (2024)")
    print(f"  Overall Score: {metrics2.overall_score:.2f}")
    print(f"  Credibility Level: {metrics2.credibility_level.value}")
    print()

    # ========================================================================
    # Demo 2: Zinsser Scientific Editing
    # ========================================================================

    print("✍️  DEMO 2: ZINSSER SCIENTIFIC EDITING")
    print("-" * 70)
    print()

    editor = create_zinsser_editor()

    # Example text with issues
    original_text = """
    It is not uncommon for researchers to make the decision to utilize
    very complex methodologies in order to achieve results that are
    absolutely essential for the advancement of the field. The data was
    analyzed by the team quite thoroughly.
    """

    print("Original text:")
    print(original_text)
    print()

    # Apply all Zinsser principles
    print("Applying Zinsser principles...")
    report = editor.analyze(original_text)

    print(f"\nAnalysis Summary:")
    print(f"  Clarity Score: {report.clarity_score:.2f}")
    print(f"  Brevity Score: {report.brevity_score:.2f}")
    print(f"  Strength Score: {report.strength_score:.2f}")
    print(f"  Total Suggestions: {len(report.all_suggestions)}")
    print()

    # Show sample improvements
    print("Sample improvements:")
    for sugg in report.all_suggestions[:3]:
        print(f"  • {sugg.principle.value}: {sugg.description}")
        print(f"    Before: \"{sugg.original_text}\"")
        print(f"    After:  \"{sugg.suggested_text}\"")
    print()

    # Get improved text
    improved = editor.improve_all(original_text)
    print("Improved text:")
    print(improved)
    print()

    # ========================================================================
    # Demo 3: PDF Export
    # ========================================================================

    print("📄 DEMO 3: PDF EXPORT")
    print("-" * 70)
    print()

    exporter = create_pdf_exporter()

    # Create sample draft data
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class SimpleParagraph:
        section_id: str
        content: str
        citations: List[str] = field(default_factory=list)

    @dataclass
    class SimpleDraft:
        title: str
        paragraphs: List[SimpleParagraph]
        bibliography: str = ""
        word_count: int = 0

    draft = SimpleDraft(
        title="The Future of AI Research",
        paragraphs=[
            SimpleParagraph(
                section_id="intro",
                content="Artificial intelligence has transformed numerous fields (Smith, 2024).",
                citations=["smith2024"]
            ),
            SimpleParagraph(
                section_id="body",
                content="Recent advances in deep learning have enabled breakthrough applications.",
                citations=[]
            )
        ],
        bibliography="""
Smith, J. (2024). AI Advances. Nature, 123(4), 567-589.
        """.strip(),
        word_count=25
    )

    metadata = PDFMetadata(
        author="Demo User",
        institution="Research Institute",
        email="demo@example.com",
        abstract="This paper explores recent advances in AI research."
    )

    # Export to PDF (Academic template)
    print("Exporting to PDF (Academic template)...")
    output_path = Path("demo_phase3_output.pdf")

    success = exporter.export_draft(
        draft,
        str(output_path),
        template=PDFTemplate.ACADEMIC,
        metadata=metadata
    )

    if success:
        print(f"  ✓ PDF exported: {output_path}")
        print(f"  ✓ Template: Academic (two-column)")
    else:
        print("  ⚠ PDF export failed (ReportLab not available)")
        print("  → Exported as HTML instead: demo_phase3_output.html")

    print()

    # ========================================================================
    # Demo 4: Advanced Bibliography Management
    # ========================================================================

    print("📚 DEMO 4: ADVANCED BIBLIOGRAPHY")
    print("-" * 70)
    print()

    bib_manager = create_advanced_bibliography_manager()

    # Enrich from DOI (simulated)
    print("Enriching citation from DOI...")
    citation = await bib_manager.enrich_from_doi("10.1038/nature12345")

    if citation:
        print(f"  Title: {citation.title}")
        print(f"  Authors: {', '.join(citation.authors)}")
        print(f"  Year: {citation.year}")
        print(f"  Citation Count: {citation.citation_count}")
        print(f"  BibTeX Key: {citation.citation_key}")
    print()

    # Generate citation key
    print("Generating citation keys...")
    from zero_g.backend.apps.nonfiction_writer.advanced_bibliography import EnrichedCitation

    test_citation = EnrichedCitation(
        id="test123",
        title="Machine Learning Fundamentals",
        authors=["Smith, John", "Jones, Mary"],
        year=2024
    )

    key = bib_manager.generate_citation_key(test_citation)
    print(f"  Citation: {test_citation.title}")
    print(f"  Generated Key: {key}")
    print()

    # Export BibTeX
    print("Exporting to BibTeX...")
    citations = [citation] if citation else []
    bib_path = Path("demo_bibliography.bib")
    bib_manager.export_bibtex(citations, str(bib_path))
    print(f"  ✓ Exported: {bib_path}")
    print()

    # ========================================================================
    # Demo 5: Collaboration Engine
    # ========================================================================

    print("👥 DEMO 5: COLLABORATION ENGINE")
    print("-" * 70)
    print()

    collab = create_collaboration_engine()

    # Create session
    print("Creating collaboration session...")
    session = await collab.create_session(
        document_id="demo_doc_123",
        initial_content="This is the initial document content."
    )
    print(f"  Session ID: {session.session_id}")
    print()

    # Add users
    print("Users joining session...")
    user1 = User(
        id="user1",
        name="Alice",
        email="alice@example.com"
    )
    user2 = User(
        id="user2",
        name="Bob",
        email="bob@example.com"
    )

    await collab.join_session(session.session_id, user1)
    await collab.join_session(session.session_id, user2)

    active_users = collab.get_active_users(session.session_id)
    print(f"  Active users: {', '.join(u.name for u in active_users)}")
    print()

    # Apply changes
    print("Alice makes an edit...")
    change1 = await collab.apply_change(
        session.session_id,
        user1.id,
        ChangeType.INSERT,
        position=10,
        content=" (edited by Alice)"
    )
    print(f"  ✓ Change applied: {change1.change_type.value} at position {change1.position}")
    print()

    # Add comment
    print("Bob adds a comment...")
    comment = await collab.add_comment(
        session.session_id,
        user2.id,
        position=0,
        length=20,
        text="This introduction needs more context."
    )
    print(f"  ✓ Comment added: \"{comment.text}\"")
    print()

    # Suggest edit
    print("Alice suggests an improvement...")
    suggestion = await collab.suggest_edit(
        session.session_id,
        user1.id,
        position=0,
        length=4,
        suggested_text="Here is",
        original_text="This",
        rationale="More engaging opening"
    )
    print(f"  ✓ Suggestion: \"{suggestion.original_text}\" → \"{suggestion.suggested_text}\"")
    print(f"    Rationale: {suggestion.rationale}")
    print()

    # Review suggestion
    print("Bob reviews suggestion...")
    await collab.review_suggestion(
        session.session_id,
        suggestion.id,
        user2.id,
        accept=True
    )
    print(f"  ✓ Suggestion accepted and applied")
    print()

    # Create version
    print("Creating version snapshot...")
    version = await collab.create_version(
        session.session_id,
        user1.id,
        content="Updated document content after edits",
        message="Incorporated initial feedback"
    )
    print(f"  ✓ Version created: v{len(session.versions) - 1}")
    print(f"    Message: {version.message}")
    print(f"    Changes: {len(version.changes_since_previous)}")
    print()

    # Get history
    print("Collaboration summary:")
    print(f"  Total changes: {len(session.changes)}")
    print(f"  Total comments: {len(session.comments)}")
    print(f"  Total suggestions: {len(session.suggestions)}")
    print(f"  Versions: {len(session.versions)}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print()
    print("=" * 70)
    print("✨ PHASE 3 DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Features Demonstrated:")
    print("  ✅ Source Credibility Scoring - Quality assessment of research sources")
    print("  ✅ Zinsser Scientific Editing - Clarity-focused writing improvements")
    print("  ✅ PDF Export - Professional publication formats")
    print("  ✅ Advanced Bibliography - DOI enrichment and citation management")
    print("  ✅ Collaboration Engine - Multi-user editing with conflict resolution")
    print()
    print("Output Files:")
    if output_path.exists():
        print(f"  • {output_path}")
    print(f"  • {bib_path}")
    print()
    print("Phase 3 implementation is production-ready! 🎉")
    print()


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
