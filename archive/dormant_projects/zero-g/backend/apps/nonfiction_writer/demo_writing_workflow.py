"""
End-to-End Nonfiction Writing Workflow Demo

Demonstrates the complete writing pipeline from research to publication:
Research → Outline → Draft → Revise → Verify → Export

This demo shows:
1. Research phase: Ingesting documents and building corpus
2. Outline phase: Generating hierarchical structure
3. Draft phase: Creating initial draft with citations
4. Revision phase: Multi-pass refinement (ELEGANCE strategy)
5. Verification phase: Fact-checking all claims
6. Export: Markdown, JSON, and bibliography

Run with: PYTHONPATH=. python zero-g/backend/apps/nonfiction_writer/demo_writing_workflow.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from zero-g.backend.apps.nonfiction_writer import (
    CitationManager,
    CitationStyle,
    ResearchPhase,
    OutlineGenerator,
    OutlineStyle,
    DraftGenerator,
    RevisionEngine,
    VerificationPhase,
)


async def main():
    """Run complete writing workflow demo"""

    print("=" * 70)
    print("NONFICTION WRITING TOOLKIT - END-TO-END DEMO")
    print("=" * 70)
    print()

    # ========================================================================
    # Phase 1: Research
    # ========================================================================

    print("📚 PHASE 1: RESEARCH")
    print("-" * 70)

    topic = "Climate Change Adaptation Strategies"
    print(f"Topic: {topic}\n")

    # Initialize citation manager (shared across all phases)
    citation_manager = CitationManager()

    # Create research phase
    research = ResearchPhase(topic, citation_manager)

    # Simulate ingesting research documents
    print("Ingesting research sources...")

    # In real usage, would be: await research.ingest_pdf("paper1.pdf")
    # For demo, we'll simulate with synthetic sources
    source1 = await research.ingest_url("https://example.com/adaptation-overview")
    print(f"  ✓ Ingested: {source1.citation.title if source1.citation else 'Source 1'}")

    source2 = await research.ingest_url("https://example.com/urban-planning")
    print(f"  ✓ Ingested: {source2.citation.title if source2.citation else 'Source 2'}")

    source3 = await research.ingest_url("https://example.com/infrastructure")
    print(f"  ✓ Ingested: {source3.citation.title if source3.citation else 'Source 3'}")

    print()

    # Build research corpus
    print("Building research corpus...")
    corpus = await research.build_corpus([
        "What are the main adaptation strategies?",
        "What evidence supports these approaches?",
        "What are current limitations?"
    ])

    print(f"  ✓ Total sources: {len(corpus.sources)}")
    print(f"  ✓ Total words: {corpus.get_total_word_count():,}")
    print(f"  ✓ Peer-reviewed: {len(corpus.get_peer_reviewed_sources())}")
    print(f"  ✓ Themes identified: {', '.join(corpus.themes)}")
    print()

    # ========================================================================
    # Phase 2: Outline
    # ========================================================================

    print("📝 PHASE 2: OUTLINE GENERATION")
    print("-" * 70)

    # Create outline generator
    outline_gen = OutlineGenerator(corpus)

    # Generate outline
    thesis = "Effective climate adaptation requires integrated multi-scale strategies"
    print(f"Thesis: {thesis}\n")

    print("Generating outline...")
    outline = await outline_gen.generate(
        thesis=thesis,
        target_sections=5,
        max_depth=3,
        target_word_count=3000,
        style=OutlineStyle.ROMAN
    )

    print(f"  ✓ Sections: {outline.total_sections}")
    print(f"  ✓ Max depth: {outline.get_max_depth()}")
    print(f"  ✓ Est. word count: {outline.get_total_word_count():,}")
    print()

    # Validate outline
    print("Validating outline...")
    validation = await outline_gen.validate(outline)

    print(f"  ✓ Valid: {validation['valid']}")
    print(f"  ✓ Coverage score: {validation['coverage_score']:.1%}")
    print(f"  ✓ Balance score: {validation['balance_score']:.1%}")

    if validation['issues']:
        print("\n  Issues:")
        for issue in validation['issues']:
            print(f"    ⚠ {issue}")

    if validation['suggestions']:
        print("\n  Suggestions:")
        for sugg in validation['suggestions']:
            print(f"    → {sugg}")

    print()

    # Show outline structure
    print("Outline structure (first 2 sections):")
    for i, section in enumerate(outline.sections[:2]):
        print(f"\n  {section.number}. {section.title}")
        print(f"     {section.description}")
        print(f"     Est. {section.estimated_word_count} words")

        for subsection in section.subsections[:2]:
            print(f"\n     {subsection.number}. {subsection.title}")
            print(f"        {subsection.description}")

    print("\n  ... (3 more sections)\n")

    # ========================================================================
    # Phase 3: Draft Generation
    # ========================================================================

    print("✍️  PHASE 3: DRAFT GENERATION")
    print("-" * 70)

    # Create draft generator
    draft_gen = DraftGenerator(outline, corpus, citation_manager)

    # Generate draft
    print("Generating draft with citations...")
    draft = await draft_gen.generate(
        citation_style=CitationStyle.APA,
        target_density='moderate'
    )

    print(f"  ✓ Paragraphs: {len(draft.paragraphs)}")
    print(f"  ✓ Word count: {draft.word_count:,}")
    print(f"  ✓ Citations: {draft.get_citation_count()}")
    print(f"  ✓ Source coverage: {draft.get_source_coverage():.1%}")
    print()

    # Show sample paragraphs
    print("Sample paragraphs (first 2):")
    for i, para in enumerate(draft.paragraphs[:2]):
        print(f"\n  Paragraph {i+1} (Section {para.section_id}):")
        print(f"  {para.content[:200]}...")
        print(f"  [{para.word_count} words, {len(para.citations)} citations]")

    print()

    # ========================================================================
    # Phase 4: Revision
    # ========================================================================

    print("🔧 PHASE 4: REVISION")
    print("-" * 70)

    # Create revision engine
    revision_engine = RevisionEngine()

    # Revise draft
    print("Applying ELEGANCE refinement (3 passes)...")
    from zero-g.backend.apps.nonfiction_writer.revision_phase import RefinementStrategy

    revision_result = await revision_engine.revise(
        draft,
        strategy=RefinementStrategy.ELEGANCE,
        max_iterations=3,
        quality_threshold=0.9
    )

    print()
    print(revision_result.get_summary())
    print()

    # Update draft to revised version
    draft = revision_result.revised_draft

    # ========================================================================
    # Phase 5: Verification
    # ========================================================================

    print("✓ PHASE 5: VERIFICATION")
    print("-" * 70)

    # Create verification phase
    verifier = VerificationPhase(corpus)

    # Verify draft
    print("Fact-checking all claims...")
    verification_report = await verifier.verify(draft)

    print()
    print(verification_report.get_summary())
    print()

    # ========================================================================
    # Phase 6: Export
    # ========================================================================

    print("💾 PHASE 6: EXPORT")
    print("-" * 70)

    # Export outline
    print("\n1. Exporting outline...")
    outline_md = outline.to_markdown()
    outline_path = Path("demo_output_outline.md")
    outline_path.write_text(outline_md)
    print(f"   ✓ Saved: {outline_path}")

    outline_json = outline.to_json()
    outline_json_path = Path("demo_output_outline.json")
    outline_json_path.write_text(outline_json)
    print(f"   ✓ Saved: {outline_json_path}")

    # Export draft
    print("\n2. Exporting draft...")
    draft_text = draft.get_full_text()
    draft_path = Path("demo_output_draft.md")
    draft_path.write_text(draft_text)
    print(f"   ✓ Saved: {draft_path}")
    print(f"   ✓ Word count: {draft.word_count:,}")

    # Export bibliography
    print("\n3. Exporting bibliography...")
    bib_path = Path("demo_output_bibliography.txt")
    bib_path.write_text(draft.bibliography)
    print(f"   ✓ Saved: {bib_path}")
    print(f"   ✓ Citations: {draft.get_citation_count()}")

    # Export verification report
    print("\n4. Exporting verification report...")
    report_text = verification_report.get_summary()
    report_path = Path("demo_output_verification.txt")
    report_path.write_text(report_text)
    print(f"   ✓ Saved: {report_path}")
    print(f"   ✓ Verification rate: {verification_report.get_verification_rate():.1%}")

    # ========================================================================
    # Summary
    # ========================================================================

    print()
    print("=" * 70)
    print("✨ WORKFLOW COMPLETE!")
    print("=" * 70)
    print()
    print("Pipeline Summary:")
    print(f"  Research    → {len(corpus.sources)} sources, {corpus.get_total_word_count():,} words")
    print(f"  Outline     → {outline.total_sections} sections, {outline.get_max_depth()} levels deep")
    print(f"  Draft       → {len(draft.paragraphs)} paragraphs, {draft.word_count:,} words")
    print(f"  Revision    → {revision_result.total_changes} improvements, +{revision_result.improvement:.2%} quality")
    print(f"  Verify      → {verification_report.get_verification_rate():.1%} claims verified")
    print()
    print("Output Files:")
    print(f"  • {outline_path}")
    print(f"  • {outline_json_path}")
    print(f"  • {draft_path}")
    print(f"  • {bib_path}")
    print(f"  • {report_path}")
    print()
    print("Next Steps:")
    print("  1. Review verification report for unsupported claims")
    print("  2. Address any issues identified during verification")
    print("  3. Apply additional revision passes if needed")
    print("  4. Export to final format (PDF, DOCX, etc.)")
    print()
    print("🎉 Your nonfiction piece is ready for publication!")
    print()


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
