"""
BossPig Auto-Fixer Demo

Demonstrates the auto-fixer fixing real business slop examples.

Usage:
    python demo_auto_fixer.py

Created: 2025-11-22
"""

from fixer.auto_fixer import AutoFixer, FixStrategy
from detector.core import FindingCategory


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_jargon_fix():
    """Demo: Corporate jargon replacement"""
    print_section("Demo 1: Corporate Jargon Replacement")

    fixer = AutoFixer(strategy=FixStrategy.BATCH)

    text = """
    We need to synergize our core competencies to leverage best practices
    and optimize our bandwidth. Let's circle back on this to ensure we're
    maximizing our value proposition.
    """.strip()

    print("BEFORE:")
    print(text)
    print()

    result = fixer.fix(text, categories=[FindingCategory.CORPORATE_JARGON])

    print("AFTER:")
    print(result.fixed_text)
    print()

    print(f"Fixes Applied: {result.fixes_applied}")
    print(f"Quality Improvement: +{result.quality_improvement} points")
    print()

    print("Changes:")
    for change in result.fix_summary:
        # Replace Unicode arrow with ASCII
        change_ascii = change.replace("→", "->")
        print(f"  * {change_ascii}")


def demo_vague_commitments():
    """Demo: Vague commitment fixes"""
    print_section("Demo 2: Vague Commitments -> Specific Actions")

    fixer = AutoFixer(strategy=FixStrategy.BATCH)

    text = """
    We will try to complete the project ASAP. The team will make a best
    effort to deliver on time. We'll get back to you soon with more details.
    """.strip()

    print("BEFORE:")
    print(text)
    print()

    result = fixer.fix(text, categories=[FindingCategory.VAGUE_COMMITMENTS])

    print("AFTER:")
    print(result.fixed_text)
    print()

    print(f"Fixes Applied: {result.fixes_applied}")
    print(f"Quality Improvement: +{result.quality_improvement} points")


def demo_missing_dates():
    """Demo: Missing date fixes"""
    print_section("Demo 3: Missing Dates -> Specific Deadlines")

    fixer = AutoFixer(strategy=FixStrategy.BATCH)

    text = """
    Project Timeline:
    - Phase 1: TBD
    - Phase 2: ASAP
    - Phase 3: Soon
    - Final Delivery: To be determined
    """.strip()

    print("BEFORE:")
    print(text)
    print()

    result = fixer.fix(text, categories=[FindingCategory.MISSING_DATES])

    print("AFTER:")
    print(result.fixed_text)
    print()

    print(f"Fixes Applied: {result.fixes_applied}")
    print(f"Suggested dates are 7-14 days from today")


def demo_ai_hallucinations():
    """Demo: AI hallucination cleanup"""
    print_section("Demo 4: AI Hallucination Cleanup")

    fixer = AutoFixer(strategy=FixStrategy.BATCH)

    text = """
    As an AI language model, I cannot provide personal opinions. However,
    [INSERT MARKET ANALYSIS HERE] shows promising trends. The data suggests
    [FILL IN CONCLUSION] based on recent metrics.
    """.strip()

    print("BEFORE:")
    print(text)
    print()

    result = fixer.fix(text, categories=[FindingCategory.AI_HALLUCINATION])

    print("AFTER:")
    print(result.fixed_text)
    print()

    print(f"Fixes Applied: {result.fixes_applied}")
    print("Note: LLM boilerplate removed, placeholders flagged for review")


def demo_full_pipeline():
    """Demo: Fix all categories together"""
    print_section("Demo 5: Full Pipeline - Fix Everything")

    fixer = AutoFixer(strategy=FixStrategy.BATCH)

    text = """
    Executive Summary

    As an AI language model, I must inform you that we need to synergize
    our core competencies ASAP. The team will try to leverage best practices
    to optimize our deliverables. [INSERT TIMELINE HERE] shows we can
    deliver soon. Mistakes were made in the previous quarter, but we're
    committed to making a best effort going forward.
    """.strip()

    print("BEFORE:")
    print(text)
    print()

    # Analyze before fixing
    print("\nISSUES DETECTED:")
    from detector.detector import BossPigDetector
    detector = BossPigDetector()
    findings = detector.analyze(text)
    print(f"  • {len(findings.findings)} total issues found")
    print(f"  • Quality Score: {findings.quality_metrics.overall_score()}/100 ({findings.quality_metrics.grade()})")
    print()

    # Fix all issues
    result = fixer.fix(text)  # No categories = fix all

    print("AFTER:")
    print(result.fixed_text)
    print()

    print("RESULTS:")
    print(f"  • Fixes Applied: {result.fixes_applied}")
    print(f"  • Fixes Failed: {result.fixes_failed}")
    print(f"  • Quality Improvement: +{result.quality_improvement} points")
    print(f"  • New Score: {result.findings_after.quality_metrics.overall_score()}/100 ({result.findings_after.quality_metrics.grade()})")
    print()

    print("ALL CHANGES:")
    for i, change in enumerate(result.fix_summary, 1):
        change_ascii = change.replace("→", "->")
        print(f"  {i}. {change_ascii}")


def demo_diff_view():
    """Demo: Unified diff output"""
    print_section("Demo 6: Unified Diff View")

    fixer = AutoFixer(strategy=FixStrategy.BATCH)

    text = "We need to synergize our bandwidth ASAP to leverage core competencies."

    result = fixer.fix(text)

    print("UNIFIED DIFF:")
    print(result.get_diff())


def main():
    """Run all demos"""
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 18 + "BossPig Auto-Fixer Demo" + " " * 27 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    demo_jargon_fix()
    demo_vague_commitments()
    demo_missing_dates()
    demo_ai_hallucinations()
    demo_full_pipeline()
    demo_diff_view()

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
