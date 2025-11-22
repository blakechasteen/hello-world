"""
BossPig Core Detection Engine Demo

Demonstrates all 5 detection categories with before/after examples.

Created: 2025-11-22
Status: Production Ready
"""

import sys
sys.path.insert(0, '..')

from bosspig.detector import BossPigDetector, QualityScorer, FindingCategory


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def analyze_and_report(detector: BossPigDetector, text: str, title: str):
    """Analyze text and print detailed report"""
    print_section(title)
    print("\n**Original Text:**")
    print("-" * 80)
    print(text)
    print("-" * 80)

    # Analyze
    results = detector.analyze(text)

    # Print summary
    print(f"\n**Quality Score:** {results.quality_score}/100 ({results.grade} - {results.grade_label})")
    print(f"\n**Findings Summary:**")
    print(f"  Critical: {results.critical_count()}")
    print(f"  Warnings: {results.warning_count()}")
    print(f"  Info: {results.info_count()}")

    # Print findings by category
    if results.findings:
        print(f"\n**Detailed Findings:**")
        for category in FindingCategory:
            category_findings = results.findings_by_category(category)
            if category_findings:
                print(f"\n{category.value.upper().replace('_', ' ')} ({len(category_findings)} issues):")
                for finding in category_findings[:3]:  # Show first 3
                    severity_icon = {
                        "critical": "[!]",
                        "warning": "[~]",
                        "info": "[i]"
                    }.get(finding.severity.value, "[ ]")

                    print(f"  {severity_icon} Line {finding.line_number}: \"{finding.text}\"")
                    print(f"      => {finding.suggestion}")

    # Print score report
    print(f"\n**Score Report:**")
    report = QualityScorer.generate_score_report(results, include_recommendations=True)
    print(report)


def main():
    """Run all demo examples"""
    print_section("BossPig Core Detection Engine - Demo")
    print("\nDemonstrating TOP 5 detection categories:")
    print("1. Corporate Jargon Detection")
    print("2. Vague Commitments Detection")
    print("3. Missing Dates/Deadlines Detection")
    print("4. AI Hallucination Markers Detection")
    print("5. Passive Voice Detection")

    # Create detector
    detector = BossPigDetector()

    # Example 1: Corporate Jargon Overload
    jargon_text = """
# Q4 Strategic Initiative

We need to synergize our core competencies to leverage the low-hanging fruit
and move the needle on our key performance indicators. This paradigm shift
will be a game changer for our thought leadership in the space.

Action Items:
- Circle back on the deep dive
- Touch base with stakeholders
- Reach out to the team
"""
    analyze_and_report(detector, jargon_text, "Example 1: Corporate Jargon Overload")

    # Example 2: Vague Commitments
    vague_text = """
# Project Update

We will try to finish the migration soon. Hopefully we can get it done
before the end of the quarter. It would be nice if we could avoid downtime.
We should probably test it first.

Progress:
- Some work has been completed
- A few issues were identified
- The team is working on it
"""
    analyze_and_report(detector, vague_text, "Example 2: Vague Commitments")

    # Example 3: AI Hallucination
    ai_text = """
# Product Specification

As an AI language model, I don't have personal opinions, but here are some
features that could be implemented:

[INSERT PRODUCT NAME HERE]
- Feature 1: [TO BE DETERMINED]
- Feature 2: TBD
- Feature 3: See Section 5 for details

Studies show that users prefer intuitive interfaces. Many experts believe
that performance is important.
"""
    analyze_and_report(detector, ai_text, "Example 3: AI Hallucination & Placeholders")

    # Example 4: Clean Text (High Score)
    clean_text = """
# Q4 Strategic Initiative

@alice will combine our product development and marketing strengths
to achieve easy wins and improve our customer satisfaction scores by 15%.
This major change will establish our expertise in customer experience.

Action Items:
- @alice: Review customer feedback analysis by Dec 1, 2025
- @bob: Meet with product team on Nov 25, 2025 to discuss roadmap
- @carol: Send email to engineering leads by Nov 23, 2025 with timeline
"""
    analyze_and_report(detector, clean_text, "Example 4: Clean, Specific Text (Good Score)")

    # Example 5: Mixed Issues (Multiple Categories)
    mixed_text = """
# Strategic Plan

We will try to leverage our bandwidth to synergize with stakeholders.
ASAP, we need to circle back on the paradigm shift. The decision was made
by the team, and mistakes were identified during the process.

As an AI language model, I suggest: [INSERT RECOMMENDATIONS HERE]
"""
    analyze_and_report(detector, mixed_text, "Example 5: Mixed Issues (All 5 Categories)")

    # Final summary
    print_section("Demo Complete")
    print("\nBossPig successfully detected and classified business slop across 5 categories:")
    print("  [!] Corporate jargon => Replace with plain language")
    print("  [!] Vague commitments => Add specific owner and deadline")
    print("  [~] Missing dates => Specify exact dates (YYYY-MM-DD)")
    print("  [!] AI hallucinations => Remove LLM boilerplate, fill placeholders")
    print("  [~] Passive voice => Convert to active voice with owner")
    print("\nWeek 1-3 MVP: COMPLETE")
    print("Next: Week 4 - Auto-Fixer Implementation")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
