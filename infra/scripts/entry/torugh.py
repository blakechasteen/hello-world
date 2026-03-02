#!/usr/bin/env python3
"""
🐷 TORUGH - The Complete Slop Detection & Classification Pipeline
=================================================================

Combines Trough (detection) + xTerminator (classification) for a complete
code quality analysis workflow.

"From pig trough to prize-winning bacon!" - The Swine Way

Usage:
    python torugh.py                           # Scan HoloLoom
    python torugh.py --dir ./my_project        # Scan custom directory
    python torugh.py --show-fixable           # Show only auto-fixable issues
"""

import sys
import asyncio
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from trough import AISlopDetector, Language, Severity, SlopIssue
from xterminator import ClassificationEngine, FixProposal


class TorughPipeline:
    """
    🐷 The complete slop-to-classification pipeline.

    Step 1: Trough detects issues (the slop detector)
    Step 2: xTerminator classifies them (the pig farmer's sorting hat)
    Step 3: Actionable report (what to fix, how to fix it)
    """

    def __init__(self):
        self.detector = AISlopDetector()
        self.classifier = ClassificationEngine()

    async def scan_file(self, file_path: str) -> Dict:
        """Scan a single file and classify all issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'issues': []
            }

        # Step 1: Detect issues with Trough
        issues = await self.detector.detect_all(code, Language.PYTHON, file_path)

        # Step 2: Classify each issue with xTerminator
        classified = []
        for issue in issues:
            try:
                proposal = await self.classifier.classify_and_propose(issue, code)
                classified.append({
                    'issue': issue,
                    'proposal': proposal
                })
            except Exception as e:
                # Skip classification errors
                classified.append({
                    'issue': issue,
                    'proposal': None,
                    'error': str(e)
                })

        return {
            'file': file_path,
            'issues': classified,
            'total': len(classified)
        }

    async def scan_directory(self, directory: str, max_files: int = 20) -> List[Dict]:
        """Scan directory with Trough + xTerminator classification."""
        import glob
        import os

        print("=" * 70)
        print("TORUGH PIPELINE - Slop Detection + Classification")
        print("  (*)< OINK! - The Pig Farmer's Code Quality Tool")
        print("=" * 70)
        print()
        print(f"Scanning: {directory}")
        print(f"Max files: {max_files}")
        print()

        # Find Python files
        py_files = glob.glob(f"{directory}/**/*.py", recursive=True)
        py_files = [f for f in py_files if '__pycache__' not in f]
        py_files = py_files[:max_files]

        print(f"Found {len(py_files)} Python files")
        print()

        # Scan each file
        results = []
        for i, file_path in enumerate(py_files):
            rel_path = file_path.replace('\\', '/').split('/')[-1]
            print(f"[{i+1}/{len(py_files)}] {rel_path}...", end=' ')

            result = await self.scan_file(file_path)

            if 'error' in result:
                print("ERROR")
            else:
                print(f"{result['total']} issues")

            results.append(result)

        return results

    def generate_report(self, results: List[Dict]) -> str:
        """Generate piglet-themed classification report."""
        # Count by strategy
        by_strategy = defaultdict(int)
        by_risk = defaultdict(int)
        auto_fixable = []
        needs_review = []
        false_positives = []

        for result in results:
            if 'error' in result:
                continue

            for classified in result['issues']:
                proposal = classified.get('proposal')
                if not proposal:
                    continue

                issue = classified['issue']
                by_strategy[proposal.fix_strategy.value] += 1
                by_risk[proposal.risk_level.value] += 1

                if proposal.safe_to_autofix:
                    auto_fixable.append({
                        'file': result['file'],
                        'issue': issue,
                        'proposal': proposal
                    })
                elif proposal.should_skip():
                    false_positives.append({
                        'file': result['file'],
                        'issue': issue,
                        'proposal': proposal
                    })
                else:
                    needs_review.append({
                        'file': result['file'],
                        'issue': issue,
                        'proposal': proposal
                    })

        # Generate report
        report = []
        report.append("=" * 70)
        report.append("TORUGH CLASSIFICATION REPORT")
        report.append("  (*)< OINK! - From Trough to Prize-Winning Code")
        report.append("=" * 70)
        report.append("")
        report.append(f"Files Scanned: {len(results)}")
        report.append(f"Total Issues: {sum(len(r.get('issues', [])) for r in results)}")
        report.append("")

        # The Pig Farmer's Wisdom
        report.append("## (*)<  The Pig Farmer's Verdict")
        report.append("")
        report.append(f"**Auto-Fixable (Prize Pigs)**: {len(auto_fixable)} issues")
        report.append(f"   >> Safe to fix automatically with high confidence")
        report.append("")
        report.append(f"**Needs Review (Piglets Need Training)**: {len(needs_review)} issues")
        report.append(f"   >> Require human judgment or more tests")
        report.append("")
        report.append(f"**False Positives (Mud, Not Slop)**: {len(false_positives)} issues")
        report.append(f"   >> Can be safely ignored")
        report.append("")

        # By strategy
        report.append("## [TOOLS] Fix Strategy Breakdown")
        report.append("")
        for strategy, count in sorted(by_strategy.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{strategy}**: {count}")
        report.append("")

        # By risk
        report.append("## [WARN] Risk Level Distribution")
        report.append("")
        for risk, count in sorted(by_risk.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{risk}**: {count}")
        report.append("")

        # Top auto-fixable
        if auto_fixable:
            report.append("## [OK] Top 10 Auto-Fixable Issues")
            report.append("")
            for i, item in enumerate(auto_fixable[:10], 1):
                rel_path = item['file'].replace('\\', '/').split('/')[-1]
                issue = item['issue']
                proposal = item['proposal']
                report.append(f"{i}. **{rel_path}:{issue.line_number}**")
                report.append(f"   - Category: {issue.category.value}")
                report.append(f"   - Strategy: {proposal.fix_strategy.value}")
                report.append(f"   - Confidence: {proposal.confidence:.3f}")
                report.append("")

        # Needs review sample
        if needs_review:
            report.append("## [REVIEW] Sample Issues Needing Review")
            report.append("")
            for i, item in enumerate(needs_review[:5], 1):
                rel_path = item['file'].replace('\\', '/').split('/')[-1]
                issue = item['issue']
                proposal = item['proposal']
                report.append(f"{i}. **{rel_path}:{issue.line_number}**")
                report.append(f"   - Category: {issue.category.value}")
                report.append(f"   - Risk: {proposal.risk_level.value}")
                report.append(f"   - Why review: {proposal.explanation[:100]}...")
                report.append("")

        report.append("=" * 70)
        report.append("(*)< May your code be clean, your tests be green,")
        report.append("     and your slop be properly classified!")
        report.append("=" * 70)

        return '\n'.join(report)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="(*)< TORUGH - Trough + xTerminator pipeline"
    )
    parser.add_argument(
        '--dir',
        default='hololoom',
        help='Directory to scan (default: HoloLoom)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=20,
        help='Max files to scan (default: 20)'
    )
    parser.add_argument(
        '--show-fixable',
        action='store_true',
        help='Show only auto-fixable issues'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = TorughPipeline()
    results = await pipeline.scan_directory(args.dir, max_files=args.max_files)

    # Generate report
    report = pipeline.generate_report(results)
    print()
    print(report)

    # Save report
    with open('TORUGH_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print()
    print("Report saved to: TORUGH_REPORT.md")


if __name__ == "__main__":
    asyncio.run(main())
