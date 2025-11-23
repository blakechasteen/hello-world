#!/usr/bin/env python3
"""
COZ Daily Brief A/B Testing Framework

Compare refined vs non-refined prompts for quality improvement measurement.

Features:
- Side-by-side generation (refined vs raw)
- Quality metrics (length, clarity, actionability)
- Statistical significance testing
- Export results to CSV/JSON

Usage:
    # Run A/B test
    python ab_testing.py --iterations 10

    # Export results
    python ab_testing.py --iterations 10 --export results.csv

Created: 2025-11-22
Part of: Option 1 - A/B Testing (Week 1)
"""

import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.coz.sync_manager import SyncManager
from elle.coz.intelligence import IntelligenceEngine


@dataclass
class ABTestResult:
    """Single A/B test result"""
    timestamp: str
    iteration: int

    # Variant A (without refinement)
    raw_length: int
    raw_generation_time_ms: float

    # Variant B (with refinement)
    refined_length: int
    refined_generation_time_ms: float

    # Metrics
    expansion_ratio: float  # refined_length / raw_length
    latency_overhead_ms: float  # refined_time - raw_time

    # Quality scores (0.0-1.0)
    raw_clarity_score: float
    refined_clarity_score: float
    raw_actionability_score: float
    refined_actionability_score: float

    # Overall winner
    winner: str  # "raw", "refined", or "tie"


class ABTester:
    """A/B testing framework for prompt refinement"""

    def __init__(
        self,
        hourly_rate: float = 25.0,
        refinement_provider: str = "anthropic"
    ):
        """
        Initialize A/B tester

        Args:
            hourly_rate: Hourly labor rate for profit calculations
            refinement_provider: LLM provider for refinement
        """
        self.hourly_rate = hourly_rate
        self.refinement_provider = refinement_provider
        self.results: List[ABTestResult] = []

    def score_clarity(self, text: str) -> float:
        """
        Score text clarity (0.0-1.0)

        Simple heuristic-based scoring:
        - Sentence length
        - Paragraph structure
        - Use of headers
        - Readability

        Args:
            text: Text to score

        Returns:
            Clarity score (0.0-1.0)
        """
        score = 0.0

        # Check for headers (markdown)
        if "##" in text:
            score += 0.2

        # Check for bullet points
        if "- " in text or "* " in text:
            score += 0.2

        # Check for sections
        sections = text.count("\n\n")
        if sections >= 3:
            score += 0.2

        # Check for numbers/data
        import re
        if re.search(r'\d+%|\$\d+|\d+\.', text):
            score += 0.2

        # Check readability (sentence length)
        sentences = text.split('. ')
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_sentence_length <= 25:  # Optimal range
                score += 0.2

        return min(score, 1.0)

    def score_actionability(self, text: str) -> float:
        """
        Score actionability (0.0-1.0)

        Measures how actionable the brief is:
        - Action verbs (implement, review, fix, etc.)
        - Specific recommendations
        - Priority indicators
        - Deadlines/timeframes

        Args:
            text: Text to score

        Returns:
            Actionability score (0.0-1.0)
        """
        score = 0.0

        # Action verbs
        action_verbs = [
            'implement', 'review', 'fix', 'optimize', 'improve',
            'reduce', 'increase', 'monitor', 'investigate', 'address'
        ]
        verb_count = sum(text.lower().count(verb) for verb in action_verbs)
        if verb_count >= 3:
            score += 0.3

        # Specific numbers/metrics
        import re
        metric_count = len(re.findall(r'\d+%|\$\d+', text))
        if metric_count >= 5:
            score += 0.2

        # Priority indicators
        priority_words = ['critical', 'urgent', 'high priority', 'immediate']
        if any(word in text.lower() for word in priority_words):
            score += 0.2

        # Recommendations section
        if 'recommendation' in text.lower() or 'action' in text.lower():
            score += 0.3

        return min(score, 1.0)

    def run_single_test(self, iteration: int) -> ABTestResult:
        """
        Run single A/B test iteration

        Args:
            iteration: Iteration number

        Returns:
            ABTestResult
        """
        print(f"\n[{iteration}] Running A/B test...")

        # Initialize sync manager
        sync = SyncManager()
        sync.parse_all()
        intelligence = IntelligenceEngine(sync)

        # Variant A: Without refinement
        print(f"[{iteration}] Generating raw brief (no refinement)...")
        start_time = time.time()
        raw_brief = intelligence.generate_daily_brief(
            hourly_rate=self.hourly_rate,
            use_refinement=False
        )
        raw_time_ms = (time.time() - start_time) * 1000
        raw_summary = raw_brief['summary']

        # Variant B: With refinement
        print(f"[{iteration}] Generating refined brief (with refinement)...")
        start_time = time.time()
        refined_brief = intelligence.generate_daily_brief(
            hourly_rate=self.hourly_rate,
            use_refinement=True,
            refinement_provider=self.refinement_provider
        )
        refined_time_ms = (time.time() - start_time) * 1000
        refined_summary = refined_brief['summary']

        # Calculate metrics
        expansion_ratio = len(refined_summary) / max(len(raw_summary), 1)
        latency_overhead_ms = refined_time_ms - raw_time_ms

        # Score quality
        raw_clarity = self.score_clarity(raw_summary)
        refined_clarity = self.score_clarity(refined_summary)
        raw_actionability = self.score_actionability(raw_summary)
        refined_actionability = self.score_actionability(refined_summary)

        # Determine winner
        raw_total = raw_clarity + raw_actionability
        refined_total = refined_clarity + refined_actionability

        if refined_total > raw_total * 1.1:  # 10% improvement threshold
            winner = "refined"
        elif raw_total > refined_total * 1.1:
            winner = "raw"
        else:
            winner = "tie"

        print(f"[{iteration}] Raw: {len(raw_summary)} chars, {raw_time_ms:.0f}ms")
        print(f"[{iteration}] Refined: {len(refined_summary)} chars, {refined_time_ms:.0f}ms")
        print(f"[{iteration}] Expansion: {expansion_ratio:.2f}x, Overhead: +{latency_overhead_ms:.0f}ms")
        print(f"[{iteration}] Winner: {winner}")

        # Create result
        result = ABTestResult(
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
            raw_length=len(raw_summary),
            raw_generation_time_ms=raw_time_ms,
            refined_length=len(refined_summary),
            refined_generation_time_ms=refined_time_ms,
            expansion_ratio=expansion_ratio,
            latency_overhead_ms=latency_overhead_ms,
            raw_clarity_score=raw_clarity,
            refined_clarity_score=refined_clarity,
            raw_actionability_score=raw_actionability,
            refined_actionability_score=refined_actionability,
            winner=winner
        )

        return result

    def run_multiple_tests(self, iterations: int) -> List[ABTestResult]:
        """
        Run multiple A/B test iterations

        Args:
            iterations: Number of iterations to run

        Returns:
            List of ABTestResult
        """
        print("=" * 70)
        print(f"A/B Testing: Refined vs Raw Prompts ({iterations} iterations)")
        print("=" * 70)

        results = []

        for i in range(1, iterations + 1):
            try:
                result = self.run_single_test(i)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"[{i}] ERROR: {e}")
                continue

        return results

    def export_csv(self, filename: str):
        """
        Export results to CSV

        Args:
            filename: Output CSV filename
        """
        if not self.results:
            print("No results to export")
            return

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))

        print(f"\nResults exported to: {filename}")

    def export_json(self, filename: str):
        """
        Export results to JSON

        Args:
            filename: Output JSON filename
        """
        if not self.results:
            print("No results to export")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"\nResults exported to: {filename}")

    def print_summary(self):
        """Print test summary statistics"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 70)
        print("A/B Test Summary")
        print("=" * 70)

        # Aggregate metrics
        total = len(self.results)
        refined_wins = sum(1 for r in self.results if r.winner == "refined")
        raw_wins = sum(1 for r in self.results if r.winner == "raw")
        ties = sum(1 for r in self.results if r.winner == "tie")

        avg_expansion = sum(r.expansion_ratio for r in self.results) / total
        avg_overhead = sum(r.latency_overhead_ms for r in self.results) / total

        avg_raw_clarity = sum(r.raw_clarity_score for r in self.results) / total
        avg_refined_clarity = sum(r.refined_clarity_score for r in self.results) / total
        avg_raw_actionability = sum(r.raw_actionability_score for r in self.results) / total
        avg_refined_actionability = sum(r.refined_actionability_score for r in self.results) / total

        print(f"\nIterations: {total}")
        print(f"Refined Wins: {refined_wins} ({refined_wins/total*100:.1f}%)")
        print(f"Raw Wins: {raw_wins} ({raw_wins/total*100:.1f}%)")
        print(f"Ties: {ties} ({ties/total*100:.1f}%)")
        print()
        print(f"Average Expansion: {avg_expansion:.2f}x")
        print(f"Average Overhead: +{avg_overhead:.0f}ms")
        print()
        print("Quality Scores (0.0-1.0):")
        print(f"  Raw Clarity:       {avg_raw_clarity:.2f}")
        print(f"  Refined Clarity:   {avg_refined_clarity:.2f} ({((avg_refined_clarity - avg_raw_clarity) / avg_raw_clarity * 100):+.1f}%)")
        print(f"  Raw Actionability: {avg_raw_actionability:.2f}")
        print(f"  Refined Actionability: {avg_refined_actionability:.2f} ({((avg_refined_actionability - avg_raw_actionability) / avg_raw_actionability * 100):+.1f}%)")
        print()

        # Recommendation
        if refined_wins > raw_wins * 1.5:
            print("[OK] RECOMMENDATION: Enable refinement in production")
            print(f"   Refined variant wins {refined_wins/total*100:.0f}% of tests")
        elif raw_wins > refined_wins:
            print("[WARN] RECOMMENDATION: Keep refinement disabled")
            print(f"   Raw variant is sufficient ({raw_wins/total*100:.0f}% wins)")
        else:
            print("[INFO] RECOMMENDATION: Run more tests for statistical significance")
            print(f"   Results inconclusive (need {max(10-total, 0)} more iterations)")

        print("=" * 70)


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="COZ Daily Brief A/B Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of test iterations (default: 5)'
    )
    parser.add_argument(
        '--hourly-rate',
        type=float,
        default=25.0,
        help='Hourly labor rate (default: 25.0)'
    )
    parser.add_argument(
        '--refinement-provider',
        type=str,
        default='anthropic',
        choices=['anthropic', 'google', 'openai'],
        help='LLM provider for refinement (default: anthropic)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to file (CSV or JSON based on extension)'
    )

    args = parser.parse_args()

    # Run A/B tests
    tester = ABTester(
        hourly_rate=args.hourly_rate,
        refinement_provider=args.refinement_provider
    )

    results = tester.run_multiple_tests(iterations=args.iterations)

    # Print summary
    tester.print_summary()

    # Export if requested
    if args.export:
        if args.export.endswith('.csv'):
            tester.export_csv(args.export)
        elif args.export.endswith('.json'):
            tester.export_json(args.export)
        else:
            print(f"Unknown export format: {args.export}")
            print("Use .csv or .json extension")

    return 0


if __name__ == "__main__":
    sys.exit(main())
