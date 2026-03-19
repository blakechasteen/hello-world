"""
Phase 3: Production Validation - Human Evaluation Framework

This module provides infrastructure for collecting human evaluations comparing traditional
vs UnifiedMRF prompts through blind side-by-side comparisons.

Created: November 2025
Status: Production validation framework
"""

import asyncio
import json
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EvaluationCriterion(Enum):
    """Criteria for human evaluation."""
    ACCURACY = "accuracy"  # Factual correctness
    COMPLETENESS = "completeness"  # Addresses all aspects
    CLARITY = "clarity"  # Clear and easy to understand
    RELEVANCE = "relevance"  # Relevant to question
    CONCISENESS = "conciseness"  # Not overly verbose
    OVERALL = "overall"  # Overall quality


class Preference(Enum):
    """Human preference between responses."""
    STRONGLY_PREFER_A = "strongly_prefer_a"  # -2
    PREFER_A = "prefer_a"  # -1
    NEUTRAL = "neutral"  # 0
    PREFER_B = "prefer_b"  # +1
    STRONGLY_PREFER_B = "strongly_prefer_b"  # +2

    def to_score(self) -> int:
        """Convert to numerical score (-2 to +2)."""
        scores = {
            "strongly_prefer_a": -2,
            "prefer_a": -1,
            "neutral": 0,
            "prefer_b": 1,
            "strongly_prefer_b": 2
        }
        return scores[self.value]


@dataclass
class EvaluationPair:
    """A pair of responses to evaluate."""

    pair_id: str  # Unique identifier
    query: str  # Original query
    response_a: str  # First response (order randomized)
    response_b: str  # Second response
    true_variant_a: str  # "traditional" or "unified_mrf"
    true_variant_b: str

    # Randomization (for blind evaluation)
    presentation_order: tuple[str, str] = ("A", "B")  # Which is shown first

    # Evaluation results (filled by evaluator)
    evaluations: dict[str, Preference] = field(default_factory=dict)
    overall_preference: Preference | None = None
    rationale: str | None = None  # Why they chose this

    # Metadata
    evaluator_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def get_winner(self) -> str | None:
        """
        Get which variant won (if any).

        Returns:
            "traditional", "unified_mrf", or None (tie)
        """
        if not self.overall_preference:
            return None

        score = self.overall_preference.to_score()

        if score > 0:  # Prefer B
            # B is better, but which variant is B?
            return self.true_variant_b
        elif score < 0:  # Prefer A
            return self.true_variant_a
        else:  # Neutral
            return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['evaluations'] = {k: v.value for k, v in self.evaluations.items()}
        result['overall_preference'] = self.overall_preference.value if self.overall_preference else None
        return result


@dataclass
class EvaluationResults:
    """Aggregated human evaluation results."""

    total_evaluations: int = 0

    # Win rates
    traditional_wins: int = 0
    mrf_wins: int = 0
    ties: int = 0

    # Preference scores by criterion
    criterion_scores: dict[str, float] = field(default_factory=dict)

    # Overall statistics
    avg_score: float = 0.0  # -2 to +2 (positive favors MRF)
    mrf_preference_rate: float = 0.0  # % of evaluations preferring MRF

    # Inter-rater reliability (if multiple evaluators)
    inter_rater_agreement: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_evaluations": self.total_evaluations,
                "mrf_wins": self.mrf_wins,
                "traditional_wins": self.traditional_wins,
                "ties": self.ties
            },
            "preferences": {
                "mrf_preference_rate": self.mrf_preference_rate,
                "average_score": self.avg_score,
                "criterion_scores": self.criterion_scores
            },
            "reliability": {
                "inter_rater_agreement": self.inter_rater_agreement
            }
        }


class HumanEvaluationCollector:
    """
    Collects human evaluations through blind side-by-side comparisons.

    Usage:
        collector = HumanEvaluationCollector("human_eval.db")

        # Create evaluation pair
        pair = await collector.create_evaluation_pair(
            query="What is Thompson Sampling?",
            traditional_response="...",
            mrf_response="..."
        )

        # Present to evaluator (blind - they don't know which is which)
        print(f"Query: {pair.query}")
        print(f"Response A: {pair.response_a}")
        print(f"Response B: {pair.response_b}")

        # Collect evaluation
        await collector.record_evaluation(
            pair_id=pair.pair_id,
            evaluator_id="evaluator_123",
            overall_preference=Preference.PREFER_B,
            criterion_preferences={
                EvaluationCriterion.ACCURACY: Preference.PREFER_B,
                EvaluationCriterion.CLARITY: Preference.NEUTRAL
            },
            rationale="Response B provides more detail and examples"
        )

        # Analyze results
        results = await collector.analyze_results()
        print(f"MRF preference rate: {results.mrf_preference_rate:.1%}")
    """

    def __init__(self, db_path: str = "human_eval.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Evaluation pairs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_pairs (
                pair_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response_a TEXT NOT NULL,
                response_b TEXT NOT NULL,
                true_variant_a TEXT NOT NULL,
                true_variant_b TEXT NOT NULL,
                presentation_order TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                evaluator_id TEXT,
                criterion TEXT NOT NULL,
                preference TEXT NOT NULL,
                rationale TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (pair_id) REFERENCES evaluation_pairs(pair_id)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_pair ON evaluations(pair_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_evaluator ON evaluations(evaluator_id)")

        conn.commit()
        conn.close()

    async def create_evaluation_pair(
        self,
        query: str,
        traditional_response: str,
        mrf_response: str,
        pair_id: str | None = None
    ) -> EvaluationPair:
        """
        Create an evaluation pair with randomized presentation order.

        Args:
            query: Original query
            traditional_response: Traditional variant response
            mrf_response: UnifiedMRF variant response
            pair_id: Optional custom pair ID

        Returns:
            EvaluationPair with randomized order
        """
        if not pair_id:
            pair_id = f"pair_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Randomize presentation order (blind evaluation)
        if random.random() < 0.5:
            response_a = traditional_response
            response_b = mrf_response
            true_variant_a = "traditional"
            true_variant_b = "unified_mrf"
        else:
            response_a = mrf_response
            response_b = traditional_response
            true_variant_a = "unified_mrf"
            true_variant_b = "traditional"

        pair = EvaluationPair(
            pair_id=pair_id,
            query=query,
            response_a=response_a,
            response_b=response_b,
            true_variant_a=true_variant_a,
            true_variant_b=true_variant_b
        )

        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO evaluation_pairs
            (pair_id, query, response_a, response_b, true_variant_a, true_variant_b,
             presentation_order, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair.pair_id,
            pair.query,
            pair.response_a,
            pair.response_b,
            pair.true_variant_a,
            pair.true_variant_b,
            json.dumps(pair.presentation_order),
            pair.timestamp.isoformat()
        ))

        conn.commit()
        conn.close()

        return pair

    async def record_evaluation(
        self,
        pair_id: str,
        evaluator_id: str,
        overall_preference: Preference,
        criterion_preferences: dict[EvaluationCriterion, Preference] | None = None,
        rationale: str | None = None
    ):
        """
        Record an evaluation for a pair.

        Args:
            pair_id: Evaluation pair identifier
            evaluator_id: Evaluator identifier
            overall_preference: Overall preference
            criterion_preferences: Preferences by criterion
            rationale: Why they chose this
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        # Record overall preference
        cursor.execute("""
            INSERT INTO evaluations
            (pair_id, evaluator_id, criterion, preference, rationale, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pair_id, evaluator_id, "overall", overall_preference.value, rationale, timestamp))

        # Record criterion preferences
        if criterion_preferences:
            for criterion, preference in criterion_preferences.items():
                cursor.execute("""
                    INSERT INTO evaluations
                    (pair_id, evaluator_id, criterion, preference, rationale, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (pair_id, evaluator_id, criterion.value, preference.value, None, timestamp))

        conn.commit()
        conn.close()

    async def get_pair(self, pair_id: str) -> EvaluationPair | None:
        """Retrieve an evaluation pair by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT pair_id, query, response_a, response_b, true_variant_a, true_variant_b,
                   presentation_order, created_at
            FROM evaluation_pairs
            WHERE pair_id = ?
        """, (pair_id,))

        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        pair = EvaluationPair(
            pair_id=row[0],
            query=row[1],
            response_a=row[2],
            response_b=row[3],
            true_variant_a=row[4],
            true_variant_b=row[5],
            presentation_order=tuple(json.loads(row[6])),
            timestamp=datetime.fromisoformat(row[7])
        )

        # Fetch evaluations
        cursor.execute("""
            SELECT criterion, preference, rationale
            FROM evaluations
            WHERE pair_id = ?
        """, (pair_id,))

        eval_rows = cursor.fetchall()

        for criterion, preference, rationale in eval_rows:
            pref = Preference(preference)

            if criterion == "overall":
                pair.overall_preference = pref
                pair.rationale = rationale
            else:
                pair.evaluations[criterion] = pref

        conn.close()
        return pair

    async def analyze_results(self) -> EvaluationResults:
        """Analyze all evaluations and return aggregated results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all pairs with overall evaluations
        cursor.execute("""
            SELECT p.pair_id, p.true_variant_a, p.true_variant_b, e.preference
            FROM evaluation_pairs p
            JOIN evaluations e ON p.pair_id = e.pair_id
            WHERE e.criterion = 'overall'
        """)

        rows = cursor.fetchall()

        results = EvaluationResults()
        results.total_evaluations = len(rows)

        scores = []

        for pair_id, true_variant_a, true_variant_b, preference in rows:
            pref = Preference(preference)
            score = pref.to_score()
            scores.append(score)

            # Determine winner
            if score > 0:  # Prefer B
                winner = true_variant_b
            elif score < 0:  # Prefer A
                winner = true_variant_a
            else:
                winner = None

            if winner == "unified_mrf":
                results.mrf_wins += 1
            elif winner == "traditional":
                results.traditional_wins += 1
            else:
                results.ties += 1

        # Calculate statistics
        if scores:
            results.avg_score = sum(scores) / len(scores)
            results.mrf_preference_rate = results.mrf_wins / len(scores)

        # Criterion scores
        cursor.execute("""
            SELECT e.criterion, AVG(
                CASE e.preference
                    WHEN 'strongly_prefer_a' THEN -2
                    WHEN 'prefer_a' THEN -1
                    WHEN 'neutral' THEN 0
                    WHEN 'prefer_b' THEN 1
                    WHEN 'strongly_prefer_b' THEN 2
                END
            )
            FROM evaluations e
            WHERE e.criterion != 'overall'
            GROUP BY e.criterion
        """)

        criterion_rows = cursor.fetchall()

        for criterion, avg_score in criterion_rows:
            results.criterion_scores[criterion] = avg_score

        conn.close()
        return results

    async def export_results(self, output_path: Path):
        """Export results to JSON file."""
        results = await self.analyze_results()

        output = {
            "summary": results.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[OK] Human evaluation results exported to: {output_path}")


# Example usage
if __name__ == "__main__":
    async def main():
        # Create collector
        collector = HumanEvaluationCollector("human_eval.db")

        # Create evaluation pairs
        queries = [
            "What is Thompson Sampling?",
            "Explain Bayesian optimization",
            "How does exploration-exploitation tradeoff work?",
        ]

        traditional_responses = [
            "Thompson Sampling is a probabilistic algorithm.",
            "Bayesian optimization uses probability.",
            "Exploration-exploitation balances trying new vs known options."
        ]

        mrf_responses = [
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It balances exploration and exploitation by sampling from posterior distributions.",
            "Bayesian optimization is a sequential design strategy for global optimization of black-box functions. It uses a probabilistic model (typically Gaussian process) to guide the search.",
            "The exploration-exploitation tradeoff is fundamental in reinforcement learning. Exploration tries new actions to gather information, while exploitation uses known information to maximize reward."
        ]

        pairs = []
        for i, (query, trad, mrf) in enumerate(zip(queries, traditional_responses, mrf_responses)):
            pair = await collector.create_evaluation_pair(query, trad, mrf)
            pairs.append(pair)

        # Simulate evaluations
        for pair in pairs:
            # Simulate evaluator preferring MRF (response with more detail)
            # In real usage, this would come from actual human evaluators

            # Determine which response is longer (as proxy for quality)
            if len(pair.response_a) > len(pair.response_b):
                preference = Preference.PREFER_A
            else:
                preference = Preference.PREFER_B

            await collector.record_evaluation(
                pair_id=pair.pair_id,
                evaluator_id="evaluator_001",
                overall_preference=preference,
                criterion_preferences={
                    EvaluationCriterion.CLARITY: preference,
                    EvaluationCriterion.COMPLETENESS: preference
                },
                rationale="More detailed and comprehensive"
            )

        # Analyze results
        results = await collector.analyze_results()

        print("\n" + "="*60)
        print("HUMAN EVALUATION RESULTS")
        print("="*60)
        print(f"\nTotal evaluations: {results.total_evaluations}")
        print(f"  UnifiedMRF wins: {results.mrf_wins}")
        print(f"  Traditional wins: {results.traditional_wins}")
        print(f"  Ties: {results.ties}")
        print(f"\nMRF preference rate: {results.mrf_preference_rate:.1%}")
        print(f"Average score: {results.avg_score:+.2f} (range: -2 to +2)")

        if results.criterion_scores:
            print("\nCriterion scores:")
            for criterion, score in results.criterion_scores.items():
                print(f"  {criterion}: {score:+.2f}")

        print("\n" + "="*60)

        # Export
        await collector.export_results(Path("human_eval_results.json"))

    asyncio.run(main())
