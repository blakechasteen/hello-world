"""
Pattern Miner - Extracts high-quality patterns from production logs
===================================================================
Mines patterns from classification telemetry to improve classifier accuracy.

Author: Claude Code
Date: November 13, 2025
Phase: 3 - Adaptive Learning
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ClassificationLog:
    """Single classification event from production."""
    query: str
    expected: str  # Ground truth (if available)
    actual: str  # Classifier output
    confidence: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class PatternScore:
    """Quality metrics for a pattern."""
    precision: float  # % correct when pattern matches
    recall: float  # % of complexity level covered
    support: int  # # queries matching pattern
    confidence: float  # Average confidence
    f1_score: float  # Harmonic mean of precision & recall


@dataclass
class Pattern:
    """Candidate pattern for classifier improvement."""
    regex: str
    complexity: str  # trivial/simple/complex/research
    score: PatternScore
    examples: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())


class PatternMiner:
    """
    Mines high-quality patterns from production classification logs.

    Strategies:
    1. Misclassification mining: Find patterns in incorrectly classified queries
    2. High-confidence mining: Extract patterns from high-confidence successes
    3. N-gram analysis: Discover multi-word patterns
    4. Regex generalization: Generalize specific patterns

    Usage:
        miner = PatternMiner(stats_path="./classification_logs")
        patterns = miner.mine_patterns(min_support=10, min_precision=0.95)
    """

    def __init__(
        self,
        stats_path: Optional[str] = None,
        min_support: int = 10,
        min_precision: float = 0.95,
        min_recall: float = 0.30,
        max_patterns: int = 50
    ):
        """
        Initialize pattern miner.

        Args:
            stats_path: Directory with classification logs
            min_support: Minimum queries matching pattern
            min_precision: Minimum precision (correctness)
            min_recall: Minimum recall (coverage)
            max_patterns: Maximum patterns to return
        """
        self.stats_path = Path(stats_path) if stats_path else None
        self.min_support = min_support
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.max_patterns = max_patterns

        # Pattern templates for generalization
        self.generalization_rules = [
            # Question patterns
            (r'\bwhat is \w+\b', r'what is \w+'),
            (r'\bwhat are \w+\b', r'what are \w+'),
            (r'\bhow (to|do|does) \w+', r'how (to|do|does) \w+'),
            (r'\bwhy (is|are|do|does) \w+', r'why (is|are|do|does) \w+'),
            (r'\bwhen (is|are|do|does) \w+', r'when (is|are|do|does) \w+'),
            (r'\bwhere (is|are|do|does) \w+', r'where (is|are|do|does) \w+'),

            # Command patterns
            (r'\b(tell|show|give|explain) me \w+', r'(tell|show|give|explain) me \w+'),
            (r'\b(define|describe|list) \w+', r'(define|describe|list) \w+'),

            # Greetings
            (r'\b(hi|hello|hey|greetings)\b', r'(hi|hello|hey|greetings)'),
            (r'\bgood (morning|afternoon|evening)\b', r'good (morning|afternoon|evening)'),
        ]

    def mine_patterns(
        self,
        days_lookback: int = 7,
        focus_on_misclassifications: bool = True
    ) -> List[Pattern]:
        """
        Mine patterns from recent classification logs.

        Args:
            days_lookback: How many days of logs to analyze
            focus_on_misclassifications: Prioritize fixing errors

        Returns:
            List of high-quality candidate patterns
        """
        logger.info(f"Mining patterns from last {days_lookback} days")

        # Load classification logs
        logs = self._load_logs(days_lookback=days_lookback)
        if not logs:
            logger.warning("No classification logs found")
            return []

        logger.info(f"Loaded {len(logs)} classification events")

        # Extract candidate patterns
        candidates = self._extract_candidates(logs, focus_on_misclassifications)

        # Score patterns
        scored_patterns = []
        for pattern_str, complexity in candidates:
            score = self._score_pattern(pattern_str, complexity, logs)

            # Filter by quality thresholds
            if (score.support >= self.min_support and
                score.precision >= self.min_precision and
                score.recall >= self.min_recall):

                # Get example queries
                examples = self._get_examples(pattern_str, logs, max_examples=5)

                pattern = Pattern(
                    regex=pattern_str,
                    complexity=complexity,
                    score=score,
                    examples=examples
                )
                scored_patterns.append(pattern)

        # Sort by F1 score (balance precision & recall)
        scored_patterns.sort(key=lambda p: p.score.f1_score, reverse=True)

        # Return top-N patterns
        top_patterns = scored_patterns[:self.max_patterns]

        logger.info(f"Mined {len(top_patterns)} high-quality patterns")
        for p in top_patterns[:5]:
            logger.info(f"  Pattern: '{p.regex}' ({p.complexity}) "
                       f"F1={p.score.f1_score:.3f} support={p.score.support}")

        return top_patterns

    def _load_logs(self, days_lookback: int) -> List[ClassificationLog]:
        """Load classification logs from disk."""
        if not self.stats_path or not self.stats_path.exists():
            return []

        logs = []
        cutoff = datetime.now() - timedelta(days=days_lookback)

        # Read log files
        for log_file in self.stats_path.glob("*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue

                        data = json.loads(line)
                        log = ClassificationLog(
                            query=data['query'],
                            expected=data.get('expected', data['actual']),
                            actual=data['actual'],
                            confidence=data['confidence'],
                            timestamp=data['timestamp'],
                            metadata=data.get('metadata', {})
                        )

                        # Filter by date
                        if datetime.fromtimestamp(log.timestamp) >= cutoff:
                            logs.append(log)

            except Exception as e:
                logger.error(f"Error loading {log_file}: {e}")

        return logs

    def _extract_candidates(
        self,
        logs: List[ClassificationLog],
        focus_on_misclassifications: bool
    ) -> List[Tuple[str, str]]:
        """
        Extract candidate patterns from logs.

        Returns:
            List of (pattern_regex, complexity) tuples
        """
        candidates = set()

        # Group logs by complexity
        by_complexity = defaultdict(list)
        for log in logs:
            by_complexity[log.actual].append(log)

        # Extract patterns for each complexity level
        for complexity, complexity_logs in by_complexity.items():
            # Filter for misclassifications if requested
            if focus_on_misclassifications:
                complexity_logs = [l for l in complexity_logs if l.actual != l.expected]

            # Extract n-grams (1-5 words)
            ngrams = self._extract_ngrams(complexity_logs)

            # Generalize to regex patterns
            for ngram, count in ngrams.most_common(100):
                if count < self.min_support:
                    break

                # Try to generalize
                generalized = self._generalize_pattern(ngram)
                if generalized:
                    candidates.add((generalized, complexity))

        return list(candidates)

    def _extract_ngrams(self, logs: List[ClassificationLog], max_n: int = 5) -> Counter:
        """Extract n-grams from queries."""
        ngrams = Counter()

        for log in logs:
            query = log.query.lower().strip()

            # Tokenize (simple whitespace split)
            tokens = query.split()

            # Extract n-grams
            for n in range(1, min(len(tokens) + 1, max_n + 1)):
                for i in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[i:i+n])
                    ngrams[ngram] += 1

        return ngrams

    def _generalize_pattern(self, text: str) -> Optional[str]:
        """Generalize text into regex pattern."""
        # Try generalization rules
        for pattern, replacement in self.generalization_rules:
            if re.search(pattern, text, re.IGNORECASE):
                return replacement

        # Exact match for short patterns
        if len(text.split()) <= 2:
            # Escape special regex characters
            escaped = re.escape(text)
            return f"^{escaped}$"

        # For longer patterns, require prefix match
        if len(text.split()) >= 3:
            escaped = re.escape(text)
            return f"^{escaped}"

        return None

    def _score_pattern(
        self,
        pattern: str,
        complexity: str,
        logs: List[ClassificationLog]
    ) -> PatternScore:
        """
        Score pattern quality.

        Metrics:
        - Precision: % correct when pattern matches
        - Recall: % of complexity level covered by pattern
        - Support: # queries matching pattern
        - Confidence: Average confidence when pattern matches
        - F1: Harmonic mean of precision & recall
        """
        # Compile regex
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except:
            # Invalid regex
            return PatternScore(
                precision=0.0,
                recall=0.0,
                support=0,
                confidence=0.0,
                f1_score=0.0
            )

        # Find matching queries
        matches = []
        for log in logs:
            if regex.search(log.query.lower()):
                matches.append(log)

        if not matches:
            return PatternScore(
                precision=0.0,
                recall=0.0,
                support=0,
                confidence=0.0,
                f1_score=0.0
            )

        # Calculate precision (% correct among matches)
        correct = sum(1 for m in matches if m.actual == complexity)
        precision = correct / len(matches) if matches else 0.0

        # Calculate recall (% of complexity level covered)
        complexity_total = sum(1 for l in logs if l.actual == complexity)
        recall = len(matches) / complexity_total if complexity_total > 0 else 0.0

        # Calculate average confidence
        avg_confidence = sum(m.confidence for m in matches) / len(matches)

        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        return PatternScore(
            precision=precision,
            recall=recall,
            support=len(matches),
            confidence=avg_confidence,
            f1_score=f1_score
        )

    def _get_examples(
        self,
        pattern: str,
        logs: List[ClassificationLog],
        max_examples: int = 5
    ) -> List[str]:
        """Get example queries matching pattern."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except:
            return []

        examples = []
        for log in logs:
            if regex.search(log.query.lower()):
                examples.append(log.query)
                if len(examples) >= max_examples:
                    break

        return examples

    def export_patterns(self, patterns: List[Pattern], output_path: str):
        """Export patterns to JSON for review/deployment."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for p in patterns:
            data.append({
                'regex': p.regex,
                'complexity': p.complexity,
                'precision': p.score.precision,
                'recall': p.score.recall,
                'support': p.score.support,
                'confidence': p.score.confidence,
                'f1_score': p.score.f1_score,
                'examples': p.examples,
                'created_at': p.created_at
            })

        with open(output, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(patterns)} patterns to {output}")
