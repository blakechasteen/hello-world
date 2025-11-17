"""
AR Pattern Miner - Extract high-quality patterns from AR interaction logs
==========================================================================
Mines patterns from AR classification logs to improve classifier accuracy.

Author: Claude Code (Agent Q)
Date: November 17, 2025
Phase: Agent Swarm Wave 5 - Adaptive Routing Integration
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timedelta

from HoloLoom.voice.ar_query_classifier import ARComplexity, ARQueryType, ARPatternRule

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ARClassificationLog:
    """Single AR classification event from production."""
    query: str
    expected: Optional[str] = None  # Ground truth (if available)
    actual_complexity: str = ""  # Classifier output
    actual_ar_type: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    context: Dict = field(default_factory=dict)
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
class ARPattern:
    """Discovered AR pattern."""
    regex: str
    complexity: ARComplexity
    ar_type: ARQueryType
    score: PatternScore
    examples: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    pattern_type: str = "discovered"  # discovered, gesture, spatial, visual


# ============================================================================
# AR Pattern Miner
# ============================================================================

class ARPatternMiner:
    """
    Mines high-quality patterns from AR interaction logs.

    Strategies:
    1. Misclassification mining: Find patterns in incorrectly classified queries
    2. High-confidence mining: Extract patterns from high-confidence successes
    3. N-gram analysis: Discover multi-word patterns
    4. AR-specific mining: Gesture patterns, spatial patterns, visual patterns
    5. Regex generalization: Generalize specific patterns

    Usage:
        miner = ARPatternMiner(log_dir="./ar_logs")
        patterns = miner.mine_patterns(
            min_support=10,
            min_precision=0.95,
            lookback_days=7
        )
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        min_support: int = 10,
        min_precision: float = 0.95,
        min_recall: float = 0.30,
        max_patterns: int = 50
    ):
        """
        Initialize AR pattern miner.

        Args:
            log_dir: Directory with classification logs
            min_support: Minimum queries matching pattern
            min_precision: Minimum precision (correctness)
            min_recall: Minimum recall (coverage)
            max_patterns: Maximum patterns to return
        """
        self.log_dir = Path(log_dir) if log_dir else Path("./ar_logs")
        self.min_support = min_support
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.max_patterns = max_patterns

        # AR-specific pattern templates
        self.ar_pattern_templates = self._initialize_ar_templates()

    def _initialize_ar_templates(self) -> Dict[str, List[str]]:
        """Initialize AR-specific pattern templates."""
        return {
            "gesture": [
                r'\b(point|tap|swipe|pinch|grab)\s+',
                r'\b(select|choose)\s+(this|that)\b',
                r'\bgesture\s+\w+\b',
            ],
            "spatial": [
                r'\b(here|there|over\s+there|right\s+there|this\s+one|that\s+one)\b',
                r'\b(left|right|above|below|behind|in\s+front)\b',
                r'\bnear(by)?\s+',
            ],
            "visual": [
                r'\bwhat\'?s\s+(this|that)\b',
                r'\bidentify\s+(this|that)\b',
                r'\blook\s+at\s+',
                r'\b(see|seeing)\s+',
            ],
            "multimodal": [
                r'\b(and\s+then|after\s+that|then)\b',
                r'\bwhile\s+\w+',
                r'\bas\s+I\s+',
            ]
        }

    def mine_patterns(
        self,
        lookback_days: int = 7,
        include_low_confidence: bool = False
    ) -> List[ARPattern]:
        """
        Mine patterns from AR classification logs.

        Args:
            lookback_days: Number of days to look back
            include_low_confidence: Include patterns from low-confidence classifications

        Returns:
            List of high-quality AR patterns
        """
        # Load logs
        logs = self._load_logs(lookback_days)
        if not logs:
            logger.warning("No logs found for pattern mining")
            return []

        logger.info(f"Mining patterns from {len(logs)} classification logs")

        # Mine different pattern types
        all_patterns = []

        # 1. N-gram patterns
        ngram_patterns = self._mine_ngram_patterns(logs)
        all_patterns.extend(ngram_patterns)

        # 2. AR-specific patterns
        ar_patterns = self._mine_ar_specific_patterns(logs)
        all_patterns.extend(ar_patterns)

        # 3. Misclassification patterns
        if include_low_confidence:
            misclass_patterns = self._mine_misclassification_patterns(logs)
            all_patterns.extend(misclass_patterns)

        # 4. High-confidence patterns
        highconf_patterns = self._mine_high_confidence_patterns(logs)
        all_patterns.extend(highconf_patterns)

        # Score and filter patterns
        scored_patterns = self._score_patterns(all_patterns, logs)
        filtered_patterns = self._filter_patterns(scored_patterns)

        # Sort by F1 score and limit
        filtered_patterns.sort(key=lambda p: p.score.f1_score, reverse=True)
        top_patterns = filtered_patterns[:self.max_patterns]

        logger.info(f"Discovered {len(top_patterns)} high-quality patterns")
        return top_patterns

    def _load_logs(self, lookback_days: int) -> List[ARClassificationLog]:
        """Load classification logs from JSONL files."""
        logs = []
        cutoff_time = datetime.now().timestamp() - (lookback_days * 86400)

        # Find all log files
        if not self.log_dir.exists():
            return logs

        for log_file in self.log_dir.glob("classifications_*.jsonl"):
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)

                        # Check timestamp
                        if entry.get("timestamp", 0) < cutoff_time:
                            continue

                        # Parse log entry
                        log = ARClassificationLog(
                            query=entry["query"],
                            actual_complexity=entry.get("complexity", ""),
                            actual_ar_type=entry.get("ar_type", ""),
                            confidence=entry.get("confidence", 0.0),
                            timestamp=entry.get("timestamp", 0.0),
                            context=entry.get("metadata", {}).get("context", {}),
                            metadata=entry.get("metadata", {})
                        )
                        logs.append(log)
            except Exception as e:
                logger.warning(f"Failed to load log file {log_file}: {e}")

        return logs

    def _mine_ngram_patterns(self, logs: List[ARClassificationLog]) -> List[ARPattern]:
        """Mine n-gram patterns (1-4 words)."""
        patterns = []

        # Group logs by complexity
        by_complexity = defaultdict(list)
        for log in logs:
            if log.actual_complexity:
                by_complexity[log.actual_complexity].append(log.query)

        # Extract n-grams for each complexity level
        for complexity_str, queries in by_complexity.items():
            # 1-grams, 2-grams, 3-grams
            for n in [1, 2, 3]:
                ngrams = self._extract_ngrams(queries, n)

                # Convert frequent n-grams to regex patterns
                for ngram, count in ngrams.most_common(50):
                    if count < self.min_support:
                        continue

                    # Create regex pattern
                    escaped_ngram = re.escape(ngram)
                    regex = rf'\b{escaped_ngram}\b'

                    # Determine AR type (default to VOICE_ONLY)
                    ar_type = ARQueryType.VOICE_ONLY

                    pattern = ARPattern(
                        regex=regex,
                        complexity=ARComplexity(complexity_str),
                        ar_type=ar_type,
                        score=PatternScore(0, 0, count, 0, 0),  # Will be scored later
                        examples=[q for q in queries if ngram in q.lower()][:3],
                        pattern_type="ngram"
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_ngrams(self, queries: List[str], n: int) -> Counter:
        """Extract n-grams from queries."""
        ngrams = Counter()

        for query in queries:
            words = query.lower().split()
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                ngrams[ngram] += 1

        return ngrams

    def _mine_ar_specific_patterns(self, logs: List[ARClassificationLog]) -> List[ARPattern]:
        """Mine AR-specific patterns (gesture, spatial, visual, multimodal)."""
        patterns = []

        # Group logs by AR type
        by_ar_type = defaultdict(list)
        for log in logs:
            if log.actual_ar_type:
                by_ar_type[log.actual_ar_type].append(log)

        # Extract patterns for each AR type
        for ar_type_str, type_logs in by_ar_type.items():
            ar_type = ARQueryType(ar_type_str)

            # Determine pattern category
            if ar_type == ARQueryType.GESTURE_COMMAND:
                templates = self.ar_pattern_templates["gesture"]
            elif ar_type == ARQueryType.SPATIAL_REFERENCE:
                templates = self.ar_pattern_templates["spatial"]
            elif ar_type == ARQueryType.VISUAL_QUERY:
                templates = self.ar_pattern_templates["visual"]
            elif ar_type == ARQueryType.MULTIMODAL:
                templates = self.ar_pattern_templates["multimodal"]
            else:
                continue

            # Test each template
            for template in templates:
                matching_logs = [
                    log for log in type_logs
                    if re.search(template, log.query.lower())
                ]

                if len(matching_logs) < self.min_support:
                    continue

                # Determine complexity (most common)
                complexity_counts = Counter(log.actual_complexity for log in matching_logs)
                most_common_complexity = complexity_counts.most_common(1)[0][0]

                pattern = ARPattern(
                    regex=template,
                    complexity=ARComplexity(most_common_complexity),
                    ar_type=ar_type,
                    score=PatternScore(0, 0, len(matching_logs), 0, 0),  # Scored later
                    examples=[log.query for log in matching_logs[:3]],
                    pattern_type=ar_type_str
                )
                patterns.append(pattern)

        return patterns

    def _mine_misclassification_patterns(self, logs: List[ARClassificationLog]) -> List[ARPattern]:
        """Mine patterns from misclassified queries."""
        patterns = []

        # Find low-confidence classifications
        low_conf_logs = [log for log in logs if log.confidence < 0.6]

        if not low_conf_logs:
            return patterns

        # Group by actual classification
        by_complexity = defaultdict(list)
        for log in low_conf_logs:
            if log.actual_complexity:
                by_complexity[log.actual_complexity].append(log.query)

        # Extract patterns
        for complexity_str, queries in by_complexity.items():
            # Find common phrases
            phrases = self._find_common_phrases(queries)

            for phrase, count in phrases.most_common(20):
                if count < 5:  # Lower threshold for misclassifications
                    continue

                escaped_phrase = re.escape(phrase)
                regex = rf'\b{escaped_phrase}\b'

                pattern = ARPattern(
                    regex=regex,
                    complexity=ARComplexity(complexity_str),
                    ar_type=ARQueryType.VOICE_ONLY,
                    score=PatternScore(0, 0, count, 0, 0),
                    examples=[q for q in queries if phrase in q.lower()][:3],
                    pattern_type="misclassification"
                )
                patterns.append(pattern)

        return patterns

    def _mine_high_confidence_patterns(self, logs: List[ARClassificationLog]) -> List[ARPattern]:
        """Mine patterns from high-confidence classifications."""
        patterns = []

        # Find high-confidence classifications
        high_conf_logs = [log for log in logs if log.confidence >= 0.85]

        if not high_conf_logs:
            return patterns

        # Group by complexity
        by_complexity = defaultdict(list)
        for log in high_conf_logs:
            if log.actual_complexity:
                by_complexity[log.actual_complexity].append(log.query)

        # Extract patterns
        for complexity_str, queries in by_complexity.items():
            # Find common phrases
            phrases = self._find_common_phrases(queries)

            for phrase, count in phrases.most_common(30):
                if count < self.min_support:
                    continue

                escaped_phrase = re.escape(phrase)
                regex = rf'\b{escaped_phrase}\b'

                pattern = ARPattern(
                    regex=regex,
                    complexity=ARComplexity(complexity_str),
                    ar_type=ARQueryType.VOICE_ONLY,
                    score=PatternScore(0, 0, count, 0, 0),
                    examples=[q for q in queries if phrase in q.lower()][:3],
                    pattern_type="high_confidence"
                )
                patterns.append(pattern)

        return patterns

    def _find_common_phrases(self, queries: List[str]) -> Counter:
        """Find common phrases in queries."""
        phrases = Counter()

        for query in queries:
            # Extract 2-4 word phrases
            words = query.lower().split()
            for n in [2, 3, 4]:
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    phrases[phrase] += 1

        return phrases

    def _score_patterns(
        self,
        patterns: List[ARPattern],
        logs: List[ARClassificationLog]
    ) -> List[ARPattern]:
        """Score patterns based on precision, recall, and support."""
        for pattern in patterns:
            # Find matching logs
            matching_logs = [
                log for log in logs
                if re.search(pattern.regex, log.query.lower())
            ]

            if not matching_logs:
                continue

            # Precision: % correct when pattern matches
            correct = sum(
                1 for log in matching_logs
                if log.actual_complexity == pattern.complexity.value
            )
            precision = correct / len(matching_logs) if matching_logs else 0.0

            # Recall: % of complexity level covered
            complexity_logs = [
                log for log in logs
                if log.actual_complexity == pattern.complexity.value
            ]
            recall = len(matching_logs) / len(complexity_logs) if complexity_logs else 0.0

            # Average confidence
            avg_confidence = (
                sum(log.confidence for log in matching_logs) / len(matching_logs)
                if matching_logs else 0.0
            )

            # F1 score
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            # Update score
            pattern.score = PatternScore(
                precision=precision,
                recall=recall,
                support=len(matching_logs),
                confidence=avg_confidence,
                f1_score=f1
            )

        return patterns

    def _filter_patterns(self, patterns: List[ARPattern]) -> List[ARPattern]:
        """Filter patterns by quality thresholds."""
        filtered = []

        for pattern in patterns:
            # Check thresholds
            if pattern.score.support < self.min_support:
                continue
            if pattern.score.precision < self.min_precision:
                continue
            if pattern.score.recall < self.min_recall:
                continue

            filtered.append(pattern)

        return filtered

    def export_patterns(self, patterns: List[ARPattern], filepath: str) -> None:
        """Export patterns to JSON file."""
        try:
            patterns_data = []
            for p in patterns:
                patterns_data.append({
                    "pattern": p.regex,
                    "complexity": p.complexity.value,
                    "ar_type": p.ar_type.value,
                    "confidence_boost": p.score.f1_score * 0.2,  # Convert F1 to boost
                    "priority": int(p.score.precision * 10),
                    "examples": p.examples,
                    "score": {
                        "precision": p.score.precision,
                        "recall": p.score.recall,
                        "support": p.score.support,
                        "confidence": p.score.confidence,
                        "f1_score": p.score.f1_score
                    },
                    "pattern_type": p.pattern_type,
                    "created_at": p.created_at
                })

            with open(filepath, "w") as f:
                json.dump(patterns_data, f, indent=2)

            logger.info(f"Exported {len(patterns)} patterns to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export patterns: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_ar_pattern_miner(
    log_dir: Optional[str] = None,
    min_support: int = 10,
    min_precision: float = 0.95
) -> ARPatternMiner:
    """
    Factory function to create AR pattern miner.

    Args:
        log_dir: Directory with classification logs
        min_support: Minimum queries matching pattern
        min_precision: Minimum precision (correctness)

    Returns:
        ARPatternMiner instance
    """
    return ARPatternMiner(
        log_dir=log_dir,
        min_support=min_support,
        min_precision=min_precision
    )
