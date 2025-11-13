"""
Auto-Detection Engine with Learning

Analyzes queries and suggests appropriate strategies. Learns from
user feedback to improve suggestions over time.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging
from datetime import datetime

from .strategy import StrategyContext
from .registry import get_registry

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Record of user feedback on a strategy suggestion"""
    context_query: str
    strategy_name: str
    was_helpful: bool
    timestamp: datetime = field(default_factory=datetime.now)

    # Context similarity features
    query_words: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    selection_length: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize to dict"""
        return {
            'context_query': self.context_query,
            'strategy_name': self.strategy_name,
            'was_helpful': self.was_helpful,
            'timestamp': self.timestamp.isoformat(),
            'query_words': self.query_words,
            'file_path': self.file_path,
            'selection_length': self.selection_length
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FeedbackRecord':
        """Deserialize from dict"""
        return cls(
            context_query=data['context_query'],
            strategy_name=data['strategy_name'],
            was_helpful=data['was_helpful'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            query_words=data.get('query_words', []),
            file_path=data.get('file_path'),
            selection_length=data.get('selection_length')
        )


class AutoDetector:
    """
    Analyzes queries and suggests appropriate strategies.
    Learns from feedback to improve suggestions over time.

    The detector uses a hybrid approach:
    1. Base confidence from each strategy's can_apply() method
    2. Historical feedback to adjust confidence scores
    3. Context similarity to find relevant past usage

    Example:
        >>> detector = AutoDetector()
        >>> context = StrategyContext(query="analyze security vulnerability")
        >>> suggestions = await detector.detect(context, top_k=3)
        >>> print(suggestions)
        [('challenge', 0.94), ('verify', 0.89), ('scaffold', 0.67)]

        >>> # Record feedback
        >>> detector.record_feedback(context, 'challenge', was_helpful=True)
        >>> # Future suggestions will improve
    """

    def __init__(self, feedback_file: Optional[Path] = None):
        """
        Initialize auto-detector.

        Args:
            feedback_file: Optional path to feedback storage file.
                          Defaults to ~/.promptly/feedback.json
        """
        self.registry = get_registry()
        self.history: List[FeedbackRecord] = []

        # Determine feedback file path
        if feedback_file is None:
            feedback_dir = Path.home() / ".promptly"
            feedback_dir.mkdir(exist_ok=True)
            feedback_file = feedback_dir / "feedback.json"

        self.feedback_file = feedback_file

        # Load existing feedback
        self._load_feedback()

        # Learning parameters
        self.learning_rate = 0.3  # How much historical feedback affects confidence
        self.similarity_threshold = 0.5  # Min similarity to use historical feedback

        logger.info(f"AutoDetector initialized with {len(self.history)} feedback records")

    async def detect(
        self,
        context: StrategyContext,
        top_k: int = 3,
        use_learning: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Detect best strategies for context.

        Args:
            context: Query context
            top_k: Number of suggestions to return
            use_learning: Whether to apply learning adjustments

        Returns:
            List of (strategy_name, confidence) tuples sorted by confidence

        Example:
            >>> detector = AutoDetector()
            >>> context = StrategyContext(query="verify this contract")
            >>> suggestions = await detector.detect(context)
            >>> print(suggestions[0])
            ('verify', 0.92)
        """
        # Get base suggestions from registry
        base_suggestions = self.registry.suggest(context, top_k=top_k * 2)

        if not use_learning or not self.history:
            return base_suggestions[:top_k]

        # Apply learning adjustments
        adjusted = self._apply_learning(context, base_suggestions)

        logger.debug(f"Auto-detection for '{context.query[:50]}': {adjusted[:top_k]}")

        return adjusted[:top_k]

    def _apply_learning(
        self,
        context: StrategyContext,
        base_suggestions: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Adjust base suggestions using historical feedback.

        Args:
            context: Current query context
            base_suggestions: Base confidence scores from strategies

        Returns:
            Adjusted confidence scores
        """
        # Find similar historical contexts
        similar = [
            record for record in self.history
            if self._is_similar(context, record)
        ]

        if not similar:
            logger.debug("No similar historical contexts found")
            return base_suggestions

        logger.debug(f"Found {len(similar)} similar historical contexts")

        # Calculate success rates for each strategy
        strategy_success: Dict[str, List[float]] = {}
        for record in similar:
            if record.strategy_name not in strategy_success:
                strategy_success[record.strategy_name] = []
            strategy_success[record.strategy_name].append(1.0 if record.was_helpful else 0.0)

        # Adjust confidence scores
        adjusted = []
        for name, base_confidence in base_suggestions:
            if name in strategy_success:
                # Calculate success rate
                successes = strategy_success[name]
                success_rate = sum(successes) / len(successes)

                # Blend base confidence with learned success rate
                adjusted_confidence = (
                    (1 - self.learning_rate) * base_confidence +
                    self.learning_rate * success_rate
                )

                logger.debug(
                    f"Strategy '{name}': base={base_confidence:.2f}, "
                    f"success_rate={success_rate:.2f}, "
                    f"adjusted={adjusted_confidence:.2f}"
                )
            else:
                # No historical data, use base confidence
                adjusted_confidence = base_confidence

            adjusted.append((name, adjusted_confidence))

        # Sort by adjusted confidence
        adjusted.sort(key=lambda x: x[1], reverse=True)

        return adjusted

    def _is_similar(self, context: StrategyContext, record: FeedbackRecord) -> bool:
        """
        Check if current context is similar to a historical record.

        Uses keyword overlap, file path matching, and selection length
        to determine similarity.

        Args:
            context: Current context
            record: Historical feedback record

        Returns:
            True if contexts are similar
        """
        # Keyword overlap
        context_words = set(context.query.lower().split())
        record_words = set(record.query_words)

        if not context_words or not record_words:
            return False

        overlap = len(context_words & record_words)
        union = len(context_words | record_words)
        keyword_similarity = overlap / union if union > 0 else 0.0

        # File path matching
        file_match = (
            context.file_path == record.file_path
            if context.file_path and record.file_path
            else False
        )

        # Selection length similarity
        selection_match = False
        if context.selection and record.selection_length:
            context_len = len(context.selection)
            ratio = min(context_len, record.selection_length) / max(context_len, record.selection_length)
            selection_match = ratio > 0.5

        # Combine signals
        if file_match:
            # Same file -> very similar
            return True

        if keyword_similarity > self.similarity_threshold:
            return True

        if selection_match and keyword_similarity > 0.3:
            return True

        return False

    def record_feedback(
        self,
        context: StrategyContext,
        strategy_name: str,
        was_helpful: bool
    ):
        """
        Record feedback for learning.

        Args:
            context: Query context
            strategy_name: Strategy that was used
            was_helpful: Whether the strategy was helpful

        Example:
            >>> detector = AutoDetector()
            >>> context = StrategyContext(query="verify contract")
            >>> detector.record_feedback(context, 'verify', was_helpful=True)
        """
        record = FeedbackRecord(
            context_query=context.query,
            strategy_name=strategy_name,
            was_helpful=was_helpful,
            query_words=context.query.lower().split(),
            file_path=context.file_path,
            selection_length=len(context.selection) if context.selection else None
        )

        self.history.append(record)

        # Trim history to last 1000 entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        # Save to disk
        self._save_feedback()

        logger.info(
            f"Recorded feedback: {strategy_name} was "
            f"{'helpful' if was_helpful else 'not helpful'} "
            f"for query '{context.query[:50]}'"
        )

    def _load_feedback(self):
        """Load feedback history from disk"""
        if not self.feedback_file.exists():
            logger.debug("No existing feedback file found")
            return

        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)

            self.history = [
                FeedbackRecord.from_dict(record)
                for record in data.get('records', [])
            ]

            logger.info(f"Loaded {len(self.history)} feedback records")

        except Exception as e:
            logger.error(f"Error loading feedback: {e}", exc_info=True)
            self.history = []

    def _save_feedback(self):
        """Save feedback history to disk"""
        try:
            data = {
                'records': [record.to_dict() for record in self.history],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.history)} feedback records")

        except Exception as e:
            logger.error(f"Error saving feedback: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about learning.

        Returns:
            Dict with learning statistics

        Example:
            >>> detector = AutoDetector()
            >>> stats = detector.get_stats()
            >>> print(stats['total_feedback'])
            1234
            >>> print(stats['strategy_success_rates'])
            {'verify': 0.87, 'challenge': 0.92, ...}
        """
        if not self.history:
            return {
                'total_feedback': 0,
                'strategy_success_rates': {},
                'most_helpful': None
            }

        # Calculate success rates per strategy
        strategy_totals: Dict[str, Tuple[int, int]] = {}  # (successes, total)
        for record in self.history:
            if record.strategy_name not in strategy_totals:
                strategy_totals[record.strategy_name] = (0, 0)

            successes, total = strategy_totals[record.strategy_name]
            if record.was_helpful:
                successes += 1
            total += 1
            strategy_totals[record.strategy_name] = (successes, total)

        # Calculate rates
        success_rates = {
            name: successes / total
            for name, (successes, total) in strategy_totals.items()
        }

        # Find most helpful
        most_helpful = max(success_rates.items(), key=lambda x: x[1]) if success_rates else None

        return {
            'total_feedback': len(self.history),
            'strategy_success_rates': success_rates,
            'most_helpful': most_helpful[0] if most_helpful else None,
            'learning_rate': self.learning_rate
        }

    def reset_learning(self):
        """Reset all learning data (for testing or fresh start)"""
        self.history = []
        self._save_feedback()
        logger.info("Reset all learning data")


# Global auto-detector instance
_global_detector: Optional[AutoDetector] = None


def get_auto_detector() -> AutoDetector:
    """Get or create global auto-detector"""
    global _global_detector
    if _global_detector is None:
        _global_detector = AutoDetector()
    return _global_detector


async def auto_detect(context: StrategyContext, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Auto-detect best strategies for context.

    Convenience function using global detector.

    Args:
        context: Query context
        top_k: Number of suggestions

    Returns:
        List of (strategy_name, confidence) tuples

    Example:
        >>> context = StrategyContext(query="analyze contract")
        >>> suggestions = await auto_detect(context)
        >>> print(suggestions[0])
        ('verify', 0.92)
    """
    detector = get_auto_detector()
    return await detector.detect(context, top_k=top_k)
