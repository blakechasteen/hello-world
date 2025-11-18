"""
Questioning Strategies for Contract-First Prompting

Different approaches to iterative gap identification.

Created: 2025-11-18
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from HoloLoom.prompting.types import Gap, GapAnalysis


class QuestioningStrategy(ABC):
    """Base class for questioning strategies."""

    @abstractmethod
    def prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Prioritize gaps for questioning."""
        pass

    @abstractmethod
    def should_continue(self, gap_analysis: GapAnalysis, confidence_threshold: float) -> bool:
        """Determine if more questions are needed."""
        pass

    @abstractmethod
    def adapt(self, gap: Gap, answer: str) -> None:
        """Adapt strategy based on user answer."""
        pass


class BreadthFirstStrategy(QuestioningStrategy):
    """
    Breadth-first questioning: Cover all dimensions broadly before deep diving.

    Strategy:
    - Ask one question from each dimension
    - Rotate through dimensions
    - Deep dive only after broad coverage
    """

    def __init__(self):
        self.dimensions_covered = set()

    def prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Prioritize by dimension coverage, then priority."""
        # First pass: One question per dimension
        uncovered = [g for g in gaps if g.dimension not in self.dimensions_covered]

        if uncovered:
            # Prefer uncovered dimensions, sorted by priority
            return sorted(uncovered, key=lambda g: g.priority, reverse=True)
        else:
            # All dimensions covered, now sort by priority
            return sorted(gaps, key=lambda g: g.priority, reverse=True)

    def should_continue(self, gap_analysis: GapAnalysis, confidence_threshold: float) -> bool:
        """Continue if confidence below threshold or essential gaps remain."""
        has_unanswered_essential = any(
            not g.asked and not g.optional for g in gap_analysis.gaps
        )
        return gap_analysis.confidence < confidence_threshold or has_unanswered_essential

    def adapt(self, gap: Gap, answer: str) -> None:
        """Track dimension coverage."""
        self.dimensions_covered.add(gap.dimension)


class DepthFirstStrategy(QuestioningStrategy):
    """
    Depth-first questioning: Deep dive into one dimension before moving to next.

    Strategy:
    - Pick highest priority dimension
    - Exhaust all questions in that dimension
    - Move to next dimension
    """

    def __init__(self):
        self.current_dimension: Optional[str] = None
        self.exhausted_dimensions = set()

    def prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Prioritize by current dimension, then highest priority dimension."""
        # If we're in a dimension, prioritize questions from that dimension
        if self.current_dimension:
            same_dimension = [g for g in gaps if g.dimension == self.current_dimension]
            if same_dimension:
                return sorted(same_dimension, key=lambda g: g.priority, reverse=True)
            else:
                # Dimension exhausted, pick new one
                self.exhausted_dimensions.add(self.current_dimension)
                self.current_dimension = None

        # Pick highest priority dimension not yet exhausted
        available = [
            g for g in gaps if g.dimension not in self.exhausted_dimensions
        ]
        if not available:
            return []

        # Group by dimension, pick highest priority dimension
        by_dimension = {}
        for gap in available:
            if gap.dimension not in by_dimension:
                by_dimension[gap.dimension] = []
            by_dimension[gap.dimension].append(gap)

        # Pick dimension with highest priority gap
        best_dimension = max(
            by_dimension.keys(),
            key=lambda d: max(g.priority for g in by_dimension[d]),
        )
        self.current_dimension = best_dimension

        return sorted(by_dimension[best_dimension], key=lambda g: g.priority, reverse=True)

    def should_continue(self, gap_analysis: GapAnalysis, confidence_threshold: float) -> bool:
        """Continue if confidence below threshold or essential gaps remain."""
        has_unanswered_essential = any(
            not g.asked and not g.optional for g in gap_analysis.gaps
        )
        return gap_analysis.confidence < confidence_threshold or has_unanswered_essential

    def adapt(self, gap: Gap, answer: str) -> None:
        """Track current dimension."""
        # Dimension tracking handled in prioritize_gaps
        pass


class EssentialOnlyStrategy(QuestioningStrategy):
    """
    Essential-only questioning: Ask only critical questions.

    Strategy:
    - Skip all optional gaps
    - Ask only essential gaps
    - Prioritize by confidence impact
    - Stop at threshold even if optional gaps remain
    """

    def prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Prioritize essential gaps by confidence impact."""
        essential = [g for g in gaps if not g.optional]
        return sorted(essential, key=lambda g: g.confidence_impact, reverse=True)

    def should_continue(self, gap_analysis: GapAnalysis, confidence_threshold: float) -> bool:
        """Continue only if essential gaps remain."""
        has_unanswered_essential = any(
            not g.asked and not g.optional for g in gap_analysis.gaps
        )
        return has_unanswered_essential

    def adapt(self, gap: Gap, answer: str) -> None:
        """No adaptation needed for essential-only."""
        pass


class AdaptiveStrategy(QuestioningStrategy):
    """
    Adaptive questioning: Adjust strategy based on user responses.

    Strategy:
    - Start breadth-first
    - If user provides detailed answers ’ stay breadth-first
    - If user provides terse answers ’ switch to depth-first
    - If user shows impatience ’ switch to essential-only
    - Track answer length and quality to adapt
    """

    def __init__(self):
        self.current_strategy: QuestioningStrategy = BreadthFirstStrategy()
        self.answer_lengths = []
        self.question_count = 0

    def prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Delegate to current strategy."""
        return self.current_strategy.prioritize_gaps(gaps)

    def should_continue(self, gap_analysis: GapAnalysis, confidence_threshold: float) -> bool:
        """Delegate to current strategy."""
        return self.current_strategy.should_continue(gap_analysis, confidence_threshold)

    def adapt(self, gap: Gap, answer: str) -> None:
        """Adapt strategy based on answer."""
        self.answer_lengths.append(len(answer))
        self.question_count += 1

        # Calculate average answer length
        avg_length = sum(self.answer_lengths) / len(self.answer_lengths)

        # Adapt strategy
        if self.question_count >= 3:  # Need at least 3 answers to adapt
            if avg_length > 200:
                # User provides detailed answers ’ breadth-first works well
                if not isinstance(self.current_strategy, BreadthFirstStrategy):
                    self.current_strategy = BreadthFirstStrategy()
            elif avg_length < 50:
                # User provides terse answers ’ try depth-first
                if not isinstance(self.current_strategy, DepthFirstStrategy):
                    self.current_strategy = DepthFirstStrategy()

        # If user seems impatient (very short answers after many questions)
        if self.question_count >= 5 and avg_length < 30:
            if not isinstance(self.current_strategy, EssentialOnlyStrategy):
                self.current_strategy = EssentialOnlyStrategy()

        # Delegate to current strategy
        self.current_strategy.adapt(gap, answer)


def create_strategy(strategy_type: str) -> QuestioningStrategy:
    """Factory function to create questioning strategies."""
    strategies = {
        "breadth_first": BreadthFirstStrategy,
        "depth_first": DepthFirstStrategy,
        "essential_only": EssentialOnlyStrategy,
        "adaptive": AdaptiveStrategy,
    }

    strategy_class = strategies.get(strategy_type.lower())
    if not strategy_class:
        raise ValueError(
            f"Unknown strategy: {strategy_type}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategy_class()
