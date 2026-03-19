"""
Base Lens
=========
Abstract base class for all conscience lenses.

Provides common functionality and structure for lens implementations.

Author: HoloLoom Team
Date: 2025-12-03
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..judgment import Concern, Wisdom

logger = logging.getLogger("hololoom.conscience.lenses")


class BaseLens(ABC):
    """
    Abstract base class for conscience lenses.

    Provides common functionality:
    - Threshold management
    - Wisdom tracking
    - Logging
    - Composition support

    Subclasses must implement:
    - name property
    - category property
    - _examine_impl method
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enabled: bool = True,
    ):
        """
        Initialize a lens.

        Args:
            threshold: Concern threshold (0.0-1.0). Higher = more lenient.
            enabled: Whether this lens is active.
        """
        self.threshold = threshold
        self.enabled = enabled
        self._wisdom: dict[str, Wisdom] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this lens."""
        ...

    @property
    @abstractmethod
    def category(self) -> str:
        """Category of concern (e.g., 'harm', 'deception', 'power')."""
        ...

    @abstractmethod
    async def _examine_impl(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """
        Internal examination implementation.

        Override this in subclasses.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            List of concerns (may be empty)
        """
        ...

    async def examine(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """
        Examine an action through this lens.

        Args:
            action: The action being considered
            context: Context for the action

        Returns:
            List of concerns raised by this lens (may be empty)
        """
        if not self.enabled:
            return []

        try:
            concerns = await self._examine_impl(action, context)

            # Filter by threshold
            filtered = [c for c in concerns if c.confidence >= self.threshold]

            if filtered:
                logger.debug(
                    f"{self.name}: Raised {len(filtered)} concerns "
                    f"(threshold: {self.threshold})"
                )

            return filtered

        except Exception as e:
            logger.error(f"{self.name}: Error during examination: {e}")
            # Graceful degradation - return no concerns on error
            return []

    async def learn(
        self,
        action: str,
        context: dict[str, Any],
        outcome: dict[str, Any]
    ) -> None:
        """
        Learn from the outcome of an action.

        Args:
            action: The action that was taken
            context: Context for the action
            outcome: What happened (success, failure, consequences)
        """
        # Generate pattern ID from action and context
        pattern_id = self._generate_pattern_id(action, context)

        # Get or create wisdom
        if pattern_id not in self._wisdom:
            self._wisdom[pattern_id] = Wisdom(
                pattern_id=pattern_id,
                description=f"{self.category}: {action[:50]}",
                contexts=[],
            )

        wisdom = self._wisdom[pattern_id]

        # Determine if outcome was successful
        success = outcome.get("success", True)
        confidence = outcome.get("confidence", 1.0)

        # Update wisdom
        if success:
            wisdom.update_success(confidence)
        else:
            wisdom.update_failure(confidence)

        logger.debug(
            f"{self.name}: Updated wisdom '{pattern_id}' - "
            f"success_rate: {wisdom.expected_success_rate:.2f}"
        )

    def _generate_pattern_id(
        self,
        action: str,
        context: dict[str, Any]
    ) -> str:
        """Generate a pattern ID from action and context."""
        # Simple hash-based pattern ID
        context_key = "_".join(sorted(context.keys())[:3])
        action_key = action[:20].replace(" ", "_").lower()
        return f"{self.category}:{action_key}:{context_key}"

    def get_wisdom(self, pattern_id: str) -> Wisdom | None:
        """Get wisdom for a pattern."""
        return self._wisdom.get(pattern_id)

    def suggest_threshold(self, context: dict[str, Any]) -> float:
        """
        Suggest a threshold based on learned patterns.

        Uses Thompson Sampling across relevant wisdom to suggest
        a threshold that balances safety and utility.
        """
        if not self._wisdom:
            return self.threshold

        # Find relevant wisdom based on context
        relevant = []
        for wisdom in self._wisdom.values():
            if any(ctx in str(context) for ctx in wisdom.contexts):
                relevant.append(wisdom)

        if not relevant:
            return self.threshold

        # Weight by confidence and success rate
        total_weight = 0.0
        weighted_sum = 0.0

        for wisdom in relevant:
            weight = wisdom.confidence
            # Higher success rate -> suggest higher (more lenient) threshold
            suggested = wisdom.expected_success_rate
            weighted_sum += suggested * weight
            total_weight += weight

        if total_weight == 0:
            return self.threshold

        return weighted_sum / total_weight

    def __or__(self, other: "BaseLens") -> "CompositeLens":
        """Compose two lenses with | operator."""
        from .composite import CompositeLens
        return CompositeLens([self, other])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
