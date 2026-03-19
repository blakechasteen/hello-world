"""
Conscience Protocols
====================
Protocol definitions for the Conscience system.

ConscienceLens is the core abstraction - a composable safety concern
that can be combined with other lenses using the | operator.

Example:
    lens = HarmLens() | DeceptionLens() | PowerLens()
    judgment = await lens.examine(action, context)

Philosophy:
    "Composable concerns, unified judgment."

Author: HoloLoom Team
Date: 2025-12-03
"""

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

from .judgment import Concern, Judgment, Wisdom


@runtime_checkable
class ConscienceLens(Protocol):
    """
    Protocol for composable safety concerns.

    A lens examines an action through a specific concern (harm, deception,
    power, etc.) and contributes to the overall Judgment.

    Lenses can be composed using the | operator:
        combined = HarmLens() | DeceptionLens()

    Design Principles:
        1. Single Responsibility: Each lens focuses on one concern
        2. Composable: Lenses combine naturally with |
        3. Learnable: Lenses can learn from outcomes
        4. Explainable: Lenses provide clear reasons for concerns
    """

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
    async def examine(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[Concern]:
        """
        Examine an action through this lens.

        Args:
            action: The action being considered
            context: Context for the action (user, session, etc.)

        Returns:
            List of concerns raised by this lens (may be empty)
        """
        ...

    @abstractmethod
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
        ...

    def __or__(self, other: "ConscienceLens") -> "CompositeLens":
        """Compose two lenses with | operator."""
        from .lenses.composite import CompositeLens
        return CompositeLens([self, other])


class WitnessProtocol(Protocol):
    """
    Protocol for witnessing and recording actions.

    Witnesses observe actions and their outcomes, building
    an audit trail and contributing to learning.
    """

    @abstractmethod
    async def record(
        self,
        action: str,
        context: dict[str, Any],
        judgment: Judgment,
        result: dict[str, Any] | None = None
    ) -> str:
        """
        Record an action and its judgment.

        Args:
            action: The action taken
            context: Context for the action
            judgment: The conscience's judgment
            result: The outcome of the action (if executed)

        Returns:
            Record ID for this observation
        """
        ...

    @abstractmethod
    async def query(
        self,
        filters: dict[str, Any],
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Query past records.

        Args:
            filters: Query filters (action, voice, time range, etc.)
            limit: Maximum records to return

        Returns:
            List of matching records
        """
        ...


class LearnerProtocol(Protocol):
    """
    Protocol for learning from experience.

    Learners improve the conscience's judgment over time
    by tracking patterns and updating parameters.
    """

    @abstractmethod
    async def update(
        self,
        pattern_id: str,
        success: bool,
        confidence: float = 1.0
    ) -> Wisdom:
        """
        Update a pattern based on outcome.

        Args:
            pattern_id: The pattern to update
            success: Whether the outcome was successful
            confidence: Confidence in this observation (0.0-1.0)

        Returns:
            Updated wisdom for this pattern
        """
        ...

    @abstractmethod
    async def get_wisdom(
        self,
        pattern_id: str
    ) -> Wisdom | None:
        """
        Get learned wisdom for a pattern.

        Args:
            pattern_id: The pattern to look up

        Returns:
            Wisdom if pattern exists, None otherwise
        """
        ...

    @abstractmethod
    async def suggest_threshold(
        self,
        context: dict[str, Any]
    ) -> float:
        """
        Suggest a threshold based on learned patterns.

        Args:
            context: Context for the suggestion

        Returns:
            Suggested threshold (0.0-1.0)
        """
        ...


class ConscienceProtocol(Protocol):
    """
    Protocol for the main Conscience interface.

    Defines the three core operations:
    - consider: Evaluate an action before execution
    - witness: Record an action and its outcome
    - learn: Improve from feedback
    """

    @abstractmethod
    async def consider(
        self,
        action: str,
        context: dict[str, Any] | None = None
    ) -> Judgment:
        """
        Consider an action before execution.

        Args:
            action: The action being considered
            context: Optional context (user, session, etc.)

        Returns:
            Judgment with voice level and concerns
        """
        ...

    @abstractmethod
    async def witness(
        self,
        action: str,
        result: dict[str, Any],
        judgment: Judgment | None = None
    ) -> str:
        """
        Witness an action and its outcome.

        Args:
            action: The action that was taken
            result: The outcome of the action
            judgment: The prior judgment (if any)

        Returns:
            Record ID for this observation
        """
        ...

    @abstractmethod
    async def learn(
        self,
        feedback: dict[str, Any]
    ) -> None:
        """
        Learn from feedback.

        Args:
            feedback: Feedback on past judgments/actions
        """
        ...


# =============================================================================
# Type Aliases for Clarity
# =============================================================================

# Action context containing all relevant information
ActionContext = dict[str, Any]

# Outcome of an action
ActionOutcome = dict[str, Any]

# Feedback for learning
Feedback = dict[str, Any]
