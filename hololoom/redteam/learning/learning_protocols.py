"""
Learning Protocols for CARTS
============================

Base protocol definitions for the unified learning system.

Provides abstract interfaces for:
- Thompson Sampling bandits (flat and contextual)
- Heat-based payload tracking
- A/B testing for attack variants
- Hierarchical learning (strategy/payload/mutation)

Philosophy: "Protocol-based design enables swappable implementations."

Author: CARTS Team
Date: 2025-12-03
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar('T')  # Generic arm/element type
ContextT = TypeVar('ContextT')  # Context key type


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LearningResult:
    """
    Result of a learning update.

    Attributes:
        success: Whether the action was successful
        reward: Numeric reward (0.0-1.0, typically severity for attacks)
        metadata: Additional context about the result
        timestamp: When the result was recorded
    """
    success: bool
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        # Clamp reward to [0, 1]
        self.reward = max(0.0, min(1.0, self.reward))


@dataclass
class HeatScore:
    """
    Heat score for a payload or pattern.

    Heat represents how "hot" (effective and recently used) an element is.

    Formula:
        heat = access_count × success_rate × avg_confidence × decay_factor
        decay_factor = 0.95^hours_since_last_success

    Attributes:
        element_id: Identifier for the tracked element
        heat: Current heat score (0.0-1.0)
        access_count: Total number of accesses
        success_count: Number of successful uses
        avg_severity: Average severity when successful
        last_success: Timestamp of last successful use
        is_hot: True if heat > 0.7
        is_cold: True if heat < 0.3
    """
    element_id: str
    heat: float = 0.0
    access_count: int = 0
    success_count: int = 0
    avg_severity: float = 0.0
    last_success: str | None = None

    @property
    def is_hot(self) -> bool:
        """Hot payloads (heat > 0.7) seed genetic evolution."""
        return self.heat > 0.7

    @property
    def is_cold(self) -> bool:
        """Cold payloads (heat < 0.3) get pruned."""
        return self.heat < 0.3

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        if self.access_count == 0:
            return 0.0
        return self.success_count / self.access_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'element_id': self.element_id,
            'heat': self.heat,
            'access_count': self.access_count,
            'success_count': self.success_count,
            'avg_severity': self.avg_severity,
            'last_success': self.last_success,
            'is_hot': self.is_hot,
            'is_cold': self.is_cold,
            'success_rate': self.success_rate,
        }


@dataclass
class ContextKey:
    """
    Context key for contextual bandits.

    Enables per-context arm tracking:
    - target_type: "llm", "api", "web_app"
    - vuln_class: "prompt_injection", "jailbreak", "unicode_bypass"
    - defense_level: "none", "basic", "advanced"

    Attributes:
        target_type: Type of target being attacked
        vuln_class: Vulnerability classification
        defense_level: Level of defensive measures
    """
    target_type: str
    vuln_class: str
    defense_level: str = "unknown"

    def __hash__(self) -> int:
        return hash((self.target_type, self.vuln_class, self.defense_level))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextKey):
            return False
        return (
            self.target_type == other.target_type and
            self.vuln_class == other.vuln_class and
            self.defense_level == other.defense_level
        )

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to tuple for legacy compatibility."""
        return (self.target_type, self.vuln_class, self.defense_level)

    @classmethod
    def from_tuple(cls, t: tuple[str, str, str]) -> 'ContextKey':
        """Create from tuple."""
        return cls(target_type=t[0], vuln_class=t[1], defense_level=t[2])


@dataclass
class ABTestResult:
    """
    Result of an A/B test comparison.

    Attributes:
        winner: "control", "treatment", or "inconclusive"
        p_value: Statistical significance (p < 0.05 = significant)
        effect_size: Cohen's d effect size
        control_mean: Mean success rate for control
        treatment_mean: Mean success rate for treatment
        sample_size_control: Number of samples in control group
        sample_size_treatment: Number of samples in treatment group
    """
    winner: str
    p_value: float
    effect_size: float
    control_mean: float
    treatment_mean: float
    sample_size_control: int
    sample_size_treatment: int

    @property
    def is_significant(self) -> bool:
        """True if p_value < 0.05."""
        return self.p_value < 0.05

    @property
    def improvement(self) -> float:
        """Percentage improvement of treatment over control."""
        if self.control_mean == 0:
            return 0.0
        return (self.treatment_mean - self.control_mean) / self.control_mean


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class LearnerProtocol(Protocol):
    """
    Base protocol for all learning components.

    Defines common interface for updating and querying learned state.
    """

    def update(self, result: LearningResult) -> None:
        """Update learner with new result."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        ...

    def save(self, path: Path) -> None:
        """Save learner state to file."""
        ...

    def load(self, path: Path) -> None:
        """Load learner state from file."""
        ...

    def reset(self) -> None:
        """Reset learner to initial state."""
        ...


@runtime_checkable
class BanditProtocol(Protocol[T]):
    """
    Protocol for Thompson Sampling bandits.

    Defines interface compatible with existing RedTeamBandit.

    Type Parameters:
        T: Type of arms (e.g., AttackStrategy)
    """

    def select(self) -> T:
        """
        Select arm using Thompson Sampling.

        Samples from Beta(α, β) for each arm,
        returns arm with highest sample.

        Returns:
            Selected arm
        """
        ...

    def select_top_k(self, k: int) -> list[T]:
        """
        Select top-k arms by Thompson Sampling.

        Args:
            k: Number of arms to select

        Returns:
            List of top-k arms
        """
        ...

    def update(self, arm: T, success: bool, reward: float = 1.0) -> None:
        """
        Update arm based on result.

        Args:
            arm: The arm that was pulled
            success: True if attack found vulnerability
            reward: Severity of vulnerability (0.0-1.0)
        """
        ...

    def get_expected_rewards(self) -> dict[T, float]:
        """Get expected rewards for all arms."""
        ...

    def get_best_arm(self) -> T:
        """Get arm with highest expected reward."""
        ...

    def get_exploration_candidates(self, uncertainty_threshold: float = 0.1) -> list[T]:
        """Get arms with high uncertainty (worth exploring)."""
        ...


@runtime_checkable
class ContextualBanditProtocol(Protocol[T, ContextT]):
    """
    Protocol for contextual Thompson Sampling bandits.

    Maintains separate arm statistics per context.

    Type Parameters:
        T: Type of arms (e.g., AttackStrategy)
        ContextT: Type of context keys (e.g., ContextKey)
    """

    def select(self, context: ContextT) -> T:
        """
        Select arm for given context using Thompson Sampling.

        Args:
            context: Context key for this selection

        Returns:
            Selected arm
        """
        ...

    def select_top_k(self, context: ContextT, k: int) -> list[T]:
        """
        Select top-k arms for given context.

        Args:
            context: Context key
            k: Number of arms to select

        Returns:
            List of top-k arms
        """
        ...

    def update(
        self,
        context: ContextT,
        arm: T,
        success: bool,
        reward: float = 1.0
    ) -> None:
        """
        Update arm for given context.

        Args:
            context: Context key
            arm: The arm that was pulled
            success: True if attack found vulnerability
            reward: Severity of vulnerability (0.0-1.0)
        """
        ...

    def get_context_stats(self, context: ContextT) -> dict[str, Any]:
        """Get statistics for a specific context."""
        ...

    def get_all_contexts(self) -> list[ContextT]:
        """Get all known contexts."""
        ...


@runtime_checkable
class HeatTrackerProtocol(Protocol[T]):
    """
    Protocol for heat-based tracking.

    Tracks access patterns and effectiveness to identify
    "hot" (effective, recent) and "cold" (ineffective, stale) elements.

    Type Parameters:
        T: Type of tracked elements (e.g., payload IDs)
    """

    def record_access(self, element_id: T, success: bool, severity: float = 0.0) -> None:
        """
        Record an access to an element.

        Args:
            element_id: ID of the accessed element
            success: Whether the access was successful
            severity: Severity score if successful (0.0-1.0)
        """
        ...

    def get_heat(self, element_id: T) -> float:
        """
        Get current heat score for an element.

        Args:
            element_id: ID of the element

        Returns:
            Heat score (0.0-1.0)
        """
        ...

    def get_hot_elements(self, limit: int = 10) -> list[HeatScore]:
        """
        Get hottest elements (heat > 0.7).

        Args:
            limit: Maximum number of elements to return

        Returns:
            List of hot elements sorted by heat descending
        """
        ...

    def get_cold_elements(self, limit: int = 10) -> list[HeatScore]:
        """
        Get coldest elements (heat < 0.3).

        Args:
            limit: Maximum number of elements to return

        Returns:
            List of cold elements sorted by heat ascending
        """
        ...

    def apply_decay(self) -> int:
        """
        Apply time-based decay to all elements.

        Returns:
            Number of elements affected
        """
        ...

    def prune_cold(self) -> int:
        """
        Remove cold elements from tracking.

        Returns:
            Number of elements pruned
        """
        ...


@runtime_checkable
class ABTestProtocol(Protocol):
    """
    Protocol for A/B testing attack variants.

    Enables statistical comparison of different attack strategies
    or payload variants.
    """

    def create_test(
        self,
        name: str,
        control: str,
        treatment: str,
        min_samples: int = 30
    ) -> None:
        """
        Create a new A/B test.

        Args:
            name: Unique test name
            control: Description of control variant
            treatment: Description of treatment variant
            min_samples: Minimum samples per variant before analysis
        """
        ...

    def record_result(
        self,
        test_name: str,
        variant: str,
        success: bool,
        severity: float = 0.0
    ) -> None:
        """
        Record a result for a test variant.

        Args:
            test_name: Name of the test
            variant: "control" or "treatment"
            success: Whether the attack succeeded
            severity: Severity if successful
        """
        ...

    def analyze(self, test_name: str) -> ABTestResult | None:
        """
        Analyze test results.

        Returns None if insufficient samples.

        Args:
            test_name: Name of the test

        Returns:
            ABTestResult or None if insufficient data
        """
        ...

    def get_active_tests(self) -> list[str]:
        """Get names of all active tests."""
        ...

    def conclude_test(self, test_name: str) -> ABTestResult | None:
        """
        Conclude a test and return final results.

        Args:
            test_name: Name of the test

        Returns:
            Final ABTestResult or None if insufficient data
        """
        ...


# =============================================================================
# Learning Level Enum
# =============================================================================

class LearningLevel(Enum):
    """
    Hierarchical learning levels.

    STRATEGY: Which attack type works? (12 arms)
    PAYLOAD: Which payload variants succeed? (per-strategy arms)
    MUTATION: Which mutations improve payloads? (10 mutation types)
    """
    STRATEGY = "strategy"
    PAYLOAD = "payload"
    MUTATION = "mutation"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type variables
    'T',
    'ContextT',

    # Data classes
    'LearningResult',
    'HeatScore',
    'ContextKey',
    'ABTestResult',

    # Protocols
    'LearnerProtocol',
    'BanditProtocol',
    'ContextualBanditProtocol',
    'HeatTrackerProtocol',
    'ABTestProtocol',

    # Enums
    'LearningLevel',
]
