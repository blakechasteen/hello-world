# HoloLoom/ralph/context_monitor.py
"""
Ralph Loop Engine - Context Window Monitor.

Monitors context window usage and triggers automatic resets
when thresholds are exceeded. Also supports the "keep at 60k"
ceiling trick for proactive context trimming.

"When the context fills up, you get a fresh agent with fresh context,
picking up where the last one left off."
- Geoff Huntley

Created: 2026-01-28
Updated: 2026-02-05 - Added context ceiling enforcement
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import asyncio
import logging
import time


logger = logging.getLogger(__name__)


class ResetTrigger(Enum):
    """What triggered a context reset."""

    THRESHOLD_WARNING = "threshold_warning"
    THRESHOLD_RESET = "threshold_reset"
    THRESHOLD_CRITICAL = "threshold_critical"
    MANUAL = "manual"
    ERROR_LIMIT = "error_limit"
    TIMEOUT = "timeout"
    CONSOLIDATION = "consolidation"
    CEILING_TRIM = "ceiling_trim"


@dataclass
class ContextUsage:
    """Snapshot of context window usage."""

    timestamp: str
    estimated_tokens: int
    estimated_percent: float
    trigger_level: Optional[ResetTrigger] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def exceeds_warning(self, threshold: float) -> bool:
        return self.estimated_percent >= threshold

    def exceeds_reset(self, threshold: float) -> bool:
        return self.estimated_percent >= threshold

    def exceeds_critical(self, threshold: float) -> bool:
        return self.estimated_percent >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "estimated_tokens": self.estimated_tokens,
            "estimated_percent": self.estimated_percent,
            "trigger_level": self.trigger_level.value if self.trigger_level else None,
            "details": self.details,
        }


@dataclass
class CeilingAction:
    """Describes a context ceiling trim action taken.

    Returned by ContextMonitor.check_ceiling() when the ceiling
    is approached and content needs to be trimmed.

    Added: 2026-02-05
    """

    triggered: bool
    tokens_before: int
    tokens_after: int
    tokens_trimmed: int
    strategy_used: str
    categories_trimmed: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    @property
    def trim_percent(self) -> float:
        """What fraction of tokens were trimmed."""
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_trimmed / self.tokens_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_trimmed": self.tokens_trimmed,
            "trim_percent": self.trim_percent,
            "strategy_used": self.strategy_used,
            "categories_trimmed": self.categories_trimmed,
            "timestamp": self.timestamp,
        }


@dataclass
class AutoResetConfig:
    """Configuration for automatic context reset."""

    # Enable auto-reset
    enabled: bool = True

    # Thresholds (as fractions 0-1)
    warning_threshold: float = 0.60
    consolidation_threshold: float = 0.75
    reset_threshold: float = 0.85
    critical_threshold: float = 0.95

    # Estimated context window size (tokens)
    context_window_size: int = 150_000

    # Reserved tokens for response generation
    response_reserve: int = 10_000

    # How often to check (iterations)
    check_interval: int = 1

    # Actions on threshold breaches
    consolidate_on_warning: bool = True
    reset_on_threshold: bool = True
    force_reset_on_critical: bool = True

    # Callbacks
    on_warning: Optional[Callable] = None
    on_consolidation: Optional[Callable] = None
    on_reset: Optional[Callable] = None
    on_critical: Optional[Callable] = None

    @classmethod
    def default(cls) -> "AutoResetConfig":
        return cls()

    @classmethod
    def conservative(cls) -> "AutoResetConfig":
        """Conservative settings - reset early."""
        return cls(
            warning_threshold=0.50,
            consolidation_threshold=0.60,
            reset_threshold=0.70,
            critical_threshold=0.80,
        )

    @classmethod
    def aggressive(cls) -> "AutoResetConfig":
        """Aggressive settings - maximize context usage."""
        return cls(
            warning_threshold=0.70,
            consolidation_threshold=0.80,
            reset_threshold=0.90,
            critical_threshold=0.95,
        )


class ContextEstimator:
    """
    Estimates context window usage.

    Uses heuristics to estimate tokens consumed by:
    - Conversation history
    - System prompts
    - Retrieved memories
    - Tool outputs
    """

    def __init__(
        self,
        context_window_size: int = 150_000,
        chars_per_token: float = 4.0,  # Rough estimate
    ):
        self.context_window_size = context_window_size
        self.chars_per_token = chars_per_token

        # Tracking
        self._message_chars: int = 0
        self._system_chars: int = 0
        self._memory_chars: int = 0
        self._tool_chars: int = 0

    def add_message(self, content: str):
        """Track a message added to context."""
        self._message_chars += len(content)

    def add_system(self, content: str):
        """Track system prompt content."""
        self._system_chars += len(content)

    def add_memory(self, content: str):
        """Track memory/retrieval content."""
        self._memory_chars += len(content)

    def add_tool_output(self, content: str):
        """Track tool output content."""
        self._tool_chars += len(content)

    def reset_tracking(self):
        """Reset all tracking (for new context window)."""
        self._message_chars = 0
        self._system_chars = 0
        self._memory_chars = 0
        self._tool_chars = 0

    def estimate_tokens(self) -> int:
        """Estimate total tokens in context."""
        total_chars = (
            self._message_chars
            + self._system_chars
            + self._memory_chars
            + self._tool_chars
        )
        return int(total_chars / self.chars_per_token)

    def estimate_percent(self) -> float:
        """Estimate context usage as percentage."""
        tokens = self.estimate_tokens()
        return tokens / self.context_window_size

    def get_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of context usage."""
        total = self.estimate_tokens()
        return {
            "total_tokens": total,
            "total_percent": self.estimate_percent(),
            "breakdown": {
                "messages": int(self._message_chars / self.chars_per_token),
                "system": int(self._system_chars / self.chars_per_token),
                "memory": int(self._memory_chars / self.chars_per_token),
                "tools": int(self._tool_chars / self.chars_per_token),
            },
            "remaining_tokens": self.context_window_size - total,
        }

    def prune_to_target(
        self,
        target_tokens: int,
        prune_ratio: float = 0.30,
        prune_order: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Prune tracked content to reach target token count.

        Removes content from categories in reverse prune_order
        (last in order = highest value = pruned last).

        Args:
            target_tokens: Target token count after pruning
            prune_ratio: Fraction of each category to prune per pass
            prune_order: Categories in order of importance
                         (first = most important, last = least important)

        Returns:
            Dict of category -> tokens_removed

        Added: 2026-02-05
        """
        if prune_order is None:
            prune_order = ["system", "messages", "memory", "tools"]

        current_tokens = self.estimate_tokens()
        if current_tokens <= target_tokens:
            return {}

        tokens_to_remove = current_tokens - target_tokens
        removed: Dict[str, int] = {}

        # Map category names to internal char counters
        category_attrs = {
            "messages": "_message_chars",
            "system": "_system_chars",
            "memory": "_memory_chars",
            "tools": "_tool_chars",
        }

        # Prune in reverse order (least important first)
        for category in reversed(prune_order):
            if tokens_to_remove <= 0:
                break

            attr = category_attrs.get(category)
            if attr is None:
                continue

            current_chars = getattr(self, attr)
            if current_chars == 0:
                continue

            # Calculate chars to remove from this category
            chars_to_remove = int(current_chars * prune_ratio)
            tokens_from_this = int(chars_to_remove / self.chars_per_token)

            # Don't remove more than we need
            if tokens_from_this > tokens_to_remove:
                chars_to_remove = int(tokens_to_remove * self.chars_per_token)
                tokens_from_this = tokens_to_remove

            # Don't remove more than exists
            chars_to_remove = min(chars_to_remove, current_chars)
            tokens_from_this = int(chars_to_remove / self.chars_per_token)

            setattr(self, attr, current_chars - chars_to_remove)
            removed[category] = tokens_from_this
            tokens_to_remove -= tokens_from_this

        return removed


class ContextMonitor:
    """
    Monitors context window usage and triggers actions.

    Integrates with Ralph engine to provide automatic context management:
    - Warns when usage exceeds warning threshold
    - Triggers consolidation when usage exceeds consolidation threshold
    - Triggers reset when usage exceeds reset threshold
    - Forces immediate reset when usage exceeds critical threshold
    """

    def __init__(
        self,
        config: Optional[AutoResetConfig] = None,
        estimator: Optional[ContextEstimator] = None,
    ):
        self.config = config or AutoResetConfig.default()
        self.estimator = estimator or ContextEstimator(
            context_window_size=self.config.context_window_size
        )

        # State
        self._usage_history: List[ContextUsage] = []
        self._last_check_time: float = 0
        self._warning_emitted: bool = False
        self._consolidation_triggered: bool = False

    def check(self) -> ContextUsage:
        """
        Check current context usage and determine if action needed.

        Returns ContextUsage with trigger_level set if threshold exceeded.
        """
        usage = ContextUsage(
            timestamp=datetime.utcnow().isoformat(),
            estimated_tokens=self.estimator.estimate_tokens(),
            estimated_percent=self.estimator.estimate_percent(),
            details=self.estimator.get_breakdown(),
        )

        # Check thresholds (highest first)
        if usage.exceeds_critical(self.config.critical_threshold):
            usage.trigger_level = ResetTrigger.THRESHOLD_CRITICAL
            if self.config.on_critical:
                try:
                    self.config.on_critical(usage)
                except Exception as e:
                    logger.warning(f"Critical callback failed: {e}")

        elif usage.exceeds_reset(self.config.reset_threshold):
            usage.trigger_level = ResetTrigger.THRESHOLD_RESET
            if self.config.on_reset:
                try:
                    self.config.on_reset(usage)
                except Exception as e:
                    logger.warning(f"Reset callback failed: {e}")

        elif usage.exceeds_warning(self.config.consolidation_threshold):
            if not self._consolidation_triggered:
                usage.trigger_level = ResetTrigger.CONSOLIDATION
                self._consolidation_triggered = True
                if self.config.on_consolidation:
                    try:
                        self.config.on_consolidation(usage)
                    except Exception as e:
                        logger.warning(f"Consolidation callback failed: {e}")

        elif usage.exceeds_warning(self.config.warning_threshold):
            if not self._warning_emitted:
                usage.trigger_level = ResetTrigger.THRESHOLD_WARNING
                self._warning_emitted = True
                if self.config.on_warning:
                    try:
                        self.config.on_warning(usage)
                    except Exception as e:
                        logger.warning(f"Warning callback failed: {e}")

        # Track history
        self._usage_history.append(usage)
        self._last_check_time = time.time()

        return usage

    def should_reset(self) -> bool:
        """Check if reset should be triggered based on latest usage."""
        if not self._usage_history:
            return False

        latest = self._usage_history[-1]
        if latest.trigger_level in (
            ResetTrigger.THRESHOLD_RESET,
            ResetTrigger.THRESHOLD_CRITICAL,
        ):
            return True
        return False

    def should_consolidate(self) -> bool:
        """Check if consolidation should be triggered."""
        if not self._usage_history:
            return False

        latest = self._usage_history[-1]
        return latest.trigger_level == ResetTrigger.CONSOLIDATION

    def reset_tracking(self):
        """Reset tracking for new context window."""
        self.estimator.reset_tracking()
        self._warning_emitted = False
        self._consolidation_triggered = False

    def get_usage_history(self) -> List[Dict[str, Any]]:
        """Get usage history for analysis."""
        return [u.to_dict() for u in self._usage_history]

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current usage summary."""
        if not self._usage_history:
            return {"status": "no_data"}

        latest = self._usage_history[-1]
        return {
            "timestamp": latest.timestamp,
            "tokens": latest.estimated_tokens,
            "percent": latest.estimated_percent,
            "trigger": latest.trigger_level.value if latest.trigger_level else None,
            "warning_threshold": self.config.warning_threshold,
            "reset_threshold": self.config.reset_threshold,
            "remaining_before_reset": (
                self.config.reset_threshold - latest.estimated_percent
            ),
        }

    def check_ceiling(
        self,
        ceiling_tokens: int,
        headroom: float = 0.10,
        prune_ratio: float = 0.30,
        prune_order: Optional[List[str]] = None,
        strategy: str = "hybrid",
    ) -> CeilingAction:
        """
        Check if context exceeds the ceiling and trim if needed.

        This implements the "keep at 60k" trick: instead of waiting for
        a full reset, proactively trim context to stay within the optimal
        performance zone.

        Args:
            ceiling_tokens: Maximum token count to maintain
            headroom: Fraction of ceiling used as buffer (start trimming at
                      ceiling * (1 - headroom))
            prune_ratio: Fraction of each category to prune per pass
            prune_order: Categories in order of importance
            strategy: "prune", "summarize", or "hybrid"

        Returns:
            CeilingAction describing what was done

        Added: 2026-02-05
        """
        current_tokens = self.estimator.estimate_tokens()
        trim_threshold = int(ceiling_tokens * (1.0 - headroom))

        if current_tokens < trim_threshold:
            return CeilingAction(
                triggered=False,
                tokens_before=current_tokens,
                tokens_after=current_tokens,
                tokens_trimmed=0,
                strategy_used=strategy,
            )

        # Calculate target after trimming (with extra headroom)
        target_after = int(ceiling_tokens * (1.0 - headroom * 2))

        logger.info(
            f"Context ceiling trim: {current_tokens} tokens "
            f"(threshold: {trim_threshold}, target: {target_after})"
        )

        # Execute the prune
        categories_trimmed = self.estimator.prune_to_target(
            target_tokens=target_after,
            prune_ratio=prune_ratio,
            prune_order=prune_order,
        )

        tokens_after = self.estimator.estimate_tokens()
        tokens_trimmed = current_tokens - tokens_after

        action = CeilingAction(
            triggered=True,
            tokens_before=current_tokens,
            tokens_after=tokens_after,
            tokens_trimmed=tokens_trimmed,
            strategy_used=strategy,
            categories_trimmed=categories_trimmed,
        )

        # Record in usage history as a ceiling trim event
        usage = ContextUsage(
            timestamp=action.timestamp,
            estimated_tokens=tokens_after,
            estimated_percent=self.estimator.estimate_percent(),
            trigger_level=ResetTrigger.CEILING_TRIM,
            details={
                "ceiling_action": action.to_dict(),
            },
        )
        self._usage_history.append(usage)

        logger.info(
            f"Ceiling trim complete: {tokens_trimmed} tokens removed "
            f"({action.trim_percent:.1%}), now at {tokens_after} tokens"
        )

        return action

    def should_trim(self, ceiling_tokens: int, headroom: float = 0.10) -> bool:
        """Check if context is approaching the ceiling and needs trimming.

        Added: 2026-02-05
        """
        current_tokens = self.estimator.estimate_tokens()
        trim_threshold = int(ceiling_tokens * (1.0 - headroom))
        return current_tokens >= trim_threshold

    def get_ceiling_status(
        self, ceiling_tokens: int, headroom: float = 0.10
    ) -> Dict[str, Any]:
        """Get current status relative to the ceiling.

        Added: 2026-02-05
        """
        current_tokens = self.estimator.estimate_tokens()
        trim_threshold = int(ceiling_tokens * (1.0 - headroom))
        target_after = int(ceiling_tokens * (1.0 - headroom * 2))

        return {
            "ceiling_tokens": ceiling_tokens,
            "current_tokens": current_tokens,
            "trim_threshold": trim_threshold,
            "target_after_trim": target_after,
            "tokens_until_trim": max(0, trim_threshold - current_tokens),
            "ceiling_percent": current_tokens / ceiling_tokens if ceiling_tokens > 0 else 0.0,
            "needs_trim": current_tokens >= trim_threshold,
            "headroom": headroom,
        }

    def get_projection(self, tokens_per_iteration: int = 5000) -> Dict[str, Any]:
        """
        Project when reset will be needed based on current rate.

        Args:
            tokens_per_iteration: Estimated tokens consumed per iteration

        Returns:
            Projection of iterations until thresholds
        """
        current_tokens = self.estimator.estimate_tokens()
        reset_tokens = int(
            self.config.context_window_size * self.config.reset_threshold
        )
        critical_tokens = int(
            self.config.context_window_size * self.config.critical_threshold
        )

        remaining_to_reset = reset_tokens - current_tokens
        remaining_to_critical = critical_tokens - current_tokens

        return {
            "current_tokens": current_tokens,
            "current_percent": self.estimator.estimate_percent(),
            "reset_threshold_tokens": reset_tokens,
            "critical_threshold_tokens": critical_tokens,
            "tokens_until_reset": remaining_to_reset,
            "tokens_until_critical": remaining_to_critical,
            "iterations_until_reset": (
                remaining_to_reset // tokens_per_iteration if tokens_per_iteration > 0 else float("inf")
            ),
            "iterations_until_critical": (
                remaining_to_critical // tokens_per_iteration if tokens_per_iteration > 0 else float("inf")
            ),
        }


def create_context_monitor(
    context_window_size: int = 150_000,
    conservative: bool = False,
    on_warning: Optional[Callable] = None,
    on_reset: Optional[Callable] = None,
) -> ContextMonitor:
    """
    Create a context monitor with common configuration.

    Args:
        context_window_size: Estimated context window (tokens)
        conservative: Use conservative thresholds (reset earlier)
        on_warning: Callback when warning threshold exceeded
        on_reset: Callback when reset threshold exceeded
    """
    config_class = AutoResetConfig.conservative if conservative else AutoResetConfig.default
    config = config_class()
    config.context_window_size = context_window_size
    config.on_warning = on_warning
    config.on_reset = on_reset

    return ContextMonitor(config=config)


def create_ceiling_monitor(
    ceiling_tokens: int = 60_000,
    context_window_size: int = 150_000,
    on_warning: Optional[Callable] = None,
    on_reset: Optional[Callable] = None,
) -> ContextMonitor:
    """
    Create a context monitor with ceiling enforcement enabled.

    Convenience function for the "keep at 60k" pattern.

    Args:
        ceiling_tokens: Target ceiling in tokens (default: 60,000)
        context_window_size: Full context window size (tokens)
        on_warning: Callback when warning threshold exceeded
        on_reset: Callback when reset threshold exceeded

    Added: 2026-02-05
    """
    config = AutoResetConfig.default()
    config.context_window_size = context_window_size
    config.on_warning = on_warning
    config.on_reset = on_reset

    return ContextMonitor(config=config)
