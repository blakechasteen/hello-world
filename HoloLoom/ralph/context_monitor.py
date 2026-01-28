# HoloLoom/ralph/context_monitor.py
"""
Ralph Loop Engine - Context Window Monitor.

Monitors context window usage and triggers automatic resets
when thresholds are exceeded.

"When the context fills up, you get a fresh agent with fresh context,
picking up where the last one left off."
- Geoff Huntley

Created: 2026-01-28
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
