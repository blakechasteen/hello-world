"""
Ephemeral Memory Pressure Policy — 2026-02-09

Tracks pressure signals, computes tier, and applies tier-specific policies.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import PressureSignals, PressureStatus, PressureTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy table: what each tier does
# ---------------------------------------------------------------------------

_TIER_POLICIES: dict[PressureTier, dict[str, Any]] = {
    PressureTier.TIER0: {
        "priming_strategy": "full",
        "min_importance": 0.0,
        "semantic_fallback_enabled": True,
        "per_query_token_cap": None,  # no cap
        "plan_write_through": True,  # always
    },
    PressureTier.TIER1: {
        "priming_strategy": "reduced",
        "min_importance": 0.2,
        "semantic_fallback_enabled": True,
        "per_query_token_cap": None,
        "plan_write_through": True,
    },
    PressureTier.TIER2: {
        "priming_strategy": "minimal",
        "min_importance": 0.4,
        "semantic_fallback_enabled": False,  # disabled at Tier2+
        "per_query_token_cap": 2000,
        "plan_write_through": True,
    },
    PressureTier.TIER3: {
        "priming_strategy": "emergency",
        "min_importance": 0.6,
        "semantic_fallback_enabled": False,
        "per_query_token_cap": 500,  # strict
        "plan_write_through": True,
    },
}


class PressureEngine:
    """
    Tracks memory pressure signals and computes policy responses.

    Usage:
        engine = PressureEngine(context_window=128_000)
        engine.update(context_tokens_used=50_000)
        status = engine.status()
        policy = engine.current_policy()
    """

    def __init__(self, context_window: int = 128_000):
        self._signals = PressureSignals(context_window_size=context_window)

    @property
    def signals(self) -> PressureSignals:
        return self._signals

    @property
    def tier(self) -> PressureTier:
        return self._signals.tier

    def update(
        self,
        *,
        context_tokens_used: int | None = None,
        retrieval_result_count: int | None = None,
        retrieval_latency_ms: float | None = None,
        promotion_queue_depth: int | None = None,
        decay_floor_population: int | None = None,
    ) -> PressureTier:
        """Update tracked signals and return current tier."""
        if context_tokens_used is not None:
            self._signals.context_tokens_used = context_tokens_used
            if self._signals.context_window_size > 0:
                self._signals.context_utilization_pct = (
                    context_tokens_used / self._signals.context_window_size
                ) * 100.0
        if retrieval_result_count is not None:
            self._signals.retrieval_result_count = retrieval_result_count
        if retrieval_latency_ms is not None:
            self._signals.retrieval_latency_ms = retrieval_latency_ms
        if promotion_queue_depth is not None:
            self._signals.promotion_queue_depth = promotion_queue_depth
        if decay_floor_population is not None:
            self._signals.decay_floor_population = decay_floor_population

        new_tier = self._signals.tier
        logger.debug("Pressure updated: tier=%s util=%.1f%%", new_tier.value, self._signals.context_utilization_pct)
        return new_tier

    def current_policy(self) -> dict[str, Any]:
        """Return the active policy dict for the current tier."""
        return dict(_TIER_POLICIES.get(self.tier, _TIER_POLICIES[PressureTier.TIER0]))

    def status(self) -> PressureStatus:
        """Return full pressure status (for memory://pressure resource)."""
        return PressureStatus(
            tier=self.tier,
            signals=self._signals.model_copy(),
            policy=self.current_policy(),
        )

    def get_token_cap(self, requested: int | None = None) -> int | None:
        """
        Return the effective per-query token cap.
        If the tier enforces a cap, return the minimum of the tier cap and the requested cap.
        """
        policy = self.current_policy()
        tier_cap = policy.get("per_query_token_cap")
        if tier_cap is None and requested is None:
            return None
        if tier_cap is None:
            return requested
        if requested is None:
            return tier_cap
        return min(tier_cap, requested)

    def is_semantic_allowed(self) -> bool:
        """Check if semantic fallback is allowed at current pressure."""
        return self.current_policy().get("semantic_fallback_enabled", True)

    def min_importance(self) -> float:
        """Return the minimum importance threshold for the current tier."""
        return self.current_policy().get("min_importance", 0.0)
