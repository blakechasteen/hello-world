"""
Feedback Protocol — universal types for multi-temporal feedback.

Layer 0: Zero HoloLoom dependencies. Defines the contracts that the entire
feedback system builds on.

The key insight: feedback operates across multiple timescales simultaneously.
A sub-second bus signal and a week-long federation promotion are both feedback —
they just have different weights and decay rates. This protocol unifies them
into a single type that the rest of the system can consume.

Timescales:
    SUB_SECOND  (<1s)   — bus signals, attention shifts
    SECONDS     (1-60s) — per-interaction rewards, tool outcomes
    MINUTES     (1-60m) — reflection analysis, background consolidation
    HOURS       (1-24h) — wave consolidation, session summaries
    DAYS        (1-7d)  — federation promotions, note quality
    WEEKS       (7d+)   — department health, long-term drift

Author: Claude Code (with HoloLoom by Blake)
Date: 2026-03-19
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable


# ============================================================================
# Timescale Taxonomy
# ============================================================================

class Timescale(str, Enum):
    """Temporal resolution of a feedback signal."""
    SUB_SECOND = "sub_second"   # <1s: bus signals, attention
    SECONDS = "seconds"          # 1-60s: per-interaction rewards
    MINUTES = "minutes"          # 1-60min: reflection, background
    HOURS = "hours"              # 1-24h: wave consolidation
    DAYS = "days"                # 1-7d: federation promotions
    WEEKS = "weeks"              # 7d+: department health


TIMESCALE_WEIGHT: dict[Timescale, float] = {
    Timescale.SUB_SECOND: 0.05,
    Timescale.SECONDS: 0.15,
    Timescale.MINUTES: 0.30,
    Timescale.HOURS: 0.60,
    Timescale.DAYS: 0.85,
    Timescale.WEEKS: 1.00,
}
"""Longer timescales carry more weight — they represent more integrated evidence."""


# ============================================================================
# Core Data Structure
# ============================================================================

@dataclass
class FeedbackSignal:
    """
    A single feedback observation at a specific timescale.

    Normalized to [-1, 1] on a named dimension. Carries provenance
    (source, causation chain) so credit assignment can trace outcomes
    back to the decisions that caused them.
    """
    source: str               # who emitted: "model_router", "federation_tracker", etc.
    timescale: Timescale
    subject: str              # what was judged: tool name, model id, note path
    dimension: str            # quality axis: "quality", "speed", "accuracy", "relevance"
    value: float              # [-1, 1] normalized score
    confidence: float = 0.5   # [0, 1] emitter's confidence in this signal
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    causation_chain: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def effective_weight(self) -> float:
        """Weight combining timescale importance and emitter confidence."""
        return TIMESCALE_WEIGHT.get(self.timescale, 0.3) * self.confidence

    def to_dict(self) -> dict:
        """Serialize for persistence or bus transport."""
        return {
            "signal_id": self.signal_id,
            "source": self.source,
            "timescale": self.timescale.value,
            "subject": self.subject,
            "dimension": self.dimension,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "causation_chain": list(self.causation_chain),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackSignal:
        """Deserialize from dict."""
        return cls(
            signal_id=data.get("signal_id", uuid.uuid4().hex[:12]),
            source=data["source"],
            timescale=Timescale(data["timescale"]),
            subject=data["subject"],
            dimension=data["dimension"],
            value=data["value"],
            confidence=data.get("confidence", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            causation_chain=data.get("causation_chain", []),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class FeedbackSource(Protocol):
    """Something that produces feedback signals (a bandit, a tracker, etc.)."""

    @property
    def source_id(self) -> str: ...

    @property
    def timescale(self) -> Timescale: ...

    def collect(self) -> list[FeedbackSignal]: ...


@runtime_checkable
class FeedbackConsumer(Protocol):
    """Something that acts on feedback signals (updates bandits, adjusts weights)."""

    @property
    def consumer_id(self) -> str: ...

    def apply_feedback(self, signals: list[FeedbackSignal]) -> None: ...

    def accepts(self, signal: FeedbackSignal) -> bool: ...
