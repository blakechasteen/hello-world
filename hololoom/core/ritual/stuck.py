"""
SnagWatcher — stuck pattern detection over journal events.
============================================================

A snag is when the weft catches, the shuttle jams, and the loom
can't advance. SnagWatcher inspects the cloth after each pass,
looking for repeated tangles.

Four signals:
  OSCILLATION — A→B→A→B cycle in ritual outcomes
  REPETITION  — same ritual failed N times consecutively
  PARALYSIS   — N dispatches with empty produces
  RUNAWAY     — session event count exceeds ceiling
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StuckSignal(Enum):
    OSCILLATION = "oscillation"
    REPETITION = "repetition"
    PARALYSIS = "paralysis"
    RUNAWAY = "runaway"


@dataclass
class StuckDetection:
    """A detected stuck pattern with evidence and recommended action."""
    signal: StuckSignal
    severity: float                 # 0.0–1.0
    evidence: list[str] = field(default_factory=list)  # journal event IDs
    recommendation: str = "defer"   # "halt" | "escalate" | "switch_mode" | "defer"
    details: str = ""


class StuckDetector:
    """Protocol for stuck detection. Implement for custom detection logic."""

    def check(
        self,
        journal: Any,     # RitualJournal
        ctx: Any,          # RitualContext
    ) -> StuckDetection | None:
        """Check for stuck patterns. Returns detection or None."""
        ...


class SlidingWindowDetector:
    """
    Default stuck detector using sliding window over journal events.

    Checks four patterns against the most recent events in the session.
    """

    def __init__(
        self,
        window_size: int = 8,
        repetition_threshold: int = 3,
        paralysis_threshold: int = 5,
        max_session_events: int = 500,
    ):
        self._window_size = window_size
        self._repetition_threshold = repetition_threshold
        self._paralysis_threshold = paralysis_threshold
        self._max_session_events = max_session_events

    def check(
        self,
        journal: Any,
        ctx: Any,
    ) -> StuckDetection | None:
        """
        Check the last N journal events for stuck patterns.
        Returns the most severe detection, or None.
        """
        if journal is None:
            return None

        try:
            events = journal.replay_session(ctx.session_id, limit=self._window_size * 2)
        except Exception as e:
            logger.debug("Journal replay failed in stuck check: %s", e)
            return None

        if not events:
            return None

        # Check RUNAWAY first (session-level, cheapest)
        total = journal.count_events(session_id=ctx.session_id)
        if total > self._max_session_events:
            return StuckDetection(
                signal=StuckSignal.RUNAWAY,
                severity=1.0,
                recommendation="halt",
                details=f"Session has {total} events (ceiling: {self._max_session_events})",
            )

        recent = events[-self._window_size:]

        # Check OSCILLATION — A→B→A→B pattern
        oscillation = self._check_oscillation(recent)
        if oscillation is not None:
            return oscillation

        # Check REPETITION — same (ritual_id, BODY_FAILED) N times
        repetition = self._check_repetition(recent)
        if repetition is not None:
            return repetition

        # Check PARALYSIS — consecutive empty produces
        paralysis = self._check_paralysis(recent)
        if paralysis is not None:
            return paralysis

        return None

    def _check_oscillation(self, events: list) -> StuckDetection | None:
        """Detect A→B→A→B pattern in (ritual_id, event_type) pairs."""
        if len(events) < 4:
            return None

        signatures = [(e.ritual_id, e.event_type) for e in events]
        # Check for 2 full cycles: [A, B, A, B]
        for i in range(len(signatures) - 3):
            a, b = signatures[i], signatures[i + 1]
            if a != b and signatures[i + 2] == a and signatures[i + 3] == b:
                return StuckDetection(
                    signal=StuckSignal.OSCILLATION,
                    severity=0.7,
                    evidence=[e.id for e in events[i:i + 4]],
                    recommendation="switch_mode",
                    details=f"Oscillation: {a} ↔ {b}",
                )
        return None

    def _check_repetition(self, events: list) -> StuckDetection | None:
        """Detect same ritual failing N times consecutively."""
        failures = [
            e for e in events
            if e.event_type in ("BODY_FAILED", "POLICY_INVOKED")
        ]
        if len(failures) < self._repetition_threshold:
            return None

        # Check last N failures are the same ritual
        recent_failures = failures[-self._repetition_threshold:]
        ritual_ids = {f.ritual_id for f in recent_failures}
        if len(ritual_ids) == 1:
            return StuckDetection(
                signal=StuckSignal.REPETITION,
                severity=0.6,
                evidence=[e.id for e in recent_failures],
                recommendation="escalate",
                details=f"Ritual '{recent_failures[0].ritual_id}' failed {len(recent_failures)}x",
            )
        return None

    def _check_paralysis(self, events: list) -> StuckDetection | None:
        """Detect N consecutive BODY_COMPLETE with empty produced."""
        completes = [e for e in events if e.event_type == "BODY_COMPLETE"]
        if len(completes) < self._paralysis_threshold:
            return None

        recent = completes[-self._paralysis_threshold:]
        all_empty = all(
            not e.payload.get("produced") for e in recent
        )
        if all_empty:
            return StuckDetection(
                signal=StuckSignal.PARALYSIS,
                severity=0.5,
                evidence=[e.id for e in recent],
                recommendation="escalate",
                details=f"{len(recent)} consecutive dispatches produced nothing",
            )
        return None
