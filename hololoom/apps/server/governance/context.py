"""
GovernanceContext — shared state flowing through the middleware chain.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GovernanceContext:
    """Immutable-ish bag carried through every middleware layer."""

    # ── Input (set once at request entry) ──
    query: str
    mode: str = "default"
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.perf_counter)
    request_metadata: dict = field(default_factory=dict)

    # ── Accumulated by middleware ──
    rbac_decision: str | None = None        # "ALLOW" / "DENY" / …
    safety_evaluation: Any | None = None     # SafetyResult from guardrails
    reasoning_result: Any | None = None      # orchestrator.reason() output
    response: Any | None = None              # final HTTP-ready response

    # ── Open annotations bag (middleware can stash anything) ──
    annotations: dict = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000
