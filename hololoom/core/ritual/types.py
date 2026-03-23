"""
Core data models for the Ritual Grammar system.

Ritual — first-class identity (what the behavior IS)
HookBinding — deployment configuration (WHEN it fires)
RitualContext — everything a ritual body needs at execution time
RitualResult — what a ritual returns after execution
RitualOutcome — the learning signal, written to Yarn and Warp
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


@dataclass
class RitualBudget:
    """Token and time ceiling for a ritual execution."""
    max_tokens: int
    max_seconds: float
    priority_class: str  # "critical" | "standard" | "best_effort"


@dataclass
class TokenBudget:
    """Agent-level token budget tracking."""
    total: int
    used: int

    @property
    def remaining(self) -> int:
        return self.total - self.used


class Priority(Enum):
    BLOCKING = 0   # halts pipeline until complete
    PARALLEL = 1   # runs concurrently, does not block
    DEFERRED = 2   # queued for async execution


@dataclass
class RitualResult:
    """What a ritual returns after execution."""
    success: bool
    produced: dict[str, Any]   # keyed by Bobbin type name
    confidence_delta: float
    notes: str = ""
    error: Exception | None = None


@dataclass
class RitualRecord:
    """Log entry for a ritual execution within a session."""
    ritual_id: str
    fired_at: datetime
    was_skipped: bool
    result: RitualResult | None


@dataclass
class RitualContext:
    """Everything a ritual body needs at execution time."""
    agent_id: str
    agent_role: str
    domain: str
    task_id: str
    task_type: str
    task_intent: str
    task_complexity: str   # "trivial" | "standard" | "complex"
    yarn: Any              # Neo4jHandle
    warp: Any              # QdrantHandle
    confidence: float
    confidence_history: list[float]
    ritual_outputs: dict[str, Any]
    ritual_log: list[RitualRecord]
    token_budget: TokenBudget
    time_budget: float
    escalation_path: list[str]
    session_id: str
    # v2 fields — ThreadPrimer, PatternCard, BindingObligation
    priming_context: dict[str, Any] = field(default_factory=dict)
    active_rules: list[Any] = field(default_factory=list)  # list[ActiveRule]
    closure_gate: Any = None  # ClosureGate


@dataclass
class Ritual:
    """
    First-class behavioral identity. Exists independently of when it fires.
    Do NOT embed trigger logic here — that belongs in HookBinding.
    """
    id: str
    version: str
    domain: str            # "coding" | "beekeeping" | "farming" | "universal"
    intent: str            # human-readable: what this ritual is FOR
    author: str            # "system" | "masterweaver" | agent role
    consumes: list[str]    # Bobbin type names required as input
    produces: list[str]    # Bobbin type names emitted as output
    failure_policy: Any    # FailurePolicy — use Any to avoid circular import
    budget: RitualBudget
    body: Callable[["RitualContext"], Awaitable["RitualResult"]]
    skip_condition: Callable[["RitualContext"], bool] | None = None
    compensation_ritual_id: str | None = None
    closure_required_by: str | None = None  # ritual_id that must run after this


@dataclass
class HookBinding:
    """
    Deployment configuration for a Ritual.
    Separating this from Ritual identity allows rebinding without changing the ritual.
    """
    ritual: Ritual
    trigger: Any           # HookEvent — use Any to avoid circular import
    condition: Callable[["RitualContext"], bool] | None
    priority: Priority


@dataclass
class RitualOutcome:
    """
    Written to Yarn as RitualHistoryNode after every execution.
    Embedded into Warp for semantic retrieval across sessions.
    This is the learning signal.
    """
    ritual_id: str
    ritual_version: str
    session_id: str
    agent_id: str
    task_context: str
    fired_at: datetime
    duration_ms: int
    tokens_used: int
    produced: dict[str, Any]
    downstream_task_success: bool | None
    confidence_delta: float
    was_skipped: bool
    skip_reason: str | None
    failure_policy_invoked: str | None
