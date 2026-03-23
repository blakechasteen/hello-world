"""
ThreadPrimer — pre-inlined context for ritual execution.
=========================================================

Primes the warp before the shuttle passes. The snapshot is tensioned
context — historical outcomes, skill hints, domain priors — injected
into RitualContext before BODY_START so ritual bodies never orient
themselves.

Protocol-based: swap JournalSnapshotBuilder for any SnapshotBuilder.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ContextSnapshot:
    """Pre-inlined context. Tensioned warp threads ready to be woven against."""
    recent_outcomes: list[dict] = field(default_factory=list)
    session_history: list[dict] = field(default_factory=list)
    skill_hint: dict[str, Any] | None = None
    domain_priors: dict[str, float] = field(default_factory=dict)
    token_cost: int = 0


class SnapshotBuilder(Protocol):
    """Protocol for building context snapshots."""
    def build(
        self,
        ctx: Any,                      # RitualContext
        active_rules: list[Any],       # list[ActiveRule]
        budget_tokens: int,
    ) -> ContextSnapshot: ...


class JournalSnapshotBuilder:
    """
    Builds snapshots from the RitualJournal.

    Sources in priority order (greedy fill until budget exhausted):
      1. Journal replay — last N events from current session
      2. SkillMemory match — Thompson-sample best playbook
      3. Confidence history — last 10 values
      4. Domain priors — from Warp semantic search
      5. Ritual outputs — already-produced Bobbins

    Active rules shape retrieval:
      - recall keywords prioritize specific topics
      - exclude keywords skip domains
    """

    def __init__(
        self,
        journal: Any = None,          # RitualJournal
        skill_extractor: Any = None,   # SkillExtractor
        max_recent: int = 5,
    ):
        self._journal = journal
        self._skill_extractor = skill_extractor
        self._max_recent = max_recent

    def build(
        self,
        ctx: Any,
        active_rules: list[Any] | None = None,
        budget_tokens: int = 200,
    ) -> ContextSnapshot:
        """
        Build a context snapshot from available sources.

        Respects budget_tokens — fills greedily in priority order,
        stops when budget exhausted.
        """
        snapshot = ContextSnapshot()
        tokens_remaining = budget_tokens

        # 1. Session history from journal
        if self._journal is not None and tokens_remaining > 0:
            try:
                events = self._journal.replay_session(
                    ctx.session_id, limit=self._max_recent * 2,
                )
                snapshot.session_history = [
                    {
                        "ritual_id": e.ritual_id,
                        "event_type": e.event_type,
                        "payload": e.payload,
                    }
                    for e in events[-self._max_recent:]
                ]
                tokens_remaining -= len(snapshot.session_history) * 20  # estimate
            except Exception as e:
                logger.debug("Journal replay failed in snapshot: %s", e)

        # 2. Skill hint from extractor
        if self._skill_extractor is not None and tokens_remaining > 0:
            try:
                skill = self._skill_extractor.select_skill(
                    ctx.task_intent,
                    domain=ctx.domain,
                )
                if skill is not None:
                    snapshot.skill_hint = skill.to_dict()
                    tokens_remaining -= 50  # estimate
            except Exception as e:
                logger.debug("Skill extraction failed in snapshot: %s", e)

        # 3. Confidence history
        if hasattr(ctx, "confidence_history") and tokens_remaining > 0:
            snapshot.domain_priors[ctx.domain] = ctx.confidence
            tokens_remaining -= 5

        snapshot.token_cost = budget_tokens - tokens_remaining
        return snapshot
