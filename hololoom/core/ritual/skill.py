"""
SkillMemory — procedural learning from successful ritual completions.
=====================================================================

Thompson Sampling learns WHICH strategies work.
SkillMemory learns HOW to execute them.

Pattern from Hermes Agent: after a successful multi-step completion,
extract the sequence of rituals + actions as a replayable playbook.

SkillMemory is a Bobbin type. It's stored in Warp for semantic retrieval
and selected via Thompson Sampling (SOP-9).
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .journal import RitualJournal, BODY_COMPLETE, SKIP

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillStep:
    """One step in a procedural playbook."""
    action_type: str       # "ritual" | "tool_call" | "memory_query"
    action_id: str         # ritual_id, tool name, etc.
    parameters: dict[str, Any]
    expected_outcome: str
    duration_budget_ms: float = 0.0


@dataclass
class SkillMemory:
    """
    Procedural playbook learned from successful completions.
    Each skill has a Beta(success_count, failure_count) prior
    for Thompson Sampling selection among competing skills.
    """
    skill_id: str
    domain: str
    task_pattern: str              # intent text for matching
    steps: list[SkillStep]
    success_count: float = 1.0     # Beta α — start uniform
    failure_count: float = 1.0     # Beta β — start uniform
    avg_duration_ms: float = 0.0
    avg_confidence_delta: float = 0.0
    created_at: str = ""
    last_used: str = ""

    def sample(self) -> float:
        """Thompson sample from Beta(α, β) prior."""
        return random.betavariate(
            max(self.success_count, 0.01),
            max(self.failure_count, 0.01),
        )

    def update(self, success: bool, confidence_delta: float = 0.0) -> None:
        """Update the Beta prior with outcome."""
        if success:
            self.success_count += 1.0
        else:
            self.failure_count += 1.0
        # EMA blend for confidence delta
        alpha = 0.3
        self.avg_confidence_delta = (
            alpha * confidence_delta + (1 - alpha) * self.avg_confidence_delta
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "domain": self.domain,
            "task_pattern": self.task_pattern,
            "steps": [
                {
                    "action_type": s.action_type,
                    "action_id": s.action_id,
                    "parameters": s.parameters,
                    "expected_outcome": s.expected_outcome,
                    "duration_budget_ms": s.duration_budget_ms,
                }
                for s in self.steps
            ],
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_duration_ms": self.avg_duration_ms,
            "avg_confidence_delta": self.avg_confidence_delta,
        }


SKILLS_COLLECTION = "skill_memories"


class SkillExtractor:
    """
    Watches the ritual journal for successful sessions and
    synthesizes SkillMemory playbooks.

    Call extract_from_session() after a session ends with
    downstream_task_success=True.

    When warp is provided, skills are embedded for semantic retrieval.
    In-memory dict serves as cache + fallback when Warp is unavailable.
    """

    def __init__(self, journal: RitualJournal, warp: Any = None):
        self._journal = journal
        self._warp = warp
        self._skills: dict[str, SkillMemory] = {}  # cache + fallback

    def extract_from_session(
        self,
        session_id: str,
        domain: str,
        task_pattern: str,
        success: bool,
    ) -> SkillMemory | None:
        """
        Extract or update a skill from a completed session.

        Returns the SkillMemory if extraction succeeded, None otherwise.
        """
        events = self._journal.replay_session(session_id)
        if not events:
            return None

        # Extract completed ritual steps
        steps: list[SkillStep] = []
        for event in events:
            if event.event_type == BODY_COMPLETE:
                steps.append(SkillStep(
                    action_type="ritual",
                    action_id=event.ritual_id,
                    parameters=event.payload.get("produced", {}),
                    expected_outcome=event.payload.get("notes", "success"),
                    duration_budget_ms=event.payload.get("duration_ms", 0),
                ))

        if not steps:
            return None

        # Check for existing skill with same pattern
        existing = self._skills.get(task_pattern)
        if existing:
            existing.update(success)
            return existing

        # Create new skill
        skill = SkillMemory(
            skill_id=f"skill_{len(self._skills)}",
            domain=domain,
            task_pattern=task_pattern,
            steps=steps,
            success_count=2.0 if success else 1.0,
            failure_count=1.0 if success else 2.0,
        )
        self._skills[task_pattern] = skill
        self._embed_skill(skill)
        return skill

    def select_skill(
        self,
        task_intent: str,
        domain: str | None = None,
    ) -> SkillMemory | None:
        """
        Thompson-sample among matching skills (SOP-9).

        Tries Warp semantic search first, falls back to in-memory substring match.
        Returns the highest-scoring match, or None.
        """
        # Try Warp semantic search first
        if self._warp is not None:
            warp_candidates = self._search_warp(task_intent, domain)
            if warp_candidates:
                return max(warp_candidates, key=lambda s: s.sample())

        # Fallback: in-memory substring match
        return self._select_from_cache(task_intent, domain)

    def _select_from_cache(
        self,
        task_intent: str,
        domain: str | None = None,
    ) -> SkillMemory | None:
        """In-memory substring match fallback."""
        candidates = []
        for skill in self._skills.values():
            if domain and skill.domain != domain:
                continue
            if skill.task_pattern in task_intent or task_intent in skill.task_pattern:
                candidates.append(skill)

        if not candidates:
            return None

        return max(candidates, key=lambda s: s.sample())

    def _embed_skill(self, skill: SkillMemory) -> None:
        """Embed a skill to Warp for semantic retrieval. Fire-and-forget."""
        if self._warp is None:
            return
        try:
            from .memory import _hash_embedding, _text_to_qdrant_id
            text = f"{skill.domain} {skill.task_pattern} {' '.join(s.action_id for s in skill.steps)}"
            embedding = _hash_embedding(text)
            point_id = _text_to_qdrant_id(skill.skill_id)
            self._warp.upsert(
                collection_name=SKILLS_COLLECTION,
                points=[{
                    "id": point_id,
                    "vector": embedding,
                    "payload": skill.to_dict(),
                }],
            )
            logger.debug("Skill embedded to Warp: %s", skill.skill_id)
        except Exception as e:
            logger.warning("Failed to embed skill to Warp: %s", e)

    def _search_warp(
        self,
        task_intent: str,
        domain: str | None = None,
    ) -> list[SkillMemory]:
        """Search Warp for semantically matching skills."""
        try:
            from .memory import _hash_embedding
            query_vector = _hash_embedding(task_intent)
            filters = {}
            if domain:
                filters["domain"] = domain

            results = self._warp.search(
                collection_name=SKILLS_COLLECTION,
                query_vector=query_vector,
                limit=5,
                query_filter=filters if filters else None,
            )

            skills = []
            for hit in results:
                payload = hit.get("payload", hit) if isinstance(hit, dict) else getattr(hit, "payload", {})
                skills.append(SkillMemory(
                    skill_id=payload.get("skill_id", ""),
                    domain=payload.get("domain", ""),
                    task_pattern=payload.get("task_pattern", ""),
                    steps=[
                        SkillStep(**s) for s in payload.get("steps", [])
                    ],
                    success_count=payload.get("success_count", 1.0),
                    failure_count=payload.get("failure_count", 1.0),
                ))
            return skills
        except Exception as e:
            logger.debug("Warp skill search failed, falling back to cache: %s", e)
            return []

    @property
    def skill_count(self) -> int:
        return len(self._skills)
