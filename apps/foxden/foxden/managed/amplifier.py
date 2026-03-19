"""Amplifier — watches signal density and scales the group dynamically.

When a persona is finding a lot of issues in a specific area,
the amplifier spawns more agents focused on that area.
When a persona is finding nothing, it doesn't waste resources on more copies.

This is Thompson Sampling applied to test strategy: explore personas
that show promise, exploit the ones that are producing results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..core.persona import Persona
from .base import Manager
from .events import Event, EventBus, EventKind


@dataclass
class PersonaStats:
    """Bandit arm for a persona — tracks hit rate."""

    persona_name: str
    sessions_run: int = 0
    issues_found: int = 0
    # Beta distribution parameters (Thompson Sampling).
    alpha: float = 1.0  # successes + prior
    beta: float = 1.0  # failures + prior

    @property
    def hit_rate(self) -> float:
        if self.sessions_run == 0:
            return 0.0
        return self.issues_found / self.sessions_run

    def sample(self) -> float:
        """Thompson sample — draw from Beta(alpha, beta)."""
        return random.betavariate(self.alpha, self.beta)

    def record_success(self) -> None:
        self.alpha += 1

    def record_no_finding(self) -> None:
        self.beta += 1


@dataclass
class AmplifierConfig:
    """Knobs for the amplifier."""

    max_total_agents: int = 30  # hard cap
    min_issues_to_amplify: int = 2  # persona needs this many before scaling
    cooldown_sessions: int = 2  # wait this many completions between scale-ups
    explore_probability: float = 0.2  # chance of spawning an underexplored persona


class Amplifier(Manager):
    """Signal-density scaling via Thompson Sampling."""

    def __init__(
        self,
        bus: EventBus,
        config: AmplifierConfig | None = None,
    ):
        super().__init__(bus, backend=None)
        self.config = config or AmplifierConfig()
        self._stats: dict[str, PersonaStats] = {}
        self._total_spawned: int = 0
        self._sessions_since_last_scale: int = 0
        self._active_count: int = 0

    @property
    def name(self) -> str:
        return "amplifier"

    def attach(self) -> None:
        self.bus.on(EventKind.SESSION_START, self._on_start)
        self.bus.on(EventKind.SESSION_END, self._on_end)
        self.bus.on(EventKind.ISSUE_FOUND, self._on_issue)
        self._attached = True

    def _ensure_stats(self, persona: str) -> PersonaStats:
        if persona not in self._stats:
            self._stats[persona] = PersonaStats(persona_name=persona)
        return self._stats[persona]

    async def _on_start(self, event: Event) -> None:
        stats = self._ensure_stats(event.persona)
        stats.sessions_run += 1
        self._active_count += 1
        self._total_spawned += 1

    async def _on_end(self, event: Event) -> None:
        self._active_count -= 1
        self._sessions_since_last_scale += 1

        stats = self._ensure_stats(event.persona)
        session_issues = event.data.get("issues", 0)
        if session_issues > 0:
            stats.record_success()
        else:
            stats.record_no_finding()

        # Consider scaling.
        if self._sessions_since_last_scale >= self.config.cooldown_sessions:
            await self._consider_scale()

    async def _on_issue(self, event: Event) -> None:
        stats = self._ensure_stats(event.persona)
        stats.issues_found += 1

    async def _consider_scale(self) -> None:
        """Decide whether to spawn more agents and which persona to use."""
        if self._total_spawned >= self.config.max_total_agents:
            return
        if self._active_count >= self.config.max_total_agents:
            return

        # Thompson sample across all personas.
        candidates = [
            (name, stats.sample())
            for name, stats in self._stats.items()
        ]
        if not candidates:
            return

        # Sort by sample value (exploit) but sometimes explore.
        if random.random() < self.config.explore_probability:
            # Explore: pick the least-tested persona.
            candidates.sort(key=lambda x: self._stats[x[0]].sessions_run)
        else:
            # Exploit: pick the highest Thompson sample.
            candidates.sort(key=lambda x: x[1], reverse=True)

        best_name, best_score = candidates[0]
        best_stats = self._stats[best_name]

        # Only amplify if we have enough signal.
        if best_stats.issues_found < self.config.min_issues_to_amplify:
            return

        self._sessions_since_last_scale = 0

        await self._emit(
            EventKind.SPAWN_AGENT,
            persona=best_name,
            data={
                "reason": "amplification",
                "score": best_score,
                "hit_rate": best_stats.hit_rate,
                "sessions_run": best_stats.sessions_run,
            },
        )

    @property
    def leaderboard(self) -> list[dict[str, Any]]:
        """Ranked persona performance."""
        return sorted(
            [
                {
                    "persona": s.persona_name,
                    "sessions": s.sessions_run,
                    "issues": s.issues_found,
                    "hit_rate": round(s.hit_rate, 2),
                    "thompson_mean": round(s.alpha / (s.alpha + s.beta), 2),
                }
                for s in self._stats.values()
            ],
            key=lambda x: x["hit_rate"],
            reverse=True,
        )
