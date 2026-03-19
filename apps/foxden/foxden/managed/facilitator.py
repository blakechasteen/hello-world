"""Facilitator — the conductor of a managed focus group.

Responsibilities:
  - Decides when the group has enough data (saturation detection)
  - Manages token/time budgets
  - Coordinates startup and shutdown
  - Does NOT micromanage individual sessions (that's the Moderator)

The facilitator watches aggregate signals, not individual actions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..llm.backend import LLMBackend, Message
from .base import Manager, ManagerState
from .events import Event, EventBus, EventKind


SATURATION_PROMPT = """\
You are monitoring a focus group of {n_agents} AI agents testing software.
Here is the current state:

Issues found: {n_issues} (new in last batch: {new_issues})
Confusion points: {n_confusion}
Goals completed: {completed} / failed: {failed}
Sessions still active: {active}
Token budget used: {tokens_used} / {tokens_max}

Recent issue descriptions:
{recent_issues}

Has this focus group reached saturation — are we finding the same kinds of issues,
or is each new session still discovering novel problems?

Respond with JSON:
{{
  "saturated": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "why",
  "recommendation": "continue | scale_down | wrap_up"
}}
"""


@dataclass
class FacilitatorConfig:
    """Knobs for the facilitator."""

    token_budget: int = 0  # 0 = unlimited
    time_budget_seconds: float = 0  # 0 = unlimited
    min_sessions: int = 3  # don't check saturation until this many finish
    saturation_check_interval: int = 3  # check every N completed sessions
    saturation_threshold: float = 0.7  # confidence above this = saturated
    auto_scale: bool = True  # let the facilitator spawn/kill agents


class Facilitator(Manager):
    """Top-level orchestrator for a managed focus group."""

    def __init__(
        self,
        bus: EventBus,
        backend: LLMBackend,
        config: FacilitatorConfig | None = None,
    ):
        super().__init__(bus, backend)
        self.config = config or FacilitatorConfig()
        self.state = ManagerState(token_budget_max=self.config.token_budget)
        self._sessions_completed = 0
        self._recent_issues: list[str] = []
        self._issue_window: list[str] = []  # issues since last saturation check
        self._saturated = False
        self._start_time: float | None = None

    @property
    def name(self) -> str:
        return "facilitator"

    def attach(self) -> None:
        self.bus.on(EventKind.SESSION_START, self._on_session_start)
        self.bus.on(EventKind.SESSION_END, self._on_session_end)
        self.bus.on(EventKind.ISSUE_FOUND, self._on_issue)
        self.bus.on(EventKind.CONFUSION_FOUND, self._on_confusion)
        self.bus.on(EventKind.GOAL_COMPLETED, self._on_goal_completed)
        self.bus.on(EventKind.GOAL_FAILED, self._on_goal_failed)
        self.bus.on(EventKind.SESSION_STEP, self._on_step)
        self._attached = True

    async def _on_session_start(self, event: Event) -> None:
        self.state.active_sessions[event.session_id] = {
            "persona": event.persona,
            "step": 0,
            "issues": 0,
        }
        if self._start_time is None:
            import time
            self._start_time = time.monotonic()

    async def _on_session_end(self, event: Event) -> None:
        self.state.active_sessions.pop(event.session_id, None)
        self._sessions_completed += 1

        # Saturation check.
        if (
            self._sessions_completed >= self.config.min_sessions
            and self._sessions_completed % self.config.saturation_check_interval == 0
            and not self._saturated
            and self.backend
        ):
            await self._check_saturation()

    async def _on_issue(self, event: Event) -> None:
        self.state.total_issues += 1
        desc = event.data.get("description", "")
        if desc:
            self._recent_issues.append(desc)
            self._issue_window.append(desc)
            self._recent_issues = self._recent_issues[-20:]

        if event.session_id in self.state.active_sessions:
            self.state.active_sessions[event.session_id]["issues"] += 1

    async def _on_confusion(self, event: Event) -> None:
        self.state.total_confusion += 1

    async def _on_goal_completed(self, event: Event) -> None:
        self.state.total_goals_completed += 1

    async def _on_goal_failed(self, event: Event) -> None:
        self.state.total_goals_failed += 1

    async def _on_step(self, event: Event) -> None:
        tokens = event.data.get("tokens", 0)
        self.state.token_budget_used += tokens

        if event.session_id in self.state.active_sessions:
            self.state.active_sessions[event.session_id]["step"] += 1

        # Budget check.
        if (
            self.config.token_budget > 0
            and self.state.token_budget_used > self.config.token_budget * 0.9
        ):
            await self._emit(
                EventKind.BUDGET_WARNING,
                data={
                    "used": self.state.token_budget_used,
                    "max": self.config.token_budget,
                    "percent": self.state.token_budget_used / self.config.token_budget,
                },
            )

        # Time budget check.
        if self.config.time_budget_seconds > 0 and self._start_time:
            import time
            elapsed = time.monotonic() - self._start_time
            if elapsed > self.config.time_budget_seconds:
                await self._emit(
                    EventKind.GROUP_COMPLETE,
                    data={"reason": "time_budget_exceeded", "elapsed": elapsed},
                )

    async def _check_saturation(self) -> None:
        """Ask the LLM if we're still finding novel issues."""
        new_in_window = len(self._issue_window)
        self._issue_window = []  # reset window

        prompt = SATURATION_PROMPT.format(
            n_agents=self._sessions_completed + len(self.state.active_sessions),
            n_issues=self.state.total_issues,
            new_issues=new_in_window,
            n_confusion=self.state.total_confusion,
            completed=self.state.total_goals_completed,
            failed=self.state.total_goals_failed,
            active=len(self.state.active_sessions),
            tokens_used=self.state.token_budget_used,
            tokens_max=self.config.token_budget or "unlimited",
            recent_issues="\n".join(f"  - {i}" for i in self._recent_issues[-10:]) or "(none)",
        )

        response = await self.backend.complete(
            [Message(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=512,
        )

        import json, re
        text = response.content
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text.strip())

        try:
            # Find JSON in response.
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return  # can't parse, keep going

        confidence = data.get("confidence", 0)
        saturated = data.get("saturated", False)
        recommendation = data.get("recommendation", "continue")

        if saturated and confidence >= self.config.saturation_threshold:
            self._saturated = True
            await self._emit(
                EventKind.GROUP_COMPLETE,
                data={
                    "reason": "saturation",
                    "confidence": confidence,
                    "recommendation": recommendation,
                    "reasoning": data.get("reasoning", ""),
                },
            )
