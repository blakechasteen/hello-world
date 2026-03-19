"""Moderator — watches individual sessions and intervenes when useful.

The moderator is the focus group facilitator in the room. It:
  - Notices when an agent is going in circles
  - Injects follow-up prompts when something interesting surfaces
  - Redirects agents that have drifted from their goals
  - Flags sessions that need human attention

It does NOT control the session loop directly. It communicates
via INJECT_PROMPT and REDIRECT_GOAL events that the session runner
picks up on the next cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.backend import LLMBackend, Message
from .base import Manager
from .events import Event, EventBus, EventKind


INTERVENTION_PROMPT = """\
You are moderating a focus group session. An agent with persona "{persona}" is testing software.

Session state:
- Current goal: {current_goal}
- Steps taken: {steps}
- Consecutive failures: {failures}
- Issues found so far: {issues}
- Frustration level: {frustration}

Recent actions (last {n_recent}):
{recent_actions}

Should you intervene? Consider:
1. Is the agent going in circles (repeating similar actions)?
2. Did it find something interesting that deserves deeper probing?
3. Is it drifting away from its goals?
4. Is it stuck and needs a nudge?

Respond with JSON:
{{
  "intervene": true/false,
  "type": "probe | redirect | nudge | none",
  "message": "the prompt to inject (if intervening)",
  "reasoning": "why"
}}
"""


@dataclass
class SessionTracker:
    """Tracks a single session's recent activity for the moderator."""

    persona: str
    session_id: str
    steps: int = 0
    consecutive_failures: int = 0
    issues_found: int = 0
    frustration: float = 0.0
    current_goal: str = ""
    recent_actions: list[str] = field(default_factory=list)
    last_intervention_step: int = -10  # prevent rapid-fire interventions


@dataclass
class ModeratorConfig:
    """Knobs for the moderator."""

    check_interval: int = 5  # evaluate every N steps per session
    min_steps_between_interventions: int = 4
    circle_detection_window: int = 6  # look back this many actions for loops
    max_actions_tracked: int = 15


class Moderator(Manager):
    """Watches sessions and injects prompts when useful."""

    def __init__(
        self,
        bus: EventBus,
        backend: LLMBackend,
        config: ModeratorConfig | None = None,
    ):
        super().__init__(bus, backend)
        self.config = config or ModeratorConfig()
        self._sessions: dict[str, SessionTracker] = {}

    @property
    def name(self) -> str:
        return "moderator"

    def attach(self) -> None:
        self.bus.on(EventKind.SESSION_START, self._on_start)
        self.bus.on(EventKind.SESSION_END, self._on_end)
        self.bus.on(EventKind.SESSION_STEP, self._on_step)
        self.bus.on(EventKind.ISSUE_FOUND, self._on_issue)
        self.bus.on(EventKind.ACTION_FAILED, self._on_failure)
        self.bus.on(EventKind.FRUSTRATION_HIGH, self._on_frustration)
        self._attached = True

    async def _on_start(self, event: Event) -> None:
        self._sessions[event.session_id] = SessionTracker(
            persona=event.persona,
            session_id=event.session_id,
        )

    async def _on_end(self, event: Event) -> None:
        self._sessions.pop(event.session_id, None)

    async def _on_step(self, event: Event) -> None:
        tracker = self._sessions.get(event.session_id)
        if not tracker:
            return

        tracker.steps += 1
        action = event.data.get("action", "")
        goal = event.data.get("current_goal", "")
        if action:
            tracker.recent_actions.append(action)
            tracker.recent_actions = tracker.recent_actions[-self.config.max_actions_tracked:]
        if goal:
            tracker.current_goal = goal

        # Reset consecutive failures on successful step.
        if not event.data.get("failed"):
            tracker.consecutive_failures = 0

        # Evaluate at interval.
        if (
            tracker.steps % self.config.check_interval == 0
            and tracker.steps - tracker.last_intervention_step >= self.config.min_steps_between_interventions
        ):
            await self._evaluate(tracker)

    async def _on_issue(self, event: Event) -> None:
        tracker = self._sessions.get(event.session_id)
        if tracker:
            tracker.issues_found += 1

    async def _on_failure(self, event: Event) -> None:
        tracker = self._sessions.get(event.session_id)
        if tracker:
            tracker.consecutive_failures += 1

    async def _on_frustration(self, event: Event) -> None:
        tracker = self._sessions.get(event.session_id)
        if tracker:
            tracker.frustration = event.data.get("level", 0)

    async def _evaluate(self, tracker: SessionTracker) -> None:
        """Ask the LLM whether to intervene in this session."""
        # Quick heuristic: detect obvious circles without LLM.
        if self._detect_circle(tracker):
            tracker.last_intervention_step = tracker.steps
            await self._emit(
                EventKind.INJECT_PROMPT,
                session_id=tracker.session_id,
                persona=tracker.persona,
                data={
                    "message": (
                        "[MODERATOR] You seem to be repeating similar actions. "
                        "Try a completely different approach, or move to your next goal."
                    ),
                    "type": "redirect",
                },
            )
            return

        # LLM evaluation for subtler patterns.
        if not self.backend:
            return

        recent = tracker.recent_actions[-self.config.circle_detection_window:]
        prompt = INTERVENTION_PROMPT.format(
            persona=tracker.persona,
            current_goal=tracker.current_goal or "unknown",
            steps=tracker.steps,
            failures=tracker.consecutive_failures,
            issues=tracker.issues_found,
            frustration=f"{tracker.frustration:.0f}",
            n_recent=len(recent),
            recent_actions="\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent)) or "(none)",
        )

        response = await self.backend.complete(
            [Message(role="user", content=prompt)],
            temperature=0.2,
            max_tokens=256,
        )

        import json, re
        text = response.content
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text.strip())

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return

        if data.get("intervene") and data.get("message"):
            tracker.last_intervention_step = tracker.steps
            intervention_type = data.get("type", "nudge")

            kind = EventKind.INJECT_PROMPT
            if intervention_type == "redirect":
                kind = EventKind.REDIRECT_GOAL

            await self._emit(
                kind,
                session_id=tracker.session_id,
                persona=tracker.persona,
                data={
                    "message": f"[MODERATOR] {data['message']}",
                    "type": intervention_type,
                    "reasoning": data.get("reasoning", ""),
                },
            )

    def _detect_circle(self, tracker: SessionTracker) -> bool:
        """Heuristic: are the last N actions mostly repeats?"""
        window = tracker.recent_actions[-self.config.circle_detection_window:]
        if len(window) < self.config.circle_detection_window:
            return False

        # Normalize and check uniqueness.
        normalized = [a.strip().lower() for a in window]
        unique = len(set(normalized))
        # If fewer than 40% unique in the window, it's a circle.
        return unique / len(normalized) < 0.4
