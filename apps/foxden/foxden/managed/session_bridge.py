"""Session bridge — connects the session runner to the event bus.

This is the adapter between the existing session loop and the managed
event system. It wraps run_session() to:
  1. Emit events as the session progresses
  2. Accept directive events (INJECT_PROMPT, REDIRECT_GOAL, TERMINATE)
  3. Feed directives into the session's message stream

The session runner itself stays clean — the bridge handles the wiring.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.persona import Persona
from ..interfaces.base import Action, Observation, TestInterface
from ..llm.backend import LLMBackend, Message
from ..recording.transcript import Transcript
from ..core.session import _build_system_prompt, _parse_agent_response, _action_help
from .events import Event, EventBus, EventKind


@dataclass
class ManagedSession:
    """A session that's wired into the event bus."""

    session_id: str
    persona: Persona
    interface: TestInterface
    backend: LLMBackend
    bus: EventBus
    max_steps: int = 50

    # Directive inbox — filled by event handlers, drained by the session loop.
    _inbox: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    _terminated: bool = field(default=False, init=False)

    def __post_init__(self):
        # Subscribe to directives for this session.
        self.bus.on(EventKind.INJECT_PROMPT, self._on_directive)
        self.bus.on(EventKind.REDIRECT_GOAL, self._on_directive)
        self.bus.on(EventKind.TERMINATE_SESSION, self._on_terminate)

    async def _on_directive(self, event: Event) -> None:
        if event.session_id == self.session_id:
            await self._inbox.put(event)

    async def _on_terminate(self, event: Event) -> None:
        if event.session_id == self.session_id or event.session_id == "*":
            self._terminated = True

    async def run(self) -> Transcript:
        """Run the session with event bus integration."""
        transcript = Transcript(persona_name=self.persona.name)
        interface_hint = type(self.interface).__name__.replace("Interface", "").lower()
        system_prompt = _build_system_prompt(self.persona, interface_hint)

        messages: list[Message] = [Message(role="system", content=system_prompt)]

        # Emit start.
        await self.bus.emit(Event(
            kind=EventKind.SESSION_START,
            session_id=self.session_id,
            persona=self.persona.name,
        ))

        # Start interface.
        obs = await self.interface.start()
        transcript.record("observe", obs.content)
        messages.append(Message(role="user", content=f"[OBSERVATION]\n{obs.content}"))

        frustration = 0.0
        consecutive_failures = 0
        behavior = self.persona.behavior

        for step in range(self.max_steps):
            if self._terminated:
                transcript.record("evaluate", "Session terminated by manager")
                break

            # Drain directive inbox.
            while not self._inbox.empty():
                try:
                    directive = self._inbox.get_nowait()
                    msg = directive.data.get("message", "")
                    if msg:
                        messages.append(Message(role="user", content=msg))
                        transcript.record("evaluate", f"Directive: {msg[:100]}")
                except asyncio.QueueEmpty:
                    break

            # Frustration hint.
            frustration_hint = ""
            if frustration > behavior.patience_budget * 0.5:
                frustration_hint = (
                    f"\n[SYSTEM] Your frustration level is {frustration:.0f}/{behavior.patience_budget}. "
                    f"You're getting {'very ' if frustration > behavior.patience_budget * 0.75 else ''}impatient."
                )
                if frustration > behavior.patience_budget * 0.75:
                    await self.bus.emit(Event(
                        kind=EventKind.FRUSTRATION_HIGH,
                        session_id=self.session_id,
                        persona=self.persona.name,
                        data={"level": frustration, "budget": behavior.patience_budget},
                    ))

            # Think.
            response = await self.backend.complete(messages, temperature=0.7)
            transcript.record("think", response.content, tokens=response.output_tokens)
            parsed = _parse_agent_response(response.content)
            messages.append(Message(role="assistant", content=response.content))

            # Emit step event.
            status = parsed.get("status", {})
            await self.bus.emit(Event(
                kind=EventKind.SESSION_STEP,
                session_id=self.session_id,
                persona=self.persona.name,
                data={
                    "step": step,
                    "tokens": response.output_tokens,
                    "current_goal": status.get("current_goal", ""),
                    "action": parsed.get("action", {}).get("value", ""),
                    "failed": False,  # updated below if it fails
                },
            ))

            # Extract findings.
            for issue in status.get("issues", []):
                if issue and issue not in transcript.issues:
                    transcript.issues.append(issue)
                    await self.bus.emit(Event(
                        kind=EventKind.ISSUE_FOUND,
                        session_id=self.session_id,
                        persona=self.persona.name,
                        data={"description": issue},
                    ))

            for confusion in status.get("confusion", []):
                if confusion and confusion not in transcript.confusion_points:
                    transcript.confusion_points.append(confusion)
                    await self.bus.emit(Event(
                        kind=EventKind.CONFUSION_FOUND,
                        session_id=self.session_id,
                        persona=self.persona.name,
                        data={"description": confusion},
                    ))

            # Goal tracking.
            if status.get("goal_completed"):
                current_goal = status.get("current_goal", "")
                if current_goal and current_goal not in transcript.goals_completed:
                    transcript.goals_completed.append(current_goal)
                    transcript.record("evaluate", f"Goal completed: {current_goal}")
                    frustration = max(0, frustration - behavior.recovery_rate * 3)
                    await self.bus.emit(Event(
                        kind=EventKind.GOAL_COMPLETED,
                        session_id=self.session_id,
                        persona=self.persona.name,
                        data={"goal": current_goal},
                    ))

            # Agent frustration blend.
            agent_frustration = status.get("frustration", 0)
            if isinstance(agent_frustration, (int, float)):
                frustration = max(frustration, agent_frustration * behavior.frustration_rate)

            # Action.
            action_data = parsed.get("action", {})
            action_kind = action_data.get("kind", "done")
            action_value = action_data.get("value", "")
            action_metadata = action_data.get("metadata")

            if action_kind == "done":
                all_goals = set(self.persona.goals)
                completed = set(transcript.goals_completed)
                for g in all_goals - completed:
                    if g not in transcript.goals_failed:
                        transcript.goals_failed.append(g)
                transcript.record("evaluate", f"Agent finished: {action_value}")
                break

            action = Action(kind=action_kind, value=action_value, metadata=action_metadata)
            transcript.record("act", f"{action.kind}: {action.value}")

            obs = await self.interface.act(action)
            transcript.record("observe", obs.content)

            # Failure detection.
            raw = obs.raw or {}
            is_failure = raw.get("exit_code", 0) != 0 or raw.get("timeout") or raw.get("error")

            if is_failure:
                consecutive_failures += 1
                frustration += behavior.frustration_rate

                await self.bus.emit(Event(
                    kind=EventKind.ACTION_FAILED,
                    session_id=self.session_id,
                    persona=self.persona.name,
                    data={"action": action_value, "consecutive": consecutive_failures},
                ))

                if consecutive_failures >= behavior.error_tolerance:
                    current_goal = status.get("current_goal", "unknown")
                    transcript.goals_failed.append(current_goal)
                    transcript.record("evaluate",
                        f"Gave up on '{current_goal}' after {consecutive_failures} failures")

                    await self.bus.emit(Event(
                        kind=EventKind.GOAL_FAILED,
                        session_id=self.session_id,
                        persona=self.persona.name,
                        data={"goal": current_goal, "failures": consecutive_failures},
                    ))

                    messages.append(Message(
                        role="user",
                        content=f"[OBSERVATION]\n{obs.content}\n\n"
                        f"[SYSTEM] You have failed {consecutive_failures} times. "
                        f"Consider moving on or giving up.",
                    ))
                    consecutive_failures = 0
                    continue
            else:
                if consecutive_failures > 0:
                    frustration = max(0, frustration - behavior.recovery_rate)
                consecutive_failures = 0

            # Patience budget.
            if frustration >= behavior.patience_budget:
                transcript.record("evaluate",
                    f"Patience exhausted ({frustration:.0f}/{behavior.patience_budget})")
                all_goals = set(self.persona.goals)
                completed = set(transcript.goals_completed)
                for g in all_goals - completed:
                    if g not in transcript.goals_failed:
                        transcript.goals_failed.append(g)

                await self.bus.emit(Event(
                    kind=EventKind.PATIENCE_EXHAUSTED,
                    session_id=self.session_id,
                    persona=self.persona.name,
                    data={"frustration": frustration, "budget": behavior.patience_budget},
                ))
                break

            obs_text = f"[OBSERVATION]\n{obs.content}"
            if frustration_hint:
                obs_text += frustration_hint
            messages.append(Message(role="user", content=obs_text))
        else:
            transcript.record("evaluate", "Session ended: max steps reached")

        await self.interface.close()
        transcript.finish()

        # Emit end.
        await self.bus.emit(Event(
            kind=EventKind.SESSION_END,
            session_id=self.session_id,
            persona=self.persona.name,
            data={
                "issues": len(transcript.issues),
                "confusion": len(transcript.confusion_points),
                "goals_completed": len(transcript.goals_completed),
                "goals_failed": len(transcript.goals_failed),
            },
        ))

        return transcript
