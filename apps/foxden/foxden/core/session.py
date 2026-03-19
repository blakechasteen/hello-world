"""Session — one agent's full interaction with software under test."""

from __future__ import annotations

import json
import re

from ..core.persona import Persona
from ..interfaces.base import Action, Observation, TestInterface
from ..llm.backend import LLMBackend, Message
from ..recording.transcript import Transcript

# Max steps before forcing session end.
MAX_STEPS = 50


def _build_system_prompt(persona: Persona, interface_hint: str) -> str:
    goals_text = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(persona.goals))

    return f"""{persona.system_prompt}

You are testing software via a {interface_hint} interface.

Your goals (complete them in order if possible):
{goals_text}

Behavioral traits:
- Patience: {persona.behavior.patience}
- Reads documentation: {persona.behavior.reads_docs}
- Exploration style: {persona.behavior.exploration}
- Speed: {persona.behavior.speed}
- Gives up after {persona.behavior.error_tolerance} consecutive failures

After each observation, respond with a JSON object containing:
{{
  "thinking": "your reasoning about the current state",
  "action": {{"kind": "<action_type>", "value": "<target>"}},
  "status": {{
    "current_goal": "which goal you're working on",
    "goal_completed": false,
    "frustration": 0-10,
    "issues": ["any bugs or problems found"],
    "confusion": ["anything confusing or unclear"]
  }}
}}

Set "goal_completed": true when you've finished your current goal, then move to the next one.

{_action_help(interface_hint)}

When you've completed all goals or given up, set action to:
{{"kind": "done", "value": "summary of what happened"}}
"""


def _action_help(interface_hint: str) -> str:
    if interface_hint == "web":
        return """Available actions for the web interface:
- {{"kind": "click", "value": "CSS selector or text"}} — click an element
- {{"kind": "type", "value": "text", "metadata": {{"selector": "CSS selector", "text": "what to type"}}}} — type into a field
- {{"kind": "navigate", "value": "URL"}} — go to a URL
- {{"kind": "press", "value": "Enter"}} — press a key (Tab, Enter, Escape, ArrowDown, etc.)
- {{"kind": "scroll", "value": "down"}} — scroll up or down
- {{"kind": "hover", "value": "CSS selector"}} — hover over an element
- {{"kind": "select", "value": "option value", "metadata": {{"selector": "CSS selector"}}}} — select dropdown option
- {{"kind": "back"}} — go back in browser history
- {{"kind": "wait", "value": "1000"}} — wait milliseconds

Use the ACCESSIBILITY TREE and VISIBLE ELEMENTS in observations to find selectors.
Prefer text selectors like 'text=Sign Up' or 'button:has-text("Submit")' over fragile CSS paths."""
    elif interface_hint == "cli":
        return """Available actions for the CLI interface:
- {{"kind": "command", "value": "the shell command to run"}}"""
    elif interface_hint == "api":
        return """Available actions for the API interface:
The action "kind" is the HTTP method and "value" is the path. Use "metadata" for body, params, and headers.

- {{"kind": "GET", "value": "/endpoint"}} — simple GET
- {{"kind": "GET", "value": "/endpoint", "metadata": {{"params": {{"page": 1}}}}}} — GET with query params
- {{"kind": "POST", "value": "/endpoint", "metadata": {{"body": {{"key": "value"}}}}}} — POST with JSON body
- {{"kind": "PUT", "value": "/endpoint", "metadata": {{"body": {{"key": "value"}}}}}} — PUT
- {{"kind": "PATCH", "value": "/endpoint", "metadata": {{"body": {{"key": "value"}}}}}} — PATCH
- {{"kind": "DELETE", "value": "/endpoint"}} — DELETE
- {{"kind": "POST", "value": "/endpoint", "metadata": {{"headers": {{"Authorization": "Bearer token"}}}}}} — with custom headers

Use the OPENAPI SPEC SUMMARY (if available) to discover endpoints and their expected inputs."""
    return ""


def _parse_agent_response(text: str) -> dict:
    """Extract JSON from the agent's response, tolerating markdown fences."""
    # Strip markdown code fences.
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())

    # Find the outermost JSON object.
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue

    return {"thinking": text, "action": {"kind": "done", "value": "parse failure"}, "status": {}}


async def run_session(
    persona: Persona,
    interface: TestInterface,
    backend: LLMBackend,
    *,
    max_steps: int = MAX_STEPS,
) -> Transcript:
    """Run a single agent session: observe → think → act → record → evaluate."""
    transcript = Transcript(persona_name=persona.name)
    interface_hint = type(interface).__name__.replace("Interface", "").lower()
    system_prompt = _build_system_prompt(persona, interface_hint)

    messages: list[Message] = [Message(role="system", content=system_prompt)]

    # Start the interface.
    obs = await interface.start()
    transcript.record("observe", obs.content)
    messages.append(Message(role="user", content=f"[OBSERVATION]\n{obs.content}"))

    # Frustration model.
    frustration = 0.0
    consecutive_failures = 0
    behavior = persona.behavior

    for step in range(max_steps):
        # Inject frustration context if it's building up.
        frustration_hint = ""
        if frustration > behavior.patience_budget * 0.5:
            frustration_hint = (
                f"\n[SYSTEM] Your frustration level is {frustration:.0f}/{behavior.patience_budget}. "
                f"You're getting {'very ' if frustration > behavior.patience_budget * 0.75 else ''}impatient."
            )

        # Think — ask the LLM.
        response = await backend.complete(messages, temperature=0.7)
        transcript.record("think", response.content, tokens=response.output_tokens)

        parsed = _parse_agent_response(response.content)
        messages.append(Message(role="assistant", content=response.content))

        # Extract status.
        status = parsed.get("status", {})
        for issue in status.get("issues", []):
            if issue and issue not in transcript.issues:
                transcript.issues.append(issue)
        for confusion in status.get("confusion", []):
            if confusion and confusion not in transcript.confusion_points:
                transcript.confusion_points.append(confusion)

        # Goal completion tracking.
        if status.get("goal_completed"):
            current_goal = status.get("current_goal", "")
            if current_goal and current_goal not in transcript.goals_completed:
                transcript.goals_completed.append(current_goal)
                transcript.record("evaluate", f"Goal completed: {current_goal}")
                # Success reduces frustration.
                frustration = max(0, frustration - behavior.recovery_rate * 3)

        # Agent-reported frustration feeds into our model.
        agent_frustration = status.get("frustration", 0)
        if isinstance(agent_frustration, (int, float)):
            # Blend agent's self-reported frustration with our model.
            frustration = max(frustration, agent_frustration * behavior.frustration_rate)

        # Extract action.
        action_data = parsed.get("action", {})
        action_kind = action_data.get("kind", "done")
        action_value = action_data.get("value", "")
        action_metadata = action_data.get("metadata")

        if action_kind == "done":
            # Mark any remaining goals as failed.
            all_goals = set(persona.goals)
            completed = set(transcript.goals_completed)
            for g in all_goals - completed:
                if g not in transcript.goals_failed:
                    transcript.goals_failed.append(g)
            transcript.record("evaluate", f"Agent finished: {action_value}")
            break

        action = Action(kind=action_kind, value=action_value, metadata=action_metadata)
        transcript.record("act", f"{action.kind}: {action.value}")

        # Execute the action.
        obs = await interface.act(action)
        transcript.record("observe", obs.content)

        # Check for failure patterns.
        raw = obs.raw or {}
        is_failure = (
            raw.get("exit_code", 0) != 0
            or raw.get("timeout")
            or raw.get("error")
        )
        if is_failure:
            consecutive_failures += 1
            frustration += behavior.frustration_rate

            if consecutive_failures >= behavior.error_tolerance:
                current_goal = status.get("current_goal", "unknown")
                transcript.goals_failed.append(current_goal)
                transcript.record(
                    "evaluate",
                    f"Gave up on '{current_goal}' after {consecutive_failures} failures "
                    f"(frustration: {frustration:.0f}/{behavior.patience_budget})",
                )
                messages.append(
                    Message(
                        role="user",
                        content=f"[OBSERVATION]\n{obs.content}\n\n"
                        f"[SYSTEM] You have failed {consecutive_failures} times in a row. "
                        f"Consider moving on or giving up.",
                    )
                )
                consecutive_failures = 0
                continue
        else:
            if consecutive_failures > 0:
                # Partial recovery on success after failures.
                frustration = max(0, frustration - behavior.recovery_rate)
            consecutive_failures = 0

        # Global patience budget check.
        if frustration >= behavior.patience_budget:
            transcript.record(
                "evaluate",
                f"Agent exhausted patience (frustration {frustration:.0f}/{behavior.patience_budget}). Quitting.",
            )
            # Mark remaining goals as failed.
            all_goals = set(persona.goals)
            completed = set(transcript.goals_completed)
            for g in all_goals - completed:
                if g not in transcript.goals_failed:
                    transcript.goals_failed.append(g)
            break

        obs_text = f"[OBSERVATION]\n{obs.content}"
        if frustration_hint:
            obs_text += frustration_hint
        messages.append(Message(role="user", content=obs_text))
    else:
        transcript.record("evaluate", "Session ended: max steps reached")

    await interface.close()
    transcript.finish()
    return transcript
