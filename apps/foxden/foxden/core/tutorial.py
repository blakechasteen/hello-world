"""Tutorial mode — agents follow written instructions and report divergences."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..interfaces.base import Action, Observation, TestInterface
from ..llm.backend import LLMBackend, Message
from ..recording.transcript import Transcript


@dataclass
class TutorialStep:
    """A single step from a tutorial or doc."""

    number: int
    instruction: str
    expected_outcome: str = ""


@dataclass
class TutorialDivergence:
    """Where reality didn't match the tutorial."""

    step: int
    instruction: str
    expected: str
    actual: str
    severity: str = "medium"  # low | medium | high


@dataclass
class TutorialResult:
    """Result of a tutorial-following session."""

    transcript: Transcript
    steps_completed: int = 0
    steps_total: int = 0
    divergences: list[TutorialDivergence] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.steps_completed / self.steps_total

    @property
    def summary(self) -> str:
        lines = [
            f"=== Tutorial Follow Result ===",
            f"Steps: {self.steps_completed}/{self.steps_total} completed ({self.success_rate:.0%})",
        ]
        if self.divergences:
            lines.append(f"\nDivergences ({len(self.divergences)}):")
            for d in self.divergences:
                lines.append(f"  Step {d.step} [{d.severity.upper()}]:")
                lines.append(f"    Instruction: {d.instruction}")
                if d.expected:
                    lines.append(f"    Expected:    {d.expected}")
                lines.append(f"    Actual:      {d.actual}")
        else:
            lines.append("\nNo divergences found — docs match reality.")
        return "\n".join(lines)


def parse_tutorial(text: str) -> list[TutorialStep]:
    """Parse a tutorial document into steps.

    Supports formats:
    - Numbered steps: "1. Do this thing"
    - Markdown headers as sections with paragraphs as steps
    - "Expected:" or "Output:" lines after a step
    """
    steps = []
    lines = text.strip().split("\n")
    i = 0
    step_num = 0

    while i < len(lines):
        line = lines[i].strip()

        # Numbered step: "1. Do something" or "Step 1: Do something"
        m = re.match(r"(?:step\s+)?(\d+)[.):]\s*(.+)", line, re.IGNORECASE)
        if m:
            step_num += 1
            instruction = m.group(2).strip()
            expected = ""

            # Look ahead for expected outcome.
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                em = re.match(r"(?:expected|output|result|you should see)[:\s]*(.+)", next_line, re.IGNORECASE)
                if em:
                    expected = em.group(1).strip()
                    j += 1
                    # Gather continuation lines.
                    while j < len(lines) and lines[j].strip() and not re.match(r"(?:step\s+)?\d+[.):]\s", lines[j].strip(), re.IGNORECASE):
                        expected += "\n" + lines[j].strip()
                        j += 1
                    break
                # Check if next line is another step.
                if re.match(r"(?:step\s+)?\d+[.):]\s", next_line, re.IGNORECASE):
                    break
                j += 1

            steps.append(TutorialStep(number=step_num, instruction=instruction, expected_outcome=expected))
            i = j if j > i + 1 else i + 1
            continue

        # Code block as a step (```bash or ``` with a command).
        if line.startswith("```"):
            lang = line[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            if code_lines:
                step_num += 1
                code = "\n".join(code_lines).strip()
                steps.append(TutorialStep(
                    number=step_num,
                    instruction=f"Run: {code}" if lang in ("bash", "sh", "shell", "") else f"Execute code:\n{code}",
                ))
            continue

        i += 1

    return steps


def load_tutorial(path_or_text: str) -> list[TutorialStep]:
    """Load tutorial from a file path or raw text."""
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        text = p.read_text()
    else:
        text = path_or_text
    return parse_tutorial(text)


TUTORIAL_SYSTEM_PROMPT = """\
You are following a tutorial step by step. Your job is to execute each step exactly as written
and verify that the result matches what the tutorial promises.

For each step, respond with a JSON object:
{{
  "thinking": "your reasoning",
  "action": {{"kind": "<action_type>", "value": "<target>"}},
  "step_result": {{
    "matches_expected": true or false,
    "actual_outcome": "what actually happened",
    "divergence": "description of mismatch (if any)",
    "severity": "low | medium | high"
  }}
}}

{action_help}

When you've completed all steps or can't continue, set action to:
{{"kind": "done", "value": "summary"}}
"""


async def follow_tutorial(
    steps: list[TutorialStep],
    interface: TestInterface,
    backend: LLMBackend,
    *,
    interface_hint: str = "cli",
) -> TutorialResult:
    """Have an agent follow a tutorial step by step, reporting divergences."""
    from .session import _action_help, _parse_agent_response

    transcript = Transcript(persona_name="tutorial-follower")
    action_help = _action_help(interface_hint)
    system_prompt = TUTORIAL_SYSTEM_PROMPT.format(action_help=action_help)

    messages: list[Message] = [Message(role="system", content=system_prompt)]

    # Start interface.
    obs = await interface.start()
    transcript.record("observe", obs.content)
    messages.append(Message(role="user", content=f"[OBSERVATION]\n{obs.content}"))

    divergences: list[TutorialDivergence] = []
    steps_completed = 0

    for step in steps:
        # Present the step to the agent.
        step_prompt = f"[TUTORIAL STEP {step.number}]\nInstruction: {step.instruction}"
        if step.expected_outcome:
            step_prompt += f"\nExpected outcome: {step.expected_outcome}"
        step_prompt += "\n\nExecute this step and report the result."

        messages.append(Message(role="user", content=step_prompt))
        transcript.record("evaluate", f"Starting step {step.number}: {step.instruction}")

        # Let the agent work through this step (up to 10 actions per step).
        step_done = False
        for _ in range(10):
            response = await backend.complete(messages, temperature=0.3)
            transcript.record("think", response.content, tokens=response.output_tokens)
            parsed = _parse_agent_response(response.content)
            messages.append(Message(role="assistant", content=response.content))

            # Check step result.
            step_result = parsed.get("step_result", {})
            if step_result:
                if step_result.get("matches_expected") is True:
                    steps_completed += 1
                    transcript.goals_completed.append(f"Step {step.number}")
                    transcript.record("evaluate", f"Step {step.number}: PASS")
                elif step_result.get("matches_expected") is False:
                    div = TutorialDivergence(
                        step=step.number,
                        instruction=step.instruction,
                        expected=step.expected_outcome,
                        actual=step_result.get("actual_outcome", ""),
                        severity=step_result.get("severity", "medium"),
                    )
                    divergences.append(div)
                    transcript.goals_failed.append(f"Step {step.number}")
                    transcript.issues.append(
                        f"Step {step.number} divergence: {step_result.get('divergence', '')}"
                    )
                    transcript.record("evaluate", f"Step {step.number}: DIVERGENCE — {step_result.get('divergence', '')}")

            # Execute action if not done.
            action_data = parsed.get("action", {})
            action_kind = action_data.get("kind", "done")
            action_value = action_data.get("value", "")

            if action_kind == "done" or step_result:
                step_done = True
                break

            action = Action(
                kind=action_kind,
                value=action_value,
                metadata=action_data.get("metadata"),
            )
            transcript.record("act", f"{action.kind}: {action.value}")

            obs = await interface.act(action)
            transcript.record("observe", obs.content)
            messages.append(Message(role="user", content=f"[OBSERVATION]\n{obs.content}"))

        if not step_done:
            transcript.record("evaluate", f"Step {step.number}: TIMEOUT (too many actions)")
            transcript.goals_failed.append(f"Step {step.number}")

    await interface.close()
    transcript.finish()

    return TutorialResult(
        transcript=transcript,
        steps_completed=steps_completed,
        steps_total=len(steps),
        divergences=divergences,
    )
