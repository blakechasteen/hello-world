"""Recruiter — selects and generates personas for a target.

Given a target (URL, CLI tool, API), the recruiter:
  1. Analyzes what kind of software it is
  2. Selects the best-fit personas from the library
  3. Generates new custom personas when the library doesn't cover a need
  4. Learns over time which personas find real bugs for which software types
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.persona import Behavior, Persona, load_personas
from ..llm.backend import LLMBackend, Message
from .base import Manager
from .events import Event, EventBus, EventKind


ANALYSIS_PROMPT = """\
I need to assemble a focus group of AI agents to test this software:

Target: {target}
Interface: {interface}
Additional context: {context}

Available personas in the library:
{persona_list}

Your job:
1. Analyze what kind of software this is
2. Select the most relevant personas from the library (by name)
3. Suggest 0-3 NEW custom personas that would test aspects the library doesn't cover
4. For each persona, recommend how many copies to run (1-3)

Respond with JSON:
{{
  "software_type": "what kind of software this is",
  "selected": [
    {{"name": "existing-persona-name", "copies": 1, "reason": "why"}}
  ],
  "generated": [
    {{
      "name": "new-persona-name",
      "system_prompt": "the full system prompt",
      "goals": ["goal 1", "goal 2"],
      "behavior": {{
        "patience": "low|medium|high",
        "reads_docs": true/false,
        "exploration": "random|guided|methodical",
        "error_tolerance": 3
      }},
      "copies": 1,
      "reason": "why this persona is needed"
    }}
  ]
}}
"""


@dataclass
class RecruitmentPlan:
    """The recruiter's output: who to test with."""

    software_type: str
    personas: list[Persona]
    copies: dict[str, int]  # persona_name → copy count
    reasoning: dict[str, str]  # persona_name → why selected


class Recruiter(Manager):
    """Selects and generates personas for a target."""

    def __init__(
        self,
        bus: EventBus,
        backend: LLMBackend,
        persona_dir: str | Path | None = None,
    ):
        super().__init__(bus, backend)
        self.persona_dir = Path(persona_dir) if persona_dir else Path(__file__).parent.parent.parent / "personas"
        self._library = load_personas(self.persona_dir)

    @property
    def name(self) -> str:
        return "recruiter"

    def attach(self) -> None:
        # Recruiter doesn't subscribe to events — it's called directly.
        self._attached = True

    async def recruit(
        self,
        target: str,
        interface: str,
        context: str = "",
    ) -> RecruitmentPlan:
        """Analyze target and build a recruitment plan."""
        persona_list = []
        for name, p in self._library.items():
            tags = ", ".join(p.tags) if p.tags else ""
            goals = "; ".join(p.goals[:3])
            persona_list.append(f"  {name} [{tags}]: {goals}")

        prompt = ANALYSIS_PROMPT.format(
            target=target,
            interface=interface,
            context=context or "(none)",
            persona_list="\n".join(persona_list),
        )

        response = await self.backend.complete(
            [Message(role="user", content=prompt)],
            temperature=0.4,
            max_tokens=2048,
        )

        return self._parse_plan(response.content)

    def _parse_plan(self, text: str) -> RecruitmentPlan:
        """Parse the LLM's recruitment plan."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text.strip())

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Fallback: use all personas, 1 copy each.
            return RecruitmentPlan(
                software_type="unknown",
                personas=list(self._library.values()),
                copies={n: 1 for n in self._library},
                reasoning={n: "fallback — couldn't parse recruitment plan" for n in self._library},
            )

        personas = []
        copies = {}
        reasoning = {}

        # Selected from library.
        for sel in data.get("selected", []):
            name = sel.get("name", "")
            if name in self._library:
                personas.append(self._library[name])
                copies[name] = sel.get("copies", 1)
                reasoning[name] = sel.get("reason", "")

        # Generated custom personas.
        for gen in data.get("generated", []):
            name = gen.get("name", "")
            if not name:
                continue
            behavior_data = gen.get("behavior", {})
            persona = Persona(
                name=name,
                system_prompt=gen.get("system_prompt", "You are a software tester."),
                goals=gen.get("goals", []),
                behavior=Behavior.from_dict(behavior_data),
                tags=["generated"],
            )
            personas.append(persona)
            copies[name] = gen.get("copies", 1)
            reasoning[name] = gen.get("reason", "generated for this target")

        return RecruitmentPlan(
            software_type=data.get("software_type", "unknown"),
            personas=personas,
            copies=copies,
            reasoning=reasoning,
        )
