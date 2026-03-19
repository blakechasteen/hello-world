"""Synthesis agent — LLM-powered analysis across all session transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..llm.backend import LLMBackend, Message
from ..recording.transcript import Transcript


SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior QA analyst reviewing results from a simulated focus group.
Multiple AI agents — each with a distinct persona and goals — tested the same software.
Your job is to synthesize their findings into a clear, actionable report.

Analyze the transcripts and produce a JSON report with this structure:
{
  "executive_summary": "2-3 sentence overview of the software's state",
  "severity_rating": "critical | high | medium | low",
  "issues": [
    {
      "title": "Short issue title",
      "severity": "critical | high | medium | low",
      "description": "What's wrong",
      "evidence": ["Which personas hit this, what they saw"],
      "recommendation": "How to fix it"
    }
  ],
  "usability_findings": [
    {
      "area": "Which part of the software",
      "observation": "What was confusing or difficult",
      "affected_personas": ["which types of users"],
      "recommendation": "How to improve"
    }
  ],
  "strengths": ["Things that worked well"],
  "coverage_gaps": ["Areas that weren't tested or need deeper testing"],
  "doc_divergences": [
    {
      "documented": "What the docs say",
      "actual": "What actually happened",
      "persona": "Who found it"
    }
  ]
}
"""


@dataclass
class SynthesizedReport:
    """Rich report produced by the synthesis agent."""

    executive_summary: str = ""
    severity_rating: str = "low"
    issues: list[dict] = field(default_factory=list)
    usability_findings: list[dict] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    doc_divergences: list[dict] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> SynthesizedReport:
        return cls(
            executive_summary=d.get("executive_summary", ""),
            severity_rating=d.get("severity_rating", "low"),
            issues=d.get("issues", []),
            usability_findings=d.get("usability_findings", []),
            strengths=d.get("strengths", []),
            coverage_gaps=d.get("coverage_gaps", []),
            doc_divergences=d.get("doc_divergences", []),
            raw_json=d,
        )

    @property
    def summary(self) -> str:
        lines = [
            f"=== Synthesized Report (severity: {self.severity_rating}) ===",
            "",
            self.executive_summary,
            "",
        ]

        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            for iss in self.issues:
                sev = iss.get("severity", "?")
                title = iss.get("title", "untitled")
                lines.append(f"  [{sev.upper()}] {title}")
                if iss.get("description"):
                    lines.append(f"    {iss['description']}")
                if iss.get("recommendation"):
                    lines.append(f"    Fix: {iss['recommendation']}")
            lines.append("")

        if self.usability_findings:
            lines.append(f"Usability ({len(self.usability_findings)}):")
            for uf in self.usability_findings:
                area = uf.get("area", "?")
                obs = uf.get("observation", "")
                lines.append(f"  {area}: {obs}")
                if uf.get("recommendation"):
                    lines.append(f"    Fix: {uf['recommendation']}")
            lines.append("")

        if self.strengths:
            lines.append("Strengths:")
            for s in self.strengths:
                lines.append(f"  + {s}")
            lines.append("")

        if self.coverage_gaps:
            lines.append("Coverage gaps:")
            for g in self.coverage_gaps:
                lines.append(f"  ? {g}")
            lines.append("")

        if self.doc_divergences:
            lines.append(f"Doc/reality divergences ({len(self.doc_divergences)}):")
            for dd in self.doc_divergences:
                lines.append(f"  Docs: {dd.get('documented', '?')}")
                lines.append(f"  Real: {dd.get('actual', '?')}")
                lines.append(f"  Found by: {dd.get('persona', '?')}")
                lines.append("")

        return "\n".join(lines)


def _build_transcript_digest(transcripts: list[Transcript], max_entries: int = 20) -> str:
    """Condense transcripts into a digestible format for the synthesis agent."""
    parts = []

    for t in transcripts:
        section = [f"--- Persona: {t.persona_name} ---"]

        if t.goals_completed:
            section.append(f"Goals completed: {', '.join(t.goals_completed)}")
        if t.goals_failed:
            section.append(f"Goals failed: {', '.join(t.goals_failed)}")
        if t.issues:
            section.append("Issues reported:")
            for iss in t.issues:
                section.append(f"  - {iss}")
        if t.confusion_points:
            section.append("Confusion points:")
            for cp in t.confusion_points:
                section.append(f"  - {cp}")

        # Include key entries (actions and evaluations, skip verbose observations).
        key_entries = [
            e for e in t.entries
            if e.phase in ("act", "evaluate")
        ][:max_entries]

        if key_entries:
            section.append("Key actions:")
            for e in key_entries:
                section.append(f"  [{e.phase}] {e.content[:200]}")

        parts.append("\n".join(section))

    return "\n\n".join(parts)


async def synthesize(
    transcripts: list[Transcript],
    backend: LLMBackend,
    *,
    max_tokens: int = 4096,
) -> SynthesizedReport:
    """Run the synthesis agent to produce a rich cross-session report."""
    digest = _build_transcript_digest(transcripts)

    persona_count = len(transcripts)
    unique_personas = len({t.persona_name for t in transcripts})
    total_issues = sum(len(t.issues) for t in transcripts)
    total_confusion = sum(len(t.confusion_points) for t in transcripts)

    user_prompt = f"""\
Here are the results from a focus group test with {persona_count} agents ({unique_personas} unique personas).
Total raw issues reported: {total_issues}
Total confusion points: {total_confusion}

Transcript digest:

{digest}

Analyze these results and produce your JSON report. Deduplicate issues that multiple agents found.
Rank issues by severity. Identify patterns across personas.
"""

    messages = [
        Message(role="system", content=SYNTHESIS_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]

    response = await backend.complete(messages, temperature=0.3, max_tokens=max_tokens)

    # Parse the JSON from the response.
    import re
    text = response.content
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text.strip())

    # Find JSON object.
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
                    data = json.loads(text[start : i + 1])
                    return SynthesizedReport.from_dict(data)
                except json.JSONDecodeError:
                    continue

    # Fallback: couldn't parse, return raw text as summary.
    return SynthesizedReport(executive_summary=text[:500])
