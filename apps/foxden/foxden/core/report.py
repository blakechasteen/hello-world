"""Report — synthesize findings across all agent sessions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..recording.transcript import Transcript


@dataclass
class Issue:
    description: str
    reported_by: list[str]  # persona names
    frequency: int = 1


@dataclass
class Report:
    """Synthesized findings from a focus group run."""

    transcripts: list[Transcript]
    issues: list[Issue] = field(default_factory=list)
    confusion_points: list[Issue] = field(default_factory=list)

    def __post_init__(self):
        if not self.issues:
            self.issues = self._deduplicate(
                [(t.persona_name, iss) for t in self.transcripts for iss in t.issues]
            )
        if not self.confusion_points:
            self.confusion_points = self._deduplicate(
                [(t.persona_name, cp) for t in self.transcripts for cp in t.confusion_points]
            )

    @staticmethod
    def _deduplicate(items: list[tuple[str, str]]) -> list[Issue]:
        """Group similar items. For now, exact match. Future: semantic similarity."""
        groups: dict[str, list[str]] = {}
        for persona, text in items:
            key = text.strip().lower()
            groups.setdefault(key, []).append(persona)
        return sorted(
            [
                Issue(description=text, reported_by=personas, frequency=len(personas))
                for text, personas in groups.items()
            ],
            key=lambda i: i.frequency,
            reverse=True,
        )

    @property
    def total_agents(self) -> int:
        return len(self.transcripts)

    @property
    def goals_completed(self) -> int:
        return sum(len(t.goals_completed) for t in self.transcripts)

    @property
    def goals_failed(self) -> int:
        return sum(len(t.goals_failed) for t in self.transcripts)

    @property
    def summary(self) -> str:
        lines = [
            f"=== Focus Group Report ===",
            f"Agents: {self.total_agents}",
            f"Goals completed: {self.goals_completed}, failed: {self.goals_failed}",
            "",
        ]

        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            for iss in self.issues:
                reporters = ", ".join(iss.reported_by)
                lines.append(f"  [{iss.frequency}x] {iss.description} (by: {reporters})")
            lines.append("")

        if self.confusion_points:
            lines.append(f"Confusion points ({len(self.confusion_points)}):")
            for cp in self.confusion_points:
                reporters = ", ".join(cp.reported_by)
                lines.append(f"  [{cp.frequency}x] {cp.description} (by: {reporters})")
            lines.append("")

        # Per-agent summaries.
        lines.append("Per-agent breakdown:")
        for t in self.transcripts:
            lines.append(f"  {t.persona_name}: {len(t.goals_completed)} completed, "
                         f"{len(t.goals_failed)} failed, {len(t.issues)} issues")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_agents": self.total_agents,
            "goals_completed": self.goals_completed,
            "goals_failed": self.goals_failed,
            "issues": [
                {"description": i.description, "reported_by": i.reported_by, "frequency": i.frequency}
                for i in self.issues
            ],
            "confusion_points": [
                {"description": c.description, "reported_by": c.reported_by, "frequency": c.frequency}
                for c in self.confusion_points
            ],
            "transcripts": [t.to_dict() for t in self.transcripts],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
