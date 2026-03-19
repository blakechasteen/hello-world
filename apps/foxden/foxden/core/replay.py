"""Replay and comparison — load past sessions and diff across runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..recording.transcript import Transcript


@dataclass
class RunSnapshot:
    """A saved focus group run that can be loaded for comparison."""

    path: Path
    transcripts: list[Transcript]
    issues: list[dict]
    confusion_points: list[dict]
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> RunSnapshot:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        transcripts = []
        for td in data.get("transcripts", []):
            t = Transcript(persona_name=td["persona"])
            t.issues = td.get("issues", [])
            t.confusion_points = td.get("confusion_points", [])
            t.goals_completed = td.get("goals_completed", [])
            t.goals_failed = td.get("goals_failed", [])
            t.started_at = td.get("started_at", "")
            t.ended_at = td.get("ended_at")
            transcripts.append(t)

        return cls(
            path=path,
            transcripts=transcripts,
            issues=data.get("issues", []),
            confusion_points=data.get("confusion_points", []),
            meta={
                "total_agents": data.get("total_agents", 0),
                "goals_completed": data.get("goals_completed", 0),
                "goals_failed": data.get("goals_failed", 0),
            },
        )


@dataclass
class Diff:
    """Comparison between two focus group runs."""

    before: RunSnapshot
    after: RunSnapshot
    new_issues: list[str] = field(default_factory=list)
    resolved_issues: list[str] = field(default_factory=list)
    persistent_issues: list[str] = field(default_factory=list)
    new_confusion: list[str] = field(default_factory=list)
    resolved_confusion: list[str] = field(default_factory=list)
    goal_delta: int = 0  # positive = more goals completed

    @property
    def summary(self) -> str:
        lines = [
            f"=== Run Comparison ===",
            f"Before: {self.before.path.name}  ({self.before.meta.get('total_agents', '?')} agents)",
            f"After:  {self.after.path.name}  ({self.after.meta.get('total_agents', '?')} agents)",
            "",
        ]

        before_completed = self.before.meta.get("goals_completed", 0)
        after_completed = self.after.meta.get("goals_completed", 0)
        delta = after_completed - before_completed
        sign = "+" if delta > 0 else ""
        lines.append(f"Goals completed: {before_completed} → {after_completed} ({sign}{delta})")
        lines.append("")

        if self.resolved_issues:
            lines.append(f"Resolved issues ({len(self.resolved_issues)}):")
            for iss in self.resolved_issues:
                lines.append(f"  - {iss}")
            lines.append("")

        if self.new_issues:
            lines.append(f"NEW issues ({len(self.new_issues)}):")
            for iss in self.new_issues:
                lines.append(f"  + {iss}")
            lines.append("")

        if self.persistent_issues:
            lines.append(f"Persistent issues ({len(self.persistent_issues)}):")
            for iss in self.persistent_issues:
                lines.append(f"  = {iss}")
            lines.append("")

        if self.resolved_confusion:
            lines.append(f"Resolved confusion ({len(self.resolved_confusion)}):")
            for c in self.resolved_confusion:
                lines.append(f"  - {c}")
            lines.append("")

        if self.new_confusion:
            lines.append(f"NEW confusion ({len(self.new_confusion)}):")
            for c in self.new_confusion:
                lines.append(f"  + {c}")

        if not (self.new_issues or self.resolved_issues or self.new_confusion or self.resolved_confusion):
            lines.append("No changes detected between runs.")

        return "\n".join(lines)


def compare(before: RunSnapshot, after: RunSnapshot) -> Diff:
    """Compare two run snapshots and produce a diff."""
    before_issues = _issue_set(before)
    after_issues = _issue_set(after)

    before_confusion = _confusion_set(before)
    after_confusion = _confusion_set(after)

    return Diff(
        before=before,
        after=after,
        new_issues=sorted(after_issues - before_issues),
        resolved_issues=sorted(before_issues - after_issues),
        persistent_issues=sorted(before_issues & after_issues),
        new_confusion=sorted(after_confusion - before_confusion),
        resolved_confusion=sorted(before_confusion - after_confusion),
        goal_delta=(
            after.meta.get("goals_completed", 0)
            - before.meta.get("goals_completed", 0)
        ),
    )


def _issue_set(snapshot: RunSnapshot) -> set[str]:
    """Extract normalized issue strings from a snapshot."""
    issues = set()
    for iss in snapshot.issues:
        desc = iss.get("description", "") if isinstance(iss, dict) else str(iss)
        issues.add(desc.strip().lower())
    for t in snapshot.transcripts:
        for iss in t.issues:
            issues.add(iss.strip().lower())
    issues.discard("")
    return issues


def _confusion_set(snapshot: RunSnapshot) -> set[str]:
    """Extract normalized confusion strings from a snapshot."""
    points = set()
    for cp in snapshot.confusion_points:
        desc = cp.get("description", "") if isinstance(cp, dict) else str(cp)
        points.add(desc.strip().lower())
    for t in snapshot.transcripts:
        for cp in t.confusion_points:
            points.add(cp.strip().lower())
    points.discard("")
    return points
