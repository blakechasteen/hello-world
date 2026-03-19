"""Transcript recorder — captures everything an agent does."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TranscriptEntry:
    timestamp: str
    phase: str  # "observe" | "think" | "act" | "evaluate"
    content: str
    metadata: dict | None = None

    def to_dict(self) -> dict:
        d = {"timestamp": self.timestamp, "phase": self.phase, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class Transcript:
    """Full recording of an agent session."""

    persona_name: str
    entries: list[TranscriptEntry] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    confusion_points: list[str] = field(default_factory=list)
    goals_completed: list[str] = field(default_factory=list)
    goals_failed: list[str] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str | None = None

    def record(self, phase: str, content: str, **metadata) -> None:
        self.entries.append(
            TranscriptEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                phase=phase,
                content=content,
                metadata=metadata or None,
            )
        )

    def finish(self) -> None:
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "persona": self.persona_name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "goals_completed": self.goals_completed,
            "goals_failed": self.goals_failed,
            "issues": self.issues,
            "confusion_points": self.confusion_points,
            "entries": [e.to_dict() for e in self.entries],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @property
    def summary(self) -> str:
        lines = [f"Agent: {self.persona_name}"]
        lines.append(f"Goals completed: {len(self.goals_completed)}/{len(self.goals_completed) + len(self.goals_failed)}")
        if self.issues:
            lines.append(f"Issues found: {len(self.issues)}")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        if self.confusion_points:
            lines.append(f"Confusion points: {len(self.confusion_points)}")
            for cp in self.confusion_points:
                lines.append(f"  - {cp}")
        return "\n".join(lines)
