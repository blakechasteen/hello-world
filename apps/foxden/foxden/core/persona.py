"""Persona definitions — YAML-driven agent personalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Behavior:
    """How an agent behaves during a session."""

    patience: str = "medium"  # low | medium | high
    reads_docs: bool = True
    exploration: str = "guided"  # random | guided | methodical
    error_tolerance: int = 3  # failures before giving up
    speed: str = "normal"  # slow | normal | fast (affects wait times, reading)
    verbosity: str = "normal"  # terse | normal | verbose (how much the agent reports)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Behavior:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def patience_budget(self) -> int:
        """Total frustration budget before the agent quits entirely."""
        return {"low": 15, "medium": 30, "high": 60}.get(self.patience, 30)

    @property
    def frustration_rate(self) -> float:
        """How fast frustration builds per failure."""
        return {"low": 3.0, "medium": 1.5, "high": 0.5}.get(self.patience, 1.5)

    @property
    def recovery_rate(self) -> float:
        """How much frustration drops per success."""
        return {"low": 0.5, "medium": 1.0, "high": 2.0}.get(self.patience, 1.0)


@dataclass(frozen=True)
class Persona:
    """A simulated user personality for testing."""

    name: str
    system_prompt: str
    goals: list[str]
    behavior: Behavior = field(default_factory=Behavior)
    success_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Persona:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"].strip(),
            goals=data.get("goals", []),
            behavior=Behavior.from_dict(data.get("behavior", {})),
            success_criteria=data.get("success_criteria", []),
            tags=data.get("tags", []),
        )


def load_personas(directory: str | Path) -> dict[str, Persona]:
    """Load all persona YAML files from a directory."""
    directory = Path(directory)
    personas = {}
    for path in sorted(directory.glob("*.yaml")):
        p = Persona.from_yaml(path)
        personas[p.name] = p
    return personas
