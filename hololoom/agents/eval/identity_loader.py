"""
Identity file loader — composes the 7-file identity system into a system prompt.

Files loaded in attention-priority order (models attend to beginning and end):
    1. agent.md     — structural constraints (highest priority)
    2. soul.md      — values and red lines
    3. tools.md     — available/forbidden tools
    4. USER.md      — who Blake is (shared)
    5. program.md   — evolution rules (lowest static priority)

heartbeat.md and tasks.md are injected as runtime context, not identity.

Complexity modes control how much identity budget is spent:
    BARE:     agent.md only (~200 tokens)
    FAST:     agent.md + soul.md (~400 tokens)
    FULL:     agent.md + soul.md + tools.md + USER.md (~800 tokens)
    RESEARCH: all 5 static files + heartbeat + tasks (~1200 tokens)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

_IDENTITIES_ROOT = Path(__file__).resolve().parent.parent / "identities"


class IdentityMode(str, Enum):
    """How much identity context to load. Maps to weaving complexity modes."""
    BARE = "bare"
    FAST = "fast"
    FULL = "full"
    RESEARCH = "research"


@dataclass
class LoadedIdentity:
    """Result of loading an agent's identity files."""
    agent_name: str
    mode: IdentityMode
    files_loaded: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)
    prompt: str = ""
    token_estimate: int = 0


# Which files to load per mode, in prompt order.
_MODE_FILES: dict[IdentityMode, list[str]] = {
    IdentityMode.BARE: ["agent.md"],
    IdentityMode.FAST: ["agent.md", "soul.md"],
    IdentityMode.FULL: ["agent.md", "soul.md", "tools.md"],
    IdentityMode.RESEARCH: ["agent.md", "soul.md", "tools.md", "program.md"],
}

# Shared files appended after agent-specific files.
_SHARED_FILES: dict[IdentityMode, list[str]] = {
    IdentityMode.BARE: [],
    IdentityMode.FAST: [],
    IdentityMode.FULL: ["USER.md"],
    IdentityMode.RESEARCH: ["USER.md"],
}

# Runtime files (agent-owned, mutable) — only in RESEARCH mode.
_RUNTIME_FILES = ["heartbeat.md", "tasks.md"]


class IdentityLoader:
    """
    Loads and composes identity files for a named agent.

    Usage:
        loader = IdentityLoader()
        identity = loader.load("promptly", mode=IdentityMode.FULL)
        system_prompt = identity.prompt
    """

    def __init__(self, root: Path | None = None):
        self.root = root or _IDENTITIES_ROOT

    def load(
        self,
        agent_name: str,
        mode: IdentityMode = IdentityMode.FULL,
    ) -> LoadedIdentity:
        """Load identity files and compose into a system prompt."""
        agent_dir = self.root / agent_name
        result = LoadedIdentity(agent_name=agent_name, mode=mode)
        sections: list[str] = []

        # Agent-specific files
        for filename in _MODE_FILES.get(mode, []):
            text = self._read(agent_dir / filename)
            if text:
                sections.append(text)
                result.files_loaded.append(filename)
            else:
                result.files_missing.append(filename)
                logger.warning("%s: missing %s", agent_name, filename)

        # Shared files (USER.md lives at identities root)
        for filename in _SHARED_FILES.get(mode, []):
            text = self._read(self.root / filename)
            if text:
                sections.append(text)
                result.files_loaded.append(filename)
            else:
                result.files_missing.append(filename)

        # Runtime files (heartbeat, tasks) — RESEARCH only
        if mode == IdentityMode.RESEARCH:
            for filename in _RUNTIME_FILES:
                text = self._read(agent_dir / filename)
                if text:
                    sections.append(text)
                    result.files_loaded.append(filename)

        result.prompt = "\n\n---\n\n".join(sections)
        result.token_estimate = len(result.prompt) // 4  # rough estimate

        logger.info(
            "%s identity loaded: mode=%s, files=%d, ~%d tokens",
            agent_name, mode.value, len(result.files_loaded), result.token_estimate,
        )
        return result

    def available_agents(self) -> list[str]:
        """List agents that have identity directories."""
        if not self.root.exists():
            return []
        return [
            d.name for d in sorted(self.root.iterdir())
            if d.is_dir() and (d / "agent.md").exists()
        ]

    @staticmethod
    def _read(path: Path) -> str:
        """Read a file, return empty string on failure."""
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        return ""
