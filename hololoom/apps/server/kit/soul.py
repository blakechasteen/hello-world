"""
Soul loading — parameterized system prompt loader for any agent.

Searches SOUL.md on disk across configurable paths, falls back to inline text.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _default_search_paths(name: str) -> list[Path]:
    """Default SOUL.md search paths for a named agent."""
    paths = []
    # Nanoclaw group path (WSL — won't exist on Windows, that's fine)
    nanoclaw = Path(f"/home/blake/nanoclaw/groups/{name}/SOUL.md")
    if nanoclaw.exists():
        paths.append(nanoclaw)
    # apps/{name}/SOUL.md
    paths.append(_REPO_ROOT / "apps" / name / "SOUL.md")
    # Server-level SOUL.md
    paths.append(Path(__file__).resolve().parent.parent / "SOUL.md")
    # Repo root fallback
    paths.append(_REPO_ROOT / "SOUL.md")
    return paths


def load_soul(
    name: str,
    search_paths: list[Path] | None = None,
    fallback: str = "",
    extras: Callable[[], str] | None = None,
) -> str:
    """
    Load a soul (system prompt) from SOUL.md files.

    Args:
        name: Agent name ("elle", "promptly") — used for default search paths
        search_paths: Override the default search path list
        fallback: Inline fallback text if no file found
        extras: Optional callable returning additional system prompt text
                (e.g., Coz context injection, vault context)

    Returns:
        Complete system prompt string.
    """
    paths = search_paths or _default_search_paths(name)

    soul = ""
    for path in paths:
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    logger.info("%s soul loaded from %s", name.capitalize(), path)
                    soul = text
                    break
        except OSError:
            continue

    if not soul:
        if fallback:
            logger.info("Using inline %s soul fallback", name)
            soul = fallback
        else:
            logger.warning("No soul found for %s", name)
            soul = f"You are {name}."

    # Append message format hint if not already present
    if "Message format" not in soul:
        soul += (
            "\n\n## Message format\n\n"
            'Messages arrive as XML: '
            '<messages><message sender="name" time="...">text</message></messages>\n'
            "Multiple messages may arrive at once — read them all, respond naturally.\n"
            "Address people by name when it's not obvious who you're talking to."
        )

    # Inject extras (Coz context, vault context, etc.)
    if extras:
        try:
            extra_text = extras()
            if extra_text:
                soul += "\n\n" + extra_text
        except Exception:
            pass  # Extras are optional — never crash for them

    return soul
