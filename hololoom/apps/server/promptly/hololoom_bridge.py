"""
HoloLoomLite integration — lazy init, graceful degradation.

Also contains vault bridge helpers used by the chat endpoint.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# HoloLoomLite (lazy singleton)
# ============================================================================

_hololoom_lite = None
_hololoom_available: bool | None = None


async def _get_hololoom():
    """Lazy-init HoloLoomLite. Returns None if unavailable."""
    global _hololoom_lite, _hololoom_available
    if _hololoom_available is False:
        return None
    if _hololoom_lite is not None:
        return _hololoom_lite

    try:
        from hololoom import HoloLoomLite
        from hololoom.config import Config
        config = Config.lite()
        loom = HoloLoomLite(config=config)
        await loom.__aenter__()
        _hololoom_lite = loom
        _hololoom_available = True
        logger.info("HoloLoomLite initialized for refinement pass")
        return loom
    except Exception as e:
        _hololoom_available = False
        logger.info("HoloLoomLite not available (refinement disabled): %s", e)
        return None


def is_hololoom_available() -> bool | None:
    """Return cached availability flag (None = not yet checked)."""
    return _hololoom_available


# ============================================================================
# Memory helpers
# ============================================================================

async def _recall_memories(loom, query: str) -> str:
    """Recall relevant memories from HoloLoomLite, return as formatted context."""
    if not loom:
        return ""
    try:
        memories = await loom.recall(query, limit=5)
        if not memories:
            return ""
        lines = []
        for m in memories:
            text = getattr(m, "text", str(m))
            relevance = getattr(m, "relevance", None)
            if relevance:
                lines.append(f"- [{relevance:.0%}] {text}")
            else:
                lines.append(f"- {text}")
        return "\n\n## Recalled from memory\n" + "\n".join(lines)
    except Exception as e:
        logger.warning("HoloLoomLite recall failed: %s", e)
        return ""


async def _recall_memories_structured(loom, query: str) -> tuple[str, list[dict]]:
    """Recall memories, returning both formatted text and structured hits.

    Returns:
        (formatted_context, hits) where hits are dicts with text/relevance/source.
    """
    if not loom:
        return "", []
    try:
        memories = await loom.recall(query, limit=5)
        if not memories:
            return "", []
        lines = []
        hits = []
        for m in memories:
            text = getattr(m, "text", str(m))
            relevance = getattr(m, "relevance", None)
            source = getattr(m, "episode", None) or getattr(m, "source", None)
            if relevance:
                lines.append(f"- [{relevance:.0%}] {text}")
            else:
                lines.append(f"- {text}")
            hits.append({
                "text": text[:200],
                "relevance": round(relevance, 3) if relevance else None,
                "source": source,
            })
        formatted = "\n\n## Recalled from memory\n" + "\n".join(lines)
        return formatted, hits
    except Exception as e:
        logger.warning("HoloLoomLite structured recall failed: %s", e)
        return "", []


async def _store_experience(loom, query: str, response: str, room_id: str,
                            pass_number: int) -> None:
    """Store an interaction as experience in HoloLoomLite."""
    if not loom:
        return
    try:
        label = f"pass {pass_number}" if pass_number > 1 else "initial"
        await loom.experience(
            f"Q: {query}\nA ({label}): {response}",
            context={"room_id": room_id, "pass": str(pass_number)},
        )
    except Exception:
        pass
