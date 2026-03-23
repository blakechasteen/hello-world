"""
HoloLoomLite integration — lazy init, graceful degradation.

Also contains the unified ``recall_smart()`` that gates between the
AwarenessGraph (spring dynamics) and HoloLoomLite based on query
complexity. Callers import from here — the bridge decides which
backend to use.
"""
from __future__ import annotations

import logging

from .weave_bridge import (  # noqa: F401 — re-export
    _get_weave_bridge,
    is_weave_available,
    recall_weaved,
)

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


# ============================================================================
# Unified smart recall — complexity-gated
# ============================================================================

async def recall_smart(
    query: str, complexity: float,
) -> tuple[str, list[dict], str]:
    """Recall memories using the best available backend.

    Complexity gating:
        >= WEAVE_THRESHOLD → AwarenessGraph (spring dynamics)
        >= 0.15            → HoloLoomLite (flat recall)
        < 0.15             → skip (trivial chat)

    Returns:
        (formatted_context, hits, backend_used)
        backend_used is one of: "awareness_graph", "hololoom_lite", "none"
    """
    from .config import WEAVE_ENABLED, WEAVE_THRESHOLD

    # Try AwarenessGraph for complex queries
    if WEAVE_ENABLED and complexity >= WEAVE_THRESHOLD:
        try:
            mem_text, mem_hits = await recall_weaved(query, limit=5)
            if mem_text:
                return mem_text, mem_hits, "awareness_graph"
            # Graph is empty — fall through to HoloLoomLite
        except Exception as e:
            logger.warning("Weave recall failed, falling back: %s", e)

    # Fall back to HoloLoomLite
    if is_hololoom_available() is not False:
        loom = await _get_hololoom()
        if loom:
            mem_text, mem_hits = await _recall_memories_structured(loom, query)
            if mem_text:
                return mem_text, mem_hits, "hololoom_lite"

    return "", [], "none"
