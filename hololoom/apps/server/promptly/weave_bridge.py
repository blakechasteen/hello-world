"""
AwarenessGraph bridge — spring dynamics memory for the chat path.

Lean bridge: initializes AwarenessGraph + SemanticCalculus directly
(no full WeavingOrchestrator). Provides physics-based memory recall
via spreading activation on the knowledge graph.

Follows the same lazy-singleton + graceful-degradation pattern as
hololoom_bridge.py. If any dependency is missing, the bridge reports
unavailable and the caller falls back to HoloLoomLite.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================================
# Types
# ============================================================================


@dataclass
class WeavedMemory:
    """A memory recalled via spring dynamics activation."""
    text: str
    relevance: float        # activation level (0-1)
    source: str | None      # context origin (room_id, episode, etc.)
    activation: float       # raw activation field level


# ============================================================================
# Bridge (lazy singleton)
# ============================================================================

_bridge: _WeaveBridge | None = None
_available: bool | None = None


class _WeaveBridge:
    """Thin wrapper around AwarenessGraph for chat-path memory."""

    def __init__(self, graph, semantic):
        self.graph = graph
        self.semantic = semantic

    async def recall(self, query: str, limit: int = 5) -> list[WeavedMemory]:
        """Perceive query → activate graph → return top-k by activation."""
        from hololoom.core.memory.awareness_types import (
            ActivationBudget,
            ActivationStrategy,
        )

        perception = await self.graph.perceive(query)
        memories = await self.graph.activate(
            perception,
            budget=ActivationBudget.for_strategy(ActivationStrategy.BALANCED),
        )

        results = []
        for m in memories[:limit]:
            act_level = self.graph.activation_field.levels.get(m.id, 0.0)
            source = m.context.get("room_id") or m.context.get("source")
            results.append(WeavedMemory(
                text=m.text,
                relevance=float(act_level),
                source=source,
                activation=float(act_level),
            ))

        return results

    async def store(self, query: str, response: str, room_id: str) -> None:
        """Store Q/A pair into the awareness graph for future retrieval."""
        content = f"Q: {query}\nA: {response}"
        perception = await self.graph.perceive(content)
        await self.graph.remember(
            content, perception, context={"room_id": room_id},
        )

    @property
    def memory_count(self) -> int:
        """Number of memories stored in the graph."""
        return self.graph.graph.number_of_nodes()


async def _get_weave_bridge() -> _WeaveBridge | None:
    """Lazy-init the AwarenessGraph bridge. Returns None if unavailable."""
    global _bridge, _available

    if _available is False:
        return None
    if _bridge is not None:
        return _bridge

    try:
        import networkx as nx

        from hololoom.core.memory.awareness_graph import AwarenessGraph
        from hololoom.semantic_calculus import MatryoshkaSemanticCalculus

        semantic = MatryoshkaSemanticCalculus()
        graph = AwarenessGraph(nx.MultiDiGraph(), semantic)
        _bridge = _WeaveBridge(graph, semantic)
        _available = True
        logger.info(
            "Weave bridge initialized (AwarenessGraph + spring dynamics)"
        )
        return _bridge
    except Exception as e:
        _available = False
        logger.info("Weave bridge unavailable: %s", e)
        return None


def is_weave_available() -> bool | None:
    """Return cached availability flag (None = not yet checked)."""
    return _available


# ============================================================================
# Formatted recall (matches hololoom_bridge output shape)
# ============================================================================


async def recall_weaved(
    query: str, limit: int = 5,
) -> tuple[str, list[dict]]:
    """Recall via AwarenessGraph, returning (formatted_text, hits).

    Returns the same shape as _recall_memories_structured() so callers
    can swap transparently.
    """
    bridge = await _get_weave_bridge()
    if not bridge:
        return "", []

    start = time.time()
    memories = await bridge.recall(query, limit=limit)
    elapsed_ms = (time.time() - start) * 1000

    if not memories:
        return "", []

    lines = []
    hits = []
    for m in memories:
        rel_str = f"[{m.relevance:.0%}]" if m.relevance else ""
        lines.append(f"- {rel_str} {m.text}")
        hits.append({
            "text": m.text[:200],
            "relevance": round(m.relevance, 3),
            "source": m.source,
        })

    formatted = "\n\n## Recalled from memory (awareness graph)\n" + "\n".join(lines)
    logger.debug(
        "Weave recall: %d hits in %.0fms (%d memories in graph)",
        len(hits), elapsed_ms, bridge.memory_count,
    )
    return formatted, hits
