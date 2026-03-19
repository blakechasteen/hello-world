"""
Ritual Memory — dual write of RitualOutcome to Yarn (Neo4j) and Warp (Qdrant).

Yarn: RitualHistoryNode with edges to AgentNode, TaskNode, SessionNode.
Warp: Embedded text of outcome for semantic retrieval across sessions.

Both writes are fire-and-forget. A backend failure never blocks ritual execution.
"""

import hashlib
import logging
import uuid
from typing import Any

from .types import RitualContext, RitualOutcome

logger = logging.getLogger(__name__)


async def write_outcome(outcome: RitualOutcome, ctx: RitualContext) -> None:
    """
    Dual write:
      1. Yarn (Neo4j) — RitualHistoryNode with relationship edges
      2. Warp (Qdrant) — embedded text for semantic retrieval
    """
    await _write_to_yarn(outcome, ctx)
    await _embed_to_warp(outcome, ctx)


# ============================================================================
# Yarn (Neo4j)
# ============================================================================

_MERGE_RITUAL_NODE = """
MERGE (r:RitualHistoryNode {id: $id})
SET r.ritual_id = $ritual_id,
    r.ritual_version = $ritual_version,
    r.session_id = $session_id,
    r.agent_id = $agent_id,
    r.task_context = $task_context,
    r.fired_at = $fired_at,
    r.duration_ms = $duration_ms,
    r.tokens_used = $tokens_used,
    r.confidence_delta = $confidence_delta,
    r.was_skipped = $was_skipped,
    r.skip_reason = $skip_reason,
    r.failure_policy_invoked = $failure_policy_invoked,
    r.downstream_task_success = $downstream_task_success
"""

_MERGE_AGENT_EDGE = """
MATCH (r:RitualHistoryNode {id: $id})
MERGE (a:AgentNode {agent_id: $agent_id})
MERGE (r)-[:EXECUTED_BY]->(a)
"""

_MERGE_SESSION_EDGE = """
MATCH (r:RitualHistoryNode {id: $id})
MERGE (s:SessionNode {session_id: $session_id})
MERGE (r)-[:IN_SESSION]->(s)
"""


async def _write_to_yarn(outcome: RitualOutcome, ctx: RitualContext) -> None:
    """
    Create RitualHistoryNode in Neo4j, attach edges to AgentNode and SessionNode.

    ctx.yarn is expected to be a Neo4j driver (with .session() method)
    or None (graceful degradation).
    """
    if ctx.yarn is None:
        logger.debug("Yarn handle not available — skipping RitualHistoryNode write")
        return

    node_id = uuid.uuid4().hex[:12]
    params = {
        "id": node_id,
        "ritual_id": outcome.ritual_id,
        "ritual_version": outcome.ritual_version,
        "session_id": outcome.session_id,
        "agent_id": outcome.agent_id,
        "task_context": outcome.task_context,
        "fired_at": outcome.fired_at.isoformat() if outcome.fired_at else "",
        "duration_ms": outcome.duration_ms,
        "tokens_used": outcome.tokens_used,
        "confidence_delta": outcome.confidence_delta,
        "was_skipped": outcome.was_skipped,
        "skip_reason": outcome.skip_reason,
        "failure_policy_invoked": outcome.failure_policy_invoked,
        "downstream_task_success": outcome.downstream_task_success,
    }

    try:
        with ctx.yarn.session() as session:
            session.run(_MERGE_RITUAL_NODE, params)
            session.run(_MERGE_AGENT_EDGE, params)
            session.run(_MERGE_SESSION_EDGE, params)
        logger.debug(
            "RitualHistoryNode written: %s v%s (skipped=%s, delta=%.3f)",
            outcome.ritual_id, outcome.ritual_version,
            outcome.was_skipped, outcome.confidence_delta,
        )
    except Exception as e:
        logger.warning("Failed to write RitualHistoryNode to Yarn: %s", e)


# ============================================================================
# Warp (Qdrant)
# ============================================================================

RITUAL_OUTCOMES_COLLECTION = "ritual_outcomes"


def _text_to_qdrant_id(text: str) -> int:
    """Convert text to a deterministic integer ID for Qdrant (MD5-based)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:16], 16)


def _hash_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Deterministic hash-based embedding for testing/fallback.
    Not semantic — use the project's Matryoshka embedder when available.
    """
    h = hashlib.sha256(text.encode()).digest()
    raw = []
    for i in range(dim):
        byte_idx = i % len(h)
        raw.append((h[byte_idx] + i) % 256 / 255.0 * 2 - 1)
    # L2 normalize
    norm = sum(x * x for x in raw) ** 0.5
    if norm > 0:
        raw = [x / norm for x in raw]
    return raw


async def _embed_to_warp(outcome: RitualOutcome, ctx: RitualContext) -> None:
    """
    Embed outcome text into Qdrant for semantic retrieval.

    ctx.warp is expected to be a Qdrant client (with .upsert() method)
    or None (graceful degradation).
    """
    if ctx.warp is None:
        logger.debug("Warp handle not available — skipping ritual outcome embedding")
        return

    text = (
        f"{outcome.ritual_id} v{outcome.ritual_version} "
        f"domain:{ctx.domain} task:{outcome.task_context} "
        f"skipped:{outcome.was_skipped} "
        f"confidence_delta:{outcome.confidence_delta} "
        f"success:{outcome.downstream_task_success}"
    )

    try:
        # Try project embedder, fall back to hash
        embed_fn = getattr(ctx, "embed_fn", None)
        if embed_fn is not None:
            embedding = await embed_fn(text)
        else:
            embedding = _hash_embedding(text)

        point_id = _text_to_qdrant_id(outcome.ritual_id + outcome.session_id)
        payload = {
            "ritual_id": outcome.ritual_id,
            "ritual_version": outcome.ritual_version,
            "domain": ctx.domain,
            "task_context": outcome.task_context,
            "was_skipped": outcome.was_skipped,
            "confidence_delta": outcome.confidence_delta,
            "downstream_task_success": outcome.downstream_task_success,
            "fired_at": outcome.fired_at.isoformat() if outcome.fired_at else "",
        }

        # Qdrant upsert — supports both qdrant_client and mock protocols
        ctx.warp.upsert(
            collection_name=RITUAL_OUTCOMES_COLLECTION,
            points=[{"id": point_id, "vector": embedding, "payload": payload}],
        )
        logger.debug("Ritual outcome embedded to Warp: %s", outcome.ritual_id)
    except Exception as e:
        logger.warning("Failed to embed ritual outcome to Warp: %s", e)
