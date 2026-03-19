"""
Memory Tools
=============

Query wrappers for Yarn (Neo4j graph) and Warp (Qdrant vectors).
Gracefully degrades when backends are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryTools:

    def __init__(self, yarn=None, warp=None):
        """
        Args:
            yarn: Neo4j knowledge graph instance (or None)
            warp: Qdrant/WarpSpace instance (or None)
        """
        self.yarn = yarn
        self.warp = warp

    async def query_yarn(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query against the Yarn graph."""
        if self.yarn is None:
            return [{"error": "Yarn (Neo4j) not available"}]
        try:
            result = await self.yarn.execute(cypher, params or {})
            return result if isinstance(result, list) else [result]
        except Exception as e:
            logger.warning(f"Yarn query failed: {e}")
            return [{"error": str(e)}]

    async def query_warp(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Run a semantic search against the Warp vector store."""
        if self.warp is None:
            return [{"error": "Warp (Qdrant) not available"}]
        try:
            result = await self.warp.search(query, limit=limit)
            return result if isinstance(result, list) else [result]
        except Exception as e:
            logger.warning(f"Warp query failed: {e}")
            return [{"error": str(e)}]
