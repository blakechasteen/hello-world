from __future__ import annotations
"""
HoloLoom Worker - Execute HoloLoom operations directly in Node Daemon.

Provides local execution of HoloLoom's memory and reasoning operations
without going through the HTTP API. True distributed intelligence.

Usage:
    worker = HoloLoomWorker()
    result = await worker.execute("recall", {"query": "Thompson Sampling", "k": 5})
"""

import asyncio
import time
from typing import Any

from ..shared.logging import get_logger

logger = get_logger(__name__, component="node")

# Graceful fallback if HoloLoom not available
try:
    from hololoom import HoloLoom
    from hololoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HoloLoom = None
    Config = None
    HOLOLOOM_AVAILABLE = False
    logger.warning("HoloLoom not available - operations will fail gracefully")

# Try to import weaving orchestrator for full reasoning
try:
    from hololoom.protocols.types import Query
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    WeavingOrchestrator = None
    Query = None
    ORCHESTRATOR_AVAILABLE = False


class HoloLoomWorker:
    """
    Executes HoloLoom operations directly (not via HTTP).

    Lazy-initializes HoloLoom on first use. Provides graceful
    degradation if HoloLoom is unavailable.
    """

    def __init__(self, mode: str = "fast"):
        """
        Initialize HoloLoom worker.

        Args:
            mode: Execution mode - "bare", "fast" (default), or "fused"
        """
        self.mode = mode
        self._loom: HoloLoom | None = None
        self._orchestrator: WeavingOrchestrator | None = None
        self._lock = asyncio.Lock()
        self._initialized_at: float | None = None

        # Statistics
        self._stats = {
            "operations_total": 0,
            "operations_success": 0,
            "operations_failed": 0,
            "recall_count": 0,
            "experience_count": 0,
            "weave_count": 0,
            "total_execution_ms": 0.0
        }

    async def _ensure_initialized(self) -> bool:
        """
        Lazy initialization of HoloLoom instance.

        Thread-safe: Only one coroutine can initialize at a time.

        Returns:
            True if HoloLoom initialized, False if unavailable
        """
        if self._loom is not None:
            return True

        if not HOLOLOOM_AVAILABLE:
            logger.error("HoloLoom not available - cannot initialize")
            return False

        async with self._lock:
            # Double-check pattern (another task may have initialized)
            if self._loom is not None:
                return True

            try:
                # Create config based on mode
                if self.mode == "bare":
                    config = Config.bare()
                elif self.mode == "fused":
                    config = Config.fused()
                else:
                    config = Config.fast()

                # Initialize HoloLoom
                self._loom = HoloLoom(config=config)
                self._initialized_at = time.time()
                logger.info(f"HoloLoom initialized in {self.mode} mode")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize HoloLoom: {e}")
                self._loom = None
                return False

    async def execute(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Main dispatch for HoloLoom operations.

        Args:
            operation: Operation name ("recall", "experience", "weave")
            params: Operation parameters

        Returns:
            Result dict with status, output, timing, and error info
        """
        start_time = time.time()

        try:
            # Ensure HoloLoom is available
            if not await self._ensure_initialized():
                return {
                    "status": "error",
                    "error": "HoloLoom not available",
                    "execution_time_ms": (time.time() - start_time) * 1000
                }

            # Track operation
            self._stats["operations_total"] += 1

            # Dispatch to operation handler
            if operation == "recall":
                self._stats["recall_count"] += 1
                result = await self.recall(
                    params.get("query", ""),
                    params.get("k", 5)
                )
            elif operation == "experience":
                self._stats["experience_count"] += 1
                result = await self.experience(params.get("content", ""))
            elif operation == "weave":
                self._stats["weave_count"] += 1
                result = await self.weave(
                    params.get("query", ""),
                    params.get("mode", self.mode)
                )
            elif operation == "status":
                result = self.status()
            elif operation == "health":
                result = await self.health_check()
            else:
                self._stats["operations_failed"] += 1
                return {
                    "status": "error",
                    "error": f"Unknown operation: {operation}",
                    "execution_time_ms": (time.time() - start_time) * 1000
                }

            # Add timing and track success
            execution_ms = (time.time() - start_time) * 1000
            result["execution_time_ms"] = execution_ms
            self._stats["total_execution_ms"] += execution_ms
            self._stats["operations_success"] += 1
            return result

        except Exception as e:
            self._stats["operations_failed"] += 1
            logger.error(f"Operation {operation} failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            }

    async def recall(self, query: str, k: int = 5) -> dict[str, Any]:
        """
        Local memory search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            Dict with status and memory results
        """
        if self._loom is None:
            return {"status": "error", "error": "HoloLoom not initialized"}

        try:
            memories = await self._loom.recall(query, limit=k)

            return {
                "status": "success",
                "query": query,
                "memories": [
                    {
                        "id": m.id,
                        "text": m.text,
                        "relevance": m.relevance,
                        "timestamp": m.timestamp
                    }
                    for m in memories
                ],
                "count": len(memories)
            }

        except Exception as e:
            logger.error(f"Recall failed for query '{query}': {e}")
            return {"status": "error", "error": str(e)}

    async def experience(self, content: str) -> dict[str, Any]:
        """
        Local memory store.

        Args:
            content: Content to remember

        Returns:
            Dict with status and memory ID
        """
        if self._loom is None:
            return {"status": "error", "error": "HoloLoom not initialized"}

        try:
            memory = await self._loom.experience(content)

            return {
                "status": "success",
                "memory_id": memory.id,
                "content": content,
                "timestamp": memory.timestamp
            }

        except Exception as e:
            logger.error(f"Experience failed for content: {e}")
            return {"status": "error", "error": str(e)}

    async def weave(self, query: str, mode: str = "fast") -> dict[str, Any]:
        """
        Local reasoning (full weaving cycle).

        Uses WeavingOrchestrator if available for full 9-step weaving,
        otherwise falls back to recall + context assembly.

        Args:
            query: Query to reason about
            mode: Execution mode ("bare", "fast", "fused")

        Returns:
            Dict with status, response, confidence, and metadata
        """
        if self._loom is None:
            return {"status": "error", "error": "HoloLoom not initialized"}

        try:
            # Use configured mode if not specified
            if mode not in ("bare", "fast", "fused"):
                mode = self.mode

            # Try full orchestrator if available
            if ORCHESTRATOR_AVAILABLE and WeavingOrchestrator is not None:
                return await self._weave_with_orchestrator(query, mode)

            # Fallback: recall + simple context assembly
            memories = await self._loom.recall(query, limit=10)
            context = "\n".join([m.text for m in memories[:3]])

            return {
                "status": "success",
                "query": query,
                "mode": mode,
                "response": f"Based on memory: {context}" if context else "No relevant memories found.",
                "memories_used": len(memories),
                "confidence": min(0.5 + (len(memories) * 0.05), 0.9),
                "orchestrator": False
            }

        except Exception as e:
            logger.error(f"Weave failed for query '{query}': {e}")
            return {"status": "error", "error": str(e)}

    async def _weave_with_orchestrator(self, query: str, mode: str) -> dict[str, Any]:
        """
        Full weaving with WeavingOrchestrator.

        Args:
            query: Query to reason about
            mode: Execution mode

        Returns:
            Dict with full weaving results
        """
        try:
            # Select config based on mode
            if mode == "bare":
                config = Config.bare()
            elif mode == "fused":
                config = Config.fused()
            else:
                config = Config.fast()

            # Create orchestrator and weave
            async with WeavingOrchestrator(cfg=config, shards=[]) as orchestrator:
                spacetime = await orchestrator.weave(Query(text=query))

                return {
                    "status": "success",
                    "query": query,
                    "mode": mode,
                    "response": spacetime.response if hasattr(spacetime, 'response') else str(spacetime),
                    "confidence": spacetime.confidence if hasattr(spacetime, 'confidence') else 0.75,
                    "tool_used": spacetime.metadata.get('tool_used') if hasattr(spacetime, 'metadata') else None,
                    "orchestrator": True
                }

        except Exception as e:
            logger.warning(f"Orchestrator failed, using fallback: {e}")
            # Fall back to simple recall
            memories = await self._loom.recall(query, limit=10)
            context = "\n".join([m.text for m in memories[:3]])

            return {
                "status": "success",
                "query": query,
                "mode": mode,
                "response": f"Based on memory: {context}" if context else "No relevant memories found.",
                "memories_used": len(memories),
                "confidence": min(0.5 + (len(memories) * 0.05), 0.9),
                "orchestrator": False,
                "orchestrator_error": str(e)
            }

    async def close(self):
        """Graceful shutdown of HoloLoom instance."""
        if self._loom is not None:
            try:
                # HoloLoom cleanup if needed
                self._loom = None
                self._orchestrator = None
                logger.info("HoloLoom worker closed")
            except Exception as e:
                logger.error(f"Error closing HoloLoom: {e}")

    def status(self) -> dict[str, Any]:
        """
        Get worker status and statistics.

        Returns:
            Dict with status, mode, availability, and stats
        """
        uptime_seconds = None
        if self._initialized_at:
            uptime_seconds = time.time() - self._initialized_at

        avg_execution_ms = 0.0
        if self._stats["operations_total"] > 0:
            avg_execution_ms = self._stats["total_execution_ms"] / self._stats["operations_total"]

        return {
            "status": "success",
            "worker": {
                "initialized": self._loom is not None,
                "mode": self.mode,
                "uptime_seconds": uptime_seconds,
                "hololoom_available": HOLOLOOM_AVAILABLE,
                "orchestrator_available": ORCHESTRATOR_AVAILABLE
            },
            "stats": {
                **self._stats,
                "avg_execution_ms": round(avg_execution_ms, 2)
            }
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on HoloLoom worker.

        Returns:
            Dict with health status and details
        """
        health = {
            "status": "healthy",
            "checks": {}
        }

        # Check 1: HoloLoom availability
        health["checks"]["hololoom_available"] = {
            "status": "pass" if HOLOLOOM_AVAILABLE else "fail",
            "message": "HoloLoom module available" if HOLOLOOM_AVAILABLE else "HoloLoom not installed"
        }

        # Check 2: Initialization
        health["checks"]["initialized"] = {
            "status": "pass" if self._loom is not None else "warn",
            "message": "HoloLoom initialized" if self._loom else "Not yet initialized (lazy init)"
        }

        # Check 3: Can initialize (if not already)
        if self._loom is None and HOLOLOOM_AVAILABLE:
            try:
                can_init = await self._ensure_initialized()
                health["checks"]["can_initialize"] = {
                    "status": "pass" if can_init else "fail",
                    "message": "Successfully initialized" if can_init else "Failed to initialize"
                }
            except Exception as e:
                health["checks"]["can_initialize"] = {
                    "status": "fail",
                    "message": f"Initialization error: {e}"
                }

        # Check 4: Memory operation (if initialized)
        if self._loom is not None:
            try:
                # Quick recall test
                memories = await self._loom.recall("health check test", limit=1)
                health["checks"]["memory_operation"] = {
                    "status": "pass",
                    "message": f"Memory recall works (found {len(memories)} results)"
                }
            except Exception as e:
                health["checks"]["memory_operation"] = {
                    "status": "fail",
                    "message": f"Memory error: {e}"
                }

        # Determine overall health
        statuses = [c["status"] for c in health["checks"].values()]
        if "fail" in statuses:
            health["status"] = "unhealthy"
        elif "warn" in statuses:
            health["status"] = "degraded"

        return health
