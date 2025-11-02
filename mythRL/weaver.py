"""High-level async Weaver API wrapping the Shuttle demo implementation."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dev.protocol_modules_mythrl import (
    DemoDecisionEngine,
    DemoFeatureExtraction,
    DemoMemoryBackend,
    DemoPatternSelection,
    DemoToolExecution,
    DemoWarpSpace,
    MythRLResult,
    MythRLShuttle,
)
from HoloLoom.protocols import ComplexityLevel


@dataclass
class WeaverResult:
    """Structured result returned by ``Weaver.query`` and ``Weaver.chat``."""

    query: str
    response: str
    pattern: str
    confidence: float
    duration_ms: float
    context_count: int
    tool: str
    raw_result: MythRLResult = field(repr=False)


class Weaver:
    """Simple facade around :class:`MythRLShuttle` for demos and tests."""

    def __init__(self, mode: str = "fast", intelligent_routing: bool = True) -> None:
        self.mode = mode.lower()
        self.intelligent_routing = intelligent_routing
        self._history: List[Dict[str, str]] = []
        self._shards: List[str] = []
        self._shuttle = MythRLShuttle()
        self._memory_backend = DemoMemoryBackend()
        self._shuttle.register_protocol("memory_backend", self._memory_backend)
        self._shuttle.register_protocol("feature_extraction", DemoFeatureExtraction())
        self._shuttle.register_protocol("pattern_selection", DemoPatternSelection())
        self._shuttle.register_protocol("decision_engine", DemoDecisionEngine())
        self._shuttle.register_protocol("warp_space", DemoWarpSpace())
        self._shuttle.register_protocol("tool_execution", DemoToolExecution())
        self._closed = False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    @classmethod
    async def create(
        cls,
        pattern: Optional[str] = None,
        mode: str = "fast",
        intelligent_routing: bool = True,
        knowledge: Optional[str] = None,
        **_: Any,
    ) -> "Weaver":
        """Factory matching the legacy ``Weaver.create`` signature."""

        weaver = cls(mode=pattern or mode, intelligent_routing=intelligent_routing)
        if knowledge:
            await weaver.ingest(knowledge)
        return weaver

    async def close(self) -> None:
        self._closed = True

    async def __aenter__(self) -> "Weaver":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def query(self, query: str) -> WeaverResult:
        self._ensure_open()
        return await self._run_query(query, use_chat_context=False)

    async def chat(self, message: str) -> WeaverResult:
        self._ensure_open()
        return await self._run_query(message, use_chat_context=True)

    async def ingest(self, knowledge: str) -> int:
        self._ensure_open()
        chunks = [chunk.strip() for chunk in knowledge.splitlines() if chunk.strip()]
        for chunk in chunks:
            shard_id = f"k_{uuid.uuid4().hex[:8]}"
            self._memory_backend.knowledge_graph[shard_id] = {
                "id": shard_id,
                "content": chunk,
                "relevance": 0.72,
                "related": [],
            }
            self._shards.append(chunk)
        return len(chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _run_query(self, query: str, use_chat_context: bool) -> WeaverResult:
        start = time.perf_counter()
        context = {
            "mode": self.mode,
            "history": list(self._history) if use_chat_context else [],
            "complexity_bias": 0.3 if self.mode == "full" else 0.0,
        }
        myth_result = await self._shuttle.weave(query, context)
        duration_ms = (time.perf_counter() - start) * 1000

        pattern = self._choose_pattern(myth_result)
        response = self._render_response(query, myth_result)
        context_count = len(myth_result.output.get("memory_results", []))
        tool = myth_result.output.get("tool_result", {}).get("tool", "respond")

        weaver_result = WeaverResult(
            query=query,
            response=response,
            pattern=pattern,
            confidence=myth_result.confidence,
            duration_ms=duration_ms,
            context_count=context_count,
            tool=tool,
            raw_result=myth_result,
        )

        self._history.append({"user": query, "response": response})
        return weaver_result

    def _choose_pattern(self, myth_result: MythRLResult) -> str:
        selected = myth_result.output.get("pattern", {}).get("selected_pattern")
        if selected:
            return selected
        if myth_result.complexity_level is ComplexityLevel.RESEARCH:
            return "research"
        if myth_result.complexity_level is ComplexityLevel.FULL:
            return "full"
        if myth_result.complexity_level is ComplexityLevel.FAST:
            return "fast"
        return "lite"

    def _render_response(self, query: str, myth_result: MythRLResult) -> str:
        if myth_result.output.get("memory_results"):
            snippets = [item["content"] for item in myth_result.output["memory_results"][:3]]
            joined = " | ".join(snippets)
            return f"SYNTHESIZED({query[:30]}): {joined}"
        return f"Processed query: {query}"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Weaver has been closed")

    # Convenience methods for tests
    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)


__all__ = ["Weaver", "WeaverResult"]
