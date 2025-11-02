"""mythRL Shuttle-centric architecture helpers.

This module provides a lightweight but fully async implementation of the
Shuttle orchestration pattern described in the repo documentation.  It keeps
protocol definitions sourced from ``HoloLoom.protocols`` as the single source of
truth and exposes convenience demo implementations for quick experiments.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from HoloLoom.protocols import (
    ComplexityLevel,
    DecisionEngineProtocol,
    FeatureExtractionProtocol,
    MythRLResult,
    PatternSelectionProtocol,
    ProvenceTrace,
    ToolExecutionProtocol,
    WarpSpaceProtocol,
)

__all__ = [
    "ComplexityLevel",
    "ProvenceTrace",
    "MythRLResult",
    "MythRLShuttle",
    "DemoMemoryBackend",
    "DemoPatternSelection",
    "DemoDecisionEngine",
    "DemoFeatureExtraction",
    "DemoWarpSpace",
    "DemoToolExecution",
]


@dataclass
class _CrawlStats:
    passes: int
    total_items: int
    depth_stats: Dict[int, int]
    fusion_events: int


class MythRLShuttle:
    """Creative orchestrator that progressively activates protocols.

    The real power of the Shuttle comes from coordinating multiple
    protocol implementations while keeping the orchestration logic here.
    This implementation focuses on the progressive 3-5-7-9 complexity
    system, multipass memory crawling, and provenance tracking.
    """

    _REGISTERABLE_PROTOCOLS: Sequence[str] = (
        "pattern_selection",
        "decision_engine",
        "memory_backend",
        "feature_extraction",
        "warp_space",
        "tool_execution",
    )

    def __init__(self) -> None:
        self.pattern_selection: Optional[PatternSelectionProtocol] = None
        self.decision_engine: Optional[DecisionEngineProtocol] = None
        self.memory_backend: Optional[Any] = None
        self.feature_extraction: Optional[FeatureExtractionProtocol] = None
        self.warp_space: Optional[WarpSpaceProtocol] = None
        self.tool_execution: Optional[ToolExecutionProtocol] = None

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_protocol(self, protocol_name: str, implementation: Any) -> None:
        if protocol_name not in self._REGISTERABLE_PROTOCOLS:
            raise ValueError(f"Unknown protocol '{protocol_name}'")
        setattr(self, protocol_name, implementation)

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------
    async def weave(self, query: str, context: Optional[Dict[str, Any]] = None) -> MythRLResult:
        context = dict(context or {})
        operation_id = f"weave_{int(time.time() * 1000)}"
        trace = ProvenceTrace(
            operation_id=operation_id,
            complexity_level=ComplexityLevel.LITE,
            start_time=time.perf_counter(),
        )
        trace.add_shuttle_event("weave_start", "Shuttle weaving initiated", {"query": query})

        complexity = await self._assess_complexity_needed(query, context, trace)
        trace.complexity_level = complexity

        features = await self._run_feature_extraction(query, complexity, trace)
        crawl_payload = await self._run_memory_crawl(query, complexity, trace)
        pattern_summary = await self._run_pattern_selection(query, context, complexity, trace)
        decision_summary = await self._run_decision_engine(features, context, complexity, trace)
        manifold = await self._run_warp_space(features, complexity, trace)
        tool_result = await self._run_tool_execution(query, context, features, trace)

        confidence = self._blend_confidence([
            features.get("confidence"),
            crawl_payload.get("fusion_score"),
            decision_summary.get("confidence"),
            tool_result.get("confidence"),
        ])

        summary = {
            "features": features,
            "memory_results": crawl_payload.get("results", []),
            "pattern": pattern_summary,
            "decision": decision_summary,
            "warp_manifold": manifold,
            "tool_result": tool_result,
        }

        spacetime = await self._compute_spacetime(summary, trace)

        trace.add_shuttle_event(
            "weave_complete",
            "Shuttle weaving completed",
            {
                "confidence": confidence,
                "passes": crawl_payload.get("stats", _CrawlStats(0, 0, {}, 0)).passes,
                "modules": trace.modules_invoked,
            },
        )

        return MythRLResult(
            query=query,
            output=summary,
            confidence=confidence,
            complexity_level=complexity,
            provenance=trace,
            spacetime_coordinates=spacetime,
        )

    # ------------------------------------------------------------------
    # Complexity helpers
    # ------------------------------------------------------------------
    async def _assess_complexity_needed(
        self,
        query: str,
        context: Dict[str, Any],
        trace: ProvenceTrace,
    ) -> ComplexityLevel:
        lowered = query.lower()
        score = 0.0
        if any(term in lowered for term in ("research", "analysis", "relationship", "optimization")):
            score += 0.6
        if len(lowered.split()) > 18:
            score += 0.4
        score += min(0.4, context.get("complexity_bias", 0.0))

        level = ComplexityLevel.LITE
        if score >= 1.2:
            level = ComplexityLevel.RESEARCH
        elif score >= 0.8:
            level = ComplexityLevel.FULL
        elif score >= 0.4:
            level = ComplexityLevel.FAST

        trace.add_shuttle_event(
            "complexity_assessed",
            f"Complexity set to {level.name}",
            {"score": score},
        )
        return level

    def _embedding_scales_for(self, level: ComplexityLevel) -> List[int]:
        if level is ComplexityLevel.RESEARCH:
            return [96, 192, 384, 768]
        if level is ComplexityLevel.FULL:
            return [96, 192, 384]
        if level is ComplexityLevel.FAST:
            return [96, 192]
        return [96]

    # ------------------------------------------------------------------
    # Protocol invocations
    # ------------------------------------------------------------------
    async def _run_feature_extraction(
        self,
        query: str,
        level: ComplexityLevel,
        trace: ProvenceTrace,
    ) -> Dict[str, Any]:
        if not self.feature_extraction:
            return {}
        scales = self._embedding_scales_for(level)
        start = time.perf_counter()
        features = await self.feature_extraction.extract_features(query, scales)
        duration = (time.perf_counter() - start) * 1000
        trace.add_protocol_call(
            "feature_extraction",
            "extract_features",
            duration,
            f"Extracted scales {scales}",
        )
        trace.modules_invoked.append("feature_extraction")
        return features or {}

    async def _run_memory_crawl(
        self,
        query: str,
        level: ComplexityLevel,
        trace: ProvenceTrace,
    ) -> Dict[str, Any]:
        backend = self.memory_backend
        if not backend:
            return {}

        config = self._get_crawl_config(level)
        results: List[Dict[str, Any]] = []
        depth_stats: Dict[int, int] = {}
        fusion_scores: List[float] = []

        for depth, threshold in enumerate(config["thresholds"], start=1):
            limit = config["initial_limit"] if depth == 1 else config["branch_limit"]
            start = time.perf_counter()
            if hasattr(backend, "retrieve_with_threshold"):
                batch = await backend.retrieve_with_threshold(query, threshold, limit=limit)
            elif hasattr(backend, "retrieve"):
                batch = await backend.retrieve(query, threshold)  # type: ignore[arg-type]
            else:
                batch = []
            duration = (time.perf_counter() - start) * 1000
            trace.add_protocol_call(
                "memory_backend",
                "retrieve_with_threshold",
                duration,
                f"depth={depth} threshold={threshold} items={len(batch)}",
            )
            trace.modules_invoked.append("memory_backend")
            depth_stats[depth] = len(batch)
            results.extend(batch)
            fusion_scores.extend(item.get("relevance", 0.5) for item in batch)

            if len(results) >= config["max_total_items"]:
                break

            if depth < config["max_depth"] and hasattr(backend, "get_related"):
                related: List[Dict[str, Any]] = []
                for item in batch[:limit]:
                    related_items = await backend.get_related(item["id"], limit=config["branch_limit"])
                    related.extend(related_items)
                results.extend(related)
                fusion_scores.extend(item.get("relevance", 0.5) for item in related)
                depth_stats[depth] += len(related)

        fusion_score = self._blend_confidence(fusion_scores) or config["importance_threshold"]
        stats = _CrawlStats(
            passes=len(config["thresholds"]),
            total_items=len(results),
            depth_stats=depth_stats,
            fusion_events=sum(depth_stats.values()),
        )
        trace.add_shuttle_event(
            "multipass_crawl_complete",
            "Completed multipass crawling",
            {
                "passes": stats.passes,
                "total_items": stats.total_items,
                "depth_stats": depth_stats,
                "fusion_events": stats.fusion_events,
            },
        )
        return {"results": results, "fusion_score": fusion_score, "stats": stats}

    async def _run_pattern_selection(
        self,
        query: str,
        context: Dict[str, Any],
        level: ComplexityLevel,
        trace: ProvenceTrace,
    ) -> Dict[str, Any]:
        pattern = {}
        if not self.pattern_selection or level is ComplexityLevel.LITE:
            return pattern
        start = time.perf_counter()
        pattern = await self.pattern_selection.select_pattern(query, context, level)
        duration = (time.perf_counter() - start) * 1000
        trace.add_protocol_call(
            "pattern_selection",
            "select_pattern",
            duration,
            f"pattern={pattern.get('selected_pattern', 'unknown')}",
        )
        trace.modules_invoked.append("pattern_selection")
        return pattern or {}

    async def _run_decision_engine(
        self,
        features: Dict[str, Any],
        context: Dict[str, Any],
        level: ComplexityLevel,
        trace: ProvenceTrace,
    ) -> Dict[str, Any]:
        if not self.decision_engine or level.value < ComplexityLevel.FULL.value:
            return {}
        options = context.get(
            "decision_options",
            [
                {"tool": "search", "confidence": 0.7},
                {"tool": "analyze", "confidence": 0.8},
            ],
        )
        start = time.perf_counter()
        decision = await self.decision_engine.make_decision(features, context, options)
        duration = (time.perf_counter() - start) * 1000
        trace.add_protocol_call(
            "decision_engine",
            "make_decision",
            duration,
            f"selected={decision.get('selected_option', {}).get('tool', 'none')}",
        )
        trace.modules_invoked.append("decision_engine")
        return decision or {}

    async def _run_warp_space(
        self,
        features: Dict[str, Any],
        level: ComplexityLevel,
        trace: ProvenceTrace,
    ) -> Dict[str, Any]:
        if not self.warp_space:
            return {}
        start = time.perf_counter()
        manifold = await self.warp_space.create_manifold(features, level)
        duration = (time.perf_counter() - start) * 1000
        trace.add_protocol_call(
            "warp_space",
            "create_manifold",
            duration,
            f"manifold={manifold.get('manifold_type', 'unknown')}",
        )
        trace.modules_invoked.append("warp_space")
        return manifold or {}

    async def _run_tool_execution(
        self,
        query: str,
        context: Dict[str, Any],
        features: Dict[str, Any],
        trace: ProvenceTrace,
    ) -> Dict[str, Any]:
        if not self.tool_execution:
            return {}
        recommended = {}
        if hasattr(self.tool_execution, "assess_tool_necessity"):
            recommended = await self.tool_execution.assess_tool_necessity(query, context)
        tool_name = recommended.get("tool", "respond")
        params = {
            "query": query,
            "features": features,
            "recommended": recommended,
        }
        start = time.perf_counter()
        result = await self.tool_execution.execute_tool(tool_name, params, context)
        duration = (time.perf_counter() - start) * 1000
        trace.add_protocol_call(
            "tool_execution",
            "execute_tool",
            duration,
            f"tool={tool_name}",
        )
        trace.modules_invoked.append("tool_execution")
        return result or {}

    async def _compute_spacetime(self, summary: Dict[str, Any], trace: ProvenceTrace) -> Dict[str, Any]:
        loops = len(summary.get("memory_results", [])) or 1
        beam = len(trace.modules_invoked) or 1
        return {
            "x": math.log1p(loops),
            "y": math.sqrt(beam),
            "z": len(trace.protocol_calls),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _blend_confidence(self, values: Iterable[Optional[float]]) -> float:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return 0.5
        return float(sum(filtered) / len(filtered))

    def _get_crawl_config(self, level: ComplexityLevel) -> Dict[str, Any]:
        if level is ComplexityLevel.RESEARCH:
            return {
                "max_depth": 4,
                "thresholds": [0.5, 0.65, 0.8, 0.9],
                "initial_limit": 12,
                "branch_limit": 8,
                "max_total_items": 50,
                "importance_threshold": 0.6,
            }
        if level is ComplexityLevel.FULL:
            return {
                "max_depth": 3,
                "thresholds": [0.6, 0.75, 0.85],
                "initial_limit": 10,
                "branch_limit": 6,
                "max_total_items": 35,
                "importance_threshold": 0.58,
            }
        if level is ComplexityLevel.FAST:
            return {
                "max_depth": 2,
                "thresholds": [0.6, 0.75],
                "initial_limit": 8,
                "branch_limit": 4,
                "max_total_items": 20,
                "importance_threshold": 0.55,
            }
        return {
            "max_depth": 1,
            "thresholds": [0.7],
            "initial_limit": 6,
            "branch_limit": 3,
            "max_total_items": 10,
            "importance_threshold": 0.5,
        }


# ----------------------------------------------------------------------
# Demo protocol implementations
# ----------------------------------------------------------------------


class DemoMemoryBackend:
    """In-memory backend with simple graph traversal for demos."""

    def __init__(self) -> None:
        self.knowledge_graph: Dict[str, Dict[str, Any]] = self._build_seed_graph()

    async def retrieve_with_threshold(self, query: str, threshold: float, limit: int = 10) -> List[Dict[str, Any]]:
        await asyncio.sleep(0)
        hits = [item for item in self.knowledge_graph.values() if item["relevance"] >= threshold]
        return hits[:limit]

    async def get_related(self, item_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        await asyncio.sleep(0)
        node = self.knowledge_graph.get(item_id)
        if not node:
            return []
        related = [self.knowledge_graph[nid] for nid in node.get("related", []) if nid in self.knowledge_graph]
        return related[:limit]

    async def health_check(self) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"status": "ok", "items": len(self.knowledge_graph)}

    def _build_seed_graph(self) -> Dict[str, Dict[str, Any]]:
        graph: Dict[str, Dict[str, Any]] = {}
        data = {
            "bee_1": {
                "content": "Bee colony health indicators",
                "relevance": 0.82,
                "related": ["hive_2", "climate_1"],
            },
            "hive_2": {
                "content": "Hive structure optimization",
                "relevance": 0.78,
                "related": ["bee_1", "agri_3"],
            },
            "climate_1": {
                "content": "Climate resilience strategies",
                "relevance": 0.74,
                "related": ["bee_1", "policy_1"],
            },
            "agri_3": {
                "content": "Agricultural impact analysis",
                "relevance": 0.69,
                "related": ["hive_2"],
            },
            "policy_1": {
                "content": "Policy levers for sustainability",
                "relevance": 0.66,
                "related": ["climate_1"],
            },
        }
        for item_id, payload in data.items():
            graph[item_id] = {
                "id": item_id,
                "content": payload["content"],
                "relevance": payload["relevance"],
                "related": payload["related"],
            }
        return graph


class DemoPatternSelection:
    async def select_pattern(self, query: str, context: Dict[str, Any], complexity: ComplexityLevel) -> Dict[str, Any]:
        await asyncio.sleep(0)
        pattern = "research" if complexity is ComplexityLevel.RESEARCH else "balanced"
        return {"selected_pattern": pattern, "confidence": 0.7}

    async def assess_pattern_necessity(self, query: str) -> float:
        await asyncio.sleep(0)
        return 0.6 if len(query) > 24 else 0.2

    async def synthesize_patterns(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"selected_pattern": primary.get("selected_pattern", "primary"), "fused": True}


class DemoDecisionEngine:
    async def make_decision(self, features: Dict[str, Any], context: Dict[str, Any], options: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        selected = max(options, key=lambda opt: opt.get("confidence", 0.5)) if options else {}
        return {"selected_option": selected, "confidence": selected.get("confidence", 0.6)}

    async def assess_decision_complexity(self, features: Dict[str, Any]) -> float:
        await asyncio.sleep(0)
        return 0.5

    async def optimize_multi_criteria(self, criteria: Sequence[Dict[str, Any]], constraints: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"criteria": list(criteria), "constraints": constraints, "confidence": 0.6}


class DemoFeatureExtraction:
    async def extract_features(self, data: Any, scales: Sequence[int]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        embeddings = {scale: [0.1 * scale, 0.2 * scale] for scale in scales}
        return {"embeddings": embeddings, "confidence": 0.65, "dominant_pattern": "base"}

    async def extract_motifs(self, data: Any, complexity: ComplexityLevel) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"motifs": ["rhythm", "structure"], "confidence": 0.6}

    async def assess_extraction_needs(self, data: Any) -> List[int]:
        await asyncio.sleep(0)
        return [96, 192]


class DemoWarpSpace:
    async def create_manifold(self, features: Dict[str, Any], complexity: ComplexityLevel) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"manifold_type": "demo", "dimensions": 3, "tensor_shape": (3, 3), "metadata": {"complexity": complexity.name}}

    async def tension_threads(self, manifold: Dict[str, Any], threads: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"tensioned": len(threads)}

    async def compute_trajectories(self, manifold: Dict[str, Any], start_points: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"trajectories": len(start_points)}

    async def experimental_operations(self, manifold: Dict[str, Any], experiments: Sequence[str]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        return {"experiments": list(experiments), "confidence": 0.55}


class DemoToolExecution:
    def __init__(self) -> None:
        self._tools = {
            "respond": "Generate textual response",
            "search": "Perform knowledge search",
        }

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        description = self._tools.get(tool_name, "Unknown tool")
        return {
            "tool": tool_name,
            "description": description,
            "output": f"{description}: {parameters.get('query', '')}",
            "confidence": 0.7,
        }

    async def list_available_tools(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        await asyncio.sleep(0)
        return [{"name": name, "description": desc} for name, desc in self._tools.items()]

    async def assess_tool_necessity(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        tool = "search" if "search" in query.lower() else "respond"
        return {"tool": tool, "confidence": 0.6}
