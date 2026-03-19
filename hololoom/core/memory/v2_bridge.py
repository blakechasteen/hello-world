"""
v2 Cognitive Bridge — Wire the v2 classical AI pipeline into the MCP server.

Bridges UnifiedMemory (v1 persistence) to the v2 cognitive architecture:
  Navigator (PPR) + Blackboard + KnowledgeSources (TMS, CBR, ChunkCompiler, Demons)

Architecture:
  - UnifiedMemory remains the durable backend (Neo4j/Qdrant/InMemory)
  - LiteMemoryBus acts as the in-memory graph layer for PPR traversal
  - Dual-write: every store goes to both UnifiedMemory and LiteMemoryBus
  - v2 runs headless (no LLM) — Claude IS the generation layer

The bridge runs the retrieval + analysis pipeline, not the full orchestrator.
MCP tools return enriched results: memories + confidence + justifications + CBR hints.

No external dependencies beyond HoloLoom.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from .bus import MemoryItem
from .lite_bus import LiteMemoryBus
from .v2.activation import ActivationAdapter
from .v2.blackboard import (
    Blackboard,
    WorkingMemory,
    compute_ppr_entropy,
    extract_seed_sources,
)
from .v2.classical_factory import create_classical_sources
from .v2.confidence import ConfidenceEstimator
from .v2.formatter import FormatConfig, Formatter
from .v2.knowledge_source import Phase
from .v2.navigator import Navigator, PPRConfig

try:
    from .graph import extract_entities_simple
except ImportError:
    # Fallback: extract capitalized words
    import re
    def extract_entities_simple(text: str) -> list[str]:
        return list(set(re.findall(r'\b[A-Z][a-z]{2,}\b', text)))

logger = logging.getLogger(__name__)


# =============================================================================
# Pattern Card → PPR Config Mapping
# =============================================================================

PATTERN_PPR = {
    "bare": PPRConfig(
        alpha=0.25,          # High reset = stay close to seeds
        max_iterations=10,
        tolerance=1e-5,
        max_results=20,
        min_score=1e-3,
    ),
    "fast": PPRConfig(
        alpha=0.15,          # Default
        max_iterations=20,
        tolerance=1e-6,
        max_results=50,
    ),
    "fused": PPRConfig(
        alpha=0.10,          # Low reset = explore wider
        max_iterations=30,
        tolerance=1e-7,
        max_results=100,
        min_score=1e-5,
    ),
}


# =============================================================================
# Enhanced Recall Result
# =============================================================================

@dataclass
class EnhancedRecallResult:
    """Result from the v2 cognitive pipeline."""
    memories: list[dict[str, Any]]     # [{id, text, score, memory_type}, ...]
    confidence: float                   # Combined confidence [0, 1]
    decision: str                       # "sufficient" / "verify" / "flag"
    context_text: str                   # Token-budgeted formatted context
    justifications: list[dict]          # TMS justification records
    retractions: list[dict]             # TMS retractions
    cbr_hints: list[dict]              # Similar past case hints
    seed_count: int
    ppr_converged: bool
    node_count: int                     # Total graph nodes
    elapsed: float                      # Seconds


# =============================================================================
# Cognitive Bridge
# =============================================================================

class CognitiveBridge:
    """Bridge from UnifiedMemory to the v2 cognitive pipeline.

    Maintains a LiteMemoryBus as in-memory graph layer. Runs the headless
    v2 pipeline on every recall: Navigate (PPR) + Confidence + KnowledgeSources.

    Usage:
        bridge = CognitiveBridge(memory)
        await bridge.initialize()
        result = await bridge.enhanced_recall("Thompson Sampling vs epsilon-greedy")
    """

    def __init__(self, memory: Any, persist_path: str | None = None):
        self.memory = memory
        self.persist_path = persist_path  # None = in-memory only
        self.bus = LiteMemoryBus()
        self.navigator: Navigator | None = None
        self.confidence_estimator: ConfidenceEstimator | None = None

        # Persistent across queries
        self.activation = ActivationAdapter()
        self.working_memory = WorkingMemory()
        self.sources = create_classical_sources(activation_adapter=self.activation)

        self._initialized = False

    async def initialize(self, warm_up_limit: int = 500):
        """Initialize bus, navigator, and warm up from existing memories.

        If persist_path is set and a snapshot exists, loads combined state
        (graph + CBR + ChunkCompiler) and skips warm-up.
        """
        await self.bus.initialize()

        loaded = False
        if self.persist_path:
            try:
                loaded = self._load_combined_snapshot()
                if loaded:
                    logger.info("CognitiveBridge: loaded snapshot, skipping warm-up")
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning("CognitiveBridge: corrupt snapshot (%s), falling back to warm-up", e)
                loaded = False

        self.navigator = Navigator.from_lite_bus(self.bus)
        self.confidence_estimator = ConfidenceEstimator(
            graph=self.navigator.graph,
        )

        if not loaded:
            await self._warm_up(limit=warm_up_limit)

        if loaded:
            await self.bus.decay()

        self._initialized = True
        logger.info(
            "CognitiveBridge initialized: %d nodes, %d edges%s",
            len(self.bus._items),
            sum(len(e) for e in self.bus._edges.values()) // 2,
            " (from snapshot)" if loaded else "",
        )

    # =========================================================================
    # Warm-up: populate graph from existing memories
    # =========================================================================

    async def _warm_up(self, limit: int = 500):
        """Load recent memories from UnifiedMemory into the graph layer."""
        try:
            # UnifiedMemory.recall() may be sync or async — try both
            memories = self._recall_from_memory("", limit)
            count = 0
            for mem in memories:
                text = getattr(mem, 'text', str(mem))
                mem_id = getattr(mem, 'id', None)
                context = getattr(mem, 'context', {}) or {}
                await self._index_memory(text, mem_id, context)
                count += 1
            logger.info("CognitiveBridge: warmed up %d memories into graph", count)
        except Exception as e:
            logger.warning("CognitiveBridge warm-up failed (starting cold): %s", e)

    def _recall_from_memory(self, query: str, limit: int) -> list:
        """Call memory.recall() handling sync/async variants."""
        import asyncio

        # Try direct call (might be sync)
        try:
            result = self.memory.recall(query=query, limit=limit)
            # If it returned a coroutine, we need to await it
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
        except Exception:
            result = []

        # Result might be a list or have .memories attribute
        if hasattr(result, 'memories'):
            return result.memories
        if isinstance(result, list):
            return result
        return []

    # =========================================================================
    # Index: add a memory to the graph layer
    # =========================================================================

    async def _index_memory(
        self,
        text: str,
        mem_id: str | None = None,
        context: dict | None = None,
    ) -> str:
        """Index a single memory into LiteMemoryBus with entity edges."""
        context = context or {}
        importance = context.get('importance', 0.5)
        if isinstance(importance, str):
            try:
                importance = float(importance)
            except ValueError:
                importance = 0.5

        item = MemoryItem(
            content=text,
            memory_type="factual",
            importance=importance,
            entity_ids=[],
        )
        node_id = await self.bus.store(item)

        # Extract entities and create graph edges
        entities = extract_entities_simple(text)
        for entity in entities:
            entity_lower = entity.lower()

            # Create or find entity node
            if entity_lower in self.bus._aliases:
                eid = self.bus._aliases[entity_lower]
            else:
                entity_item = MemoryItem(
                    content=entity,
                    memory_type="entity",
                    importance=0.6,
                )
                eid = await self.bus.store(entity_item)
                # store() auto-aliases entity-type items

            # Link memory → entity
            await self.bus.store_edge(node_id, eid, "MENTIONS")

        return node_id

    async def store_and_index(
        self,
        text: str,
        context: dict = None,
        tags: list[str] = None,
        user_id: str = "",
    ):
        """Public: index a newly stored memory into the graph layer."""
        await self._index_memory(text, mem_id=None, context=context)
        self._save_if_persistent()

    def _save_if_persistent(self):
        """Save combined snapshot (graph + CBR + ChunkCompiler) if persist_path is set."""
        if self.persist_path:
            try:
                self._save_combined_snapshot()
            except Exception as e:
                logger.warning("Failed to save cognitive snapshot: %s", e)

    def _save_combined_snapshot(self):
        """Write graph + classical AI state to a single atomic JSON file."""
        from pathlib import Path

        combined = {
            "_version": 2,
            "_saved_at": datetime.utcnow().isoformat(),
            "graph": self.bus.to_snapshot_dict(),
        }

        # Persist CBR case library
        cbr = self._find_source("cbr")
        if cbr is not None:
            combined["cbr"] = cbr.to_snapshot()

        # Persist ChunkCompiler chunks + experience
        cc = self._find_source("chunk_compiler")
        if cc is not None:
            combined["chunks"] = cc.to_snapshot()

        p = Path(self.persist_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(combined, f, indent=2)
        tmp.replace(p)

    def _load_combined_snapshot(self) -> bool:
        """Load combined snapshot. Returns True if loaded."""
        from pathlib import Path

        p = Path(self.persist_path)
        if not p.exists():
            return False

        with open(p) as f:
            combined = json.load(f)

        version = combined.get("_version", 0)

        # v1 snapshots are graph-only (from the bus directly)
        if version <= 1:
            self.bus.from_snapshot_dict(combined)
            logger.info("CognitiveBridge: loaded v1 graph-only snapshot")
            return True

        # v2+ combined snapshots
        if "graph" in combined:
            self.bus.from_snapshot_dict(combined["graph"])

        if "cbr" in combined:
            cbr = self._find_source("cbr")
            if cbr is not None:
                cbr.from_snapshot(combined["cbr"])

        if "chunks" in combined:
            cc = self._find_source("chunk_compiler")
            if cc is not None:
                cc.from_snapshot(combined["chunks"])

        logger.info(
            "CognitiveBridge: loaded v%d combined snapshot (%d nodes, %d edges)",
            version, len(self.bus._items),
            sum(len(e) for e in self.bus._edges.values()) // 2,
        )
        return True

    def _find_source(self, name: str):
        """Find a KnowledgeSource by name in the registry."""
        for src in self.sources._sources:
            if src.name == name:
                return src
        return None

    # =========================================================================
    # Enhanced Recall: headless v2 pipeline
    # =========================================================================

    async def enhanced_recall(
        self,
        query: str,
        limit: int = 10,
        pattern: str = "fast",
    ) -> EnhancedRecallResult:
        """Run the headless v2 pipeline: Navigate + Confidence + KnowledgeSources.

        Args:
            query: Search query
            limit: Max results to return
            pattern: Pattern card ("bare", "fast", "fused") — controls PPR traversal depth

        Returns enriched results with confidence scores, TMS justifications,
        CBR hints, and formatted context.
        """
        if not self._initialized:
            raise RuntimeError("CognitiveBridge not initialized — call initialize() first")

        t0 = time.perf_counter()

        # 1. Blackboard + working memory
        bb = Blackboard(query=query, working_memory=self.working_memory)
        self.working_memory.update_turn()
        self.activation.decay_step()

        # 2. Navigate: seed extraction + PPR (pattern card selects PPR config)
        ppr_override = PATTERN_PPR.get(pattern.lower())
        nav_result = await self.navigator.navigate(
            query=query,
            activation_adapter=self.activation,
            override_config=ppr_override,
        )

        # 3. Post navigation to blackboard
        ppr_scores = [score for _, score in nav_result.ranked_nodes]
        seed_sources = extract_seed_sources(nav_result.seed_nodes)
        entity_ids = {nid for nid, _ in nav_result.ranked_nodes[:20]}
        bb.post_navigation(
            seed_count=len(nav_result.seed_nodes),
            seed_sources=seed_sources,
            ppr_entropy=compute_ppr_entropy(ppr_scores),
            ppr_converged=nav_result.converged,
            entity_ids=entity_ids,
        )

        # 4. KS: POST_NAVIGATE (TMS justifications, CBR case retrieval)
        self.sources.invoke(
            Phase.POST_NAVIGATE, bb, self.navigator.graph,
            {"nav_result": nav_result},
        )

        # 5. Confidence estimation
        confidence = self.confidence_estimator.estimate(nav_result, query=query)
        bb.post_confidence(
            combined=confidence.combined,
            structural=confidence.structural.score,
            narrative=confidence.narrative.score,
            decision=confidence.decision,
        )

        # 6. KS: POST_CONFIDENCE (TMS retraction propagation)
        self.sources.invoke(
            Phase.POST_CONFIDENCE, bb, self.navigator.graph,
            {"confidence": confidence, "nav_result": nav_result},
        )

        # 7. Format context block
        format_config = FormatConfig(token_budget=2048)
        formatter = Formatter(config=format_config)
        context_block = formatter.pack(
            ranked_nodes=nav_result.ranked_nodes,
            node_data=nav_result.node_data,
            confidence_scores=confidence.per_node,
        )

        # 8. KS: POST_LOOP (CBR retain, ChunkCompiler record)
        # Build synthetic LoopResult for KS consumption
        synthetic_loop = SimpleNamespace(
            response=context_block.text[:200],
            shells_executed=[],
            final_confidence=confidence,
            total_tokens=context_block.token_count,
            total_elapsed=time.perf_counter() - t0,
            blackboard=bb,
            shell_count=1,
        )
        self.sources.invoke(
            Phase.POST_LOOP, bb, self.navigator.graph,
            {"loop_result": synthetic_loop},
        )

        # 9. Assemble result
        memories = []
        for nid, score in nav_result.ranked_nodes[:limit]:
            data = nav_result.node_data.get(nid, {})
            memories.append({
                "id": nid,
                "text": data.get("content", data.get("text", "")),
                "score": round(score, 4),
                "memory_type": data.get("memory_type", "unknown"),
            })

        elapsed = time.perf_counter() - t0

        # Normalize justifications: TMS stores as dict {belief_id: [records]}
        raw_justifications = bb.flags.get("justifications", {})
        if isinstance(raw_justifications, dict):
            justification_list = []
            for belief_id, records in raw_justifications.items():
                for rec in records:
                    justification_list.append({
                        "belief_id": belief_id,
                        "seed_id": getattr(rec, "seed_id", ""),
                        "path_length": len(getattr(rec, "path", [])),
                        "ppr_score": getattr(rec, "ppr_score", 0.0),
                    })
        else:
            justification_list = raw_justifications if isinstance(raw_justifications, list) else []

        self._save_if_persistent()

        return EnhancedRecallResult(
            memories=memories,
            confidence=confidence.combined,
            decision=confidence.decision,
            context_text=context_block.text,
            justifications=justification_list,
            retractions=bb.flags.get("retractions", []),
            cbr_hints=bb.flags.get("cbr_hints", []),
            seed_count=len(nav_result.seed_nodes),
            ppr_converged=nav_result.converged,
            node_count=len(self.bus._items),
            elapsed=elapsed,
        )

    # =========================================================================
    # Diagnostic
    # =========================================================================

    async def feedback(self, query: str, reward: float):
        """Report feedback on a recall result for CBR/ChunkCompiler learning.

        Args:
            query: The original query
            reward: Quality score [0.0, 1.0] — how useful the recall was
        """
        if not self._initialized:
            return

        cbr = self._find_source("cbr")
        if cbr is not None:
            # Update the most recent case matching this query
            for case in reversed(cbr._cases):
                if case.query == query:
                    # Running average
                    n = case.retrieval_count + 1
                    case.avg_reuse_reward = (
                        (case.avg_reuse_reward * case.retrieval_count + reward) / n
                    )
                    case.retrieval_count = n
                    break

        self._save_if_persistent()

    def demon_report(self) -> dict[str, Any]:
        """Get the latest demon alerts report."""
        demons = self._find_source("demons")
        if demons is not None:
            return demons.report()
        return {"alerts": [], "total_alerts": 0}

    def report(self) -> dict[str, Any]:
        """Diagnostic info about bridge state."""
        return {
            "initialized": self._initialized,
            "graph_nodes": len(self.bus._items),
            "graph_edges": sum(len(e) for e in self.bus._edges.values()) // 2,
            "aliases": len(self.bus._aliases),
            "working_memory_turns": self.working_memory.turn_count,
            "activation_levels": len(self.activation.levels),
            "classical_ai": {
                src.name: src.activation_level(None)
                for src in self.sources._sources
            } if hasattr(self.sources, '_sources') else {},
        }


# =============================================================================
# Formatting for MCP output
# =============================================================================

def format_enhanced_result(result: EnhancedRecallResult, limit: int = 10) -> str:
    """Format EnhancedRecallResult as text for MCP tool output."""
    if not result.memories:
        return "No memories found."

    lines = [
        f"Found {len(result.memories)} memories (v2 enhanced, "
        f"confidence={result.confidence:.2f} [{result.decision}]):",
        "",
    ]

    for i, mem in enumerate(result.memories[:limit], 1):
        lines.append(f"{i}. [{mem['score']:.3f}] {mem['text']}")
        lines.append(f"   Type: {mem['memory_type']} | ID: {mem['id']}")
        lines.append("")

    # Cognitive analysis section
    lines.append("--- v2 Cognitive Analysis ---")
    lines.append(f"Confidence: {result.confidence:.2f} ({result.decision})")
    lines.append(f"Seeds: {result.seed_count} | PPR converged: {result.ppr_converged}")
    lines.append(f"Graph: {result.node_count} nodes")

    if result.justifications:
        lines.append(f"TMS: {len(result.justifications)} justifications recorded")
    if result.retractions:
        lines.append(f"TMS: {len(result.retractions)} retractions (contradictions resolved)")
    if result.cbr_hints:
        lines.append(f"CBR: {len(result.cbr_hints)} similar past cases found")
        for hint in result.cbr_hints[:2]:
            lines.append(f"  - [{hint.get('similarity', 0):.2f}] {hint.get('query', '')[:60]}")

    lines.append(f"Elapsed: {result.elapsed:.3f}s")

    return "\n".join(lines)
