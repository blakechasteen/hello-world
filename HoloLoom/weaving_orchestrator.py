#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaving Orchestrator - Complete Weaving Cycle
==============================================
The unified orchestrator that wires all 6 weaving modules together.

This is the REALIZATION of the weaving metaphor described in CLAUDE.md.
All independent "warp thread" modules are woven together into a complete cycle.

Complete Weaving Cycle:
1. LoomCommand → Selects Pattern Card (BARE/FAST/FUSED)
2. ChronoTrigger → Fires temporal window
3. ResonanceShed → Lifts feature threads, creates DotPlasma
4. WarpSpace → Tensions threads into continuous manifold
5. ConvergenceEngine → Collapses to discrete decision
6. Spacetime → Woven fabric with complete trace

Philosophy:
The WeavingOrchestrator is the "shuttle" that moves across all warp threads,
weaving them into finished "fabric" (responses with full computational provenance).

Usage:
    weaver = WeavingOrchestrator(config=Config.fused())
    result = await weaver.weave(query="What is HoloLoom?")
    trace = result.spacetime.weaving_trace
"""

import sys
import os
# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import all 6 weaving modules
try:
    from HoloLoom.loom.command import LoomCommand, PatternCard
    from HoloLoom.chrono.trigger import ChronoTrigger
    from HoloLoom.resonance.shed import ResonanceShed
    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.convergence.engine import ConvergenceEngine, CollapseStrategy
    from HoloLoom.convergence.mcts_engine import MCTSConvergenceEngine  # MCTS FLUX CAPACITOR
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace

    # Supporting modules
    from HoloLoom.config import Config
    from HoloLoom.Documentation.types import Query
    from HoloLoom.motif.base import create_motif_detector
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings, SpectralFusion
    from HoloLoom.memory.cache import MemoryManager
    from HoloLoom.synthesis_bridge import SynthesisBridge

except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you run from repository root with PYTHONPATH set")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Weaving Orchestrator
# ============================================================================

class WeavingOrchestrator:
    """
    Complete weaving orchestrator that coordinates all 6 modules.

    This class brings together the entire weaving architecture into a
    unified processing pipeline with full computational provenance.

    The weaving cycle:
    1. LoomCommand selects pattern based on query
    2. ChronoTrigger fires, creates temporal window
    3. ResonanceShed lifts feature threads (motif, embedding, spectral)
    4. WarpSpace tensions threads into continuous manifold
    5. ConvergenceEngine collapses to discrete tool decision
    6. Spacetime captures complete trace

    Attributes:
        loom: LoomCommand for pattern selection
        chrono: ChronoTrigger for temporal control
        shed: ResonanceShed for feature extraction
        warp: WarpSpace for tensioned computation
        convergence: ConvergenceEngine for decision collapse
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        default_pattern: str = "fast",
        collapse_strategy: CollapseStrategy = CollapseStrategy.EPSILON_GREEDY,
        use_mcts: bool = True,
        mcts_simulations: int = 100
    ):
        """
        Initialize Weaving Orchestrator.

        Args:
            config: HoloLoom config (defaults to Config.fast())
            default_pattern: Default pattern card ("bare", "fast", "fused")
            collapse_strategy: Strategy for decision collapse (if not using MCTS)
            use_mcts: Use MCTS Flux Capacitor for decisions (TS all the way down!)
            mcts_simulations: Number of MCTS simulations per decision
        """
        self.config = config or Config.fast()
        self.use_mcts = use_mcts

        # Initialize embeddings and spectral fusion
        self.embedder = MatryoshkaEmbeddings()
        self.spectral_fusion = SpectralFusion()

        # Initialize motif detector
        self.motif_detector = create_motif_detector(mode="hybrid")

        # Initialize memory - ACTUAL WORKING MEMORY!
        self._init_memory()

        # ====================================================================
        # Initialize all 6 weaving modules
        # ====================================================================

        # 1. Loom Command - Pattern selection
        self.loom = LoomCommand(
            default_pattern=PatternCard(default_pattern),
            auto_select=True
        )

        # 2. Chrono Trigger - Temporal control
        self.chrono = ChronoTrigger(
            config=self.config,
            enable_heartbeat=False  # Disable for now
        )

        # 3. Resonance Shed - Feature extraction
        self.shed = ResonanceShed(
            motif_detector=self.motif_detector,
            embedder=self.embedder,
            spectral_fusion=self.spectral_fusion,
            interference_mode="weighted_sum"
        )

        # 4. Warp Space - Tensioned manifold
        self.warp = WarpSpace(
            embedder=self.embedder,
            scales=self.config.scales,
            spectral_fusion=self.spectral_fusion
        )

        # 5. Convergence Engine - Decision collapse with MCTS FLUX CAPACITOR!
        if use_mcts:
            logger.info("Using MCTS FLUX CAPACITOR for decision-making!")
            self.convergence = MCTSConvergenceEngine(
                tools=self._get_available_tools(),
                n_simulations=mcts_simulations,
                exploration_constant=1.414  # sqrt(2) for balanced exploration
            )
        else:
            self.convergence = ConvergenceEngine(
                tools=self._get_available_tools(),
                default_strategy=collapse_strategy,
                epsilon=0.1
            )

        # 6. Synthesis Bridge - Pattern extraction and enrichment
        self.synthesis = SynthesisBridge(
            enable_enrichment=True,
            enable_pattern_extraction=True,
            min_pattern_confidence=0.4
        )

        # Weaving statistics
        self.weaving_count = 0
        self.pattern_usage = {"bare": 0, "fast": 0, "fused": 0}

        logger.info("WeavingOrchestrator initialized")
        logger.info(f"  Pattern: {default_pattern}")
        logger.info(f"  Collapse: {collapse_strategy.value}")
        logger.info(f"  Scales: {self.config.scales}")

    def _init_memory(self, backend: str = "hybrid", data_dir: str = "./memory_data"):
        """
        Initialize memory with hybrid store + fallback.

        Args:
            backend: "hybrid" (Qdrant+Neo4j+file), "file" (file only)
            data_dir: Directory for file persistence
        """
        self.memory_backend = backend
        self.memory_store = None

        if backend == "hybrid":
            # Try hybrid store (Qdrant + Neo4j + Mem0)
            try:
                from HoloLoom.memory.stores.hybrid_store import HybridMemoryStore, BackendConfig
                from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore
                from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore
                from HoloLoom.memory.stores.file_store import FileMemoryStore

                logger.info("Attempting to initialize hybrid memory store...")

                # Create backends (all with file fallback)
                backends = []

                # Try Qdrant
                try:
                    qdrant = QdrantMemoryStore(embedder=self.embedder)
                    backends.append(BackendConfig(qdrant, weight=0.4, name="qdrant"))
                    logger.info("  ✓ Qdrant backend available")
                except Exception as e:
                    logger.warning(f"  ✗ Qdrant unavailable: {e}")

                # Try Neo4j
                try:
                    neo4j = Neo4jMemoryStore()
                    backends.append(BackendConfig(neo4j, weight=0.3, name="neo4j"))
                    logger.info("  ✓ Neo4j backend available")
                except Exception as e:
                    logger.warning(f"  ✗ Neo4j unavailable: {e}")

                # Always include file backend (guaranteed to work)
                file_store = FileMemoryStore(data_dir=data_dir, embedder=self.embedder)
                backends.append(BackendConfig(file_store, weight=0.3, name="file"))
                logger.info("  ✓ File backend available")

                if len(backends) > 1:
                    # Use hybrid fusion
                    self.memory_store = HybridMemoryStore(
                        backends=backends,
                        fusion_method="weighted"
                    )
                    logger.info(f"Hybrid memory initialized with {len(backends)} backends")
                else:
                    # Only file backend available
                    self.memory_store = file_store
                    logger.info("Using file-only memory (other backends unavailable)")

            except ImportError as e:
                logger.warning(f"Hybrid store unavailable: {e}, falling back to file")
                backend = "file"

        if backend == "file" or self.memory_store is None:
            # File-only fallback
            try:
                from HoloLoom.memory.stores.file_store import FileMemoryStore
                self.memory_store = FileMemoryStore(data_dir=data_dir, embedder=self.embedder)
                logger.info(f"File memory initialized: {data_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize file store: {e}")
                # Ultimate fallback: in-memory list
                self.memory_store = []
                logger.warning("Using in-memory fallback (no persistence)")

    async def add_knowledge(self, text: str, metadata: Optional[Dict] = None):
        """Add knowledge to memory (async)."""
        from HoloLoom.memory.protocol import Memory

        # Create Memory object
        memory = Memory(
            id=f"mem_{datetime.now().timestamp()}",
            text=text,
            timestamp=datetime.now(),
            context={},
            metadata=metadata or {}
        )

        # Store based on backend type
        if isinstance(self.memory_store, list):
            # In-memory fallback (legacy)
            from HoloLoom.Documentation.types import MemoryShard
            shard = MemoryShard(
                id=memory.id,
                text=text,
                episode=metadata.get("episode", "default") if metadata else "default",
                entities=metadata.get("entities", []) if metadata else [],
                motifs=metadata.get("motifs", []) if metadata else [],
                metadata=metadata or {}
            )
            self.memory_store.append(shard)
            logger.info(f"Added knowledge (in-memory): {text[:50]}...")
        else:
            # Protocol-based store (async)
            await self.memory_store.store(memory)
            logger.info(f"Added knowledge ({self.memory_backend}): {text[:50]}...")

    async def _retrieve_context(self, query: str, limit: int = 5) -> List:
        """Retrieve relevant context from memory (async)."""
        if isinstance(self.memory_store, list):
            # Legacy in-memory retrieval
            if not self.memory_store:
                return []

            # Encode query and all memories
            query_embed = self.embedder.encode([query])[0]
            memory_texts = [shard.text for shard in self.memory_store]
            memory_embeds = self.embedder.encode(memory_texts)

            # Compute similarities
            query_norm = query_embed / (np.linalg.norm(query_embed) + 1e-8)
            mem_norm = memory_embeds / (np.linalg.norm(memory_embeds, axis=1, keepdims=True) + 1e-8)
            similarities = mem_norm @ query_norm

            # Get top-K
            top_k = min(limit, len(self.memory_store))
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = [self.memory_store[i] for i in top_indices]
            logger.info(f"Retrieved {len(results)} context shards (scores: {[f'{similarities[i]:.2f}' for i in top_indices]})")

            return results
        else:
            # Protocol-based store (async)
            from HoloLoom.memory.protocol import MemoryQuery, Strategy
            from HoloLoom.Documentation.types import MemoryShard

            query_obj = MemoryQuery(text=query, limit=limit)
            result = await self.memory_store.retrieve(query_obj, strategy=Strategy.FUSED)

            # Convert Memory objects back to MemoryShards for compatibility
            shards = []
            for mem in result.memories:
                shard = MemoryShard(
                    id=mem.id,
                    text=mem.text,
                    episode=mem.context.get("episode", "default"),
                    entities=mem.context.get("entities", []),
                    motifs=mem.context.get("motifs", []),
                    metadata=mem.metadata
                )
                shards.append(shard)

            logger.info(
                f"Retrieved {len(shards)} context shards "
                f"(backend: {result.metadata.get('backend', self.memory_backend)}, "
                f"scores: {[f'{s:.2f}' for s in result.scores[:3]]})"
            )

            return shards

    def _get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return [
            "search",
            "summarize",
            "extract",
            "respond",
            "clarify"
        ]

    async def weave(
        self,
        query: str,
        user_pattern: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Spacetime:
        """
        Execute complete weaving cycle.

        This is the main API that coordinates all 6 modules into a
        complete processing pipeline with full trace.

        Args:
            query: User query text
            user_pattern: Optional explicit pattern ("bare", "fast", "fused")
            context: Optional additional context

        Returns:
            Spacetime fabric with result and complete trace
        """
        self.weaving_count += 1
        cycle_start = datetime.now()

        logger.info("="*80)
        logger.info(f"WEAVING CYCLE #{self.weaving_count}")
        logger.info("="*80)

        # Initialize trace
        trace = WeavingTrace(
            start_time=cycle_start,
            end_time=cycle_start,  # Updated at end
            duration_ms=0.0
        )

        try:
            # ================================================================
            # STAGE 1: Loom Command - Select Pattern Card
            # ================================================================
            logger.info("\n[STAGE 1] Loom Command - Pattern Selection")
            pattern_spec = self.loom.select_pattern(
                query_text=query,
                user_preference=user_pattern
            )

            pattern_name = pattern_spec.card.value
            self.pattern_usage[pattern_name] += 1

            trace.pattern_card = pattern_name
            trace.pattern_spec = {
                "scales": pattern_spec.scales,
                "quality_target": pattern_spec.quality_target,
                "speed_priority": pattern_spec.speed_priority,
                "timeout": pattern_spec.pipeline_timeout
            }

            logger.info(f"  Selected: {pattern_name.upper()}")
            logger.info(f"  Quality: {pattern_spec.quality_target:.1%}")
            logger.info(f"  Speed: {pattern_spec.speed_priority:.1%}")

            # ================================================================
            # STAGE 2: Chrono Trigger - Fire Temporal Window
            # ================================================================
            logger.info("\n[STAGE 2] Chrono Trigger - Temporal Activation")
            stage_start = datetime.now()

            window = await self.chrono.fire(
                query_time=datetime.now(),
                pattern_card_mode=pattern_name
            )

            trace.temporal_window = {
                "start": str(window.start),
                "end": str(window.end),
                "max_age": str(window.max_age),
                "recency_bias": window.recency_bias
            }

            logger.info(f"  Window: {window.start} → {window.end}")
            logger.info(f"  Recency bias: {window.recency_bias:.1%}")

            # ================================================================
            # STAGE 3: Resonance Shed - Feature Extraction
            # ================================================================
            logger.info("\n[STAGE 3] Resonance Shed - Feature Interference")
            stage_start = datetime.now()

            # Lift threads and create DotPlasma
            dot_plasma = await self.chrono.monitor(
                operation=lambda: self.shed.weave(
                    text=query,
                    thread_weights={
                        "motif": 1.0,
                        "embedding": 1.0,
                        "spectral": 0.5 if pattern_spec.enable_spectral else 0.0
                    }
                ),
                timeout=pattern_spec.stage_timeouts.get("features", 2.0),
                stage="features"
            )

            # Record in trace
            if isinstance(dot_plasma, dict) and "motifs" in dot_plasma:
                trace.motifs_detected = dot_plasma.get("motifs", [])
                trace.embedding_scales_used = pattern_spec.scales
                trace.dot_plasma_metadata = dot_plasma.get("metadata", {})

                logger.info(f"  Motifs: {len(trace.motifs_detected)}")
                logger.info(f"  Embedding scales: {trace.embedding_scales_used}")
                logger.info(f"  Threads: {dot_plasma.get('metadata', {}).get('thread_count', 0)}")

            # ================================================================
            # STAGE 3.5: Synthesis - Pattern Extraction & Enrichment
            # ================================================================
            logger.info("\n[STAGE 3.5] Synthesis - Pattern Extraction")
            stage_start = datetime.now()

            # Retrieve actual context from memory!
            context_shards = await self._retrieve_context(query, limit=5)

            # Run synthesis
            synthesis_result = await self.synthesis.synthesize(
                query_text=query,
                dot_plasma=dot_plasma if isinstance(dot_plasma, dict) else {},
                context_shards=context_shards,
                pattern_spec=pattern_spec
            )

            # Record in trace
            trace.synthesis_result = synthesis_result.to_trace_dict()

            logger.info(f"  Entities: {len(synthesis_result.key_entities)}")
            logger.info(f"  Patterns: {len(synthesis_result.patterns)}")
            logger.info(f"  Reasoning: {synthesis_result.reasoning_type}")
            logger.info(f"  Confidence: {synthesis_result.confidence:.2f}")

            # ================================================================
            # STAGE 4: Warp Space - Tension Threads
            # ================================================================
            logger.info("\n[STAGE 4] Warp Space - Thread Tensioning")
            stage_start = datetime.now()

            # context_shards already retrieved in Stage 3.5
            trace.context_shards_count = len(context_shards)
            trace.threads_activated = [s.text[:50] for s in context_shards[:5]]

            # Tension context into Warp Space
            if context_shards:
                context_texts = [shard.text for shard in context_shards]
                await self.warp.tension(
                    thread_texts=context_texts[:10],  # Limit for performance
                    tension_weights=[1.0] * min(10, len(context_texts))
                )

                logger.info(f"  Tensioned: {len(self.warp.threads)} threads")
                logger.info(f"  Context shards: {len(context_shards)}")
            else:
                logger.warning("  No context shards retrieved")

            # ================================================================
            # STAGE 5: Convergence Engine - Decision Collapse
            # ================================================================
            logger.info("\n[STAGE 5] Convergence Engine - Decision Collapse")
            stage_start = datetime.now()

            # Create features dict for policy
            features_dict = {
                "motifs": trace.motifs_detected,
                "psi": dot_plasma.get("psi", []) if isinstance(dot_plasma, dict) else [],
                "spectral": dot_plasma.get("spectral") if isinstance(dot_plasma, dict) else None
            }

            # Generate mock neural probabilities (TODO: replace with actual policy network)
            import numpy as np
            n_tools = len(self._get_available_tools())
            neural_probs = np.random.dirichlet(np.ones(n_tools))  # Random but valid distribution

            # Collapse to decision
            decision = self.convergence.collapse(neural_probs=neural_probs)

            trace.tool_selected = decision.tool
            trace.tool_confidence = decision.confidence
            trace.policy_adapter = decision.strategy_used

            logger.info(f"  Tool: {decision.tool}")
            logger.info(f"  Confidence: {decision.confidence:.1%}")
            logger.info(f"  Strategy: {decision.strategy_used}")

            # ================================================================
            # STAGE 6: Tool Execution
            # ================================================================
            logger.info("\n[STAGE 6] Tool Execution")
            stage_start = datetime.now()

            # Execute selected tool
            result = await self._execute_tool(
                tool=decision.tool,
                query=query,
                context=context_shards,
                features=features_dict
            )

            trace.execution_result = {
                "tool": decision.tool,
                "success": result.get("success", False),
                "output_length": len(str(result.get("output", "")))
            }

            logger.info(f"  Result: {result.get('success', False)}")
            logger.info(f"  Output: {len(str(result.get('output', '')))} chars")

            # ================================================================
            # Finalize Trace
            # ================================================================
            cycle_end = datetime.now()
            duration_ms = (cycle_end - cycle_start).total_seconds() * 1000

            trace.end_time = cycle_end
            trace.duration_ms = duration_ms

            # Chrono metrics
            chrono_metrics = self.chrono.record_completion()
            trace.chrono_metrics = chrono_metrics

            # ================================================================
            # Create Spacetime Fabric
            # ================================================================
            spacetime = Spacetime(
                query_text=query,
                response=result.get("output", ""),
                tool_used=decision.tool,
                confidence=decision.confidence,
                trace=trace
            )

            logger.info("\n" + "="*80)
            logger.info(f"WEAVING COMPLETE: {duration_ms:.0f}ms")
            logger.info(f"  Pattern: {pattern_name}")
            logger.info(f"  Tool: {decision.tool}")
            logger.info(f"  Confidence: {decision.confidence:.1%}")
            logger.info("="*80)

            return spacetime

        except Exception as e:
            logger.error(f"Weaving failed: {e}")
            import traceback
            traceback.print_exc()

            # Create error spacetime
            trace.end_time = datetime.now()
            trace.duration_ms = (trace.end_time - cycle_start).total_seconds() * 1000
            trace.errors.append({"error": str(e), "stage": "weaving"})

            return Spacetime(
                query_text=query,
                response=f"Error: {e}",
                tool_used="error",
                confidence=0.0,
                trace=trace
            )

    async def _execute_tool(
        self,
        tool: str,
        query: str,
        context: List,
        features: Dict
    ) -> Dict[str, Any]:
        """
        Execute selected tool.

        Args:
            tool: Tool name
            query: User query
            context: Context shards
            features: Extracted features

        Returns:
            Execution result dict
        """
        # Simple tool execution (expand as needed)
        if tool == "search":
            output = f"Search results for: {query}"
            if context:
                output += f"\n\nFound {len(context)} relevant results"

        elif tool == "summarize":
            output = f"Summary: {query}\n\nBased on {len(context)} context shards"

        elif tool == "extract":
            motifs = features.get("motifs", [])
            output = f"Extracted {len(motifs)} key patterns: {motifs}"

        elif tool == "respond":
            output = f"Response to: {query}\n\nContext: {len(context)} shards"

        elif tool == "clarify":
            output = f"Could you clarify: {query}?"

        else:
            output = f"Tool '{tool}' executed for: {query}"

        return {
            "success": True,
            "tool": tool,
            "output": output,
            "query": query
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get weaving statistics.

        Returns:
            Dict with statistics from all modules
        """
        stats = {
            "total_weavings": self.weaving_count,
            "pattern_usage": self.pattern_usage,
            "loom_stats": self.loom.get_statistics(),
            "chrono_stats": self.chrono.get_evolution_stats(),
        }

        # Add convergence stats (different for MCTS vs regular)
        if self.use_mcts:
            stats["mcts_stats"] = self.convergence.get_statistics()
        else:
            stats["bandit_stats"] = self.convergence.bandit.get_statistics()

        return stats

    def stop(self):
        """Stop all background processes."""
        self.chrono.stop()
        logger.info("WeavingOrchestrator stopped")


# ============================================================================
# Factory Functions
# ============================================================================

def create_weaving_orchestrator(
    pattern: str = "fast",
    strategy: str = "epsilon_greedy"
) -> WeavingOrchestrator:
    """
    Create WeavingOrchestrator with defaults.

    Args:
        pattern: Default pattern ("bare", "fast", "fused")
        strategy: Collapse strategy

    Returns:
        Configured WeavingOrchestrator
    """
    strategy_enum = CollapseStrategy(strategy)
    return WeavingOrchestrator(
        default_pattern=pattern,
        collapse_strategy=strategy_enum
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("WEAVING ORCHESTRATOR WITH MCTS FLUX CAPACITOR")
        print("="*80)
        print("\nInitializing complete weaving architecture...")
        print("Thompson Sampling ALL THE WAY DOWN with MCTS search!")
        print()

        # Create orchestrator with MCTS FLUX CAPACITOR
        weaver = WeavingOrchestrator(
            config=Config.fast(),
            default_pattern="fast",
            use_mcts=True,
            mcts_simulations=50  # 50 simulations per decision
        )

        # Test queries
        queries = [
            "What is HoloLoom?",
            "Explain the weaving metaphor",
            "How does Thompson Sampling work?"
        ]

        for idx, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"QUERY {idx}: {query}")
            print('='*80)

            # Execute weaving cycle
            spacetime = await weaver.weave(query)

            # Show result
            print(f"\nResult:")
            print(f"  Tool: {spacetime.tool_used}")
            print(f"  Output: {spacetime.response[:200]}...")

            # Show trace
            print(f"\nWeaving Trace:")
            print(f"  Duration: {spacetime.trace.duration_ms:.0f}ms")
            print(f"  Pattern: {spacetime.trace.pattern_card}")
            print(f"  Motifs: {len(spacetime.trace.motifs_detected)}")
            print(f"  Context shards: {spacetime.trace.context_shards_count}")
            print(f"  Confidence: {spacetime.trace.tool_confidence:.1%}")

        # Show statistics
        print(f"\n{'='*80}")
        print("STATISTICS")
        print('='*80)
        stats = weaver.get_statistics()
        print(f"  Total weavings: {stats['total_weavings']}")
        print(f"  Pattern usage: {stats['pattern_usage']}")

        if 'mcts_stats' in stats:
            print(f"\n  MCTS FLUX CAPACITOR:")
            mcts = stats['mcts_stats']
            print(f"    Total simulations: {mcts['flux_stats']['total_simulations']}")
            print(f"    Decisions made: {mcts['decision_count']}")
            print(f"    Tool distribution: {mcts['flux_stats']['tool_distribution']}")
            print(f"    Thompson priors: {[f'{p:.3f}' for p in mcts['flux_stats']['thompson_priors']]}")

        # Stop
        weaver.stop()

        print("\nFlux Capacitor operational! Thompson Sampling ALL THE WAY DOWN!")

    asyncio.run(demo())
