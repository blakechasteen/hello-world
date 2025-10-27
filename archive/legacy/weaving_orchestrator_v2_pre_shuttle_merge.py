#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Weaving Orchestrator - Full 9-Step Weaving Cycle
===========================================================
The canonical orchestrator implementing complete weaving architecture.

MIGRATION NOTE (2025-10-27):
This file was promoted from weaving_shuttle.py to weaving_orchestrator.py
as part of Phase 1 consolidation. The old weaving_orchestrator.py (6-step)
is archived at archive/legacy/weaving_orchestrator_v1.py.

9-Step Weaving Cycle:
1. Loom Command → Pattern Card selection (BARE/FAST/FUSED)
2. Chrono Trigger → Temporal window creation
3. Yarn Graph → Thread selection from memory
4. Resonance Shed → Feature extraction, DotPlasma
5. Warp Space → Continuous manifold tensioning
6. Convergence Engine → Discrete decision collapse
7. Tool Execution → Action with results
8. Reflection Buffer → Learning from outcome
9. Chrono Detension → Cycle completion

Philosophy:
The true embodiment of the weaving metaphor - coordinates all architectural
components into an elegant dance of symbolic ↔ continuous transformations.

Author: HoloLoom Team (Blake + Claude)
Date: 2025-10-27 (promoted from shuttle)
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Shared types
from HoloLoom.Documentation.types import Query, Context, Features, MemoryShard

# Weaving architecture components
from HoloLoom.loom.command import LoomCommand, PatternCard, PatternSpec
from HoloLoom.chrono.trigger import ChronoTrigger, TemporalWindow, ExecutionLimits
from HoloLoom.resonance.shed import ResonanceShed
from HoloLoom.warp.space import WarpSpace
from HoloLoom.convergence.engine import ConvergenceEngine, CollapseStrategy, CollapseResult
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.reflection.buffer import ReflectionBuffer, LearningSignal

# Core modules
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.motif.base import create_motif_detector
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings, SpectralFusion
from HoloLoom.memory.base import create_retriever
from HoloLoom.policy.unified import create_policy

logging.basicConfig(level=logging.INFO)


# ============================================================================
# Tool Execution
# ============================================================================

class ToolExecutor:
    """
    Executes tools based on convergence engine decisions.

    In production, this would call actual APIs, databases, etc.
    """

    def __init__(self):
        self.tools = ["answer", "search", "notion_write", "calc"]
        self.logger = logging.getLogger(__name__)

    async def execute(self, tool: str, query: Query, context: Context) -> Dict:
        """
        Execute a tool based on the convergence decision.

        Args:
            tool: Tool name from CollapseResult
            query: Original query
            context: Retrieved context

        Returns:
            Dict with execution results
        """
        self.logger.info(f"Executing tool: {tool}")

        # Tool implementations (stubs - replace with real implementations)
        tool_handlers = {
            "answer": self._handle_answer,
            "search": self._handle_search,
            "notion_write": self._handle_notion_write,
            "calc": self._handle_calc
        }

        handler = tool_handlers.get(tool, self._handle_unknown)
        return await handler(query, context)

    async def _handle_answer(self, query: Query, context: Context) -> Dict:
        """Generate an answer based on context."""
        return {
            "tool": "answer",
            "result": f"Generated answer for: {query.text}",
            "confidence": 0.85,
            "sources": len(context.shards) if context and hasattr(context, 'shards') else 0
        }

    async def _handle_search(self, query: Query, context: Context) -> Dict:
        """Perform a search."""
        return {
            "tool": "search",
            "result": "Search results based on query",
            "sources": ["source1", "source2", "source3"],
            "count": 3
        }

    async def _handle_notion_write(self, query: Query, context: Context) -> Dict:
        """Write to Notion database."""
        return {
            "tool": "notion_write",
            "result": "Successfully wrote to Notion database",
            "status": "success",
            "page_id": "mock_page_123"
        }

    async def _handle_calc(self, query: Query, context: Context) -> Dict:
        """Perform calculation."""
        return {
            "tool": "calc",
            "result": "Calculation completed",
            "value": 42,
            "expression": "mock_calculation"
        }

    async def _handle_unknown(self, query: Query, context: Context) -> Dict:
        """Handle unknown tool."""
        return {
            "tool": "unknown",
            "result": "Unknown tool",
            "error": "Tool not implemented",
            "status": "error"
        }


# ============================================================================
# Yarn Graph (Simple Implementation)
# ============================================================================

class YarnGraph:
    """
    Simple in-memory Yarn Graph for thread storage.

    In production, this would be backed by Neo4j or NetworkX.
    For now, we use a simple dict-based implementation.
    """

    def __init__(self, shards: List[MemoryShard]):
        """Initialize with memory shards."""
        self.shards = {shard.id: shard for shard in shards}
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"YarnGraph initialized with {len(shards)} threads")

    def select_threads(self, temporal_window: TemporalWindow, query: Query) -> List[MemoryShard]:
        """
        Select threads based on temporal window.

        For now, returns all shards. In production, would filter by:
        - Temporal window bounds
        - Recency weighting
        - Episode filter
        - Query relevance

        Args:
            temporal_window: Time bounds for selection
            query: Query for relevance filtering

        Returns:
            List of relevant memory shards
        """
        # Simple implementation: return all threads
        threads = list(self.shards.values())
        self.logger.debug(f"Selected {len(threads)} threads from YarnGraph")
        return threads


# ============================================================================
# Weaving Orchestrator - Full 9-Step Cycle
# ============================================================================

class WeavingOrchestrator:
    """
    The canonical HoloLoom orchestrator implementing the full 9-step weaving cycle.

    Coordinates all architectural components:
    - Loom Command (pattern selection)
    - Chrono Trigger (temporal control)
    - Yarn Graph (thread storage)
    - Resonance Shed (feature interference)
    - Warp Space (tensor tensioning)
    - Convergence Engine (continuous → discrete)
    - Tool Execution (action)
    - Spacetime Fabric (provenance)
    - Reflection (learning)

    Usage:
        config = Config.fused()
        weaver = WeavingOrchestrator(cfg=config, shards=memory_shards)
        spacetime = await weaver.weave(Query(text="What is Thompson Sampling?"))
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        shards: Optional[List[MemoryShard]] = None,
        memory=None,  # Unified memory backend
        pattern_preference: Optional[PatternCard] = None,
        enable_reflection: bool = True,
        reflection_capacity: int = 1000,
        # Orchestrator modes (consolidated from smart/analytical variants)
        orchestrator_mode: str = "standard",  # standard|smart|analytical|simple
        enable_math_pipeline: bool = False,
        enable_analytical_verification: bool = False,
        # Backward compatibility
        config: Optional[Config] = None
    ):
        """
        Initialize the Weaving Orchestrator.

        Args:
            cfg: Configuration object (preferred)
            shards: List of memory shards (optional if memory is provided)
            memory: Unified memory backend (optional, overrides shards)
            pattern_preference: Optional pattern card preference (overrides config)
            enable_reflection: Enable reflection loop for learning
            reflection_capacity: Maximum episodes to store in reflection buffer
            orchestrator_mode: Mode of operation:
                - "standard": Normal 9-step weaving (default)
                - "smart": Adds smart math selection and synthesis
                - "analytical": Adds mathematical verification and analysis
                - "simple": Compatibility mode for legacy demos
            enable_math_pipeline: Enable math->meaning pipeline (smart mode)
            enable_analytical_verification: Enable analytical verification (analytical mode)
            config: Backward compatibility alias for cfg

        Note:
            Either shards OR memory must be provided. If memory is provided,
            it will be used for dynamic queries instead of static shards.
        """
        # Backward compatibility: accept both cfg and config
        if cfg is None and config is not None:
            cfg = config
        if cfg is None:
            cfg = Config.fast()  # Default
            
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Orchestrator mode configuration
        self.orchestrator_mode = orchestrator_mode
        self.enable_math_pipeline = enable_math_pipeline
        self.enable_analytical_verification = enable_analytical_verification
        
        # Auto-configure based on mode
        if orchestrator_mode == "smart":
            self.enable_math_pipeline = True
            self.logger.info("Smart mode enabled: math pipeline active")
        elif orchestrator_mode == "analytical":
            self.enable_analytical_verification = True
            self.logger.info("Analytical mode enabled: verification active")
        elif orchestrator_mode == "simple":
            # Simple mode: minimal features for compatibility
            enable_reflection = False
            self.logger.info("Simple mode enabled: compatibility mode")

        # Validate memory configuration
        if memory is None and shards is None:
            raise ValueError("Either 'shards' or 'memory' must be provided")

        self.memory = memory  # Backend memory store
        self.shards = shards or []  # Static shards (backward compatibility)

        # Determine pattern card from config or preference
        if pattern_preference:
            self.default_pattern = pattern_preference
        elif cfg.mode == ExecutionMode.BARE:
            self.default_pattern = PatternCard.BARE
        elif cfg.mode == ExecutionMode.FAST:
            self.default_pattern = PatternCard.FAST
        else:
            self.default_pattern = PatternCard.FUSED

        self.logger.info(f"Initializing WeavingOrchestrator (mode: {orchestrator_mode}) with pattern: {self.default_pattern.value}")

        # Lifecycle management
        self._background_tasks: List[asyncio.Task] = []
        self._closed = False

        # Initialize weaving components
        self._initialize_components()

        # Initialize reflection loop
        self.enable_reflection = enable_reflection
        if enable_reflection:
            self.reflection_buffer = ReflectionBuffer(
                capacity=reflection_capacity,
                persist_path="./reflections",
                learning_window=100
            )
            self.logger.info("Reflection loop enabled")
        else:
            self.reflection_buffer = None
            self.logger.info("Reflection loop disabled")

        self.logger.info("WeavingOrchestrator initialization complete")

    def _initialize_components(self):
        """Initialize all weaving architecture components."""

        # 1. Loom Command - Pattern selection
        self.loom_command = LoomCommand(
            default_pattern=self.default_pattern,
            auto_select=True
        )

        # 2. Yarn Graph / Memory Backend - Thread storage
        if self.memory:
            # Use persistent memory backend (UnifiedMemory, etc.)
            from HoloLoom.memory.weaving_adapter import WeavingMemoryAdapter
            if isinstance(self.memory, WeavingMemoryAdapter):
                self.yarn_graph = self.memory
            else:
                # Wrap raw memory in adapter (use "factory" type for protocol backends)
                self.yarn_graph = WeavingMemoryAdapter(backend=self.memory, backend_type="factory")
            self.logger.info("Using persistent memory backend")
        else:
            # Use in-memory YarnGraph with shards
            self.yarn_graph = YarnGraph(self.shards)
            self.logger.info(f"Using in-memory YarnGraph with {len(self.shards)} shards")

        # 3. Component factories (will be instantiated per-query with pattern spec)
        self.embedder = MatryoshkaEmbeddings(
            sizes=self.cfg.scales,
            base_model_name=self.cfg.base_model_name
        )

        # 4. Tool Executor
        self.tool_executor = ToolExecutor()

        # 5. Retriever (for context)
        # Only create traditional retriever if using shards
        if self.shards:
            self.retriever = create_retriever(
                shards=list(self.yarn_graph.shards.values()),
                emb=self.embedder,
                fusion_weights=self.cfg.fusion_weights
            )
        else:
            # Using dynamic memory backend - retriever not needed
            self.retriever = None

        self.logger.debug("All weaving components initialized")

    async def weave(
        self,
        query: Query,
        pattern_override: Optional[PatternCard] = None
    ) -> Spacetime:
        """
        Execute the complete 9-step weaving cycle.

        This is the main API - takes a query and returns a Spacetime artifact
        with complete computational lineage.

        Args:
            query: User query
            pattern_override: Optional pattern card override

        Returns:
            Spacetime fabric with response and full trace
        """
        start_time = datetime.now()
        stage_timings = {}
        errors = []
        warnings = []

        self.logger.info(f"[WEAVING] Beginning weaving cycle for query: '{query.text}'")

        try:
            # ================================================================
            # STEP 1: Loom Command selects Pattern Card
            # ================================================================
            step_start = time.time()

            pattern_spec = self.loom_command.select_pattern(
                query.text,
                user_preference=pattern_override.value if pattern_override else None
            )

            self.logger.info(f"  [1] Pattern selected: {pattern_spec.name}")
            stage_timings['pattern_selection'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 2: Chrono Trigger fires, creates TemporalWindow
            # ================================================================
            step_start = time.time()

            # Create a minimal config-like object for Chrono
            class ChronoConfig:
                def __init__(self, timeout):
                    self.pipeline_timeout = timeout

            chrono = ChronoTrigger(
                config=ChronoConfig(pattern_spec.pipeline_timeout),
                enable_heartbeat=False
            )

            temporal_window = TemporalWindow(
                start=datetime.now() - timedelta(days=365),  # Look back 1 year
                end=datetime.now(),
                max_age=timedelta(days=365),
                recency_bias=0.5
            )

            self.logger.info(f"  [2] Chrono Trigger fired")
            stage_timings['temporal_setup'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 3: Yarn Graph threads selected
            # ================================================================
            step_start = time.time()

            threads = self.yarn_graph.select_threads(temporal_window, query)
            thread_ids = [s.id for s in threads]

            self.logger.info(f"  [3] Selected {len(threads)} threads from Yarn Graph")
            stage_timings['thread_selection'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 4: Resonance Shed lifts feature threads, creates DotPlasma
            # ================================================================
            step_start = time.time()

            # Create components based on pattern spec
            motif_detector = create_motif_detector(mode=pattern_spec.motif_mode)
            spectral_fusion = SpectralFusion() if pattern_spec.enable_spectral else None

            # Create embedder with pattern-specific scales
            pattern_embedder = MatryoshkaEmbeddings(
                sizes=pattern_spec.scales,
                base_model_name=self.cfg.base_model_name
            )

            resonance_shed = ResonanceShed(
                motif_detector=motif_detector,
                embedder=pattern_embedder,
                spectral_fusion=spectral_fusion,
                interference_mode="weighted_sum"
            )

            # Extract features through Resonance Shed
            dot_plasma = await resonance_shed.weave(
                text=query.text,
                context_graph=None  # Could add KG here
            )

            thread_count = len(dot_plasma.get('threads', []))
            self.logger.info(f"  [4] DotPlasma created with {thread_count} feature threads")
            stage_timings['feature_extraction'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 5: Warp Space tensions threads into continuous manifold
            # ================================================================
            step_start = time.time()

            warp_space = WarpSpace(
                embedder=self.embedder,
                scales=pattern_spec.scales,
                spectral_fusion=spectral_fusion
            )

            # Tension threads from Yarn Graph
            await warp_space.tension(thread_ids, self.yarn_graph.shards)
            warp_operations = [(datetime.now().isoformat(), "tension", len(thread_ids))]

            self.logger.info(f"  [5] Warp Space tensioned with {len(thread_ids)} threads")
            stage_timings['warp_tensioning'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 6: Retrieve context (still needed for policy)
            # ================================================================
            step_start = time.time()

            # Retrieve context shards - either from static retriever or dynamic backend
            if self.retriever:
                # Traditional static shard retrieval
                hits = await self.retriever.search(
                    query=query.text,
                    k=pattern_spec.retrieval_k,
                    fast=(pattern_spec.retrieval_mode == "fast")
                )
                shards = [shard for shard, _ in hits]
                shard_texts = [shard.text for shard in shards]

            elif self.memory:
                # Dynamic backend query
                shards = await self._query_memory_backend(
                    query_text=query.text,
                    limit=pattern_spec.retrieval_k
                )
                shard_texts = [shard.text for shard in shards]
                # Create hits format for compatibility
                hits = [(shard, 1.0) for shard in shards]

            else:
                # No memory source available
                self.logger.warning("No memory source available (no shards or memory backend)")
                shards = []
                shard_texts = []
                hits = []

            context = Context(
                shards=shards,
                hits=hits,
                shard_texts=shard_texts,
                query=query,
                features=None  # Will be set from dot_plasma
            )

            self.logger.info(f"  [6] Retrieved {len(hits)} context shards")
            stage_timings['retrieval'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 7: Convergence Engine collapses to discrete tool selection
            # ================================================================
            step_start = time.time()

            # Create policy for neural predictions (use pattern_embedder)
            policy = create_policy(
                mem_dim=max(pattern_spec.scales),
                emb=pattern_embedder,
                scales=pattern_spec.scales,
                device=None,
                n_layers=pattern_spec.n_transformer_layers,
                n_heads=pattern_spec.n_attention_heads,
                bandit_strategy=self.cfg.bandit_strategy,
                epsilon=self.cfg.epsilon
            )

            # Convert dot_plasma to Features object for policy
            # Note: plasma uses 'psi' for embeddings and 'motifs' (plural)
            psi_array = dot_plasma.get('psi', [])
            psi_list = psi_array.tolist() if hasattr(psi_array, 'tolist') else list(psi_array)

            features = Features(
                psi=psi_list,
                motifs=dot_plasma.get('motifs', []),
                metrics={'spectral': dot_plasma.get('spectral')},
                metadata=dot_plasma.get('metadata', {})
            )
            context.features = features

            # Get neural predictions
            action_plan = await policy.decide(features=features, context=context)

            # Get tool probabilities (mock for now - would come from policy)
            import numpy as np
            neural_probs = np.array([
                action_plan.tool_probs.get(tool, 0.0)
                for tool in self.tool_executor.tools
            ])

            # Convergence Engine collapse
            convergence = ConvergenceEngine(
                tools=self.tool_executor.tools,
                default_strategy=self._map_bandit_to_collapse(self.cfg.bandit_strategy),
                epsilon=self.cfg.epsilon
            )

            collapse_result = convergence.collapse(neural_probs)

            self.logger.info(f"  [7] Convergence collapsed to tool: {collapse_result.tool} (confidence={collapse_result.confidence:.2f})")
            stage_timings['convergence'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 8: Tool executes
            # ================================================================
            step_start = time.time()

            # Execute the selected tool
            tool_result = await self.tool_executor.execute(
                collapse_result.tool,
                query,
                context
            )

            self.logger.info(f"  [8] Tool executed: {collapse_result.tool}")
            stage_timings['tool_execution'] = (time.time() - step_start) * 1000

            # ================================================================
            # STEP 9: Results woven into Spacetime fabric
            # ================================================================
            step_start = time.time()

            # Detension Warp Space
            warp_updates = warp_space.collapse()
            warp_operations.append((datetime.now().isoformat(), "detension", len(warp_updates)))

            # Create WeavingTrace
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            trace = WeavingTrace(
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                stage_durations=stage_timings,
                motifs_detected=[m.pattern if hasattr(m, 'pattern') else str(m) for m in features.motifs],
                embedding_scales_used=pattern_spec.scales,
                spectral_features=features.metrics.get('spectral'),
                threads_activated=thread_ids,
                context_shards_count=len(context.shards),
                retrieval_mode=pattern_spec.retrieval_mode,
                policy_adapter=action_plan.adapter,
                tool_selected=collapse_result.tool,
                tool_confidence=collapse_result.confidence,
                bandit_statistics=collapse_result.bandit_stats,
                warp_operations=warp_operations,
                tensor_field_stats={"threads_tensioned": len(thread_ids)},
                errors=errors,
                warnings=warnings
            )

            # Create Spacetime artifact
            spacetime = Spacetime(
                query_text=query.text,
                response=tool_result.get('result', 'No response'),
                tool_used=collapse_result.tool,
                confidence=collapse_result.confidence,
                trace=trace,
                metadata={
                    'pattern_card': pattern_spec.name,
                    'execution_mode': pattern_spec.card.value,
                    'loom_command': 'auto',
                    'chrono_timeout': pattern_spec.pipeline_timeout
                },
                context_summary=f"{len(context.shards)} shards",
                sources_used=[s.id for s in context.shards[:3]]
            )

            self.logger.info(f"  [9] Spacetime fabric woven!")
            stage_timings['spacetime_assembly'] = (time.time() - step_start) * 1000

            self.logger.info(f"[SUCCESS] Weaving cycle complete! Total duration: {duration_ms:.1f}ms")

            return spacetime

        except Exception as e:
            self.logger.error(f"[ERROR] Weaving cycle failed: {e}", exc_info=True)
            errors.append({
                'stage': 'unknown',
                'error': str(e),
                'type': type(e).__name__
            })

            # Return error Spacetime
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            trace = WeavingTrace(
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                stage_durations=stage_timings,
                errors=errors,
                warnings=warnings
            )

            return Spacetime(
                query_text=query.text,
                response=f"Error: {str(e)}",
                tool_used="error",
                confidence=0.0,
                trace=trace,
                metadata={'status': 'error', 'error_type': type(e).__name__}
            )

    def _map_bandit_to_collapse(self, bandit_strategy) -> CollapseStrategy:
        """Map Config BanditStrategy to Convergence CollapseStrategy."""
        from HoloLoom.config import BanditStrategy

        mapping = {
            BanditStrategy.EPSILON_GREEDY: CollapseStrategy.EPSILON_GREEDY,
            BanditStrategy.BAYESIAN_BLEND: CollapseStrategy.BAYESIAN_BLEND,
            BanditStrategy.PURE_THOMPSON: CollapseStrategy.PURE_THOMPSON
        }

        return mapping.get(bandit_strategy, CollapseStrategy.EPSILON_GREEDY)

    async def reflect(
        self,
        spacetime: Spacetime,
        feedback: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None
    ) -> None:
        """
        Store Spacetime in reflection buffer for learning.

        Call this after each weaving cycle to enable continuous improvement.

        Args:
            spacetime: Spacetime artifact from weaving
            feedback: Optional user feedback dict
            reward: Optional explicit reward (0-1)
        """
        if not self.enable_reflection:
            return

        await self.reflection_buffer.store(spacetime, feedback=feedback, reward=reward)
        self.logger.debug(f"Reflected on {spacetime.tool_used} (confidence={spacetime.confidence:.2f})")

    async def learn(self, force: bool = False) -> List[LearningSignal]:
        """
        Analyze reflection buffer and generate learning signals.

        Performs periodic analysis to identify improvement opportunities.

        Args:
            force: Force analysis even if not enough time has passed

        Returns:
            List of learning signals
        """
        if not self.enable_reflection:
            return []

        signals = await self.reflection_buffer.analyze_and_learn(force=force)

        if signals:
            self.logger.info(f"Generated {len(signals)} learning signals")

        return signals

    async def apply_learning_signals(self, signals: List[LearningSignal]) -> None:
        """
        Apply learning signals to adapt the system.

        Args:
            signals: Learning signals from reflection analysis
        """
        if not signals:
            return

        applied_count = 0

        for signal in signals:
            try:
                if signal.signal_type == "bandit_update":
                    # Update bandit statistics (future: integrate with policy)
                    self.logger.info(f"Bandit update for {signal.tool}: reward={signal.reward:.2f}")
                    applied_count += 1

                elif signal.signal_type == "pattern_preference":
                    # Adjust pattern card preference (future: dynamic adaptation)
                    self.logger.info(f"Pattern preference: {signal.pattern}")
                    applied_count += 1

                elif signal.signal_type == "threshold_adjustment":
                    # Adjust confidence thresholds (future: dynamic thresholds)
                    self.logger.info(f"Threshold adjustment recommended: {signal.recommendation}")
                    applied_count += 1

            except Exception as e:
                self.logger.warning(f"Failed to apply learning signal: {e}")

        self.logger.info(f"Applied {applied_count}/{len(signals)} learning signals")

    def get_reflection_metrics(self) -> Optional[Dict[str, Any]]:
        """Get reflection metrics if reflection is enabled."""
        if not self.enable_reflection:
            return None

        metrics = self.reflection_buffer.get_metrics()
        return {
            'total_cycles': metrics.total_cycles,
            'success_rate': self.reflection_buffer.get_success_rate(),
            'tool_success_rates': metrics.tool_success_rates,
            'tool_recommendations': self.reflection_buffer.get_tool_recommendations(),
            'pattern_success_rates': metrics.pattern_success_rates
        }

    async def weave_and_reflect(
        self,
        query: Query,
        feedback: Optional[Dict[str, Any]] = None,
        pattern_override: Optional[PatternCard] = None
    ) -> Spacetime:
        """
        Weave and automatically reflect on the outcome.

        Convenience method that combines weaving and reflection.

        Args:
            query: User query
            feedback: Optional feedback to store
            pattern_override: Optional pattern card override

        Returns:
            Spacetime artifact
        """
        # Weave
        spacetime = await self.weave(query, pattern_override=pattern_override)

        # Reflect
        await self.reflect(spacetime, feedback=feedback)

        # Periodically learn
        if len(self.reflection_buffer) % 10 == 0:  # Every 10 cycles
            signals = await self.learn(force=False)
            if signals:
                await self.apply_learning_signals(signals)

        return spacetime

    # ========================================================================
    # Memory Backend Helpers
    # ========================================================================

    async def _query_memory_backend(
        self,
        query_text: str,
        limit: int = 5
    ) -> List[MemoryShard]:
        """
        Query the unified memory backend and convert results to MemoryShards.

        Args:
            query_text: Query string
            limit: Maximum number of results

        Returns:
            List of MemoryShard objects
        """
        if not self.memory:
            return []

        try:
            # Import protocol types
            from HoloLoom.memory.protocol import MemoryQuery

            # Create query
            mem_query = MemoryQuery(
                text=query_text,
                user_id=getattr(self.cfg, 'user_id', 'default'),
                limit=limit
            )

            # Query backend
            result = await self.memory.recall(mem_query)

            # Convert backend Memory objects to MemoryShards
            shards = []
            for mem in result.memories:
                shard = MemoryShard(
                    id=mem.id,
                    text=mem.text,
                    episode=mem.context.get('episode', 'default'),
                    entities=mem.context.get('entities', []),
                    motifs=mem.metadata.get('motifs', []),
                    metadata=mem.metadata
                )
                shards.append(shard)

            self.logger.debug(f"Retrieved {len(shards)} shards from memory backend")
            return shards

        except Exception as e:
            self.logger.error(f"Failed to query memory backend: {e}")
            return []

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def __aenter__(self):
        """
        Async context manager entry.

        Usage:
            async with WeavingOrchestrator(cfg, shards) as shuttle:
                spacetime = await shuttle.weave(query)
                # Automatic cleanup on exit
        """
        self.logger.debug("WeavingOrchestrator context manager entered")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit with cleanup.

        Performs graceful shutdown:
        - Cancels background tasks
        - Flushes reflection buffer
        - Closes connections
        - Cleans up resources

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.logger.debug("WeavingOrchestrator context manager exiting")

        # Cleanup
        await self.close()

        # Don't suppress exceptions
        return False

    async def close(self) -> None:
        """
        Clean up all resources.

        Can be called manually or automatically via context manager.
        Safe to call multiple times (idempotent).
        """
        if self._closed:
            return

        self.logger.info("Closing WeavingOrchestrator...")

        # Cancel background tasks
        if self._background_tasks:
            self.logger.info(f"Cancelling {len(self._background_tasks)} background tasks")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for cancellation with timeout
            if self._background_tasks:
                try:
                    await asyncio.wait(
                        self._background_tasks,
                        timeout=5.0,
                        return_when=asyncio.ALL_COMPLETED
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Some background tasks did not complete within timeout")

            self._background_tasks.clear()

        # Close reflection buffer
        if self.enable_reflection and self.reflection_buffer:
            self.logger.info("Closing reflection buffer...")
            await self.reflection_buffer.flush()
            await self.reflection_buffer.close()

        # Close memory backend connections
        if self.memory:
            self.logger.info("Closing memory backend connections...")
            try:
                # Check if backend has close method
                if hasattr(self.memory, 'close'):
                    if asyncio.iscoroutinefunction(self.memory.close):
                        await self.memory.close()
                    else:
                        self.memory.close()

                # Close individual backend connections (hybrid stores)
                if hasattr(self.memory, 'neo4j') and self.memory.neo4j:
                    if hasattr(self.memory.neo4j, 'close'):
                        self.logger.debug("Closing Neo4j connection...")
                        self.memory.neo4j.close()

                if hasattr(self.memory, 'qdrant') and self.memory.qdrant:
                    if hasattr(self.memory.qdrant, 'close'):
                        self.logger.debug("Closing Qdrant connection...")
                        self.memory.qdrant.close()

                self.logger.info("Memory backend connections closed")

            except Exception as e:
                self.logger.warning(f"Error closing memory backend: {e}")

        self._closed = True
        self.logger.info("WeavingOrchestrator closed successfully")

    def spawn_background_task(self, coro) -> asyncio.Task:
        """
        Spawn a background task and track it for cleanup.

        Args:
            coro: Coroutine to run in background

        Returns:
            asyncio.Task object

        Usage:
            task = shuttle.spawn_background_task(some_async_function())
        """
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)

        # Clean up completed tasks
        task.add_done_callback(lambda t: self._background_tasks.remove(t) if t in self._background_tasks else None)

        return task


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

async def main():
    """Example usage of WeavingOrchestrator."""
    print("\n" + "="*80)
    print("HoloLoom Weaving Shuttle - Full Architecture Demo")
    print("="*80 + "\n")

    # Create sample memory shards
    shards = [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="docs",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["ALGORITHM", "OPTIMIZATION"]
        ),
        MemoryShard(
            id="shard_002",
            text="The algorithm balances exploration and exploitation by sampling from posterior distributions.",
            episode="docs",
            entities=["exploration", "exploitation", "posterior"],
            motifs=["ALGORITHM", "PROBABILITY"]
        ),
        MemoryShard(
            id="shard_003",
            text="Hive Jodi has 8 frames of brood and is very active with goldenrod flow.",
            episode="inspection_2025_10_13",
            entities=["Hive Jodi", "brood", "goldenrod"],
            motifs=["HIVE_INSPECTION", "SEASONAL"]
        )
    ]

    # Test all three patterns
    for mode in [ExecutionMode.BARE, ExecutionMode.FAST, ExecutionMode.FUSED]:
        print(f"\n{'='*80}")
        print(f"Testing {mode.value.upper()} Mode")
        print(f"{'='*80}\n")

        # Create config
        if mode == ExecutionMode.BARE:
            config = Config.bare()
        elif mode == ExecutionMode.FAST:
            config = Config.fast()
        else:
            config = Config.fused()

        # Create shuttle
        print("Initializing WeavingOrchestrator...")
        shuttle = WeavingOrchestrator(cfg=config, shards=shards)
        print("Shuttle ready!\n")

        # Process a query
        query = Query(text="What is Thompson Sampling?")
        print(f"Processing query: '{query.text}'")
        print("-" * 80)

        spacetime = await shuttle.weave(query)

        # Print spacetime
        print("\n" + "="*80)
        print("SPACETIME FABRIC")
        print("="*80)
        print(f"Query: {spacetime.query_text}")
        print(f"Tool Used: {spacetime.tool_used}")
        print(f"Confidence: {spacetime.confidence:.2f}")
        print(f"Response: {spacetime.response}")
        print(f"\nTrace:")
        print(f"  Duration: {spacetime.trace.duration_ms:.1f}ms")
        print(f"  Motifs: {len(spacetime.trace.motifs_detected)}")
        print(f"  Scales: {spacetime.trace.embedding_scales_used}")
        print(f"  Threads: {len(spacetime.trace.threads_activated)}")
        print(f"  Context Shards: {spacetime.trace.context_shards_count}")
        print(f"\nStage Timings:")
        for stage, duration in spacetime.trace.stage_durations.items():
            print(f"  {stage:25s}: {duration:6.1f}ms")
        print("="*80)


# ============================================================================
# Mode-based Orchestrator Aliases (Backward Compatibility)
# ============================================================================

class SmartWeavingOrchestrator(WeavingOrchestrator):
    """
    Backward compatibility alias for smart mode orchestrator.
    
    Usage:
        orchestrator = SmartWeavingOrchestrator(
            config=config,
            enable_math_pipeline=True
        )
    """
    def __init__(self, *args, **kwargs):
        kwargs['orchestrator_mode'] = 'smart'
        kwargs.setdefault('enable_math_pipeline', True)
        super().__init__(*args, **kwargs)


class AnalyticalWeavingOrchestrator(WeavingOrchestrator):
    """
    Backward compatibility alias for analytical mode orchestrator.
    
    Usage:
        orchestrator = AnalyticalWeavingOrchestrator(
            config=config,
            enable_analytical_verification=True
        )
    """
    def __init__(self, *args, **kwargs):
        kwargs['orchestrator_mode'] = 'analytical'
        kwargs.setdefault('enable_analytical_verification', True)
        super().__init__(*args, **kwargs)


class SimpleOrchestrator(WeavingOrchestrator):
    """
    Backward compatibility alias for simple mode orchestrator.
    
    Usage:
        orchestrator = SimpleOrchestrator(config=config)
    """
    def __init__(self, *args, **kwargs):
        kwargs['orchestrator_mode'] = 'simple'
        kwargs.setdefault('enable_reflection', False)
        super().__init__(*args, **kwargs)


# ============================================================================
# Demonstration
# ============================================================================


if __name__ == "__main__":
    asyncio.run(main())
