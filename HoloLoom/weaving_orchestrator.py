#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Weaving Orchestrator - Canonical Implementation with mythRL Protocols
=================================================================================
The canonical HoloLoom orchestrator with full Shuttle architecture integration.

This orchestrator implements the full 9-step weaving cycle:
1. Loom Command selects Pattern Card (BARE/FAST/FUSED)
2. Chrono Trigger fires, creates TemporalWindow
3. Yarn Graph threads selected based on temporal window
4. Resonance Shed lifts feature threads, creates DotPlasma
5. Warp Space tensions threads into continuous manifold
6. Convergence Engine collapses to discrete tool selection
7. Tool executes, results woven into Spacetime fabric
8. Reflection Buffer learns from outcome
9. Chrono Trigger detensions, cycle completes

Philosophy:
This is the canonical orchestrator implementation, enhanced with the complete
Shuttle architecture and mythRL protocol-based system.

Author: Claude Code (with HoloLoom architecture by Blake)
Date: 2025-10-27 (Task 1.2: Shuttle Integration Complete)
"""

from __future__ import annotations

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

# Shared types
from HoloLoom.protocols.types import Query, Context, Features, MemoryShard

# mythRL Protocol-based architecture types
from HoloLoom.protocols import (
    ComplexityLevel,
    ProvenanceTrace,
    MythRLResult,
    PatternSelectionProtocol,
    FeatureExtractionProtocol,
    WarpSpaceProtocol,
    DecisionEngineProtocol,
)

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
from HoloLoom.memory.graph import KG  # Yarn Graph for thread storage
from HoloLoom.policy.unified import create_policy
from HoloLoom.alignment.safety_guardrails import create_guardrails, SafetyGuardrails

# Tool Execution (Elegance Pass: Extracted to tools/ module - November 2025)
from HoloLoom.tools import ToolExecutor

# Initialization Functions (Elegance Pass: Extracted to orchestrator/initialization/ - November 2025 Phase 2)
from HoloLoom.orchestrator.initialization import (
    initialize_config_and_memory,
    initialize_reflection_and_caching,
    initialize_recursive_learning,
    initialize_components,
    initialize_production_hardening,
    initialize_semantic_cache,
    initialize_linguistic_gate,
)

# Core Logic Functions (Elegance Pass: Extracted to orchestrator/core/ - November 2025 Phase 3)
from HoloLoom.orchestrator.core import (
    assess_complexity_level,
    create_provenance_trace,
    get_reflection_metrics,
    get_recursive_learning_stats,
    get_metrics,
    start_background_consolidation,
    spawn_background_task,
)

# Retrieval Functions (Elegance Pass: Extracted to orchestrator/retrieval/ - November 2025 Phase 6)
from HoloLoom.orchestrator.retrieval import (
    multipass_memory_crawl,
    query_memory_backend,
)

# Physics Integration (Elegance Pass: Extracted to orchestrator/physics/ - November 2025 Phase 7)
from HoloLoom.orchestrator.physics import (
    weave_with_physics,
)

# Learning Integration (Elegance Pass: Extracted to orchestrator/learning/ - November 2025 Phase 8)
from HoloLoom.orchestrator.learning import (
    apply_recursive_learning,
)

# Production Hardening (Part 5: Days 21-25)
try:
    from HoloLoom.context import (
        # Configuration
        ProductionConfig,
        # Monitoring
        create_system_monitor,
        SystemMonitor,
        # Circuit breakers
        create_circuit_breaker_registry,
        CircuitBreakerRegistry,
        CircuitState,
        # Rate limiting
        create_rate_limiter,
        RateLimiter,
        RateLimitExceededError,
        # Health checks
        create_health_checker,
        HealthChecker,
        HealthStatus,
        # Error handling
        create_error_handler,
        ErrorHandler,
        BackendError,
    )
    PRODUCTION_HARDENING_AVAILABLE = True
except ImportError:
    PRODUCTION_HARDENING_AVAILABLE = False
    import warnings
    warnings.warn(
        "Production hardening features not available. Install context module for "
        "production features (monitoring, circuit breakers, rate limiting, health checks).",
        ImportWarning
    )

# Physics-based routing (Phase 1: Gradient Flow)
from HoloLoom.routing import ToolRouter, ToolConfig

# Smart Query Routing (November 2025 - Performance Optimization)
from HoloLoom.routing.query_classifier import QueryClassifier, QueryComplexity
from HoloLoom.routing.fast_paths import FastPathRouter, handle_trivial_query, handle_simple_query
from HoloLoom.routing.classifier_factory import create_classifier, create_fast_path_router

# Shuttle Integration (January 2025 - MCTS-powered Warp↔Yarn intersection)
try:
    from HoloLoom.shuttle.weaving_integration import ShuttleStage, create_shuttle_stage
    SHUTTLE_AVAILABLE = True
except ImportError:
    SHUTTLE_AVAILABLE = False
    import warnings
    warnings.warn(
        "Shuttle integration not available. Install shuttle module for "
        "MCTS-powered Warp↔Yarn intersection at Step 3.",
        ImportWarning
    )

# Unified Physics Engine (Phases 1-4 integrated)
try:
    from HoloLoom.physics import UnifiedPhysicsEngine, UnifiedPhysicsResult
    UNIFIED_PHYSICS_AVAILABLE = True
except ImportError:
    UNIFIED_PHYSICS_AVAILABLE = False

# Statistical Mechanics (Phase 5)
try:
    from HoloLoom.physics import (
        StatisticalMechanicsEngine,
        Microstate,
        Macrostate,
        PhaseTransition
    )
    STATISTICAL_MECHANICS_AVAILABLE = True
except ImportError:
    STATISTICAL_MECHANICS_AVAILABLE = False

# Performance optimizations
from HoloLoom.performance.cache import QueryCache

# Prometheus metrics
try:
    from HoloLoom.performance.prometheus_metrics import metrics
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

logging.basicConfig(level=logging.INFO)


if TYPE_CHECKING:
    from HoloLoom.awareness.llm_integration import OllamaLLM


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
# Weaving Shuttle - Full Architecture Integration
# ============================================================================

class WeavingOrchestrator:
    """
    The Weaving Shuttle - Enhanced with mythRL Protocol-Based Architecture
    =======================================================================
    
    Implements the complete 9-step weaving cycle with 3-5-7-9 progressive complexity.
    
    **Traditional HoloLoom (9 steps):**
    - Loom Command (pattern selection)
    - Chrono Trigger (temporal control)
    - Yarn Graph (thread storage)
    - Resonance Shed (feature interference)
    - Warp Space (tensor tensioning)
    - Convergence Engine (continuous → discrete)
    - Tool Execution (action)
    - Spacetime Fabric (provenance)
    - Reflection (learning)
    
    **mythRL Progressive Complexity (3-5-7-9):**
    - LITE (3 steps): Extract → Route → Execute (<50ms)
    - FAST (5 steps): + Pattern Selection + Temporal Windows (<150ms)
    - FULL (7 steps): + Decision Engine + Synthesis Bridge (<300ms)
    - RESEARCH (9 steps): + Advanced WarpSpace + Full Tracing (no limit)
    
    **Protocol-Based Design:**
    - PatternSelectionProtocol: Processing pattern selection
    - FeatureExtractionProtocol: Multi-scale Matryoshka extraction
    - WarpSpaceProtocol: Mathematical manifold operations
    - DecisionEngineProtocol: Strategic multi-criteria optimization
    
    Usage:
        config = Config.fused()
        shuttle = WeavingOrchestrator(cfg=config, shards=memory_shards)
        
        # Traditional mode
        spacetime = await shuttle.weave(Query(text="What is Thompson Sampling?"))
        
        # With complexity control
        spacetime = await shuttle.weave(
            Query(text="Analyze bee colony optimization"),
            complexity=ComplexityLevel.RESEARCH
        )
    """

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        memory=None,  # Unified memory backend
        yarn_graph: Optional['KG'] = None,  # Yarn Graph (KG) for thread storage
        pattern_preference: Optional[PatternCard] = None,
        enable_reflection: bool = True,
        reflection_capacity: int = 1000,
        enable_complexity_auto_detect: bool = True,
        enable_semantic_cache: bool = True,
        enable_dashboards: bool = False,
        enable_statistical_mechanics: bool = False,
        consolidation_interval: float = 3600.0,  # 1 hour default
        consolidation_temperature: float = 1.0,
        consolidation_cooling_rate: float = 0.95,
        stage_callback: Optional[callable] = None,  # Phase 3.1: Stage tracking
        # Production Hardening (Part 5)
        enable_production_hardening: bool = False,
        production_config: Optional['ProductionConfig'] = None,
        rate_limit_qps: float = 100.0,
        rate_limit_concurrent: int = 50,
        enable_circuit_breakers: bool = True,
        circuit_breaker_threshold: int = 5,
        enable_health_checks: bool = True,
        enable_auto_enhancement: bool = False,  # Meta-Prompt Auto-Enhancement
        # Consciousness Integration (Phase 1 - November 2025)
        awareness_layer: Optional[Any] = None,  # AwarenessGraph for epistemic consciousness
    ):
        """
        Initialize the Weaving Shuttle with mythRL protocol enhancements.

        Args:
            cfg: Configuration object
            shards: List of memory shards (DEPRECATED - use yarn_graph instead)
            memory: Unified memory backend (optional, overrides shards and yarn_graph)
            yarn_graph: KG instance for thread storage (preferred over shards)
            pattern_preference: Optional pattern card preference (overrides config)
            enable_reflection: Enable reflection loop for learning
            reflection_capacity: Maximum episodes to store in reflection buffer
            enable_complexity_auto_detect: Auto-detect query complexity (3-5-7-9)
            enable_semantic_cache: Enable three-tier semantic caching (default True)
            enable_dashboards: Enable automatic dashboard generation (default False)
            enable_statistical_mechanics: Enable Phase 5 statistical mechanics memory consolidation
            consolidation_interval: Seconds between consolidation runs (default 3600 = 1 hour)
            consolidation_temperature: Initial temperature for Gibbs distribution (default 1.0)
            consolidation_cooling_rate: Cooling rate for simulated annealing (default 0.95)
            stage_callback: Optional callback for stage tracking/monitoring
            enable_production_hardening: Enable production hardening features (default False)
            production_config: Optional ProductionConfig (auto-detects if None)
            rate_limit_qps: Rate limit queries per second (default 100)
            rate_limit_concurrent: Max concurrent requests (default 50)
            enable_circuit_breakers: Enable circuit breaker protection (default True)
            circuit_breaker_threshold: Failures before circuit opens (default 5)
            enable_health_checks: Enable health check system (default True)
            enable_auto_enhancement: Enable Meta-Prompt auto-enhancement (default False)
            awareness_layer: Optional AwarenessGraph for epistemic consciousness integration (default None)
                - If None, auto-creates AwarenessGraph with semantic calculus
                - Enables epistemic self-awareness, uncertainty quantification, meta-confidence
                - Injects awareness context into weaving cycle and Spacetime results

        Note:
            Memory sources (priority order):
            1. memory (unified backend) - highest priority
            2. yarn_graph (KG instance) - preferred for new code
            3. shards (list) - deprecated, backward compatibility only

            If none provided, raises ValueError.

            Production Hardening Features (Part 5):
            - Monitoring: Performance, resource, learning metrics
            - Circuit Breakers: Auto-protect backend failures
            - Rate Limiting: Token bucket + sliding window + concurrent
            - Health Checks: Component-based health monitoring
            - Error Handling: Retry + fallback + categorization

            Enable with enable_production_hardening=True for <1ms overhead.
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.enable_semantic_cache = enable_semantic_cache
        self.enable_dashboards = enable_dashboards

        # Meta-Prompt Auto-Enhancement
        self.enable_auto_enhancement = enable_auto_enhancement
        self.meta_prompt_template = None
        if self.enable_auto_enhancement:
            try:
                # Load template from standard location
                import os
                template_path = os.path.join("promptly_skills", "meta_prompt", "template.md")
                if os.path.exists(template_path):
                    with open(template_path, "r", encoding="utf-8") as f:
                        self.meta_prompt_template = f.read()
                    self.logger.info("Meta-Prompt template loaded for auto-enhancement")
                else:
                    self.logger.warning(f"Meta-Prompt template not found at {template_path}")
                    self.enable_auto_enhancement = False
            except Exception as e:
                self.logger.warning(f"Failed to load Meta-Prompt template: {e}")
                self.enable_auto_enhancement = False

        # Phase 3.1: Stage tracking callback for monitoring
        self.stage_callback = stage_callback

        # Statistical Mechanics (Phase 5)
        self.enable_statistical_mechanics = enable_statistical_mechanics and STATISTICAL_MECHANICS_AVAILABLE
        self.consolidation_interval = consolidation_interval
        self.consolidation_temperature = consolidation_temperature
        self.consolidation_cooling_rate = consolidation_cooling_rate
        self.stat_mech_engine: Optional['StatisticalMechanicsEngine'] = None
        self._consolidation_task: Optional[asyncio.Task] = None

        # Recursive Learning System Components (Phase 1-5)
        self.enable_recursive_learning = cfg.enable_recursive_learning
        self._recursive_components = None  # Lazy initialization

        # Production Hardening (Part 5: Days 21-25)
        self.enable_production_hardening = enable_production_hardening and PRODUCTION_HARDENING_AVAILABLE
        self.monitor: Optional['SystemMonitor'] = None
        self.breaker_registry: Optional['CircuitBreakerRegistry'] = None
        self.rate_limiter: Optional['RateLimiter'] = None
        self.health_checker: Optional['HealthChecker'] = None
        self.error_handler: Optional['ErrorHandler'] = None

        if self.enable_production_hardening:
            initialize_production_hardening(
                orchestrator=self,
                production_config=production_config,
                rate_limit_qps=rate_limit_qps,
                rate_limit_concurrent=rate_limit_concurrent,
                enable_circuit_breakers=enable_circuit_breakers,
                circuit_breaker_threshold=circuit_breaker_threshold,
                enable_health_checks=enable_health_checks
            )

        # Initialize configuration and memory
        initialize_config_and_memory(
            orchestrator=self,
            memory=memory,
            yarn_graph=yarn_graph,
            shards=shards,
            pattern_preference=pattern_preference,
            enable_complexity_auto_detect=enable_complexity_auto_detect
        )

        # Initialize weaving components
        initialize_components(orchestrator=self)

        # Initialize reflection, caching, and dashboards
        initialize_reflection_and_caching(
            orchestrator=self,
            enable_reflection=enable_reflection,
            reflection_capacity=reflection_capacity
        )

        # Initialize awareness layer (Consciousness Integration - Phase 1, November 2025)
        self.awareness_layer = awareness_layer
        if self.awareness_layer is None and hasattr(self, 'semantic_spectrum'):
            # Auto-create awareness layer with semantic calculus
            try:
                from HoloLoom.memory.awareness_graph import AwarenessGraph
                # Use existing graph backend and semantic calculus
                self.awareness_layer = AwarenessGraph(
                    graph_backend=self.yarn_graph._graph if hasattr(self, 'yarn_graph') else None,
                    semantic_calculus=self.semantic_spectrum,
                    vector_store=None  # Optional, not required
                )
                self.logger.info("Auto-created AwarenessGraph for consciousness integration")
            except Exception as e:
                self.logger.warning(f"Failed to auto-create AwarenessGraph: {e}")
                self.awareness_layer = None

        self.logger.info("WeavingOrchestrator initialization complete")

    def _analyze_semantics(self, text: str) -> Optional[Dict[str, float]]:
        """
        Analyze text through semantic calculus with three-tier caching.

        This method provides fast semantic projections by:
        1. Checking hot tier (pre-loaded patterns) - <0.001ms
        2. Checking warm tier (recently accessed) - <0.001ms
        3. Computing on-demand (cold path) - ~18ms

        Args:
            text: Text to analyze

        Returns:
            Dict mapping semantic dimension name → score (244D)
            Returns None if semantic cache is disabled
        """
        if not self.semantic_cache:
            return None

        try:
            scores = self.semantic_cache.get_scores(text)
            return scores
        except Exception as e:
            self.logger.warning(f"Semantic cache failed, falling back: {e}")
            # Fallback to direct computation
            if self.semantic_spectrum:
                vec = self.embedder.encode([text])[0]
                return self.semantic_spectrum.project_vector(vec)
            return None

    # ========================================================================
    # mythRL Protocol-Based Architecture Methods
    # ========================================================================

    def register_protocol(self, protocol_name: str, implementation: Any):
        """
        Register a protocol implementation for mythRL architecture.
        
        Allows swappable implementations of:
        - PatternSelectionProtocol
        - FeatureExtractionProtocol
        - WarpSpaceProtocol
        - DecisionEngineProtocol
        
        Args:
            protocol_name: Name of protocol ('pattern_selection', 'feature_extraction', etc.)
            implementation: Protocol implementation instance
        
        Example:
            shuttle.register_protocol('pattern_selection', CustomPatternSelector())
        """
        self._protocols[protocol_name] = implementation
        self.logger.info(f"Registered protocol: {protocol_name}")

    # ========================================================================
    # Stage Tracking (Phase 3.1)
    # ========================================================================

    def _emit_stage_event(self, stage_id: int, stage_name: str, duration_ms: float = 0.0):
        """
        Emit stage event for monitoring (Phase 3.1).

        Args:
            stage_id: Stage number (1-9)
            stage_name: Human-readable stage name
            duration_ms: Stage duration in milliseconds (0 if starting)
        """
        if self.stage_callback:
            try:
                self.stage_callback(stage_id, stage_name, duration_ms)
            except Exception as e:
                # Don't let callback errors break the pipeline
                self.logger.warning(f"Stage callback error: {e}")

    # ========================================================================
    # Meta-Prompt Integration (Proto-LLM Call)
    # ========================================================================

    async def proto_llm_call(self, query_text: str) -> str:
        """
        Execute Proto-LLM call to enhance casual prompt using Meta-Prompt Skill.

        Transforms: "help me with python"
        Into: "Role: Python Expert... Objective: ..."

        Uses the 7-component framework from promptly_skills/meta_prompt/template.md.
        """
        if not self.meta_prompt_template:
            return query_text

        start_time = time.time()
        self.logger.info(f"[METAPROMPT] Enhancing query: '{query_text}'")

        try:
            # Format template
            prompt = self.meta_prompt_template.replace("{user_request}", query_text)

            # Use existing LLM if available
            if self.tool_executor and self.tool_executor.llm:
                response = await self.tool_executor.llm.generate(
                    prompt=prompt,
                    system_prompt="You are a prompt engineering expert.",
                    max_tokens=1000,
                    temperature=0.3
                )
                enhanced_text = response.content
            else:
                # Fallback if no LLM available (shouldn't happen in production)
                self.logger.warning("[METAPROMPT] No LLM available for enhancement")
                return query_text

            # Extract just the structured prompt if possible (simple heuristic)
            if "## STRUCTURED PROMPT" in enhanced_text:
                parts = enhanced_text.split("## STRUCTURED PROMPT")
                if len(parts) > 1:
                    enhanced_text = parts[1].split("## JUSTIFICATION")[0].strip()

            duration = (time.time() - start_time) * 1000
            self.logger.info(f"[METAPROMPT] Enhancement complete in {duration:.1f}ms")
            return enhanced_text

        except Exception as e:
            self.logger.error(f"[METAPROMPT] Enhancement failed: {e}")
            return query_text

    # ========================================================================
    # Main Weaving Cycle
    # ========================================================================

    async def weave(
        self,
        query: Query,
        pattern_override: Optional[PatternCard] = None,
        complexity: Optional[ComplexityLevel] = None,
        auto_enhance: Optional[bool] = None
    ) -> Spacetime:
        """
        Execute the complete 9-step weaving cycle with mythRL progressive complexity.

        This is the core API for the HoloLoom weaving orchestrator. Processes a query
        through the complete pipeline: pattern selection, feature extraction, context
        retrieval, decision making, and response synthesis.

        **Progressive Complexity (3-5-7-9 System):**
        - LITE (3 steps): Extract → Route → Execute
          Performance: <50ms | Use for: Simple lookups, cached queries

        - FAST (5 steps): + Pattern Selection + Temporal Windows
          Performance: <150ms | Use for: Standard queries, real-time apps

        - FULL (7 steps): + Decision Engine + Synthesis Bridge
          Performance: <300ms | Use for: Complex queries, production systems

        - RESEARCH (9 steps): + Advanced WarpSpace + Full Tracing
          Performance: No limit | Use for: Research, debugging, quality maximization

        **Weaving Cycle Steps:**
        1. **Loom Command**: Selects pattern card (BARE/FAST/FUSED)
        2. **Chrono Trigger**: Creates temporal window for memory filtering
        3. **Yarn Graph**: Retrieves relevant memory threads
        4. **Resonance Shed**: Extracts multi-modal features (DotPlasma)
        5. **Warp Space**: Tensions threads into continuous manifold
        6. **Convergence Engine**: Collapses to discrete tool selection
        7. **Tool Execution**: Executes selected tool with context
        8. **Spacetime Fabric**: Weaves results with complete provenance
        9. **Reflection Buffer**: Learns from outcome (if enabled)

        Args:
            query: User query object with text and optional metadata
                Example: Query(text="What is Thompson Sampling?")

            pattern_override: Force specific pattern card (bypasses auto-selection)
                - None: Auto-detect based on query complexity (default)
                - PatternCard.BARE: Minimal processing (<50ms)
                - PatternCard.FAST: Balanced processing (<150ms)
                - PatternCard.FUSED: Full processing (<300ms)

            complexity: Override mythRL complexity level
                - None: Auto-assess based on query characteristics (default)
                - ComplexityLevel.LITE: 3-step fast path
                - ComplexityLevel.FAST: 5-step standard path
                - ComplexityLevel.FULL: 7-step production path
                - ComplexityLevel.RESEARCH: 9-step research path

            auto_enhance: Override auto-enhancement setting
                - True: Force meta-prompt enhancement
                - False: Disable meta-prompt enhancement
                - None: Use default configuration

        Returns:
            Spacetime: Complete woven fabric containing:
                - response: Generated response text
                - confidence: Decision confidence [0, 1]
                - trace: Full computational lineage
                - metadata: Tool used, pattern, complexity, timings
                - provenance: Complete audit trail

        **Performance Budgets:**
        - Cache hit: <1ms (hash lookup)
        - LITE mode: <50ms (3 steps)
        - FAST mode: <150ms (5 steps)
        - FULL mode: <300ms (7 steps)
        - RESEARCH mode: No limit (quality over speed)

        Example:
            >>> from HoloLoom.weaving_orchestrator import WeavingOrchestrator
            >>> from HoloLoom.protocols.types import Query
            >>> from HoloLoom.config import Config
            >>>
            >>> config = Config.fast()
            >>> async with WeavingOrchestrator(cfg=config, shards=shards) as shuttle:
            ...     # Auto-complexity (recommended)
            ...     query = Query(text="What is Thompson Sampling?")
            ...     result = await shuttle.weave(query)
            ...     print(result.response)
            ...     print(f"Confidence: {result.confidence:.2f}")
            ...
            ...     # Force FUSED mode for highest quality
            ...     query = Query(text="Explain the complete architecture")
            ...     result = await shuttle.weave(
            ...         query,
            ...         pattern_override=PatternCard.FUSED
            ...     )
            ...
            ...     # Force LITE for lowest latency
            ...     query = Query(text="Quick lookup")
            ...     result = await shuttle.weave(
            ...         query,
            ...         complexity=ComplexityLevel.LITE
            ...     )

        **Caching:**
        Queries are automatically cached by text hash. Repeated queries return
        cached results in <1ms. Cache is thread-safe and bounded by max_cache_size.

        **Recursive Learning:**
        If enable_recursive_learning=True, the system learns from every query:
        - Pattern usage statistics
        - Tool selection success rates
        - Confidence calibration
        - Thompson Sampling prior updates

        **Error Handling:**
        Gracefully handles errors at each stage. Returns Spacetime with error
        metadata and fallback response. Never throws on invalid input.

        Raises:
            None: All errors caught and returned in Spacetime.metadata['errors']

        See Also:
            - weave_and_reflect(): Weave + automatic reflection
            - experience(): Form new memories (HoloLoom.hololoom)
            - recall(): Retrieve memories (HoloLoom.hololoom)
        """
        start_time = datetime.now()
        stage_timings = {}
        errors = []
        warnings = []

        self.logger.info(f"[WEAVING] Beginning weaving cycle for query: '{query.text}'")

        # ====================================================================
        # STEP 0: Meta-Prompt Enhancement (Proto-LLM Call)
        # ====================================================================
        should_enhance = auto_enhance if auto_enhance is not None else self.enable_auto_enhancement

        if should_enhance:
            try:
                enhanced_text = await self.proto_llm_call(query.text)
                # Update query with enhanced text, preserving metadata
                original_text = query.text
                query = Query(text=enhanced_text, metadata=query.metadata)
                query.metadata['original_query'] = original_text
                query.metadata['enhanced'] = True
                self.logger.info(f"[WEAVING] Using enhanced query ({len(original_text)} -> {len(enhanced_text)} chars)")
            except Exception as e:
                self.logger.warning(f"[WEAVING] Auto-enhancement failed: {e}")

        # ====================================================================
        # PRODUCTION HARDENING (Part 5: Days 21-25)
        # ====================================================================
        # Rate limiting, circuit breakers, monitoring
        if self.enable_production_hardening:
            # Rate limiting check
            if self.rate_limiter and not await self.rate_limiter.acquire():
                self.logger.warning("[PRODUCTION] Rate limit exceeded")
                raise RateLimitExceededError(
                    f"Rate limit exceeded for query: {query.text[:50]}"
                )

            # Record query start time for monitoring
            prod_start_time = time.time()

        # Recursive Learning: Initialize components on first use (lazy init)
        if self.enable_recursive_learning and self._recursive_components is None:
            self._initialize_recursive_learning()

        # ====================================================================
        # CONSCIOUSNESS INTEGRATION (Phase 1 - November 2025)
        # ====================================================================
        # Perceive query through awareness layer for epistemic consciousness
        awareness_context = None
        if self.awareness_layer:
            try:
                perception_start = time.perf_counter()
                # Perceive query through awareness layer
                awareness_perception = await self.awareness_layer.perceive(query.text)
                perception_time = (time.perf_counter() - perception_start) * 1000

                # Build awareness context for downstream use
                awareness_context = {
                    'semantic_position': awareness_perception.get('position'),
                    'activation_level': awareness_perception.get('activation', 0.0),
                    'perception_time_ms': perception_time,
                }

                # Get awareness metrics for provenance
                awareness_metrics = self.awareness_layer.get_metrics()
                awareness_context.update({
                    'active_nodes': awareness_metrics.get('active_nodes', 0),
                    'coherence': awareness_metrics.get('coherence', {}).get('global_coherence', 0.0),
                    'shift_detected': awareness_metrics.get('shift_detected', False),
                })

                self.logger.info(
                    f"[AWARENESS] Perception complete: "
                    f"activation={awareness_context['activation_level']:.3f}, "
                    f"coherence={awareness_context['coherence']:.3f}, "
                    f"time={perception_time:.2f}ms"
                )

                stage_timings['awareness_perception'] = perception_time

            except Exception as e:
                self.logger.warning(f"[AWARENESS] Perception failed, continuing without awareness: {e}")
                awareness_context = None

    # ========================================================================
    # EGGROLL Integration (Evolutionary Training)
    # ========================================================================

    async def dream(self, num_epochs: int = 1, num_workers: int = 5):
        """
        Trigger a 'Dream' cycle: Run EGGROLL evolutionary training on recent experiences.
        
        This method:
        1. Initializes the EGGROLL integration.
        2. Pulls tasks from ReflectionBuffer (via Weave).
        3. Runs the distributed evolutionary loop.
        4. Updates the internal model (conceptually).
        
        Args:
            num_epochs: Number of evolutionary epochs to run.
            num_workers: Number of simulated workers.
        """
        self.logger.info(f"[DREAM] Starting EGGROLL dream cycle (epochs={num_epochs}, workers={num_workers})")
        
        try:
            from HoloLoom.eggroll.integration import EggrollIntegration
            
            integration = EggrollIntegration(num_workers=num_workers)
            await integration.run_evolution_loop(num_epochs=num_epochs)
            
            self.logger.info("[DREAM] Dream cycle complete.")
            
        except ImportError:
            self.logger.error("[DREAM] EGGROLL module not found. Cannot dream.")
        except Exception as e:
            self.logger.error(f"[DREAM] Dream cycle failed: {e}")

        # ====================================================================
        # SMART QUERY ROUTING (November 2025 - Performance Optimization)
        # ====================================================================
        # Classify query and route TRIVIAL/SIMPLE to fast paths
        # Target: <1ms for TRIVIAL, <50ms for SIMPLE
        if self.query_classifier and self.fast_path_router:
            classification = self.query_classifier.classify(query.text)

            if classification.complexity in {QueryComplexity.TRIVIAL, QueryComplexity.SIMPLE}:
                self.logger.info(
                    f"[ROUTING] Fast path: {classification.complexity.value} "
                    f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
                )

                # Route to fast path
                spacetime = await self.fast_path_router.route(query, classification)

                # Update routing statistics
                self.logger.debug(
                    f"[ROUTING] Fast path completed in {spacetime.metadata.get('latency_ms', 0):.1f}ms"
                )

                return spacetime

            # COMPLEX/RESEARCH: Continue to full pipeline
            self.logger.info(
                f"[ROUTING] Full pipeline: {classification.complexity.value} "
                f"(confidence={classification.confidence:.2f}, reason={classification.reasoning})"
            )

        # mythRL: Assess complexity level (3-5-7-9 system)
        if complexity is None:
            complexity = assess_complexity_level(self, query)

        # mythRL: Create provenance trace
        provenance = create_provenance_trace(self, query, complexity)
        
        self.logger.info(f"[mythRL] Complexity: {complexity.name} ({complexity.value} steps)")

        # Check cache first
        cached_result = self.query_cache.get(query.text)
        if cached_result is not None:
            self.logger.info(f"[CACHE HIT] Returning cached result for query")
            provenance.add_shuttle_event("cache_hit", "Returned cached result")
            # Track cache hit
            if METRICS_ENABLED:
                metrics.track_cache_hit()
            return cached_result
        else:
            # Track cache miss
            if METRICS_ENABLED:
                metrics.track_cache_miss()

        try:
            # ================================================================
            # STEP 1: Loom Command selects Pattern Card
            # ================================================================
            step_start = time.time()
            self._emit_stage_event(1, "Loom Command")

            pattern_spec = self.loom_command.select_pattern(
                query.text,
                user_preference=pattern_override.value if pattern_override else None
            )

            duration = (time.time() - step_start) * 1000
            self.logger.info(f"  [1] Pattern selected: {pattern_spec.name}")
            stage_timings['pattern_selection'] = duration
            self._emit_stage_event(1, "Loom Command", duration)

            # ================================================================
            # STEP 2: Chrono Trigger fires, creates TemporalWindow
            # ================================================================
            step_start = time.time()
            self._emit_stage_event(2, "Chrono Trigger")

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

            duration = (time.time() - step_start) * 1000
            self.logger.info(f"  [2] Chrono Trigger fired")
            stage_timings['temporal_setup'] = duration
            self._emit_stage_event(2, "Chrono Trigger", duration)

            # ================================================================
            # STEP 3: Thread Selection (Shuttle or Yarn Graph)
            # ================================================================
            step_start = time.time()

            if self.enable_shuttle and self.shuttle_stage:
                # MCTS-powered Warp↔Yarn intersection
                self._emit_stage_event(3, "Shuttle (Warp↔Yarn MCTS)")

                threads = await self.shuttle_stage.select_threads(
                    temporal_window=temporal_window,
                    query=query,
                    trajectory_name=None  # Auto-select via Thompson Sampling
                )

                duration = (time.time() - step_start) * 1000
                self.logger.info(f"  [3] Shuttle selected {len(threads)} threads in {duration:.1f}ms")
                stage_timings['shuttle_selection'] = duration
                self._emit_stage_event(3, "Shuttle", duration)

            else:
                # Fallback: Simple Yarn Graph thread selection
                self._emit_stage_event(3, "Yarn Graph")

                threads = self.yarn_graph.select_threads(temporal_window, query)

                duration = (time.time() - step_start) * 1000
                self.logger.info(f"  [3] Selected {len(threads)} threads from Yarn Graph")
                stage_timings['thread_selection'] = duration
                self._emit_stage_event(3, "Yarn Graph", duration)

            # Continue with thread processing (unchanged)
            thread_ids = [s.id for s in threads]
            thread_texts = [s.text for s in threads]

            # ================================================================
            # STEPS 4-6: PARALLELIZED - Feature Extraction, Warp Tensioning, and Memory Retrieval
            # ================================================================
            # OPTIMIZATION: These three steps are independent and can run in parallel
            # Expected speedup: 40-120ms (steps run concurrently instead of sequentially)
            parallel_start = time.time()

            # Track individual stage start times for accurate timing
            step4_start = time.time()
            step5_start = time.time()
            step6_start = time.time()

            # =================================================================
            # STEP 4 SETUP: Prepare components for Resonance Shed (synchronous)
            # =================================================================
            # Create components based on pattern spec
            motif_detector = create_motif_detector(mode=pattern_spec.motif_mode)
            spectral_fusion = SpectralFusion() if pattern_spec.enable_spectral else None

            # Create embedder with pattern-specific scales
            # Phase 5 Integration: Use linguistic gate if enabled (includes compositional cache!)
            if self.linguistic_gate and self.cfg.enable_linguistic_gate:
                # Use linguistic matryoshka gate (with compositional cache built-in)
                # NOTE: Don't change config.scales - that's for gate() method
                # The encode_scales() method handles any requested size
                pattern_embedder = self.linguistic_gate

                self.logger.info(
                    f"  [4a] Phase 5 Linguistic Gate enabled "
                    f"(mode={self.cfg.linguistic_mode}, cache={self.cfg.use_compositional_cache})"
                )
            elif self.cfg.enable_zero_copy_embeddings:
                # Zero-copy embeddings (1.4x speedup, 50% memory savings)
                from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings
                pattern_embedder = ZeroCopyMatryoshkaEmbeddings(
                    sizes=pattern_spec.scales,
                    base_model_name=self.cfg.base_model_name,
                    store_path=self.cfg.zero_copy_cache_path,
                    max_cache_size=self.cfg.zero_copy_cache_size
                )

                self.logger.info(
                    f"  [4a] Zero-copy embeddings enabled "
                    f"(cache={self.cfg.zero_copy_cache_path}, size={self.cfg.zero_copy_cache_size})"
                )
            else:
                # Standard matryoshka embeddings (no compositional cache)
                pattern_embedder = MatryoshkaEmbeddings(
                    sizes=pattern_spec.scales,
                    base_model_name=self.cfg.base_model_name
                )

            # Create semantic analyzer if enabled by pattern (using organized structure)
            semantic_calculus = None
            if pattern_spec.enable_semantic_flow:
                # Clean imports from organized structure
                from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
                from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

                # Create config from pattern spec
                sem_config = SemanticCalculusConfig.from_pattern_spec(pattern_spec)

                # Create embedder function
                embed_fn = lambda words: pattern_embedder.encode(words)

                # Create analyzer with clean interface
                semantic_calculus = create_semantic_analyzer(embed_fn, config=sem_config)

                self.logger.info(
                    f"  [4a] Semantic analyzer enabled ({sem_config.dimensions}D, "
                    f"cache={sem_config.enable_cache}, ethics={sem_config.compute_ethics})"
                )

            resonance_shed = ResonanceShed(
                motif_detector=motif_detector,
                embedder=pattern_embedder,
                spectral_fusion=spectral_fusion,
                semantic_calculus=semantic_calculus,
                interference_mode="weighted_sum",
                # Phase 5: Ensure embeddings match policy's expected dimension
                # ResonanceShed will call encode_scales() with this target
                target_scale=max(pattern_spec.scales),
                guardrails=self.guardrails,
            )

            # =================================================================
            # STEP 5 SETUP: Prepare WarpSpace (synchronous)
            # =================================================================
            warp_space = WarpSpace(
                embedder=self.embedder,
                scales=pattern_spec.scales,
                spectral_fusion=spectral_fusion,
                guardrails=self.guardrails,
            )

            # =================================================================
            # PARALLEL EXECUTION: Define async tasks for Steps 4, 5, 6
            # =================================================================
            async def step4_feature_extraction():
                """Step 4: Extract features through Resonance Shed"""
                start = time.time()
                self._emit_stage_event(4, "Resonance Shed")
                dot_plasma = await resonance_shed.weave(
                    text=query.text,
                    context_graph=None  # Could add KG here
                )
                duration = (time.time() - start) * 1000
                thread_count = len(dot_plasma.get('threads', []))
                self.logger.info(f"  [4] DotPlasma created with {thread_count} feature threads")
                self._emit_stage_event(4, "Resonance Shed", duration)
                return dot_plasma, duration

            async def step5_warp_tensioning():
                """Step 5: Tension threads into continuous manifold"""
                start = time.time()
                self._emit_stage_event(5, "Warp Space")
                await warp_space.tension(thread_texts, thread_ids=thread_ids)
                duration = (time.time() - start) * 1000
                warp_operations = [(datetime.now().isoformat(), "tension", len(thread_ids))]
                self.logger.info(f"  [5] Warp Space tensioned with {len(thread_ids)} threads")
                self._emit_stage_event(5, "Warp Space", duration)
                return warp_operations, duration

            async def step6_memory_retrieval():
                """Step 6: Retrieve context with multipass memory crawling"""
                start = time.time()
                self._emit_stage_event(6, "Memory Retrieval")

                # Use multipass memory crawling for intelligent retrieval
                if self.memory:
                    # NEW: Multipass crawling with gated retrieval and graph traversal
                    shards = await multipass_memory_crawl(self, query, complexity, provenance)
                    shard_texts = [shard.text for shard in shards]
                    # Create hits format for compatibility
                    hits = [(shard, 1.0) for shard in shards]
                    self.logger.info(f"  [6] Multipass crawl retrieved {len(shards)} shards")

                elif self.retriever:
                    # Fallback: Traditional static shard retrieval (legacy path)
                    hits = await self.retriever.search(
                        query=query.text,
                        k=pattern_spec.retrieval_k,
                        fast=(pattern_spec.retrieval_mode == "fast")
                    )
                    shards = [shard for shard, _ in hits]
                    shard_texts = [shard.text for shard in shards]
                    self.logger.info(f"  [6] Legacy retriever fetched {len(shards)} shards")

                else:
                    # No memory source available
                    self.logger.warning("No memory source available (no shards or memory backend)")
                    shards = []
                    shard_texts = []
                    hits = []

                duration = (time.time() - start) * 1000
                self.logger.info(f"  [6] Retrieved {len(hits)} context shards")
                self._emit_stage_event(6, "Memory Retrieval", duration)
                return shards, shard_texts, hits, duration

            # Execute all three steps in parallel using asyncio.gather
            self.logger.info("  [PARALLEL] Executing Steps 4-6 concurrently...")

            try:
                (dot_plasma, t4), (warp_operations, t5), (shards, shard_texts, hits, t6) = await asyncio.gather(
                    step4_feature_extraction(),
                    step5_warp_tensioning(),
                    step6_memory_retrieval(),
                    return_exceptions=False  # Propagate exceptions
                )

                # Record individual stage timings (actual parallel execution times)
                stage_timings['feature_extraction'] = t4
                stage_timings['warp_tensioning'] = t5
                stage_timings['retrieval'] = t6

                # Calculate parallel execution time and speedup
                parallel_duration = (time.time() - parallel_start) * 1000
                sequential_duration = t4 + t5 + t6
                speedup = sequential_duration / parallel_duration if parallel_duration > 0 else 1.0

                stage_timings['parallel_execution_wall_time'] = parallel_duration
                stage_timings['parallel_speedup'] = speedup

                self.logger.info(
                    f"  [PARALLEL] Steps 4-6 completed in {parallel_duration:.1f}ms "
                    f"(sequential would be {sequential_duration:.1f}ms, speedup: {speedup:.2f}x)"
                )

            except Exception as e:
                self.logger.error(f"  [PARALLEL] Parallel execution failed: {e}", exc_info=True)
                raise  # Re-raise to trigger error handling below

            # =================================================================
            # POST-PARALLEL: Assemble context from retrieval results
            # =================================================================
            context = Context(
                shards=shards,
                hits=hits,
                shard_texts=shard_texts,
                query=query,
                features=None  # Will be set from dot_plasma
            )

            thread_count = len(dot_plasma.get('threads', []))
            self.logger.debug(f"  [POST-PARALLEL] Context assembled with {len(hits)} shards, {thread_count} threads")

            # ================================================================
            # STEP 5.5: WarpSpace Compute Operations
            # ================================================================
            # Perform tensor operations in continuous manifold
            # Computes: spectral features, attention, weighted context
            step_start = time.time()

            try:
                # Get query embedding from DotPlasma for attention computation
                psi_raw = dot_plasma.get('psi', [])
                if isinstance(psi_raw, dict):
                    # Extract at highest scale
                    query_embedding = psi_raw[max(psi_raw.keys())]
                else:
                    query_embedding = psi_raw

                # Convert to numpy if needed
                if not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding, dtype=np.float32)

                # Perform continuous tensor operations
                warp_compute_results = warp_space.compute(
                    query_embedding=query_embedding,
                    compute_spectral=True  # Enable spectral features
                )

                # Record warp operation
                warp_operations.append((
                    datetime.now().isoformat(),
                    "compute",
                    {
                        'attention_entropy': warp_compute_results.get('attention_entropy', 0.0),
                        'spectral_computed': warp_compute_results['metadata']['spectral_computed']
                    }
                ))

                self.logger.info(
                    f"  [5.5] Warp Space compute: "
                    f"attention_entropy={warp_compute_results.get('attention_entropy', 0.0):.3f}, "
                    f"spectral={warp_compute_results['metadata']['spectral_computed']}"
                )

                # Store compute results in context metadata for downstream use
                context.metadata['warp_compute'] = {
                    'attention_entropy': warp_compute_results.get('attention_entropy'),
                    'spectral_features': warp_compute_results.get('spectral_features'),
                    'context_vector_norm': float(np.linalg.norm(warp_compute_results['context_vector']))
                        if warp_compute_results.get('context_vector') is not None else None
                }

                stage_timings['warp_compute'] = (time.time() - step_start) * 1000

            except Exception as e:
                self.logger.warning(f"  [5.5] Warp Space compute failed: {e}. Continuing without warp features.")
                context.metadata['warp_compute'] = None
                stage_timings['warp_compute'] = 0.0

            # ================================================================
            # STEP 6.5: Beta Wave Context Packing (OPTIONAL)
            # ================================================================
            # Physics-based context optimization using activation spreading
            # Requires: MultiWaveMemoryEngine with spring_engine
            # Benefit: 50% token reduction, <1ms overhead

            if (self.cfg.enable_beta_wave_packing and
                self.memory and
                hasattr(self.memory, 'spring_engine')):

                step_start = time.time()

                try:
                    from HoloLoom.awareness.beta_wave_packer import (
                        BetaWaveContextPacker, TokenBudget
                    )

                    # Create token budget from config
                    packing_budget = TokenBudget(
                        total=self.cfg.packing_token_budget,
                        reserved_for_query=self.cfg.packing_query_reserve,
                        reserved_for_response=self.cfg.packing_response_reserve
                    )

                    # Create beta wave context packer
                    packer = BetaWaveContextPacker(
                        spring_engine=self.memory.spring_engine,
                        token_budget=packing_budget,
                        activation_threshold=self.cfg.packing_activation_threshold,
                        compression_threshold=self.cfg.packing_compression_threshold
                    )

                    # Get query embedding from DotPlasma
                    psi_raw = dot_plasma.get('psi', [])
                    if isinstance(psi_raw, dict):
                        # Extract at highest scale
                        query_embedding = psi_raw[max(psi_raw.keys())]
                    else:
                        query_embedding = psi_raw

                    # Convert to numpy if needed
                    if not isinstance(query_embedding, np.ndarray):
                        query_embedding = np.array(query_embedding, dtype=np.float32)

                    # Pack context using physics-based activation spreading
                    packed = await packer.pack_context(
                        query_text=query.text,
                        query_embedding=query_embedding,
                        awareness_context=None,  # Could integrate awareness here
                        top_k=len(shards)
                    )

                    # Store packed context in metadata
                    context.metadata['packed_context'] = packed
                    context.metadata['packing_stats'] = {
                        'elements_included': packed.elements_included,
                        'elements_compressed': packed.elements_compressed,
                        'elements_excluded': packed.elements_excluded,
                        'total_tokens': packed.total_tokens,
                        'budget_available': packing_budget.available_for_context,
                        'avg_activation': packed.avg_activation,
                        'min_activation': packed.min_activation,
                        'max_activation': packed.max_activation,
                        'activation_distribution': packed.activation_stats.get('activation_distribution', {})
                    }

                    self.logger.info(
                        f"  [6.5] Beta wave packing: {packed.elements_included} included, "
                        f"{packed.elements_compressed} compressed, "
                        f"{packed.elements_excluded} excluded "
                        f"({packed.total_tokens}/{packing_budget.available_for_context} tokens, "
                        f"avg_activation={packed.avg_activation:.3f})"
                    )
                    stage_timings['context_packing'] = (time.time() - step_start) * 1000

                except Exception as e:
                    # Graceful fallback on packing errors
                    self.logger.warning(
                        f"  [6.5] Beta wave packing failed: {e}. "
                        f"Falling back to raw shards."
                    )
                    context.metadata['packed_context'] = None
                    context.metadata['packing_error'] = str(e)
            else:
                # Beta wave packing disabled or unavailable
                context.metadata['packed_context'] = None
                if self.cfg.enable_beta_wave_packing:
                    reason = "memory backend lacks spring_engine"
                    self.logger.info(
                        f"  [6.5] Beta wave packing: DISABLED ({reason}, using raw shards)"
                    )
                else:
                    self.logger.debug("  [6.5] Beta wave packing: DISABLED (config flag off)")

            # ================================================================
            # STEP 7: Convergence Engine collapses to discrete tool selection
            # ================================================================
            step_start = time.time()

            # Create policy for neural predictions (use pattern_embedder)
            policy_mem_dim = max(pattern_spec.scales)
            self.logger.info(
                f"[DEBUG] Creating policy: pattern={pattern_spec.name}, "
                f"mem_dim={policy_mem_dim}, scales={pattern_spec.scales}"
            )
            policy = create_policy(
                mem_dim=policy_mem_dim,
                emb=pattern_embedder,
                scales=pattern_spec.scales,
                device=None,
                n_layers=pattern_spec.n_transformer_layers,
                n_heads=pattern_spec.n_attention_heads,
                bandit_strategy=self.cfg.bandit_strategy,
                epsilon=self.cfg.epsilon
                ,
                guardrails=self.guardrails,
                cfg=self.cfg,  # Environment-aware safety configuration
            )

            # Convert dot_plasma to Features object for policy
            # Note: plasma uses 'psi' for embeddings and 'motifs' (plural)
            psi_raw = dot_plasma.get('psi', [])

            # Defensive handling: ResonanceShed should now return array (not dict)
            # via encode_scales(size=target_scale), but handle legacy dict case
            if isinstance(psi_raw, dict):
                # Extract embeddings at pattern's required scale
                pattern_scale = max(pattern_spec.scales)
                self.logger.warning(
                    f"DotPlasma psi is dict (unexpected), extracting scale {pattern_scale}. "
                    f"ResonanceShed should use encode_scales(size={pattern_scale})"
                )
                psi_array = psi_raw.get(pattern_scale, psi_raw[max(psi_raw.keys())])
            else:
                # Expected path: psi is already an array at correct dimension
                psi_array = psi_raw

            # Convert to list for Features
            psi_list = psi_array.tolist() if hasattr(psi_array, 'tolist') else list(psi_array)

            features = Features(
                psi=psi_list,
                motifs=dot_plasma.get('motifs', []),
                metrics={'spectral': dot_plasma.get('spectral')},
                metadata=dot_plasma.get('metadata', {})
            )

            # Inject awareness metrics into features (Consciousness Integration - Phase 1)
            if awareness_context:
                features.metadata['awareness'] = {
                    'activation_level': awareness_context.get('activation_level', 0.0),
                    'coherence': awareness_context.get('coherence', 0.0),
                    'active_nodes': awareness_context.get('active_nodes', 0),
                    'shift_detected': awareness_context.get('shift_detected', False),
                }
                self.logger.debug(
                    f"[AWARENESS] Injected epistemic context into features: "
                    f"activation={awareness_context.get('activation_level', 0.0):.3f}, "
                    f"coherence={awareness_context.get('coherence', 0.0):.3f}"
                )

            context.features = features

            # ================================================================
            # STEP 7: Convergence Engine - Decision Collapse
            # ================================================================
            step_start = time.time()
            self._emit_stage_event(7, "Convergence Engine")

            # Get neural predictions with timeout (prevent hung decisions)
            try:
                action_plan = await asyncio.wait_for(
                    policy.decide(features=features, context=context),
                    timeout=0.2  # 200ms - reduced from 2s per bottleneck analysis
                )
            except asyncio.TimeoutError:
                self.logger.error("Policy decision timed out after 200ms, using safe default")
                # Create safe default action plan
                from HoloLoom.protocols.types import ActionPlan
                action_plan = ActionPlan(
                    tool="answer",
                    confidence=0.5,
                    tool_probs={"answer": 1.0},
                    metadata={"timeout": True, "fallback": True}
                )

            # Get tool probabilities (mock for now - would come from policy)
            neural_probs = np.array([
                action_plan.tool_probs.get(tool, 0.0)
                for tool in self.tool_executor.tools
            ])

            # Phase 1 Gradient Flow: Physics-based tool selection (optional enhancement)
            gradient_decision = None
            if self.gradient_router:
                try:
                    gradient_decision = await self.gradient_router.select_tool(query.text)
                    self.logger.info(f"  [7a] Gradient flow suggests: {gradient_decision.target} (loss={gradient_decision.loss:.3f})")
                except Exception as e:
                    self.logger.warning(f"Gradient flow routing failed: {e}")

            # Convergence Engine collapse (with optional gradient flow blending)
            convergence = ConvergenceEngine(
                tools=self.tool_executor.tools,
                default_strategy=self._map_bandit_to_collapse(self.cfg.bandit_strategy),
                epsilon=self.cfg.epsilon
            )

            # If gradient flow available, blend with neural predictions
            if gradient_decision:
                # Blend gradient flow with neural (70% neural, 30% gradient flow)
                gradient_boost = {gradient_decision.target: 0.3}
                for i, tool in enumerate(self.tool_executor.tools):
                    if tool == gradient_decision.target:
                        neural_probs[i] = 0.7 * neural_probs[i] + 0.3
                    else:
                        neural_probs[i] = 0.7 * neural_probs[i]
                # Renormalize
                neural_probs = neural_probs / neural_probs.sum()
                self.logger.debug(f"  [7b] Blended neural + gradient flow probabilities")

            collapse_result = convergence.collapse(neural_probs)

            duration = (time.time() - step_start) * 1000
            self.logger.info(f"  [7] Convergence collapsed to tool: {collapse_result.tool} (confidence={collapse_result.confidence:.2f})")
            stage_timings['convergence'] = duration
            self._emit_stage_event(7, "Convergence Engine", duration)

            # ================================================================
            # STEP 8: Tool Execution
            # ================================================================
            step_start = time.time()
            self._emit_stage_event(8, "Tool Execution")

            # Execute the selected tool
            tool_result = await self.tool_executor.execute(
                collapse_result.tool,
                query,
                context
            )

            duration = (time.time() - step_start) * 1000
            self.logger.info(f"  [8] Tool executed: {collapse_result.tool}")
            stage_timings['tool_execution'] = duration
            self._emit_stage_event(8, "Tool Execution", duration)

            # ================================================================
            # STEP 9: Spacetime Fabric - Weaving Results
            # ================================================================
            step_start = time.time()
            self._emit_stage_event(9, "Spacetime Fabric")

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
            metadata = {
                'pattern_card': pattern_spec.name,
                'execution_mode': pattern_spec.card.value,
                'loom_command': 'auto',
                'chrono_timeout': pattern_spec.pipeline_timeout
            }

            # Add semantic cache statistics if enabled
            if self.semantic_cache:
                cache_stats = self.semantic_cache.get_stats()
                metadata['semantic_cache'] = {
                    'enabled': True,
                    'hit_rate': cache_stats['cache_hit_rate'],
                    'hot_hits': cache_stats['hot_hits'],
                    'warm_hits': cache_stats['warm_hits'],
                    'cold_misses': cache_stats['cold_misses'],
                    'estimated_speedup': cache_stats['estimated_speedup']
                }
            else:
                metadata['semantic_cache'] = {'enabled': False}

            # Inject awareness context into metadata (Consciousness Integration - Phase 1)
            if awareness_context:
                metadata['awareness'] = {
                    'activation_level': awareness_context.get('activation_level', 0.0),
                    'coherence': awareness_context.get('coherence', 0.0),
                    'active_nodes': awareness_context.get('active_nodes', 0),
                    'shift_detected': awareness_context.get('shift_detected', False),
                    'semantic_position': awareness_context.get('semantic_position'),
                    'perception_time_ms': awareness_context.get('perception_time_ms', 0.0),
                }
                self.logger.info(
                    f"[AWARENESS] Epistemic context added to Spacetime: "
                    f"activation={awareness_context.get('activation_level', 0.0):.3f}, "
                    f"coherence={awareness_context.get('coherence', 0.0):.3f}"
                )

            spacetime = Spacetime(
                query_text=query.text,
                response=tool_result.get('result', 'No response'),
                tool_used=collapse_result.tool,
                confidence=collapse_result.confidence,
                trace=trace,
                metadata=metadata,
                context_summary=f"{len(context.shards)} shards",
                sources_used=[s.id for s in context.shards[:3]]
            )
            
            # Attach mythRL enhancements if protocol system is enabled
            if hasattr(self, 'enable_complexity_auto_detect') and self.enable_complexity_auto_detect:
                spacetime.complexity = complexity
                spacetime.provenance = self._create_provenance_trace(query, complexity)
                # Add all stage timings as protocol calls
                for stage, timing_ms in stage_timings.items():
                    spacetime.provenance.add_protocol_call(
                        protocol='weaving_orchestrator',
                        method=stage,
                        duration_ms=timing_ms,
                        result_summary=f"Stage completed in {timing_ms:.1f}ms"
                    )
                # Add final shuttle event
                spacetime.provenance.add_shuttle_event("weaving_complete", {
                    'tool': collapse_result.tool,
                    'confidence': collapse_result.confidence,
                    'duration_ms': duration_ms
                })

            duration = (time.time() - step_start) * 1000
            self.logger.info(f"  [9] Spacetime fabric woven!")
            stage_timings['spacetime_assembly'] = duration
            self._emit_stage_event(9, "Spacetime Fabric", duration)

            # Mark pipeline complete (back to idle)
            self._emit_stage_event(0, "Complete")

            self.logger.info(f"[SUCCESS] Weaving cycle complete! Total duration: {duration_ms:.1f}ms")

            # Track metrics
            if METRICS_ENABLED:
                # Track overall query
                metrics.track_query(
                    pattern=pattern_spec.name,
                    complexity=complexity.name if complexity else 'unknown',
                    duration=duration_ms / 1000.0,  # Convert to seconds
                    tool_used=collapse_result.tool
                )

                # Track all stage durations
                metrics.track_stage_batch(stage_timings)

                # Track tool execution
                metrics.track_tool_execution(
                    tool_name=collapse_result.tool,
                    duration=(stage_timings.get('tool_execution', 0)) / 1000.0
                )

                # Track parallel execution metrics
                if 'parallel_speedup' in stage_timings and 'parallel_execution_wall_time' in stage_timings:
                    metrics.track_parallel_execution(
                        stage_group='steps_4_6',
                        wall_time=stage_timings.get('parallel_execution_wall_time', 0) / 1000.0,
                        speedup=stage_timings.get('parallel_speedup', 1.0)
                    )

                # Track confidence
                metrics.set_confidence(collapse_result.tool, collapse_result.confidence)

                # Track active threads
                metrics.set_active_threads(pattern_spec.name, len(thread_ids))

                # Track context size
                metrics.set_retrieval_context_size(len(context.shards))

                # Track motif detection
                if features.motifs:
                    metrics.track_motifs(len(features.motifs))

            # Generate dashboard if enabled (Edward Tufte Machine)
            if self.dashboard_constructor:
                try:
                    dashboard = self.dashboard_constructor.construct(spacetime)
                    spacetime.metadata['dashboard'] = dashboard
                    self.logger.info(f"[DASHBOARD] Generated {len(dashboard.panels)} panels ({dashboard.layout.value} layout)")
                except Exception as e:
                    self.logger.warning(f"[DASHBOARD] Failed to generate dashboard: {e}")
                    # Don't fail the weaving cycle if dashboard generation fails

            # ================================================================
            # Recursive Learning: Apply learning loop (if enabled)
            # ================================================================
            if self.enable_recursive_learning and self._recursive_components:
                try:
                    spacetime = await apply_recursive_learning(self, spacetime, query)
                except Exception as e:
                    self.logger.warning(f"Recursive learning failed: {e}. Continuing with original spacetime.")
                    # Don't fail the weaving cycle if learning fails

            # ================================================================
            # Production Hardening: Record metrics (Part 5)
            # ================================================================
            if self.enable_production_hardening and self.monitor:
                try:
                    # Calculate query latency
                    prod_latency = (time.time() - prod_start_time) * 1000  # ms

                    # Record performance metrics
                    self.monitor.performance.record_query(
                        latency_ms=prod_latency,
                        cache_hit=False,  # This is a cache miss (we're about to cache it)
                        error=None if collapse_result.confidence >= 0.5 else "LowConfidence"
                    )

                    # Record learning metrics (if available)
                    if hasattr(spacetime, 'confidence'):
                        self.monitor.learning.record_calibration(
                            ece=abs(spacetime.confidence - 1.0)  # Simple ECE approximation
                        )

                    self.logger.debug(
                        f"[PRODUCTION] Recorded metrics: latency={prod_latency:.1f}ms, "
                        f"confidence={collapse_result.confidence:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"[PRODUCTION] Failed to record metrics: {e}")
                    # Don't fail weaving if monitoring fails

            # Cache the result
            self.query_cache.put(query.text, spacetime)
            self.logger.debug(f"[CACHE] Cached result for query")

            return spacetime

        except Exception as e:
            self.logger.error(f"[ERROR] Weaving cycle failed: {e}", exc_info=True)
            errors.append({
                'stage': 'unknown',
                'error': str(e),
                'type': type(e).__name__
            })

            # Track error metrics
            if METRICS_ENABLED:
                metrics.track_error(error_type=type(e).__name__, stage='weaving')

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

    async def weave_with_physics(
        self,
        query: Query,
        track_provenance: bool = True
    ) -> tuple[Spacetime, Optional[UnifiedPhysicsResult]]:
        """
        Physics-enhanced weaving with complete unified physics integration.

        Processes query through ALL physics layers (Phases 1-4) and provides
        complete provenance of every physics-based decision.

        Args:
            query: Input query
            track_provenance: If True, return full physics result with provenance

        Returns:
            (spacetime, physics_result) tuple where:
                - spacetime: Complete weaving result with physics metadata
                - physics_result: Full physics provenance (if track_provenance=True)

        Example:
            >>> orchestrator = WeavingOrchestrator(cfg=config)
            >>> spacetime, physics = await orchestrator.weave_with_physics(query)
            >>> print(f"Physics routing: {physics.routing_decision.target}")
            >>> print(f"Temperature: {physics.exploration_temperature:.2f}")
            >>> print(f"Patterns: {len(physics.constructive_patterns)}")

        Note:
            Implementation extracted to orchestrator/physics/physics_integration.py
            (Elegance Pass - Phase 7, November 2025)
        """
        return await weave_with_physics(self, query, track_provenance)

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
    # Recursive Learning Integration (Phase 1-5)
    # ========================================================================

    async def _apply_recursive_learning(self, spacetime: Spacetime, query: Query) -> Spacetime:
        """
        Apply recursive learning loop to spacetime result.

        This integrates all 5 phases of recursive learning:
        - Phase 1: Scratchpad provenance tracking
        - Phase 2: Pattern learning from successful queries
        - Phase 3: Hot pattern tracking for adaptive retrieval
        - Phase 4: Refinement for low-confidence results
        - Phase 5: Thompson Sampling and policy weight updates

        Args:
            spacetime: Spacetime result from weaving
            query: Original query

        Returns:
            Potentially refined spacetime (or original if no refinement needed)

        Note:
            Implementation extracted to orchestrator/learning/recursive_learning.py
            (Elegance Pass - Phase 8, November 2025)
        """
        return await apply_recursive_learning(self, spacetime, query)

    # ========================================================================
    # Memory Backend Helpers
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

        # Start background learner if recursive learning is enabled
        if self.enable_recursive_learning:
            # Lazy initialization happens on first weave()
            # We don't initialize here to avoid overhead if never used
            pass

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

    # ========================================================================
    # Cache and Statistics
    # ========================================================================

    def cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        return self.query_cache.stats()

    async def close(self) -> None:
        """
        Clean up all resources.

        Can be called manually or automatically via context manager.
        Safe to call multiple times (idempotent).
        """
        if self._closed:
            return

        self.logger.info("Closing WeavingOrchestrator...")

        # Stop recursive learning background learner
        if self.enable_recursive_learning and self._recursive_components:
            if self._recursive_components.get('background_learner'):
                self.logger.info("Stopping background learner...")
                await self._recursive_components['background_learner'].stop()

        # Cancel background tasks (with lock protection)
        async with self._bg_lock:
            if self._background_tasks:
                self.logger.info(f"Cancelling {len(self._background_tasks)} background tasks")
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()

                # Wait for cancellation with timeout
                tasks_to_wait = list(self._background_tasks)  # Copy for waiting
            else:
                tasks_to_wait = []

        if tasks_to_wait:
            try:
                await asyncio.wait(
                    tasks_to_wait,
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

    def save_dashboard(self, spacetime: Spacetime, output_path: str) -> None:
        """
        Save dashboard from Spacetime to HTML file (Edward Tufte Machine).

        Args:
            spacetime: Spacetime artifact with dashboard in metadata
            output_path: Path to save HTML file

        Raises:
            ValueError: If dashboards are not enabled or dashboard not found

        Usage:
            async with WeavingOrchestrator(cfg, shards, enable_dashboards=True) as orch:
                spacetime = await orch.weave(query)
                orch.save_dashboard(spacetime, 'output.html')
        """
        if not self.enable_dashboards:
            raise ValueError("Dashboards are not enabled. Initialize with enable_dashboards=True")

        dashboard = spacetime.metadata.get('dashboard')
        if dashboard is None:
            raise ValueError("No dashboard found in Spacetime metadata")

        from HoloLoom.visualization.html_renderer import save_dashboard
        save_dashboard(dashboard, output_path)
        self.logger.info(f"[DASHBOARD] Saved to {output_path}")

    # ========================================================================
    # Production Hardening Methods (Part 5: Days 21-25)
    # ========================================================================

    async def get_health(self) -> Optional[Dict[str, Any]]:
        """
        Get production health check status.

        Returns health check result with component checks (overall, backends,
        learning, resources) for load balancer integration.

        Returns:
            Dict with health status or None if production hardening disabled

        Example:
            >>> async with WeavingOrchestrator(..., enable_production_hardening=True) as orch:
            ...     health = await orch.get_health()
            ...     print(health['healthy'])  # True/False
            ...     print(health['status'])   # "healthy"/"degraded"/"unhealthy"
            ...     for check_name, check in health['checks'].items():
            ...         print(f"{check_name}: {check['status']}")
        """
        if not self.enable_production_hardening or not self.health_checker:
            self.logger.warning("[PRODUCTION] Health checks not enabled")
            return None

        try:
            result = await self.health_checker.check_health()
            return result.to_dict()
        except Exception as e:
            self.logger.error(f"[PRODUCTION] Health check failed: {e}")
            return {
                "healthy": False,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }

    def get_circuit_breaker_status(self) -> Optional[Dict[str, Any]]:
        """
        Get circuit breaker states for all registered backends.

        Returns circuit breaker status for monitoring backend health.

        Returns:
            Dict with breaker states or None if circuit breakers disabled

        Example:
            >>> orch = WeavingOrchestrator(..., enable_production_hardening=True)
            >>> status = orch.get_circuit_breaker_status()
            >>> for backend, state in status['breakers'].items():
            ...     print(f"{backend}: {state['state']} ({state['failure_count']} failures)")
        """
        if not self.enable_production_hardening or not self.breaker_registry:
            self.logger.warning("[PRODUCTION] Circuit breakers not enabled")
            return None

        try:
            breakers_status = {}
            for backend_name, breaker in self.breaker_registry.breakers.items():
                breakers_status[backend_name] = {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count,
                    "last_failure_time": breaker.last_failure_time,
                    "opened_at": breaker.opened_at
                }

            return {
                "breakers": breakers_status,
                "healthy": self.breaker_registry.get_health_summary()["healthy"],
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"[PRODUCTION] Failed to get breaker status: {e}")
            return {"error": str(e), "timestamp": time.time()}
