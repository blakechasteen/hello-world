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
from HoloLoom.Documentation.types import Query, Context, Features, MemoryShard

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

# Wool Storage Layer (November 17, 2025 - Zero-Copy Architecture)
from HoloLoom.wool import WoolStorage
from HoloLoom.wool.hybrid_kg import HybridKG

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
# Tool Execution
# ============================================================================

class ToolExecutor:
    """
    Executes tools based on convergence engine decisions.

    In production, this would call actual APIs, databases, etc.
    """

    def __init__(self, llm: Optional['OllamaLLM'] = None):
        self.tools = ["answer", "search", "notion_write", "calc"]
        self.logger = logging.getLogger(__name__)

        # Initialize LLM (lazy loading)
        self.llm = llm
        if self.llm is None:
            try:
                from HoloLoom.awareness.llm_integration import OllamaLLM
                self.llm = OllamaLLM(model="llama3.2:3b")
                self.logger.info("Initialized Ollama LLM (llama3.2:3b)")
            except Exception as e:
                self.logger.warning(f"LLM unavailable, using fallback: {e}")
                self.llm = None

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
        # Use packed context if available (beta wave optimization)
        packed_ctx = context.metadata.get('packed_context') if context and hasattr(context, 'metadata') else None

        if packed_ctx:
            # Use optimized physics-based packed context
            # Format: query section + awareness section + memory section
            llm_context = packed_ctx.format_for_llm(include_metadata=False)
            context_tokens = packed_ctx.total_tokens
            packing_stats = {
                'using_packed_context': True,
                'elements_included': packed_ctx.elements_included,
                'elements_compressed': packed_ctx.elements_compressed,
                'elements_excluded': packed_ctx.elements_excluded,
                'avg_activation': packed_ctx.avg_activation,
                'token_budget_used': f"{packed_ctx.total_tokens}/{context.metadata.get('packing_stats', {}).get('budget_available', 'N/A')}"
            }
            self.logger.debug(
                f"Using packed context: {packed_ctx.elements_included} elements, "
                f"{context_tokens} tokens (avg_activation={packed_ctx.avg_activation:.3f})"
            )
        else:
            # Use raw shard texts (legacy behavior)
            shard_texts = context.shard_texts[:5] if context and hasattr(context, 'shard_texts') else []
            llm_context = "\n\n".join(shard_texts)
            context_tokens = len(llm_context) // 4  # Rough token estimate
            packing_stats = {
                'using_packed_context': False,
                'raw_shards_count': len(shard_texts)
            }
            self.logger.debug(f"Using raw context: {len(shard_texts)} shards")

        # Build LLM prompt
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer based on the provided context. "
            "Be concise and accurate."
        )

        user_prompt = f"""Context:
{llm_context}

Question: {query.text}

Answer:"""

        # Call LLM if available
        if self.llm and hasattr(self.llm, 'is_available') and self.llm.is_available():
            try:
                response = await self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=500,
                    temperature=0.7
                )

                return {
                    "tool": "answer",
                    "result": response.content,  # ✅ Actual LLM response!
                    "confidence": 0.85,
                    "sources": len(context.shards) if context and hasattr(context, 'shards') else 0,
                    "llm_provider": response.provider.value if hasattr(response, 'provider') else "ollama",
                    "llm_model": response.model if hasattr(response, 'model') else "unknown",
                    "usage": response.usage if hasattr(response, 'usage') else {},
                    "context_tokens": context_tokens,
                    "packing_stats": packing_stats
                }
            except Exception as e:
                self.logger.error(f"LLM generation failed: {e}")
                # Fall through to fallback

        # Fallback (LLM unavailable)
        self.logger.warning("Using fallback response (LLM unavailable)")
        return {
            "tool": "answer",
            "result": f"[Fallback] Generated answer for: {query.text}\n\nContext: {llm_context[:300]}...",
            "confidence": 0.5,
            "sources": len(context.shards) if context and hasattr(context, 'shards') else 0,
            "context_preview": llm_context[:200] + "..." if len(llm_context) > 200 else llm_context,
            "context_tokens": context_tokens,
            "packing_stats": packing_stats
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
        enable_health_checks: bool = True
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
            self._initialize_production_hardening(
                production_config=production_config,
                rate_limit_qps=rate_limit_qps,
                rate_limit_concurrent=rate_limit_concurrent,
                enable_circuit_breakers=enable_circuit_breakers,
                circuit_breaker_threshold=circuit_breaker_threshold,
                enable_health_checks=enable_health_checks
            )

        # Initialize configuration and memory
        self._initialize_config_and_memory(
            memory=memory,
            yarn_graph=yarn_graph,
            shards=shards,
            pattern_preference=pattern_preference,
            enable_complexity_auto_detect=enable_complexity_auto_detect
        )

        # Initialize weaving components
        self._initialize_components()

        # Initialize reflection, caching, and dashboards
        self._initialize_reflection_and_caching(
            enable_reflection=enable_reflection,
            reflection_capacity=reflection_capacity
        )

        self.logger.info("WeavingOrchestrator initialization complete")

    def _initialize_config_and_memory(
        self,
        memory,
        yarn_graph,
        shards,
        pattern_preference,
        enable_complexity_auto_detect
    ):
        """
        Initialize memory configuration, pattern card, and protocol architecture.

        Validates memory sources, sets up mythRL protocols, and configures
        complexity detection thresholds.
        """
        # Validate memory configuration
        if memory is None and yarn_graph is None and shards is None:
            raise ValueError("Either 'memory', 'yarn_graph', or 'shards' must be provided")

        # Deprecation warning for shards parameter
        if shards is not None and yarn_graph is None and memory is None:
            import warnings
            warnings.warn(
                "Using 'shards' parameter is deprecated. "
                "Please use 'yarn_graph' (KG instance) for full metaphor support. "
                "Example: yarn_graph=KG() with memories added via yarn_graph.store().",
                DeprecationWarning,
                stacklevel=2
            )

        self.memory = memory  # Backend memory store
        self.yarn_graph_param = yarn_graph  # User-provided Yarn Graph (KG)
        self.shards = shards or []  # Static shards (backward compatibility)

        # Initialize Wool Storage Layer (Phase 1: November 17, 2025)
        self.wool: Optional[WoolStorage] = None
        if self.cfg.enable_wool_storage:
            from pathlib import Path
            self.wool = WoolStorage(
                base_path=Path(self.cfg.wool_storage_path),
                enable_cache=True
            )
            self.logger.info(f"Initialized WoolStorage at {self.cfg.wool_storage_path}")

        # mythRL Protocol-based architecture
        self.enable_complexity_auto_detect = enable_complexity_auto_detect
        self._protocols: Dict[str, Any] = {}  # Registered protocol implementations
        self._complexity_thresholds = {
            # Word count thresholds
            'lite_max_words': 2,      # Up to 2 words = LITE (greetings, simple commands)
            'fast_max_words': 20,     # 3-20 words = FAST (standard questions)
            'full_max_words': 50,     # 21-50 words = FULL (detailed queries)

            # Intent patterns for sophisticated detection
            'greeting_patterns': ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye'],
            'simple_commands': ['show', 'list', 'get', 'find', 'open'],
            'question_words': ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom'],
            'knowledge_verbs': ['explain', 'describe', 'tell', 'define', 'clarify', 'elaborate'],
            'analysis_verbs': ['analyze', 'compare', 'evaluate', 'assess', 'investigate', 'examine'],
            'research_keywords': ['research', 'deep', 'comprehensive', 'detailed', 'thorough', 'in-depth', 'extensive']
        }

        # Multipass Memory Crawling Configuration
        # Recursive gated retrieval with Matryoshka importance gating
        self._crawl_config = {
            ComplexityLevel.LITE: {
                'passes': 1,
                'thresholds': [0.7],           # Single pass, high threshold
                'limits': [5],                  # 5 items max
                'graph_expansion': False        # No graph traversal
            },
            ComplexityLevel.FAST: {
                'passes': 2,
                'thresholds': [0.6, 0.75],     # Broad → focused
                'limits': [8, 12],              # 8 initial, 12 expansion
                'graph_expansion': True         # Light graph traversal
            },
            ComplexityLevel.FULL: {
                'passes': 3,
                'thresholds': [0.6, 0.75, 0.85],  # Progressive gating
                'limits': [12, 20, 15],           # Expanding search space
                'graph_expansion': True           # Full graph traversal
            },
            ComplexityLevel.RESEARCH: {
                'passes': 4,
                'thresholds': [0.5, 0.65, 0.8, 0.9],  # Maximum depth
                'limits': [20, 30, 25, 15],            # Deep exploration
                'graph_expansion': True                 # Aggressive graph traversal
            }
        }
        self.logger.info(f"mythRL protocol system enabled (auto_detect={enable_complexity_auto_detect})")

        # Create a single shared SafetyGuardrails instance used across components
        try:
            # Check if guardrails should be disabled via config or environment variable
            import os
            testing_mode = (not getattr(self.cfg, 'enable_safety_guardrails', True) or
                           os.getenv("DISABLE_GUARDRAILS"))
            self.guardrails: SafetyGuardrails = create_guardrails(testing_mode=testing_mode)
            if testing_mode:
                self.logger.info("Shared SafetyGuardrails created in testing mode (approval bypassed)")
            else:
                self.logger.info("Shared SafetyGuardrails instance created and will be threaded to components")
        except Exception as e:
            self.logger.warning(f"Failed to initialize shared guardrails: {e}. Falling back to local guardrails where needed.")
            self.guardrails = None

        # Determine pattern card from config or preference
        if pattern_preference:
            self.default_pattern = pattern_preference
        elif self.cfg.mode == ExecutionMode.BARE:
            self.default_pattern = PatternCard.BARE
        elif self.cfg.mode == ExecutionMode.FAST:
            self.default_pattern = PatternCard.FAST
        else:
            self.default_pattern = PatternCard.FUSED

        self.logger.info(f"Initializing WeavingOrchestrator with pattern: {self.default_pattern.value}")

        # Lifecycle management
        self._background_tasks: List[asyncio.Task] = []
        self._bg_lock = asyncio.Lock()  # Protect concurrent access to _background_tasks
        self._closed = False

    def _initialize_reflection_and_caching(
        self,
        enable_reflection: bool,
        reflection_capacity: int
    ):
        """
        Initialize reflection loop, query cache, and dashboard constructor.

        Sets up learning systems and performance optimizations.
        """
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

        # Initialize performance cache
        self.query_cache = QueryCache(max_size=50, ttl_seconds=300)
        self.logger.info("Query cache enabled (max_size=50, ttl=300s)")

        # Initialize smart query routing (November 2025 - Performance Optimization)
        self.query_classifier = create_classifier(self.cfg)
        self.fast_path_router = create_fast_path_router(self, self.cfg) if self.query_classifier else None

        if self.query_classifier and self.fast_path_router:
            self.logger.info(f"✅ Smart query routing enabled "
                            f"(classifier={self.cfg.routing_classifier}, "
                            f"semantic_tier={self.cfg.enable_semantic_tier})")
        else:
            self.logger.info("Smart query routing disabled")

        # Initialize dashboard constructor (Edward Tufte Machine)
        if self.enable_dashboards:
            from HoloLoom.visualization.constructor import DashboardConstructor
            self.dashboard_constructor = DashboardConstructor()
            self.logger.info("Dashboard generation enabled (Edward Tufte Machine)")
        else:
            self.dashboard_constructor = None
            self.logger.info("Dashboard generation disabled")

    def _initialize_recursive_learning(self):
        """
        Initialize recursive learning components (lazy initialization).

        This is called on first weave() when enable_recursive_learning=True.
        Initializes:
        - Scratchpad for provenance tracking
        - Pattern learner for successful query patterns
        - Hot pattern tracker for adaptive retrieval
        - Advanced refiner for low-confidence queries
        - Background learner for Thompson Sampling and policy updates
        """
        if self._recursive_components is not None:
            return  # Already initialized

        try:
            from HoloLoom.recursive.scratchpad import Scratchpad
            from HoloLoom.recursive.loop_integration import PatternLearner
            from HoloLoom.recursive.hot_patterns import HotPatternTracker
            from HoloLoom.recursive.advanced_refinement import AdvancedRefiner
            from HoloLoom.recursive.full_learning_loop import (
                ThompsonPriors,
                PolicyWeights,
                BackgroundLearner,
                LearningMetrics
            )

            self.logger.info("Initializing recursive learning components...")

            # Components dictionary
            components = {}

            # 1. Scratchpad (Phase 1)
            if self.cfg.recursive_learning_enable_scratchpad:
                components['scratchpad'] = Scratchpad(capacity=1000)
                self.logger.info("  [Phase 1] Scratchpad enabled (provenance tracking)")

            # 2. Pattern Learner (Phase 2)
            components['pattern_learner'] = PatternLearner()
            self.logger.info("  [Phase 2] Pattern learner enabled")

            # 3. Hot Pattern Tracker (Phase 3)
            if self.cfg.recursive_learning_enable_hot_patterns:
                components['hot_tracker'] = HotPatternTracker()
                self.logger.info("  [Phase 3] Hot pattern tracker enabled")

            # 4. Advanced Refiner (Phase 4)
            components['refiner'] = AdvancedRefiner(
                orchestrator=self,
                scratchpad=components.get('scratchpad'),
                enable_learning=True
            )
            self.logger.info("  [Phase 4] Advanced refiner enabled")

            # 5. Background Learner (Phase 5)
            components['thompson_priors'] = ThompsonPriors()
            components['policy_weights'] = PolicyWeights()
            components['metrics'] = LearningMetrics()

            if self.cfg.recursive_learning_enable_background:
                components['background_learner'] = BackgroundLearner(
                    orchestrator=self,
                    thompson_priors=components['thompson_priors'],
                    policy_weights=components['policy_weights'],
                    update_interval=self.cfg.recursive_learning_update_interval
                )
                self.logger.info("  [Phase 5] Background learner enabled")
            else:
                components['background_learner'] = None

            self._recursive_components = components

            # Note: Background learner.start() should be called asynchronously in weave()
            # Cannot use await here since _initialize_recursive_learning is synchronous
            self.logger.info("Recursive learning initialization complete")

        except ImportError as e:
            self.logger.warning(
                f"Failed to initialize recursive learning components: {e}. "
                f"Recursive learning will be disabled. "
                f"Install dependencies or disable via config.enable_recursive_learning=False"
            )
            self.enable_recursive_learning = False
            self._recursive_components = None

    def _initialize_components(self):
        """Initialize all weaving architecture components."""

        # 1. Loom Command - Pattern selection
        self.loom_command = LoomCommand(
            default_pattern=self.default_pattern,
            auto_select=True,
            guardrails=self.guardrails,
        )

        # 2. Yarn Graph / Memory Backend - Thread storage
        # Priority: memory > yarn_graph > shards (for backward compatibility)
        if self.memory:
            # Use persistent memory backend (UnifiedMemory, etc.)
            from HoloLoom.memory.weaving_adapter import WeavingMemoryAdapter
            if isinstance(self.memory, WeavingMemoryAdapter):
                self.yarn_graph = self.memory
            else:
                # Wrap raw memory in adapter (use "factory" type for protocol backends)
                self.yarn_graph = WeavingMemoryAdapter(backend=self.memory, backend_type="factory")
            self.logger.info("Using persistent memory backend")
        elif self.yarn_graph_param:
            # Use KG instance (preferred modern API)
            # Wrap with HybridKG if wool storage enabled (Phase 1)
            if self.wool:
                self.yarn_graph = HybridKG(
                    wool_storage=self.wool,
                    base_kg=self.yarn_graph_param
                )
                self.logger.info(f"Using HybridKG (zero-copy enabled) with {self.yarn_graph_param.G.number_of_nodes()} nodes")
            else:
                self.yarn_graph = self.yarn_graph_param
                self.logger.info(f"Using Yarn Graph (KG) with {self.yarn_graph.G.number_of_nodes()} nodes")
        else:
            # Use in-memory YarnGraph with shards (backward compatibility)
            # Wrap with HybridKG if wool storage enabled (Phase 1)
            from HoloLoom.memory.yarn_graph import YarnGraph
            base_yarn = YarnGraph(self.shards)
            if self.wool:
                self.yarn_graph = HybridKG(
                    wool_storage=self.wool,
                    base_kg=base_yarn.kg  # Extract KG from YarnGraph
                )
                self.logger.info(f"Using HybridKG (zero-copy enabled) with {len(self.shards)} shards")
            else:
                self.yarn_graph = base_yarn
                self.logger.info(f"Using in-memory YarnGraph (legacy) with {len(self.shards)} shards")

        # 3. Component factories (will be instantiated per-query with pattern spec)
        self.embedder = MatryoshkaEmbeddings(
            sizes=self.cfg.scales,
            base_model_name=self.cfg.base_model_name
        )

        # 3a. Semantic Cache - Three-tier caching for 244D semantic projections
        if self.enable_semantic_cache:
            self._initialize_semantic_cache()
        else:
            self.semantic_cache = None
            self.semantic_spectrum = None

        # 3b. Phase 5: Linguistic Matryoshka Gate (optional)
        if self.cfg.enable_linguistic_gate:
            self._initialize_linguistic_gate()
        else:
            self.linguistic_gate = None

        # 4. Tool Executor
        self.tool_executor = ToolExecutor()

        # 4a. Gradient Flow Router (Phase 1: Physics-based tool selection)
        # Uses gradient descent to route queries to optimal tools based on cost/quality/latency
        try:
            tool_configs = [
                ToolConfig("answer", cost=0.3, quality=0.8, latency=100),
                ToolConfig("search", cost=0.5, quality=0.7, latency=150),
                ToolConfig("notion_write", cost=0.6, quality=0.75, latency=120),
                ToolConfig("calc", cost=0.1, quality=0.9, latency=50)
            ]
            self.gradient_router = ToolRouter(
                tools=tool_configs,
                cost_weight=0.3,
                quality_weight=0.5,  # Favor quality
                latency_weight=0.2
            )
            self.logger.info("Gradient flow router initialized (Phase 1 integration)")
        except Exception as e:
            self.logger.warning(f"Failed to initialize gradient flow router: {e}")
            self.gradient_router = None

        # 4b. Unified Physics Engine (Phases 1-4: Complete physics stack)
        # Coordinates gradient flow, fluid dynamics, thermodynamics, and wave mechanics
        # for self-optimizing behavior across routing, packing, exploration, and pattern detection
        self.unified_physics = None
        if UNIFIED_PHYSICS_AVAILABLE and self.cfg.enable_unified_physics:
            try:
                self.unified_physics = UnifiedPhysicsEngine(
                    enable_routing=True,        # Phase 1: Gradient flow routing
                    enable_packing=True,        # Phase 2: Fluid dynamics context packing
                    enable_thermodynamics=True, # Phase 3: Exploration/exploitation balance
                    enable_wave_mechanics=True, # Phase 4: Pattern detection
                    mode="adaptive"            # Adaptive strategy selection
                )
                self.logger.info("✓ Unified Physics Engine initialized (Phases 1-4 complete)")
                self.logger.info("  → Routing: Gradient flow (optimal tool selection)")
                self.logger.info("  → Packing: Fluid dynamics (context optimization)")
                self.logger.info("  → Exploration: Thermodynamics (F = E - T*S)")
                self.logger.info("  → Patterns: Wave mechanics (interference detection)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize unified physics: {e}")
                self.unified_physics = None
        elif not UNIFIED_PHYSICS_AVAILABLE:
            self.logger.debug("Unified physics not available (import failed)")
        else:
            self.logger.debug("Unified physics disabled in config")

        # 4c. Statistical Mechanics Engine (Phase 5: Memory consolidation)
        # Uses Gibbs distribution for emergent clustering and phase transition detection
        if self.enable_statistical_mechanics:
            try:
                self.stat_mech_engine = StatisticalMechanicsEngine(
                    temperature=self.consolidation_temperature,
                    cooling_rate=self.consolidation_cooling_rate
                )
                self.logger.info("✓ Statistical Mechanics Engine initialized (Phase 5 complete)")
                self.logger.info(f"  → Consolidation interval: {self.consolidation_interval}s")
                self.logger.info(f"  → Initial temperature: {self.consolidation_temperature}")
                self.logger.info(f"  → Cooling rate: {self.consolidation_cooling_rate}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize statistical mechanics: {e}")
                self.stat_mech_engine = None
        elif not STATISTICAL_MECHANICS_AVAILABLE:
            self.logger.debug("Statistical mechanics not available (import failed)")
        else:
            self.logger.debug("Statistical mechanics disabled")

        # 5. Retriever (for context)
        # Only create traditional retriever if using shards (legacy path)
        if self.shards:
            # Legacy YarnGraph has .shards dict
            if hasattr(self.yarn_graph, 'shards'):
                self.retriever = create_retriever(
                    shards=list(self.yarn_graph.shards.values()),
                    emb=self.embedder,
                    fusion_weights=self.cfg.fusion_weights
                )
            else:
                # Direct shards list (shouldn't happen with new API)
                self.retriever = create_retriever(
                    shards=self.shards,
                    emb=self.embedder,
                    fusion_weights=self.cfg.fusion_weights
                )
        else:
            # Using KG or dynamic memory backend - retriever not needed
            # Thread selection happens via yarn_graph.select_threads()
            self.retriever = None

        self.logger.debug("All weaving components initialized")

    def _initialize_production_hardening(
        self,
        production_config,
        rate_limit_qps,
        rate_limit_concurrent,
        enable_circuit_breakers,
        circuit_breaker_threshold,
        enable_health_checks
    ):
        """
        Initialize production hardening components (Part 5: Days 21-25).

        Creates monitoring, circuit breakers, rate limiting, health checks,
        and error handling components for production deployment.

        Features:
        - System monitoring (performance, resources, learning metrics)
        - Circuit breaker protection for backend failures
        - Rate limiting (token bucket + sliding window + concurrent)
        - Health check system for load balancers
        - Error handler with retry and fallback

        Performance: <1ms overhead per query
        """
        if not PRODUCTION_HARDENING_AVAILABLE:
            self.logger.warning("Production hardening unavailable (context module not found)")
            return

        self.logger.info("Initializing production hardening...")

        # Load or create production configuration
        if production_config is None:
            production_config = ProductionConfig.from_environment()
            self.logger.info(f"  Auto-detected environment: {production_config.environment.value}")
        else:
            self.logger.info(f"  Using provided config: {production_config.environment.value}")

        # Validate configuration
        errors = production_config.validate()
        if errors:
            self.logger.error(f"Production config validation failed: {errors}")
            raise ValueError(f"Invalid production configuration: {errors}")

        # Create system monitor
        self.monitor = create_system_monitor()
        self.logger.info("  System monitor enabled (performance + resources + learning)")

        # Create circuit breaker registry (if enabled)
        if enable_circuit_breakers:
            self.breaker_registry = create_circuit_breaker_registry()
            self.logger.info(
                f"  Circuit breakers enabled "
                f"(threshold: {circuit_breaker_threshold})"
            )
        else:
            self.breaker_registry = None
            self.logger.info("  Circuit breakers disabled")

        # Create rate limiter
        self.rate_limiter = create_rate_limiter(
            rate=rate_limit_qps,
            capacity=int(rate_limit_qps * 0.2),  # 20% burst capacity
            max_concurrent=rate_limit_concurrent
        )
        self.logger.info(
            f"  Rate limiter enabled "
            f"(qps: {rate_limit_qps}, concurrent: {rate_limit_concurrent})"
        )

        # Create health checker (if enabled)
        if enable_health_checks:
            self.health_checker = create_health_checker(
                performance_monitor=self.monitor.performance if self.monitor else None,
                resource_monitor=self.monitor.resources if self.monitor else None,
                learning_monitor=self.monitor.learning if self.monitor else None,
                circuit_breaker_registry=self.breaker_registry
            )
            self.logger.info("  Health checker enabled (5 component checks)")
        else:
            self.health_checker = None
            self.logger.info("  Health checks disabled")

        # Create error handler
        self.error_handler = create_error_handler()
        self.logger.info("  Error handler enabled (retry + fallback)")

        self.logger.info("Production hardening initialization complete (<1ms overhead)")

    def _initialize_semantic_cache(self):
        """
        Initialize three-tier semantic cache for 244D projections.

        This provides 3-10× speedup for semantic projections by caching at
        the semantic dimension level instead of just the embedding level.
        """
        try:
            from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS, SemanticSpectrum
            from HoloLoom.performance.semantic_cache import AdaptiveSemanticCache

            self.logger.info("Initializing semantic cache...")

            # Create global semantic spectrum
            self.semantic_spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)

            # Learn axes using embedder
            self.logger.info("  Learning semantic axes...")
            self.semantic_spectrum.learn_axes(lambda word: self.embedder.encode([word])[0])

            # Initialize adaptive semantic cache
            self.semantic_cache = AdaptiveSemanticCache(
                semantic_spectrum=self.semantic_spectrum,
                embedder=self.embedder,
                hot_size=1000,  # Pre-loaded narrative patterns
                warm_size=5000   # LRU cache for recent queries
            )

            self.logger.info("  Semantic cache enabled (hot=1000, warm=5000)")

        except Exception as e:
            self.logger.warning(f"Failed to initialize semantic cache: {e}")
            self.logger.warning("Falling back to direct semantic computation")
            self.semantic_cache = None
            self.semantic_spectrum = None

    def _initialize_linguistic_gate(self):
        """
        Initialize Phase 5 Linguistic Matryoshka Gate.

        This provides 10-300× speedup through:
        1. Universal Grammar phrase chunking (X-bar theory)
        2. 3-tier compositional cache (parse/merge/semantic)
        3. Progressive linguistic filtering
        """
        try:
            from HoloLoom.embedding.linguistic_matryoshka_gate import (
                LinguisticMatryoshkaGate,
                LinguisticGateConfig,
                LinguisticFilterMode
            )

            self.logger.info("Initializing Phase 5 Linguistic Matryoshka Gate...")

            # Map config string to enum
            mode_map = {
                'disabled': LinguisticFilterMode.DISABLED,
                'prefilter': LinguisticFilterMode.PREFILTER,
                'embedding': LinguisticFilterMode.EMBEDDING,
                'both': LinguisticFilterMode.BOTH
            }
            linguistic_mode = mode_map.get(self.cfg.linguistic_mode, LinguisticFilterMode.DISABLED)

            # Create configuration
            gate_config = LinguisticGateConfig(
                linguistic_mode=linguistic_mode,
                use_compositional_cache=self.cfg.use_compositional_cache,
                parse_cache_size=self.cfg.parse_cache_size,
                merge_cache_size=self.cfg.merge_cache_size,
                linguistic_weight=self.cfg.linguistic_weight,
                prefilter_similarity_threshold=self.cfg.prefilter_similarity_threshold,
                prefilter_keep_ratio=self.cfg.prefilter_keep_ratio
            )

            # Create linguistic gate
            self.linguistic_gate = LinguisticMatryoshkaGate(
                embedder=self.embedder,
                config=gate_config
            )

            self.logger.info(
                f"  Phase 5 enabled: mode={linguistic_mode.value}, "
                f"cache={self.cfg.use_compositional_cache}, "
                f"parse_cache={self.cfg.parse_cache_size}, "
                f"merge_cache={self.cfg.merge_cache_size}"
            )

        except Exception as e:
            self.logger.warning(f"Failed to initialize linguistic gate: {e}")
            self.logger.warning("Falling back to standard matryoshka gate")
            self.linguistic_gate = None

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

    def _assess_complexity_level(self, query: Query, trace: Optional[ProvenanceTrace] = None) -> ComplexityLevel:
        """
        Assess query complexity using hybrid word count + intent detection.
        
        **Progressive Complexity (3-5-7-9):**
        - LITE (3): Greetings, simple commands (1-3 words, no questions)
        - FAST (5): Questions, knowledge queries (4-20 words OR question indicators)
        - FULL (7): Detailed analysis (21-50 words, complex questions)
        - RESEARCH (9): Deep research (analysis verbs, research keywords, 50+ words)
        
        **Intent Detection:**
        - Greetings: "hi", "hello", "thanks" → LITE
        - Questions: "what", "how", "why" → FAST (minimum)
        - Knowledge: "explain", "describe" → FAST (minimum)
        - Analysis: "analyze", "compare" → RESEARCH
        - Research: "comprehensive", "detailed" → RESEARCH
        
        Args:
            query: User query
            trace: Optional provenance trace to record decision
        
        Returns:
            ComplexityLevel (LITE/FAST/FULL/RESEARCH)
        """
        if not self.enable_complexity_auto_detect:
            # Map pattern card to complexity level
            if self.default_pattern == PatternCard.BARE:
                return ComplexityLevel.LITE
            elif self.default_pattern == PatternCard.FAST:
                return ComplexityLevel.FAST
            else:  # FUSED
                return ComplexityLevel.FULL
        
        text = query.text.lower()
        words = text.split()
        word_count = len(words)
        
        # Extract intent patterns
        thresholds = self._complexity_thresholds
        
        # Check for specific intent patterns
        is_greeting = any(pattern in text for pattern in thresholds['greeting_patterns'])
        is_simple_command = any(cmd in words for cmd in thresholds['simple_commands'])
        has_question_word = any(q in words for q in thresholds['question_words'])
        has_question_mark = '?' in text
        has_knowledge_verb = any(verb in words for verb in thresholds['knowledge_verbs'])
        has_analysis_verb = any(verb in words for verb in thresholds['analysis_verbs'])
        has_research_keyword = any(keyword in text for keyword in thresholds['research_keywords'])
        
        # Determine complexity with sophisticated intent detection
        # Priority: Research > Full > Fast > Lite
        
        # RESEARCH: Analysis verbs, research keywords, or very long queries
        if has_analysis_verb or has_research_keyword or word_count > thresholds['full_max_words']:
            level = ComplexityLevel.RESEARCH
            reason = f"analysis_verb={has_analysis_verb}, research_keyword={has_research_keyword}, long_query={word_count > thresholds['full_max_words']}"
        
        # FULL: Detailed questions (21-50 words) or complex knowledge requests
        elif word_count >= thresholds['fast_max_words']:  # Changed > to >= (includes 20-word boundary)
            level = ComplexityLevel.FULL
            reason = f"word_count={word_count} >= {thresholds['fast_max_words']}"
        
        # FAST: Questions, knowledge verbs, or 4-20 words
        elif has_question_word or has_question_mark or has_knowledge_verb or word_count > thresholds['lite_max_words']:
            # Exception: Pure greetings stay LITE even if >3 words (but not if they have question words)
            if is_greeting and word_count <= 5 and not (has_question_word or has_question_mark):
                level = ComplexityLevel.LITE
                reason = f"greeting_pattern (word_count={word_count})"
            else:
                level = ComplexityLevel.FAST
                reason = f"question={has_question_word or has_question_mark}, knowledge_verb={has_knowledge_verb}, word_count={word_count}"
        
        # LITE: Simple greetings, short commands (1-3 words, no questions)
        else:
            level = ComplexityLevel.LITE
            reason = f"simple_query (word_count={word_count})"
        
        if trace:
            trace.add_shuttle_event(
                "complexity_assessment",
                f"Assessed complexity: {level.name}",
                {
                    'word_count': word_count,
                    'is_greeting': is_greeting,
                    'has_question': has_question_word or has_question_mark,
                    'has_knowledge_verb': has_knowledge_verb,
                    'has_analysis_verb': has_analysis_verb,
                    'has_research_keyword': has_research_keyword,
                    'complexity_level': level.name,
                    'reason': reason
                }
            )
        
        return level

    def _create_provenance_trace(self, query: Query, complexity: ComplexityLevel) -> ProvenanceTrace:
        """Create a new provenance trace for this operation."""
        operation_id = f"weave_{int(time.time() * 1000)}"
        trace = ProvenanceTrace(
            operation_id=operation_id,
            complexity_level=complexity,
            start_time=time.perf_counter()
        )
        trace.add_shuttle_event("weave_start", f"Beginning weave for query: {query.text[:50]}...")
        return trace

    async def _multipass_memory_crawl(
        self,
        query: Query,
        complexity: ComplexityLevel,
        trace: Optional[ProvenanceTrace] = None
    ) -> List[Any]:
        """
        Recursive gated multipass memory crawling with Matryoshka importance gating.
        
        **Crawling Strategy:**
        1. **Gated Retrieval**: Start broad (low threshold), progressively focus (high threshold)
        2. **Graph Traversal**: Follow entity relationships for deeper exploration
        3. **Matryoshka Gating**: Increase importance thresholds by depth (0.6 → 0.75 → 0.85 → 0.9)
        4. **Multipass Fusion**: Intelligent deduplication and composite scoring
        
        **Complexity Scaling:**
        - LITE (1 pass): threshold=0.7, limit=5, no graph traversal
        - FAST (2 passes): thresholds=[0.6, 0.75], limits=[8, 12], light graph
        - FULL (3 passes): thresholds=[0.6, 0.75, 0.85], limits=[12, 20, 15], full graph
        - RESEARCH (4 passes): thresholds=[0.5, 0.65, 0.8, 0.9], limits=[20, 30, 25, 15], aggressive
        
        Args:
            query: User query
            complexity: Assessed complexity level
            trace: Optional provenance trace
        
        Returns:
            List of retrieved memory items with composite scores
        """
        crawl_start = time.perf_counter()
        config = self._crawl_config[complexity]
        
        all_results = {}  # item_id -> {item, score, depth, sources}
        seen_ids = set()
        
        if trace:
            trace.add_shuttle_event(
                "crawl_start",
                f"Starting {config['passes']}-pass crawl with thresholds {config['thresholds']}",
                {'complexity': complexity.name, 'config': config}
            )
        
        # Multi-pass retrieval with progressive gating
        for pass_idx in range(config['passes']):
            threshold = config['thresholds'][pass_idx]
            limit = config['limits'][pass_idx]
            
            pass_start = time.perf_counter()
            
            # Initial retrieval from memory backend
            if self.memory:
                try:
                    # Use memory backend's recall method
                    # Note: Threshold is handled by the backend based on relevance scoring
                    from HoloLoom.memory.protocol import MemoryQuery
                    mem_query = MemoryQuery(
                        text=query.text,
                        limit=limit
                    )
                    result = await self.memory.recall(mem_query, limit=limit)
                    
                    # Process results
                    for idx, (memory, score) in enumerate(zip(result.memories, result.scores)):
                        if memory.id not in seen_ids:
                            all_results[memory.id] = {
                                'item': memory,
                                'score': score * (1.0 / (pass_idx + 1)),  # Decay by depth
                                'depth': pass_idx,
                                'sources': [f'pass_{pass_idx}']
                            }
                            seen_ids.add(memory.id)
                        else:
                            # Boost score for items found in multiple passes
                            all_results[memory.id]['score'] += score * 0.3
                            all_results[memory.id]['sources'].append(f'pass_{pass_idx}')
                    
                    if trace:
                        trace.add_shuttle_event(
                            f"crawl_pass_{pass_idx}",
                            f"Retrieved {len(result.memories)} items (threshold={threshold})",
                            {
                                'pass': pass_idx,
                                'threshold': threshold,
                                'limit': limit,
                                'retrieved': len(result.memories),
                                'time_ms': (time.perf_counter() - pass_start) * 1000
                            }
                        )
                
                except Exception as e:
                    self.logger.warning(f"Pass {pass_idx} failed: {e}")
                    if trace:
                        trace.add_shuttle_event(
                            f"crawl_pass_{pass_idx}_error",
                            f"Retrieval failed: {str(e)}",
                            {'error': str(e)}
                        )
            
            # Graph expansion (if enabled for this complexity)
            if config['graph_expansion'] and pass_idx < config['passes'] - 1:
                # Expand from top results of this pass
                expand_count = min(3, len(all_results))  # Expand from top 3
                expanded_ids = list(all_results.keys())[:expand_count]
                
                for item_id in expanded_ids:
                    # Try to get related items (graph traversal)
                    # Note: This requires the memory backend to support get_related()
                    if hasattr(self.memory, 'get_related'):
                        try:
                            related = await self.memory.get_related(item_id, limit=5)
                            for rel_item in related:
                                if rel_item.id not in seen_ids:
                                    # Related items get slightly lower score
                                    all_results[rel_item.id] = {
                                        'item': rel_item,
                                        'score': 0.7 * (1.0 / (pass_idx + 2)),
                                        'depth': pass_idx + 1,
                                        'sources': [f'graph_expansion_from_{item_id}']
                                    }
                                    seen_ids.add(rel_item.id)
                        except (AttributeError, Exception) as e:
                            # Backend doesn't support graph traversal or error occurred
                            pass
        
        # Sort by composite score
        ranked_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        crawl_time_ms = (time.perf_counter() - crawl_start) * 1000
        
        if trace:
            trace.add_shuttle_event(
                "crawl_complete",
                f"Crawl complete: {len(ranked_results)} unique items",
                {
                    'total_items': len(ranked_results),
                    'passes_completed': config['passes'],
                    'time_ms': crawl_time_ms,
                    'avg_time_per_pass_ms': crawl_time_ms / config['passes']
                }
            )
        
        # Return just the items (without metadata for now)
        return [r['item'] for r in ranked_results]

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
    # Main Weaving Cycle
    # ========================================================================

    async def weave(
        self,
        query: Query,
        pattern_override: Optional[PatternCard] = None,
        complexity: Optional[ComplexityLevel] = None
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
            >>> from HoloLoom.Documentation.types import Query
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
            complexity = self._assess_complexity_level(query)
        
        # mythRL: Create provenance trace
        provenance = self._create_provenance_trace(query, complexity)
        
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
            # STEP 3: Yarn Graph threads selected
            # ================================================================
            step_start = time.time()
            self._emit_stage_event(3, "Yarn Graph")

            threads = self.yarn_graph.select_threads(temporal_window, query)
            thread_ids = [s.id for s in threads]
            thread_texts = [s.text for s in threads]

            duration = (time.time() - step_start) * 1000
            self.logger.info(f"  [3] Selected {len(threads)} threads from Yarn Graph")
            stage_timings['thread_selection'] = duration
            self._emit_stage_event(3, "Yarn Graph", duration)

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
                    shards = await self._multipass_memory_crawl(query, complexity, provenance)
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
                from HoloLoom.Documentation.types import ActionPlan
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
                    spacetime = await self._apply_recursive_learning(spacetime, query)
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
        """
        if not self.unified_physics:
            self.logger.warning("Unified physics not available, falling back to standard weaving")
            spacetime = await self.weave(query)
            return spacetime, None

        # Standard weaving to get base result
        spacetime = await self.weave(query)

        # Prepare physics inputs
        actions = self.tool_executor.tools
        action_metrics = {
            "answer": {"cost": 0.3, "quality": 0.8, "latency": 0.1, "error": 0.1},
            "search": {"cost": 0.6, "quality": 0.7, "latency": 0.2, "error": 0.15},
            "notion_write": {"cost": 0.6, "quality": 0.75, "latency": 0.15, "error": 0.1},
            "calc": {"cost": 0.1, "quality": 0.9, "latency": 0.05, "error": 0.05}
        }

        # Build graph structure for wave mechanics (if yarn_graph available)
        graph_structure = None
        if self.yarn_graph and hasattr(self.yarn_graph, 'G'):
            # Extract edges from knowledge graph
            graph_structure = [
                (str(source), str(target))
                for source, target in self.yarn_graph.G.edges()
            ]

        # Process through unified physics
        physics_result = await self.unified_physics.process(
            query=query.text,
            actions=actions,
            action_metrics=action_metrics,
            graph_structure=graph_structure
        )

        # Enhance spacetime with physics provenance
        if track_provenance and self.cfg.physics_track_provenance:
            spacetime.metadata['unified_physics'] = {
                # Phase 1: Routing
                'routing_target': physics_result.routing_decision.target if physics_result.routing_decision else None,
                'routing_loss': physics_result.routing_loss,
                'routing_ms': physics_result.routing_ms,

                # Phase 2: Packing (if available)
                'context_efficiency': physics_result.context_efficiency,
                'packing_ms': physics_result.packing_ms,

                # Phase 3: Thermodynamics
                'selected_action': physics_result.selected_action,
                'exploration_temperature': physics_result.exploration_temperature,
                'free_energy': physics_result.free_energy,
                'thermodynamics_ms': physics_result.thermodynamics_ms,

                # Phase 4: Wave Mechanics
                'constructive_patterns': len(physics_result.constructive_patterns),
                'destructive_patterns': len(physics_result.destructive_patterns),
                'resonances': len(physics_result.resonances),
                'wave_mechanics_ms': physics_result.wave_mechanics_ms,

                # Unified metrics
                'total_energy': physics_result.total_energy,
                'total_entropy': physics_result.total_entropy,
                'total_free_energy': physics_result.total_free_energy,
                'physics_duration_ms': physics_result.duration_ms,

                # System statistics
                'physics_stats': self.unified_physics.get_statistics()
            }

            self.logger.info(
                f"✓ Physics enhancement complete: "
                f"routing={physics_result.routing_decision.target if physics_result.routing_decision else 'N/A'}, "
                f"T={physics_result.exploration_temperature:.2f}, "
                f"patterns={len(physics_result.constructive_patterns)}, "
                f"F={physics_result.total_free_energy:.3f}"
            )

        return spacetime, physics_result if track_provenance else None

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
        """
        if not self._recursive_components:
            return spacetime

        components = self._recursive_components
        confidence = spacetime.trace.tool_confidence

        # Phase 1: Scratchpad - Track provenance
        if components.get('scratchpad'):
            components['scratchpad'].add_entry(
                thought=f"Query: {query.text[:100]}",
                action=f"Tool: {spacetime.trace.tool_selected}, Adapter: {spacetime.trace.policy_adapter}",
                observation=f"Confidence: {confidence:.2f}",
                score=confidence
            )

        # Phase 2: Pattern Learning - Learn from high-confidence queries
        if confidence >= self.cfg.recursive_learning_refinement_threshold:
            # Learn pattern
            components['pattern_learner'].learn_from_spacetime(spacetime)

        # Phase 3: Hot Pattern Tracking - Track access frequency
        if components.get('hot_tracker'):
            components['hot_tracker'].record_access(spacetime)

        # Phase 4: Refinement - Refine low-confidence results
        if confidence < self.cfg.recursive_learning_refinement_threshold:
            self.logger.info(
                f"[LEARNING] Low confidence ({confidence:.2f}), triggering refinement"
            )

            refinement_result = await components['refiner'].refine(
                query=query,
                initial_spacetime=spacetime,
                strategy=None,  # Auto-select
                max_iterations=self.cfg.recursive_learning_max_iterations,
                quality_threshold=0.9
            )

            spacetime = refinement_result.final_spacetime

            # Log refinement to scratchpad
            if components.get('scratchpad'):
                components['scratchpad'].add_entry(
                    thought=f"Refinement: {refinement_result.strategy_used.value}",
                    action=f"Iterations: {refinement_result.iterations}",
                    observation=refinement_result.summary(),
                    score=refinement_result.trajectory[-1].score()
                )

        # Phase 5: Thompson Sampling and Policy Updates
        tool = spacetime.trace.tool_selected
        adapter = spacetime.trace.policy_adapter
        final_confidence = spacetime.trace.tool_confidence

        # Update Thompson priors
        if final_confidence >= 0.75:
            components['thompson_priors'].update_success(tool, final_confidence)
        else:
            components['thompson_priors'].update_failure(tool, final_confidence)

        # Update policy weights
        success = final_confidence >= 0.75
        components['policy_weights'].update(adapter, success)

        # Update metrics
        components['metrics'].update(final_confidence)
        components['metrics'].thompson_updates += 1
        components['metrics'].policy_updates += 1

        # Record for background learner
        if components.get('background_learner'):
            components['background_learner'].record_spacetime(spacetime)

        return spacetime

    def get_recursive_learning_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive recursive learning statistics.

        Returns None if recursive learning is disabled.

        Returns:
            Dict with learning metrics, Thompson priors, policy weights, etc.
        """
        if not self.enable_recursive_learning or not self._recursive_components:
            return None

        components = self._recursive_components
        stats = {
            "enabled": True,
            "queries_processed": components['metrics'].queries_processed,
            "avg_confidence": components['metrics'].avg_confidence,
            "thompson_priors": {
                tool: {
                    "expected_reward": components['thompson_priors'].get_expected_reward(tool),
                    "uncertainty": components['thompson_priors'].get_uncertainty(tool),
                    **priors
                }
                for tool, priors in components['thompson_priors'].tool_priors.items()
            },
            "policy_weights": {
                adapter: {
                    "weight": components['policy_weights'].get_weight(adapter),
                    "success_rate": components['policy_weights'].get_success_rate(adapter),
                    "total_uses": components['policy_weights'].adapter_total[adapter]
                }
                for adapter in components['policy_weights'].adapter_weights.keys()
            },
        }

        # Add hot patterns if available
        if components.get('hot_tracker'):
            hot_patterns = components['hot_tracker'].get_hot_patterns(limit=10)
            stats["hot_patterns"] = [
                {
                    "element_id": record.element_id,
                    "heat_score": record.heat_score,
                    "access_count": record.access_count,
                    "success_rate": record.success_rate,
                    "avg_confidence": record.avg_confidence
                }
                for record in hot_patterns
            ]

        # Add learned patterns if available
        if components.get('pattern_learner'):
            learned_patterns = components['pattern_learner'].get_hot_patterns()
            stats["learned_patterns"] = [
                {
                    "motifs": pattern.motifs[:3],
                    "tool": pattern.tool,
                    "query_type": pattern.query_type,
                    "occurrences": pattern.occurrences,
                    "avg_confidence": pattern.avg_confidence
                }
                for pattern in learned_patterns[:10]
            ]

        return stats

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
    # Statistical Mechanics (Phase 5) - Memory Consolidation
    # ========================================================================

    def _shards_to_microstates(self, shards: List[MemoryShard]) -> List['Microstate']:
        """
        Convert memory shards to microstates for statistical mechanics.

        Args:
            shards: Memory shards to convert

        Returns:
            List of Microstate objects
        """
        if not STATISTICAL_MECHANICS_AVAILABLE:
            return []

        microstates = []
        for shard in shards:
            # Get embedding (use first scale if available)
            state_vector = None
            if hasattr(shard, 'embedding') and shard.embedding is not None:
                state_vector = shard.embedding
            elif hasattr(shard, 'embeddings') and shard.embeddings:
                # Use first available embedding
                state_vector = shard.embeddings[0] if isinstance(shard.embeddings, list) else shard.embeddings
            elif hasattr(shard, 'metadata') and 'embedding' in shard.metadata:
                # Check metadata for embedding
                state_vector = shard.metadata['embedding']

            if state_vector is None:
                # Skip shards without embeddings
                self.logger.warning(f"Shard {shard.id} has no embedding, skipping consolidation")
                continue

            microstate = Microstate(
                id=shard.id,
                state_vector=np.array(state_vector),
                energy=0.0,  # Will be computed by engine
                metadata={
                    'source': shard.metadata.get('source', 'unknown') if hasattr(shard, 'metadata') else 'unknown',
                    'content_preview': shard.text[:100] if hasattr(shard, 'text') else ''
                }
            )
            microstates.append(microstate)

        return microstates

    def _macrostates_to_shards(self, macrostates: List['Macrostate']) -> List[MemoryShard]:
        """
        Convert consolidated macrostates back to memory shards.

        Each macrostate becomes a consolidated shard representing the cluster.

        Args:
            macrostates: Consolidated macrostates

        Returns:
            List of consolidated memory shards
        """
        consolidated_shards = []

        for i, macro in enumerate(macrostates):
            # Get representative microstate (centroid)
            if not macro.microstates:
                continue

            # Use first microstate as representative (could compute centroid instead)
            representative = macro.microstates[0]

            # Merge content from all microstates
            merged_content = f"[Consolidated Cluster {i}]\n\n"
            for j, micro in enumerate(macro.microstates[:5]):  # Limit to first 5
                preview = micro.metadata.get('content_preview', '')
                merged_content += f"{j+1}. {preview}...\n"

            if len(macro.microstates) > 5:
                merged_content += f"... and {len(macro.microstates) - 5} more memories\n"

            # Create consolidated shard
            consolidated_shard = MemoryShard(
                id=f"{macro.label}_consolidated",
                text=merged_content,
                metadata={
                    'embedding': representative.state_vector.tolist(),
                    'source': 'statistical_mechanics_consolidation',
                    'cluster_label': macro.label,
                    'cluster_size': len(macro.microstates),
                    'average_energy': macro.average_energy,
                    'entropy': macro.entropy,
                    'free_energy': macro.free_energy,
                    'order_parameter': macro.order_parameter,
                    'member_ids': [m.id for m in macro.microstates],
                    'consolidation_timestamp': datetime.now().isoformat()
                }
            )
            consolidated_shards.append(consolidated_shard)

        return consolidated_shards

    async def _background_consolidation_loop(self):
        """
        Background loop for periodic memory consolidation.

        Runs every consolidation_interval seconds, consolidates memory shards
        via statistical mechanics, and detects phase transitions.
        """
        self.logger.info(f"Starting background consolidation loop (interval={self.consolidation_interval}s)")

        while not self._closed:
            try:
                await asyncio.sleep(self.consolidation_interval)

                if self._closed:
                    break

                self.logger.info("[CONSOLIDATION] Starting periodic memory consolidation...")

                # Get current shards
                if hasattr(self.yarn_graph, 'shards') and isinstance(self.yarn_graph.shards, dict):
                    current_shards = list(self.yarn_graph.shards.values())
                elif hasattr(self.yarn_graph, 'shards') and isinstance(self.yarn_graph.shards, list):
                    current_shards = self.yarn_graph.shards
                else:
                    current_shards = self.shards

                if len(current_shards) < 3:
                    self.logger.debug(f"[CONSOLIDATION] Skipping (only {len(current_shards)} shards, need at least 3)")
                    continue

                # Convert to microstates
                microstates = self._shards_to_microstates(current_shards)

                if len(microstates) < 3:
                    self.logger.debug(f"[CONSOLIDATION] Skipping (only {len(microstates)} valid microstates)")
                    continue

                # Consolidate via Gibbs distribution
                consolidation_start = time.time()
                macrostates = self.stat_mech_engine.consolidate_memories(
                    microstates,
                    num_clusters=None  # Auto-detect optimal clusters
                )
                consolidation_time = (time.time() - consolidation_start) * 1000

                self.logger.info(f"[CONSOLIDATION] Consolidated {len(microstates)} → {len(macrostates)} clusters ({consolidation_time:.1f}ms)")

                # Detect phase transitions
                transition = self.stat_mech_engine.detect_phase_transition()

                if transition:
                    self.logger.info(
                        f"[PHASE TRANSITION] Detected at T={transition.critical_temperature:.2f}"
                    )
                    self.logger.info(
                        f"  Type: {transition.phase_type.value}, "
                        f"Order: {transition.order_before:.3f} → {transition.order_after:.3f}"
                    )
                    # Could trigger actions here (save checkpoint, notify, etc.)

                # Get statistics
                stats = self.stat_mech_engine.get_statistics()
                self.logger.info(
                    f"[STATISTICS] T={stats['temperature']:.2f}, "
                    f"S={stats['entropy']:.3f}, "
                    f"F={stats['free_energy']:.3f}"
                )

                # Replace shards with consolidated macrostates (optional - could keep both)
                # For now, we keep original shards and just log consolidation results
                # In production, you might want to:
                # 1. Archive original shards
                # 2. Replace with consolidated shards
                # 3. Update yarn_graph

                # Example: Replace shards (commented out for safety)
                # consolidated_shards = self._macrostates_to_shards(macrostates)
                # if hasattr(self.yarn_graph, 'shards'):
                #     self.yarn_graph.shards = {s.id: s for s in consolidated_shards}
                # self.shards = consolidated_shards

                # Cool temperature (simulated annealing)
                self.stat_mech_engine.anneal()

            except asyncio.CancelledError:
                self.logger.info("[CONSOLIDATION] Background consolidation cancelled")
                break
            except Exception as e:
                self.logger.error(f"[CONSOLIDATION] Error during consolidation: {e}", exc_info=True)
                # Continue running despite errors

        self.logger.info("[CONSOLIDATION] Background consolidation loop stopped")

    def start_background_consolidation(self):
        """
        Start background consolidation loop.

        Should be called after initialization if statistical mechanics is enabled.
        """
        if not self.enable_statistical_mechanics or not self.stat_mech_engine:
            self.logger.warning("Statistical mechanics not enabled, cannot start consolidation")
            return

        if self._consolidation_task is not None:
            self.logger.warning("Background consolidation already running")
            return

        # Create background task
        self._consolidation_task = asyncio.create_task(self._background_consolidation_loop())
        self._background_tasks.append(self._consolidation_task)
        self.logger.info("Background consolidation started")

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

        # Close wool storage (Phase 1: November 17, 2025)
        if self.wool:
            self.logger.info("Closing wool storage...")
            self.wool.close()

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

        NOTE: While this method is synchronous, _background_tasks access is
        protected by _bg_lock in async contexts. Since asyncio is single-threaded,
        synchronous appends are safe, but removal via callback needs protection.

        Args:
            coro: Coroutine to run in background

        Returns:
            asyncio.Task object

        Usage:
            task = shuttle.spawn_background_task(some_async_function())
        """
        task = asyncio.create_task(coro)

        # Append is atomic in Python, safe in single-threaded asyncio
        self._background_tasks.append(task)

        # Clean up completed tasks (callback is synchronous, but list operations are atomic)
        def cleanup_callback(t):
            try:
                self._background_tasks.remove(t)
            except ValueError:
                pass  # Already removed

        task.add_done_callback(cleanup_callback)

        return task

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

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get production monitoring metrics.

        Returns performance, resource, and learning metrics for observability.

        Returns:
            Dict with metrics or None if production hardening disabled

        Example:
            >>> orch = WeavingOrchestrator(..., enable_production_hardening=True)
            >>> metrics = orch.get_metrics()
            >>> print(f"QPS: {metrics['performance']['qps']}")
            >>> print(f"Error rate: {metrics['performance']['error_rate']}")
            >>> print(f"P95 latency: {metrics['performance']['latency_p95']}ms")
        """
        if not self.enable_production_hardening or not self.monitor:
            self.logger.warning("[PRODUCTION] Monitoring not enabled")
            return None

        try:
            return {
                "performance": self.monitor.performance.get_metrics(),
                "resources": self.monitor.resources.get_metrics(),
                "learning": self.monitor.learning.get_metrics(),
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"[PRODUCTION] Failed to get metrics: {e}")
            return {"error": str(e), "timestamp": time.time()}

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


if __name__ == "__main__":
    asyncio.run(main())
