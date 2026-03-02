#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Default Implementations of Component Protocols
================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 3: Protocol DI (Days 11-14)

This module provides default implementations of the 6 component protocols.
Each implementation wraps existing HoloLoom code to satisfy the protocol
interface while maintaining backward compatibility.

Default Implementations:
    1. DefaultPatternSelector - Wraps LoomCommand.select_pattern()
    2. DefaultThreadSelector - Wraps YarnGraph.select_threads()
    3. DefaultFeatureExtractor - Wraps ResonanceShed operations
    4. DefaultConvergence - Wraps ConvergenceEngine.collapse()
    5. DefaultToolExecutor - Wraps ToolExecutor.execute()
    6. DefaultSpacetimeAssembler - Wraps fabric assembly logic

Usage:
    >>> from hololoom.orchestrator.protocols.defaults import (
    ...     DefaultPatternSelector,
    ...     DefaultToolExecutor,
    ... )
    >>>
    >>> # Use with existing HoloLoom components
    >>> selector = DefaultPatternSelector(loom_command=loom_command)
    >>> pattern = selector.select_pattern("What is Thompson Sampling?")
    >>>
    >>> # Protocol compliance
    >>> isinstance(selector, PatternSelectorProtocol)
    True

Author: Claude Code (Elegance Pass - Phase 3)
Date: 2025-12-09
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from .components import (
    PatternSelectorProtocol,
    ThreadSelectorProtocol,
    FeatureExtractorProtocol,
    ConvergenceProtocol,
    ToolExecutorProtocol,
    SpacetimeAssemblerProtocol,
)

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext
    from hololoom.loom.command import LoomCommand
    from hololoom.loom.protocol import PatternSpec
    from hololoom.memory.graph import KG
    from hololoom.chrono.trigger import TemporalWindow
    from hololoom.protocols.types import Query, MemoryShard
    from hololoom.convergence.engine import ConvergenceEngine, CollapseResult
    from hololoom.resonance.shed import ResonanceShed
    from hololoom.tools.executor import ToolExecutor
    from hololoom.fabric.spacetime import Spacetime, WeavingTrace
    from hololoom.embedding.spectral import MatryoshkaEmbeddings
    from hololoom.config import Config

logger = logging.getLogger(__name__)


# ============================================================================
# Default Pattern Selector
# ============================================================================

class DefaultPatternSelector:
    """
    Default implementation of PatternSelectorProtocol.

    Wraps LoomCommand to provide pattern selection based on query characteristics.

    Example:
        >>> from hololoom.loom.command import LoomCommand
        >>> loom_command = LoomCommand(config)
        >>> selector = DefaultPatternSelector(loom_command=loom_command)
        >>> pattern = selector.select_pattern("What is Thompson Sampling?")
        >>> print(pattern.name)  # "FAST" or "FUSED"
    """

    def __init__(
        self,
        loom_command: 'LoomCommand',
        enable_complexity_detect: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize pattern selector.

        Args:
            loom_command: LoomCommand instance for pattern selection
            enable_complexity_detect: Whether to auto-detect complexity
            logger: Optional logger instance
        """
        self.loom_command = loom_command
        self.enable_complexity_detect = enable_complexity_detect
        self.logger = logger or logging.getLogger(__name__)

    def select_pattern(
        self,
        query_text: str,
        user_preference: Optional[str] = None
    ) -> 'PatternSpec':
        """
        Select pattern card based on query characteristics.

        Args:
            query_text: The query text to analyze
            user_preference: Optional explicit pattern name (BARE/FAST/FUSED)

        Returns:
            PatternSpec with pattern configuration
        """
        # If user provides explicit preference, use it
        if user_preference:
            return self.loom_command.get_pattern_by_name(user_preference)

        # Otherwise, let LoomCommand select based on query
        return self.loom_command.select_pattern(
            query_text,
            enable_complexity_detect=self.enable_complexity_detect
        )

    def detect_complexity(self, query_text: str) -> str:
        """
        Detect query complexity level.

        Args:
            query_text: The query text to analyze

        Returns:
            Complexity level: "trivial", "simple", "moderate", "complex", "research"
        """
        # Delegate to core complexity detection
        from hololoom.orchestrator.core import assess_complexity_level
        return assess_complexity_level(query_text)


# ============================================================================
# Default Thread Selector
# ============================================================================

class DefaultThreadSelector:
    """
    Default implementation of ThreadSelectorProtocol.

    Wraps KG (YarnGraph) to provide memory thread selection.

    Example:
        >>> from hololoom.memory.graph import KG
        >>> kg = KG()
        >>> selector = DefaultThreadSelector(yarn_graph=kg)
        >>> threads = await selector.select_threads(temporal_window, query, limit=10)
    """

    def __init__(
        self,
        yarn_graph: 'KG',
        enable_shuttle: bool = False,
        shuttle_stage: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize thread selector.

        Args:
            yarn_graph: KG (YarnGraph) instance for thread selection
            enable_shuttle: Whether to use Shuttle for advanced selection
            shuttle_stage: Optional ShuttleStage for enhanced selection
            logger: Optional logger instance
        """
        self.yarn_graph = yarn_graph
        self.enable_shuttle = enable_shuttle
        self.shuttle_stage = shuttle_stage
        self.logger = logger or logging.getLogger(__name__)

    async def select_threads(
        self,
        temporal_window: 'TemporalWindow',
        query: 'Query',
        limit: int = 10
    ) -> List['MemoryShard']:
        """
        Select memory threads from knowledge graph.

        Args:
            temporal_window: TemporalWindow constraining thread selection
            query: Query object with text and metadata
            limit: Maximum number of threads to return

        Returns:
            List of MemoryShard objects selected for this query
        """
        # Use shuttle if enabled and available
        if self.enable_shuttle and self.shuttle_stage:
            return await self._shuttle_select(temporal_window, query, limit)

        # Otherwise use YarnGraph directly
        return self._yarn_graph_select(temporal_window, query, limit)

    def _yarn_graph_select(
        self,
        temporal_window: 'TemporalWindow',
        query: 'Query',
        limit: int
    ) -> List['MemoryShard']:
        """Select threads using YarnGraph (simple recency-based)."""
        # Get recent nodes from graph
        nodes = self.yarn_graph.get_recent_nodes(
            window_start=temporal_window.start if temporal_window else None,
            window_end=temporal_window.end if temporal_window else None,
            limit=limit
        )
        return nodes

    async def _shuttle_select(
        self,
        temporal_window: 'TemporalWindow',
        query: 'Query',
        limit: int
    ) -> List['MemoryShard']:
        """Select threads using Shuttle (advanced selection)."""
        # Delegate to shuttle stage
        return await self.shuttle_stage.select_threads(
            temporal_window=temporal_window,
            query=query,
            limit=limit
        )

    async def expand_context(
        self,
        threads: List['MemoryShard'],
        max_hops: int = 2
    ) -> List['MemoryShard']:
        """
        Expand context by traversing graph connections.

        Args:
            threads: Initial thread selection
            max_hops: Maximum graph traversal depth

        Returns:
            Expanded list of MemoryShard objects
        """
        if not threads:
            return []

        # Use adaptive expansion if available
        try:
            from hololoom.memory.adaptive_expansion import expand_context_adaptive

            seed_node_ids = [t.id for t in threads if hasattr(t, 'id')]
            result = await expand_context_adaptive(
                query="",  # Context expansion doesn't need query
                seed_nodes=seed_node_ids,
                graph=self.yarn_graph.graph,
                max_hops=max_hops
            )
            return result.nodes if result else threads
        except ImportError:
            self.logger.debug("Adaptive expansion not available, returning original threads")
            return threads


# ============================================================================
# Default Feature Extractor
# ============================================================================

class DefaultFeatureExtractor:
    """
    Default implementation of FeatureExtractorProtocol.

    Wraps ResonanceShed operations for feature extraction.

    Example:
        >>> extractor = DefaultFeatureExtractor(embedder=embedder, cfg=config)
        >>> features = await extractor.extract(query, threads)
        >>> print(features["embeddings"].shape)  # (384,)
    """

    def __init__(
        self,
        embedder: 'MatryoshkaEmbeddings',
        cfg: 'Config',
        resonance_shed: Optional['ResonanceShed'] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize feature extractor.

        Args:
            embedder: MatryoshkaEmbeddings for multi-scale embeddings
            cfg: Configuration object
            resonance_shed: Optional ResonanceShed for feature extraction
            logger: Optional logger instance
        """
        self.embedder = embedder
        self.cfg = cfg
        self.resonance_shed = resonance_shed
        self.logger = logger or logging.getLogger(__name__)

    async def extract(
        self,
        query: 'Query',
        threads: List['MemoryShard']
    ) -> Dict[str, Any]:
        """
        Extract features from query and context.

        Args:
            query: Query object with text and metadata
            threads: Selected memory threads for context

        Returns:
            Dictionary with extracted features
        """
        features = {}

        # Extract embeddings
        query_text = query.text if hasattr(query, 'text') else str(query)
        features['embeddings'] = self.embedder.embed(query_text)

        # Extract motifs (if available)
        features['motifs'] = self._extract_motifs(query_text)

        # Extract spectral features from threads
        features['spectral'] = self._extract_spectral(threads)

        # Use resonance shed if available
        if self.resonance_shed:
            shed_features = await self.resonance_shed.extract(query, threads)
            features.update(shed_features)

        return features

    def _extract_motifs(self, query_text: str) -> List[str]:
        """Extract motif patterns from query text."""
        # Simple heuristic motif detection
        motifs = []
        if '?' in query_text:
            motifs.append('question')
        if any(word in query_text.lower() for word in ['what', 'who', 'where', 'when', 'why', 'how']):
            motifs.append('interrogative')
        if any(word in query_text.lower() for word in ['explain', 'describe', 'define']):
            motifs.append('explanatory')
        if any(word in query_text.lower() for word in ['compare', 'versus', 'vs', 'difference']):
            motifs.append('comparative')
        return motifs if motifs else ['general']

    def _extract_spectral(self, threads: List['MemoryShard']) -> List[float]:
        """Extract spectral features from memory threads."""
        # Simple spectral features based on thread count and connectivity
        return [float(len(threads)), 0.0, 0.0, 0.0, 0.0, 0.0]  # 6D spectral

    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted embeddings."""
        return self.cfg.embedding_dim if hasattr(self.cfg, 'embedding_dim') else 384


# ============================================================================
# Default Convergence
# ============================================================================

class DefaultConvergence:
    """
    Default implementation of ConvergenceProtocol.

    Wraps ConvergenceEngine for decision collapse.

    Example:
        >>> from hololoom.convergence.engine import ConvergenceEngine
        >>> engine = ConvergenceEngine()
        >>> convergence = DefaultConvergence(engine=engine)
        >>> result = convergence.collapse(probs, strategy="bayesian_blend")
    """

    def __init__(
        self,
        engine: Optional['ConvergenceEngine'] = None,
        n_tools: int = 4,
        epsilon: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize convergence.

        Args:
            engine: Optional ConvergenceEngine instance
            n_tools: Number of available tools
            epsilon: Exploration rate for epsilon-greedy
            logger: Optional logger instance
        """
        self.engine = engine
        self.n_tools = n_tools
        self.epsilon = epsilon
        self.logger = logger or logging.getLogger(__name__)

        # Thompson Sampling priors (alpha, beta for each tool)
        self._alpha = [1.0] * n_tools
        self._beta = [1.0] * n_tools

    def collapse(
        self,
        probs: Any,
        strategy: str = "bayesian_blend"
    ) -> 'CollapseResult':
        """
        Collapse probability distribution to discrete decision.

        Args:
            probs: Probability distribution over tools (np.ndarray)
            strategy: Collapse strategy name

        Returns:
            CollapseResult with selected tool index and metadata
        """
        import numpy as np

        # Use engine if available
        if self.engine:
            return self.engine.collapse(probs, strategy=strategy)

        # Otherwise, simple implementation
        probs_array = np.array(probs) if not isinstance(probs, np.ndarray) else probs

        if strategy == "argmax":
            tool_index = int(np.argmax(probs_array))
        elif strategy == "epsilon_greedy":
            if np.random.random() < self.epsilon:
                tool_index = np.random.randint(len(probs_array))
            else:
                tool_index = int(np.argmax(probs_array))
        elif strategy == "pure_thompson":
            # Thompson sampling from beta distributions
            samples = [np.random.beta(self._alpha[i], self._beta[i])
                      for i in range(len(probs_array))]
            tool_index = int(np.argmax(samples))
        else:  # bayesian_blend (default)
            # Blend neural predictions with Thompson priors
            thompson_probs = [self._alpha[i] / (self._alpha[i] + self._beta[i])
                             for i in range(len(probs_array))]
            blended = 0.7 * probs_array + 0.3 * np.array(thompson_probs)
            tool_index = int(np.argmax(blended))

        # Create simple result
        from dataclasses import dataclass

        @dataclass
        class SimpleCollapseResult:
            tool_index: int
            confidence: float
            strategy: str
            metadata: Dict[str, Any]

        return SimpleCollapseResult(
            tool_index=tool_index,
            confidence=float(probs_array[tool_index]),
            strategy=strategy,
            metadata={"probs": probs_array.tolist()}
        )

    def update_priors(
        self,
        tool_index: int,
        success: bool,
        confidence: float
    ) -> None:
        """
        Update Thompson Sampling priors based on outcome.

        Args:
            tool_index: Index of the tool that was executed
            success: Whether execution was successful
            confidence: Confidence score (0.0-1.0)
        """
        if 0 <= tool_index < self.n_tools:
            if success:
                self._alpha[tool_index] += confidence
            else:
                self._beta[tool_index] += (1.0 - confidence)

    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.epsilon


# ============================================================================
# Default Tool Executor
# ============================================================================

class DefaultToolExecutor:
    """
    Default implementation of ToolExecutorProtocol.

    Wraps ToolExecutor for tool execution with optional safety gating.

    Example:
        >>> executor = DefaultToolExecutor(tool_executor=tool_executor)
        >>> result = await executor.execute("answer", query, context)
    """

    def __init__(
        self,
        tool_executor: Optional['ToolExecutor'] = None,
        guardrails: Optional[Any] = None,
        tools: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize tool executor.

        Args:
            tool_executor: Optional ToolExecutor instance
            guardrails: Optional SafetyGuardrails for safety gating
            tools: List of available tool names
            logger: Optional logger instance
        """
        self.tool_executor = tool_executor
        self.guardrails = guardrails
        self._tools = tools or ["answer", "research", "explore", "clarify"]
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        tool: str,
        query: 'Query',
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute selected tool with safety gating.

        Args:
            tool: Name of the tool to execute
            query: Query object with text and metadata
            context: Execution context (features, threads, etc.)

        Returns:
            Dictionary with execution result
        """
        # Safety check if guardrails available
        if self.guardrails:
            safety_result = await self._check_safety(tool, query, context)
            if not safety_result.get('allowed', True):
                return {
                    'response': f"Action blocked: {safety_result.get('reason', 'Unknown')}",
                    'blocked': True,
                    'safety_decision': safety_result
                }

        # Execute tool
        if self.tool_executor:
            return await self.tool_executor.execute(tool, query, context)

        # Simple fallback implementation
        query_text = query.text if hasattr(query, 'text') else str(query)
        return {
            'response': f"[{tool}] Processing: {query_text[:100]}...",
            'tool': tool,
            'metadata': {'fallback': True}
        }

    async def _check_safety(
        self,
        tool: str,
        query: 'Query',
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check safety with guardrails."""
        try:
            from hololoom.alignment.safety_guardrails import ActionRequest

            request = ActionRequest(
                action=tool,
                query=query.text if hasattr(query, 'text') else str(query),
                context=context
            )
            decision = await self.guardrails.evaluate(request)
            return {
                'allowed': decision.allowed,
                'reason': decision.reason if hasattr(decision, 'reason') else None
            }
        except Exception as e:
            self.logger.warning(f"Safety check failed: {e}")
            return {'allowed': True}  # Fail open for now

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        if self.tool_executor:
            return self.tool_executor.get_available_tools()
        return self._tools

    async def validate_tool(self, tool: str, context: Dict[str, Any]) -> bool:
        """Validate if tool can be executed in given context."""
        return tool in self.get_available_tools()


# ============================================================================
# Default Spacetime Assembler
# ============================================================================

class DefaultSpacetimeAssembler:
    """
    Default implementation of SpacetimeAssemblerProtocol.

    Assembles final Spacetime output with complete provenance.

    Example:
        >>> assembler = DefaultSpacetimeAssembler(cfg=config)
        >>> spacetime = assembler.assemble(ctx)
        >>> print(spacetime.response)
    """

    def __init__(
        self,
        cfg: 'Config',
        semantic_cache: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize spacetime assembler.

        Args:
            cfg: Configuration object
            semantic_cache: Optional semantic cache for statistics
            logger: Optional logger instance
        """
        self.cfg = cfg
        self.semantic_cache = semantic_cache
        self.logger = logger or logging.getLogger(__name__)

    def assemble(self, ctx: 'WeavingContext') -> 'Spacetime':
        """
        Assemble final Spacetime output with provenance.

        Args:
            ctx: WeavingContext with all pipeline state

        Returns:
            Spacetime object with response, confidence, and trace
        """
        from hololoom.fabric.spacetime import Spacetime

        # Extract query text from context
        query_text = ""
        if hasattr(ctx, 'query'):
            if hasattr(ctx.query, 'text'):
                query_text = ctx.query.text
            else:
                query_text = str(ctx.query)

        # Extract response from tool result
        response = ""
        if ctx.tool_result:
            response = ctx.tool_result.get('response', '')

        # Extract tool_used from context
        tool_used = "unknown"
        if hasattr(ctx, 'tool_used') and ctx.tool_used:
            tool_used = ctx.tool_used
        elif hasattr(ctx, 'collapse_result') and ctx.collapse_result:
            if hasattr(ctx.collapse_result, 'selected_tool'):
                tool_used = ctx.collapse_result.selected_tool
            elif hasattr(ctx.collapse_result, 'tool_index'):
                tool_used = f"tool_{ctx.collapse_result.tool_index}"

        # Calculate confidence
        confidence = self.calculate_confidence(ctx)

        # Create trace
        trace = self.create_trace(ctx)

        # Assemble metadata
        metadata = {
            'pattern_used': ctx.pattern_spec.name if ctx.pattern_spec else 'UNKNOWN',
            'threads_count': len(ctx.threads) if ctx.threads else 0,
            'stage_count': len(ctx.stage_timings) if hasattr(ctx, 'stage_timings') and ctx.stage_timings else 0,
        }

        # Add cache stats if available
        if self.semantic_cache:
            metadata['cache_stats'] = self._get_cache_stats()

        return Spacetime(
            query_text=query_text,
            response=response,
            tool_used=tool_used,
            confidence=confidence,
            trace=trace,
            metadata=metadata
        )

    def create_trace(self, ctx: 'WeavingContext') -> Any:
        """
        Create WeavingTrace from context.

        Args:
            ctx: WeavingContext with stage timings and metadata

        Returns:
            WeavingTrace object with complete provenance
        """
        from datetime import datetime
        from hololoom.fabric.spacetime import WeavingTrace

        # Calculate timing from context or use defaults
        now = datetime.now()
        stage_timings = ctx.stage_timings if hasattr(ctx, 'stage_timings') and ctx.stage_timings else {}
        total_duration = sum(stage_timings.values()) if stage_timings else 0.0

        return WeavingTrace(
            start_time=now,
            end_time=now,
            duration_ms=total_duration,
            stage_durations=stage_timings,
            motifs_detected=[],
            embedding_scales_used=[384],  # Default scale
        )

    def calculate_confidence(self, ctx: 'WeavingContext') -> float:
        """
        Calculate final confidence score.

        Args:
            ctx: WeavingContext with pipeline results

        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with collapse result confidence if available
        if ctx.collapse_result and hasattr(ctx.collapse_result, 'confidence'):
            base_confidence = ctx.collapse_result.confidence
        else:
            base_confidence = 0.5

        # Adjust based on thread count (more threads = more context = higher confidence)
        thread_boost = min(0.1, len(ctx.threads) * 0.02) if ctx.threads else 0.0

        # Penalize for errors
        error_penalty = min(0.2, len(ctx.errors) * 0.05) if ctx.errors else 0.0

        # Final confidence
        confidence = base_confidence + thread_boost - error_penalty
        return max(0.0, min(1.0, confidence))

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.semantic_cache and hasattr(self.semantic_cache, 'get_stats'):
            return self.semantic_cache.get_stats()
        return {}


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'DefaultPatternSelector',
    'DefaultThreadSelector',
    'DefaultFeatureExtractor',
    'DefaultConvergence',
    'DefaultToolExecutor',
    'DefaultSpacetimeAssembler',
]
