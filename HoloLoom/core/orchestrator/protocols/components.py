#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Component Protocols for Dependency Injection
=============================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 3: Protocol DI (Days 11-14)

This module defines 6 component protocols that enable full dependency injection
into the weaving pipeline. Each protocol defines the minimal interface for a
capability, allowing implementations to be swapped without changing orchestrator code.

Protocols:
    1. PatternSelectorProtocol - Pattern card selection
    2. ThreadSelectorProtocol - Memory thread selection
    3. FeatureExtractorProtocol - Feature extraction
    4. ConvergenceProtocol - Decision collapse
    5. ToolExecutorProtocol - Tool execution
    6. SpacetimeAssemblerProtocol - Output assembly

Philosophy:
    "Program to interfaces, not implementations"
    These protocols define WHAT, not HOW. Default implementations wrap
    existing code, but custom implementations can be injected for:
    - Testing (mock implementations)
    - A/B testing (alternative algorithms)
    - Domain-specific optimization

Usage:
    >>> from HoloLoom.core.orchestrator.protocols import PatternSelectorProtocol
    >>>
    >>> class MyPatternSelector:
    ...     def select_pattern(self, query_text, user_preference=None):
    ...         return my_custom_pattern_spec
    >>>
    >>> # Protocol compliance check
    >>> isinstance(MyPatternSelector(), PatternSelectorProtocol)
    True

Author: Claude Code (Elegance Pass - Phase 3)
Date: 2025-12-09
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.core.orchestrator.context import WeavingContext
    from HoloLoom.core.loom.protocol import PatternSpec
    from HoloLoom.core.chrono.trigger import TemporalWindow
    from HoloLoom.core.protocols.types import Query, MemoryShard
    from HoloLoom.core.convergence.engine import CollapseResult
    from HoloLoom.core.fabric.spacetime import Spacetime


# ============================================================================
# Pattern Selection Protocol
# ============================================================================

@runtime_checkable
class PatternSelectorProtocol(Protocol):
    """
    Protocol for pattern card selection.

    Pattern selection determines the execution template (BARE/FAST/FUSED)
    based on query characteristics and optional user preferences.

    Default Implementation: Wraps LoomCommand.select_pattern()

    Example:
        >>> class CustomPatternSelector:
        ...     def select_pattern(self, query_text, user_preference=None):
        ...         if len(query_text) < 20:
        ...             return PatternSpec(name="BARE", scales=[96])
        ...         return PatternSpec(name="FUSED", scales=[96, 192, 384])
        ...
        ...     def detect_complexity(self, query_text):
        ...         return "simple" if len(query_text) < 20 else "complex"
    """

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
            PatternSpec with pattern configuration (name, scales, features, timeout)
        """
        ...

    def detect_complexity(self, query_text: str) -> str:
        """
        Detect query complexity level.

        Args:
            query_text: The query text to analyze

        Returns:
            Complexity level string: "trivial", "simple", "moderate", "complex", "research"
        """
        ...


# ============================================================================
# Thread Selection Protocol
# ============================================================================

@runtime_checkable
class ThreadSelectorProtocol(Protocol):
    """
    Protocol for memory thread selection from knowledge graph.

    Thread selection retrieves relevant memory shards based on the
    temporal window and query context.

    Default Implementation: Wraps YarnGraph.select_threads()

    Example:
        >>> class CustomThreadSelector:
        ...     async def select_threads(self, temporal_window, query, limit=10):
        ...         # Custom selection logic
        ...         return relevant_shards[:limit]
        ...
        ...     async def expand_context(self, threads, max_hops=2):
        ...         # Graph traversal for context expansion
        ...         return expanded_threads
    """

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
        ...

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
        ...


# ============================================================================
# Feature Extraction Protocol
# ============================================================================

@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """
    Protocol for feature extraction from query and context.

    Feature extraction creates the DotPlasma (feature fusion) from
    query text and selected memory threads.

    Default Implementation: Wraps ResonanceShed operations

    Example:
        >>> class CustomFeatureExtractor:
        ...     async def extract(self, query, threads):
        ...         motifs = extract_motifs(query.text)
        ...         embeddings = embed(query.text)
        ...         spectral = compute_spectral_features(threads)
        ...         return {"motifs": motifs, "embeddings": embeddings, "spectral": spectral}
        ...
        ...     def get_feature_dim(self):
        ...         return 384
    """

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
            Dictionary with extracted features:
            - motifs: Symbolic patterns (list of strings)
            - embeddings: Multi-scale vectors (np.ndarray)
            - spectral: Topological features (np.ndarray)
            - Additional domain-specific features
        """
        ...

    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of extracted embeddings.

        Returns:
            Feature dimension (e.g., 384 for Matryoshka embeddings)
        """
        ...


# ============================================================================
# Convergence Protocol
# ============================================================================

@runtime_checkable
class ConvergenceProtocol(Protocol):
    """
    Protocol for decision collapse from continuous to discrete.

    Convergence collapses probability distributions over tools/actions
    to a discrete selection using various strategies.

    Default Implementation: Wraps ConvergenceEngine.collapse()

    Strategies:
        - ARGMAX: Select highest probability
        - EPSILON_GREEDY: Explore with probability epsilon
        - BAYESIAN_BLEND: Combine neural + bandit priors
        - PURE_THOMPSON: Thompson Sampling only

    Example:
        >>> class CustomConvergence:
        ...     def collapse(self, probs, strategy="bayesian_blend"):
        ...         if strategy == "argmax":
        ...             return CollapseResult(tool_index=np.argmax(probs), ...)
        ...         # Custom collapse logic
        ...
        ...     def update_priors(self, tool_index, success, confidence):
        ...         # Update Thompson Sampling priors
        ...         pass
    """

    def collapse(
        self,
        probs: Any,  # np.ndarray, but avoid import for protocol
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
        ...

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
        ...

    def get_exploration_rate(self) -> float:
        """
        Get current exploration rate.

        Returns:
            Exploration probability (0.0-1.0)
        """
        ...


# ============================================================================
# Tool Executor Protocol
# ============================================================================

@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """
    Protocol for tool execution with optional safety gating.

    Tool execution runs the selected tool with the query and context,
    optionally applying safety guardrails.

    Default Implementation: Wraps ToolExecutor.execute()

    Example:
        >>> class CustomToolExecutor:
        ...     async def execute(self, tool, query, context):
        ...         if tool == "answer":
        ...             return {"response": generate_answer(query, context)}
        ...         elif tool == "research":
        ...             return {"response": conduct_research(query, context)}
        ...
        ...     def get_available_tools(self):
        ...         return ["answer", "research", "explore", "clarify"]
    """

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
            Dictionary with execution result:
            - response: Generated response text
            - metadata: Execution metadata
            - safety_decision: Safety evaluation (if applicable)
        """
        ...

    def get_available_tools(self) -> List[str]:
        """
        Get list of available tools.

        Returns:
            List of tool names
        """
        ...

    async def validate_tool(self, tool: str, context: Dict[str, Any]) -> bool:
        """
        Validate if tool can be executed in given context.

        Args:
            tool: Name of the tool to validate
            context: Execution context

        Returns:
            True if tool can be executed, False otherwise
        """
        ...


# ============================================================================
# Spacetime Assembler Protocol
# ============================================================================

@runtime_checkable
class SpacetimeAssemblerProtocol(Protocol):
    """
    Protocol for assembling final Spacetime output with provenance.

    Spacetime assembly creates the final output artifact with complete
    computational lineage for debugging and reflection learning.

    Default Implementation: Wraps fabric assembly logic

    Example:
        >>> class CustomSpacetimeAssembler:
        ...     def assemble(self, ctx):
        ...         return Spacetime(
        ...             response=ctx.tool_result["response"],
        ...             confidence=ctx.collapse_result.confidence,
        ...             trace=self.create_trace(ctx),
        ...             metadata={"custom": "data"}
        ...         )
        ...
        ...     def create_trace(self, ctx):
        ...         return WeavingTrace(
        ...             stage_timings=ctx.stage_timings,
        ...             pattern_used=ctx.pattern_spec.name,
        ...             ...
        ...         )
    """

    def assemble(self, ctx: 'WeavingContext') -> 'Spacetime':
        """
        Assemble final Spacetime output with provenance.

        Args:
            ctx: WeavingContext with all pipeline state

        Returns:
            Spacetime object with:
            - response: Generated response text
            - confidence: Confidence score (0.0-1.0)
            - trace: WeavingTrace with complete provenance
            - metadata: Additional metadata
        """
        ...

    def create_trace(self, ctx: 'WeavingContext') -> Any:
        """
        Create WeavingTrace from context.

        Args:
            ctx: WeavingContext with stage timings and metadata

        Returns:
            WeavingTrace object with complete provenance
        """
        ...

    def calculate_confidence(self, ctx: 'WeavingContext') -> float:
        """
        Calculate final confidence score.

        Args:
            ctx: WeavingContext with pipeline results

        Returns:
            Confidence score (0.0-1.0)
        """
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Component Protocols (Phase 3)
    'PatternSelectorProtocol',
    'ThreadSelectorProtocol',
    'FeatureExtractorProtocol',
    'ConvergenceProtocol',
    'ToolExecutorProtocol',
    'SpacetimeAssemblerProtocol',
]
