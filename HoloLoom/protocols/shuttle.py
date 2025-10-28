"""
mythRL Shuttle Protocols
========================
Protocol definitions for the Shuttle-centric architecture.

These protocols support the 3-5-7-9 progressive complexity system:
- LITE (3 steps): Extract → Route → Execute
- FAST (5 steps): + Pattern Selection + Temporal Windows
- FULL (7 steps): + Decision Engine + Synthesis Bridge
- RESEARCH (9 steps): + Advanced WarpSpace + Full Tracing

All implementations are swappable via dependency injection.

Author: mythRL Team
Date: 2025-10-27 (Phase 1 - Task 1.1: Protocol Standardization)
"""

from typing import Protocol, runtime_checkable, List, Dict, Any, Optional
from HoloLoom.protocols.types import ComplexityLevel


# ============================================================================
# Pattern Selection Protocol
# ============================================================================

@runtime_checkable
class PatternSelectionProtocol(Protocol):
    """
    Protocol for processing pattern selection.

    Grows in complexity/necessity based on ComplexityLevel:
    - LITE: Skip (use default patterns)
    - FAST: Basic pattern recognition
    - FULL: Advanced pattern synthesis
    - RESEARCH: Emergent pattern discovery

    Note: This is different from PatternDetector which detects patterns
    in memory access. PatternSelectionProtocol selects PROCESSING patterns.
    """

    async def select_pattern(
        self,
        query: str,
        context: Dict,
        complexity: ComplexityLevel
    ) -> Dict:
        """
        Select processing pattern based on query complexity.

        Args:
            query: User query string
            context: Execution context
            complexity: Target complexity level

        Returns:
            Dict with pattern selection:
            {
                'selected_pattern': str,
                'pattern_config': Dict,
                'confidence': float
            }
        """
        ...

    async def assess_pattern_necessity(self, query: str) -> float:
        """
        Assess how much pattern selection is needed (0.0-1.0).

        Args:
            query: User query

        Returns:
            Necessity score (0.0 = skip, 1.0 = essential)
        """
        ...

    async def synthesize_patterns(
        self,
        primary: Dict,
        secondary: Dict
    ) -> Dict:
        """
        Synthesize multiple patterns (FULL+ only).

        Args:
            primary: Primary pattern config
            secondary: Secondary pattern config

        Returns:
            Synthesized pattern configuration
        """
        ...


# ============================================================================
# Feature Extraction Protocol
# ============================================================================

@runtime_checkable
class FeatureExtractionProtocol(Protocol):
    """
    Protocol for multi-scale feature extraction.

    Matryoshka scaling based on ComplexityLevel:
    - LITE: 96d embeddings only (~50ms)
    - FAST: 96d + 192d embeddings (~150ms)
    - FULL: 96d + 192d + 384d embeddings (~300ms)
    - RESEARCH: All scales + experimental features

    Note: Uses Embedder protocol for low-level vector generation.
    This is high-level orchestration of feature extraction.
    """

    async def extract_features(
        self,
        data: Any,
        scales: List[int]
    ) -> Dict:
        """
        Extract features at specified embedding scales.

        Args:
            data: Input data (text, structured data, etc.)
            scales: List of embedding dimensions [96, 192, 384, etc.]

        Returns:
            Dict with multi-scale features:
            {
                'embeddings': {96: [...], 192: [...], ...},
                'motifs': [...],
                'metadata': {...},
                'confidence': float
            }
        """
        ...

    async def extract_motifs(
        self,
        data: Any,
        complexity: ComplexityLevel
    ) -> Dict:
        """
        Extract recurring patterns/motifs.

        Args:
            data: Input data
            complexity: Determines motif extraction depth

        Returns:
            Dict with detected motifs
        """
        ...

    async def assess_extraction_needs(self, data: Any) -> List[int]:
        """
        Recommend embedding scales needed for this data.

        Args:
            data: Input data to assess

        Returns:
            List of recommended scales [96, 192, ...]
        """
        ...


# ============================================================================
# WarpSpace Protocol
# ============================================================================

@runtime_checkable
class WarpSpaceProtocol(Protocol):
    """
    Protocol for mathematical manifold operations.

    NON-NEGOTIABLE: Always present in Shuttle, but complexity-gated.

    Complexity-based operations:
    - LITE: Basic tensor operations
    - FAST: Standard manifold operations
    - FULL: Advanced mathematical features
    - RESEARCH: Experimental manifold research

    This is the mathematical heart of HoloLoom's tensor manifold.
    """

    async def create_manifold(
        self,
        features: Dict,
        complexity: ComplexityLevel
    ) -> Dict:
        """
        Create tensor manifold for computation.

        Args:
            features: Extracted features
            complexity: Determines manifold complexity

        Returns:
            Dict representing manifold structure:
            {
                'manifold_type': str,
                'dimensions': int,
                'tensor_shape': tuple,
                'metadata': Dict
            }
        """
        ...

    async def tension_threads(
        self,
        manifold: Dict,
        threads: List[Dict]
    ) -> Dict:
        """
        Apply tension to threads in manifold space.

        Args:
            manifold: Manifold structure
            threads: Threads to tension

        Returns:
            Updated manifold with tensioned threads
        """
        ...

    async def compute_trajectories(
        self,
        manifold: Dict,
        start_points: List[Dict]
    ) -> Dict:
        """
        Compute trajectories through manifold (FULL+).

        Args:
            manifold: Manifold structure
            start_points: Starting points for trajectories

        Returns:
            Computed trajectories with metadata
        """
        ...

    async def experimental_operations(
        self,
        manifold: Dict,
        experiments: List[str]
    ) -> Dict:
        """
        Experimental manifold operations (RESEARCH only).

        Args:
            manifold: Manifold structure
            experiments: List of experiment names

        Returns:
            Experimental results
        """
        ...


# ============================================================================
# Decision Engine Protocol
# ============================================================================

@runtime_checkable
class DecisionEngineProtocol(Protocol):
    """
    Protocol for strategic decision-making.

    Grows in sophistication based on ComplexityLevel:
    - LITE: Skip (direct routing)
    - FAST: Skip (direct routing)
    - FULL: Intelligent decision making
    - RESEARCH: Advanced multi-criteria optimization

    Note: Different from PolicyEngine which is reactive.
    DecisionEngineProtocol is strategic multi-criteria optimization.
    """

    async def make_decision(
        self,
        features: Dict,
        context: Dict,
        options: List[Dict]
    ) -> Dict:
        """
        Make decision based on extracted features and available options.

        Args:
            features: Extracted features from input
            context: Decision context
            options: List of available options

        Returns:
            Dict with decision:
            {
                'selected_option': Dict,
                'reasoning': str,
                'confidence': float,
                'alternatives': List[Dict]
            }
        """
        ...

    async def assess_decision_complexity(self, features: Dict) -> float:
        """
        Assess decision complexity needed (0.0-1.0).

        Args:
            features: Extracted features

        Returns:
            Complexity score (0.0 = skip, 1.0 = full decision engine)
        """
        ...

    async def optimize_multi_criteria(
        self,
        criteria: List[Dict],
        constraints: Dict
    ) -> Dict:
        """
        Multi-criteria optimization (RESEARCH only).

        Args:
            criteria: List of optimization criteria
            constraints: Constraints on optimization

        Returns:
            Optimized solution with Pareto frontier
        """
        ...


# ============================================================================
# Tool Execution Protocol
# ============================================================================

@runtime_checkable
class ToolExecutor(Protocol):
    """
    Protocol for tool execution.

    Standardized interface for executing tools/actions across
    different complexity levels.
    """

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict,
        context: Dict
    ) -> Dict:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            context: Execution context

        Returns:
            Dict with execution result:
            {
                'tool': str,
                'output': Any,
                'confidence': float,
                'execution_time_ms': float
            }
        """
        ...

    async def list_available_tools(self, context: Dict) -> List[Dict]:
        """
        List available tools for current context.

        Args:
            context: Execution context

        Returns:
            List of tool descriptions
        """
        ...

    async def assess_tool_necessity(
        self,
        query: str,
        context: Dict
    ) -> Dict:
        """
        Assess which tools are needed for query.

        Args:
            query: User query
            context: Execution context

        Returns:
            Dict with tool recommendations and confidence scores
        """
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'PatternSelectionProtocol',
    'FeatureExtractionProtocol',
    'WarpSpaceProtocol',
    'DecisionEngineProtocol',
    'ToolExecutor',
]