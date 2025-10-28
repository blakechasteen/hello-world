"""
Core Feature Protocol Definitions
==================================
Protocols for core HoloLoom features (embedding, motif detection, policy).

These are the fundamental building blocks used across the system.

Author: mythRL Team
Date: 2025-10-27 (Phase 1 - Task 1.1: Protocol Standardization)
"""

from typing import Protocol, runtime_checkable, List, Any, Tuple, Optional

# Import shared types
try:
    from HoloLoom.documentation.types import (
        Features, Context, ActionPlan, Query, Vector
    )
except ImportError:
    # Fallback if types not available
    Features = Any
    Context = Any
    ActionPlan = Any
    Query = Any
    Vector = Any


# ============================================================================
# Embedding Protocol
# ============================================================================

@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for embedding implementations.

    All embedders must implement encode methods.
    This enables swappable embedding backends without code changes.

    Implementations: MatryoshkaEmbeddings, SpectralEmbedder, etc.
    """

    def encode(self, texts: List[str]) -> List[Vector]:
        """
        Encode texts into vectors (SYNCHRONOUS).

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors

        Note: This is synchronous - do NOT await it!
        """
        ...

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...


# ============================================================================
# Motif Detection Protocol
# ============================================================================

@runtime_checkable
class MotifDetector(Protocol):
    """
    Protocol for pattern/motif detection in text.

    Detects recurring patterns, entities, and structural elements.

    Implementations: RegexMotifDetector, SpaCyMotifDetector, etc.
    """

    async def detect(self, text: str) -> List[Tuple[str, int, int, float]]:
        """
        Detect motifs in text.

        Args:
            text: Input text

        Returns:
            List of (pattern, start, end, score) tuples
        """
        ...


# ============================================================================
# Policy Engine Protocol
# ============================================================================

@runtime_checkable
class PolicyEngine(Protocol):
    """
    Protocol for decision-making/policy engines.

    Makes decisions about which actions to take based on features and context.

    Implementations: UnifiedPolicy (neural + Thompson Sampling), etc.
    """

    async def choose_action(
        self,
        query: Query,
        features: Features,
        context: Context
    ) -> ActionPlan:
        """
        Choose action based on query, features, and context.

        Args:
            query: User query
            features: Extracted features
            context: Retrieved context

        Returns:
            ActionPlan with chosen tool and metadata
        """
        ...

    async def update(self, reward: float, metadata: Optional[Any] = None):
        """
        Update policy based on reward (for learning policies).

        Args:
            reward: Reward signal (0.0 to 1.0)
            metadata: Optional metadata about the outcome
        """
        ...


# ============================================================================
# Routing Strategy Protocol
# ============================================================================

@runtime_checkable
class RoutingStrategy(Protocol):
    """
    Protocol for routing strategies (mode selection).

    Determines which execution mode to use based on query characteristics.

    Implementations: StaticRouting, LearnedRouting, AdaptiveRouting
    """

    async def select_mode(
        self,
        query: Query,
        context: Optional[Any] = None
    ) -> str:
        """
        Select execution mode based on query.

        Args:
            query: User query
            context: Optional context

        Returns:
            Mode name: 'lite', 'fast', 'full', or 'research'
        """
        ...

    async def update(self, mode: str, latency: float, quality: float):
        """
        Update strategy based on outcomes (for learned routing).

        Args:
            mode: Mode that was used
            latency: Execution time in ms
            quality: Quality score (0.0 to 1.0)
        """
        ...


# ============================================================================
# Execution Engine Protocol
# ============================================================================

@runtime_checkable
class ExecutionEngine(Protocol):
    """
    Protocol for execution engines that run tools/actions.

    Executes the chosen actions and returns results.
    """

    async def execute(
        self,
        tool: str,
        args: Any,
        context: Optional[Context] = None
    ) -> Any:
        """
        Execute a tool with arguments.

        Args:
            tool: Tool name
            args: Tool arguments
            context: Optional context

        Returns:
            Tool execution result
        """
        ...


# ============================================================================
# Tool Registry Protocol
# ============================================================================

@runtime_checkable
class ToolRegistry(Protocol):
    """Protocol for tool registries."""

    def register(self, name: str, tool: Any):
        """Register a tool."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        ...

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        ...


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'Embedder',
    'MotifDetector',
    'PolicyEngine',
    'RoutingStrategy',
    'ExecutionEngine',
    'ToolRegistry',
]