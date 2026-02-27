#!/usr/bin/env python3
"""
Recursive Reasoning Protocol - Promptly Integration
===================================================

Defines the protocol for recursive reasoning loops integrated into HoloLoom's
weaving architecture. Inspired by Promptly's recursive intelligence system.

Created: 2025-11-15
Integration: Promptly → HoloLoom convergence layer
"""

from typing import Protocol, List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import abstractmethod

from hololoom.core.protocols.types import Features, Query
from hololoom.core.fabric.spacetime import Spacetime


class ReasoningStrategy(Enum):
    """Recursive reasoning strategies (from Promptly's LoopType)"""

    # Linear strategies (HoloLoom native)
    DIRECT = "direct"           # Single-pass answer (HoloLoom's LITE mode)
    VERIFY = "verify"           # Answer + verification (HoloLoom's FAST mode)

    # Recursive strategies (from Promptly)
    REFINE = "refine"           # Iterative refinement until quality threshold
    CRITIQUE = "critique"       # Self-critique → improve → repeat
    DECOMPOSE = "decompose"     # Break down → solve → synthesize
    EXPLORE = "explore"         # Multiple approaches → select best
    HOFSTADTER = "hofstadter"   # Strange loops (meta-reasoning)

    # Hybrid strategies (new)
    ADAPTIVE = "adaptive"       # Auto-select strategy based on query complexity


class StopCondition(Enum):
    """When to stop recursive refinement"""
    MAX_ITERATIONS = "max_iterations"       # Hard limit on passes
    QUALITY_THRESHOLD = "quality_threshold" # Confidence >= target
    NO_IMPROVEMENT = "no_improvement"       # Δconfidence < min_improvement
    CONVERGENCE = "convergence"             # Output stabilizes
    BUDGET_EXCEEDED = "budget_exceeded"     # Cost/time limit reached


@dataclass
class ReasoningTrace:
    """Single iteration in reasoning process (Promptly's ScratchpadEntry)"""
    iteration: int
    thought: str                    # Internal reasoning
    action: str                     # What the system did
    observation: str                # Result of action
    confidence: float               # Quality score (0.0-1.0)
    features_extracted: Features    # Features at this iteration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Human-readable representation"""
        return f"""
## Iteration {self.iteration}

**Thought:** {self.thought}
**Action:** {self.action}
**Observation:** {self.observation}
**Confidence:** {self.confidence:.2f}
"""


@dataclass
class ReasoningJournal:
    """Complete reasoning provenance (Promptly's Scratchpad)"""
    traces: List[ReasoningTrace] = field(default_factory=list)
    final_spacetime: Optional[Spacetime] = None
    strategy_used: Optional[ReasoningStrategy] = None

    def add_trace(self, trace: ReasoningTrace):
        """Add iteration trace"""
        self.traces.append(trace)

    def get_history(self) -> str:
        """Full reasoning history"""
        return "\n".join(t.to_text() for t in self.traces)

    def get_latest(self) -> Optional[ReasoningTrace]:
        """Most recent iteration"""
        return self.traces[-1] if self.traces else None

    def get_confidence_trajectory(self) -> List[float]:
        """Confidence over time"""
        return [t.confidence for t in self.traces]

    def converged(self, min_improvement: float = 0.05) -> bool:
        """Check if reasoning has converged"""
        if len(self.traces) < 2:
            return False

        recent = self.traces[-2:]
        delta = recent[1].confidence - recent[0].confidence
        return delta < min_improvement


@dataclass
class RecursiveConfig:
    """Configuration for recursive reasoning"""
    strategy: ReasoningStrategy = ReasoningStrategy.REFINE
    max_iterations: int = 5
    quality_threshold: float = 0.9
    min_improvement: float = 0.05
    time_budget_ms: Optional[float] = None
    cost_budget: Optional[float] = None
    enable_journal: bool = True  # Track provenance

    # Advanced settings
    parallel_explore_branches: int = 3  # For EXPLORE strategy
    hofstadter_depth: int = 3           # For HOFSTADTER strategy


class RecursiveReasoningProtocol(Protocol):
    """
    Protocol for recursive reasoning implementations.

    All recursive reasoners must implement this interface to integrate
    with HoloLoom's convergence engine.
    """

    @abstractmethod
    async def reason(
        self,
        query: Query,
        initial_features: Features,
        config: RecursiveConfig
    ) -> tuple[Spacetime, ReasoningJournal]:
        """
        Execute recursive reasoning loop.

        Args:
            query: User's query
            initial_features: Features from first pass
            config: Recursive reasoning configuration

        Returns:
            (final_spacetime, reasoning_journal)
            - final_spacetime: Best result after refinement
            - reasoning_journal: Complete provenance of reasoning process
        """
        ...

    @abstractmethod
    def should_continue(
        self,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> tuple[bool, StopCondition]:
        """
        Determine if recursive loop should continue.

        Args:
            journal: Current reasoning history
            config: Configuration with thresholds

        Returns:
            (should_continue, reason)
            - should_continue: True if more iterations needed
            - reason: Which stop condition triggered (if stopping)
        """
        ...

    @abstractmethod
    async def refine_iteration(
        self,
        previous_spacetime: Spacetime,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> tuple[Spacetime, ReasoningTrace]:
        """
        Execute one refinement iteration.

        Args:
            previous_spacetime: Result from last iteration
            journal: History so far
            config: Refinement parameters

        Returns:
            (improved_spacetime, trace)
            - improved_spacetime: Refined result
            - trace: Provenance for this iteration
        """
        ...


class QualityScoringProtocol(Protocol):
    """Protocol for scoring spacetime quality (Promptly's auto-scoring)"""

    @abstractmethod
    async def score(self, spacetime: Spacetime, query: Query) -> float:
        """
        Score spacetime quality (0.0-1.0).

        Factors:
        - Response completeness
        - Context relevance
        - Factual accuracy
        - Consistency with query intent

        Returns:
            Quality score 0.0 (poor) to 1.0 (excellent)
        """
        ...


# Type aliases for convenience
ReasoningResult = tuple[Spacetime, ReasoningJournal]
StopDecision = tuple[bool, StopCondition]
RefinementResult = tuple[Spacetime, ReasoningTrace]
