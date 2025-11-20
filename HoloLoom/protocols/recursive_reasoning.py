<<<<<<< HEAD
"""
Recursive Reasoning Protocols
==============================

Protocol definitions for recursive reasoning and self-improving refinement loops.

Created: 2025-11-20
Author: HoloLoom Team
"""

from typing import Protocol, List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================

class ReasoningStrategy(Enum):
    """Available recursive reasoning strategies."""
    REFINE = "refine"           # Iterative refinement
    CRITIQUE = "critique"       # Self-critique and improve
    VERIFY = "verify"           # Multi-pass verification
    ELEGANCE = "elegance"       # Polish for clarity/simplicity
    DECOMPOSE = "decompose"     # Break into sub-queries
    EXPLORE = "explore"         # Multiple perspectives
    HOFSTADTER = "hofstadter"   # Strange loops / meta-reasoning


class StopCondition(Enum):
    """Reasons for stopping recursive reasoning."""
    QUALITY_THRESHOLD = "quality_threshold"  # Target quality reached
    MAX_ITERATIONS = "max_iterations"         # Iteration limit reached
    NO_IMPROVEMENT = "no_improvement"         # Convergence (plateau)
    TIME_BUDGET = "time_budget"               # Time limit exceeded
    COMPLEXITY_LIMIT = "complexity_limit"     # Query too complex


class RefinementStrategy(Enum):
    """Refinement strategies for low-confidence responses."""
    EXPAND_SEARCH = "expand_search"           # Retrieve more sources
    RERANK = "rerank"                         # Better source ranking
    ALTERNATE_MODE = "alternate_mode"         # Try different reasoning mode
    DECOMPOSE = "decompose"                   # Break into sub-queries
    VERIFY_AND_CORRECT = "verify_and_correct" # Check and fix errors
    MULTI_PERSPECTIVE = "multi_perspective"   # Multiple angles


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReasoningTrace:
    """Single step in reasoning process."""
    iteration: int
    thought: str
    action: str
    observation: str
    confidence: float
    features_extracted: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementStep:
    """Single refinement step in recursive reasoning."""
    iteration: int
    query: str
    response: str
    confidence: float
    improvement: float  # Improvement from previous step
    strategy_used: str  # Which refinement strategy
    reasoning: str      # Why this refinement was needed
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionTree:
    """Tree of decomposed queries."""
    root_query: str
    sub_queries: List[str]
    depth: int
    complexity_score: float
    synthesis_strategy: str  # How to combine sub-answers
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
=======
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

from HoloLoom.documentation.types import Features, Query, Spacetime


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
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0


@dataclass
class ReasoningJournal:
<<<<<<< HEAD
    """Complete journal of reasoning process."""
    strategy_used: ReasoningStrategy
    traces: List[ReasoningTrace] = field(default_factory=list)
    final_spacetime: Optional[Any] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    def add_trace(self, trace: ReasoningTrace):
        """Add trace to journal."""
        self.traces.append(trace)

    def get_latest(self) -> Optional[ReasoningTrace]:
        """Get latest trace."""
        return self.traces[-1] if self.traces else None

    def converged(self, min_improvement: float) -> bool:
        """Check if reasoning has converged (no improvement)."""
        if len(self.traces) < 2:
            return False

        # Check last 2 iterations
        recent = self.traces[-2:]
        improvement = recent[-1].confidence - recent[-2].confidence

        return improvement < min_improvement

    def get_trajectory(self) -> List[float]:
        """Get confidence trajectory."""
        return [trace.confidence for trace in self.traces]


@dataclass
class RecursiveResult:
    """Result from recursive reasoning."""
    final_response: str
    final_confidence: float
    iterations: int
    refinement_steps: List[RefinementStep]
    convergence_reason: str
    total_latency_ms: float
    improvement_trajectory: List[float]  # Confidence at each step
    journal: ReasoningJournal
    decomposition: Optional[DecompositionTree] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
=======
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
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0


@dataclass
class RecursiveConfig:
<<<<<<< HEAD
    """Configuration for recursive reasoning."""
    strategy: ReasoningStrategy = ReasoningStrategy.REFINE
    max_iterations: int = 5
    quality_threshold: float = 0.85
    min_improvement: float = 0.05
    time_budget_ms: Optional[float] = None
    enable_decomposition: bool = True
    enable_learning: bool = True
    parallel_explore_branches: int = 3  # For EXPLORE strategy

    # Convergence detection
    convergence_strategy: str = "multi_criteria"  # multi_criteria, confidence, improvement

    # Refinement
    refinement_threshold: float = 0.75  # Auto-refine if confidence < this
    max_refinement_iterations: int = 3


# ============================================================================
# Type Aliases
# ============================================================================

StopDecision = Tuple[bool, Optional[StopCondition]]
RefinementResult = Tuple[Any, ReasoningTrace]  # (improved_spacetime, trace)
ReasoningResult = Tuple[Any, ReasoningJournal]  # (final_spacetime, journal)


# ============================================================================
# Protocols
# ============================================================================

class RecursiveReasoningProtocol(Protocol):
    """Protocol for recursive reasoning implementations."""

    async def reason(
        self,
        query: Any,
        initial_features: Any,
        config: RecursiveConfig
    ) -> ReasoningResult:
=======
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
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0
        """
        Execute recursive reasoning loop.

        Args:
<<<<<<< HEAD
            query: Query to reason about
            initial_features: Initial extracted features
            config: Reasoning configuration

        Returns:
            (final_spacetime, journal) tuple
        """
        ...

=======
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
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0
    def should_continue(
        self,
        journal: ReasoningJournal,
        config: RecursiveConfig
<<<<<<< HEAD
    ) -> StopDecision:
        """
        Check if reasoning should continue.

        Args:
            journal: Current reasoning journal
            config: Reasoning configuration

        Returns:
            (should_continue, stop_reason) tuple
        """
        ...

    async def refine_iteration(
        self,
        previous_spacetime: Any,
        journal: ReasoningJournal,
        config: RecursiveConfig
    ) -> RefinementResult:
        """
        Execute single refinement iteration.

        Args:
            previous_spacetime: Previous iteration result
            journal: Current reasoning journal
            config: Reasoning configuration

        Returns:
            (improved_spacetime, trace) tuple
=======
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
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0
        """
        ...


class QualityScoringProtocol(Protocol):
<<<<<<< HEAD
    """Protocol for quality scoring implementations."""

    async def score(self, spacetime: Any, query: Any) -> float:
        """
        Score quality of spacetime result.

        Args:
            spacetime: Result to score
            query: Original query

        Returns:
            Quality score (0.0-1.0)
=======
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
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0
        """
        ...


<<<<<<< HEAD
class ConvergenceDetectorProtocol(Protocol):
    """Protocol for convergence detection."""

    def detect(self, history: List[RefinementStep]) -> Tuple[bool, str]:
        """
        Detect if refinement has converged.

        Args:
            history: List of refinement steps

        Returns:
            (converged, reason) tuple
        """
        ...


class QueryDecomposerProtocol(Protocol):
    """Protocol for query decomposition."""

    async def decompose(
        self,
        query: str,
        complexity_threshold: float = 0.7
    ) -> DecompositionTree:
        """
        Decompose query into sub-queries.

        Args:
            query: Query to decompose
            complexity_threshold: Minimum complexity to trigger decomposition

        Returns:
            DecompositionTree with sub-queries
        """
        ...

    def detect_complexity(self, query: str) -> float:
        """
        Estimate query complexity.

        Args:
            query: Query to analyze

        Returns:
            Complexity score (0.0-1.0)
        """
        ...

    async def synthesize_sub_answers(
        self,
        sub_results: List[Any]
    ) -> str:
        """
        Combine sub-query answers into final answer.

        Args:
            sub_results: List of sub-query results

        Returns:
            Synthesized final answer
        """
        ...


class StrategySelec torProtocol(Protocol):
    """Protocol for refinement strategy selection."""

    async def select_strategy(
        self,
        query: str,
        current_confidence: float,
        refinement_history: List[RefinementStep]
    ) -> RefinementStrategy:
        """
        Select optimal refinement strategy.

        Args:
            query: Current query
            current_confidence: Current confidence score
            refinement_history: History of refinements

        Returns:
            Selected refinement strategy
        """
        ...

    async def learn_from_outcome(
        self,
        strategy: RefinementStrategy,
        improvement: float,
        query_type: str
    ):
        """
        Learn from refinement outcome.

        Args:
            strategy: Strategy that was used
            improvement: Quality improvement achieved
            query_type: Type of query (factual, procedural, etc.)
        """
        ...
=======
# Type aliases for convenience
ReasoningResult = tuple[Spacetime, ReasoningJournal]
StopDecision = tuple[bool, StopCondition]
RefinementResult = tuple[Spacetime, ReasoningTrace]
>>>>>>> 5e8972dfa632a6a2091c36c692d1a6443ceb4ab0
