"""
Ultra-Fast Optimizer (Level 1)

Updates: Every sub-query (~1-10ms)
Purpose: Agentic sub-query routing, token-level decisions, department self-resolution

Key Features:
- <1ms inference latency
- Lightweight MLP (no transformers)
- Exponential moving average for working memory
- Sub-query routing to agents/departments
- Token-level confidence gating for streaming

Use Cases:
- "Should I verify this claim?" → instant decision
- "Which department handles this?" → <1ms routing
- "Is this token confident enough?" → per-token gating
- "Which sub-question next?" → real-time prioritization

Philosophy:
"Not every decision needs deep reasoning. Ultra-fast decisions enable
fluid agentic reasoning and natural conversation flow."
"""

import time
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class AgentType(Enum):
    """Available agent types for routing"""
    HOLOLOOM_QUERY = "hololoom_query"  # Full weaving cycle
    MEMORY_SEARCH = "memory_search"    # Direct memory lookup
    VERIFICATION = "verification"       # Fact checking
    SYNTHESIS = "synthesis"             # Multi-source aggregation
    CODE_ANALYSIS = "code_analysis"     # Code-specific reasoning
    MATH_SOLVER = "math_solver"         # Mathematical reasoning
    CHAT = "chat"                       # Simple conversational response


class DecisionType(Enum):
    """Types of ultra-fast decisions"""
    ROUTE_SUBQUERY = "route_subquery"           # Which agent to invoke
    GATE_TOKEN = "gate_token"                   # Should we output this token?
    PRIORITIZE_SUBQUERY = "prioritize_subquery" # Which sub-question next?
    VERIFY_CLAIM = "verify_claim"               # Should we verify this?
    DEPARTMENT_ROUTE = "department_route"       # Which department?


@dataclass
class SubQueryContext:
    """Context for a sub-query decision"""
    text: str
    parent_query: str
    depth: int  # How many layers deep in recursion
    confidence_so_far: float  # Cumulative confidence
    tokens_generated: int  # For streaming
    time_elapsed_ms: float


@dataclass
class UltraFastDecision:
    """Result of an ultra-fast decision"""
    decision_type: DecisionType
    agent_type: AgentType | None
    confidence: float
    reasoning: str
    latency_ms: float
    working_memory_state: torch.Tensor


class WorkingMemory:
    """
    Volatile working memory for current query context.

    Decay: <1ms (replaced every query)
    Capacity: Single 244D vector (current features only)

    Uses exponential moving average (EMA) to blend incoming features:
        working_memory ← α * working_memory + (1-α) * new_features

    Fast decay (α=0.9) ensures working memory reflects current context.
    """

    def __init__(self, embedding_dim: int = 244, decay_rate: float = 0.9):
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.activations = torch.zeros(1, embedding_dim)
        self.last_update_time = time.time()

    def update(self, new_features: torch.Tensor) -> torch.Tensor:
        """
        Update working memory with new features (EMA).

        Args:
            new_features: [1, embedding_dim] new feature vector

        Returns:
            Updated working memory state
        """
        # Exponential moving average
        self.activations = (
            self.decay_rate * self.activations +
            (1 - self.decay_rate) * new_features
        )
        self.last_update_time = time.time()
        return self.activations

    def reset(self):
        """Reset working memory (new query starts)"""
        self.activations = torch.zeros(1, self.embedding_dim)
        self.last_update_time = time.time()

    def get_state(self) -> torch.Tensor:
        """Get current working memory state"""
        return self.activations.clone()

    def time_since_update_ms(self) -> float:
        """Time since last update in milliseconds"""
        return (time.time() - self.last_update_time) * 1000


class SubQueryRouter(nn.Module):
    """
    Lightweight MLP for sub-query routing.

    Architecture:
        Input (244D) → Hidden (128D) → Output (n_agents)

    Design Philosophy:
        - No attention (too slow)
        - No normalization (adds latency)
        - No dropout (inference only)
        - Simple ReLU activation

    Target Latency: <0.5ms on CPU
    """

    def __init__(
        self,
        input_dim: int = 244,
        hidden_dim: int = 128,
        n_agents: int = 7
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        # Lightweight 2-layer MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents)
        )

    def forward(self, working_memory: torch.Tensor) -> torch.Tensor:
        """
        Route sub-query to agent.

        Args:
            working_memory: [1, input_dim] current working memory

        Returns:
            logits: [1, n_agents] routing logits
        """
        return self.network(working_memory)

    def select_agent(
        self,
        logits: torch.Tensor,
        temperature: float = 0.1
    ) -> tuple[AgentType, float]:
        """
        Select agent from logits.

        Args:
            logits: [1, n_agents] routing logits
            temperature: Lower = more confident (0.1 for near-deterministic)

        Returns:
            (agent_type, confidence)
        """
        probs = torch.softmax(logits / temperature, dim=-1)
        agent_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, agent_idx].item()

        agent_types = list(AgentType)
        return agent_types[agent_idx], confidence


class TokenGate(nn.Module):
    """
    Token-level confidence gating for streaming responses.

    During streaming generation, decides whether each token is confident
    enough to output or should be held back for verification.

    Architecture: Single linear layer (no hidden layers for speed)
        Input (244D working memory) → Output (1D confidence)

    Target Latency: <0.1ms per token
    """

    def __init__(self, input_dim: int = 244):
        super().__init__()
        self.gate = nn.Linear(input_dim, 1)

    def forward(self, working_memory: torch.Tensor) -> float:
        """
        Gate token output.

        Args:
            working_memory: [1, input_dim] current state

        Returns:
            confidence: float in [0, 1]
        """
        logit = self.gate(working_memory)
        confidence = torch.sigmoid(logit).item()
        return confidence

    def should_output_token(
        self,
        working_memory: torch.Tensor,
        threshold: float = 0.7
    ) -> bool:
        """
        Decide whether to output token.

        Args:
            working_memory: Current working memory state
            threshold: Minimum confidence to output

        Returns:
            True if token should be output
        """
        confidence = self.forward(working_memory)
        return confidence >= threshold


class UltraFastOptimizer:
    """
    Level 1: Ultra-Fast Optimizer

    Updates: Every sub-query (~1-10ms)
    Latency: <1ms per decision

    Responsibilities:
    1. Sub-query routing (which agent/department)
    2. Token-level gating (streaming confidence)
    3. Verification triggering (should we fact-check?)
    4. Sub-query prioritization (which question next?)

    Design Philosophy:
        Speed over accuracy. Better to be 90% right in <1ms than
        99% right in 10ms. Agentic reasoning needs fluidity.

    Example Usage:
        >>> optimizer = UltraFastOptimizer(embedding_dim=244)
        >>> decision = await optimizer.route_subquery(
        ...     text="Is this claim accurate?",
        ...     parent_query="What is Thompson Sampling?",
        ...     working_memory_state=current_features
        ... )
        >>> print(decision.agent_type)  # AgentType.VERIFICATION
        >>> print(decision.latency_ms)  # ~0.5ms
    """

    def __init__(
        self,
        embedding_dim: int = 244,
        hidden_dim: int = 128,
        n_agents: int = 7,
        decay_rate: float = 0.9
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        # Working memory (volatile)
        self.working_memory = WorkingMemory(embedding_dim, decay_rate)

        # Sub-query router
        self.router = SubQueryRouter(embedding_dim, hidden_dim, n_agents)

        # Token gate
        self.token_gate = TokenGate(embedding_dim)

        # Statistics tracking
        self.stats = {
            'total_decisions': 0,
            'decisions_by_type': dict.fromkeys(DecisionType, 0),
            'average_latency_ms': 0.0,
            'routing_distribution': dict.fromkeys(AgentType, 0)
        }

    async def route_subquery(
        self,
        text: str,
        parent_query: str,
        working_memory_state: torch.Tensor,
        depth: int = 0,
        confidence_so_far: float = 0.0
    ) -> UltraFastDecision:
        """
        Route sub-query to appropriate agent (ultra-fast).

        Args:
            text: Sub-query text
            parent_query: Parent query for context
            working_memory_state: Current feature encoding
            depth: Recursion depth
            confidence_so_far: Cumulative confidence

        Returns:
            UltraFastDecision with routing information
        """
        start_time = time.perf_counter()

        # Update working memory
        self.working_memory.update(working_memory_state)

        # Route via lightweight MLP
        logits = self.router(self.working_memory.get_state())
        agent_type, confidence = self.router.select_agent(logits)

        # Record latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self._update_stats(DecisionType.ROUTE_SUBQUERY, agent_type, latency_ms)

        # Build decision
        decision = UltraFastDecision(
            decision_type=DecisionType.ROUTE_SUBQUERY,
            agent_type=agent_type,
            confidence=confidence,
            reasoning=f"Routed to {agent_type.value} (depth={depth}, conf={confidence:.2f})",
            latency_ms=latency_ms,
            working_memory_state=self.working_memory.get_state()
        )

        return decision

    async def gate_token(
        self,
        token: str,
        working_memory_state: torch.Tensor,
        threshold: float = 0.7
    ) -> UltraFastDecision:
        """
        Decide whether to output token (streaming).

        Args:
            token: Token to potentially output
            working_memory_state: Current state
            threshold: Minimum confidence

        Returns:
            UltraFastDecision with gating information
        """
        start_time = time.perf_counter()

        # Update working memory
        self.working_memory.update(working_memory_state)

        # Gate token
        should_output = self.token_gate.should_output_token(
            self.working_memory.get_state(),
            threshold
        )
        confidence = self.token_gate.forward(self.working_memory.get_state())

        # Record latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self._update_stats(DecisionType.GATE_TOKEN, None, latency_ms)

        decision = UltraFastDecision(
            decision_type=DecisionType.GATE_TOKEN,
            agent_type=None,
            confidence=confidence,
            reasoning=f"Token '{token}' {'PASSED' if should_output else 'BLOCKED'} gate (conf={confidence:.2f})",
            latency_ms=latency_ms,
            working_memory_state=self.working_memory.get_state()
        )

        return decision

    async def should_verify(
        self,
        claim: str,
        working_memory_state: torch.Tensor,
        verification_threshold: float = 0.6
    ) -> UltraFastDecision:
        """
        Decide whether claim needs verification.

        Uses working memory state to detect:
        - Factual claims (need verification)
        - Opinions (skip verification)
        - Already-verified facts (skip)

        Args:
            claim: Claim to potentially verify
            working_memory_state: Current state
            verification_threshold: Trigger threshold

        Returns:
            UltraFastDecision with verification recommendation
        """
        start_time = time.perf_counter()

        # Update working memory
        self.working_memory.update(working_memory_state)

        # Simple heuristic: Route to VERIFICATION agent
        # In practice, this would use learned patterns
        logits = self.router(self.working_memory.get_state())
        agent_type, confidence = self.router.select_agent(logits)

        # Trigger verification if agent suggests it OR confidence low
        should_verify = (
            agent_type == AgentType.VERIFICATION or
            confidence < verification_threshold
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(DecisionType.VERIFY_CLAIM, None, latency_ms)

        decision = UltraFastDecision(
            decision_type=DecisionType.VERIFY_CLAIM,
            agent_type=AgentType.VERIFICATION if should_verify else None,
            confidence=confidence,
            reasoning=f"Verification {'NEEDED' if should_verify else 'SKIPPED'} (conf={confidence:.2f})",
            latency_ms=latency_ms,
            working_memory_state=self.working_memory.get_state()
        )

        return decision

    async def prioritize_subqueries(
        self,
        subqueries: list[str],
        working_memory_state: torch.Tensor
    ) -> list[tuple[str, float]]:
        """
        Prioritize sub-queries for execution order.

        Returns sub-queries ranked by urgency/relevance.

        Args:
            subqueries: List of sub-query texts
            working_memory_state: Current state

        Returns:
            List of (subquery, priority_score) sorted by priority
        """
        start_time = time.perf_counter()

        # Update working memory
        self.working_memory.update(working_memory_state)

        # Score each sub-query
        # In practice, this would use learned scoring
        # For now, use simple heuristic (similarity to working memory)
        priorities = []
        for subquery in subqueries:
            # Mock scoring (replace with learned model)
            score = torch.rand(1).item()  # Placeholder
            priorities.append((subquery, score))

        # Sort by priority
        priorities.sort(key=lambda x: x[1], reverse=True)

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(DecisionType.PRIORITIZE_SUBQUERY, None, latency_ms)

        return priorities

    def reset_working_memory(self):
        """Reset working memory (new query session)"""
        self.working_memory.reset()

    def get_statistics(self) -> dict:
        """Get optimizer statistics"""
        return self.stats.copy()

    def _update_stats(
        self,
        decision_type: DecisionType,
        agent_type: AgentType | None,
        latency_ms: float
    ):
        """Update internal statistics"""
        self.stats['total_decisions'] += 1
        self.stats['decisions_by_type'][decision_type] += 1

        # Update average latency (running average)
        n = self.stats['total_decisions']
        old_avg = self.stats['average_latency_ms']
        self.stats['average_latency_ms'] = (old_avg * (n - 1) + latency_ms) / n

        # Update routing distribution
        if agent_type:
            self.stats['routing_distribution'][agent_type] += 1

    def summary(self) -> str:
        """Human-readable statistics summary"""
        stats = self.stats
        lines = [
            "UltraFastOptimizer Statistics",
            "=" * 40,
            f"Total Decisions: {stats['total_decisions']}",
            f"Average Latency: {stats['average_latency_ms']:.3f}ms",
            "",
            "Decisions by Type:",
        ]

        for dt, count in stats['decisions_by_type'].items():
            if count > 0:
                lines.append(f"  {dt.value}: {count}")

        lines.append("")
        lines.append("Routing Distribution:")

        for at, count in stats['routing_distribution'].items():
            if count > 0:
                pct = 100 * count / stats['total_decisions']
                lines.append(f"  {at.value}: {count} ({pct:.1f}%)")

        return "\n".join(lines)


# Convenience functions for quick usage

async def route_subquery_fast(
    text: str,
    parent_query: str,
    optimizer: UltraFastOptimizer | None = None
) -> UltraFastDecision:
    """
    Quick sub-query routing without manual memory management.

    Usage:
        >>> decision = await route_subquery_fast(
        ...     "Is this claim accurate?",
        ...     "What is Thompson Sampling?"
        ... )
    """
    if optimizer is None:
        optimizer = UltraFastOptimizer()

    # Mock working memory (replace with actual encoding in production)
    working_memory = torch.randn(1, 244)

    return await optimizer.route_subquery(
        text, parent_query, working_memory
    )


async def should_verify_fast(
    claim: str,
    optimizer: UltraFastOptimizer | None = None
) -> bool:
    """
    Quick verification check.

    Usage:
        >>> if await should_verify_fast("Python is 30 years old"):
        ...     result = await verify_claim(claim)
    """
    if optimizer is None:
        optimizer = UltraFastOptimizer()

    working_memory = torch.randn(1, 244)
    decision = await optimizer.should_verify(claim, working_memory)

    return decision.agent_type == AgentType.VERIFICATION
