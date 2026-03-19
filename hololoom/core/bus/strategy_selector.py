"""
Strategy Selector — MCTS-powered execution strategy selection.
===============================================================

When a QUERY signal arrives, the strategic MCTS explores the space of
possible execution strategies (chain patterns, workflow templates,
direct pass) and selects the one most likely to succeed for this query's
domain.

Composes:
- MCTSEngine + MCTSStateSpace from hololoom/agents/mcts_core.py
- ChainPatterns from hololoom/chaining/patterns.py (17 strategies)
- SignalJournal historical performance data
- Bus signal flow (subscribes to QUERY, emits MCTS_DECISION)

The strategic MCTS learns over time:
- After each execution completes, the result (success, confidence,
  duration) feeds back as reward signal
- Domain-strategy associations strengthen with successful executions
- Thompson Sampling from model_router.py is preserved for model selection
  within strategies — MCTS handles strategy selection above that

Usage:
    from hololoom.core.bus import UnifiedBus
    from hololoom.core.bus.strategy_selector import StrategySelector

    bus = UnifiedBus(journal=journal)
    selector = StrategySelector(bus)
    # Automatically subscribes to QUERY signals and emits MCTS_DECISION
"""

import logging
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hololoom.agents.mcts_core import MCTSEngine, MCTSNode, MCTSStateSpace

from .signal import Signal, SignalKind, SignalPriority
from .unified_bus import UnifiedBus

logger = logging.getLogger(__name__)


# ============================================================================
# Strategy Definitions
# ============================================================================

@dataclass
class Strategy:
    """An execution strategy that MCTS can select."""
    name: str
    execution_type: str     # "chain", "workflow", "direct"
    description: str
    default_simulations: int = 50
    # Historical performance (updated by backprop)
    successes: float = 1.0  # Beta prior alpha
    failures: float = 1.0   # Beta prior beta
    total_confidence: float = 0.0
    total_duration_ms: float = 0.0
    uses: int = 0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.uses if self.uses > 0 else 0.5

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.uses if self.uses > 0 else 1000.0


# Default strategies mapped from ChainPatterns
DEFAULT_STRATEGIES = [
    Strategy("simple_query", "chain", "Execute only — fastest, no verification"),
    Strategy("verified_query", "chain", "Execute then verify — balanced quality/speed"),
    Strategy("auto_refine", "chain", "Execute, verify, refine if low confidence"),
    Strategy("iterative_improve", "chain", "Loop execute/verify until high confidence"),
    Strategy("research_pipeline", "chain", "Full research cycle with learning"),
    Strategy("quality_first", "chain", "Prioritize quality over speed"),
    Strategy("quick_answer", "chain", "Minimum latency, skip verification"),
    Strategy("fact_check", "chain", "Extract claims, verify, verdict"),
    Strategy("code_review", "chain", "Security scan, style check, fixes"),
    Strategy("direct_pass", "direct", "No chain — single LLM call"),
]


# ============================================================================
# MCTS State Space for Strategy Selection
# ============================================================================

@dataclass
class StrategyState:
    """State in the strategy selection search space."""
    query_text: str = ""
    query_domain: str = ""      # classified domain (code, chat, reasoning, etc.)
    speed_weight: float = 0.5   # 0 = quality, 1 = speed
    selected: str | None = None  # strategy name (terminal when set)


class StrategyStateSpace(MCTSStateSpace):
    """
    MCTS state space where actions are execution strategies.

    Evaluation uses historical performance data to guide selection.
    Over time, the tree learns which strategies work for which domains.
    """

    def __init__(self, strategies: list[Strategy]):
        self.strategies = {s.name: s for s in strategies}
        self._strategy_names = [s.name for s in strategies]

    def get_legal_actions(self, state: StrategyState) -> list[str]:
        """All available strategy names."""
        if state.selected is not None:
            return []  # Terminal
        return self._strategy_names

    def apply_action(self, state: StrategyState, action: str) -> StrategyState:
        """Select a strategy."""
        new_state = deepcopy(state)
        new_state.selected = action
        return new_state

    async def evaluate(self, state: StrategyState) -> float:
        """
        Evaluate strategy selection quality.

        Combines:
        - Historical success rate (exploitation)
        - Speed/quality tradeoff alignment
        - Domain-specific performance
        """
        if state.selected is None:
            return 0.5  # Neutral for non-terminal states

        strategy = self.strategies.get(state.selected)
        if not strategy:
            return 0.0

        # Base score from historical success rate
        score = strategy.success_rate * 0.5

        # Confidence contribution
        score += strategy.avg_confidence * 0.3

        # Speed alignment (penalize slow strategies when speed is preferred)
        if state.speed_weight > 0.7:
            # User wants speed — prefer simple/quick strategies
            speed_bonus = {
                "quick_answer": 0.2,
                "simple_query": 0.15,
                "direct_pass": 0.2,
                "verified_query": 0.05,
            }.get(state.selected, 0.0)
            score += speed_bonus
        elif state.speed_weight < 0.3:
            # User wants quality — prefer thorough strategies
            quality_bonus = {
                "research_pipeline": 0.2,
                "quality_first": 0.2,
                "iterative_improve": 0.15,
                "auto_refine": 0.1,
            }.get(state.selected, 0.0)
            score += quality_bonus

        # Domain-specific bonuses
        domain = state.query_domain
        if domain == "code":
            if state.selected == "code_review":
                score += 0.15
        elif domain == "factual":
            if state.selected == "fact_check":
                score += 0.15

        return min(1.0, max(0.0, score))

    def is_terminal(self, state: StrategyState) -> bool:
        """Terminal when a strategy has been selected."""
        return state.selected is not None

    def copy_state(self, state: StrategyState) -> StrategyState:
        return deepcopy(state)


# ============================================================================
# Domain Classifier (lightweight, from model_router.py pattern)
# ============================================================================

_DOMAIN_KEYWORDS = {
    "code": {"code", "function", "class", "bug", "error", "implement", "refactor",
             "python", "javascript", "typescript", "rust", "api", "debug", "test"},
    "factual": {"what", "when", "where", "who", "how many", "define", "explain",
                "history", "fact", "capital", "population"},
    "reasoning": {"why", "analyze", "compare", "evaluate", "tradeoff", "design",
                  "architecture", "strategy", "plan", "think"},
    "creative": {"write", "story", "poem", "imagine", "creative", "brainstorm",
                 "generate", "draft", "compose"},
    "chat": {"hello", "hi", "thanks", "hey", "how are you", "goodbye"},
}


def classify_domain(query: str) -> tuple[str, float]:
    """
    Lightweight domain classification from query text.
    Mirrors model_router.py intent classifier.

    Returns:
        (domain, confidence) tuple
    """
    query_lower = query.lower()
    words = set(query_lower.split())

    best_domain = "general"
    best_score = 0

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        overlap = len(words & keywords)
        if overlap > best_score:
            best_score = overlap
            best_domain = domain

    confidence = min(1.0, best_score / 3.0) if best_score > 0 else 0.3
    return best_domain, confidence


# ============================================================================
# Strategy Selector (Bus-Connected)
# ============================================================================

class StrategySelector:
    """
    Bus-connected strategic MCTS for execution strategy selection.

    Subscribes to QUERY signals, runs MCTS search, emits MCTS_DECISION.
    Subscribes to EXECUTION_COMPLETE for backpropagation (learning).

    The selector learns over time which strategies work best for which
    domains, speed preferences, and query types.
    """

    def __init__(
        self,
        bus: UnifiedBus,
        strategies: list[Strategy] | None = None,
        simulations: int = 50,
        auto_subscribe: bool = True,
    ):
        """
        Args:
            bus: UnifiedBus to subscribe to
            strategies: Custom strategies (default: DEFAULT_STRATEGIES)
            simulations: MCTS simulations per decision
            auto_subscribe: Auto-subscribe to QUERY and EXECUTION_COMPLETE
        """
        self.bus = bus
        self.strategies = {
            s.name: s for s in (strategies or DEFAULT_STRATEGIES)
        }
        self.simulations = simulations

        # MCTS engine with strategy state space
        self._state_space = StrategyStateSpace(
            list(self.strategies.values())
        )
        self._engine = MCTSEngine(
            state_space=self._state_space,
            exploration_weight=1.414,
            max_depth=2,  # Shallow: select one strategy
        )

        # Track pending decisions (correlation_id -> selected strategy)
        self._pending: dict[str, str] = {}

        if auto_subscribe:
            self.bus.on(SignalKind.QUERY, self._on_query)
            self.bus.on(SignalKind.EXECUTION_COMPLETE, self._on_execution_complete)
            self.bus.on(SignalKind.EXECUTION_FAILED, self._on_execution_failed)

    async def select_strategy(
        self,
        query_text: str,
        speed_weight: float = 0.5,
    ) -> tuple[str, dict[str, Any]]:
        """
        Select the best execution strategy for a query.

        Args:
            query_text: The query text
            speed_weight: 0.0 = quality, 1.0 = speed

        Returns:
            (strategy_name, decision_metadata)
        """
        # Classify domain
        domain, domain_confidence = classify_domain(query_text)

        # Build initial state
        initial_state = StrategyState(
            query_text=query_text,
            query_domain=domain,
            speed_weight=speed_weight,
        )

        # Run MCTS search
        start_time = time.time()
        best_action, root = await self._engine.search(
            initial_state,
            n_simulations=self.simulations,
        )
        search_time_ms = (time.time() - start_time) * 1000

        # Fallback to verified_query if MCTS fails
        strategy_name = best_action or "verified_query"

        # Build decision metadata
        metadata = {
            "strategy": strategy_name,
            "domain": domain,
            "domain_confidence": domain_confidence,
            "speed_weight": speed_weight,
            "simulations_run": self.simulations,
            "search_time_ms": search_time_ms,
            "root_visits": root.visits,
            "alternatives": self._get_alternatives(root),
        }

        return strategy_name, metadata

    async def _on_query(self, signal: Signal) -> Signal | None:
        """Handle QUERY signal — select strategy and emit MCTS_DECISION."""
        query_text = signal.payload.get("text", "")
        speed_weight = signal.payload.get("speed_weight", 0.5)

        strategy_name, metadata = await self.select_strategy(
            query_text=query_text,
            speed_weight=speed_weight,
        )

        # Track the pending decision
        correlation_id = signal.correlation_id or signal.id
        self._pending[correlation_id] = strategy_name

        # Record to journal
        if self.bus._journal:
            await self.bus._journal.record_mcts_decision(
                decision_id=uuid.uuid4().hex[:12],
                tree_level="strategic",
                created_at=datetime.now().isoformat(),
                correlation_id=correlation_id,
                domain=metadata["domain"],
                simulations_run=metadata["simulations_run"],
                selected_action=strategy_name,
                root_visits=metadata["root_visits"],
                best_value=self.strategies[strategy_name].success_rate,
                alternatives=metadata["alternatives"],
            )

        # Emit MCTS_DECISION
        return Signal(
            kind=SignalKind.MCTS_DECISION,
            priority=SignalPriority.NORMAL,
            source="strategy_selector",
            payload=metadata,
        )

    async def _on_execution_complete(self, signal: Signal) -> Signal | None:
        """Handle EXECUTION_COMPLETE — backprop reward to strategy."""
        correlation_id = signal.correlation_id
        if not correlation_id or correlation_id not in self._pending:
            return None

        strategy_name = self._pending.pop(correlation_id)
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return None

        # Update strategy performance
        confidence = signal.payload.get("confidence", 0.5)
        duration_ms = signal.payload.get("duration_ms", 0.0)

        strategy.successes += 1.0
        strategy.total_confidence += confidence
        strategy.total_duration_ms += duration_ms
        strategy.uses += 1

        logger.info(
            f"Strategy '{strategy_name}' succeeded "
            f"(rate={strategy.success_rate:.2f}, "
            f"avg_conf={strategy.avg_confidence:.2f})"
        )

        return None

    async def _on_execution_failed(self, signal: Signal) -> Signal | None:
        """Handle EXECUTION_FAILED — backprop negative reward."""
        correlation_id = signal.correlation_id
        if not correlation_id or correlation_id not in self._pending:
            return None

        strategy_name = self._pending.pop(correlation_id)
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            return None

        strategy.failures += 1.0
        strategy.uses += 1

        logger.info(
            f"Strategy '{strategy_name}' failed "
            f"(rate={strategy.success_rate:.2f})"
        )

        return None

    def _get_alternatives(self, root: MCTSNode) -> list[dict[str, Any]]:
        """Get alternative strategies considered by MCTS."""
        alternatives = []
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            if child.action:
                alternatives.append({
                    "strategy": child.action,
                    "visits": child.visits,
                    "value": round(child.value(), 3),
                })
        return alternatives[:5]  # Top 5

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get performance statistics for all strategies."""
        return {
            name: {
                "success_rate": round(s.success_rate, 3),
                "avg_confidence": round(s.avg_confidence, 3),
                "avg_duration_ms": round(s.avg_duration_ms, 1),
                "uses": s.uses,
            }
            for name, s in self.strategies.items()
        }


__all__ = [
    "Strategy",
    "StrategySelector",
    "StrategyStateSpace",
    "classify_domain",
    "DEFAULT_STRATEGIES",
]
