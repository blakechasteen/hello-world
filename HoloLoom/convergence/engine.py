"""
Convergence Engine - Continuous to Discrete Collapse
=====================================================
The decision-making engine that collapses continuous representations
into discrete tool selections.

Philosophy:
After features interfere in the Resonance Shed and threads tension in
Warp Space, the Convergence Engine performs the critical "collapse" -
transforming continuous probability distributions into discrete decisions.

Like quantum mechanics where wave functions collapse to definite states,
the Convergence Engine takes the fluid DotPlasma and probabilistic neural
outputs and "snaps" them into concrete tool choices.

This is where exploration meets exploitation, where Thompson Sampling
samples from the space of possibilities, and where the final decision
emerges.

Components:
- Neural Core: Deep learning decision network
- Thompson Sampling Bandit: Bayesian exploration
- Collapse Strategies: How to convert probabilities → decisions
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Collapse Strategies
# ============================================================================

class CollapseStrategy(Enum):
    """
    Strategies for collapsing continuous to discrete.

    Different ways to make the final decision from probabilities.
    """
    ARGMAX = "argmax"  # Pick highest probability (pure exploitation)
    EPSILON_GREEDY = "epsilon_greedy"  # Explore with probability epsilon
    BAYESIAN_BLEND = "bayesian_blend"  # Mix neural + bandit priors
    PURE_THOMPSON = "pure_thompson"  # Pure Thompson Sampling


# ============================================================================
# Thompson Sampling Bandit
# ============================================================================

class ThompsonBandit:
    """
    Thompson Sampling bandit for tool selection.

    Maintains Beta distributions for each tool (arm), updating based on
    observed rewards. Naturally balances exploration and exploitation.
    """

    def __init__(self, n_tools: int):
        """
        Initialize bandit.

        Args:
            n_tools: Number of tools (arms)
        """
        # Beta distribution parameters
        self.successes = np.ones(n_tools)  # α (alpha)
        self.failures = np.ones(n_tools)   # β (beta)
        self.n_tools = n_tools

        # Tracking
        self.pulls = np.zeros(n_tools)
        self.total_reward = np.zeros(n_tools)

        logger.info(f"ThompsonBandit initialized with {n_tools} tools")

    def sample(self) -> int:
        """
        Sample tool index using Thompson Sampling.

        Draws from Beta distributions and picks the best sample.

        Returns:
            Tool index
        """
        # Sample from each arm's Beta distribution
        samples = np.random.beta(self.successes, self.failures)

        # Pick arm with highest sample
        tool_idx = int(np.argmax(samples))

        self.pulls[tool_idx] += 1
        return tool_idx

    def get_priors(self) -> np.ndarray:
        """
        Get prior probabilities (expected values).

        Mean of Beta(α, β) = α / (α + β)

        Returns:
            Array of prior probabilities [n_tools]
        """
        return self.successes / (self.successes + self.failures)

    def update(self, tool_idx: int, reward: float) -> None:
        """
        Update bandit based on observed reward.

        Args:
            tool_idx: Tool that was used
            reward: Observed reward (positive for success)
        """
        if reward > 0:
            self.successes[tool_idx] += reward
        else:
            self.failures[tool_idx] += abs(reward)

        self.total_reward[tool_idx] += reward

        logger.debug(f"Updated bandit: tool={tool_idx}, reward={reward:.3f}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get bandit statistics.

        Returns:
            Dict with per-tool statistics
        """
        return {
            "tool_priors": self.get_priors().tolist(),
            "pulls": self.pulls.tolist(),
            "total_rewards": self.total_reward.tolist(),
            "success_counts": self.successes.tolist(),
            "failure_counts": self.failures.tolist()
        }


# ============================================================================
# Convergence Engine
# ============================================================================

@dataclass
class CollapseResult:
    """Result of collapse operation."""
    tool: str  # Selected tool
    tool_idx: int  # Tool index
    confidence: float  # Decision confidence
    neural_probs: List[float]  # Neural network probabilities
    strategy_used: str  # Which collapse strategy was used
    bandit_stats: Optional[Dict] = None  # Optional bandit statistics


class ConvergenceEngine:
    """
    Convergence Engine - Collapses continuous → discrete.

    The engine takes continuous probability distributions from neural networks
    and/or bandit priors, and makes discrete tool selections.

    It supports multiple collapse strategies:
    1. **ARGMAX**: Pure exploitation (pick highest probability)
    2. **EPSILON_GREEDY**: Explore ε% of time, exploit (1-ε)%
    3. **BAYESIAN_BLEND**: Mix neural predictions with bandit priors
    4. **PURE_THOMPSON**: Pure Bayesian exploration (ignore neural net)

    Usage:
        engine = ConvergenceEngine(tools=["answer", "search", "calc"])
        result = engine.collapse(
            neural_probs=[0.6, 0.3, 0.1],
            strategy=CollapseStrategy.EPSILON_GREEDY,
            epsilon=0.1
        )
    """

    def __init__(
        self,
        tools: List[str],
        default_strategy: CollapseStrategy = CollapseStrategy.EPSILON_GREEDY,
        epsilon: float = 0.1,
        entropy_temperature: float = 0.1
    ):
        """
        Initialize Convergence Engine.

        Args:
            tools: List of tool names
            default_strategy: Default collapse strategy
            epsilon: Exploration rate for epsilon-greedy
            entropy_temperature: Temperature for entropy injection (0-1)
                                0.0 = no noise (deterministic)
                                0.1 = light noise (default)
                                0.5 = moderate noise
                                1.0 = high noise (maximum exploration)
        """
        self.tools = tools
        self.n_tools = len(tools)
        self.default_strategy = default_strategy
        self.epsilon = epsilon
        self.entropy_temperature = entropy_temperature

        # Initialize Thompson bandit
        self.bandit = ThompsonBandit(self.n_tools)

        # Track collapses
        self.collapse_history: List[CollapseResult] = []

        logger.info(f"ConvergenceEngine initialized: {self.n_tools} tools, strategy={default_strategy.value}, entropy_temp={entropy_temperature}")

    def inject_entropy(self, probs: np.ndarray, temperature: Optional[float] = None) -> np.ndarray:
        """
        Inject entropy (controlled noise) into probabilities.

        Adds Gumbel noise for "fresh air" - prevents deterministic stagnation
        while maintaining roughly the same probability distribution.

        Args:
            probs: Probability distribution [n_tools]
            temperature: Override default temperature (higher = more noise)

        Returns:
            Noisy probabilities (still normalized)
        """
        temperature = temperature or self.entropy_temperature

        if temperature == 0:
            return probs  # No noise

        # Add Gumbel noise: -log(-log(uniform))
        # This is the "Gumbel-Max trick" for sampling
        gumbel_noise = -np.log(-np.log(np.random.uniform(0.0001, 0.9999, size=len(probs))))

        # Scale by temperature and add to log-probs
        log_probs = np.log(probs + 1e-10)
        noisy_log_probs = log_probs + temperature * gumbel_noise

        # Convert back to probabilities
        noisy_probs = np.exp(noisy_log_probs - np.max(noisy_log_probs))
        noisy_probs = noisy_probs / np.sum(noisy_probs)

        return noisy_probs

    def collapse(
        self,
        neural_probs: np.ndarray,
        strategy: Optional[CollapseStrategy] = None,
        epsilon: Optional[float] = None,
        inject_entropy: bool = True
    ) -> CollapseResult:
        """
        Perform collapse: continuous probabilities → discrete tool selection.

        Args:
            neural_probs: Probability distribution from neural network [n_tools]
            strategy: Override default collapse strategy
            epsilon: Override default epsilon (for epsilon-greedy)
            inject_entropy: Whether to inject entropy for exploration

        Returns:
            CollapseResult with selected tool and metadata
        """
        strategy = strategy or self.default_strategy
        epsilon = epsilon or self.epsilon

        logger.debug(f"Collapsing with strategy={strategy.value}, neural_probs={neural_probs}")

        # Normalize probabilities
        neural_probs = neural_probs / np.sum(neural_probs)

        # Inject entropy if enabled (adds "fresh air")
        if inject_entropy and self.entropy_temperature > 0:
            neural_probs = self.inject_entropy(neural_probs)
            logger.debug(f"Entropy injected (temperature={self.entropy_temperature})")

        # Select tool based on strategy
        if strategy == CollapseStrategy.ARGMAX:
            tool_idx = int(np.argmax(neural_probs))
            confidence = float(neural_probs[tool_idx])
            strategy_info = "argmax_exploitation"

        elif strategy == CollapseStrategy.EPSILON_GREEDY:
            if np.random.rand() < epsilon:
                # EXPLORE: Thompson Sampling
                tool_idx = self.bandit.sample()
                confidence = float(neural_probs[tool_idx])
                strategy_info = f"epsilon_explore_({epsilon})"
            else:
                # EXPLOIT: Neural network
                tool_idx = int(np.argmax(neural_probs))
                confidence = float(neural_probs[tool_idx])
                strategy_info = f"epsilon_exploit_({1-epsilon})"

        elif strategy == CollapseStrategy.BAYESIAN_BLEND:
            # Blend neural predictions with bandit priors
            bandit_priors = self.bandit.get_priors()

            # Weighted combination (70% neural, 30% bandit)
            blended = 0.7 * neural_probs + 0.3 * bandit_priors
            tool_idx = int(np.argmax(blended))
            confidence = float(blended[tool_idx])
            strategy_info = "bayesian_blend_(0.7neural+0.3bandit)"

        elif strategy == CollapseStrategy.PURE_THOMPSON:
            # Pure Thompson: ignore neural network
            tool_idx = self.bandit.sample()
            bandit_priors = self.bandit.get_priors()
            confidence = float(bandit_priors[tool_idx])
            strategy_info = "pure_thompson"

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Get tool name
        tool = self.tools[tool_idx]

        # Create result
        result = CollapseResult(
            tool=tool,
            tool_idx=tool_idx,
            confidence=confidence,
            neural_probs=neural_probs.tolist(),
            strategy_used=strategy_info,
            bandit_stats=self.bandit.get_statistics()
        )

        # Record collapse
        self.collapse_history.append(result)

        logger.info(f"Collapsed → tool={tool} (idx={tool_idx}), confidence={confidence:.3f}, strategy={strategy_info}")

        return result

    def update_from_outcome(
        self,
        tool_idx: int,
        success: bool,
        reward: Optional[float] = None
    ) -> None:
        """
        Update bandit based on outcome.

        Args:
            tool_idx: Tool that was used
            success: Whether the tool succeeded
            reward: Optional explicit reward (if None, uses success/failure)
        """
        if reward is None:
            reward = 1.0 if success else -1.0

        self.bandit.update(tool_idx, reward)

        logger.info(f"Updated from outcome: tool_idx={tool_idx}, success={success}, reward={reward:.3f}")

    def get_trace(self) -> Dict[str, Any]:
        """
        Get convergence trace.

        Returns:
            Dict with convergence history and statistics
        """
        return {
            "total_collapses": len(self.collapse_history),
            "tools": self.tools,
            "default_strategy": self.default_strategy.value,
            "epsilon": self.epsilon,
            "bandit_stats": self.bandit.get_statistics(),
            "recent_collapses": [
                {
                    "tool": r.tool,
                    "confidence": r.confidence,
                    "strategy": r.strategy_used
                }
                for r in self.collapse_history[-10:]  # Last 10
            ]
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_convergence_engine(
    tools: List[str],
    strategy: CollapseStrategy = CollapseStrategy.EPSILON_GREEDY,
    epsilon: float = 0.1
) -> ConvergenceEngine:
    """
    Create Convergence Engine.

    Args:
        tools: List of tool names
        strategy: Collapse strategy
        epsilon: Exploration rate

    Returns:
        Configured ConvergenceEngine
    """
    return ConvergenceEngine(
        tools=tools,
        default_strategy=strategy,
        epsilon=epsilon
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Convergence Engine Demo")
    print("="*80 + "\n")

    # Create engine
    tools = ["answer", "search", "notion_write", "calc"]
    engine = ConvergenceEngine(tools, default_strategy=CollapseStrategy.EPSILON_GREEDY, epsilon=0.2)

    print("Testing collapse strategies:\n")

    # Simulate neural network probabilities
    neural_probs = np.array([0.5, 0.3, 0.15, 0.05])

    # Test different strategies
    strategies = [
        (CollapseStrategy.ARGMAX, "Pure Exploitation"),
        (CollapseStrategy.EPSILON_GREEDY, "Epsilon-Greedy (20% explore)"),
        (CollapseStrategy.BAYESIAN_BLEND, "Bayesian Blend"),
        (CollapseStrategy.PURE_THOMPSON, "Pure Thompson Sampling")
    ]

    for strategy, description in strategies:
        print(f"\n{description}")
        print("-" * 40)

        # Run 10 collapses
        tool_counts = {tool: 0 for tool in tools}

        for i in range(10):
            result = engine.collapse(neural_probs, strategy=strategy)
            tool_counts[result.tool] += 1

            # Simulate outcome (higher prob = higher success rate)
            success = np.random.rand() < result.confidence
            engine.update_from_outcome(result.tool_idx, success)

        # Display distribution
        print("Tool selection distribution:")
        for tool, count in tool_counts.items():
            bar = '█' * count
            print(f"  {tool:15s} {bar} ({count}/10)")

    # Show final bandit statistics
    print("\n" + "="*80)
    print("Final Bandit Statistics:")
    print("="*80)
    stats = engine.bandit.get_statistics()
    for i, tool in enumerate(tools):
        prior = stats['tool_priors'][i]
        pulls = stats['pulls'][i]
        print(f"  {tool:15s} Prior={prior:.3f}, Pulls={int(pulls)}, "
              f"Success={stats['success_counts'][i]:.1f}, "
              f"Fail={stats['failure_counts'][i]:.1f}")

    print("\n✓ Demo complete!")
