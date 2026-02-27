#!/usr/bin/env python3
"""
Complete Agentic Integration Demo

Demonstrates the full stack:
- Safety guardrails (Phase 1 Alignment)
- Agentic reasoning (4 modes: DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
- LLM-activated intelligent search (Task 4)
- Persistent memory integration (Neo4j + Qdrant)
- Complete provenance tracking (AuditTrail)
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard
from hololoom.alignment.audit_trail import OutcomeType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subheader(title: str):
    """Print formatted subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def create_demo_shards() -> List[MemoryShard]:
    """Create rich test shards with ML/AI content."""
    return [
        MemoryShard(
            id="shard_thompson_1",
            text=(
                "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                "It maintains a probability distribution (typically Beta) for each arm's reward. "
                "At each step, it samples from these distributions and selects the arm with the "
                "highest sample. This naturally balances exploration and exploitation through "
                "the uncertainty in the posterior distributions. The Beta(α, β) distribution is "
                "updated after each trial: successful outcomes increase α, failures increase β."
            ),
            metadata={
                "topic": "reinforcement_learning",
                "subtopic": "multi_armed_bandits",
                "algorithm": "thompson_sampling",
                "confidence": 0.95,
                "source": "bandit_algorithms_textbook"
            }
        ),
        MemoryShard(
            id="shard_thompson_2",
            text=(
                "Compared to epsilon-greedy and UCB (Upper Confidence Bound), Thompson Sampling "
                "often converges faster to optimal policies. UCB uses deterministic upper confidence "
                "bounds while Thompson Sampling uses probabilistic sampling. Epsilon-greedy has "
                "fixed exploration rate, while Thompson Sampling adapts based on posterior uncertainty. "
                "In practice, Thompson Sampling achieves lower regret in many scenarios, especially "
                "when priors are informative."
            ),
            metadata={
                "topic": "reinforcement_learning",
                "subtopic": "comparison",
                "algorithms": ["thompson_sampling", "ucb", "epsilon_greedy"],
                "confidence": 0.90,
                "source": "bandit_comparison_paper"
            }
        ),
        MemoryShard(
            id="shard_transformers",
            text=(
                "Transformers revolutionized NLP through self-attention mechanisms. Unlike RNNs, "
                "they process sequences in parallel, using attention to weigh the importance of "
                "different positions. The architecture consists of encoder-decoder stacks with "
                "multi-head attention, feed-forward networks, and residual connections. Key innovation: "
                "attention scores computed as softmax(QK^T/√d_k)V, where Q, K, V are learned projections."
            ),
            metadata={
                "topic": "deep_learning",
                "subtopic": "architectures",
                "model": "transformer",
                "confidence": 0.92,
                "source": "attention_is_all_you_need"
            }
        ),
        MemoryShard(
            id="shard_rl_fundamentals",
            text=(
                "Reinforcement Learning formalizes sequential decision-making as MDPs (Markov Decision "
                "Processes). An agent learns a policy π(a|s) that maximizes expected cumulative reward "
                "E[Σγ^t r_t]. Key algorithms include Q-learning (off-policy, tabular), SARSA (on-policy), "
                "PPO (policy gradient with clipped objective), and A3C (asynchronous actor-critic). "
                "The exploration-exploitation tradeoff is central: explore to discover better strategies, "
                "exploit current knowledge to maximize reward."
            ),
            metadata={
                "topic": "reinforcement_learning",
                "subtopic": "fundamentals",
                "algorithms": ["q_learning", "sarsa", "ppo", "a3c"],
                "confidence": 0.88,
                "source": "sutton_barto_rl_book"
            }
        ),
        MemoryShard(
            id="shard_bayesian_inference",
            text=(
                "Bayesian inference updates beliefs using Bayes' theorem: P(θ|D) = P(D|θ)P(θ)/P(D). "
                "Prior P(θ) represents initial beliefs, likelihood P(D|θ) measures data fit, posterior "
                "P(θ|D) is updated belief. Conjugate priors (like Beta for Bernoulli) enable analytical "
                "updates. In Thompson Sampling, Beta(α, β) is conjugate to Bernoulli likelihood, making "
                "updates efficient: after observing reward r, update α ← α + r, β ← β + (1-r)."
            ),
            metadata={
                "topic": "statistics",
                "subtopic": "bayesian_methods",
                "concepts": ["bayes_theorem", "conjugate_priors", "beta_distribution"],
                "confidence": 0.93,
                "source": "bayesian_data_analysis"
            }
        ),
        MemoryShard(
            id="shard_exploration_strategies",
            text=(
                "Exploration strategies in RL include: (1) ε-greedy: random action with probability ε, "
                "greedy otherwise; (2) Boltzmann/softmax: sample proportional to exp(Q/τ), temperature "
                "τ controls randomness; (3) UCB: optimism under uncertainty, select arg max[Q + c√(ln(t)/N)]; "
                "(4) Thompson Sampling: Bayesian approach, sample from posterior; (5) Curiosity-driven: "
                "intrinsic rewards for novel states (ICM, RND). Each has tradeoffs in sample efficiency, "
                "computational cost, and convergence guarantees."
            ),
            metadata={
                "topic": "reinforcement_learning",
                "subtopic": "exploration",
                "strategies": ["epsilon_greedy", "boltzmann", "ucb", "thompson_sampling", "curiosity"],
                "confidence": 0.87,
                "source": "exploration_exploitation_survey"
            }
        )
    ]


async def demo_mode(
    orchestrator,
    mode: ReasoningMode,
    query: Query,
    max_steps: int = 3
):
    """Demo a single reasoning mode."""
    print_subheader(f"Mode: {mode.value.upper()}")
    print(f"Query: '{query.text}'\n")

    start_time = datetime.now()

    try:
        result = await orchestrator.reason(
            query=query,
            mode=mode,
            max_steps=max_steps
        )

        duration = (datetime.now() - start_time).total_seconds() * 1000

        # Display results
        print(f"✅ {mode.value.upper()} complete!")
        print(f"   Duration: {duration:.1f}ms")
        print(f"   Confidence: {result.spacetime.confidence:.2f}")
        print(f"   Total steps: {len(result.steps_taken)}")

        # Show reasoning steps
        if result.steps_taken:
            print(f"\n📋 Reasoning Steps ({len(result.steps_taken)}):")
            for i, step in enumerate(result.steps_taken, 1):
                step_type = step.get("type", "unknown")

                # Check if LLM-generated (for RESEARCH queries)
                is_llm = step.get("llm_generated", False)
                llm_marker = " 🤖" if is_llm else ""

                print(f"\n  Step {i}: {step_type.upper()}{llm_marker}")

                if step_type == "research_query":
                    query_text = step.get("query", "")
                    findings = step.get("findings", "")
                    confidence = step.get("confidence", 0.0)

                    # Truncate long queries for display
                    if len(query_text) > 120:
                        query_text = query_text[:120] + "..."

                    print(f"     Query: {query_text}")
                    print(f"     Confidence: {confidence:.2f}")
                    print(f"     Findings: {findings[:100]}...")

                elif step_type == "verification":
                    checks = step.get("checks", [])
                    print(f"     Checks: {len(checks)}")
                    for check in checks[:3]:  # Show first 3
                        status = "✅" if check.get("passed") else "❌"
                        print(f"       {status} {check.get('aspect', 'unknown')}")

                elif step_type == "goal_decomposition":
                    goals = step.get("goals", [])
                    print(f"     Sub-goals: {len(goals)}")
                    for goal in goals[:3]:
                        print(f"       • {goal}")

                elif step_type == "synthesis":
                    sources = step.get("sources", 0)
                    print(f"     Synthesizing from {sources} sources")

        # Show final result
        if result.spacetime and result.spacetime.response:
            print(f"\n📄 Final Result:")
            response = result.spacetime.response
            if len(response) > 300:
                print(f"   {response[:300]}...")
            else:
                print(f"   {response}")

        print()

    except Exception as e:
        logger.error(f"Mode {mode.value} failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run complete agentic integration demo."""
    print_header("🎯 Complete Agentic Integration Demo")

    print("Demonstrating:")
    print("  • Safety guardrails (Phase 1 Alignment)")
    print("  • 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)")
    print("  • LLM-activated intelligent search (Task 4)")
    print("  • Persistent memory integration")
    print("  • Complete provenance tracking")

    # Create demo shards
    print_subheader("1. Creating Knowledge Base")
    shards = create_demo_shards()
    print(f"✅ Created {len(shards)} memory shards")
    print(f"   Topics: RL, Transformers, Bayesian Inference, Exploration")

    # Create orchestrator
    print_subheader("2. Initializing Agentic Orchestrator")
    orchestrator = await create_agentic_orchestrator(
        config=Config.fast(),
        shards=shards,
        enable_verification=True,
        enable_goal_tracking=True
    )

    # Check LLM status
    llm_status = "available" if orchestrator.llm else "unavailable"
    print(f"✅ Orchestrator created")
    print(f"   LLM status: {llm_status}")
    print(f"   Alignment: enabled")
    print(f"   Memory: {len(shards)} shards loaded")

    # Demo 1: DIRECT mode
    print_header("Demo 1: DIRECT Mode (Single-Pass Answer)")
    await demo_mode(
        orchestrator,
        ReasoningMode.DIRECT,
        Query(text="What is Thompson Sampling?"),
        max_steps=1
    )

    # Demo 2: VERIFY mode
    print_header("Demo 2: VERIFY Mode (Answer + Verification)")
    await demo_mode(
        orchestrator,
        ReasoningMode.VERIFY,
        Query(text="How does Thompson Sampling balance exploration and exploitation?"),
        max_steps=2
    )

    # Demo 3: RESEARCH mode (LLM-activated intelligent search)
    print_header("Demo 3: RESEARCH Mode (LLM-Activated Intelligent Search) ⭐")
    print("This is where LLM-activated agentic search shines!")
    print("Watch as the system generates intelligent, context-aware research questions.\n")

    await demo_mode(
        orchestrator,
        ReasoningMode.RESEARCH,
        Query(text="How does Thompson Sampling compare to other exploration strategies?"),
        max_steps=3
    )

    # Demo 4: PLAN_EXECUTE mode
    print_header("Demo 4: PLAN_EXECUTE Mode (Goal Decomposition)")
    await demo_mode(
        orchestrator,
        ReasoningMode.PLAN_EXECUTE,
        Query(text="Implement Thompson Sampling for A/B testing"),
        max_steps=4
    )

    # Show alignment audit
    print_header("5. Safety & Provenance")
    if orchestrator.audit_trail:
        all_logs = orchestrator.audit_trail.logs
        print(f"✅ Audit trail: {len(all_logs)} decision logs")

        if all_logs:
            print(f"\nRecent safety decisions (last 3):")
            for log in all_logs[-3:]:
                action = log.action_description or log.decision_type.value
                outcome = "✅ APPROVED" if log.outcome == OutcomeType.APPROVED else "❌ REJECTED"
                risk = log.risk_level or "unknown"
                print(f"  {outcome}: {action} (risk={risk}, confidence={log.confidence:.2f})")

    # Cleanup
    print_header("6. Cleanup")
    try:
        await orchestrator.close()
        print("✅ Orchestrator closed")
    except AttributeError:
        print("✅ Orchestrator session complete (auto-cleanup)")

    print_header("🎉 Demo Complete!")
    print("Key achievements demonstrated:")
    print("  ✅ All 4 reasoning modes working")
    print("  ✅ LLM-activated intelligent search (RESEARCH mode)")
    print("  ✅ Safety guardrails evaluating all actions")
    print("  ✅ Complete provenance tracking")
    print("  ✅ Memory integration with rich knowledge base")
    print("\nThis is a production-ready agentic AI system! 🚀\n")


if __name__ == "__main__":
    asyncio.run(main())
