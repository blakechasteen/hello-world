"""
Routing Commands for ChatOps
=============================

New commands for intelligent backend routing and learning.

Commands:
- !routing stats - Show learned routing preferences
- !routing experiment start <a> <b> - A/B test strategies
- !routing winner - Show best performing strategy
- !routing explain <query> - Explain routing decision
- !routing learn - Trigger optimization
"""

from typing import Dict, Any, Optional
from datetime import datetime

from HoloLoom.memory.routing import (
    RuleBasedRouter,
    LearnedRouter,
    RoutingExperiment,
    BackendType,
    QueryType
)
from HoloLoom.memory.routing.orchestrator import (
    RoutingOrchestrator,
    OrchestratorExperiment
)


class RoutingCommandHandler:
    """Handler for routing-related commands."""

    def __init__(self, orchestrator: RoutingOrchestrator):
        self.orchestrator = orchestrator
        self.active_experiment: Optional[OrchestratorExperiment] = None

    async def handle_routing_stats(self) -> str:
        """
        !routing stats

        Shows learned backend preferences by query type.
        """
        stats = self.orchestrator.get_statistics()
        routing_stats = stats['routing_strategy']

        response = "**Routing Statistics**\n\n"

        # Overall metrics
        if 'overall_accuracy' in routing_stats:
            accuracy = routing_stats['overall_accuracy']
            response += f"Overall Accuracy: {accuracy:.1%}\n"
            response += f"Total Executions: {stats['total_executions']}\n\n"

        # Backend distribution
        if 'backend_counts' in routing_stats:
            response += "**Backend Usage:**\n"
            for backend, count in routing_stats['backend_counts'].items():
                response += f"  • {backend}: {count} queries\n"
            response += "\n"

        # Query type classification
        if 'query_type_counts' in routing_stats:
            response += "**Query Types:**\n"
            for qtype, count in routing_stats['query_type_counts'].items():
                response += f"  • {qtype}: {count} queries\n"
            response += "\n"

        # Learned bandit stats (if using LearnedRouter)
        if 'bandits' in routing_stats:
            response += "**Learned Preferences (by Query Type):**\n\n"

            for qtype, bandit_stats in routing_stats['bandits'].items():
                response += f"**{qtype.upper()}:**\n"

                # Sort backends by mean reward
                sorted_backends = sorted(
                    bandit_stats.items(),
                    key=lambda x: x[1]['mean_reward'],
                    reverse=True
                )

                for backend, bstats in sorted_backends[:3]:  # Top 3
                    reward = bstats['mean_reward']
                    trials = bstats['total_trials']
                    response += f"  • {backend}: {reward:.2f} reward ({trials} trials)\n"

                response += "\n"

        # Performance by backend
        if 'avg_relevance_by_backend' in routing_stats:
            response += "**Average Relevance by Backend:**\n"
            for backend, relevance in routing_stats['avg_relevance_by_backend'].items():
                response += f"  • {backend}: {relevance:.2f}\n"

        return response

    async def handle_routing_experiment_start(
        self,
        variant_a: str,
        variant_b: str
    ) -> str:
        """
        !routing experiment start <variant-a> <variant-b>

        Start A/B test between two routing strategies.

        Examples:
          !routing experiment start rule_based learned
          !routing experiment start baseline optimized
        """
        # Create experiment
        self.active_experiment = OrchestratorExperiment()

        # Map variant names to strategies
        strategy_map = {
            'rule_based': RuleBasedRouter(),
            'learned': LearnedRouter(),
            'baseline': RuleBasedRouter(),
        }

        if variant_a not in strategy_map:
            return f"Unknown variant: {variant_a}. Options: {', '.join(strategy_map.keys())}"

        if variant_b not in strategy_map:
            return f"Unknown variant: {variant_b}. Options: {', '.join(strategy_map.keys())}"

        # Add variants
        from HoloLoom.memory.routing.execution_patterns import (
            FeedForwardEngine,
            RecursiveEngine,
            ExecutionPattern
        )

        self.active_experiment.add_variant(
            variant_a,
            routing=strategy_map[variant_a],
            execution={ExecutionPattern.FEED_FORWARD: FeedForwardEngine()},
            weight=0.5
        )

        self.active_experiment.add_variant(
            variant_b,
            routing=strategy_map[variant_b],
            execution={ExecutionPattern.RECURSIVE: RecursiveEngine()},
            weight=0.5
        )

        return f"""**Routing Experiment Started**

Variant A: {variant_a} (50% traffic)
Variant B: {variant_b} (50% traffic)

Queries will be randomly routed to test variants.
React with 👍/👎 to provide feedback!

Use `!routing winner` to see results (min 20 queries)
"""

    async def handle_routing_winner(self) -> str:
        """
        !routing winner

        Show results from active experiment.
        """
        if not self.active_experiment:
            return "No active experiment. Start one with:\n`!routing experiment start <a> <b>`"

        report = self.active_experiment.generate_report()

        winner = report['winner']
        variants = report['variants']

        if winner == "insufficient_data":
            return """**Experiment In Progress**

Need at least 10 queries per variant.
Keep using !weave and reacting with 👍/👎
"""

        response = f"**Experiment Results**\n\n"
        response += f"**Winner: {winner}** 🏆\n\n"

        response += "**Performance by Variant:**\n\n"
        for variant_name, metrics in variants.items():
            is_winner = variant_name == winner
            marker = "🏆 " if is_winner else "   "

            response += f"{marker}**{variant_name}:**\n"
            response += f"    Success Rate: {metrics['success_rate']:.1%}\n"
            response += f"    Avg Relevance: {metrics['avg_relevance']:.2f}\n"
            response += f"    Avg Latency: {metrics['avg_latency']:.0f}ms\n"
            response += f"    Total Queries: {metrics['total']}\n"

            if 'lift_over_baseline' in metrics:
                lift = metrics['lift_over_baseline']
                response += f"    Lift: {lift:+.1f}%\n"

            response += "\n"

        response += f"**Recommendation:**\n{report.get('recommendation', 'Continue testing')}"

        return response

    async def handle_routing_explain(self, query_text: str) -> str:
        """
        !routing explain <query>

        Explain why router would choose a particular backend.
        """
        from HoloLoom.memory.protocol import MemoryQuery

        # Get routing decision (without executing)
        available = [
            BackendType.NEO4J,
            BackendType.QDRANT,
            BackendType.MEM0,
            BackendType.INMEMORY
        ]

        decision = self.orchestrator.routing_strategy.select_backend(
            query_text,
            available,
            context={}
        )

        response = f"**Routing Analysis for:**\n`{query_text}`\n\n"

        response += f"**Selected Backend:** {decision.backend_type.value}\n"
        response += f"**Confidence:** {decision.confidence:.2%}\n"
        response += f"**Query Type:** {decision.query_type.value}\n\n"

        response += f"**Reasoning:**\n{decision.reasoning}\n\n"

        response += "**Alternatives Considered:**\n"
        for alt in decision.alternatives:
            response += f"  • {alt.value}\n"

        # If learned router, show stats
        if hasattr(self.orchestrator.routing_strategy, 'bandits'):
            response += f"\n**Learned Statistics:**\n"

            router = self.orchestrator.routing_strategy
            bandit_stats = router.get_statistics()

            if 'bandits' in bandit_stats:
                qtype_stats = bandit_stats['bandits'].get(decision.query_type.value, {})

                if qtype_stats:
                    backend_stats = qtype_stats.get(decision.backend_type.value, {})

                    response += f"  Mean Reward: {backend_stats.get('mean_reward', 0):.2f}\n"
                    response += f"  Total Trials: {backend_stats.get('total_trials', 0)}\n"

        # Execution pattern suggestion
        from HoloLoom.memory.routing.execution_patterns import select_execution_pattern

        suggested_pattern = select_execution_pattern(
            MemoryQuery(text=query_text, limit=5),
            decision.confidence,
            available
        )

        response += f"\n**Suggested Execution Pattern:** {suggested_pattern.value}\n"

        return response

    async def handle_routing_learn(self, force: bool = False) -> str:
        """
        !routing learn [--force]

        Trigger routing optimization from recent outcomes.
        """
        stats_before = self.orchestrator.get_statistics()

        # Save learned parameters (if using LearnedRouter)
        if hasattr(self.orchestrator.routing_strategy, 'save'):
            save_path = "memory_data/routing_learned.json"
            self.orchestrator.routing_strategy.save(save_path)

            response = f"""**Routing Optimization Complete**

Parameters saved to: {save_path}

**Current Performance:**
  • Total Executions: {stats_before['total_executions']}
"""

            routing_stats = stats_before['routing_strategy']
            if 'overall_accuracy' in routing_stats:
                response += f"  • Overall Accuracy: {routing_stats['overall_accuracy']:.1%}\n"

            return response
        else:
            return """**Routing Learn**

Current router: Rule-based (doesn't learn)

To enable learning, use:
`!routing experiment start rule_based learned`

This will test a learned router that adapts from feedback.
"""


# ============================================================================
# Reaction Feedback Loop
# ============================================================================

class ReactionFeedbackHandler:
    """
    Handles Matrix reactions and converts them to routing outcomes.

    Flow:
    1. User queries with !weave
    2. Bot responds with Spacetime
    3. User reacts: 👍 (helpful), 👎 (not helpful), ⭐ (excellent)
    4. Reaction converted to RoutingOutcome
    5. Fed into routing orchestrator for learning
    """

    def __init__(self, orchestrator: RoutingOrchestrator):
        self.orchestrator = orchestrator
        self.spacetime_cache: Dict[str, Any] = {}  # event_id -> spacetime

    def cache_spacetime(self, event_id: str, spacetime: Any):
        """Cache spacetime for later reaction processing."""
        self.spacetime_cache[event_id] = spacetime

    async def handle_reaction(
        self,
        event_id: str,
        reaction: str,
        user_id: str
    ):
        """
        Process reaction on a bot response.

        Reactions:
        - 👍 or ✅ = Helpful (0.9 relevance)
        - 👎 or ❌ = Not helpful (0.3 relevance)
        - ⭐ or 🌟 = Excellent (1.0 relevance)
        - 🔥 = Excellent (1.0 relevance)
        """
        # Get cached spacetime
        spacetime = self.spacetime_cache.get(event_id)
        if not spacetime:
            # Maybe old message, skip
            return

        # Map reaction to relevance score
        reaction_scores = {
            '👍': 0.9,
            '✅': 0.9,
            '👎': 0.3,
            '❌': 0.3,
            '⭐': 1.0,
            '🌟': 1.0,
            '🔥': 1.0,
        }

        relevance = reaction_scores.get(reaction, 0.7)  # Default neutral

        # Create routing outcome
        from HoloLoom.memory.routing.protocol import RoutingOutcome

        # Extract routing decision from spacetime metadata
        routing_meta = spacetime.metadata.get('routing_decision', {})

        outcome = RoutingOutcome(
            decision=routing_meta,  # Contains backend choice, confidence, etc.
            query=spacetime.query.text,
            result_count=len(spacetime.context_shards) if hasattr(spacetime, 'context_shards') else 5,
            avg_relevance=relevance,
            latency_ms=spacetime.timings.get('total_ms', 0) if hasattr(spacetime, 'timings') else 1000,
            user_feedback=relevance,
            timestamp=datetime.now().isoformat()
        )

        # Feed into orchestrator for learning!
        self.orchestrator.record_outcome(outcome)

        # Log for observability
        print(f"[Reaction Feedback] User {user_id} reacted {reaction} "
              f"(relevance: {relevance}) on query: {spacetime.query.text[:50]}")


# ============================================================================
# Integration Helper
# ============================================================================

def setup_routing_handlers(orchestrator: RoutingOrchestrator):
    """
    Create routing command and reaction handlers.

    Returns:
        tuple: (routing_commands, reaction_handler)

    Usage in chatops:
        routing_cmds, reaction_handler = setup_routing_handlers(orchestrator)

        # Handle commands
        if message.startswith("!routing"):
            response = await routing_cmds.handle(message)

        # Handle reactions
        if event.type == "m.reaction":
            await reaction_handler.handle_reaction(event_id, reaction, user_id)
    """
    routing_commands = RoutingCommandHandler(orchestrator)
    reaction_handler = ReactionFeedbackHandler(orchestrator)

    return routing_commands, reaction_handler