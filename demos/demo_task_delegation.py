"""
Demo: Task Delegation System (Phase 7.2)
=========================================

Demonstrates the complete task delegation pipeline:
1. Auto capability detection (no explicit capability)
2. Multi-capability query → ensemble trigger
3. Thompson Sampling learning over time
4. Ensemble voting vs weighted comparison
5. Performance statistics

Author: Claude Code
Date: 2025-12-09
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.agentic import (
    # Multi-Agent Protocol (Phase 7.1)
    AgentCapability,
    AgentStatus,
    AgentRegistry,
    MultiAgentCoordinator,
    create_coordinator,
    create_research_agent,
    create_synthesis_agent,
    create_code_agent,
    # Task Delegation (Phase 7.2)
    QueryCapabilityAnalyzer,
    CapabilityAnalysis,
    QueryComplexity,
    analyze_query_capabilities,
    ExpertRouter,
    SelectionStrategy,
    create_expert_router,
    EnsembleAggregator,
    EnsembleStrategy,
    EnsembleResult,
    AgentResponse,
    aggregate_responses,
    create_agent_response,
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n--- {text} ---\n")


# =============================================================================
# Demo 1: Auto Capability Detection
# =============================================================================

def demo_capability_detection():
    """Demonstrate automatic capability detection from queries."""
    print_header("Demo 1: Auto Capability Detection")

    analyzer = QueryCapabilityAnalyzer()

    # Test queries with expected capabilities
    test_queries = [
        ("Research machine learning papers on attention", AgentCapability.RESEARCH),
        ("Review this Python code for bugs", AgentCapability.CODE_REVIEW),
        ("Write a function to calculate fibonacci", AgentCapability.CODE_GENERATION),
        ("Summarize the key findings from the report", AgentCapability.SUMMARIZATION),
        ("Analyze the tradeoffs between approaches", AgentCapability.ANALYSIS),
        ("Debug this error: IndexError", AgentCapability.DEBUGGING),
        ("Explain how transformers work", AgentCapability.REASONING),
        ("Compare supervised vs unsupervised learning", AgentCapability.ANALYSIS),
    ]

    print("Query → Capability Mapping:\n")

    correct = 0
    for query, expected in test_queries:
        analysis = analyzer.analyze(query)
        match = "✓" if analysis.primary_capability == expected else "✗"
        if analysis.primary_capability == expected:
            correct += 1

        print(f"  {match} \"{query[:50]}...\"")
        print(f"     Primary: {analysis.primary_capability.value}")
        print(f"     Secondary: {[c.value for c in analysis.secondary_capabilities[:2]]}")
        print(f"     Confidence: {analysis.confidence:.2f}")
        print(f"     Complexity: {analysis.complexity.value}")
        print()

    accuracy = correct / len(test_queries) * 100
    print(f"Accuracy: {correct}/{len(test_queries)} ({accuracy:.0f}%)")

    # Show complexity distribution
    print_section("Complexity Detection")

    complexity_queries = [
        ("hi", QueryComplexity.TRIVIAL),
        ("what is Python?", QueryComplexity.SIMPLE),
        ("explain decorators in detail", QueryComplexity.MODERATE),
        ("compare and analyze all ML algorithms", QueryComplexity.COMPLEX),
        ("comprehensive research on neural architecture search", QueryComplexity.RESEARCH),
    ]

    for query, expected in complexity_queries:
        analysis = analyzer.analyze(query)
        match = "✓" if analysis.complexity == expected else "✗"
        print(f"  {match} \"{query}\" → {analysis.complexity.value} (expected: {expected.value})")


# =============================================================================
# Demo 2: Expert Router with Thompson Sampling
# =============================================================================

def demo_expert_router():
    """Demonstrate Thompson Sampling-based agent selection."""
    print_header("Demo 2: Expert Router (Thompson Sampling)")

    # Create registry with agents
    registry = AgentRegistry()

    # Register multiple research agents with different performance
    agents = [
        ("research_alpha", AgentCapability.RESEARCH, 0.9),  # Good
        ("research_beta", AgentCapability.RESEARCH, 0.7),   # Moderate
        ("research_gamma", AgentCapability.RESEARCH, 0.5),  # Poor
        ("code_expert", AgentCapability.CODE_GENERATION, 0.95),
        ("analyzer_pro", AgentCapability.ANALYSIS, 0.85),
    ]

    for agent_id, capability, success_rate in agents:
        registry.register(
            agent_id=agent_id,
            capabilities=[capability],
            metadata={"success_rate": success_rate}
        )
        # Set agent status to IDLE
        registry.update_status(agent_id, AgentStatus.IDLE)

    # Create router with Thompson Sampling enabled
    router = ExpertRouter(registry, enable_thompson=True)

    print_section("Initial Agent Priors (Uniform)")
    for agent_id, _, _ in agents[:3]:  # Research agents
        prior = router.get_agent_prior(agent_id)
        print(f"  {agent_id}: α={prior['alpha']:.1f}, β={prior['beta']:.1f}, E[X]={prior['expected_value']:.2f}")

    # Simulate learning: update priors based on outcomes
    print_section("Simulating 20 Tasks with Feedback")

    outcomes = [
        ("research_alpha", True, 0.9),
        ("research_alpha", True, 0.85),
        ("research_alpha", True, 0.92),
        ("research_beta", True, 0.7),
        ("research_beta", False, 0.3),
        ("research_beta", True, 0.65),
        ("research_gamma", False, 0.4),
        ("research_gamma", False, 0.35),
        ("research_gamma", True, 0.55),
        ("research_alpha", True, 0.88),
        ("research_alpha", True, 0.91),
        ("research_beta", True, 0.72),
        ("research_beta", False, 0.4),
        ("research_gamma", False, 0.3),
        ("research_gamma", False, 0.25),
        ("research_alpha", True, 0.95),
        ("research_alpha", True, 0.87),
        ("research_beta", True, 0.68),
        ("research_gamma", False, 0.2),
        ("research_alpha", True, 0.93),
    ]

    for agent_id, success, confidence in outcomes:
        router.update_performance(agent_id, success, confidence)

    print("\nUpdated Priors After Learning:")
    for agent_id, _, _ in agents[:3]:  # Research agents
        prior = router.get_agent_prior(agent_id)
        perf = router.get_agent_performance(agent_id)
        print(f"  {agent_id}:")
        print(f"    Prior: α={prior['alpha']:.1f}, β={prior['beta']:.1f}, E[X]={prior['expected_value']:.2f}")
        print(f"    Performance: {perf['successes']} successes, {perf['failures']} failures")
        print(f"    Success rate: {perf['success_rate']:.1%}")

    # Show routing decisions with different strategies
    print_section("Routing Decisions by Strategy")

    strategies = [
        SelectionStrategy.BALANCED,
        SelectionStrategy.RELIABLE,
        SelectionStrategy.THOMPSON,
    ]

    for strategy in strategies:
        decision = router.route(
            required_capability=AgentCapability.RESEARCH,
            strategy=strategy,
            top_k=3
        )
        print(f"\n  Strategy: {strategy.value.upper()}")
        print(f"    Selected: {decision.selected_agent}")
        print(f"    Score: {decision.score.total_score:.3f}")
        print(f"    Thompson sample: {decision.score.thompson_sample:.3f}")
        print(f"    Expected reward: {decision.score.expected_reward:.3f}")
        print(f"    Alternatives: {[a.agent_id for a in decision.alternatives]}")

    # Statistics
    print_section("Routing Statistics")
    stats = router.get_routing_statistics()
    print(f"  Total routings: {stats['total_routings']}")
    print(f"  Thompson selections: {stats['thompson_selections']}")
    print(f"  Agents tracked: {stats['agents_tracked']}")


# =============================================================================
# Demo 3: Ensemble Decision Making
# =============================================================================

def demo_ensemble_decisions():
    """Demonstrate multi-agent result aggregation."""
    print_header("Demo 3: Ensemble Decision Making")

    # Create aggregator
    aggregator = EnsembleAggregator()

    # Scenario 1: Text responses (majority vote)
    print_section("Scenario 1: Text Responses (Majority Vote)")

    text_responses = [
        create_agent_response("agent_1", "Thompson Sampling", 0.9, 150.0),
        create_agent_response("agent_2", "Thompson Sampling", 0.85, 120.0),
        create_agent_response("agent_3", "UCB Algorithm", 0.7, 100.0),
    ]

    result = aggregator.aggregate(text_responses, EnsembleStrategy.MAJORITY_VOTE)

    print(f"  Individual responses:")
    for r in text_responses:
        print(f"    {r.agent_id}: \"{r.response}\" (conf: {r.confidence:.2f})")

    print(f"\n  Aggregated result:")
    print(f"    Final response: \"{result.final_response}\"")
    print(f"    Strategy: {result.strategy_used.value}")
    print(f"    Agreement: {result.agreement_score:.2f}")
    print(f"    Unanimous: {result.unanimous}")

    # Scenario 2: Numeric responses (confidence weighted)
    print_section("Scenario 2: Numeric Responses (Confidence Weighted)")

    numeric_responses = [
        create_agent_response("model_a", 0.85, 0.9, 200.0),
        create_agent_response("model_b", 0.78, 0.7, 180.0),
        create_agent_response("model_c", 0.92, 0.6, 220.0),
    ]

    result = aggregator.aggregate(numeric_responses, EnsembleStrategy.CONFIDENCE_WEIGHTED)

    print(f"  Individual predictions:")
    for r in numeric_responses:
        print(f"    {r.agent_id}: {r.response:.2f} (conf: {r.confidence:.2f})")

    print(f"\n  Weighted average:")
    print(f"    Final: {result.final_response:.3f}")
    print(f"    Agreement: {result.agreement_score:.2f}")
    print(f"    Avg confidence: {result.avg_confidence:.2f}")

    # Scenario 3: Best confidence selection
    print_section("Scenario 3: Best Confidence Selection")

    mixed_responses = [
        create_agent_response("expert_1", "Recommended approach A", 0.95, 300.0),
        create_agent_response("expert_2", "Recommended approach B", 0.75, 250.0),
        create_agent_response("expert_3", "Recommended approach C", 0.60, 200.0),
    ]

    result = aggregator.aggregate(mixed_responses, EnsembleStrategy.BEST_CONFIDENCE)

    print(f"  Individual recommendations:")
    for r in mixed_responses:
        print(f"    {r.agent_id}: \"{r.response}\" (conf: {r.confidence:.2f})")

    print(f"\n  Best confidence wins:")
    print(f"    Selected: \"{result.final_response}\"")
    print(f"    From highest confidence agent")

    # Scenario 4: Unanimous agreement (with fallback)
    print_section("Scenario 4: Unanimous (With Fallback)")

    disagreeing = [
        create_agent_response("validator_1", "Valid", 0.9, 100.0),
        create_agent_response("validator_2", "Invalid", 0.8, 100.0),
        create_agent_response("validator_3", "Valid", 0.85, 100.0),
    ]

    result = aggregator.aggregate(disagreeing, EnsembleStrategy.UNANIMOUS)

    print(f"  Validators disagree:")
    for r in disagreeing:
        print(f"    {r.agent_id}: \"{r.response}\" (conf: {r.confidence:.2f})")

    print(f"\n  Unanimous failed, fallback to weighted:")
    print(f"    Final: \"{result.final_response}\"")
    print(f"    Strategy used: {result.strategy_used.value}")

    # Compare all strategies on same data
    print_section("Strategy Comparison")

    comparison_responses = [
        create_agent_response("a1", "Option X", 0.9, 100.0),
        create_agent_response("a2", "Option X", 0.6, 100.0),
        create_agent_response("a3", "Option Y", 0.85, 100.0),
    ]

    print("  Responses: X(0.9), X(0.6), Y(0.85)")
    print()

    for strategy in EnsembleStrategy:
        result = aggregator.aggregate(comparison_responses, strategy)
        print(f"  {strategy.value:20s} → \"{result.final_response}\" (agreement: {result.agreement_score:.2f})")


# =============================================================================
# Demo 4: Integrated Task Delegation
# =============================================================================

async def demo_integrated_delegation():
    """Demonstrate the complete integrated delegation system."""
    print_header("Demo 4: Integrated Task Delegation")

    # Create coordinator
    coordinator = create_coordinator()

    # Register agents with various capabilities
    print_section("Setting Up Agent Network")

    agent_configs = [
        ("research_lead", [AgentCapability.RESEARCH, AgentCapability.ANALYSIS]),
        ("code_master", [AgentCapability.CODE_GENERATION, AgentCapability.CODE_REVIEW]),
        ("data_analyst", [AgentCapability.DATA_ANALYSIS, AgentCapability.ANALYSIS]),
        ("synthesizer", [AgentCapability.SUMMARIZATION, AgentCapability.CONTENT_CREATION]),
        ("debugger_bot", [AgentCapability.DEBUGGING, AgentCapability.CODE_REVIEW]),
    ]

    for agent_id, capabilities in agent_configs:
        coordinator.registry.register(
            agent_id=agent_id,
            capabilities=capabilities,
            metadata={"created": datetime.now().isoformat()}
        )
        coordinator.registry.update_status(agent_id, AgentStatus.IDLE)
        print(f"  ✓ Registered {agent_id} with {[c.value for c in capabilities]}")

    # Demo single-agent delegation with auto-detect
    print_section("Single Agent Delegation (Auto-Detect)")

    queries = [
        "Research the latest papers on transformer architectures",
        "Review this code for potential bugs",
        "Analyze the performance metrics",
    ]

    for query in queries:
        print(f"\n  Query: \"{query[:50]}...\"")

        # Analyze capability
        analysis = analyze_query_capabilities(query)
        print(f"  Detected: {analysis.primary_capability.value} (conf: {analysis.confidence:.2f})")

        # Find best agent
        router = create_expert_router(coordinator.registry)
        decision = router.route(analysis.primary_capability)

        if decision.selected_agent:
            print(f"  Routed to: {decision.selected_agent}")
            print(f"  Reason: {decision.routing_reason}")
        else:
            print("  No available agent found")

    # Demo ensemble delegation
    print_section("Ensemble Delegation")

    # Get multiple agents for ANALYSIS capability
    router = create_expert_router(coordinator.registry)

    print("  Query: \"Analyze the tradeoffs of different ML approaches\"")
    print("  Capability: ANALYSIS (multiple agents available)")

    decision = router.route(
        required_capability=AgentCapability.ANALYSIS,
        strategy=SelectionStrategy.BALANCED,
        top_k=3
    )

    print(f"\n  Selected for ensemble:")
    print(f"    Primary: {decision.selected_agent} (score: {decision.score.total_score:.3f})")
    for alt in decision.alternatives:
        print(f"    Alternative: {alt.agent_id} (score: {alt.total_score:.3f})")

    # Simulate ensemble responses
    print("\n  Simulating ensemble responses...")

    responses = [
        create_agent_response(
            decision.selected_agent,
            "Deep learning excels at pattern recognition but requires large datasets",
            0.88,
            250.0
        ),
    ]

    for alt in decision.alternatives:
        responses.append(create_agent_response(
            alt.agent_id,
            "Deep learning excels at pattern recognition but requires large datasets",
            0.75 + (0.1 * len(responses)),
            200.0 + (50 * len(responses))
        ))

    # Aggregate
    aggregator = EnsembleAggregator()
    result = aggregator.aggregate(responses, EnsembleStrategy.CONFIDENCE_WEIGHTED)

    print(f"\n  Ensemble Result:")
    print(f"    Agreement: {result.agreement_score:.2f}")
    print(f"    Avg confidence: {result.avg_confidence:.2f}")
    print(f"    Avg latency: {result.avg_latency_ms:.1f}ms")


# =============================================================================
# Demo 5: Performance Statistics
# =============================================================================

def demo_performance_stats():
    """Demonstrate performance tracking and statistics."""
    print_header("Demo 5: Performance Statistics")

    # Create registry and router
    registry = AgentRegistry()

    # Register agents
    for i in range(5):
        agent_id = f"agent_{i}"
        registry.register(
            agent_id=agent_id,
            capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
        )
        registry.update_status(agent_id, AgentStatus.IDLE)

    router = ExpertRouter(registry, enable_thompson=True)

    # Simulate many routing decisions
    print_section("Simulating 100 Routing Decisions")

    import random
    random.seed(42)

    for i in range(100):
        decision = router.route(
            required_capability=AgentCapability.RESEARCH,
            strategy=SelectionStrategy.THOMPSON if i % 3 == 0 else SelectionStrategy.BALANCED,
            top_k=1
        )

        # Simulate outcome
        success = random.random() < 0.7 + (0.1 if "agent_0" in decision.selected_agent else 0)
        confidence = random.uniform(0.6, 0.95) if success else random.uniform(0.3, 0.6)

        router.update_performance(
            decision.selected_agent,
            success=success,
            confidence=confidence,
            latency_ms=random.uniform(100, 300)
        )

    # Show statistics
    print_section("Routing Statistics")

    stats = router.get_routing_statistics()
    print(f"  Total routings: {stats['total_routings']}")
    print(f"  Thompson selections: {stats['thompson_selections']}")
    print(f"  Thompson rate: {stats['thompson_rate']:.1%}")
    print(f"  Agents tracked: {stats['agents_tracked']}")

    print_section("Per-Agent Performance")

    for agent_id, perf in stats['agent_performances'].items():
        print(f"\n  {agent_id}:")
        print(f"    Tasks: {perf['total_tasks']}")
        print(f"    Success rate: {perf['success_rate']:.1%}")
        print(f"    Avg confidence: {perf['avg_confidence']:.2f}")
        print(f"    Avg latency: {perf['avg_latency_ms']:.1f}ms")

        prior = router.get_agent_prior(agent_id)
        print(f"    Thompson prior: E[X]={prior['expected_value']:.3f}")

    # Show top agents
    print_section("Top Agents by Expected Reward")

    top_agents = router.get_top_agents(AgentCapability.RESEARCH, k=3)
    for rank, (agent_id, score) in enumerate(top_agents, 1):
        print(f"  {rank}. {agent_id}: {score:.3f}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  Phase 7.2: Task Delegation System - Demo Suite")
    print("  " + "=" * 56)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run demos
    demo_capability_detection()
    demo_expert_router()
    demo_ensemble_decisions()
    await demo_integrated_delegation()
    demo_performance_stats()

    print_header("Demo Complete!")
    print("Phase 7.2 Task Delegation System features demonstrated:")
    print("  1. ✓ Auto capability detection from queries")
    print("  2. ✓ Thompson Sampling-based agent selection")
    print("  3. ✓ Multiple selection strategies (BALANCED, RELIABLE, THOMPSON)")
    print("  4. ✓ Ensemble aggregation (MAJORITY_VOTE, CONFIDENCE_WEIGHTED, etc.)")
    print("  5. ✓ Performance tracking and statistics")
    print()


if __name__ == "__main__":
    asyncio.run(main())
