"""
Background Learning Demo

Demonstrates agents that learn continuously in the background.

Shows:
1. Agent pool with multiple specialized agents
2. Foreground queries (user requests)
3. Background learning from experiences
4. Pattern accumulation over time
5. Persistent patterns across sessions

Key Insight: Agents "think" when idle, building expertise that serves future queries.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import time

from HoloLoom.agents.background_learner import create_agent_pool
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query


async def create_test_knowledge():
    """Create test knowledge graph and embeddings"""
    kg = KG()

    # Budget domain knowledge
    edges = [
        # Q4 2024
        KGEdge("Q4_2024", "revenue", "HAS_METRIC", 1.0, {"value": 500000}),
        KGEdge("Q4_2024", "expenses", "HAS_METRIC", 1.0, {"value": 320000}),
        KGEdge("Q4_2024", "profit", "DERIVED", 1.0, {"value": 180000}),

        # Q3 2024 (for comparison)
        KGEdge("Q3_2024", "revenue", "HAS_METRIC", 1.0, {"value": 450000}),
        KGEdge("Q3_2024", "expenses", "HAS_METRIC", 1.0, {"value": 300000}),
        KGEdge("Q3_2024", "profit", "DERIVED", 1.0, {"value": 150000}),

        # Expense breakdown
        KGEdge("expenses", "marketing", "BREAKDOWN", 1.0, {"value": 120000}),
        KGEdge("expenses", "engineering", "BREAKDOWN", 1.0, {"value": 150000}),
        KGEdge("expenses", "operations", "BREAKDOWN", 1.0, {"value": 50000}),

        # Trends
        KGEdge("revenue", "growth_trend", "HAS_TREND", 1.0),
        KGEdge("profit", "increasing_trend", "HAS_TREND", 1.0),
        KGEdge("marketing", "roi_analysis", "NEEDS", 1.0),
    ]

    kg.add_edges(edges)

    # Create embeddings
    emb = MatryoshkaEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        sizes=[768, 512, 256, 128],
        default_size=768
    )

    return kg, emb


async def demo_interactive_learning():
    """
    Demo 1: Interactive Learning

    User makes queries, provides feedback, agents learn in background.
    """
    print("\n" + "="*70)
    print("DEMO 1: INTERACTIVE LEARNING WITH FEEDBACK")
    print("="*70)

    kg, emb = await create_test_knowledge()

    async with await create_agent_pool(
        kg,
        emb,
        persist_path=Path("./demo_agent_patterns"),
        enable_background_learning=True,
        mcts_simulations=50,  # Lower for demo speed
        exploration_simulations=25
    ) as pool:
        print("\n[AgentPool] Started with background learning enabled")

        # Simulate user session
        queries = [
            ("budget", "What is Q4 revenue?", True),
            ("budget", "What is Q4 profit?", True),
            ("budget", "How much was spent on marketing?", True),
            ("budget", "How does profit compare to Q3?", False),  # Bad result
            ("budget", "What are spending trends?", True),
        ]

        print("\n--- User Session (foreground queries + feedback) ---")
        for agent_name, query_text, helpful in queries:
            query = Query(text=query_text)

            # Process query
            start = time.time()
            result = await pool.query(agent_name, query, use_mcts=True)
            elapsed = time.time() - start

            print(f"\nQ: {query_text}")
            print(f"A: {result.response[:100]}...")
            print(f"Confidence: {result.confidence:.3f} | Time: {elapsed*1000:.1f}ms")

            # Provide feedback (goes to background learning queue)
            feedback = {
                'helpful': helpful,
                'confidence': result.confidence,
                'latency_ms': elapsed * 1000
            }
            await pool.feedback(agent_name, query, result, feedback)

            print(f"Feedback: {'Helpful' if helpful else 'Not helpful'} "
                  f"(queued for background learning)")

            # Give background learner time to process
            await asyncio.sleep(0.5)

        # Wait for background learning to catch up
        print("\n--- Waiting for background learning (5s) ---")
        await asyncio.sleep(5.0)

        # Show statistics
        stats = pool.stats()
        print("\n--- Statistics ---")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Learning queue: {stats['learning_queue']['pending']} pending")
        print(f"Background learner:")
        print(f"  Cycles: {stats['background_learner']['learning_cycles']}")
        print(f"  Patterns learned: {stats['background_learner']['patterns_learned']}")
        print(f"  Avg time/cycle: {stats['background_learner']['avg_time_per_cycle_ms']:.1f}ms")

        print("\nPatterns by agent:")
        for agent, count in stats['patterns_by_agent'].items():
            print(f"  {agent}: {count} patterns")


async def demo_persistent_learning():
    """
    Demo 2: Persistent Learning

    Shows patterns persisting across sessions.
    """
    print("\n" + "="*70)
    print("DEMO 2: PERSISTENT LEARNING (across sessions)")
    print("="*70)

    kg, emb = await create_test_knowledge()

    # Session 1: Build patterns
    print("\n--- Session 1: Initial Learning ---")
    async with await create_agent_pool(
        kg,
        emb,
        persist_path=Path("./demo_agent_patterns"),
        enable_background_learning=True,
        mcts_simulations=50,
        exploration_simulations=25
    ) as pool:
        # Train agent
        queries = [
            "What is Q4 revenue?",
            "What is Q4 profit?",
            "What is Q4 margin?"
        ]

        for q in queries:
            result = await pool.query('budget', Query(text=q), use_mcts=True)
            print(f"  {q} -> {result.confidence:.3f}")

        # Wait for learning
        await asyncio.sleep(3.0)

        stats = pool.stats()
        patterns_session1 = stats['patterns_by_agent'].get('budget', 0)
        print(f"\nPatterns learned in session 1: {patterns_session1}")

    # Session 2: Reload patterns
    print("\n--- Session 2: Reload Patterns ---")
    async with await create_agent_pool(
        kg,
        emb,
        persist_path=Path("./demo_agent_patterns"),
        enable_background_learning=True,
        mcts_simulations=50,
        exploration_simulations=25
    ) as pool:
        # Patterns should be reloaded
        stats = pool.stats()
        patterns_session2 = stats['patterns_by_agent'].get('budget', 0)
        print(f"Patterns available in session 2: {patterns_session2}")

        if patterns_session2 >= patterns_session1:
            print("[OK] Patterns persisted across sessions!")
        else:
            print("[WARN] Patterns not fully restored")

        # Continue learning
        new_queries = [
            "How does revenue compare to Q3?",
            "What are the main expense categories?"
        ]

        for q in new_queries:
            result = await pool.query('budget', Query(text=q), use_mcts=True)
            print(f"  {q} -> {result.confidence:.3f}")

        await asyncio.sleep(3.0)

        stats = pool.stats()
        patterns_final = stats['patterns_by_agent'].get('budget', 0)
        print(f"\nPatterns after session 2: {patterns_final}")
        print(f"Total learning: +{patterns_final - patterns_session1} patterns")


async def demo_multi_agent_learning():
    """
    Demo 3: Multi-Agent Learning

    Multiple specialized agents learning simultaneously.
    """
    print("\n" + "="*70)
    print("DEMO 3: MULTI-AGENT BACKGROUND LEARNING")
    print("="*70)

    kg, emb = await create_test_knowledge()

    async with await create_agent_pool(
        kg,
        emb,
        persist_path=Path("./demo_agent_patterns"),
        enable_background_learning=True,
        mcts_simulations=50,
        exploration_simulations=25
    ) as pool:
        print("\n[AgentPool] Multiple agents learning in parallel")

        # Queries for different agents
        queries = [
            ("budget", "What is Q4 revenue?"),
            ("budget", "What is Q4 profit?"),
            ("research", "What are growth trends?"),
            ("research", "What is ROI analysis?"),
            ("budget", "What are expenses?"),
        ]

        print("\n--- Processing Queries (different agents) ---")
        for agent_name, query_text in queries:
            result = await pool.query(agent_name, Query(text=query_text), use_mcts=True)
            print(f"[{agent_name}] {query_text[:40]} -> {result.confidence:.3f}")

            # Queue for learning
            await pool.feedback(agent_name, Query(text=query_text), result, {'helpful': True})

        # Wait for background learning
        print("\n--- Background learning (5s) ---")
        await asyncio.sleep(5.0)

        # Statistics
        stats = pool.stats()
        print("\n--- Multi-Agent Statistics ---")
        print(f"Active agents: {', '.join(stats['active_agents'])}")
        print(f"Total queries: {stats['total_queries']}")

        print("\nQueries by agent:")
        for agent, count in stats['query_count_by_agent'].items():
            print(f"  {agent}: {count} queries")

        print("\nPatterns by agent:")
        for agent, count in stats['patterns_by_agent'].items():
            print(f"  {agent}: {count} patterns")

        print(f"\nBackground learner:")
        bg_stats = stats['background_learner']
        print(f"  Cycles: {bg_stats['learning_cycles']}")
        print(f"  Patterns learned: {bg_stats['patterns_learned']}")
        print(f"  Total time: {bg_stats['total_exploration_time_s']:.1f}s")


async def main():
    """Run all background learning demos"""
    print("\n" + "="*70)
    print("BACKGROUND MCTS LEARNING - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nAgents that learn continuously in the background:")
    print("  * Process user queries (foreground)")
    print("  * Learn from feedback (background)")
    print("  * Build expertise over time")
    print("  * Persist patterns across sessions")

    await demo_interactive_learning()
    await demo_persistent_learning()
    await demo_multi_agent_learning()

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  * Agents learn from every interaction")
    print("  * Background MCTS explores continuously")
    print("  * Patterns persist across sessions")
    print("  * Multiple agents learn in parallel")
    print("  * Zero impact on foreground query latency")


if __name__ == "__main__":
    asyncio.run(main())
