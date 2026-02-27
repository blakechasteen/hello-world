"""
MCTS Agent System Demo

Demonstrates Monte Carlo Tree Search at every decision point:
1. MCTS Working Memory - Explores semantic space
2. MCTS Pattern Validation - Simulates outcomes before applying
3. MCTS Orchestrator - Full MCTS integration
4. Hierarchical MCTS - Multi-step planning

Shows:
- Performance comparison: Standard vs MCTS
- Visualization of MCTS trees
- Statistics and improvements
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio

from hololoom.agents.orchestrator_mcts import create_mcts_agent
from hololoom.agents import create_agent
from hololoom.memory.graph import KG, KGEdge
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.documentation.types import Query
from hololoom.config import Config


def create_test_knowledge() -> KG:
    """Create test knowledge graph"""
    kg = KG()

    # Budget domain
    kg.add_edges([
        KGEdge("Q4_2024", "revenue", "HAS", 1.0),
        KGEdge("revenue", "500000", "AMOUNT", 1.0),
        KGEdge("Q4_2024", "expenses", "HAS", 1.0),
        KGEdge("expenses", "300000", "AMOUNT", 1.0),
        KGEdge("expenses", "marketing", "CATEGORY", 1.0),
        KGEdge("marketing", "120000", "AMOUNT", 1.0),
        KGEdge("expenses", "engineering", "CATEGORY", 1.0),
        KGEdge("engineering", "180000", "AMOUNT", 1.0),
        KGEdge("Q4_2024", "profit", "HAS", 1.0),
        KGEdge("profit", "200000", "AMOUNT", 1.0)
    ])

    return kg


async def demo_mcts_working_memory():
    """Demo 1: MCTS Working Memory vs Standard"""
    print("\n" + "="*70)
    print("DEMO 1: MCTS WORKING MEMORY")
    print("="*70)

    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()

    print("\n--- Baseline: Standard Working Memory ---")
    async with create_agent('budget', kg, emb) as standard_agent:
        query = Query(text="What is Q4 profit?")
        result = await standard_agent.query(query)

        print(f"Query: {query.text}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Response: {result.response[:150]}...")

        wm_summary = standard_agent.get_working_memory_summary()
        print(f"\nWorking Memory:")
        focus_dims = [str(d.name) if hasattr(d, 'name') else str(d) for d in wm_summary['semantic_focus'][:3]]
        print(f"  Semantic focus: {', '.join(focus_dims)}")
        print(f"  Activated nodes: {wm_summary['activated_nodes']}")

    print("\n--- MCTS: Explores 100 Focus Trajectories ---")
    async with create_mcts_agent(
        'budget',
        kg,
        emb,
        mcts_working_memory=True,
        mcts_wm_simulations=100
    ) as mcts_agent:
        query = Query(text="What is Q4 profit?")
        result = await mcts_agent.query(query, use_mcts=True)

        print(f"Query: {query.text}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Response: {result.response[:150]}...")

        wm_summary = mcts_agent.get_working_memory_summary()
        print(f"\nWorking Memory:")
        focus_dims = [str(d.name) if hasattr(d, 'name') else str(d) for d in wm_summary['semantic_focus'][:3]]
        print(f"  Semantic focus: {', '.join(focus_dims)}")
        print(f"  Activated nodes: {wm_summary['activated_nodes']}")

        # MCTS stats
        mcts_stats = mcts_agent.get_mcts_statistics()
        if mcts_stats['working_memory']:
            wm_stats = mcts_stats['working_memory']
            print(f"\nMCTS Statistics:")
            print(f"  Total searches: {wm_stats['total_searches']}")
            print(f"  Avg simulations: {wm_stats['avg_simulations']:.0f}")
            print(f"  Avg time: {wm_stats['avg_time']*1000:.1f}ms")
            print(f"  Avg reward improvement: {wm_stats['avg_reward_improvement']:.3f}")


async def demo_mcts_pattern_validation():
    """Demo 2: MCTS Pattern Validation"""
    print("\n" + "="*70)
    print("DEMO 2: MCTS PATTERN VALIDATION")
    print("="*70)

    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()
    persist_dir = Path('./demo_mcts_memory')

    print("\n--- Train Agent (build patterns) ---")
    async with create_mcts_agent(
        'budget',
        kg,
        emb,
        persist_dir=persist_dir,
        mcts_pattern_validation=True,
        mcts_pattern_simulations=50
    ) as agent:
        # Train with similar queries
        training_queries = [
            "What is Q4 revenue?",
            "What are Q4 expenses?",
            "What is Q4 profit?"
        ]

        for q_text in training_queries:
            result = await agent.query(Query(text=q_text), use_mcts=True)
            print(f"  {q_text} -> confidence: {result.confidence:.3f}")

        learning_stats = agent.get_learning_stats()
        print(f"\nPatterns learned: {learning_stats.get('patterns_learned', 0)}")

    print("\n--- Query with Pattern Validation ---")
    async with create_mcts_agent(
        'budget',
        kg,
        emb,
        persist_dir=persist_dir,
        mcts_pattern_validation=True,
        mcts_pattern_simulations=50
    ) as agent:
        # New query (similar to training)
        query = Query(text="What is Q4 margin?")
        result = await agent.query(query, use_mcts=True, apply_learned_patterns=True)

        print(f"Query: {query.text}")
        print(f"Confidence: {result.confidence:.3f}")

        # MCTS validation stats
        mcts_stats = agent.get_mcts_statistics()
        if mcts_stats['pattern_validation']:
            pv_stats = mcts_stats['pattern_validation']
            print(f"\nPattern Validation Statistics:")
            print(f"  Total validations: {pv_stats['total_validations']}")
            print(f"  Patterns accepted: {pv_stats['patterns_accepted']}")
            print(f"  Patterns rejected: {pv_stats['patterns_rejected']}")
            print(f"  Acceptance rate: {pv_stats['acceptance_rate']:.1%}")
            print(f"  Avg validation time: {pv_stats['avg_validation_time']*1000:.1f}ms")


async def demo_mcts_vs_standard():
    """Demo 3: MCTS vs Standard Performance Comparison"""
    print("\n" + "="*70)
    print("DEMO 3: PERFORMANCE COMPARISON")
    print("="*70)

    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()

    queries = [
        "What is Q4 revenue?",
        "What are Q4 expenses?",
        "What is Q4 profit?",
        "How much was spent on marketing?",
        "How much was spent on engineering?"
    ]

    print("\n--- Standard Agent ---")
    standard_confidences = []
    async with create_agent('budget', kg, emb) as agent:
        for q in queries:
            result = await agent.query(Query(text=q))
            standard_confidences.append(result.confidence)
            print(f"  {q} -> {result.confidence:.3f}")

        stats = agent.get_agent_stats()
        print(f"\nStandard Agent Stats:")
        print(f"  Avg confidence: {stats.avg_confidence:.3f}")
        print(f"  Success rate: {stats.success_rate():.1%}")

    print("\n--- MCTS Agent (100 simulations) ---")
    mcts_confidences = []
    async with create_mcts_agent(
        'budget',
        kg,
        emb,
        mcts_working_memory=True,
        mcts_wm_simulations=100
    ) as agent:
        for q in queries:
            result = await agent.query(Query(text=q), use_mcts=True)
            mcts_confidences.append(result.confidence)
            print(f"  {q} -> {result.confidence:.3f}")

        stats = agent.get_agent_stats()
        print(f"\nMCTS Agent Stats:")
        print(f"  Avg confidence: {stats.avg_confidence:.3f}")
        print(f"  Success rate: {stats.success_rate():.1%}")

        mcts_stats = agent.get_mcts_statistics()
        if mcts_stats['working_memory']:
            wm_stats = mcts_stats['working_memory']
            print(f"  Avg MCTS time: {wm_stats['avg_time']*1000:.1f}ms")

    # Comparison
    import numpy as np
    improvement = np.mean(mcts_confidences) - np.mean(standard_confidences)
    print(f"\n--- Comparison ---")
    print(f"Standard avg confidence: {np.mean(standard_confidences):.3f}")
    print(f"MCTS avg confidence: {np.mean(mcts_confidences):.3f}")
    print(f"Improvement: {improvement:+.3f} ({improvement/np.mean(standard_confidences)*100:+.1f}%)")


async def demo_hierarchical_planning():
    """Demo 4: Hierarchical MCTS Planning"""
    print("\n" + "="*70)
    print("DEMO 4: HIERARCHICAL MCTS PLANNING")
    print("="*70)

    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()

    print("\n--- Planning: Create Q4 Budget Report ---")

    async with create_mcts_agent('budget', kg, emb) as agent:
        from hololoom.agents.planner_mcts import HierarchicalMCTSPlanner

        planner = HierarchicalMCTSPlanner(
            agent=agent,
            macro_budget=30,   # Reduced for demo speed
            meso_budget=50,
            micro_budget=100
        )

        target_goal = "Create Q4 budget report"
        plan = await planner.plan(target_goal)

        print(f"\nTarget Goal: {target_goal}")
        print(f"Generated Plan ({len(plan)} steps):\n")

        for i, step in enumerate(plan, 1):
            print(f"{i}. Goal: {step['goal']}")
            print(f"   Query: {step['query']}")
            print(f"   Parameters: {step['parameters'].mcts_simulations} simulations")
            print(f"   Estimated confidence: {step['estimated_confidence']:.2f}\n")

        print("--- Execute First 3 Steps ---")
        results = []
        for step in plan[:3]:
            query = Query(text=step['query'])
            result = await agent.query(query, use_mcts=True)
            results.append(result)
            print(f"[OK] {step['query']}")
            print(f"  Confidence: {result.confidence:.3f}")


async def main():
    """Run all MCTS demos"""
    print("\n" + "="*70)
    print("MCTS AGENT SYSTEM - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nMonte Carlo Tree Search at EVERY decision point:")
    print("  * Working Memory: Explore semantic space")
    print("  * Pattern Validation: Simulate before applying")
    print("  * Orchestrator: Integrated MCTS pipeline")
    print("  * Hierarchical Planning: Multi-step MCTS")

    await demo_mcts_working_memory()
    await demo_mcts_pattern_validation()
    await demo_mcts_vs_standard()
    await demo_hierarchical_planning()

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  * MCTS improves context retrieval through exploration")
    print("  * Pattern validation prevents bad patterns")
    print("  * Hierarchical planning enables multi-step reasoning")
    print("  * All with transparent statistics and provenance")


if __name__ == "__main__":
    asyncio.run(main())
