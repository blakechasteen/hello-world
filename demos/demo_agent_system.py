"""
Agent System Demo

Demonstrates specialized agent bots with working memory and learning:
1. Budget Advisor - Financial planning
2. Architecture Reviewer - Software design
3. Shared knowledge graph
4. Pattern learning from success
5. Persistent working memory (pins)

Shows:
- Domain-specific agents
- Trinity working memory (semantic + graph + computational)
- Learning from successful queries
- Cross-session persistence
"""

import asyncio
from pathlib import Path

from HoloLoom.agents import create_agent, list_profiles
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config


def create_test_knowledge() -> KG:
    """Create shared knowledge graph for demo"""
    kg = KG()

    # Budget domain knowledge
    kg.add_edges([
        KGEdge("Q4_2024", "marketing_budget", "HAS_BUDGET", 1.0),
        KGEdge("marketing_budget", "50000", "ALLOCATED_AMOUNT", 1.0),
        KGEdge("marketing_budget", "digital_ads", "INCLUDES", 1.0),
        KGEdge("marketing_budget", "content_creation", "INCLUDES", 1.0),
        KGEdge("digital_ads", "30000", "ALLOCATED_AMOUNT", 1.0),
        KGEdge("content_creation", "20000", "ALLOCATED_AMOUNT", 1.0),
        KGEdge("Q4_2024", "engineering_budget", "HAS_BUDGET", 1.0),
        KGEdge("engineering_budget", "200000", "ALLOCATED_AMOUNT", 1.0),
    ])

    # Architecture domain knowledge
    kg.add_edges([
        KGEdge("microservices", "architecture_pattern", "IS_A", 1.0),
        KGEdge("monolith", "architecture_pattern", "IS_A", 1.0),
        KGEdge("microservices", "scalability", "IMPROVES", 1.0),
        KGEdge("microservices", "complexity", "INCREASES", 1.0),
        KGEdge("microservices", "deployment_overhead", "INCREASES", 1.0),
        KGEdge("monolith", "simplicity", "IMPROVES", 1.0),
        KGEdge("monolith", "scalability", "LIMITS", 1.0),
        KGEdge("API_gateway", "microservices", "COMPONENT_OF", 1.0),
        KGEdge("service_mesh", "microservices", "COMPONENT_OF", 1.0),
    ])

    # Cross-domain knowledge
    kg.add_edges([
        KGEdge("microservices", "infrastructure_cost", "INCREASES", 1.0),
        KGEdge("monolith", "infrastructure_cost", "REDUCES", 1.0),
    ])

    return kg


async def demo_budget_advisor():
    """Demonstrate budget advisor agent"""
    print("\n" + "="*60)
    print("BUDGET ADVISOR DEMO")
    print("="*60)

    # Create shared knowledge
    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()

    # Create budget advisor
    async with create_agent(
        'budget',
        shared_knowledge=kg,
        embedding_model=emb,
        config=Config.fast()
    ) as budget_agent:

        print(f"\nAgent: {budget_agent.profile.name}")
        print(f"Domain: {budget_agent.profile.domain.value}")
        print(f"Priorities: {', '.join(budget_agent.profile.priorities)}")

        # Pin persistent context
        print("\n--- Pinning Context ---")
        budget_agent.pin("company budget policy 2024", weight=0.7)
        budget_agent.pin("Q4 2024 fiscal planning", weight=0.6)
        print("✓ Pinned: company budget policy 2024")
        print("✓ Pinned: Q4 2024 fiscal planning")

        # Query 1: Marketing budget
        print("\n--- Query 1: Marketing Budget ---")
        query1 = Query(text="What is the Q4 2024 marketing budget?")
        result1 = await budget_agent.query(query1)
        print(f"Query: {query1.text}")
        print(f"Confidence: {result1.confidence:.2f}")
        print(f"Response: {result1.response[:200]}...")

        # View working memory
        wm_summary = budget_agent.get_working_memory_summary()
        print(f"\nWorking Memory:")
        print(f"  Semantic focus: {', '.join(wm_summary['semantic_focus'][:3])}")
        print(f"  Activated nodes: {wm_summary['activated_nodes']}")
        print(f"  Tensioned threads: {wm_summary['tensioned_threads']}")
        print(f"  Pinned concepts: {wm_summary['pinned_concepts']}")

        # Query 2: Budget breakdown
        print("\n--- Query 2: Budget Breakdown ---")
        query2 = Query(text="How is the marketing budget allocated?")
        result2 = await budget_agent.query(query2)
        print(f"Query: {query2.text}")
        print(f"Confidence: {result2.confidence:.2f}")
        print(f"Response: {result2.response[:200]}...")

        # Query 3: Cost analysis
        print("\n--- Query 3: Cost Analysis ---")
        query3 = Query(text="Compare marketing vs engineering budget")
        result3 = await budget_agent.query(query3)
        print(f"Query: {query3.text}")
        print(f"Confidence: {result3.confidence:.2f}")

        # View learning stats
        learning_stats = budget_agent.get_learning_stats()
        print(f"\nLearning Stats:")
        print(f"  Total snapshots: {learning_stats.get('total_snapshots', 0)}")
        print(f"  Successful: {learning_stats.get('successful_snapshots', 0)}")
        print(f"  Patterns learned: {learning_stats.get('patterns_learned', 0)}")
        if learning_stats.get('successful_snapshots', 0) > 0:
            print(f"  Avg confidence: {learning_stats.get('avg_confidence_successful', 0):.2f}")

        # View agent stats
        stats = budget_agent.get_agent_stats()
        print(f"\nAgent Stats:")
        print(f"  Total queries: {stats.total_queries}")
        print(f"  Successful: {stats.successful_queries}")
        print(f"  Success rate: {stats.success_rate():.1%}")
        print(f"  Avg confidence: {stats.avg_confidence:.2f}")


async def demo_architecture_reviewer():
    """Demonstrate architecture reviewer agent"""
    print("\n" + "="*60)
    print("ARCHITECTURE REVIEWER DEMO")
    print("="*60)

    # Create shared knowledge
    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()

    # Create architecture reviewer
    async with create_agent(
        'architecture',
        shared_knowledge=kg,
        embedding_model=emb,
        config=Config.fast()
    ) as arch_agent:

        print(f"\nAgent: {arch_agent.profile.name}")
        print(f"Domain: {arch_agent.profile.domain.value}")
        print(f"Priorities: {', '.join(arch_agent.profile.priorities)}")

        # Pin persistent context
        print("\n--- Pinning Context ---")
        arch_agent.pin("SOLID principles", weight=0.7)
        arch_agent.pin("scalability requirements", weight=0.6)
        print("✓ Pinned: SOLID principles")
        print("✓ Pinned: scalability requirements")

        # Query 1: Microservices vs Monolith
        print("\n--- Query 1: Architecture Pattern ---")
        query1 = Query(text="Should we use microservices or monolith?")
        result1 = await arch_agent.query(query1)
        print(f"Query: {query1.text}")
        print(f"Confidence: {result1.confidence:.2f}")
        print(f"Response: {result1.response[:200]}...")

        # View working memory
        wm_summary = arch_agent.get_working_memory_summary()
        print(f"\nWorking Memory:")
        print(f"  Semantic focus: {', '.join(wm_summary['semantic_focus'][:3])}")
        print(f"  Activated nodes: {wm_summary['activated_nodes']}")
        print(f"  Attention radius: {wm_summary['attention_radius']}")

        # Query 2: Scalability
        print("\n--- Query 2: Scalability Analysis ---")
        query2 = Query(text="How does microservices improve scalability?")
        result2 = await arch_agent.query(query2)
        print(f"Query: {query2.text}")
        print(f"Confidence: {result2.confidence:.2f}")
        print(f"Response: {result2.response[:200]}...")

        # View learning
        learning_stats = arch_agent.get_learning_stats()
        print(f"\nLearning Stats:")
        print(f"  Patterns learned: {learning_stats.get('patterns_learned', 0)}")

        # View stats
        stats = arch_agent.get_agent_stats()
        print(f"\nAgent Stats:")
        print(f"  Total queries: {stats.total_queries}")
        print(f"  Success rate: {stats.success_rate():.1%}")


async def demo_cross_agent_knowledge():
    """Demonstrate agents sharing knowledge graph"""
    print("\n" + "="*60)
    print("CROSS-AGENT KNOWLEDGE SHARING DEMO")
    print("="*60)

    # Shared knowledge
    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()

    # Create both agents sharing same knowledge graph
    budget_agent = create_agent('budget', kg, emb)
    arch_agent = create_agent('architecture', kg, emb)

    async with budget_agent, arch_agent:
        # Budget agent queries architecture impact
        print("\n--- Budget Agent Queries Architecture Cost ---")
        query = Query(text="What's the infrastructure cost of microservices?")
        result = await budget_agent.query(query)
        print(f"Budget Agent Query: {query.text}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response: {result.response[:200]}...")

        # Architecture agent queries budget constraints
        print("\n--- Architecture Agent Queries Budget ---")
        query2 = Query(text="Do we have budget for microservices infrastructure?")
        result2 = await arch_agent.query(query2)
        print(f"Architecture Agent Query: {query2.text}")
        print(f"Confidence: {result2.confidence:.2f}")
        print(f"Response: {result2.response[:200]}...")

        print("\n✓ Both agents accessed shared knowledge graph")
        print("✓ Each prioritized different semantic regions")


async def demo_learning_persistence():
    """Demonstrate pattern learning and persistence"""
    print("\n" + "="*60)
    print("LEARNING & PERSISTENCE DEMO")
    print("="*60)

    kg = create_test_knowledge()
    emb = MatryoshkaEmbeddings()
    persist_dir = Path('./demo_agents_memory')

    # Session 1: Train budget agent
    print("\n--- Session 1: Initial Queries ---")
    async with create_agent(
        'budget',
        kg,
        emb,
        persist_dir=persist_dir
    ) as agent:
        # Query similar questions to build pattern
        queries = [
            "What is Q4 marketing budget?",
            "What is Q4 engineering budget?",
            "Show me Q4 budget breakdown"
        ]

        for q in queries:
            result = await agent.query(Query(text=q))
            print(f"  {q} → confidence: {result.confidence:.2f}")

        learning_stats = agent.get_learning_stats()
        print(f"\nPatterns learned: {learning_stats.get('patterns_learned', 0)}")

    # Session 2: Load agent and verify learning persisted
    print("\n--- Session 2: Verify Persistence ---")
    async with create_agent(
        'budget',
        kg,
        emb,
        persist_dir=persist_dir
    ) as agent:
        learning_stats = agent.get_learning_stats()
        print(f"Loaded patterns: {learning_stats.get('patterns_learned', 0)}")
        print(f"Loaded snapshots: {learning_stats.get('total_snapshots', 0)}")

        # Query should benefit from learned patterns
        result = await agent.query(Query(text="What's the Q4 budget?"))
        print(f"\nNew query with learned patterns:")
        print(f"  Confidence: {result.confidence:.2f}")

    print("\n✓ Learning state persisted across sessions")


async def main():
    """Run all demos"""
    print("\n🚀 HOLOLOOM AGENT SYSTEM DEMO")
    print("\nAvailable agent profiles:")
    for profile_name in list_profiles():
        print(f"  - {profile_name}")

    # Run demos
    await demo_budget_advisor()
    await demo_architecture_reviewer()
    await demo_cross_agent_knowledge()
    await demo_learning_persistence()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Domain-specific agents (budget, architecture)")
    print("  ✓ Trinity working memory (semantic + graph + computational)")
    print("  ✓ Pinned persistent context")
    print("  ✓ Pattern learning from success")
    print("  ✓ Cross-agent knowledge sharing")
    print("  ✓ Persistent learning across sessions")


if __name__ == "__main__":
    asyncio.run(main())
