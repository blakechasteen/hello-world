"""
Demo: Agentic Web Researcher
=============================
Autonomous multi-step research agent with planning, execution, and verification.
"""

import asyncio
from hololoom.agentic.web_researcher import (
    AgenticWebResearcher,
    ResearchStrategy,
    research_web
)


async def demo_quick_research():
    """Demo 1: Quick research (no crawling)"""
    print("=" * 80)
    print("DEMO 1: QUICK RESEARCH")
    print("=" * 80)
    print()

    result = await research_web(
        query="What is Thompson Sampling?",
        strategy=ResearchStrategy.QUICK,
        enable_verification=False
    )

    print(f"Strategy: {result.plan.strategy.value}")
    print(f"Sub-queries: {len(result.plan.sub_queries)}")
    print(f"Total pages: {result.total_pages_crawled}")
    print(f"Total shards: {result.total_shards_created}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Duration: {result.total_duration_ms:.1f}ms")
    print()
    print("Synthesis:")
    print("-" * 80)
    print(result.synthesis[:500])
    print("...")
    print()
    print("-" * 80)
    print()


async def demo_standard_research():
    """Demo 2: Standard research with verification"""
    print("=" * 80)
    print("DEMO 2: STANDARD RESEARCH (with verification)")
    print("=" * 80)
    print()

    result = await research_web(
        query="How does Bayesian inference work?",
        strategy=ResearchStrategy.STANDARD,
        enable_verification=True,
        max_sub_queries=3
    )

    print(f"Strategy: {result.plan.strategy.value}")
    print(f"Sub-queries: {len(result.plan.sub_queries)}")
    for i, sq in enumerate(result.plan.sub_queries, 1):
        print(f"  {i}. {sq}")
    print()

    print(f"Total pages: {result.total_pages_crawled}")
    print(f"Total shards: {result.total_shards_created}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Duration: {result.total_duration_ms:.1f}ms")
    print()

    if result.verification_results:
        print("Verification:")
        print(f"  Verified: {result.verification_results['verified']}")
        print(f"  Confidence: {result.verification_results['confidence']:.2f}")
        if result.verification_results['issues']:
            print(f"  Issues: {result.verification_results['issues']}")
    print()

    print("Key Findings:")
    print("-" * 80)
    lines = result.synthesis.split('\n')
    for line in lines[6:16]:  # Print key findings section
        print(line)
    print()
    print("-" * 80)
    print()


async def demo_comprehensive_research():
    """Demo 3: Comprehensive research (deep crawl)"""
    print("=" * 80)
    print("DEMO 3: COMPREHENSIVE RESEARCH (deep crawl)")
    print("=" * 80)
    print()

    result = await research_web(
        query="What are the tradeoffs of Thompson Sampling versus UCB?",
        strategy=ResearchStrategy.COMPREHENSIVE,
        enable_verification=True,
        max_sub_queries=4
    )

    print(f"Strategy: {result.plan.strategy.value}")
    print(f"Sub-queries: {len(result.plan.sub_queries)}")
    print(f"Total searches: {len(result.search_results)}")
    print(f"Total pages: {result.total_pages_crawled}")
    print(f"Total shards: {result.total_shards_created}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Duration: {result.total_duration_ms:.1f}ms")
    print()

    print("Search Results Breakdown:")
    for i, sr in enumerate(result.search_results, 1):
        print(f"  Query {i}: {len(sr.search_results)} results, "
              f"{len(sr.crawled_pages)} pages, "
              f"{len(sr.shards)} shards")
    print()

    if result.verification_results:
        print("Verification:")
        print(f"  Verified: {result.verification_results['verified']}")
        if 'shard_count' in result.verification_results:
            print(f"  Shard count: {result.verification_results['shard_count']}")
        if 'domain_count' in result.verification_results:
            print(f"  Domain count: {result.verification_results['domain_count']}")
    print()
    print("-" * 80)
    print()


async def demo_exploratory_research():
    """Demo 4: Exploratory research (broad exploration)"""
    print("=" * 80)
    print("DEMO 4: EXPLORATORY RESEARCH (broad exploration)")
    print("=" * 80)
    print()

    researcher = AgenticWebResearcher(
        strategy=ResearchStrategy.EXPLORATORY,
        enable_verification=True
    )

    result = await researcher.research(
        query="machine learning",
        max_sub_queries=3
    )

    print(f"Strategy: {result.plan.strategy.value}")
    print(f"Sub-queries: {len(result.plan.sub_queries)}")
    print(f"Total pages: {result.total_pages_crawled}")
    print(f"Total shards: {result.total_shards_created}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Duration: {result.total_duration_ms:.1f}ms")
    print()

    # Get statistics
    stats = researcher.get_stats()
    print("Researcher Statistics:")
    print(f"  Total research queries: {stats['total_research_queries']}")
    print(f"  Total sub-queries: {stats['total_sub_queries']}")
    print(f"  Avg pages per query: {stats['avg_pages_per_query']:.1f}")
    print(f"  Avg shards per query: {stats['avg_shards_per_query']:.1f}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    if stats['verifications_passed'] + stats['verifications_failed'] > 0:
        print(f"  Verification pass rate: {stats['verification_pass_rate']:.2%}")
    print()
    print("-" * 80)
    print()


async def demo_strategy_comparison():
    """Demo 5: Compare all strategies"""
    print("=" * 80)
    print("DEMO 5: STRATEGY COMPARISON")
    print("=" * 80)
    print()

    query = "What is Thompson Sampling?"
    strategies = [
        ResearchStrategy.QUICK,
        ResearchStrategy.STANDARD,
        ResearchStrategy.COMPREHENSIVE,
        ResearchStrategy.EXPLORATORY
    ]

    results = []
    for strategy in strategies:
        print(f"Running {strategy.value}...")
        result = await research_web(
            query,
            strategy=strategy,
            enable_verification=False,  # Fast comparison
            max_sub_queries=2
        )
        results.append(result)

    print()
    print("Comparison Table:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'Duration (ms)':<15} {'Pages':<8} {'Shards':<8} {'Confidence':<12}")
    print("-" * 80)

    for i, result in enumerate(results):
        print(f"{strategies[i].value:<15} "
              f"{result.total_duration_ms:<15.1f} "
              f"{result.total_pages_crawled:<8} "
              f"{result.total_shards_created:<8} "
              f"{result.confidence:<12.2f}")

    print("-" * 80)
    print()


async def main():
    """Run all demos"""
    print()
    print("=" * 80)
    print(" " * 20 + "AGENTIC WEB RESEARCHER DEMO")
    print("=" * 80)
    print()

    try:
        # Demo 1: Quick
        await demo_quick_research()

        # Demo 2: Standard
        await demo_standard_research()

        # Demo 3: Comprehensive
        await demo_comprehensive_research()

        # Demo 4: Exploratory
        await demo_exploratory_research()

        # Demo 5: Comparison
        await demo_strategy_comparison()

        print("=" * 80)
        print("ALL DEMOS COMPLETE!")
        print("=" * 80)
        print()
        print("Features Demonstrated:")
        print("  [x] Autonomous planning (query decomposition)")
        print("  [x] Multi-step execution (sub-queries)")
        print("  [x] MatryoshkaWebSearch (3-stage filtering)")
        print("  [x] RecursiveCrawler (importance gating)")
        print("  [x] Verification (consistency checking)")
        print("  [x] Synthesis (comprehensive reports)")
        print("  [x] 4 research strategies (quick/standard/comprehensive/exploratory)")
        print("  [x] Statistics tracking")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
