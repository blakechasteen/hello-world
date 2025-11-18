#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: Context Compression Demo

Demonstrates:
1. Basic compression usage
2. Strategy comparison (merge vs keep_best vs summarize)
3. Token savings measurement
4. Quality impact analysis
5. Compression insights and recommendations
6. Integration with Phases 1-3 (adaptive budgeting + outcome tracking)

Expected Results:
- 10-30% token reduction for typical workloads
- <2% quality drop with merge_similar strategy
- Clear recommendations for optimization
"""

import asyncio
from typing import List

# Import Phase 4 components
from HoloLoom.awareness.semantic_deduplicator import (
    SemanticDeduplicator,
    DeduplicationStrategy,
    SimilarityMethod,
    ContextElement
)

from HoloLoom.awareness.compression_analyzer import (
    CompressionAnalyzer
)


# ============================================================================
# Demo 1: Basic Compression Usage
# ============================================================================

def demo_1_basic_compression():
    """Demo 1: Basic compression with duplicate detection"""
    print("\n" + "="*80)
    print("DEMO 1: Basic Compression Usage")
    print("="*80)

    # Create deduplicator
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.8,  # 80% similar
        min_tokens_to_deduplicate=50
    )

    # Create context elements with some duplicates
    elements = [
        ContextElement(
            content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            importance=0.9,
            token_count=100,
            source="memory"
        ),
        ContextElement(
            content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration.",
            importance=0.8,
            token_count=120,
            source="memory"
        ),
        ContextElement(
            content="Epsilon-greedy is another approach to exploration-exploitation tradeoffs.",
            importance=0.7,
            token_count=80,
            source="memory"
        ),
        ContextElement(
            content="Thompson Sampling uses probability matching for optimal decisions.",
            importance=0.75,
            token_count=90,
            source="memory"
        ),
    ]

    print(f"\n📊 Original Context:")
    print(f"   Elements: {len(elements)}")
    print(f"   Total tokens: {sum(e.token_count for e in elements)}")

    # Apply compression
    result = deduplicator.deduplicate(
        elements,
        strategy=DeduplicationStrategy.MERGE_SIMILAR
    )

    print(f"\n✨ Compressed Context:")
    print(f"   Elements: {len(result.deduplicated_elements)}")
    print(f"   Total tokens: {result.deduplicated_tokens}")
    print(f"   Token savings: {result.token_savings} ({(1 - result.compression_ratio) * 100:.1f}%)")
    print(f"   Elements removed: {result.elements_removed}")
    print(f"   Elements merged: {result.elements_merged}")
    print(f"   Duplicate groups found: {len(result.duplicate_groups)}")

    print(f"\n{result.summary()}")


# ============================================================================
# Demo 2: Strategy Comparison
# ============================================================================

def demo_2_strategy_comparison():
    """Demo 2: Compare different deduplication strategies"""
    print("\n" + "="*80)
    print("DEMO 2: Strategy Comparison")
    print("="*80)

    # Create elements with duplicates
    elements = [
        ContextElement("Python is great for data science", 0.9, 100, "memory"),
        ContextElement("Python is great for data science and machine learning", 0.8, 120, "memory"),
        ContextElement("Java is also good", 0.6, 50, "memory"),
        ContextElement("Python is great for data science", 0.7, 100, "memory"),  # Exact duplicate
    ]

    original_tokens = sum(e.token_count for e in elements)

    strategies = [
        ("MERGE_SIMILAR", DeduplicationStrategy.MERGE_SIMILAR),
        ("KEEP_BEST", DeduplicationStrategy.KEEP_BEST),
        ("SUMMARIZE", DeduplicationStrategy.SUMMARIZE),
        ("REMOVE_EXACT", DeduplicationStrategy.REMOVE_EXACT),
    ]

    print(f"\n📊 Original: {len(elements)} elements, {original_tokens} tokens\n")

    for name, strategy in strategies:
        deduplicator = SemanticDeduplicator(
            similarity_method=SimilarityMethod.NGRAM,
            similarity_threshold=0.8
        )

        result = deduplicator.deduplicate(elements, strategy=strategy)

        print(f"  {name}:")
        print(f"    Elements: {len(result.deduplicated_elements)} ({result.elements_removed} removed)")
        print(f"    Tokens: {result.deduplicated_tokens} (saved {result.token_savings}, {(1-result.compression_ratio)*100:.1f}%)")
        print(f"    Compression ratio: {result.compression_ratio:.2f}")
        print()


# ============================================================================
# Demo 3: Token Savings Measurement
# ============================================================================

def demo_3_token_savings():
    """Demo 3: Measure token savings across workload"""
    print("\n" + "="*80)
    print("DEMO 3: Token Savings Measurement")
    print("="*80)

    analyzer = CompressionAnalyzer()
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.8
    )

    # Simulate 20 queries with varying duplication levels
    workload_sizes = [500, 800, 1200, 600, 900, 1100, 700, 1000, 850, 950,
                      600, 1300, 800, 1100, 950, 700, 1000, 900, 800, 1200]

    total_original = 0
    total_compressed = 0

    print("\n🔄 Processing 20 queries...\n")

    for i, size in enumerate(workload_sizes):
        # Create elements (simulate some duplication)
        num_elements = size // 100
        elements = []

        for j in range(num_elements):
            # Some elements are duplicates
            if j < num_elements // 3:
                content = f"Common content about topic {j % 3}"
            else:
                content = f"Unique content {i}-{j}"

            elements.append(ContextElement(
                content=content,
                importance=0.7 + (j * 0.01),
                token_count=100,
                source="memory"
            ))

        # Deduplicate
        result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.MERGE_SIMILAR)

        # Track with analyzer
        analyzer.track_compression(
            original_tokens=result.original_tokens,
            compressed_tokens=result.deduplicated_tokens,
            strategy="merge_similar",
            quality_before=0.90,  # Simulated
            quality_after=0.88,   # Simulated (slight quality drop)
            latency_ms=5.0,
            duplicate_groups_found=len(result.duplicate_groups),
            elements_removed=result.elements_removed,
            elements_merged=result.elements_merged
        )

        total_original += result.original_tokens
        total_compressed += result.deduplicated_tokens

    # Get insights
    insights = analyzer.get_insights()

    print(f"📊 Workload Results:")
    print(f"   Queries processed: {len(workload_sizes)}")
    print(f"   Total original tokens: {total_original:,}")
    print(f"   Total compressed tokens: {total_compressed:,}")
    print(f"   Total tokens saved: {total_original - total_compressed:,}")
    print(f"   Average compression ratio: {insights.avg_compression_ratio:.2f}")
    print(f"   Average savings: {insights.avg_savings_pct:.1f}%")
    print(f"\n{insights.summary()}")


# ============================================================================
# Demo 4: Quality Impact Analysis
# ============================================================================

def demo_4_quality_impact():
    """Demo 4: Analyze quality impact of compression"""
    print("\n" + "="*80)
    print("DEMO 4: Quality Impact Analysis")
    print("="*80)

    analyzer = CompressionAnalyzer()

    # Simulate compressions with varying quality impact
    scenarios = [
        ("Conservative (0.9 threshold)", 0.9, 0.05, 0.90, 0.89),  # Minimal compression, minimal quality drop
        ("Balanced (0.8 threshold)", 0.8, 0.20, 0.90, 0.88),      # Moderate compression, slight quality drop
        ("Aggressive (0.6 threshold)", 0.6, 0.35, 0.90, 0.85),    # High compression, higher quality drop
    ]

    print("\n🔬 Testing different compression aggressiveness...\n")

    for name, threshold, savings_pct, quality_before, quality_after in scenarios:
        # Track 10 compressions for each scenario
        for _ in range(10):
            analyzer.track_compression(
                original_tokens=1000,
                compressed_tokens=int(1000 * (1 - savings_pct)),
                strategy=name.lower().replace(" ", "_"),
                quality_before=quality_before,
                quality_after=quality_after,
                latency_ms=5.0
            )

    # Get insights by strategy
    insights = analyzer.get_insights()

    print("📊 Quality Impact by Strategy:\n")

    for strategy, metrics in insights.strategy_performance.items():
        print(f"  {strategy}:")
        print(f"    Savings: {metrics['savings_pct']:.1f}%")
        print(f"    Quality delta: {metrics['quality_delta']:+.3f}")
        print(f"    Quality preserved: {metrics['quality_preserved_pct']:.1f}%")
        print()

    print(f"\n✅ Best strategy (quality + savings): {insights.best_strategy}")


# ============================================================================
# Demo 5: Compression Insights and Recommendations
# ============================================================================

def demo_5_insights_and_recommendations():
    """Demo 5: Get actionable compression insights"""
    print("\n" + "="*80)
    print("DEMO 5: Compression Insights and Recommendations")
    print("="*80)

    analyzer = CompressionAnalyzer()

    # Scenario 1: Highly effective compression
    print("\n📊 Scenario 1: Highly Effective Compression")
    print("   (30% savings, quality preserved)\n")

    for _ in range(15):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="merge_similar",
            quality_before=0.90,
            quality_after=0.89,
            latency_ms=5.0
        )

    recommendations = analyzer.get_recommendations()
    for rec in recommendations:
        print(f"   • {rec}")

    # Reset for next scenario
    analyzer.reset_statistics()

    # Scenario 2: Compression hurts quality
    print("\n⚠️  Scenario 2: Compression Hurts Quality")
    print("   (30% savings, but >5% quality drop)\n")

    for _ in range(15):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="aggressive",
            quality_before=0.90,
            quality_after=0.80,  # 11% quality drop
            latency_ms=5.0
        )

    recommendations = analyzer.get_recommendations()
    for rec in recommendations:
        print(f"   • {rec}")

    # Reset for next scenario
    analyzer.reset_statistics()

    # Scenario 3: Minimal savings
    print("\n🟡 Scenario 3: Minimal Savings")
    print("   (Only 5% savings)\n")

    for _ in range(15):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=950,
            strategy="conservative",
            quality_before=0.90,
            quality_after=0.89,
            latency_ms=5.0
        )

    recommendations = analyzer.get_recommendations()
    for rec in recommendations:
        print(f"   • {rec}")


# ============================================================================
# Demo 6: Full Integration (Phases 1-4)
# ============================================================================

def demo_6_full_integration():
    """Demo 6: Compression integrated with Phases 1-3"""
    print("\n" + "="*80)
    print("DEMO 6: Full Integration (Phases 1-4)")
    print("="*80)

    print("\n📦 Phase 4 integrates seamlessly with:")
    print("   • Phase 1: LLM generation + quality scoring")
    print("   • Phase 2: Adaptive budgeting")
    print("   • Phase 3: Outcome tracking + adaptive tuning")
    print("   • Phase 4: Context compression")

    print("\n🔄 Compression Flow:")
    print("   1. Pack context (Phases 1-3)")
    print("   2. ✨ Compress context (Phase 4 - NEW)")
    print("   3. Generate with LLM")
    print("   4. Track compression metrics")
    print("   5. Learn from outcomes")

    print("\n⚙️  Configuration:")
    print("""
    packer = LLMContextPacker(
        # Phase 1: LLM integration
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022",

        # Phase 2: Adaptive budgeting
        enable_adaptive_budgeting=True,

        # Phase 3: Outcome tracking
        enable_outcome_tracking=True,
        enable_adaptive_tuning=True,

        # Phase 4: Compression (NEW!)
        enable_compression=True,
        compression_strategy="merge_similar",
        compression_similarity_threshold=0.8,
        compression_similarity_method="ngram",
        compression_min_savings=50
    )
    """)

    print("\n📊 Expected Benefits:")
    print("   • 10-30% token reduction")
    print("   • <2% quality impact (merge_similar strategy)")
    print("   • 20-40% cost savings (from Phase 2 + Phase 4 combined)")
    print("   • Automatic threshold tuning (Phase 3)")
    print("   • Continuous learning and improvement")

    print("\n✅ Phase 4 Complete!")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all Phase 4 demos"""
    print("\n" + "="*80)
    print("  PHASE 4: CONTEXT COMPRESSION - COMPREHENSIVE DEMO")
    print("="*80)
    print("\n  Reducing token waste through intelligent semantic deduplication")
    print("  while preserving response quality.\n")

    # Run all demos
    demo_1_basic_compression()
    demo_2_strategy_comparison()
    demo_3_token_savings()
    demo_4_quality_impact()
    demo_5_insights_and_recommendations()
    demo_6_full_integration()

    print("\n" + "="*80)
    print("  All Phase 4 demos complete!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
