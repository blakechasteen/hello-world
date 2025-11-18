"""
Demo: Episodic → Semantic Memory Transition

Demonstrates the complete semantic transition system:
1. Creating episodic memories (repeated queries)
2. Detecting patterns across episodes
3. Promoting patterns to semantic concepts
4. Querying semantic memory (fast path)
5. Provenance tracking
6. Background transition

This demo shows how HoloLoom learns from repeated interactions,
converting episodic memories into reusable semantic knowledge.

Created: November 2025
"""

import asyncio
from datetime import datetime, timedelta
from typing import List
import time

from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.memory.semantic_transition import (
    SemanticTransitionEngine,
    SemanticTransitionConfig,
    EpisodicPattern,
    SemanticConcept
)


# ============================================================================
# Utilities
# ============================================================================

def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_pattern(pattern: EpisodicPattern):
    """Pretty-print a pattern."""
    print(f"  Pattern ID: {pattern.pattern_id}")
    print(f"  Type:       {pattern.pattern_type}")
    print(f"  Frequency:  {pattern.frequency} episodes")
    print(f"  Similarity: {pattern.similarity_score:.2f}")
    print(f"  Signature:  {pattern.signature}")
    print(f"  Time Span:  {pattern.first_seen.strftime('%H:%M:%S')} → "
          f"{pattern.last_seen.strftime('%H:%M:%S')}")


def print_concept(concept: SemanticConcept):
    """Pretty-print a concept."""
    print(f"  Concept ID:   {concept.concept_id}")
    print(f"  Text:         {concept.concept_text}")
    print(f"  Confidence:   {concept.confidence:.2f}")
    print(f"  Source:       {len(concept.source_episode_ids)} episodes")
    print(f"  Scope:        {concept.scope.value}")
    print(f"  Access Count: {concept.access_count}")
    if concept.last_accessed:
        print(f"  Last Access:  {concept.last_accessed.strftime('%H:%M:%S')}")


# ============================================================================
# Demo 1: Basic Pattern Detection
# ============================================================================

async def demo_basic_pattern_detection():
    """Demo 1: Detect patterns from repeated queries."""
    print_header("Demo 1: Basic Pattern Detection")

    # Initialize
    loom = HoloLoom(config=Config.fast())
    engine = SemanticTransitionEngine(loom)

    print("Creating episodic memories (simulating repeated user queries)...\n")

    # Simulate user asking about Thompson Sampling 5 times
    queries = [
        "What is Thompson Sampling?",
        "Tell me about Thompson Sampling",
        "Explain Thompson Sampling",
        "How does Thompson Sampling work?",
        "What is Thompson Sampling used for?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"  [{i}/5] User: {query}")
        await loom.experience(query, context={'scope': 'SESSION'})
        await asyncio.sleep(0.1)  # Small delay to simulate real usage

    print(f"\n✓ Created {len(queries)} episodic memories\n")

    # Detect patterns
    print("Detecting patterns...\n")
    patterns = await engine.detect_patterns()

    print(f"✓ Detected {len(patterns)} patterns\n")

    # Show detected patterns
    for i, pattern in enumerate(patterns[:3], 1):  # Show first 3
        print(f"Pattern {i}:")
        print_pattern(pattern)
        print()

    # Statistics
    stats = engine.get_statistics()
    print(f"Statistics:")
    print(f"  Total patterns detected: {stats['patterns_detected']}")
    print(f"  Cached patterns:         {stats['detected_patterns']}")


async def demo_pattern_types():
    """Demo 2: Different pattern detection strategies."""
    print_header("Demo 2: Pattern Detection Strategies")

    loom = HoloLoom(config=Config.fast())

    # Configure to enable all strategies
    config = SemanticTransitionConfig(
        pattern_threshold=3,
        similarity_threshold=0.6,
        enable_query_clustering=True,
        enable_entity_cooccurrence=True,
        enable_motif_patterns=True,
        enable_response_similarity=True
    )

    engine = SemanticTransitionEngine(loom, config=config)

    print("Creating diverse episodic memories...\n")

    # 1. Query clustering (similar queries)
    print("  Strategy 1: Query Clustering")
    for i in range(4):
        await loom.experience(
            f"What is Bayesian optimization? (query {i+1})",
            context={'scope': 'SESSION'}
        )
    print("    ✓ Created 4 similar queries about 'Bayesian optimization'\n")

    # 2. Entity co-occurrence (entities appearing together)
    print("  Strategy 2: Entity Co-occurrence")
    for i in range(4):
        mem = await loom.experience(
            f"Thompson Sampling and exploration (query {i+1})",
            context={'scope': 'SESSION'}
        )

        # Add entities to graph
        loom.graph.add_edge(mem.id, 'Thompson_Sampling', type='MENTIONS')
        loom.graph.add_edge(mem.id, 'exploration', type='MENTIONS')

    print("    ✓ Created 4 episodes with entity pair [Thompson_Sampling, exploration]\n")

    # 3. Motif patterns (recurring motifs)
    print("  Strategy 3: Motif Patterns")
    for i in range(3):
        mem = await loom.experience(
            f"Query with reinforcement learning {i+1}",
            context={'scope': 'SESSION'}
        )

        # Add motif to metadata
        if mem.id in loom.graph.nodes():
            loom.graph.nodes[mem.id]['metadata'] = {
                'motifs': ['reinforcement_learning']
            }

    print("    ✓ Created 3 episodes with motif 'reinforcement_learning'\n")

    # 4. Response similarity (similar answers)
    print("  Strategy 4: Response Similarity")
    for i in range(3):
        mem = await loom.experience(
            f"Different query {i+1}",
            context={'scope': 'SESSION'}
        )

        # Add response to metadata
        if mem.id in loom.graph.nodes():
            loom.graph.nodes[mem.id]['metadata'] = {
                'response': 'UCB algorithm balances exploration and exploitation'
            }

    print("    ✓ Created 3 episodes with similar response\n")

    # Detect patterns
    print("Detecting patterns across all strategies...\n")
    patterns = await engine.detect_patterns()

    # Group by pattern type
    from collections import Counter
    pattern_types = Counter(p.pattern_type for p in patterns)

    print(f"✓ Detected {len(patterns)} patterns:\n")

    for pattern_type, count in pattern_types.items():
        print(f"  {pattern_type:25s}: {count} patterns")

    # Show one pattern of each type
    print("\nExample patterns:\n")

    seen_types = set()
    for pattern in patterns:
        if pattern.pattern_type not in seen_types:
            print(f"{pattern.pattern_type}:")
            print_pattern(pattern)
            print()
            seen_types.add(pattern.pattern_type)


# ============================================================================
# Demo 3: Concept Promotion
# ============================================================================

async def demo_concept_promotion():
    """Demo 3: Promote patterns to semantic concepts."""
    print_header("Demo 3: Concept Promotion")

    loom = HoloLoom(config=Config.fast())
    engine = SemanticTransitionEngine(loom)

    print("Creating episodic memories...\n")

    # Create 6 similar queries
    for i in range(6):
        await loom.experience(
            "What is Thompson Sampling?",
            context={'scope': 'SESSION'}
        )

    print("✓ Created 6 episodic memories\n")

    # Detect patterns
    print("Detecting patterns...\n")
    patterns = await engine.detect_patterns()

    if not patterns:
        print("⚠ No patterns detected (try lowering thresholds)")
        return

    print(f"✓ Detected {len(patterns)} patterns\n")

    # Promote first pattern to semantic
    print("Promoting pattern to semantic concept...\n")

    pattern = patterns[0]
    print("Source Pattern:")
    print_pattern(pattern)
    print()

    # Promote
    concept = await engine.promote_to_semantic(pattern)

    print("✓ Created Semantic Concept:\n")
    print_concept(concept)

    # Show provenance
    print("\nProvenance:")
    provenance = engine.get_concept_provenance(concept.concept_id)
    print(f"  This concept was formed from {len(provenance)} episodic memories:")
    for i, ep_id in enumerate(provenance[:5], 1):  # Show first 5
        print(f"    {i}. {ep_id}")

    if len(provenance) > 5:
        print(f"    ... and {len(provenance) - 5} more")

    # Statistics
    stats = engine.get_statistics()
    print(f"\nStatistics:")
    print(f"  Concepts created:        {stats['concepts_created']}")
    print(f"  Episodes transitioned:   {stats['episodes_transitioned']}")


# ============================================================================
# Demo 4: Semantic Query (Fast Path)
# ============================================================================

async def demo_semantic_query():
    """Demo 4: Query semantic memory (faster than episodic)."""
    print_header("Demo 4: Semantic Query vs Episodic Query")

    loom = HoloLoom(config=Config.fast())
    engine = SemanticTransitionEngine(loom)

    print("Setting up: Creating and promoting concepts...\n")

    # Create 5 queries about Thompson Sampling
    for i in range(5):
        await loom.experience(
            "What is Thompson Sampling?",
            context={'scope': 'SESSION'}
        )

    # Detect and promote
    patterns = await engine.detect_patterns()
    if patterns:
        concept = await engine.promote_to_semantic(patterns[0])
        print(f"✓ Created semantic concept: {concept.concept_text[:50]}...\n")

    # Now benchmark semantic query vs episodic query
    print("Benchmarking queries...\n")

    # 1. Semantic query (fast path)
    query = "Thompson Sampling"

    start = time.perf_counter()
    semantic_result = await engine.query_semantic(query)
    semantic_time_ms = (time.perf_counter() - start) * 1000

    # 2. Episodic query (slow path)
    start = time.perf_counter()
    episodic_result = await loom.recall(query)
    episodic_time_ms = (time.perf_counter() - start) * 1000

    # Results
    print(f"Query: '{query}'\n")

    print(f"Semantic Query (fast path):")
    print(f"  Time:   {semantic_time_ms:.2f}ms")
    print(f"  Result: {semantic_result.concept_text if semantic_result else 'None'}")
    print()

    print(f"Episodic Query (slow path):")
    print(f"  Time:   {episodic_time_ms:.2f}ms")
    print(f"  Result: {len(episodic_result)} memories found")
    print()

    if semantic_time_ms > 0:
        speedup = episodic_time_ms / semantic_time_ms
        print(f"Speedup: {speedup:.1f}× faster with semantic memory")

    # Access tracking
    if semantic_result:
        print(f"\nAccess Tracking:")
        print(f"  Concept accessed: {semantic_result.access_count} times")
        print(f"  Last accessed:    {semantic_result.last_accessed.strftime('%H:%M:%S')}")


# ============================================================================
# Demo 5: Provenance Tracking
# ============================================================================

async def demo_provenance_tracking():
    """Demo 5: Track provenance from episodes to concepts."""
    print_header("Demo 5: Provenance Tracking")

    loom = HoloLoom(config=Config.fast())
    engine = SemanticTransitionEngine(loom)

    print("Creating episodic memories with metadata...\n")

    # Create episodes with metadata
    episode_data = []

    for i in range(4):
        mem = await loom.experience(
            f"Thompson Sampling query {i+1}",
            context={
                'scope': 'SESSION',
                'user': 'alice',
                'session_id': 'session_123',
                'query_num': i+1
            }
        )

        episode_data.append({
            'id': mem.id,
            'text': mem.text,
            'timestamp': mem.timestamp
        })

    print(f"✓ Created {len(episode_data)} episodic memories\n")

    # Show episodes
    print("Episodic Memories:")
    for i, ep in enumerate(episode_data, 1):
        print(f"  {i}. ID: {ep['id'][:16]}... | Text: {ep['text'][:40]}...")

    print()

    # Detect and promote
    patterns = await engine.detect_patterns()

    if patterns:
        pattern = patterns[0]
        concept = await engine.promote_to_semantic(pattern)

        print("✓ Promoted to Semantic Concept\n")
        print_concept(concept)

        # Get provenance
        print("\nProvenance Chain:")
        provenance = engine.get_concept_provenance(concept.concept_id)

        print(f"  Concept '{concept.concept_id}' ←")

        for i, ep_id in enumerate(provenance, 1):
            # Find episode data
            ep_data = next((e for e in episode_data if e['id'] == ep_id), None)

            if ep_data:
                print(f"    ↑ Episode {i}: {ep_id[:16]}... ({ep_data['text'][:30]}...)")

        # Check graph edges
        print("\nProvenance Edges in Knowledge Graph:")
        edges = list(engine.kg.graph.edges(concept.concept_id, data=True))

        derived_from_edges = [
            e for e in edges
            if e[2].get('type') == 'DERIVED_FROM'
        ]

        print(f"  Found {len(derived_from_edges)} DERIVED_FROM edges:")

        for src, dst, data in derived_from_edges[:3]:  # Show first 3
            print(f"    {src[:16]}... → {dst[:16]}... (type: {data.get('type')})")


# ============================================================================
# Demo 6: Background Transition
# ============================================================================

async def demo_background_transition():
    """Demo 6: Automatic background transition."""
    print_header("Demo 6: Background Transition")

    loom = HoloLoom(config=Config.fast())

    # Configure background transition
    config = SemanticTransitionConfig(
        enable_background_transition=True,
        background_interval_seconds=5.0,  # Short interval for demo
        pattern_threshold=3,
        similarity_threshold=0.6
    )

    print("Starting background transition (5-second interval)...\n")

    async with SemanticTransitionEngine(loom, config) as engine:
        print("✓ Background loop started\n")

        # Simulate user interactions over time
        print("Simulating user queries over 20 seconds...\n")

        for i in range(20):
            # Create queries (3 different topics rotating)
            topic = ["Thompson Sampling", "UCB algorithm", "Bayesian optimization"][i % 3]

            await loom.experience(
                f"What is {topic}?",
                context={'scope': 'SESSION'}
            )

            print(f"  [{i+1}/20] Query: 'What is {topic}?'")

            # Check stats every 5 queries
            if (i + 1) % 5 == 0:
                stats = engine.get_statistics()
                print(f"    Stats: {stats['concepts_created']} concepts created so far")

            await asyncio.sleep(1)

        # Wait for one more background cycle
        print("\nWaiting for final background cycle...\n")
        await asyncio.sleep(6)

        # Final statistics
        stats = engine.get_statistics()

        print("✓ Background transition complete\n")
        print("Final Statistics:")
        print(f"  Patterns detected:       {stats['patterns_detected']}")
        print(f"  Concepts created:        {stats['concepts_created']}")
        print(f"  Episodes transitioned:   {stats['episodes_transitioned']}")
        print(f"  Semantic concepts:       {stats['semantic_concepts']}")

        # Show created concepts
        print("\nCreated Concepts:")
        for concept_id, concept in engine.semantic_concepts.items():
            print(f"  • {concept.concept_text[:60]}...")

    print("\n✓ Background loop stopped (context manager exit)")


# ============================================================================
# Demo 7: Complete End-to-End Flow
# ============================================================================

async def demo_end_to_end():
    """Demo 7: Complete end-to-end transition flow."""
    print_header("Demo 7: Complete End-to-End Flow")

    loom = HoloLoom(config=Config.fast())
    engine = SemanticTransitionEngine(loom)

    # Step 1: User interactions (episodic memories)
    print("Step 1: User Interactions (Episodic Memory)\n")

    user_queries = [
        "What is Thompson Sampling?",
        "How does Thompson Sampling work?",
        "Explain Thompson Sampling",
        "Tell me about Thompson Sampling",
        "What is Thompson Sampling used for?"
    ]

    for i, query in enumerate(user_queries, 1):
        print(f"  User Query {i}: {query}")
        await loom.experience(query, context={'scope': 'SESSION'})

    print(f"\n✓ Created {len(user_queries)} episodic memories\n")

    # Step 2: Pattern detection
    print("Step 2: Pattern Detection\n")

    patterns = await engine.detect_patterns()

    print(f"✓ Detected {len(patterns)} patterns\n")

    if patterns:
        print("Top Pattern:")
        print_pattern(patterns[0])

    print()

    # Step 3: Concept promotion
    print("Step 3: Concept Promotion (Episodic → Semantic)\n")

    concepts_created = []

    for pattern in patterns[:2]:  # Promote top 2 patterns
        concept = await engine.promote_to_semantic(pattern)
        concepts_created.append(concept)
        print(f"✓ Created concept: {concept.concept_text[:60]}...")

    print()

    # Step 4: Semantic query (fast path)
    print("Step 4: Semantic Query (Fast Path)\n")

    query = "Thompson Sampling"

    result = await engine.query_semantic(query)

    if result:
        print(f"Query: '{query}'")
        print(f"✓ Found: {result.concept_text}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Access count: {result.access_count}")
    else:
        print(f"⚠ No semantic concept found for '{query}'")

    print()

    # Step 5: Provenance tracking
    print("Step 5: Provenance Tracking\n")

    if result:
        provenance = engine.get_concept_provenance(result.concept_id)
        print(f"Concept provenance:")
        print(f"  Formed from {len(provenance)} source episodes")
        print(f"  Episode IDs: {', '.join(p[:12] + '...' for p in provenance[:3])}")

    print()

    # Step 6: Final statistics
    print("Step 6: Final Statistics\n")

    stats = engine.get_statistics()

    print("Engine Statistics:")
    print(f"  Patterns detected:       {stats['patterns_detected']}")
    print(f"  Concepts created:        {stats['concepts_created']}")
    print(f"  Episodes transitioned:   {stats['episodes_transitioned']}")
    print(f"  Semantic queries:        {stats['semantic_queries']}")
    print(f"  Episodic queries:        {stats['episodic_queries']}")

    print("\n✓ End-to-End Flow Complete!")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def main():
    """Run all demos."""
    demos = [
        ("Basic Pattern Detection", demo_basic_pattern_detection),
        ("Pattern Detection Strategies", demo_pattern_types),
        ("Concept Promotion", demo_concept_promotion),
        ("Semantic Query vs Episodic", demo_semantic_query),
        ("Provenance Tracking", demo_provenance_tracking),
        ("Background Transition", demo_background_transition),
        ("Complete End-to-End Flow", demo_end_to_end)
    ]

    print("\n" + "=" * 70)
    print("  Episodic → Semantic Memory Transition Demo")
    print("  Week 6: Memory System Enhancement")
    print("=" * 70)

    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\n  0. Run all demos")
    print()

    try:
        choice = input("Select demo (0-7): ").strip()

        if choice == '0':
            # Run all demos
            for name, demo_func in demos:
                await demo_func()
                await asyncio.sleep(1)

        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            # Run selected demo
            name, demo_func = demos[int(choice) - 1]
            await demo_func()

        else:
            print("Invalid choice. Running Demo 1 by default.")
            await demos[0][1]()

    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
