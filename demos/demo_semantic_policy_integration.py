#!/usr/bin/env python3
"""
Demo: Semantic State -> Policy Integration (Phase 8)

This demo showcases how HoloLoom's semantic calculus influences tool selection:
1. SemanticState: 244D -> 8D feature vector for policy network
2. Topic shift detection: Automatic branching when conversation drifts
3. SemanticToolSelector: Category-aware tool suggestions
4. SemanticAwareBandit: Thompson Sampling with semantic prior adjustments

Architecture Flow:
    Query -> 244D embedding -> SemanticState -> 8D features -> NeuralCore -> Tool selection
                                    |
                                    v
                            SemanticAwareBandit
                            (adjusts priors based on dominant category)

Author: HoloLoom Team
Date: December 2025
"""

import numpy as np
import asyncio
from dataclasses import dataclass
from typing import List, Dict

# Phase 8 imports
from HoloLoom.semantic_calculus import (
    PolicySemanticState as SemanticState,
    SemanticToolSelector,
    SemanticAwareBandit,
    SEMANTIC_CATEGORIES,
    TOPIC_SHIFT_THRESHOLD,
    SIGNIFICANT_SHIFT_THRESHOLD,
)


def create_mock_embedding(category_bias: str = "Core", dim: int = 244) -> np.ndarray:
    """
    Create a mock 244D embedding biased toward a specific semantic category.

    The 244D semantic space is organized into 16 categories (from SEMANTIC_CATEGORIES).
    Categories are mapped as defined in semantic_state.py fallback.
    """
    np.random.seed(42)
    embedding = np.random.randn(dim) * 0.5

    # Use the actual SEMANTIC_CATEGORIES mapping
    if category_bias in SEMANTIC_CATEGORIES:
        indices = SEMANTIC_CATEGORIES[category_bias]
        for idx in indices:
            if idx < dim:
                embedding[idx] += 2.0  # Strong positive bias

    return embedding


def demo_semantic_state_basics():
    """Demo 1: SemanticState creation and feature extraction."""
    print("=" * 70)
    print("DEMO 1: SemanticState Basics")
    print("=" * 70)
    print()

    # Create embedding biased toward "Creative" category
    embedding = create_mock_embedding("Creative")

    # Create SemanticState from embedding
    state = SemanticState.from_embedding(embedding)

    print(f"Input: 244D embedding with Creative bias")
    print(f"Output: {state}")
    print()
    print(f"Key metrics:")
    print(f"  - Dominant category: {state.dominant_category}")
    print(f"  - Momentum: {state.momentum:.3f}")
    print(f"  - Complexity: {state.complexity:.3f}")
    print(f"  - Shift magnitude: {state.shift_magnitude:.3f}")
    print()

    # Show 8D feature vector for policy network
    features = state.to_feature_vector()
    print(f"8D Feature Vector for NeuralCore:")
    print(f"  [momentum, complexity, top5_dims[0-4], velocity_mag]")
    print(f"  {features}")
    print()

    # Show category scores
    print(f"Top 5 Category Scores:")
    sorted_cats = sorted(state.category_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for cat, score in sorted_cats:
        print(f"  {cat}: {score:.3f}")
    print()


def demo_topic_shift_detection():
    """Demo 2: Topic shift detection across conversation turns."""
    print("=" * 70)
    print("DEMO 2: Topic Shift Detection")
    print("=" * 70)
    print()

    print(f"Thresholds:")
    print(f"  - Minor shift: velocity > {TOPIC_SHIFT_THRESHOLD}")
    print(f"  - Major shift: velocity > {SIGNIFICANT_SHIFT_THRESHOLD}")
    print()

    # Simulate conversation with topic changes
    conversation = [
        ("Creative", "Let's write a poem about nature"),
        ("Creative", "Add more metaphors to the poem"),
        ("Cognitive", "Now analyze the poem's structure"),
        ("Temporal", "How has poetry evolved over time?"),
        ("Emotional", "Tell me how this makes you feel"),
    ]

    print("Simulated Conversation:")
    print("-" * 50)

    prev_state = None
    for category, query in conversation:
        embedding = create_mock_embedding(category)
        state = SemanticState.from_embedding(embedding, prev_state)

        shift_type = "no shift"
        if state.topic_shift_detected:
            shift_type = "MAJOR SHIFT" if state.shift_magnitude > SIGNIFICANT_SHIFT_THRESHOLD else "minor shift"

        print(f"Query: \"{query}\"")
        print(f"  Category: {state.dominant_category}, Velocity: {state.shift_magnitude:.3f}, {shift_type}")

        prev_state = state

    print()


def demo_semantic_tool_selector():
    """Demo 3: SemanticToolSelector category-aware suggestions."""
    print("=" * 70)
    print("DEMO 3: SemanticToolSelector")
    print("=" * 70)
    print()

    selector = SemanticToolSelector()

    # Test different scenarios
    scenarios = [
        ("Creative", 0.8, 0.3, 0.2, "High momentum, low complexity, Creative focus"),
        ("Cognitive", 0.2, 0.9, 0.1, "Low momentum, high complexity (needs simplification)"),
        ("Emotional", 0.9, 0.5, 0.6, "Topic shift detected (branching needed)"),
    ]

    for category, momentum, complexity, velocity, description in scenarios:
        embedding = create_mock_embedding(category)
        state = SemanticState.from_embedding(embedding)
        # Override for demo
        state.momentum = momentum
        state.complexity = complexity
        state.shift_magnitude = velocity
        state.topic_shift_detected = velocity > TOPIC_SHIFT_THRESHOLD

        suggested = selector.suggest_tool(state)
        should_branch = selector.should_branch_thread(state)

        print(f"Scenario: {description}")
        print(f"  Suggested tool: {suggested}")
        print(f"  Should branch thread: {should_branch}")
        print()


def demo_semantic_aware_bandit():
    """Demo 4: SemanticAwareBandit Thompson Sampling with semantic adjustments."""
    print("=" * 70)
    print("DEMO 4: SemanticAwareBandit")
    print("=" * 70)
    print()

    bandit = SemanticAwareBandit(
        tools=["answer", "search", "notion_write", "calc"],
        semantic_weight=0.3
    )

    print("Category -> Tool Affinities (boost factors):")
    print("-" * 50)
    categories_to_show = ["Creative", "Cognitive", "Emotional", "Narrative"]
    for cat in categories_to_show:
        affinities = SemanticAwareBandit.CATEGORY_TOOL_AFFINITIES[cat]
        print(f"  {cat}:")
        for tool, boost in sorted(affinities.items(), key=lambda x: x[1], reverse=True):
            print(f"    {tool}: {boost:.1f}x")
    print()

    # Demo prior adjustment
    print("Prior Adjustment Example:")
    print("-" * 50)

    # Create Creative-biased state
    creative_embedding = create_mock_embedding("Creative")
    creative_state = SemanticState.from_embedding(creative_embedding)
    creative_state.dominant_category = "Creative"

    adjusted = bandit.adjust_priors(creative_state)
    print(f"With Creative dominant category (semantic_weight={bandit.semantic_weight}):")
    for tool, (alpha, beta) in adjusted.items():
        expected = alpha / (alpha + beta)
        print(f"  {tool}: alpha={alpha:.2f}, beta={beta:.2f}, E[X]={expected:.3f}")
    print()

    # Demo sampling distribution
    print("Tool Selection Probabilities:")
    print("-" * 50)

    categories_test = ["Creative", "Cognitive", "Emotional"]
    for cat in categories_test:
        embedding = create_mock_embedding(cat)
        state = SemanticState.from_embedding(embedding)
        state.dominant_category = cat

        rewards = bandit.get_expected_rewards(state)
        print(f"  {cat} context:")
        for tool, reward in sorted(rewards.items(), key=lambda x: x[1], reverse=True):
            bar = "=" * int(reward * 30)
            print(f"    {tool:15s} [{bar:30s}] {reward:.3f}")
        print()


def demo_learning_over_time():
    """Demo 5: Bandit learning from outcomes."""
    print("=" * 70)
    print("DEMO 5: Learning Over Time")
    print("=" * 70)
    print()

    bandit = SemanticAwareBandit(semantic_weight=0.3)

    # Simulate a learning session
    outcomes = [
        ("answer", True, 0.9),
        ("answer", True, 0.85),
        ("search", False, 0.4),
        ("search", True, 0.7),
        ("notion_write", True, 0.95),
        ("calc", False, 0.3),
    ]

    print("Simulated Learning Session:")
    print("-" * 50)

    for tool, success, confidence in outcomes:
        bandit.update(tool, success, confidence)
        outcome_str = "SUCCESS" if success else "FAILURE"
        print(f"  {tool}: {outcome_str} (conf={confidence:.2f})")

    print()
    print("Updated Statistics:")
    stats = bandit.get_stats()
    for tool in bandit.tools:
        alpha = stats["alphas"][tool]
        beta = stats["betas"][tool]
        expected = alpha / (alpha + beta)
        print(f"  {tool}: alpha={alpha:.2f}, beta={beta:.2f}, E[X]={expected:.3f}")
    print()

    # Show how semantic context affects sampling after learning
    print("Sampling Comparison (after learning):")
    print("-" * 50)

    np.random.seed(123)

    # Sample without semantic context
    samples_no_ctx = {}
    for _ in range(100):
        tool = bandit.sample(semantic_state=None)
        samples_no_ctx[tool] = samples_no_ctx.get(tool, 0) + 1

    # Sample with Creative context
    creative_state = SemanticState.from_embedding(create_mock_embedding("Creative"))
    creative_state.dominant_category = "Creative"

    samples_creative = {}
    for _ in range(100):
        tool = bandit.sample(semantic_state=creative_state)
        samples_creative[tool] = samples_creative.get(tool, 0) + 1

    print("Without semantic context (100 samples):")
    for tool in bandit.tools:
        count = samples_no_ctx.get(tool, 0)
        bar = "*" * (count // 2)
        print(f"  {tool:15s} [{bar:50s}] {count}")

    print()
    print("With Creative context (100 samples):")
    for tool in bandit.tools:
        count = samples_creative.get(tool, 0)
        bar = "*" * (count // 2)
        print(f"  {tool:15s} [{bar:50s}] {count}")
    print()


def demo_full_pipeline():
    """Demo 6: Full integration showing semantic state -> tool selection."""
    print("=" * 70)
    print("DEMO 6: Full Pipeline Integration")
    print("=" * 70)
    print()

    print("Pipeline: Query -> 244D -> SemanticState -> 8D -> Tool Selection")
    print()

    # Initialize components
    selector = SemanticToolSelector()
    bandit = SemanticAwareBandit(semantic_weight=0.3)

    # Simulate multi-turn conversation
    queries = [
        ("Creative", "Write me a haiku about the ocean"),
        ("Creative", "Make it more melancholic"),
        ("Cognitive", "What's the syllable structure of haikus?"),
        ("Narrative", "Tell me the history of haiku poetry"),
        ("Emotional", "Does this haiku evoke sadness?"),
    ]

    print("Multi-Turn Conversation Simulation:")
    print("=" * 60)

    prev_state = None
    for category, query in queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 60)

        # Step 1: Create embedding (would be from actual embedder in production)
        embedding = create_mock_embedding(category, dim=244)

        # Step 2: Build SemanticState
        state = SemanticState.from_embedding(embedding, prev_state)

        # Step 3: Get 8D feature vector (would feed into NeuralCore)
        features = state.to_feature_vector()

        # Step 4: Tool selection via selector + bandit
        suggested_tool = selector.suggest_tool(state)
        bandit_tool = bandit.sample(semantic_state=state)
        should_branch = selector.should_branch_thread(state)

        # Display results
        print(f"  Semantic State:")
        print(f"    Dominant: {state.dominant_category}")
        print(f"    Momentum: {state.momentum:.3f}")
        print(f"    Velocity: {state.shift_magnitude:.3f}")
        print(f"    Topic Shifted: {state.topic_shift_detected}")

        print(f"  8D Features: [{', '.join(f'{x:.2f}' for x in features[:4])}...]")

        print(f"  Tool Selection:")
        print(f"    Selector suggests: {suggested_tool}")
        print(f"    Bandit samples: {bandit_tool}")
        print(f"    Should branch: {should_branch}")

        # Simulate success/failure and update bandit
        success = np.random.random() > 0.3  # 70% success rate
        confidence = 0.7 + np.random.random() * 0.25
        bandit.update(bandit_tool, success, confidence)

        prev_state = state

    print()
    print("=" * 60)
    print("Final Bandit State (after learning):")
    stats = bandit.get_stats()
    for tool in bandit.tools:
        alpha = stats["alphas"][tool]
        beta = stats["betas"][tool]
        expected = alpha / (alpha + beta)
        print(f"  {tool}: E[X]={expected:.3f}")
    print()


def main():
    """Run all Phase 8 demos."""
    print()
    print("*" * 70)
    print("*  Phase 8: Semantic State -> Policy Integration Demo")
    print("*  HoloLoom - Smart Multithreaded Chat with Topic Branching")
    print("*" * 70)
    print()

    cats = list(SEMANTIC_CATEGORIES.keys())
    print(f"16 Semantic Categories: {', '.join(cats[:8])}...")
    print(f"                        {', '.join(cats[8:])}")
    print()

    # Run demos
    demo_semantic_state_basics()
    demo_topic_shift_detection()
    demo_semantic_tool_selector()
    demo_semantic_aware_bandit()
    demo_learning_over_time()
    demo_full_pipeline()

    print("=" * 70)
    print("Phase 8 Demo Complete!")
    print()
    print("Key Takeaways:")
    print("  1. SemanticState bridges 244D calculus -> 8D policy features")
    print("  2. Topic shifts are detected via velocity magnitude thresholds")
    print("  3. SemanticToolSelector provides category-aware suggestions")
    print("  4. SemanticAwareBandit adjusts Thompson priors based on context")
    print("  5. The system learns from outcomes and adapts over time")
    print("=" * 70)


if __name__ == "__main__":
    main()
