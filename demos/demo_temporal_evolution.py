"""
Temporal Evolution Demo
=======================

Demonstrates temporal tracking of concept understanding evolution.

Features demonstrated:
- State transitions (UNKNOWN → MASTERY)
- Milestone detection (first learned, mastery achieved)
- Temporal queries ("What did I know on date X?")
- Confidence trajectory visualization
- Evolution summaries
- Decay tracking (dormancy and forgetting)

Author: HoloLoom Demo Team
Date: 2025-11-18 (Week 7)
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom import HoloLoom
from HoloLoom.memory.temporal_evolution import (
    TemporalEvolutionTracker,
    TemporalEvolutionConfig,
    UnderstandingState
)


# ============================================================================
# Simulated Learning Journey
# ============================================================================

async def simulate_learning_journey():
    """
    Simulate learning about Reinforcement Learning over 4 weeks.

    Week 1: Introduction (UNKNOWN → INTRODUCED)
    Week 2: Active Learning (INTRODUCED → LEARNING)
    Week 3: Regular Use (LEARNING → FAMILIAR)
    Week 4: Deep Understanding (FAMILIAR → MASTERY)
    """
    print("\n" + "=" * 70)
    print("DEMO: Simulated Learning Journey")
    print("=" * 70)

    # Initialize
    loom = HoloLoom()
    config = TemporalEvolutionConfig(
        learning_threshold=3,
        familiar_threshold=10,
        mastery_confidence=0.85
    )

    async with TemporalEvolutionTracker(loom, config) as tracker:
        concept = "Reinforcement Learning"

        # Week 1: Introduction
        print("\n📚 Week 1: Introduction")
        print("-" * 70)

        week1_queries = [
            "What is Reinforcement Learning?",
            "How is RL different from supervised learning?"
        ]

        base_time = datetime.now() - timedelta(days=28)  # 4 weeks ago

        for i, query in enumerate(week1_queries):
            timestamp = base_time + timedelta(days=i)
            await tracker.track_interaction(
                query=query,
                entities=[concept],
                confidence=0.5 + (i * 0.05),
                timestamp=timestamp
            )
            print(f"  Day {i+1}: {query} (conf: {0.5 + i*0.05:.2f})")

        history = await tracker.get_concept_history(concept)
        print(f"\n  ✓ State: {history.current_state.value.upper()}")
        print(f"    Interactions: {history.total_interactions}")

        # Week 2: Active Learning
        print("\n🔬 Week 2: Active Learning")
        print("-" * 70)

        week2_queries = [
            "Explain Q-learning algorithm",
            "What is the Bellman equation?",
            "How does policy gradient work?",
            "Difference between on-policy and off-policy",
            "What is exploration-exploitation tradeoff?"
        ]

        for i, query in enumerate(week2_queries):
            timestamp = base_time + timedelta(days=7+i)
            await tracker.track_interaction(
                query=query,
                entities=[concept],
                confidence=0.60 + (i * 0.03),
                timestamp=timestamp
            )
            print(f"  Day {7+i+1}: {query} (conf: {0.60 + i*0.03:.2f})")

        history = await tracker.get_concept_history(concept)
        print(f"\n  ✓ State: {history.current_state.value.upper()}")
        print(f"    Interactions: {history.total_interactions}")

        # Week 3: Regular Use
        print("\n💪 Week 3: Regular Use")
        print("-" * 70)

        week3_queries = [
            "Implement simple Q-learning agent",
            "Compare SARSA vs Q-learning",
            "How to tune learning rate in RL?",
            "Debugging RL training instability",
            "Implement epsilon-greedy exploration",
            "Understanding value function approximation"
        ]

        for i, query in enumerate(week3_queries):
            timestamp = base_time + timedelta(days=14+i)
            await tracker.track_interaction(
                query=query,
                entities=[concept],
                confidence=0.70 + (i * 0.02),
                timestamp=timestamp
            )
            print(f"  Day {14+i+1}: {query} (conf: {0.70 + i*0.02:.2f})")

        history = await tracker.get_concept_history(concept)
        print(f"\n  ✓ State: {history.current_state.value.upper()}")
        print(f"    Interactions: {history.total_interactions}")

        # Week 4: Deep Understanding
        print("\n🎓 Week 4: Deep Understanding")
        print("-" * 70)

        week4_queries = [
            "Compare PPO vs A3C algorithms",
            "Implement policy gradient from scratch",
            "Advanced RL: TRPO vs PPO",
            "Handling sparse rewards in RL",
            "Multi-agent RL coordination",
            "RL in continuous action spaces"
        ]

        for i, query in enumerate(week4_queries):
            timestamp = base_time + timedelta(days=21+i)
            await tracker.track_interaction(
                query=query,
                entities=[concept],
                confidence=0.85 + (i * 0.01),
                timestamp=timestamp
            )
            print(f"  Day {21+i+1}: {query} (conf: {0.85 + i*0.01:.2f})")

        history = await tracker.get_concept_history(concept)
        print(f"\n  ✓ State: {history.current_state.value.upper()}")
        print(f"    Interactions: {history.total_interactions}")

        # Show state progression
        print("\n📈 State Progression:")
        print("-" * 70)
        for i, transition in enumerate(history.state_transitions):
            print(f"  {i+1}. {transition.timestamp.strftime('%Y-%m-%d')}: "
                  f"{transition.old_state.value} → {transition.new_state.value}")
            print(f"     Trigger: {transition.trigger}")

        # Show milestones
        print("\n🏆 Milestones:")
        print("-" * 70)
        for milestone in history.milestones:
            print(f"  {milestone.type.upper()}: {milestone.description}")
            print(f"    When: {milestone.timestamp.strftime('%Y-%m-%d')}")
            print(f"    Confidence: {milestone.confidence:.2f}")
            print()

        # Visualize trajectory
        print("\n📊 Understanding Trajectory:")
        print("-" * 70)
        viz = tracker.visualize_trajectory(concept, output_format='ascii')
        print(viz)


# ============================================================================
# Temporal Queries Demo
# ============================================================================

async def demonstrate_temporal_queries():
    """Demonstrate point-in-time understanding queries."""
    print("\n" + "=" * 70)
    print("DEMO: Temporal Queries")
    print("=" * 70)

    loom = HoloLoom()
    async with TemporalEvolutionTracker(loom) as tracker:
        concept = "Neural Networks"

        # Simulate learning over 60 days
        print(f"\n📝 Simulating 60 days of learning about {concept}...")

        base_time = datetime.now() - timedelta(days=60)

        for day in range(60):
            # Increasing confidence over time
            confidence = 0.4 + (day / 60) * 0.5  # 0.4 → 0.9

            await tracker.track_interaction(
                query=f"Question about neural networks (day {day})",
                entities=[concept],
                confidence=confidence,
                timestamp=base_time + timedelta(days=day)
            )

        print("  ✓ Completed\n")

        # Query at different points in time
        print("🔍 Point-in-Time Queries:")
        print("-" * 70)

        query_dates = [
            ("Day 10", base_time + timedelta(days=10)),
            ("Day 20", base_time + timedelta(days=20)),
            ("Day 30", base_time + timedelta(days=30)),
            ("Day 45", base_time + timedelta(days=45)),
            ("Day 60", base_time + timedelta(days=60))
        ]

        for label, timestamp in query_dates:
            snapshot = await tracker.query_at_time(concept, timestamp)

            print(f"\n{label} ({timestamp.strftime('%Y-%m-%d')}):")
            print(f"  State: {snapshot.state.value.upper()}")
            print(f"  Confidence: {snapshot.confidence:.2%}")
            print(f"  Query Count: {snapshot.query_count}")

        # Show "what changed" analysis
        print("\n\n📉 Understanding Evolution Analysis:")
        print("-" * 70)

        snap_10 = await tracker.query_at_time(concept, base_time + timedelta(days=10))
        snap_30 = await tracker.query_at_time(concept, base_time + timedelta(days=30))
        snap_60 = await tracker.query_at_time(concept, base_time + timedelta(days=60))

        print(f"Day 10 → Day 30:")
        print(f"  State: {snap_10.state.value} → {snap_30.state.value}")
        print(f"  Confidence: {snap_10.confidence:.2f} → {snap_30.confidence:.2f} "
              f"(+{snap_30.confidence - snap_10.confidence:.2f})")

        print(f"\nDay 30 → Day 60:")
        print(f"  State: {snap_30.state.value} → {snap_60.state.value}")
        print(f"  Confidence: {snap_30.confidence:.2f} → {snap_60.confidence:.2f} "
              f"(+{snap_60.confidence - snap_10.confidence:.2f})")


# ============================================================================
# Multi-Concept Tracking
# ============================================================================

async def demonstrate_multi_concept():
    """Track multiple concepts simultaneously."""
    print("\n" + "=" * 70)
    print("DEMO: Multi-Concept Tracking")
    print("=" * 70)

    loom = HoloLoom()
    async with TemporalEvolutionTracker(loom) as tracker:

        # Define concepts with different learning patterns
        concepts = {
            "Thompson Sampling": {
                "interactions": 20,
                "confidence_range": (0.6, 0.95)
            },
            "UCB Algorithm": {
                "interactions": 15,
                "confidence_range": (0.65, 0.88)
            },
            "Epsilon-Greedy": {
                "interactions": 10,
                "confidence_range": (0.7, 0.85)
            },
            "Policy Gradient": {
                "interactions": 5,
                "confidence_range": (0.5, 0.7)
            }
        }

        print("\n📚 Tracking multiple concepts...")
        print()

        base_time = datetime.now() - timedelta(days=30)

        for concept, params in concepts.items():
            conf_min, conf_max = params['confidence_range']
            n_interactions = params['interactions']

            for i in range(n_interactions):
                # Linear confidence growth
                confidence = conf_min + (i / n_interactions) * (conf_max - conf_min)

                await tracker.track_interaction(
                    query=f"Question about {concept}",
                    entities=[concept],
                    confidence=confidence,
                    timestamp=base_time + timedelta(days=i)
                )

            print(f"  ✓ {concept}: {n_interactions} interactions")

        # Compare understanding across concepts
        print("\n\n📊 Understanding Comparison:")
        print("-" * 70)
        print(f"{'Concept':<25} {'State':<12} {'Interactions':<14} {'Avg Confidence':<15}")
        print("-" * 70)

        comparison = []
        for concept in concepts:
            history = await tracker.get_concept_history(concept)
            avg_conf = tracker._get_average_confidence(history)

            comparison.append({
                'concept': concept,
                'state': history.current_state,
                'interactions': history.total_interactions,
                'avg_conf': avg_conf
            })

        # Sort by state level
        state_order = {
            UnderstandingState.MASTERY: 5,
            UnderstandingState.FAMILIAR: 4,
            UnderstandingState.LEARNING: 3,
            UnderstandingState.INTRODUCED: 2,
            UnderstandingState.UNKNOWN: 1
        }

        comparison.sort(
            key=lambda x: (state_order.get(x['state'], 0), x['avg_conf']),
            reverse=True
        )

        for item in comparison:
            print(f"{item['concept']:<25} "
                  f"{item['state'].value:<12} "
                  f"{item['interactions']:<14} "
                  f"{item['avg_conf']:.2f}")


# ============================================================================
# Evolution Summary Demo
# ============================================================================

async def demonstrate_evolution_summary():
    """Demonstrate evolution summary over time periods."""
    print("\n" + "=" * 70)
    print("DEMO: Evolution Summary")
    print("=" * 70)

    loom = HoloLoom()
    async with TemporalEvolutionTracker(loom) as tracker:

        # Simulate diverse learning activity
        print("\n📚 Simulating diverse learning activity...")

        base_time = datetime.now()

        # New concepts (learned in last 30 days)
        new_concepts = ["Concept A", "Concept B", "Concept C"]
        for concept in new_concepts:
            await tracker.track_interaction(
                query=f"Learn about {concept}",
                entities=[concept],
                confidence=0.6,
                timestamp=base_time - timedelta(days=15)
            )

        # Concepts reaching mastery
        mastery_concepts = ["Deep Learning", "Transformers"]
        for concept in mastery_concepts:
            # Many high-confidence interactions
            for i in range(20):
                await tracker.track_interaction(
                    query=f"Advanced {concept}",
                    entities=[concept],
                    confidence=0.9,
                    timestamp=base_time - timedelta(days=25-i)
                )

        # Active concepts
        active_concepts = ["Python", "Git", "Docker"]
        for concept in active_concepts:
            for i in range(8):
                await tracker.track_interaction(
                    query=f"Using {concept}",
                    entities=[concept],
                    confidence=0.8,
                    timestamp=base_time - timedelta(days=10-i)
                )

        # Dormant concept
        await tracker.track_interaction(
            query="Old topic",
            entities=["Old Concept"],
            confidence=0.7,
            timestamp=base_time - timedelta(days=45)
        )

        print("  ✓ Completed\n")

        # Get evolution summary
        summary = await tracker.get_evolution_summary(days=30)

        print("\n📈 Evolution Summary (Last 30 Days):")
        print("-" * 70)

        print(f"\n🆕 New Concepts: {summary['new_concepts']['count']}")
        for concept in summary['new_concepts']['concepts'][:5]:
            print(f"   - {concept}")

        print(f"\n🎓 Concepts Mastered: {summary['concepts_mastered']['count']}")
        for concept in summary['concepts_mastered']['concepts'][:5]:
            print(f"   - {concept}")

        print(f"\n⚡ Active Concepts: {summary['active_concepts']['count']}")
        for concept in summary['active_concepts']['concepts'][:5]:
            print(f"   - {concept}")

        print(f"\n😴 Forgotten Concepts: {summary['forgotten_concepts']['count']}")
        for concept in summary['forgotten_concepts']['concepts'][:5]:
            print(f"   - {concept}")

        print(f"\n📊 Total Concepts Tracked: {summary['total_concepts']}")


# ============================================================================
# Decay Monitoring Demo
# ============================================================================

async def demonstrate_decay_monitoring():
    """Demonstrate monitoring of concept decay."""
    print("\n" + "=" * 70)
    print("DEMO: Decay Monitoring")
    print("=" * 70)

    loom = HoloLoom()
    config = TemporalEvolutionConfig(dormant_days=30)

    async with TemporalEvolutionTracker(loom, config) as tracker:

        print("\n📚 Simulating concepts with different access patterns...")

        base_time = datetime.now()

        # Recently active (no decay)
        for i in range(5):
            await tracker.track_interaction(
                query="Recent query",
                entities=["Recent Concept"],
                confidence=0.8,
                timestamp=base_time - timedelta(days=i)
            )

        # Slightly inactive (20 days)
        for i in range(5):
            await tracker.track_interaction(
                query="Slightly old",
                entities=["At Risk Concept"],
                confidence=0.75,
                timestamp=base_time - timedelta(days=25-i)
            )

        # Dormant (40 days)
        for i in range(10):
            await tracker.track_interaction(
                query="Old query",
                entities=["Dormant Concept"],
                confidence=0.8,
                timestamp=base_time - timedelta(days=50-i)
            )

        # Very old, low confidence (forgotten)
        for i in range(3):
            await tracker.track_interaction(
                query="Very old",
                entities=["Forgotten Concept"],
                confidence=0.3,
                timestamp=base_time - timedelta(days=100-i)
            )

        print("  ✓ Completed\n")

        # Analyze decay
        print("\n🔍 Decay Analysis:")
        print("-" * 70)

        concepts_by_state = {
            'active': [],
            'at_risk': [],
            'dormant': [],
            'forgotten': []
        }

        for concept, history in tracker.concept_histories.items():
            if history.last_accessed:
                days_inactive = (base_time - history.last_accessed).days

                if history.current_state == UnderstandingState.FORGOTTEN:
                    concepts_by_state['forgotten'].append((concept, days_inactive))
                elif history.current_state == UnderstandingState.DORMANT:
                    concepts_by_state['dormant'].append((concept, days_inactive))
                elif days_inactive > 20:
                    concepts_by_state['at_risk'].append((concept, days_inactive))
                else:
                    concepts_by_state['active'].append((concept, days_inactive))

        print(f"\n✅ Active Concepts ({len(concepts_by_state['active'])}):")
        for concept, days in concepts_by_state['active']:
            history = await tracker.get_concept_history(concept)
            print(f"   {concept}: Last accessed {days} days ago")
            print(f"     State: {history.current_state.value}")

        print(f"\n⚠️  At Risk Concepts ({len(concepts_by_state['at_risk'])}):")
        for concept, days in concepts_by_state['at_risk']:
            history = await tracker.get_concept_history(concept)
            print(f"   {concept}: {days} days inactive")
            print(f"     State: {history.current_state.value}")

        print(f"\n😴 Dormant Concepts ({len(concepts_by_state['dormant'])}):")
        for concept, days in concepts_by_state['dormant']:
            print(f"   {concept}: {days} days inactive")

        print(f"\n🚫 Forgotten Concepts ({len(concepts_by_state['forgotten'])}):")
        for concept, days in concepts_by_state['forgotten']:
            print(f"   {concept}: {days} days inactive")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TEMPORAL EVOLUTION TRACKING DEMO" + " " * 21 + "║")
    print("║" + " " * 20 + "Week 7: HoloLoom Memory System" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")

    demos = [
        ("Learning Journey", simulate_learning_journey),
        ("Temporal Queries", demonstrate_temporal_queries),
        ("Multi-Concept Tracking", demonstrate_multi_concept),
        ("Evolution Summary", demonstrate_evolution_summary),
        ("Decay Monitoring", demonstrate_decay_monitoring)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            await demo_func()

            # Pause between demos
            if i < len(demos):
                print("\n" + "-" * 70)
                print("Press Enter to continue to next demo...")
                input()

        except Exception as e:
            print(f"\n❌ Error in {name} demo: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Track understanding evolution from UNKNOWN to MASTERY")
    print("  2. Query understanding at any point in time")
    print("  3. Detect milestones (first learned, mastery achieved)")
    print("  4. Monitor multiple concepts simultaneously")
    print("  5. Track decay (dormancy and forgetting)")
    print("\nSee TEMPORAL_EVOLUTION_README.md for complete documentation.")
    print()


if __name__ == '__main__':
    asyncio.run(main())
