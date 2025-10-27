"""
Basic Observation Example
==========================
Demonstrates how to use darkTrace to observe LLM outputs in real-time.

This example:
1. Creates a SemanticObserver with narrative domain
2. Observes a sequence of tokens
3. Displays semantic velocity, curvature, and dominant dimensions
4. Saves the trajectory for later analysis
"""

import asyncio
from darkTrace import DarkTraceConfig
from darkTrace.observers import SemanticObserver, TrajectoryRecorder, Trajectory


# Sample text from The Odyssey
ODYSSEY_TEXT = """
In the shadow of Mount Olympus, Odysseus stood before the gods, his journey finally complete.
Athena appeared in a shimmer of light, her wise eyes reflecting centuries of knowledge.
'The treasure you bring,' she said, 'is not gold or silver, but wisdom earned through suffering.'
Odysseus bowed his head. 'The trials I faced taught me that the greatest enemy is pride.'
'And the greatest victory,' Athena replied, 'is the return home with humility.'
"""


def main():
    """Run basic observation example."""
    print("=" * 70)
    print("darkTrace - Basic Observation Example")
    print("=" * 70)
    print()

    # 1. Create observer configuration
    print("1. Creating observer with narrative domain...")
    config = DarkTraceConfig.narrative()
    observer = SemanticObserver(config)
    print(f"   ✓ Observer initialized ({config.observer.dimensions}D semantic space)")
    print(f"   ✓ Domain: {config.observer.domain}")
    print(f"   ✓ Strategy: {config.observer.selection_strategy}")
    print()

    # 2. Observe text token by token
    print("2. Observing text tokens...")
    print()

    # Split into words for demonstration
    tokens = ODYSSEY_TEXT.split()

    print("Token-by-Token Analysis:")
    print("-" * 70)
    print(f"{'Token':<20} {'Velocity':<12} {'Curvature':<12} {'Top Dimension'}")
    print("-" * 70)

    for i, token in enumerate(tokens[:20]):  # First 20 tokens
        state = observer.observe(token + " ")

        # Display key metrics
        top_dim = state.dominant_dimensions[0] if state.dominant_dimensions else "N/A"
        print(f"{token:<20} {state.velocity_magnitude:>10.4f}  "
              f"{state.curvature if state.curvature else 0.0:>10.4f}  {top_dim}")

    print("-" * 70)
    print()

    # 3. Display trajectory statistics
    print("3. Trajectory Statistics:")
    print()
    stats = observer.get_statistics()
    print(f"   Total tokens:     {stats['token_count']}")
    print(f"   Trajectory length: {stats['trajectory_length']}")
    print(f"   Avg velocity:     {stats['avg_velocity']:.4f}")
    print(f"   Max velocity:     {stats['max_velocity']:.4f}")
    print(f"   Avg curvature:    {stats['avg_curvature']:.4f}")
    print(f"   Max curvature:    {stats['max_curvature']:.4f}")
    print()

    # 4. Get current semantic state
    print("4. Current Semantic State:")
    print()
    current = observer.get_current_state()
    if current:
        print(f"   Position (first 5 dims): {[f'{p:.3f}' for p in current.position[:5]]}")
        print(f"   Dominant dimensions:")
        for i, dim in enumerate(current.dominant_dimensions[:5], 1):
            score = current.dimension_scores.get(dim, 0.0)
            print(f"      {i}. {dim:<30} ({score:.4f})")
    print()

    # 5. Save trajectory
    print("5. Saving trajectory...")
    recorder = TrajectoryRecorder(storage_dir="./darkTrace_trajectories")

    trajectory = Trajectory(
        trajectory_id="odyssey_narrative_001",
        model_name="example",
        prompt="Odyssey excerpt",
        snapshots=observer.get_trajectory(),
    )

    saved_path = recorder.save(trajectory, overwrite=True)
    print(f"   ✓ Trajectory saved to {saved_path}")
    print()

    # 6. Load and verify
    print("6. Loading and verifying...")
    loaded = recorder.load("odyssey_narrative_001")
    print(f"   ✓ Loaded trajectory: {loaded.total_tokens} tokens")
    print(f"   ✓ Avg velocity: {loaded.avg_velocity:.4f}")
    print(f"   ✓ Tags: {loaded.tags or 'none'}")
    print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  • Try different domains: dialogue, technical, general")
    print("  • Analyze your own LLM outputs")
    print("  • Run trajectory prediction (see trajectory_prediction.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()
