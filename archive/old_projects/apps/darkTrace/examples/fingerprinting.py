"""
LLM Fingerprinting Example
===========================
Demonstrates how to generate and compare semantic fingerprints.

This example:
1. Generates fingerprints for different "models" (simulated)
2. Compares fingerprints to measure similarity
3. Identifies unique characteristics
4. Detects attractors and patterns
"""

from darkTrace import DarkTraceConfig
from darkTrace.observers import SemanticObserver, Trajectory
from darkTrace.analyzers import FingerprintGenerator, AttractorDetector, PatternRecognizer


# Simulated outputs from different models
MODEL_OUTPUTS = {
    "narrative_model": [
        "In the depths of darkness, the hero found light within.",
        "The journey transformed both traveler and destination.",
        "Wisdom comes not from avoiding trials but embracing them.",
        "The mentor's gift was not knowledge but the quest for it.",
        "Return home brings the greatest treasure: changed perspective.",
    ],

    "technical_model": [
        "The algorithm optimizes for efficiency and correctness.",
        "Implementation requires careful consideration of edge cases.",
        "Performance metrics indicate linear scaling with input size.",
        "Code documentation ensures maintainability over time.",
        "Testing validates assumptions through empirical evidence.",
    ],

    "philosophical_model": [
        "Existence precedes essence in the realm of consciousness.",
        "The self is constructed through dialectical engagement.",
        "Meaning emerges from the interplay of being and nothingness.",
        "Truth is not absolute but contextually contingent.",
        "Freedom entails responsibility for one's choices.",
    ],
}


def generate_model_trajectories(model_name: str, texts: list[str], config: DarkTraceConfig) -> list[Trajectory]:
    """Generate trajectories for a model."""
    trajectories = []

    for i, text in enumerate(texts):
        observer = SemanticObserver(config)

        # Observe tokens
        for token in text.split():
            observer.observe(token + " ")

        # Create trajectory
        trajectory = Trajectory(
            trajectory_id=f"{model_name}_{i:03d}",
            model_name=model_name,
            prompt=f"Sample {i+1}",
            snapshots=observer.get_trajectory(),
        )

        trajectories.append(trajectory)

    return trajectories


def main():
    """Run fingerprinting example."""
    print("=" * 70)
    print("darkTrace - LLM Fingerprinting Example")
    print("=" * 70)
    print()

    # 1. Setup
    print("1. Generating trajectories for each model...")
    config = DarkTraceConfig.fused()  # Use full analysis

    model_trajectories = {}
    for model_name, outputs in MODEL_OUTPUTS.items():
        trajectories = generate_model_trajectories(model_name, outputs, config)
        model_trajectories[model_name] = trajectories
        print(f"   ✓ {model_name}: {len(trajectories)} trajectories")

    print()

    # 2. Generate fingerprints
    print("2. Generating semantic fingerprints...")
    generator = FingerprintGenerator(dimensions=128)

    fingerprints = {}
    for model_name, trajectories in model_trajectories.items():
        fingerprint = generator.generate(trajectories, model_name=model_name)
        fingerprints[model_name] = fingerprint
        print(f"   ✓ {model_name}")
        print(f"      • Tokens: {fingerprint.total_tokens}")
        print(f"      • Avg velocity: {fingerprint.avg_velocity:.4f}")
        print(f"      • Top dimension: {list(fingerprint.dimension_preferences.keys())[0]}")

    print()

    # 3. Compare fingerprints
    print("3. Comparing fingerprints...")
    print()
    print("Pairwise Similarity Matrix:")
    print("-" * 70)

    model_names = list(fingerprints.keys())

    # Header
    print(f"{'Model':<20}", end="")
    for name in model_names:
        print(f"{name[:15]:<18}", end="")
    print()
    print("-" * 70)

    # Similarity matrix
    for model1 in model_names:
        print(f"{model1:<20}", end="")

        for model2 in model_names:
            if model1 == model2:
                print(f"{'1.000':>18}", end="")
            else:
                similarity = generator.compare(fingerprints[model1], fingerprints[model2])
                print(f"{similarity['overall_similarity']:>18.3f}", end="")

        print()

    print("-" * 70)
    print()

    # 4. Detailed comparison
    print("4. Detailed Comparison (narrative vs technical):")
    print()

    comparison = generator.compare(
        fingerprints["narrative_model"],
        fingerprints["technical_model"]
    )

    print("   Similarity Breakdown:")
    for metric, value in comparison.items():
        print(f"      • {metric:<25} {value:>8.3f}")

    print()

    # 5. Detect attractors
    print("5. Detecting semantic attractors...")
    detector = AttractorDetector(n_attractors=3)

    print()
    for model_name, trajectories in model_trajectories.items():
        attractors = detector.detect(trajectories)

        print(f"   {model_name}:")
        if attractors:
            for i, attr in enumerate(attractors[:3], 1):
                dims = ", ".join(attr.dominant_dimensions[:2])
                print(f"      {i}. {attr.attractor_type.value:<12} "
                      f"strength={attr.strength:.3f}  dims=[{dims}]")
        else:
            print("      No attractors detected")

    print()

    # 6. Pattern recognition
    print("6. Recognizing semantic patterns...")
    recognizer = PatternRecognizer(min_pattern_length=3)

    print()
    for model_name, trajectories in model_trajectories.items():
        all_patterns = []

        for traj in trajectories:
            patterns = recognizer.detect(traj)
            all_patterns.extend(patterns)

        # Count pattern types
        from collections import Counter
        pattern_counts = Counter(p.pattern_type.value for p in all_patterns)

        print(f"   {model_name}:")
        if pattern_counts:
            for pattern_type, count in pattern_counts.most_common(3):
                print(f"      • {pattern_type:<15} {count:>3} occurrences")
        else:
            print("      No patterns detected")

    print()

    # 7. Unique characteristics
    print("7. Unique Characteristics:")
    print()

    for model_name, fp in fingerprints.items():
        print(f"   {model_name}:")
        print(f"      Top dimensions: {', '.join(list(fp.dimension_preferences.keys())[:3])}")

        if fp.signature_patterns:
            print(f"      Signature patterns: {fp.signature_patterns[0]}")

        print(f"      Trajectory style: ", end="")
        if fp.avg_velocity < 0.5:
            print("steady, deliberate")
        elif fp.avg_velocity < 1.5:
            print("dynamic, flowing")
        else:
            print("rapid, exploratory")

        print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Key insights:")
    print("  • Each model has distinct semantic signature")
    print("  • Fingerprints enable model identification")
    print("  • Attractors reveal preferred semantic regions")
    print("  • Patterns show characteristic behaviors")
    print()
    print("Next steps:")
    print("  • Fingerprint real LLM outputs")
    print("  • Build model detection system")
    print("  • Track model drift over time")
    print("=" * 70)


if __name__ == "__main__":
    main()
