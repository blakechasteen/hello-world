"""
Trajectory Prediction Example
==============================
Demonstrates how to predict future semantic states using darkTrace analyzers.

This example:
1. Loads multiple recorded trajectories
2. Trains a trajectory predictor
3. Predicts future semantic states
4. Evaluates prediction accuracy
"""

from darkTrace import DarkTraceConfig
from darkTrace.observers import SemanticObserver, TrajectoryRecorder, Trajectory
from darkTrace.analyzers import TrajectoryPredictor


# Sample training texts (different narrative passages)
TRAINING_TEXTS = [
    """
    The hero stood at the threshold, fear and courage warring within.
    Ahead lay darkness, behind lay safety. The choice was clear yet impossible.
    With trembling hands, he grasped the sword and stepped forward.
    """,

    """
    In the depths of the labyrinth, she faced her greatest fear.
    The monster was not external, but the shadow of her own doubts.
    Only by accepting this truth could she emerge transformed.
    """,

    """
    The mentor appeared when hope was lost, offering wisdom but no answers.
    'The treasure you seek,' he said, 'was within you all along.'
    The student finally understood: the journey was the destination.
    """,

    """
    Victory came not through strength but through surrender.
    The hero returned home bearing gifts: wisdom, humility, grace.
    The community welcomed him, transformed by his transformation.
    """,
]


def create_training_trajectories(config: DarkTraceConfig) -> list[Trajectory]:
    """Create training trajectories from sample texts."""
    print("Creating training trajectories...")
    trajectories = []

    for i, text in enumerate(TRAINING_TEXTS):
        observer = SemanticObserver(config)

        # Observe token by token
        tokens = text.split()
        for token in tokens:
            observer.observe(token + " ")

        # Create trajectory
        trajectory = Trajectory(
            trajectory_id=f"training_{i:03d}",
            model_name="example",
            prompt=f"Training passage {i+1}",
            snapshots=observer.get_trajectory(),
        )

        trajectories.append(trajectory)

    print(f"   ✓ Created {len(trajectories)} training trajectories")
    return trajectories


def main():
    """Run trajectory prediction example."""
    print("=" * 70)
    print("darkTrace - Trajectory Prediction Example")
    print("=" * 70)
    print()

    # 1. Setup
    print("1. Setting up observer and predictor...")
    config = DarkTraceConfig.narrative()
    print(f"   ✓ Config: {config.observer.domain} domain")
    print()

    # 2. Create training data
    print("2. Creating training data...")
    train_trajectories = create_training_trajectories(config)
    print()

    # 3. Train predictor
    print("3. Training trajectory predictor...")
    predictor = TrajectoryPredictor(method="linear", regularization=0.01)
    predictor.fit(train_trajectories)
    print(f"   ✓ Predictor trained ({predictor.method} method)")
    print(f"   ✓ Dimensions: {predictor.dimensions}D")
    print()

    # 4. Create test observation
    print("4. Creating test observation...")
    test_text = "The odyssey began with a single step into the unknown."
    observer = SemanticObserver(config)

    tokens = test_text.split()
    for token in tokens[:5]:  # First 5 words
        observer.observe(token + " ")

    current_state = observer.get_current_state()
    print(f"   ✓ Observed: '{' '.join(tokens[:5])}'")
    print(f"   ✓ Current velocity: {current_state.velocity_magnitude:.4f}")
    print()

    # 5. Predict future states
    print("5. Predicting next 10 semantic states...")
    predictions = predictor.predict(current_state, horizon=10)

    print()
    print("Predictions:")
    print("-" * 70)
    print(f"{'Step':<8} {'Confidence':<12} {'Position (first 3 dims)'}")
    print("-" * 70)

    for pred in predictions:
        pos_str = f"[{pred.position[0]:>7.3f}, {pred.position[1]:>7.3f}, {pred.position[2]:>7.3f}]"
        print(f"{pred.step:<8} {pred.confidence:>10.2%}  {pos_str}")

    print("-" * 70)
    print()

    # 6. Continue observing and compare
    print("6. Observing actual continuation...")
    remaining_tokens = tokens[5:]

    print()
    print("Actual vs Predicted:")
    print("-" * 70)
    print(f"{'Token':<20} {'Actual Pos[0]':<15} {'Predicted Pos[0]':<15} {'Error'}")
    print("-" * 70)

    for i, token in enumerate(remaining_tokens[:10]):
        # Observe actual
        state = observer.observe(token + " ")
        actual_pos = state.position[0]

        # Get prediction
        if i < len(predictions):
            predicted_pos = predictions[i].position[0]
            error = abs(actual_pos - predicted_pos)

            print(f"{token:<20} {actual_pos:>13.4f}  {predicted_pos:>13.4f}  {error:>8.4f}")

    print("-" * 70)
    print()

    # 7. Evaluate on test set
    print("7. Evaluating predictor...")
    test_trajectories = create_training_trajectories(config)  # Use fresh trajectories
    metrics = predictor.evaluate(test_trajectories, horizon=5)

    print()
    print("Evaluation Metrics:")
    print(f"   MAE (Mean Absolute Error):  {metrics['mae']:.6f}")
    print(f"   RMSE (Root Mean Squared):   {metrics['rmse']:.6f}")
    print(f"   Test samples:               {metrics['n_samples']}")
    print()

    # 8. Try different methods
    print("8. Comparing prediction methods...")
    methods = ["linear", "polynomial"]

    print()
    print(f"{'Method':<15} {'MAE':<12} {'RMSE':<12} {'Time (ms)'}")
    print("-" * 70)

    import time
    for method in methods:
        start = time.time()

        pred = TrajectoryPredictor(method=method)
        pred.fit(train_trajectories)
        metrics = pred.evaluate(test_trajectories, horizon=5)

        duration_ms = (time.time() - start) * 1000

        print(f"{method:<15} {metrics['mae']:>10.6f}  {metrics['rmse']:>10.6f}  {duration_ms:>8.1f}")

    print("-" * 70)
    print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  • Train on real LLM outputs")
    print("  • Try neural method for better predictions")
    print("  • Use predictions for semantic steering")
    print("  • Generate fingerprints (see fingerprinting.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()
