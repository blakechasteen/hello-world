"""
Continuous Validator Demo
==========================
Demonstrates how ContinuousValidator monitors classifier accuracy and detects regressions.

Author: Claude Code
Date: November 13, 2025
"""

import asyncio
import time
from pathlib import Path
from hololoom.routing.learning import (
    ContinuousValidator,
    ValidationQuery,
    create_validation_set
)


class MockClassifier:
    """Mock classifier for demonstration."""

    def __init__(self, accuracy_mode="high"):
        """
        Initialize mock classifier.

        Args:
            accuracy_mode: "high" (100%), "good" (95%), "degraded" (90%)
        """
        self.accuracy_mode = accuracy_mode
        self.call_count = 0

    def classify(self, query_text: str):
        """Simulate classification with configurable accuracy."""
        self.call_count += 1

        # Ground truth mapping (for demo purposes)
        ground_truth = {
            "hi": "trivial",
            "hello": "trivial",
            "good morning": "trivial",
            "what is Python?": "simple",
            "what is machine learning?": "simple",
            "tell me about transformers": "simple",
            "how does backpropagation work?": "complex",
            "how does attention work?": "complex",
            "explain gradient descent": "complex",
            "analyze Thompson Sampling vs UCB": "research",
            "compare reinforcement learning algorithms": "research",
            "evaluate different optimization techniques": "research"
        }

        expected = ground_truth.get(query_text, "simple")

        # Simulate accuracy based on mode
        import random
        if self.accuracy_mode == "high":
            # 100% accuracy
            predicted = expected
            confidence = 0.95 + random.random() * 0.05
        elif self.accuracy_mode == "good":
            # 95% accuracy
            if random.random() < 0.95:
                predicted = expected
                confidence = 0.90 + random.random() * 0.08
            else:
                # Misclassify 5%
                levels = ["trivial", "simple", "complex", "research"]
                predicted = random.choice([l for l in levels if l != expected])
                confidence = 0.65 + random.random() * 0.15
        else:  # degraded
            # 90% accuracy (regression)
            if random.random() < 0.90:
                predicted = expected
                confidence = 0.85 + random.random() * 0.10
            else:
                # Misclassify 10%
                levels = ["trivial", "simple", "complex", "research"]
                predicted = random.choice([l for l in levels if l != expected])
                confidence = 0.60 + random.random() * 0.20

        # Return mock classification result
        from dataclasses import dataclass

        @dataclass
        class MockClassification:
            complexity: str
            confidence: float

        # Convert to enum-like object
        class ComplexityValue:
            def __init__(self, value):
                self.value = value

        classification = MockClassification(
            complexity=ComplexityValue(predicted),
            confidence=confidence
        )

        return classification


def create_sample_validation_set():
    """Create sample validation queries."""
    queries = [
        # TRIVIAL (10 queries)
        ("hi", "trivial"),
        ("hello", "trivial"),
        ("hey", "trivial"),
        ("good morning", "trivial"),
        ("good afternoon", "trivial"),
        ("greetings", "trivial"),
        ("hi there", "trivial"),
        ("hello world", "trivial"),
        ("hey there", "trivial"),
        ("good evening", "trivial"),

        # SIMPLE (15 queries)
        ("what is Python?", "simple"),
        ("what is machine learning?", "simple"),
        ("what is a neural network?", "simple"),
        ("what are transformers?", "simple"),
        ("what is reinforcement learning?", "simple"),
        ("tell me about Python", "simple"),
        ("tell me about transformers", "simple"),
        ("tell me about neural networks", "simple"),
        ("describe machine learning", "simple"),
        ("describe gradient descent", "simple"),
        ("explain what a tensor is", "simple"),
        ("explain embeddings", "simple"),
        ("define supervised learning", "simple"),
        ("define unsupervised learning", "simple"),
        ("list optimization algorithms", "simple"),

        # COMPLEX (12 queries)
        ("how does backpropagation work?", "complex"),
        ("how does attention work?", "complex"),
        ("how does gradient descent work?", "complex"),
        ("how does a transformer work?", "complex"),
        ("how does BERT work?", "complex"),
        ("explain gradient descent", "complex"),
        ("explain backpropagation", "complex"),
        ("explain attention mechanisms", "complex"),
        ("explain how transformers work", "complex"),
        ("walk me through Q-learning", "complex"),
        ("walk me through policy gradient", "complex"),
        ("walk me through Adam optimizer", "complex"),

        # RESEARCH (8 queries)
        ("analyze Thompson Sampling vs UCB", "research"),
        ("compare reinforcement learning algorithms", "research"),
        ("compare GPT and BERT architectures", "research"),
        ("compare supervised and unsupervised learning", "research"),
        ("evaluate different optimization techniques", "research"),
        ("evaluate transformer vs RNN performance", "research"),
        ("analyze attention mechanisms in detail", "research"),
        ("analyze optimization landscape", "research"),
    ]

    return queries


async def main():
    print("=" * 80)
    print("CONTINUOUS VALIDATOR DEMO")
    print("=" * 80)
    print()

    # Step 1: Create validation set
    print("Step 1: Creating validation set...")
    queries = create_sample_validation_set()
    print(f"Created {len(queries)} validation queries")
    print(f"  - TRIVIAL: {sum(1 for _, c in queries if c == 'trivial')} queries")
    print(f"  - SIMPLE: {sum(1 for _, c in queries if c == 'simple')} queries")
    print(f"  - COMPLEX: {sum(1 for _, c in queries if c == 'complex')} queries")
    print(f"  - RESEARCH: {sum(1 for _, c in queries if c == 'research')} queries")
    print()

    # Save validation set to file
    validation_dir = Path("./test_validation")
    validation_file = validation_dir / "validation_set.json"
    create_validation_set(queries, str(validation_file))
    print(f"Saved validation set to {validation_file}")
    print()

    # Step 2: Initialize ContinuousValidator with high-accuracy classifier
    print("Step 2: Initializing ContinuousValidator (baseline)...")
    classifier = MockClassifier(accuracy_mode="high")

    validator = ContinuousValidator(
        classifier=classifier,
        validation_set_path=str(validation_file),
        regression_threshold=0.02,  # 2% drop triggers regression
        baseline_accuracy=1.0  # 100% baseline
    )
    print("[OK] ContinuousValidator initialized")
    print(f"  - Baseline accuracy: {validator.baseline_accuracy:.1%}")
    print(f"  - Regression threshold: {validator.regression_threshold:.1%}")
    print(f"  - Validation set size: {len(validator.validation_set)}")
    print()

    # Step 3: Run hourly validation (100 queries)
    print("Step 3: Running hourly validation (quick check)...")
    print("-" * 80)
    result = await validator.validate_hourly(sample_size=20)  # Use 20 for demo
    print()
    print("Hourly Validation Result:")
    print(f"  Overall Accuracy: {result.overall_accuracy:.1%}")
    print(f"  Queries Validated: {result.total_queries}")
    print(f"  Correct: {result.correct}")
    print(f"  Incorrect: {result.incorrect}")
    print(f"  Regression Detected: {'YES' if result.regression_detected else 'NO'}")
    print()
    print("  By Complexity:")
    for complexity, accuracy in result.by_complexity.items():
        print(f"    {complexity.upper():10s}: {accuracy:.1%}")
    print()

    # Step 4: Run daily validation (all queries)
    print("Step 4: Running daily validation (comprehensive)...")
    print("-" * 80)
    result = await validator.validate_daily()
    print()
    print("Daily Validation Result:")
    print(f"  Overall Accuracy: {result.overall_accuracy:.1%}")
    print(f"  Queries Validated: {result.total_queries}")
    print(f"  Correct: {result.correct}")
    print(f"  Incorrect: {result.incorrect}")
    print(f"  Regression Detected: {'YES' if result.regression_detected else 'NO'}")
    print()
    print("  By Complexity:")
    for complexity, accuracy in result.by_complexity.items():
        print(f"    {complexity.upper():10s}: {accuracy:.1%}")
    print()

    # Step 5: Simulate classifier degradation
    print("Step 5: Simulating classifier degradation...")
    print("-" * 80)
    print("Switching to degraded classifier (90% accuracy)...")
    validator.classifier = MockClassifier(accuracy_mode="degraded")
    print()

    # Run validation with degraded classifier
    result = await validator.validate_daily()
    print()
    print("Degraded Classifier Validation:")
    print(f"  Overall Accuracy: {result.overall_accuracy:.1%}")
    print(f"  Baseline Accuracy: {result.baseline_accuracy:.1%}")
    print(f"  Accuracy Drop: {(result.baseline_accuracy - result.overall_accuracy):.1%}")
    print(f"  Regression Detected: {'YES' if result.regression_detected else 'NO'}")
    print()

    if result.regression_detected:
        print("  [ALERT] Regression detected!")
        print()

    # Step 6: View alerts
    print("Step 6: Viewing alerts...")
    print("-" * 80)
    alerts = validator.get_alerts()

    if alerts:
        print(f"Found {len(alerts)} alert(s):")
        print()
        for i, alert in enumerate(alerts, 1):
            print(f"Alert {i}:")
            print(f"  Severity: {alert.severity.upper()}")
            print(f"  Current Accuracy: {alert.current_accuracy:.1%}")
            print(f"  Baseline Accuracy: {alert.baseline_accuracy:.1%}")
            print(f"  Drop: {alert.drop_percentage:.1%}")
            print(f"  Affected: {', '.join(alert.affected_complexity) if alert.affected_complexity else 'Overall'}")
            print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
            print()
            print("  Message:")
            for line in alert.message.split('\n'):
                print(f"    {line}")
            print()
    else:
        print("No alerts found")
    print()

    # Step 7: Export validation history
    print("Step 7: Exporting validation history...")
    history_file = validation_dir / "validation_history.json"
    validator.export_validation_history(str(history_file))
    print(f"Exported validation history to {history_file}")
    print(f"Total validations: {len(validator.validation_history)}")
    print()

    # Summary
    print("=" * 80)
    print("CONTINUOUS VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Hourly validation: 20 queries -> {result.overall_accuracy:.1%} accuracy")
    print(f"  - Daily validation: {result.total_queries} queries -> {result.overall_accuracy:.1%} accuracy")
    print(f"  - Regression detection: {len(alerts)} alert(s)")
    print(f"  - Validation history: {len(validator.validation_history)} records")
    print()
    print("Key Features Demonstrated:")
    print("  [OK] Validation set management")
    print("  [OK] Hourly/daily validation schedules")
    print("  [OK] Regression detection (>2% threshold)")
    print("  [OK] Alert generation and retrieval")
    print("  [OK] Validation history tracking")
    print("  [OK] Export functionality")
    print()
    print("Next Steps:")
    print("  1. Integrate into MoonshotQueryClassifier")
    print("  2. Connect to production telemetry logs")
    print("  3. Set up automated validation schedule")
    print("  4. Configure alerting (Slack, email)")


if __name__ == "__main__":
    asyncio.run(main())
