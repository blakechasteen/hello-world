"""
Adaptive Updater Demo
======================
Demonstrates safe pattern deployment with shadow mode, A/B testing, and automatic rollback.

Author: Claude Code
Date: November 13, 2025
"""

import asyncio
from pathlib import Path
from HoloLoom.routing.learning import (
    AdaptiveUpdater,
    DeploymentStrategy,
    Pattern,
    PatternScore,
    ContinuousValidator
)


class MockClassifier:
    """Mock classifier for demonstration."""

    def __init__(self):
        self.patterns = []

    def classify(self, query_text: str):
        """Simulate classification."""
        from dataclasses import dataclass

        @dataclass
        class MockClassification:
            complexity: str
            confidence: float

        class ComplexityValue:
            def __init__(self, value):
                self.value = value

        # Simple mock classification
        if "?" in query_text:
            complexity = "simple"
        else:
            complexity = "trivial"

        return MockClassification(
            complexity=ComplexityValue(complexity),
            confidence=0.95
        )


def create_sample_patterns():
    """Create sample patterns to deploy."""
    return [
        Pattern(
            regex=r"^(hi|hello|hey)$",
            complexity="trivial",
            score=PatternScore(
                precision=1.0,
                recall=0.95,
                support=1250,
                confidence=0.98,
                f1_score=0.974
            ),
            examples=["hi", "hello", "hey"]
        ),
        Pattern(
            regex=r"what (is|are) \w+",
            complexity="simple",
            score=PatternScore(
                precision=0.98,
                recall=0.87,
                support=3421,
                confidence=0.95,
                f1_score=0.921
            ),
            examples=["what is Python?", "what are embeddings?"]
        ),
        Pattern(
            regex=r"how (to|do|does) \w+",
            complexity="complex",
            score=PatternScore(
                precision=0.95,
                recall=0.82,
                support=2134,
                confidence=0.92,
                f1_score=0.882
            ),
            examples=["how does attention work?", "how to implement backprop?"]
        )
    ]


async def main():
    print("=" * 80)
    print("ADAPTIVE UPDATER DEMO")
    print("=" * 80)
    print()

    # Step 1: Setup
    print("Step 1: Initializing components...")
    classifier = MockClassifier()

    # Create validator with validation set
    from demos.demo_continuous_validator import create_sample_validation_set
    validation_dir = Path("./test_deployment")
    validation_file = validation_dir / "validation_set.json"

    from HoloLoom.routing.learning import create_validation_set
    queries = create_sample_validation_set()
    create_validation_set(queries, str(validation_file))

    validator = ContinuousValidator(
        classifier=classifier,
        validation_set_path=str(validation_file),
        regression_threshold=0.02,
        baseline_accuracy=1.0
    )

    # Create updater
    updater = AdaptiveUpdater(
        classifier=classifier,
        validator=validator,
        strategy=DeploymentStrategy.GRADUAL,
        regression_threshold=0.02
    )

    print("[OK] MockClassifier initialized")
    print("[OK] ContinuousValidator initialized")
    print("[OK] AdaptiveUpdater initialized")
    print(f"  - Default strategy: {updater.strategy.value}")
    print(f"  - Regression threshold: {updater.regression_threshold:.1%}")
    print()

    # Step 2: Create patterns to deploy
    print("Step 2: Creating patterns to deploy...")
    new_patterns = create_sample_patterns()
    print(f"Created {len(new_patterns)} new patterns:")
    for i, pattern in enumerate(new_patterns, 1):
        print(f"  {i}. {pattern.regex} ({pattern.complexity})")
        print(f"     Precision: {pattern.score.precision:.1%}, "
              f"Recall: {pattern.score.recall:.1%}, "
              f"F1: {pattern.score.f1_score:.3f}")
    print()

    # Step 3: Shadow mode deployment
    print("Step 3: Deploying in SHADOW mode...")
    print("-" * 80)
    result = await updater.deploy_patterns(
        new_patterns,
        force_strategy=DeploymentStrategy.SHADOW
    )
    print()
    print("Shadow Mode Result:")
    print(f"  Success: {result.success}")
    print(f"  Current Phase: {result.current_phase.value}")
    print(f"  Patterns Deployed: {result.patterns_deployed}")
    print(f"  Traffic Split: {updater.traffic_split:.0%}")
    print(f"  Message: {result.message}")
    print()

    # Step 4: A/B testing (10/90 split)
    print("Step 4: Deploying with A/B testing (10/90 split)...")
    print("-" * 80)
    result = await updater.deploy_patterns(
        new_patterns,
        force_strategy=DeploymentStrategy.AB_TEST
    )
    print()
    print("A/B Test Result:")
    print(f"  Success: {result.success}")
    print(f"  Current Phase: {result.current_phase.value}")
    print(f"  Traffic Split: {updater.traffic_split:.0%}")
    print(f"  Queries (New/Old): {result.metrics.queries_new_patterns}/{result.metrics.queries_old_patterns}")
    print(f"  Accuracy (New/Old): {result.metrics.accuracy_new:.1%}/{result.metrics.accuracy_old:.1%}")
    print(f"  Rollback Triggered: {result.rollback_triggered}")
    print(f"  Message: {result.message}")
    print()

    # Step 5: Gradual rollout (10% -> 50% -> 100%)
    print("Step 5: Deploying with GRADUAL rollout...")
    print("-" * 80)
    print("This will go through phases: 10% -> 50% -> 100%")
    print()

    result = await updater.deploy_patterns(
        new_patterns,
        force_strategy=DeploymentStrategy.GRADUAL
    )
    print()
    print("Gradual Rollout Result:")
    print(f"  Success: {result.success}")
    print(f"  Final Phase: {result.current_phase.value}")
    print(f"  Traffic Split: {updater.traffic_split:.0%}")
    print(f"  Final Accuracy: {result.metrics.accuracy_new:.1%}")
    print(f"  Rollback Triggered: {result.rollback_triggered}")
    print(f"  Message: {result.message}")
    print()

    # Step 6: Demonstrate rollback
    print("Step 6: Demonstrating automatic rollback...")
    print("-" * 80)
    print("Simulating regression scenario...")
    print()

    # Create a "bad" validator that will trigger regression
    class BadValidator(ContinuousValidator):
        """Validator that simulates regression."""
        async def validate_hourly(self, sample_size=100):
            result = await super().validate_hourly(sample_size)
            # Simulate regression
            result.overall_accuracy = 0.85  # 15% drop
            result.regression_detected = True
            return result

    bad_validator = BadValidator(
        classifier=classifier,
        validation_set_path=str(validation_file),
        regression_threshold=0.02,
        baseline_accuracy=1.0
    )

    # Create updater with bad validator
    rollback_updater = AdaptiveUpdater(
        classifier=classifier,
        validator=bad_validator,
        strategy=DeploymentStrategy.AB_TEST,
        regression_threshold=0.02
    )

    # Try to deploy (should trigger rollback)
    result = await rollback_updater.deploy_patterns(
        new_patterns,
        force_strategy=DeploymentStrategy.AB_TEST
    )
    print()
    print("Rollback Result:")
    print(f"  Success: {result.success}")
    print(f"  Current Phase: {result.current_phase.value}")
    print(f"  Rollback Triggered: {result.rollback_triggered}")
    print(f"  Rollback Reason: {result.rollback_reason}")
    print(f"  Message: {result.message}")
    print()

    # Step 7: View deployment status
    print("Step 7: Deployment status...")
    print("-" * 80)
    status = updater.get_deployment_status()
    print("Current Deployment Status:")
    print(f"  Current Phase: {status['current_phase']}")
    print(f"  Traffic Split: {status['traffic_split']:.0%}")
    print(f"  Pattern Versions: {status['pattern_versions']}")
    print(f"  Current Version: {status['current_version']}")
    print(f"  Metrics Collected: {status['metrics_collected']}")
    print()

    # Step 8: View metrics history
    print("Step 8: Metrics history...")
    print("-" * 80)
    metrics = updater.get_metrics_history()
    print(f"Total deployment events: {len(metrics)}")
    print()

    if metrics:
        print("Recent Deployments:")
        for i, m in enumerate(metrics[-5:], 1):
            print(f"\n  {i}. Phase: {m.phase.value}")
            print(f"     Queries: {m.queries_total} "
                  f"(new: {m.queries_new_patterns}, old: {m.queries_old_patterns})")
            print(f"     Accuracy: new={m.accuracy_new:.1%}, old={m.accuracy_old:.1%}")
    print()

    # Step 9: Export deployment history
    print("Step 9: Exporting deployment history...")
    history_file = validation_dir / "deployment_history.json"
    updater.export_deployment_history(str(history_file))
    print(f"Exported deployment history to {history_file}")
    print()

    # Summary
    print("=" * 80)
    print("ADAPTIVE UPDATER DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  [OK] Shadow mode deployment (no production impact)")
    print("  [OK] A/B testing (10/90 traffic split)")
    print("  [OK] Gradual rollout (10% -> 50% -> 100%)")
    print("  [OK] Automatic rollback on regression")
    print("  [OK] Pattern versioning for rollback")
    print("  [OK] Deployment metrics tracking")
    print("  [OK] Deployment history export")
    print()
    print("Deployment Timeline:")
    print("  Day 1-2: Shadow mode (collect metrics, no impact)")
    print("  Day 3: A/B test at 10% (validate improvement)")
    print("  Day 4-5: Increase to 50% (continued monitoring)")
    print("  Day 6-7: Full deployment at 100%")
    print("  Anytime: Automatic rollback if regression detected")
    print()
    print("Safety Mechanisms:")
    print("  • Shadow mode: Test patterns without risk")
    print("  • Traffic splitting: Gradual exposure")
    print("  • Continuous validation: Monitor accuracy at each phase")
    print("  • Automatic rollback: Revert on >2% accuracy drop")
    print("  • Version history: Keep last 10 pattern versions")
    print()
    print("Next Steps:")
    print("  1. Integrate AdaptiveUpdater into MoonshotQueryClassifier")
    print("  2. Connect to production classification telemetry")
    print("  3. Set up automated deployment pipeline")
    print("  4. Configure rollback triggers and alerts")


if __name__ == "__main__":
    asyncio.run(main())
