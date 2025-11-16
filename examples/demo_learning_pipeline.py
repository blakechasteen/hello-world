"""
AutoFix Learning Pipeline Demo
================================

Demonstrates the complete learning pipeline workflow:
1. Generate synthetic autofix tracking data
2. Run learning pipeline
3. Analyze results
4. Apply recommendations

Author: mythRL Team
Date: November 16, 2025
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autofix_tracker import AutoFixTracker, AutoFixResult
from autofix_learning_pipeline import (
    AutoFixLearningPipeline,
    TrainingDataExporter,
    PatternLearner,
    CalibrationMonitor
)


def generate_synthetic_data(num_fixes: int = 100) -> str:
    """
    Generate synthetic autofix tracking data for demonstration.

    Returns:
        Path to generated sessions file
    """
    print("=" * 70)
    print("GENERATING SYNTHETIC AUTOFIX DATA")
    print("=" * 70)
    print()

    tracker = AutoFixTracker(output_dir="./demo_tracking")

    # Start session
    session_id = tracker.start_session(
        max_files=50,
        categories=['dead_code', 'hardcoded_values', 'missing_docstrings', 'type_errors'],
        confidence_threshold=0.85
    )

    # Generate diverse fixes with realistic patterns
    categories = ['dead_code', 'hardcoded_values', 'missing_docstrings', 'type_errors']
    severities = ['low', 'medium', 'high']
    strategies = ['ast', 'template', 'manual']

    base_time = datetime.now()

    for i in range(num_fixes):
        category = categories[i % len(categories)]
        severity = severities[i % len(severities)]
        strategy = strategies[i % len(strategies)]

        # Realistic confidence based on category and strategy
        if category == 'dead_code' and strategy == 'ast':
            confidence = 0.90 + (i % 10) * 0.01  # High confidence
            applied = confidence > 0.85
        elif category == 'hardcoded_values' and strategy == 'template':
            confidence = 0.75 + (i % 15) * 0.01  # Medium confidence
            applied = confidence > 0.80
        elif category == 'type_errors' and strategy == 'manual':
            confidence = 0.60 + (i % 20) * 0.01  # Lower confidence
            applied = confidence > 0.75
        else:
            confidence = 0.70 + (i % 30) * 0.01
            applied = confidence > 0.80

        # Add some noise
        import random
        if random.random() < 0.1:  # 10% randomness
            applied = not applied

        # Create fix result
        fix = AutoFixResult(
            fix_id=f"fix_{i:03d}",
            category=category,
            severity=severity,
            message=f"Fix {category} issue",
            confidence=min(0.99, confidence),
            file_path=f"src/module_{i % 10}.py",
            line_number=10 + i,
            code_snippet=f"# Sample code {i}",
            strategy=strategy,
            original_code=f"old_code_{i}",
            fixed_code=f"new_code_{i}",
            diff=f"-old_code_{i}\n+new_code_{i}",
            applied=applied,
            validation_passed=applied,  # Simplified: applied == validated
            errors=[] if applied else [f"Validation error {i}"],
            warnings=[f"Warning {i}"] if i % 5 == 0 else [],
            branch=f"autofix/batch_{session_id}",
            commit=f"abc{i:04d}",
            timestamp=(base_time + timedelta(minutes=i)).isoformat(),
            duration_ms=50.0 + i * 2
        )

        tracker.track_fix(fix)

    # End session
    tracker.end_session()

    # Export to standard location
    all_sessions_file = "demo_tracking/all_sessions.json"
    tracker.export_json(all_sessions_file)

    print(f"✓ Generated {num_fixes} synthetic fixes")
    print()

    return all_sessions_file


def demo_training_data_export(sessions_file: str):
    """Demonstrate training data export."""
    print("=" * 70)
    print("DEMO 1: TRAINING DATA EXPORT")
    print("=" * 70)
    print()

    exporter = TrainingDataExporter(sessions_file)

    # Extract features
    features, labels = exporter.extract_features()

    print(f"Extracted {len(features)} samples")
    print(f"  Positive: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Negative: {len(labels) - sum(labels)} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")
    print()

    # Show sample features
    print("Sample Feature Vector:")
    if features:
        sample = features[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")
    print()

    # Export to CSV
    exporter.export_csv("demo_tracking/training_data.csv")

    # Get train/validation split
    X_train, y_train, X_val, y_val = exporter.get_train_val_split(
        val_ratio=0.2,
        stratify=True
    )
    print()


def demo_pattern_discovery(sessions_file: str):
    """Demonstrate pattern discovery."""
    print("=" * 70)
    print("DEMO 2: PATTERN DISCOVERY")
    print("=" * 70)
    print()

    learner = PatternLearner(sessions_file)

    # Discover patterns
    patterns = learner.discover_patterns(
        min_support=5,
        min_success_rate=0.7
    )

    # Show top patterns
    print("\nTop Patterns by Support:\n")

    top_patterns = learner.get_top_patterns(n=10)

    for i, pattern in enumerate(top_patterns, 1):
        print(f"{i}. {pattern.pattern_type}")
        print(f"   Conditions: {pattern.conditions}")
        print(f"   Success Rate: {pattern.success_rate:.1%}")
        print(f"   Support: {pattern.support} instances")
        print(f"   Avg Confidence: {pattern.confidence_avg:.2f}")
        print(f"   Recommendation: {pattern.recommendation}")
        print()

    # Export patterns
    learner.export_patterns("demo_tracking/learned_patterns.json")
    print()


def demo_calibration_monitoring(sessions_file: str):
    """Demonstrate calibration monitoring."""
    print("=" * 70)
    print("DEMO 3: CALIBRATION MONITORING")
    print("=" * 70)
    print()

    monitor = CalibrationMonitor(sessions_file)

    # Compute metrics
    metrics = monitor.compute_calibration_metrics(n_bins=10)

    print("Calibration Metrics:\n")
    print(f"  Brier Score: {metrics.brier_score:.3f} (0.0 = perfect)")
    print(f"  Expected Calibration Error (ECE): {metrics.ece:.3f}")
    print(f"  Maximum Calibration Error (MCE): {metrics.mce:.3f}")
    print(f"  Overconfident Predictions: {metrics.overconfident_ratio:.1%}")
    print(f"  Underconfident Predictions: {metrics.underconfident_ratio:.1%}")
    print(f"  Recommended Adjustment: {metrics.recommended_adjustment:+.3f}")
    print()

    # Interpret results
    print("Interpretation:\n")

    if metrics.brier_score < 0.1:
        print("  ✅ Excellent Brier score - predictions are well-calibrated")
    elif metrics.brier_score < 0.2:
        print("  ✓ Good Brier score - reasonable calibration")
    else:
        print("  ⚠️  High Brier score - poor calibration, needs improvement")

    if metrics.ece < 0.05:
        print("  ✅ Excellent ECE - minimal calibration error")
    elif metrics.ece < 0.1:
        print("  ✓ Good ECE - acceptable calibration error")
    else:
        print("  ⚠️  High ECE - significant calibration error")

    if metrics.overconfident_ratio > 0.6:
        print("  ⚠️  High overconfidence - model predicts higher than actual")
        print(f"     → Increase confidence threshold by ~{abs(metrics.recommended_adjustment):.3f}")
    elif metrics.underconfident_ratio > 0.6:
        print("  ⚠️  High underconfidence - model predicts lower than actual")
        print(f"     → Decrease confidence threshold by ~{abs(metrics.recommended_adjustment):.3f}")
    else:
        print("  ✅ Balanced confidence - no major bias")

    print()

    # Export recommendations
    monitor.export_recommendations("demo_tracking/calibration_report.json")
    print()


def demo_full_pipeline(sessions_file: str):
    """Demonstrate full learning pipeline."""
    print("=" * 70)
    print("DEMO 4: FULL LEARNING PIPELINE")
    print("=" * 70)
    print()

    pipeline = AutoFixLearningPipeline(sessions_file)

    # Run complete pipeline
    pipeline.run_full_pipeline(output_dir="demo_tracking/learning_output")

    print("\n" + "=" * 70)
    print("OUTPUTS GENERATED")
    print("=" * 70)
    print()

    output_dir = Path("demo_tracking/learning_output")

    outputs = [
        ("training_data.csv", "ML-ready training data (CSV format)"),
        ("training_data.json", "Training data with metadata (JSON format)"),
        ("learned_patterns.json", "Discovered patterns and recommendations"),
        ("calibration_report.json", "Calibration analysis and metrics"),
        ("learning_summary.md", "Comprehensive summary report")
    ]

    for filename, description in outputs:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✓ {filename}")
            print(f"  {description}")
            print(f"  Size: {size:,} bytes")
            print()


def demo_pattern_integration():
    """Demonstrate how to integrate learned patterns."""
    print("=" * 70)
    print("DEMO 5: PATTERN INTEGRATION")
    print("=" * 70)
    print()

    print("How to integrate learned patterns into AutofixPolicy:\n")

    print("1. Review learned patterns:")
    print("   - Check learned_patterns.json for high-confidence patterns")
    print("   - Look for category → strategy combinations with >85% success")
    print()

    print("2. Update policy thresholds:")
    print("   ```python")
    print("   # If calibration shows overconfidence:")
    print("   policy = AutofixPolicy.balanced()")
    print("   policy.min_confidence_auto = 0.90  # Increase from 0.85")
    print("   ```")
    print()

    print("3. Adjust strategy preferences:")
    print("   ```python")
    print("   # If patterns show AST > template for dead_code:")
    print("   policy.prefer_ast_over_template = True")
    print("   policy.allow_template_autofix = False  # For high-risk categories")
    print("   ```")
    print()

    print("4. Configure category-specific policies:")
    print("   ```python")
    print("   # Create domain-specific policies based on patterns")
    print("   if category == 'dead_code' and learned_success_rate > 0.9:")
    print("       return AutofixPolicy.aggressive()")
    print("   elif category == 'type_errors' and learned_success_rate < 0.7:")
    print("       return AutofixPolicy.conservative()")
    print("   ```")
    print()

    print("5. Monitor and iterate:")
    print("   - Run learning pipeline after each session")
    print("   - Track calibration drift over time")
    print("   - A/B test new thresholds before deployment")
    print()


def main():
    """Run all demos."""
    print()
    print("*" * 70)
    print("AUTOFIX LEARNING PIPELINE - COMPLETE DEMO")
    print("*" * 70)
    print()

    # Generate synthetic data
    sessions_file = generate_synthetic_data(num_fixes=100)

    # Demo 1: Training data export
    demo_training_data_export(sessions_file)

    # Demo 2: Pattern discovery
    demo_pattern_discovery(sessions_file)

    # Demo 3: Calibration monitoring
    demo_calibration_monitoring(sessions_file)

    # Demo 4: Full pipeline
    demo_full_pipeline(sessions_file)

    # Demo 5: Pattern integration
    demo_pattern_integration()

    print("*" * 70)
    print("DEMO COMPLETE")
    print("*" * 70)
    print()
    print("Next Steps:")
    print("1. Review outputs in demo_tracking/learning_output/")
    print("2. Check learning_summary.md for comprehensive report")
    print("3. Apply recommendations to AutofixPolicy")
    print("4. Set up continuous learning pipeline for production")
    print()


if __name__ == '__main__':
    main()
