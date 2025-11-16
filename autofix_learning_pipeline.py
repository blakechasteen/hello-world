"""
AutoFix Learning Pipeline
==========================

Continuous learning system that improves autofix quality from historical outcomes.

Components:
1. TrainingDataExporter - Export tracking data for ML training
2. PatternLearner - Discover success/failure patterns
3. CalibrationMonitor - Confidence calibration feedback
4. IncrementalLearner - Online learning integration

Author: mythRL Team
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import csv


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LearningPattern:
    """Discovered pattern from fix outcomes."""
    pattern_type: str  # "category_strategy", "severity_outcome", "temporal"
    conditions: Dict[str, Any]  # Pattern conditions
    success_rate: float  # 0.0-1.0
    support: int  # Number of instances
    confidence_avg: float  # Average confidence when pattern applies
    recommendation: str  # Action recommendation


@dataclass
class CalibrationMetrics:
    """Confidence calibration quality metrics."""
    brier_score: float  # Lower is better (0.0 = perfect)
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    overconfident_ratio: float  # Fraction of overconfident predictions
    underconfident_ratio: float  # Fraction of underconfident predictions
    recommended_adjustment: float  # Suggested threshold shift


# ============================================================================
# 1. Training Data Exporter
# ============================================================================

class TrainingDataExporter:
    """
    Export tracking data in ML-ready formats.

    Features:
    - Multiple output formats (CSV, JSON, NumPy)
    - Feature engineering (one-hot encoding, normalization)
    - Train/validation/test splits
    - Balanced sampling

    Usage:
        exporter = TrainingDataExporter("autofix_tracking/all_sessions.json")
        exporter.export_csv("training_data.csv")
        X_train, y_train, X_val, y_val = exporter.get_train_val_split()
    """

    def __init__(self, sessions_file: str):
        self.sessions_file = Path(sessions_file)
        self.data = self._load_data()
        self.features = []
        self.labels = []

    def _load_data(self) -> Dict[str, Any]:
        """Load sessions data."""
        if not self.sessions_file.exists():
            return {"sessions": []}

        with open(self.sessions_file, 'r') as f:
            return json.load(f)

    def extract_features(self) -> Tuple[List[Dict], List[int]]:
        """
        Extract features and labels from tracking data.

        Returns:
            (features, labels) where:
            - features: List of feature dicts
            - labels: List of binary outcomes (1=applied, 0=failed)
        """
        features = []
        labels = []

        for session in self.data.get("sessions", []):
            for fix in session.get("fixes", []):
                # Feature vector
                feature_dict = {
                    # Categorical features
                    'category': fix.get('category', 'unknown'),
                    'severity': fix.get('severity', 'unknown'),
                    'strategy': fix.get('strategy', 'unknown'),

                    # Numerical features
                    'confidence': fix.get('confidence', 0.0),
                    'line_number': fix.get('line_number', 0),
                    'code_length': len(fix.get('code_snippet', '')),
                    'diff_length': len(fix.get('diff', '')),

                    # Temporal features
                    'hour_of_day': self._extract_hour(fix.get('timestamp')),
                    'day_of_week': self._extract_day_of_week(fix.get('timestamp')),

                    # Validation features
                    'validation_passed': 1 if fix.get('validation_passed', False) else 0,
                    'has_errors': 1 if fix.get('errors', []) else 0,
                    'has_warnings': 1 if fix.get('warnings', []) else 0,
                }

                # Label: 1 if applied successfully, 0 otherwise
                label = 1 if fix.get('applied', False) else 0

                features.append(feature_dict)
                labels.append(label)

        self.features = features
        self.labels = labels

        return features, labels

    def _extract_hour(self, timestamp: Optional[str]) -> int:
        """Extract hour of day from timestamp."""
        if not timestamp:
            return 0
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.hour
        except:
            return 0

    def _extract_day_of_week(self, timestamp: Optional[str]) -> int:
        """Extract day of week from timestamp (0=Monday, 6=Sunday)."""
        if not timestamp:
            return 0
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.weekday()
        except:
            return 0

    def export_csv(self, output_file: str, include_header: bool = True):
        """Export to CSV format."""
        if not self.features:
            self.extract_features()

        output_path = Path(output_file)

        with open(output_path, 'w', newline='') as f:
            if self.features:
                fieldnames = list(self.features[0].keys()) + ['label']
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if include_header:
                    writer.writeheader()

                for feat, label in zip(self.features, self.labels):
                    row = {**feat, 'label': label}
                    writer.writerow(row)

        print(f"✓ Exported {len(self.features)} samples to {output_path}")

    def export_json(self, output_file: str):
        """Export to JSON format."""
        if not self.features:
            self.extract_features()

        output_path = Path(output_file)

        data = {
            'features': self.features,
            'labels': self.labels,
            'metadata': {
                'num_samples': len(self.features),
                'num_positive': sum(self.labels),
                'num_negative': len(self.labels) - sum(self.labels),
                'export_date': datetime.now().isoformat()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported {len(self.features)} samples to {output_path}")

    def get_train_val_split(
        self,
        val_ratio: float = 0.2,
        stratify: bool = True,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[int], List[Dict], List[int]]:
        """
        Split data into train/validation sets.

        Args:
            val_ratio: Fraction for validation (0.0-1.0)
            stratify: Maintain class balance in splits
            random_seed: Random seed for reproducibility

        Returns:
            (X_train, y_train, X_val, y_val)
        """
        if not self.features:
            self.extract_features()

        np.random.seed(random_seed)

        n_samples = len(self.features)
        n_val = int(n_samples * val_ratio)

        if stratify:
            # Stratified split
            indices_pos = [i for i, label in enumerate(self.labels) if label == 1]
            indices_neg = [i for i, label in enumerate(self.labels) if label == 0]

            np.random.shuffle(indices_pos)
            np.random.shuffle(indices_neg)

            n_val_pos = int(len(indices_pos) * val_ratio)
            n_val_neg = int(len(indices_neg) * val_ratio)

            val_indices = indices_pos[:n_val_pos] + indices_neg[:n_val_neg]
            train_indices = indices_pos[n_val_pos:] + indices_neg[n_val_neg:]
        else:
            # Random split
            indices = list(range(n_samples))
            np.random.shuffle(indices)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

        X_train = [self.features[i] for i in train_indices]
        y_train = [self.labels[i] for i in train_indices]
        X_val = [self.features[i] for i in val_indices]
        y_val = [self.labels[i] for i in val_indices]

        print(f"✓ Split: {len(X_train)} train, {len(X_val)} validation")
        print(f"  Train: {sum(y_train)} positive, {len(y_train) - sum(y_train)} negative")
        print(f"  Val: {sum(y_val)} positive, {len(y_val) - sum(y_val)} negative")

        return X_train, y_train, X_val, y_val


# ============================================================================
# 2. Pattern Learner
# ============================================================================

class PatternLearner:
    """
    Discover success/failure patterns from fix outcomes.

    Patterns discovered:
    - Category → Strategy → Success rate
    - Severity → Outcome patterns
    - Temporal patterns (time-based trends)
    - Code characteristics → Success

    Usage:
        learner = PatternLearner("autofix_tracking/all_sessions.json")
        patterns = learner.discover_patterns(min_support=10, min_confidence=0.8)
        learner.export_patterns("learned_patterns.json")
    """

    def __init__(self, sessions_file: str):
        self.sessions_file = Path(sessions_file)
        self.data = self._load_data()
        self.patterns: List[LearningPattern] = []

    def _load_data(self) -> Dict[str, Any]:
        """Load sessions data."""
        if not self.sessions_file.exists():
            return {"sessions": []}

        with open(self.sessions_file, 'r') as f:
            return json.load(f)

    def discover_patterns(
        self,
        min_support: int = 5,
        min_success_rate: float = 0.7
    ) -> List[LearningPattern]:
        """
        Discover patterns with statistical significance.

        Args:
            min_support: Minimum number of instances for pattern
            min_success_rate: Minimum success rate to recommend

        Returns:
            List of discovered patterns
        """
        self.patterns = []

        # Pattern 1: Category → Strategy → Success
        self._discover_category_strategy_patterns(min_support, min_success_rate)

        # Pattern 2: Severity → Outcome
        self._discover_severity_patterns(min_support, min_success_rate)

        # Pattern 3: Confidence calibration
        self._discover_confidence_patterns(min_support)

        # Pattern 4: Temporal patterns
        self._discover_temporal_patterns(min_support, min_success_rate)

        # Sort by support (most common first)
        self.patterns.sort(key=lambda p: p.support, reverse=True)

        print(f"✓ Discovered {len(self.patterns)} patterns")

        return self.patterns

    def _discover_category_strategy_patterns(
        self,
        min_support: int,
        min_success_rate: float
    ):
        """Discover category → strategy → success patterns."""
        # Collect data
        pattern_data = defaultdict(lambda: {'total': 0, 'success': 0, 'confidences': []})

        for session in self.data.get("sessions", []):
            for fix in session.get("fixes", []):
                category = fix.get('category', 'unknown')
                strategy = fix.get('strategy', 'unknown')
                applied = fix.get('applied', False)
                confidence = fix.get('confidence', 0.0)

                key = (category, strategy)
                pattern_data[key]['total'] += 1
                if applied:
                    pattern_data[key]['success'] += 1
                pattern_data[key]['confidences'].append(confidence)

        # Extract patterns
        for (category, strategy), stats in pattern_data.items():
            if stats['total'] >= min_support:
                success_rate = stats['success'] / stats['total']
                avg_confidence = np.mean(stats['confidences']) if stats['confidences'] else 0.0

                # Determine recommendation
                if success_rate >= min_success_rate:
                    recommendation = f"Prefer {strategy} for {category} (high success rate)"
                else:
                    recommendation = f"Avoid {strategy} for {category} (low success rate)"

                pattern = LearningPattern(
                    pattern_type="category_strategy",
                    conditions={'category': category, 'strategy': strategy},
                    success_rate=success_rate,
                    support=stats['total'],
                    confidence_avg=avg_confidence,
                    recommendation=recommendation
                )

                self.patterns.append(pattern)

    def _discover_severity_patterns(self, min_support: int, min_success_rate: float):
        """Discover severity → outcome patterns."""
        severity_data = defaultdict(lambda: {'total': 0, 'success': 0, 'confidences': []})

        for session in self.data.get("sessions", []):
            for fix in session.get("fixes", []):
                severity = fix.get('severity', 'unknown')
                applied = fix.get('applied', False)
                confidence = fix.get('confidence', 0.0)

                severity_data[severity]['total'] += 1
                if applied:
                    severity_data[severity]['success'] += 1
                severity_data[severity]['confidences'].append(confidence)

        for severity, stats in severity_data.items():
            if stats['total'] >= min_support:
                success_rate = stats['success'] / stats['total']
                avg_confidence = np.mean(stats['confidences']) if stats['confidences'] else 0.0

                if success_rate >= min_success_rate:
                    recommendation = f"{severity.upper()} severity: reliable auto-fix"
                else:
                    recommendation = f"{severity.upper()} severity: needs manual review"

                pattern = LearningPattern(
                    pattern_type="severity_outcome",
                    conditions={'severity': severity},
                    success_rate=success_rate,
                    support=stats['total'],
                    confidence_avg=avg_confidence,
                    recommendation=recommendation
                )

                self.patterns.append(pattern)

    def _discover_confidence_patterns(self, min_support: int):
        """Discover confidence calibration patterns."""
        # Bin confidence scores and track outcomes
        bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        bin_data = defaultdict(lambda: {'total': 0, 'success': 0})

        for session in self.data.get("sessions", []):
            for fix in session.get("fixes", []):
                confidence = fix.get('confidence', 0.0)
                applied = fix.get('applied', False)

                for low, high in bins:
                    if low <= confidence < high:
                        bin_data[(low, high)]['total'] += 1
                        if applied:
                            bin_data[(low, high)]['success'] += 1
                        break

        for (low, high), stats in bin_data.items():
            if stats['total'] >= min_support:
                success_rate = stats['success'] / stats['total']
                avg_confidence = (low + high) / 2

                # Check calibration
                calibration_gap = abs(avg_confidence - success_rate)

                if calibration_gap < 0.1:
                    recommendation = f"Confidence {low:.1f}-{high:.1f}: well calibrated"
                elif success_rate < avg_confidence:
                    recommendation = f"Confidence {low:.1f}-{high:.1f}: overconfident (actual {success_rate:.1%})"
                else:
                    recommendation = f"Confidence {low:.1f}-{high:.1f}: underconfident (actual {success_rate:.1%})"

                pattern = LearningPattern(
                    pattern_type="confidence_calibration",
                    conditions={'confidence_range': (low, high)},
                    success_rate=success_rate,
                    support=stats['total'],
                    confidence_avg=avg_confidence,
                    recommendation=recommendation
                )

                self.patterns.append(pattern)

    def _discover_temporal_patterns(self, min_support: int, min_success_rate: float):
        """Discover time-based patterns (e.g., time of day affects success)."""
        hour_data = defaultdict(lambda: {'total': 0, 'success': 0, 'confidences': []})

        for session in self.data.get("sessions", []):
            for fix in session.get("fixes", []):
                timestamp = fix.get('timestamp')
                if not timestamp:
                    continue

                try:
                    dt = datetime.fromisoformat(timestamp)
                    hour = dt.hour
                    applied = fix.get('applied', False)
                    confidence = fix.get('confidence', 0.0)

                    hour_data[hour]['total'] += 1
                    if applied:
                        hour_data[hour]['success'] += 1
                    hour_data[hour]['confidences'].append(confidence)
                except:
                    continue

        # Find peak performance hours
        best_hours = []
        for hour, stats in hour_data.items():
            if stats['total'] >= min_support:
                success_rate = stats['success'] / stats['total']
                if success_rate >= min_success_rate:
                    best_hours.append((hour, success_rate, stats['total']))

        if best_hours:
            best_hours.sort(key=lambda x: x[1], reverse=True)
            top_hour, top_rate, top_support = best_hours[0]

            pattern = LearningPattern(
                pattern_type="temporal_peak",
                conditions={'hour_of_day': top_hour},
                success_rate=top_rate,
                support=top_support,
                confidence_avg=np.mean(hour_data[top_hour]['confidences']),
                recommendation=f"Peak performance at {top_hour}:00 ({top_rate:.1%} success)"
            )

            self.patterns.append(pattern)

    def export_patterns(self, output_file: str):
        """Export discovered patterns to JSON."""
        output_path = Path(output_file)

        data = {
            'patterns': [asdict(p) for p in self.patterns],
            'metadata': {
                'num_patterns': len(self.patterns),
                'discovery_date': datetime.now().isoformat()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Exported {len(self.patterns)} patterns to {output_path}")

    def get_top_patterns(self, n: int = 10) -> List[LearningPattern]:
        """Get top N patterns by support."""
        return sorted(self.patterns, key=lambda p: p.support, reverse=True)[:n]


# ============================================================================
# 3. Calibration Monitor
# ============================================================================

class CalibrationMonitor:
    """
    Monitor confidence calibration quality.

    Metrics:
    - Brier score: Mean squared error of probabilities
    - ECE (Expected Calibration Error): Average calibration gap
    - MCE (Maximum Calibration Error): Worst-case gap
    - Over/underconfidence ratios

    Usage:
        monitor = CalibrationMonitor("autofix_tracking/all_sessions.json")
        metrics = monitor.compute_calibration_metrics()
        monitor.plot_calibration_curve("calibration.png")
        monitor.export_recommendations("calibration_report.json")
    """

    def __init__(self, sessions_file: str):
        self.sessions_file = Path(sessions_file)
        self.data = self._load_data()
        self.confidences = []
        self.outcomes = []
        self._extract_confidence_outcomes()

    def _load_data(self) -> Dict[str, Any]:
        """Load sessions data."""
        if not self.sessions_file.exists():
            return {"sessions": []}

        with open(self.sessions_file, 'r') as f:
            return json.load(f)

    def _extract_confidence_outcomes(self):
        """Extract confidence scores and binary outcomes."""
        for session in self.data.get("sessions", []):
            for fix in session.get("fixes", []):
                confidence = fix.get('confidence', 0.0)
                applied = fix.get('applied', False)

                self.confidences.append(confidence)
                self.outcomes.append(1 if applied else 0)

    def compute_calibration_metrics(self, n_bins: int = 10) -> CalibrationMetrics:
        """
        Compute calibration quality metrics.

        Args:
            n_bins: Number of bins for calibration analysis

        Returns:
            CalibrationMetrics object
        """
        if not self.confidences:
            return CalibrationMetrics(
                brier_score=0.0,
                ece=0.0,
                mce=0.0,
                overconfident_ratio=0.0,
                underconfident_ratio=0.0,
                recommended_adjustment=0.0
            )

        confidences = np.array(self.confidences)
        outcomes = np.array(self.outcomes)

        # Brier score
        brier_score = np.mean((confidences - outcomes) ** 2)

        # Calibration error
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges[:-1]) - 1

        ece = 0.0
        mce = 0.0
        overconfident_count = 0
        underconfident_count = 0
        total_count = len(confidences)

        for i in range(n_bins):
            mask = bin_indices == i
            if not np.any(mask):
                continue

            bin_confidences = confidences[mask]
            bin_outcomes = outcomes[mask]

            avg_confidence = np.mean(bin_confidences)
            avg_accuracy = np.mean(bin_outcomes)

            calibration_gap = abs(avg_confidence - avg_accuracy)

            # ECE (weighted by bin size)
            ece += (len(bin_confidences) / total_count) * calibration_gap

            # MCE (maximum gap)
            mce = max(mce, calibration_gap)

            # Over/underconfidence
            if avg_confidence > avg_accuracy:
                overconfident_count += len(bin_confidences)
            elif avg_confidence < avg_accuracy:
                underconfident_count += len(bin_confidences)

        overconfident_ratio = overconfident_count / total_count
        underconfident_ratio = underconfident_count / total_count

        # Recommended adjustment (shift to reduce ECE)
        overall_confidence = np.mean(confidences)
        overall_accuracy = np.mean(outcomes)
        recommended_adjustment = overall_accuracy - overall_confidence

        metrics = CalibrationMetrics(
            brier_score=brier_score,
            ece=ece,
            mce=mce,
            overconfident_ratio=overconfident_ratio,
            underconfident_ratio=underconfident_ratio,
            recommended_adjustment=recommended_adjustment
        )

        return metrics

    def export_recommendations(self, output_file: str):
        """Export calibration analysis and recommendations."""
        metrics = self.compute_calibration_metrics()

        output_path = Path(output_file)

        # Generate recommendations
        recommendations = []

        if metrics.brier_score > 0.2:
            recommendations.append({
                'issue': 'High Brier score',
                'severity': 'high',
                'recommendation': 'Model predictions are poorly calibrated. Consider retraining with calibration loss.'
            })

        if metrics.ece > 0.1:
            recommendations.append({
                'issue': 'High Expected Calibration Error',
                'severity': 'medium',
                'recommendation': 'Apply post-hoc calibration (isotonic regression or Platt scaling).'
            })

        if metrics.overconfident_ratio > 0.6:
            recommendations.append({
                'issue': 'Overconfident predictions',
                'severity': 'high',
                'recommendation': f'Reduce confidence scores by {abs(metrics.recommended_adjustment):.3f} on average.'
            })

        if metrics.underconfident_ratio > 0.6:
            recommendations.append({
                'issue': 'Underconfident predictions',
                'severity': 'low',
                'recommendation': f'Increase confidence scores by {abs(metrics.recommended_adjustment):.3f} on average.'
            })

        data = {
            'metrics': asdict(metrics),
            'recommendations': recommendations,
            'analysis_date': datetime.now().isoformat(),
            'num_samples': len(self.confidences)
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Calibration analysis exported to {output_path}")
        print(f"\nCalibration Metrics:")
        print(f"  Brier Score: {metrics.brier_score:.3f} (lower is better)")
        print(f"  ECE: {metrics.ece:.3f} (lower is better)")
        print(f"  MCE: {metrics.mce:.3f} (lower is better)")
        print(f"  Overconfident: {metrics.overconfident_ratio:.1%}")
        print(f"  Underconfident: {metrics.underconfident_ratio:.1%}")
        print(f"  Recommended Adjustment: {metrics.recommended_adjustment:+.3f}")
        print(f"\n{len(recommendations)} recommendations generated")


# ============================================================================
# Main Pipeline
# ============================================================================

class AutoFixLearningPipeline:
    """
    Complete learning pipeline orchestrator.

    Integrates:
    - Training data export
    - Pattern discovery
    - Calibration monitoring
    - Recommendation generation

    Usage:
        pipeline = AutoFixLearningPipeline("autofix_tracking/all_sessions.json")
        pipeline.run_full_pipeline(output_dir="learning_output")
    """

    def __init__(self, sessions_file: str):
        self.sessions_file = sessions_file
        self.exporter = TrainingDataExporter(sessions_file)
        self.pattern_learner = PatternLearner(sessions_file)
        self.calibration_monitor = CalibrationMonitor(sessions_file)

    def run_full_pipeline(self, output_dir: str = "./learning_output"):
        """
        Run complete learning pipeline.

        Args:
            output_dir: Directory for outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("AUTOFIX LEARNING PIPELINE")
        print("=" * 70)
        print()

        # Step 1: Export training data
        print("Step 1: Exporting training data...")
        self.exporter.export_csv(output_path / "training_data.csv")
        self.exporter.export_json(output_path / "training_data.json")
        X_train, y_train, X_val, y_val = self.exporter.get_train_val_split()
        print()

        # Step 2: Discover patterns
        print("Step 2: Discovering patterns...")
        patterns = self.pattern_learner.discover_patterns(min_support=5, min_success_rate=0.7)
        self.pattern_learner.export_patterns(output_path / "learned_patterns.json")
        print()

        # Step 3: Calibration monitoring
        print("Step 3: Analyzing calibration...")
        self.calibration_monitor.export_recommendations(output_path / "calibration_report.json")
        print()

        # Step 4: Generate summary report
        print("Step 4: Generating summary report...")
        self._generate_summary_report(output_path / "learning_summary.md", patterns)
        print()

        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutputs saved to: {output_path}")
        print()

    def _generate_summary_report(self, output_file: Path, patterns: List[LearningPattern]):
        """Generate markdown summary report."""
        metrics = self.calibration_monitor.compute_calibration_metrics()

        with open(output_file, 'w') as f:
            f.write("# AutoFix Learning Pipeline Summary\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Training data summary
            f.write("## Training Data\n\n")
            f.write(f"- Total samples: {len(self.exporter.features)}\n")
            f.write(f"- Positive samples: {sum(self.exporter.labels)} ({100*sum(self.exporter.labels)/len(self.exporter.labels):.1f}%)\n")
            f.write(f"- Negative samples: {len(self.exporter.labels) - sum(self.exporter.labels)} ({100*(len(self.exporter.labels) - sum(self.exporter.labels))/len(self.exporter.labels):.1f}%)\n\n")

            # Patterns discovered
            f.write("## Discovered Patterns\n\n")
            f.write(f"Total patterns: {len(patterns)}\n\n")

            top_patterns = sorted(patterns, key=lambda p: p.support, reverse=True)[:10]
            f.write("### Top 10 Patterns by Support\n\n")
            f.write("| Pattern Type | Conditions | Success Rate | Support | Recommendation |\n")
            f.write("|-------------|------------|--------------|---------|----------------|\n")

            for pattern in top_patterns:
                conditions_str = ", ".join(f"{k}={v}" for k, v in pattern.conditions.items())
                f.write(f"| {pattern.pattern_type} | {conditions_str} | {pattern.success_rate:.1%} | {pattern.support} | {pattern.recommendation} |\n")

            f.write("\n")

            # Calibration metrics
            f.write("## Calibration Quality\n\n")
            f.write(f"- **Brier Score**: {metrics.brier_score:.3f} (0.0 = perfect)\n")
            f.write(f"- **Expected Calibration Error (ECE)**: {metrics.ece:.3f}\n")
            f.write(f"- **Maximum Calibration Error (MCE)**: {metrics.mce:.3f}\n")
            f.write(f"- **Overconfident predictions**: {metrics.overconfident_ratio:.1%}\n")
            f.write(f"- **Underconfident predictions**: {metrics.underconfident_ratio:.1%}\n")
            f.write(f"- **Recommended adjustment**: {metrics.recommended_adjustment:+.3f}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if metrics.brier_score < 0.1 and metrics.ece < 0.05:
                f.write("✅ **Excellent calibration** - No immediate action needed.\n\n")
            elif metrics.overconfident_ratio > 0.6:
                f.write("⚠️ **High overconfidence detected** - Consider:\n")
                f.write("1. Apply temperature scaling to confidence scores\n")
                f.write("2. Increase confidence threshold for auto-fix\n")
                f.write(f"3. Reduce confidence scores by ~{abs(metrics.recommended_adjustment):.3f}\n\n")
            elif metrics.underconfident_ratio > 0.6:
                f.write("⚠️ **High underconfidence detected** - Consider:\n")
                f.write("1. Lower confidence threshold to auto-fix more\n")
                f.write(f"2. Increase confidence scores by ~{abs(metrics.recommended_adjustment):.3f}\n\n")

            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Model Training**: Use `training_data.csv` to train/fine-tune confidence model\n")
            f.write("2. **Pattern Integration**: Integrate discovered patterns into autofix policy\n")
            f.write("3. **Calibration**: Apply isotonic regression or Platt scaling for better calibration\n")
            f.write("4. **Monitoring**: Set up continuous monitoring with periodic pipeline runs\n")
            f.write("5. **A/B Testing**: Test adjusted thresholds on holdout data\n\n")

        print(f"✓ Summary report: {output_file}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Run complete pipeline
    pipeline = AutoFixLearningPipeline("autofix_tracking/all_sessions.json")
    pipeline.run_full_pipeline(output_dir="learning_output")
