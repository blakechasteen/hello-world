# AutoFix Learning Pipeline

**Date**: November 16, 2025
**Status**: ✅ Production Ready
**Location**: `/autofix_learning_pipeline.py`

## Overview

The AutoFix Learning Pipeline is a continuous learning system that improves autofix quality from historical outcomes. It analyzes tracking data to discover patterns, calibrate confidence scores, and generate recommendations for model improvement.

## Architecture

```
AutoFix Learning Pipeline
│
├─ TrainingDataExporter
│  ├─ Feature extraction
│  ├─ Train/validation splits
│  └─ Multiple export formats (CSV, JSON)
│
├─ PatternLearner
│  ├─ Category → Strategy patterns
│  ├─ Severity → Outcome patterns
│  ├─ Confidence calibration patterns
│  └─ Temporal patterns
│
├─ CalibrationMonitor
│  ├─ Brier score computation
│  ├─ Expected Calibration Error (ECE)
│  ├─ Maximum Calibration Error (MCE)
│  └─ Over/underconfidence analysis
│
└─ AutoFixLearningPipeline (Orchestrator)
   ├─ Full pipeline execution
   ├─ Summary report generation
   └─ Recommendation synthesis
```

## Components

### 1. TrainingDataExporter

**Purpose**: Export tracking data in ML-ready formats for model training.

**Features**:
- Feature engineering (categorical encoding, temporal features)
- Train/validation splits with stratification
- Multiple output formats (CSV, JSON, NumPy arrays)
- Balanced sampling for class imbalance

**Usage**:
```python
from autofix_learning_pipeline import TrainingDataExporter

exporter = TrainingDataExporter("autofix_tracking/all_sessions.json")

# Export to CSV
exporter.export_csv("training_data.csv")

# Get train/validation split
X_train, y_train, X_val, y_val = exporter.get_train_val_split(
    val_ratio=0.2,
    stratify=True,
    random_seed=42
)
```

**Extracted Features**:
| Feature | Type | Description |
|---------|------|-------------|
| `category` | Categorical | Fix category (dead_code, hardcoded_values, etc.) |
| `severity` | Categorical | Severity level (low, medium, high, critical) |
| `strategy` | Categorical | Fix strategy (ast, template, manual) |
| `confidence` | Numerical | Predicted confidence (0.0-1.0) |
| `line_number` | Numerical | Line number in file |
| `code_length` | Numerical | Length of code snippet |
| `diff_length` | Numerical | Length of diff |
| `hour_of_day` | Numerical | Hour when fix was applied (0-23) |
| `day_of_week` | Numerical | Day of week (0=Monday, 6=Sunday) |
| `validation_passed` | Binary | Whether validation passed |
| `has_errors` | Binary | Whether errors occurred |
| `has_warnings` | Binary | Whether warnings occurred |
| `label` | Binary | **Target**: 1=applied, 0=failed |

### 2. PatternLearner

**Purpose**: Discover success/failure patterns from fix outcomes.

**Patterns Discovered**:

1. **Category → Strategy Patterns**
   - Which strategies work best for each category
   - Example: "AST strategy for dead_code: 95% success rate"

2. **Severity → Outcome Patterns**
   - Which severity levels are reliably auto-fixable
   - Example: "LOW severity: reliable auto-fix (92% success)"

3. **Confidence Calibration Patterns**
   - Whether predicted confidence matches actual outcomes
   - Example: "Confidence 0.8-0.9: overconfident (actual 72%)"

4. **Temporal Patterns**
   - Time-based trends in fix success
   - Example: "Peak performance at 14:00 (86% success)"

**Usage**:
```python
from autofix_learning_pipeline import PatternLearner

learner = PatternLearner("autofix_tracking/all_sessions.json")

# Discover patterns
patterns = learner.discover_patterns(
    min_support=5,        # Minimum 5 instances
    min_success_rate=0.7  # 70% success threshold
)

# Export patterns
learner.export_patterns("learned_patterns.json")

# Get top patterns
top_10 = learner.get_top_patterns(n=10)

for pattern in top_10:
    print(f"{pattern.pattern_type}: {pattern.recommendation}")
    print(f"  Success Rate: {pattern.success_rate:.1%}")
    print(f"  Support: {pattern.support} instances")
```

**Pattern Data Structure**:
```python
@dataclass
class LearningPattern:
    pattern_type: str           # Type of pattern
    conditions: Dict[str, Any]  # When pattern applies
    success_rate: float         # 0.0-1.0
    support: int                # Number of instances
    confidence_avg: float       # Average confidence
    recommendation: str         # Action recommendation
```

### 3. CalibrationMonitor

**Purpose**: Monitor confidence calibration quality and provide feedback.

**Metrics**:

| Metric | Description | Good Value | Bad Value |
|--------|-------------|------------|-----------|
| **Brier Score** | Mean squared error of probabilities | <0.1 | >0.2 |
| **ECE** | Expected Calibration Error (average gap) | <0.05 | >0.1 |
| **MCE** | Maximum Calibration Error (worst gap) | <0.1 | >0.2 |
| **Overconfident Ratio** | Fraction of overconfident predictions | <0.3 | >0.6 |
| **Underconfident Ratio** | Fraction of underconfident predictions | <0.3 | >0.6 |

**Usage**:
```python
from autofix_learning_pipeline import CalibrationMonitor

monitor = CalibrationMonitor("autofix_tracking/all_sessions.json")

# Compute metrics
metrics = monitor.compute_calibration_metrics(n_bins=10)

print(f"Brier Score: {metrics.brier_score:.3f}")
print(f"ECE: {metrics.ece:.3f}")
print(f"Recommended Adjustment: {metrics.recommended_adjustment:+.3f}")

# Export recommendations
monitor.export_recommendations("calibration_report.json")
```

**Calibration Issues Detected**:

1. **Overconfidence** (predicted > actual)
   - Recommendation: Reduce confidence scores or increase threshold
   - Example: "Predicted 85% but actual 72% - overconfident by 13%"

2. **Underconfidence** (predicted < actual)
   - Recommendation: Increase confidence scores or lower threshold
   - Example: "Predicted 70% but actual 88% - underconfident by 18%"

3. **High Brier Score** (poor probability predictions)
   - Recommendation: Retrain model with calibration loss
   - Example: "Brier score 0.25 - model needs recalibration"

### 4. AutoFixLearningPipeline (Orchestrator)

**Purpose**: Run complete learning pipeline and generate comprehensive report.

**Usage**:
```python
from autofix_learning_pipeline import AutoFixLearningPipeline

pipeline = AutoFixLearningPipeline("autofix_tracking/all_sessions.json")

# Run full pipeline
pipeline.run_full_pipeline(output_dir="learning_output")
```

**Outputs** (in `learning_output/`):
- `training_data.csv` - ML-ready training data
- `training_data.json` - JSON format with metadata
- `learned_patterns.json` - Discovered patterns
- `calibration_report.json` - Calibration analysis + recommendations
- `learning_summary.md` - Comprehensive summary report

## Integration Workflow

### Phase 1: Initial Learning (Batch)

```python
# 1. Run autofix with tracking enabled
from autofix_tracker import AutoFixTracker

tracker = AutoFixTracker()
session_id = tracker.start_session(
    max_files=100,
    categories=['dead_code', 'hardcoded_values', 'missing_docstrings'],
    confidence_threshold=0.85
)

# ... apply fixes and track results ...

tracker.end_session()
tracker.export_json("autofix_tracking/all_sessions.json")

# 2. Run learning pipeline
from autofix_learning_pipeline import AutoFixLearningPipeline

pipeline = AutoFixLearningPipeline("autofix_tracking/all_sessions.json")
pipeline.run_full_pipeline(output_dir="learning_output")

# 3. Review recommendations and adjust policy
# - Check calibration_report.json for confidence adjustments
# - Check learned_patterns.json for strategy improvements
# - Update AutofixPolicy thresholds based on recommendations
```

### Phase 2: Continuous Learning (Incremental)

```python
# Run learning pipeline after each autofix session
from autofix_learning_pipeline import AutoFixLearningPipeline

def post_autofix_hook(session_file: str):
    """Called after each autofix session."""
    # Run learning pipeline
    pipeline = AutoFixLearningPipeline(session_file)
    pipeline.run_full_pipeline(output_dir=f"learning_output/{datetime.now():%Y%m%d}")

    # Get recommendations
    calibration_monitor = pipeline.calibration_monitor
    metrics = calibration_monitor.compute_calibration_metrics()

    # Auto-adjust thresholds if needed
    if metrics.overconfident_ratio > 0.7:
        print("⚠️  High overconfidence detected - increasing threshold")
        # Increase min_confidence_auto by 0.05
        return {'action': 'increase_threshold', 'amount': 0.05}

    elif metrics.underconfident_ratio > 0.7:
        print("⚠️  High underconfidence detected - decreasing threshold")
        # Decrease min_confidence_auto by 0.05
        return {'action': 'decrease_threshold', 'amount': 0.05}

    return {'action': 'no_change'}

# Integrate into autofix workflow
tracker.end_session()
recommendations = post_autofix_hook("autofix_tracking/all_sessions.json")
```

### Phase 3: Model Retraining

```python
# Use exported training data to retrain confidence model

from autofix_learning_pipeline import TrainingDataExporter
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# 1. Load training data
exporter = TrainingDataExporter("autofix_tracking/all_sessions.json")
X_train, y_train, X_val, y_val = exporter.get_train_val_split()

# 2. Convert to DataFrame for easier handling
train_df = pd.DataFrame(X_train)
val_df = pd.DataFrame(X_val)

# 3. Encode categorical features
from sklearn.preprocessing import LabelEncoder

categorical_features = ['category', 'severity', 'strategy']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    train_df[feature] = le.fit_transform(train_df[feature])
    val_df[feature] = le.transform(val_df[feature])
    label_encoders[feature] = le

# 4. Train classifier
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf.fit(train_df, y_train)

# 5. Calibrate probabilities (fix overconfidence)
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
calibrated_clf.fit(train_df, y_train)

# 6. Evaluate on validation set
y_pred = calibrated_clf.predict_proba(val_df)[:, 1]
val_accuracy = np.mean((y_pred > 0.5) == y_val)

print(f"Validation Accuracy: {val_accuracy:.1%}")

# 7. Save model
import joblib
joblib.dump(calibrated_clf, 'models/autofix_confidence_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
```

## Performance Characteristics

| Operation | Latency | Memory | Notes |
|-----------|---------|--------|-------|
| Feature Extraction | ~1ms per fix | Low | Scales linearly |
| Pattern Discovery | ~50ms per session | Low | Depends on # patterns |
| Calibration Metrics | ~10ms | Low | Depends on # samples |
| Full Pipeline | ~100-500ms | Medium | For 100-1000 fixes |

## Best Practices

### 1. Minimum Data Requirements

- **Initial Learning**: ≥50 fixes (minimum for meaningful patterns)
- **Production Learning**: ≥200 fixes (better statistical significance)
- **Model Retraining**: ≥500 fixes (sufficient for ML training)

### 2. Pattern Discovery Thresholds

```python
# Conservative (high confidence patterns only)
patterns = learner.discover_patterns(min_support=20, min_success_rate=0.85)

# Balanced (default)
patterns = learner.discover_patterns(min_support=10, min_success_rate=0.75)

# Aggressive (explore more patterns)
patterns = learner.discover_patterns(min_support=5, min_success_rate=0.65)
```

### 3. Calibration Monitoring Schedule

- **Development**: After each autofix session
- **Staging**: Daily (aggregate all sessions)
- **Production**: Weekly (large-scale analysis)

### 4. Threshold Adjustment Strategy

```python
# Auto-adjust thresholds based on calibration

metrics = monitor.compute_calibration_metrics()

if metrics.overconfident_ratio > 0.7:
    # Increase threshold by 0.05
    new_threshold = current_threshold + 0.05

elif metrics.underconfident_ratio > 0.7:
    # Decrease threshold by 0.05
    new_threshold = current_threshold - 0.05

else:
    # No change needed
    new_threshold = current_threshold

# Clamp to reasonable range [0.70, 0.95]
new_threshold = max(0.70, min(0.95, new_threshold))
```

## Roadmap

### Phase 1: Batch Learning (✅ Complete)
- ✅ Training data export
- ✅ Pattern discovery
- ✅ Calibration monitoring
- ✅ Report generation

### Phase 2: Incremental Learning (Next)
- Online pattern updates
- Drift detection
- Automatic threshold adjustment
- A/B testing framework

### Phase 3: Advanced ML (Future)
- Deep learning confidence models
- Multi-task learning (confidence + strategy selection)
- Active learning (query most informative fixes)
- Federated learning (learn across departments)

### Phase 4: Production Integration (Future)
- Real-time monitoring dashboard
- Prometheus metrics export
- Slack/email alerts on calibration drift
- Automatic model retraining pipeline

## Troubleshooting

### Issue: "No patterns discovered"

**Cause**: Insufficient data or thresholds too strict

**Solution**:
```python
# Lower thresholds
patterns = learner.discover_patterns(min_support=3, min_success_rate=0.5)
```

### Issue: "High Brier score (>0.3)"

**Cause**: Poor confidence predictions

**Solution**:
1. Retrain confidence model with more data
2. Apply post-hoc calibration (isotonic regression)
3. Use simpler model (fewer features)

### Issue: "High overconfidence (>80%)"

**Cause**: Model systematically overestimates confidence

**Solution**:
1. Apply temperature scaling: `confidence = confidence / temperature`
2. Increase confidence threshold by 0.05-0.10
3. Retrain with calibration loss

### Issue: "All patterns have low support"

**Cause**: Not enough data

**Solution**:
1. Collect more autofix sessions (aim for ≥200 fixes)
2. Lower `min_support` threshold temporarily
3. Aggregate across multiple sessions

## Examples

See `examples/demo_learning_pipeline.py` for complete working examples.

## References

- **Calibration**: Guo et al. (2017) - "On Calibration of Modern Neural Networks"
- **Pattern Mining**: Agrawal & Srikant (1994) - "Fast Algorithms for Mining Association Rules"
- **Continuous Learning**: Losing et al. (2018) - "Incremental On-line Learning"

## Support

For questions or issues:
1. Check [AUTOFIX_LEARNING_PIPELINE.md](AUTOFIX_LEARNING_PIPELINE.md)
2. Review example usage in `examples/`
3. File issue on GitHub

---

**Last Updated**: November 16, 2025
**Version**: 1.0.0
**Author**: mythRL Team
