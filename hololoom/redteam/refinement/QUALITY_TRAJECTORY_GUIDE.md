# Quality Trajectory Tracking System - Complete Guide

**Status**: ✅ Production Ready (December 2025)
**Location**: `HoloLoom/redteam/refinement/`
**Performance**: <5ms per quality record, <50ms for full analysis
**Test Coverage**: 28+ comprehensive tests

## Overview

The Quality Trajectory Tracking System (CARTS - Collaborative Attack Refinement Tracking System) monitors attack quality evolution over time, enabling:

- **Plateau Detection**: Identify when refinement efforts yield diminishing returns
- **Regression Alerts**: Catch quality degradation before it becomes critical
- **Pattern Discovery**: Automatically extract successful refinement patterns
- **Strategy Analysis**: Compare effectiveness of different attack strategies
- **Actionable Insights**: Generate specific, implementable recommendations

## Core Concepts

### Quality Dimensions

Each attack is evaluated on 4 dimensions:

| Dimension | Weight | Meaning |
|-----------|--------|---------|
| **Effectiveness** | 40% | Attack achieves objective (0.0-1.0) |
| **Stealth** | 30% | Attack avoids detection (0.0-1.0) |
| **Reliability** | 20% | Attack succeeds consistently (0.0-1.0) |
| **Elegance** | 10% | Attack uses minimal resources (0.0-1.0) |

**Composite Quality Score** = 0.4×effectiveness + 0.3×stealth + 0.2×reliability + 0.1×elegance

### Trend Types

Three trend states guide refinement decisions:

| Trend | Indicator | Action |
|-------|-----------|--------|
| **IMPROVING** | Change > +0.01 per step | Continue current approach |
| **PLATEAU** | Change ≤ ±0.01 per step | Pivot or intensify efforts |
| **DEGRADING** | Change < -0.05 per step | Alert! Investigate root cause |

### Core Data Structures

#### StrategyTrajectory
Complete history of quality evolution for one attack strategy.

```python
trajectory = tracker.get_trajectory("obfuscation")

# Access complete history
print(f"Quality scores: {trajectory.scores}")
print(f"Current: {trajectory.final_quality:.1%}")
print(f"Best: {trajectory.max_quality:.1%}")
print(f"Average: {trajectory.avg_quality:.1%}")
print(f"Trend: {trajectory.trend}")
```

#### RefinementPattern
Discovered successful pattern for improving attack quality.

```python
pattern = discovered_patterns[0]

print(f"Type: {pattern.pattern_type}")
print(f"Improvement: +{pattern.improvement_pct:.1f}%")
print(f"Success rate: {pattern.success_rate:.1%}")
print(f"Confidence: {pattern.confidence:.1%}")
```

## API Reference

### 1. QualityTrajectoryTracker()
Main class for tracking attack quality evolution.

```python
from HoloLoom.redteam.refinement import QualityTrajectoryTracker

tracker = QualityTrajectoryTracker(
    plateau_threshold=0.01,           # Min change to not be plateau
    regression_threshold=-0.05,       # Threshold for regression alert
    window_size=10,                   # Rolling window for trend
    pattern_quality_threshold=0.3,    # Min improvement for pattern
    pattern_support_threshold=3       # Min occurrences for pattern
)
```

### 2. record_quality()
Record a quality score for a strategy.

```python
tracker.record_quality(
    strategy="obfuscation",
    score=0.75,  # 0.0-1.0
    metadata={
        "method": "string_concat",
        "iterations": 5,
        "execution_time_ms": 125
    }
)
```

**Performance**: <1ms per call

### 3. get_trajectory()
Get complete quality history for a strategy.

```python
trajectory = tracker.get_trajectory("obfuscation")

if trajectory:
    print(f"Iterations: {trajectory.iterations}")
    print(f"Trend: {trajectory.trend}")
    print(f"Improvement rate: {trajectory.improvement_rate:.4f} per iteration")
```

**Performance**: <0.5ms

### 4. detect_plateau()
Detect if a strategy has plateaued.

```python
if tracker.detect_plateau("obfuscation"):
    print("⚠️  Obfuscation strategy has plateaued")
    print("Consider: refining parameters, trying new techniques, or pivoting")
```

**Returns**: Boolean
**Performance**: <0.5ms

### 5. detect_regression()
Detect quality decline in a strategy.

```python
is_regressing, amount = tracker.detect_regression("obfuscation")

if is_regressing:
    print(f"🚨 Strategy regressing: {amount:.3f} per step")
    print(f"Quality drop: {abs(amount)*100:.1f}%")
```

**Returns**: Tuple of (is_regressing: bool, regression_amount: float)
**Performance**: <0.5ms

### 6. discover_patterns()
Find successful refinement patterns from history.

```python
patterns = tracker.discover_patterns()

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"  Improvement: +{pattern.improvement_pct:.1f}%")
    print(f"  Success rate: {pattern.success_rate:.1%}")
    print(f"  Confidence: {pattern.confidence:.1%}")
```

**Returns**: List[RefinementPattern]
**Performance**: <20ms (once per analysis cycle)

### 7. get_best_strategy() ⭐ NEW
Get the most effective attack strategy by average quality.

```python
best = tracker.get_best_strategy()

if best:
    trajectory = tracker.get_trajectory(best)
    print(f"Best strategy: {best}")
    print(f"  Average quality: {trajectory.avg_quality:.1%}")
    print(f"  Current quality: {trajectory.final_quality:.1%}")
    print(f"  Max achieved: {trajectory.max_quality:.1%}")
```

**Returns**: Optional[str] - Strategy name or None if no data
**Performance**: <0.5ms
**Use Case**: Strategy selection, resource allocation

### 8. get_improvement_rate() ⭐ NEW
Get quality improvement per iteration for a strategy.

```python
rate = tracker.get_improvement_rate("obfuscation")

if rate > 0.01:
    print(f"✅ Strong improvement: +{rate:.3f} per iteration")
    print(f"   At this rate, +0.20 quality in {0.20/rate:.0f} iterations")
elif rate < 0:
    print(f"⚠️  Regressing: {rate:.3f} per iteration")
else:
    print(f"📊 Plateau: minimal change per iteration")
```

**Returns**: float - Quality points per iteration
**Performance**: <0.5ms
**Use Case**: Progress tracking, effort allocation

### 9. analyze_patterns() ⭐ NEW
Comprehensive analysis of patterns across all strategies.

```python
analysis = tracker.analyze_patterns()

# Summary
print(f"Total patterns: {analysis['total_patterns']}")
print(f"Strategies tracked: {len(analysis['strategy_effectiveness'])}")

# Top patterns
print("\nTop patterns by impact:")
for i, pattern in enumerate(analysis['top_patterns'], 1):
    print(f"  {i}. {pattern.description}")
    print(f"     Improvement: +{pattern.improvement_pct:.1f}%")

# Strategy comparison
print("\nStrategy effectiveness:")
for strat, metrics in analysis['strategy_effectiveness'].items():
    print(f"  {strat}:")
    print(f"    Current: {metrics['current_quality']:.1%}")
    print(f"    Stability: {metrics['stability']:.1%}")
    print(f"    Max achieved: {metrics['max_quality']:.1%}")

# Recommendations
print("\nRecommendations:")
for rec in analysis['recommendations']:
    print(f"  - {rec}")
```

**Returns**: Dictionary with complete analysis
**Performance**: <50ms
**Use Case**: Strategic planning, progress reports

## Usage Examples

### Basic Quality Tracking

```python
from HoloLoom.redteam.refinement import QualityTrajectoryTracker

# Initialize tracker
tracker = QualityTrajectoryTracker()

# Record quality scores for different strategies
strategies = ["obfuscation", "mutation", "chaining"]
qualities = [
    [0.65, 0.68, 0.70, 0.71, 0.72, 0.72, 0.72],  # obfuscation
    [0.60, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80],  # mutation
    [0.55, 0.57, 0.59, 0.58, 0.57, 0.57, 0.56]   # chaining (regressing)
]

for strategy, quality_list in zip(strategies, qualities):
    for score in quality_list:
        tracker.record_quality(strategy, score)

# Analyze results
print("📊 Quality Analysis")
print("=" * 50)

for strategy in strategies:
    trajectory = tracker.get_trajectory(strategy)
    is_plateau = tracker.detect_plateau(strategy)
    is_regressing, regression_amt = tracker.detect_regression(strategy)
    trend = tracker.get_trend(strategy)

    print(f"\n{strategy.upper()}")
    print(f"  Current: {trajectory.final_quality:.1%}")
    print(f"  Average: {trajectory.avg_quality:.1%}")
    print(f"  Trend: {trend}")

    if is_plateau:
        print(f"  ⚠️  PLATEAUED - Consider pivoting")
    elif is_regressing:
        print(f"  🚨 REGRESSING: {regression_amt:.3f}/step")
    else:
        print(f"  ✅ IMPROVING")
```

### Strategy Selection and Comparison

```python
# Find best strategy
best_strategy = tracker.get_best_strategy()
print(f"🏆 Best strategy: {best_strategy}")

# Compare improvement rates
print("\nImprovement Rates:")
for strategy in tracker.get_all_trajectories().keys():
    rate = tracker.get_improvement_rate(strategy)
    print(f"  {strategy}: +{rate:.4f} per iteration")

# Get detailed analysis
analysis = tracker.analyze_patterns()

print(f"\n📈 Analysis Summary:")
print(f"  Total patterns discovered: {analysis['total_patterns']}")
print(f"  Recent success rate: {analysis['temporal_analysis']['recent_success_rate']:.1%}")

# View recommendations
print(f"\n💡 Recommendations:")
for rec in analysis['recommendations']:
    print(f"  • {rec}")
```

### Monitoring and Alerting

```python
import time

# Continuous monitoring
tracker = QualityTrajectoryTracker()

strategies_to_monitor = ["obfuscation", "mutation"]

for i in range(20):
    # Simulate quality scores
    for strategy in strategies_to_monitor:
        score = simulate_attack_quality(strategy)
        tracker.record_quality(strategy, score)

    # Check for problems
    for strategy in strategies_to_monitor:
        is_regressing, amt = tracker.detect_regression(strategy)
        if is_regressing:
            print(f"⚠️  ALERT: {strategy} regressing by {abs(amt):.3f}/step")

        is_plateau = tracker.detect_plateau(strategy)
        if is_plateau and i > 10:  # After 10 iterations
            print(f"📊 INFO: {strategy} plateaued at {tracker.get_trajectory(strategy).final_quality:.1%}")

    time.sleep(0.5)
```

### Complete Analysis Report

```python
def generate_quality_report(tracker: QualityTrajectoryTracker) -> str:
    """Generate comprehensive quality analysis report."""

    # Get all data
    analysis = tracker.analyze_patterns()
    metrics = tracker.export_metrics()
    best = tracker.get_best_strategy()

    # Build report
    report = []
    report.append("=" * 70)
    report.append("QUALITY TRAJECTORY ANALYSIS REPORT")
    report.append("=" * 70)

    # Overall metrics
    report.append(f"\nOVERALL METRICS")
    report.append(f"  Total strategies: {metrics['overall']['total_strategies']}")
    report.append(f"  Total patterns: {analysis['total_patterns']}")
    report.append(f"  Average quality: {metrics['overall']['avg_quality']:.1%}")
    report.append(f"  Best strategy: {best} ({metrics['overall']['best_strategy']}")

    # Strategy details
    report.append(f"\nSTRATEGY DETAILS")
    report.append("-" * 70)
    for strategy, metrics_dict in metrics['strategies'].items():
        report.append(f"\n  {strategy.upper()}")
        report.append(f"    Current:  {metrics_dict['current_quality']:.1%}")
        report.append(f"    Average:  {metrics_dict['avg_quality']:.1%}")
        report.append(f"    Max:      {metrics_dict['max_quality']:.1%}")
        report.append(f"    Trend:    {metrics_dict['trend']}")
        report.append(f"    Improvement: +{metrics_dict['improvement_rate']:.4f}/iteration")

    # Top patterns
    report.append(f"\nTOP PATTERNS")
    report.append("-" * 70)
    for i, pattern in enumerate(analysis['top_patterns'], 1):
        report.append(f"\n  {i}. {pattern.description}")
        report.append(f"     Improvement: +{pattern.improvement_pct:.1f}%")
        report.append(f"     Success rate: {pattern.success_rate:.1%}")
        report.append(f"     Confidence: {pattern.confidence:.1%}")

    # Recommendations
    report.append(f"\nRECOMMENDATIONS")
    report.append("-" * 70)
    for rec in analysis['recommendations']:
        report.append(f"  • {rec}")

    return "\n".join(report)

# Generate and print report
print(generate_quality_report(tracker))
```

## Performance Characteristics

| Operation | Latency | Complexity | When |
|-----------|---------|-----------|------|
| `record_quality()` | <1ms | O(1) | Every attack iteration |
| `get_trajectory()` | <0.5ms | O(1) | Anytime |
| `detect_plateau()` | <0.5ms | O(n) | Periodic checks |
| `detect_regression()` | <0.5ms | O(n) | Periodic checks |
| `discover_patterns()` | <20ms | O(p) | Once per cycle |
| `analyze_patterns()` | <50ms | O(p+s) | Once per report |
| **Total per-query overhead** | **<5ms** | | Production use |

**Legend**: p = patterns, s = strategies, n = trajectory length

## Configuration Guide

```python
tracker = QualityTrajectoryTracker(
    # Plateau threshold (default: 0.01)
    # - Minimum change per step to not be considered plateau
    # - Higher = less sensitive to improvements
    # - Lower = more sensitive (faster plateau detection)
    plateau_threshold=0.01,

    # Regression threshold (default: -0.05)
    # - Minimum decline per step to trigger regression alert
    # - More negative = less sensitive to decline
    # - Less negative = more sensitive (faster regression alert)
    regression_threshold=-0.05,

    # Window size (default: 10)
    # - Number of recent scores to analyze for trend
    # - Larger = smoother trends, less noise
    # - Smaller = faster response to changes
    window_size=10,

    # Pattern quality threshold (default: 0.3)
    # - Minimum improvement magnitude for pattern detection
    # - Higher = only strong patterns detected
    # - Lower = more patterns, some may be weak
    pattern_quality_threshold=0.3,

    # Pattern support threshold (default: 3)
    # - Minimum occurrences for pattern validity
    # - Higher = more reliable patterns
    # - Lower = patterns discovered sooner
    pattern_support_threshold=3
)
```

## Integration with Refinement Pipeline

```python
from HoloLoom.redteam.refinement import QualityTrajectoryTracker
from HoloLoom.recursive.advanced_refinement import AdvancedRefiner

# Initialize systems
tracker = QualityTrajectoryTracker()
refiner = AdvancedRefiner(orchestrator)

# Refinement loop
for iteration in range(100):
    # Record current quality
    tracker.record_quality(
        strategy="refinement",
        score=current_attack.quality,
        metadata={"iteration": iteration}
    )

    # Detect issues
    if tracker.detect_plateau("refinement"):
        print("⚠️  Plateau detected - switching refinement strategy")
        # Use advanced refiner
        result = await refiner.refine(
            query=attack_prompt,
            strategy=RefinementStrategy.ELEGANCE
        )
        current_attack = result.final_spacetime

    is_regressing, amt = tracker.detect_regression("refinement")
    if is_regressing:
        print(f"🚨 Regression detected: {amt}")
        # Rollback or take corrective action
        continue_iteration = False

    # Proceed with attack
    execute_attack(current_attack)

# Final analysis
analysis = tracker.analyze_patterns()
print(f"Completed: {len(analysis['recommendations'])} improvements identified")
```

## Best Practices

### 1. Regular Monitoring
Record quality scores frequently for responsive trend detection:
```python
# Good: Record after each iteration
tracker.record_quality(strategy, quality)

# Avoid: Batch recording loses temporal information
scores = [...]
for score in scores:
    tracker.record_quality(strategy, score)  # Still good, but do more often
```

### 2. Strategy Naming
Use consistent, descriptive strategy names:
```python
# Good
tracker.record_quality("obfuscation_string_concat", quality)
tracker.record_quality("obfuscation_regex_encode", quality)

# Avoid
tracker.record_quality("method1", quality)
tracker.record_quality("method2", quality)
```

### 3. Metadata Tracking
Include relevant context for pattern discovery:
```python
tracker.record_quality(
    strategy="obfuscation",
    score=quality,
    metadata={
        "technique": "string_concat",
        "parameters": {"iterations": 3},
        "environment": "test_server",
        "timestamp_wall_clock": datetime.now().isoformat()
    }
)
```

### 4. Threshold Tuning
Adjust thresholds based on your domain:
```python
# For stable, slow-improving attacks
tracker = QualityTrajectoryTracker(
    plateau_threshold=0.002,  # Very conservative
    window_size=20           # Smooth trends
)

# For volatile, fast-improving attacks
tracker = QualityTrajectoryTracker(
    plateau_threshold=0.05,  # Aggressive
    window_size=5            # Responsive
)
```

### 5. Periodic Analysis
Run full analysis at regular intervals:
```python
import asyncio

async def monitoring_loop(tracker: QualityTrajectoryTracker):
    while True:
        # Record scores (continuous)
        tracker.record_quality(strategy, quality)

        # Analyze periodically (every 10 iterations)
        if iteration % 10 == 0:
            analysis = tracker.analyze_patterns()
            # Use recommendations
            for rec in analysis['recommendations']:
                apply_recommendation(rec)

        await asyncio.sleep(0.1)
```

## Testing and Validation

```python
import pytest
from HoloLoom.redteam.refinement import QualityTrajectoryTracker

def test_quality_tracking():
    """Test basic quality recording and retrieval."""
    tracker = QualityTrajectoryTracker()

    tracker.record_quality("test", 0.5)
    trajectory = tracker.get_trajectory("test")

    assert trajectory is not None
    assert trajectory.final_quality == 0.5
    assert trajectory.avg_quality == 0.5

def test_plateau_detection():
    """Test plateau detection."""
    tracker = QualityTrajectoryTracker(plateau_threshold=0.01)

    # Record stable scores
    for _ in range(10):
        tracker.record_quality("test", 0.75)

    assert tracker.detect_plateau("test") == True

def test_improvement_rate():
    """Test improvement rate calculation."""
    tracker = QualityTrajectoryTracker()

    # Linear improvement: 0.0 → 1.0 in 10 steps
    for i in range(11):
        tracker.record_quality("test", i / 10.0)

    rate = tracker.get_improvement_rate("test")
    assert rate == pytest.approx(0.1, abs=0.01)

def test_pattern_discovery():
    """Test pattern discovery."""
    tracker = QualityTrajectoryTracker()

    # Record improving quality
    for score in [0.5, 0.6, 0.7, 0.8]:
        tracker.record_quality("test", score)

    patterns = tracker.discover_patterns()
    assert len(patterns) > 0
    assert patterns[0].improvement > 0
```

## Troubleshooting

### Problem: All strategies show plateau
**Cause**: Quality thresholds too tight
**Solution**: Reduce `plateau_threshold`:
```python
tracker = QualityTrajectoryTracker(plateau_threshold=0.001)
```

### Problem: Patterns never discovered
**Cause**: Not enough history or thresholds too high
**Solution**: Lower pattern thresholds:
```python
tracker = QualityTrajectoryTracker(
    pattern_quality_threshold=0.1,
    pattern_support_threshold=1
)
```

### Problem: Trends too noisy
**Cause**: Window size too small
**Solution**: Increase window size:
```python
tracker = QualityTrajectoryTracker(window_size=20)
```

## See Also

- [HoloLoom Recursive Learning](../../../recursive/README.md)
- [Attack Refinement Strategies](../strategies.py)
- [Quality Metrics](quality_trajectory.py)
- [CARTS Documentation](../ROADMAP.md)

---

**Last Updated**: December 2025
**Maintainer**: HoloLoom Redteam
**License**: Apache 2.0
