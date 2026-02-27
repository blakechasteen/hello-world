# CARTS Visualization Usage Guide

## Overview

The CARTS visualization foundation provides **Tufte-style attack trajectory visualization** for red team analytics. It follows Edward Tufte's principles:

- **Maximize data-ink ratio** - Show meaning first
- **Minimize chartjunk** - No unnecessary decoration
- **Small multiples** - Compare strategies side-by-side
- **High information density** - Pack metrics efficiently
- **Zero dependencies** - Pure HTML/CSS/SVG output

## Quick Start

### Basic Usage (Simplest)

```python
from hololoom.redteam.visualization import render_attack_trajectory

# Data from your attack runs
strategies = ["prompt_injection", "jailbreak", "context_overflow"]
success_rates = [0.65, 0.42, 0.28]
attack_counts = [10, 10, 10]
bypass_counts = [6, 4, 3]

# Render visualization
html = render_attack_trajectory(
    strategies, success_rates, attack_counts, bypass_counts,
    title="Red Team Attack Analysis",
    subtitle="Phase 1: Initial attack vectors (2025-12-05)"
)

# Save to file
with open("attack_trajectory.html", "w") as f:
    f.write(html)

# Open in browser
import webbrowser
webbrowser.open("attack_trajectory.html")
```

### Advanced Usage (Full Control)

```python
from hololoom.redteam.visualization import (
    AttackPoint,
    AttackTrajectoryRenderer
)

# Create detailed attack points with metadata
points = [
    AttackPoint(
        index=0,
        strategy="prompt_injection",
        success_rate=0.60,
        attack_count=10,
        bypass_count=6,
        avg_severity=0.75,
        metadata={"batch": 1, "model": "gpt-4"}
    ),
    AttackPoint(
        index=1,
        strategy="prompt_injection",
        success_rate=0.65,
        attack_count=10,
        bypass_count=7,
        avg_severity=0.78,
        metadata={"batch": 2, "model": "gpt-4"}
    ),
    AttackPoint(
        index=2,
        strategy="jailbreak",
        success_rate=0.40,
        attack_count=10,
        bypass_count=4,
        avg_severity=0.65,
        metadata={"batch": 1, "model": "gpt-4"}
    ),
    AttackPoint(
        index=3,
        strategy="jailbreak",
        success_rate=0.42,
        attack_count=10,
        bypass_count=4,
        avg_severity=0.68,
        metadata={"batch": 2, "model": "gpt-4"}
    ),
]

# Create renderer with custom settings
renderer = AttackTrajectoryRenderer(
    detect_anomalies=True,           # Enable anomaly detection
    show_strategy_breakdown=True,    # Show strategy comparison
    max_width=1200,                  # Chart width
    chart_height=300                 # Chart height
)

# Render
html = renderer.render(
    points,
    title="CARTS Attack Trajectory Analysis",
    subtitle="Multi-strategy red team campaign (2025-12-05)"
)

# Save and view
with open("detailed_analysis.html", "w") as f:
    f.write(html)
```

## Data Classes

### AttackPoint

Represents a single measurement in the attack trajectory.

```python
from hololoom.redteam.visualization import AttackPoint

point = AttackPoint(
    index=0,                    # Sequence position
    strategy="prompt_injection", # Attack strategy name
    success_rate=0.65,          # 0.0-1.0 (1.0 = all bypassed)
    attack_count=10,            # Total attacks in period
    bypass_count=6,             # Successful bypasses
    avg_severity=0.75,          # 0.0-1.0 average severity
    timestamp=1701726000.0,     # Optional Unix timestamp
    metadata={                  # Optional context
        "batch": 1,
        "model": "gpt-4",
        "temperature": 0.7
    }
)
```

**Validation**:
- `success_rate` must be 0.0-1.0
- `avg_severity` must be 0.0-1.0
- `bypass_count` cannot exceed `attack_count`

### StrategyMetrics

Aggregated metrics for a strategy (automatically computed).

```python
from hololoom.redteam.visualization import StrategyMetrics

metrics = StrategyMetrics(
    strategy="prompt_injection",
    total_attacks=100,
    total_bypasses=65,
    bypass_rate=0.65,           # Computed: total_bypasses / total_attacks
    avg_severity=0.75,
    trend="degrading",          # "improving", "degrading", or "stable"
    sparkline_data=[0.60, 0.65, 0.70, 0.72],  # Success rates over time
    min_success=0.60,           # Computed automatically
    max_success=0.72,           # Computed automatically
    points_count=4              # Computed automatically
)
```

## Visualization Features

### Main Chart

**Line chart** showing attack success rate over time.

- **Y-axis**: Success rate 0-100%
- **X-axis**: Attack sequence/time
- **Line color**: Green (trend indicator)
- **Points**:
  - Red circle: Success rate >50% (attack succeeded)
  - Green circle: Success rate <50% (attack blocked)
- **Reference line**: 50% threshold (dashed gray)
- **Grid**: Subtle background grid for readability

### Anomaly Detection

Automatically detects 5 types of anomalies:

1. **SUDDEN_BREAKTHROUGH** (Red diamond)
   - Success rate spike >20% in single step
   - Indicates new attack vector works

2. **SUSTAINED_SUCCESS** (Dark red diamond)
   - High success (>60%) for 3+ consecutive points
   - Indicates persistent vulnerability

3. **PLATEAU** (Orange diamond)
   - Success rate stable ±5% for 5+ points
   - Indicates plateau in attack effectiveness

4. **DEGRADATION** (Green diamond)
   - Success rate drops >15%
   - **Good**: Guardrails improving

5. **STRATEGY_SHIFT** (Blue diamond)
   - Strategy changes mid-sequence
   - Tactical shift in attack approach

### Statistics Panel

Key metrics with trend indicators:

| Metric | Purpose |
|--------|---------|
| Overall Bypass Rate | Attack success percentage |
| Total Attacks | Number of attacks executed |
| Successful Bypasses | Number that bypassed guardrails |
| Average Severity | Mean severity of successful attacks |
| Peak Success Rate | Maximum success observed |
| Success Volatility | Std dev of success rates |

**Trend indicators**:
- ↓ Improving (lower success rate = better)
- ↑ Degrading (higher success rate = worse)
- → Stable (no significant change)
- — Neutral (not applicable)

### Strategy Comparison (Small Multiples)

Table showing each strategy with:

| Column | Content |
|--------|---------|
| Strategy | Attack strategy name |
| Bypass Rate | Success percentage (color-coded) |
| Attacks | Total attacks for strategy |
| Trend | Inline sparkline showing trajectory |
| Status | Trend indicator (↑↓→) |

**Sparkline colors**:
- Green line: Trending down (success decreasing = good)
- Red line: Trending up (success increasing = bad)
- Endpoint circles: Start/end values

## Color Scheme

**Semantic colors** (accessible, clear):

```
COLOR_SUCCESS  = #2ecc71  # Green: good news (attacks blocked)
COLOR_BLOCKED  = #e74c3c  # Red: bad news (attacks succeeded)
COLOR_WARNING  = #f39c12  # Orange: warning/anomaly
COLOR_CRITICAL = #c0392b  # Dark red: critical threat
```

**Anomaly-specific colors**:
- SUDDEN_BREAKTHROUGH: Red (#e74c3c)
- SUSTAINED_SUCCESS: Dark red (#c0392b)
- PLATEAU: Orange (#f39c12)
- DEGRADATION: Green (#2ecc71)
- STRATEGY_SHIFT: Blue (#3498db)

## Integration Examples

### Real Attack Campaign Tracking

```python
from hololoom.redteam.visualization import AttackPoint, AttackTrajectoryRenderer
import json
from datetime import datetime

# Load attack results from CARTS pipeline
with open("attack_results.json") as f:
    results = json.load(f)

# Convert to AttackPoints
points = []
for i, result in enumerate(results):
    point = AttackPoint(
        index=i,
        strategy=result["strategy"],
        success_rate=result["bypass_count"] / result["total_attacks"],
        attack_count=result["total_attacks"],
        bypass_count=result["bypass_count"],
        avg_severity=result["avg_severity"],
        timestamp=result["timestamp"],
        metadata={
            "model": result["model"],
            "temperature": result["temperature"],
            "version": result["carts_version"]
        }
    )
    points.append(point)

# Render
renderer = AttackTrajectoryRenderer()
html = renderer.render(
    points,
    title="CARTS Real-Time Attack Campaign",
    subtitle=f"Generated {datetime.now().isoformat()}"
)

# Save
with open("campaign_analysis.html", "w") as f:
    f.write(html)
```

### Multiple Model Comparison

```python
from hololoom.redteam.visualization import render_attack_trajectory

models = ["gpt-4", "claude-3", "gemini"]
colors_per_model = {
    "gpt-4": [0.65, 0.70, 0.72],
    "claude-3": [0.42, 0.40, 0.38],
    "gemini": [0.28, 0.26, 0.24]
}

# Flatten data for visualization
all_strategies = []
all_rates = []
all_attacks = []
all_bypasses = []

for model in models:
    rates = colors_per_model[model]
    for i, rate in enumerate(rates):
        all_strategies.append(f"{model} (batch {i+1})")
        all_rates.append(rate)
        all_attacks.append(10)
        all_bypasses.append(int(rate * 10))

html = render_attack_trajectory(
    all_strategies, all_rates, all_attacks, all_bypasses,
    title="Model Comparison: Attack Effectiveness",
    subtitle="Cross-model red team analysis"
)
```

### Monitoring Dashboard Integration

```python
from hololoom.redteam.visualization import render_attack_trajectory
from flask import Flask, render_template_string

app = Flask(__name__)

@app.route("/dashboard")
def dashboard():
    # Fetch latest attack data
    strategies = ["prompt_injection", "jailbreak", "overflow"]
    success_rates = [0.65, 0.42, 0.28]
    attack_counts = [100, 100, 100]
    bypass_counts = [65, 42, 28]

    # Generate visualization
    html = render_attack_trajectory(
        strategies, success_rates, attack_counts, bypass_counts,
        title="Live Attack Dashboard"
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>CARTS Dashboard</title></head>
    <body>
        {html}
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

## Output Format

The visualizer produces **pure HTML/CSS/SVG** with:

- **No external dependencies** - No CDN links, no JavaScript frameworks
- **Self-contained** - Single HTML file can be saved and shared
- **Responsive** - Works on mobile (CSS media queries)
- **Fast** - No rendering overhead, instant display
- **Accessible** - High contrast, semantic colors
- **Printable** - Looks good on paper with @media print styles

## Performance

| Metric | Value |
|--------|-------|
| Render time | <100ms for 100 data points |
| File size | ~35KB for typical visualization |
| Memory usage | Minimal (string concatenation) |
| Dependencies | Zero external |

## Known Limitations

1. **Maximum points**: ~1000 points (performance degrades above this)
2. **No interactivity**: Pure SVG (no hover tooltips, click events)
3. **No real-time updates**: Render once, static HTML
4. **Chart complexity**: Single line chart (no subplots)

## Future Enhancements

Planned features:

- Interactive tooltips (requires JS)
- Comparison mode (multiple campaigns side-by-side)
- Export to PNG/PDF
- Dark mode CSS variant
- Animated transitions
- Confidence intervals (shaded bands)

## Troubleshooting

### All data points show same success rate

Check your data:
```python
# Verify variety in success_rates
assert min(success_rates) != max(success_rates), "No variation in data"
```

### Anomalies not detected

Anomalies require minimum variation:
- SUDDEN_BREAKTHROUGH: >20% jump required
- SUSTAINED_SUCCESS: 3+ points >60% required
- PLATEAU: ±5% stability for 5+ points

**Solution**: Add more varied data or disable anomalies:
```python
renderer = AttackTrajectoryRenderer(detect_anomalies=False)
```

### Chart appears stretched/compressed

Adjust dimensions:
```python
renderer = AttackTrajectoryRenderer(
    max_width=1400,  # Wider
    chart_height=400  # Taller
)
```

### Numbers hard to read

Increase font sizes in CSS (edit `_render_styles()` method) or use a larger display.

## Contributing

To extend the visualization:

1. **Add new anomaly type**: Add to `AnomalyType` enum, update `_detect_anomalies()`
2. **Add new metric**: Update `_compute_metrics()` and `_render_statistics()`
3. **Change colors**: Edit class variables `COLOR_*` and `ANOMALY_COLORS`
4. **Modify layout**: Edit `_render_chart()` and CSS in `_render_styles()`

## References

- **Tufte, E. R.** (2001). "The Visual Display of Quantitative Information" (2nd ed.)
- **SVG specification**: https://www.w3.org/TR/SVG2/
- **Color accessibility**: https://www.w3.org/WAI/WCAG21/Understanding/use-of-color.html
