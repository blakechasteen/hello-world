# Red Team Visualization - Quick Reference

**Status**: ✅ Production Ready (November 2025)
**Location**: `hololoom/redteam/visualization/`

---

## 30-Second Getting Started

```python
from hololoom.redteam.visualization import render_attack_trajectory

html = render_attack_trajectory(
    strategies=["prompt_injection", "jailbreak", "overflow"],
    success_rates=[0.65, 0.42, 0.28],
    attack_counts=[100, 100, 100],
    bypass_counts=[65, 42, 28],
    title="Attack Campaign Analysis"
)

with open("report.html", "w") as f:
    f.write(html)
```

Open `report.html` in any browser. Done!

---

## API Reference (TL;DR)

### Convenience Function

```python
render_attack_trajectory(
    strategies: List[str],              # ["strategy_1", "strategy_2"]
    success_rates: List[float],         # [0.65, 0.42]  (0.0-1.0)
    attack_counts: List[int],           # [100, 100]
    bypass_counts: List[int],           # [65, 42]
    title: str = "Attack Trajectory",   # Chart title
    subtitle: Optional[str] = None,     # Optional subtitle
    detect_anomalies: bool = True,      # Enable pattern detection
    show_strategy_breakdown: bool = True # Show strategy sparklines
) -> str                                 # HTML output
```

### Core Classes

**AttackPoint** - One data point
```python
AttackPoint(
    index=0,                        # Sequence number (0, 1, 2, ...)
    strategy="prompt_injection",    # Attack type
    success_rate=0.65,              # 0.0-1.0 success rate
    attack_count=100,               # Total attacks
    bypass_count=65,                # Successful bypasses
    avg_severity=0.75,              # 0.0-1.0 severity
    timestamp=None,                 # Optional Unix timestamp
    metadata=None                   # Optional extra data
)
```

**AttackTrajectoryRenderer** - Main engine
```python
renderer = AttackTrajectoryRenderer(
    detect_anomalies=True,          # Detect patterns
    show_strategy_breakdown=True,   # Show comparisons
    max_width=1200,                 # SVG width
    chart_height=300                # Chart height
)

html = renderer.render(
    points: List[AttackPoint],
    title: str,
    subtitle: Optional[str]
)
```

**StrategyMetrics** - Per-strategy summary (auto-computed)
```python
metrics = StrategyMetrics(
    strategy="prompt_injection",
    total_attacks=500,
    total_bypasses=325,
    bypass_rate=0.65,
    avg_severity=0.72,
    trend="degrading",              # "improving"/"degrading"/"stable"
    sparkline_data=[0.6, 0.63, 0.65, 0.64]
)
```

---

## Usage Patterns

### Pattern 1: Simple Visualization

```python
from hololoom.redteam.visualization import render_attack_trajectory

html = render_attack_trajectory(
    strategies=["technique_a", "technique_b"],
    success_rates=[0.75, 0.55],
    attack_counts=[50, 50],
    bypass_counts=[37, 27]
)

with open("output.html", "w") as f:
    f.write(html)
```

### Pattern 2: Detailed Rendering

```python
from hololoom.redteam.visualization import AttackTrajectoryRenderer, AttackPoint
from datetime import datetime

# Create points with metadata
points = [
    AttackPoint(
        index=i,
        strategy="prompt_injection",
        success_rate=0.25 + (i * 0.05),  # Improving over time
        attack_count=100,
        bypass_count=int((0.25 + i * 0.05) * 100),
        avg_severity=0.8,
        timestamp=datetime.now().timestamp() + (i * 3600),
        metadata={"model": "gpt-4", "api_version": "v1"}
    )
    for i in range(10)
]

# Render with configuration
renderer = AttackTrajectoryRenderer(
    detect_anomalies=True,
    show_strategy_breakdown=True,
    chart_height=400
)

html = renderer.render(
    points,
    title="Adversarial Campaign: Q4 2025",
    subtitle="Prompt Injection Attack Series"
)

with open("detailed_report.html", "w") as f:
    f.write(html)
```

### Pattern 3: Integration with CARTS

```python
from hololoom.redteam.visualization import AttackTrajectoryRenderer
from hololoom.redteam.tracker import AttackTracker

# Get data from CARTS tracker
tracker = AttackTracker()
campaign_data = tracker.get_campaign("carts_2025_final")

# Convert to visualization points
points = [
    AttackPoint(
        index=i,
        strategy=attack.strategy_name,
        success_rate=attack.success_rate,
        attack_count=attack.total_count,
        bypass_count=attack.bypass_count,
        avg_severity=attack.severity,
        timestamp=attack.timestamp,
        metadata=attack.extra_data
    )
    for i, attack in enumerate(campaign_data.attacks)
]

# Generate report
renderer = AttackTrajectoryRenderer()
html = renderer.render(
    points,
    title=f"Campaign: {campaign_data.name}",
    subtitle=f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# Save
with open(f"campaign_{campaign_data.id}.html", "w") as f:
    f.write(html)
```

---

## Anomaly Types (Automatic Detection)

When `detect_anomalies=True`, system detects:

| Type | Meaning | Color |
|------|---------|-------|
| **SUDDEN_BREAKTHROUGH** | Success rate jumps >20% | 🔴 Red |
| **SUSTAINED_SUCCESS** | High success for 3+ points | 🔴 Dark Red |
| **PLATEAU** | Success stable for 5+ points | 🟠 Orange |
| **DEGRADATION** | Success drops >15% (good!) | 🟢 Green |
| **STRATEGY_SHIFT** | Attack technique changes | 🔵 Blue |

---

## Configuration Reference

### Renderer Options

```python
renderer = AttackTrajectoryRenderer(
    # Display options
    detect_anomalies=True,          # Find patterns in data
    show_strategy_breakdown=True,   # Show strategy sparklines

    # Sizing
    max_width=1200,                 # Max SVG width (px)
    chart_height=300                # Chart height (px)
)
```

### Render Options

```python
html = renderer.render(
    points: List[AttackPoint],      # Your data
    title="Analysis Title",          # Main heading
    subtitle="Context info"          # Optional subheading
)
```

---

## Output Example

**File**: `report.html`
**Size**: ~12-15 KB
**Format**: Complete HTML with inline CSS and SVG
**Compatibility**: All modern browsers
**Dependencies**: None

The HTML file is **self-contained** - no external files needed, works offline.

---

## Common Tasks

### Task: Visualize Single Attack Campaign

```python
from hololoom.redteam.visualization import render_attack_trajectory

results = [
    ("attempt_1", 0.35, 50, 17),
    ("attempt_2", 0.42, 50, 21),
    ("attempt_3", 0.40, 50, 20),
]

strategies, rates, counts, bypasses = zip(*results)

html = render_attack_trajectory(
    strategies=list(strategies),
    success_rates=list(rates),
    attack_counts=list(counts),
    bypass_counts=list(bypasses)
)

with open("campaign.html", "w") as f:
    f.write(html)
```

### Task: Compare Multiple Strategies

```python
from hololoom.redteam.visualization import AttackPoint, AttackTrajectoryRenderer

# Data from your attack runs
strategies_data = {
    "prompt_injection": [0.25, 0.28, 0.32, 0.35],
    "jailbreak": [0.15, 0.18, 0.20, 0.22],
    "overflow": [0.08, 0.10, 0.12, 0.14]
}

points = []
for idx, (strategy, rates) in enumerate(strategies_data.items()):
    for seq, rate in enumerate(rates):
        points.append(AttackPoint(
            index=idx * len(rates) + seq,
            strategy=strategy,
            success_rate=rate,
            attack_count=100,
            bypass_count=int(rate * 100),
            avg_severity=rate * 0.9
        ))

renderer = AttackTrajectoryRenderer()
html = renderer.render(points, title="Strategy Comparison Q4 2025")

with open("comparison.html", "w") as f:
    f.write(html)
```

### Task: Generate Executive Report

```python
from hololoom.redteam.visualization import render_attack_trajectory
from datetime import datetime

# Your campaign metrics
summary = {
    "prompt_injection": (45, 250, 112),    # rate, count, bypasses
    "jailbreak": (32, 250, 80),
    "context_overflow": (28, 250, 70),
    "token_smuggling": (15, 250, 37),
}

html = render_attack_trajectory(
    strategies=list(summary.keys()),
    success_rates=[rate for rate, _, _ in summary.values()],
    attack_counts=[count for _, count, _ in summary.values()],
    bypass_counts=[bypasses for _, _, bypasses in summary.values()],
    title="Red Team Campaign - Q4 2025 Final Report",
    subtitle=f"Generated {datetime.now().strftime('%B %d, %Y')}"
)

# Add to report template or email
print(f"Report ready for distribution:")
print(f"- {len(summary)} strategies analyzed")
print(f"- {sum(c for _, c, _ in summary.values())} total attacks")
print(f"- {sum(b for _, _, b in summary.values())} successful bypasses")

with open("Q4_2025_RedTeam_Report.html", "w") as f:
    f.write(html)
```

---

## Troubleshooting

### Problem: "ValueError: All lists must have same length"

**Solution**: Ensure all lists have same number of elements

```python
# ❌ Wrong
render_attack_trajectory(
    strategies=["a", "b"],              # 2 items
    success_rates=[0.5, 0.6, 0.7],      # 3 items - MISMATCH!
    ...
)

# ✅ Correct
render_attack_trajectory(
    strategies=["a", "b", "c"],         # 3 items
    success_rates=[0.5, 0.6, 0.7],      # 3 items
    attack_counts=[100, 100, 100],      # 3 items
    bypass_counts=[50, 60, 70]          # 3 items
)
```

### Problem: "ValueError: success_rate must be 0.0-1.0"

**Solution**: Ensure success rates are between 0.0 and 1.0

```python
# ❌ Wrong
success_rates=[50, 60, 70]  # Should be 0.5, 0.6, 0.7

# ✅ Correct
success_rates=[0.5, 0.6, 0.7]
```

### Problem: HTML output looks plain/no styling

**Solution**: Ensure you're saving as `.html` and opening in browser

```python
# ✅ Correct approach
with open("report.html", "w") as f:  # .html extension!
    f.write(html)

# Open in browser
import webbrowser
webbrowser.open("report.html")
```

---

## Performance Tips

1. **Large datasets (>1000 points)**: Rendering takes <500ms
2. **Multiple reports**: Reuse renderer instance
3. **Batch processing**: Render in parallel for speed

```python
# Efficient: Reuse renderer
renderer = AttackTrajectoryRenderer()

for campaign in campaigns:
    html = renderer.render(campaign.points, title=campaign.name)
    save(html)
```

---

## Demo

Run the built-in demo:

```bash
cd hololoom/redteam/visualization
PYTHONPATH=../.. python demo_attack_trajectory.py

# Output: demo_output_simple.html
#         demo_output_advanced.html
#         demo_output_production.html
```

---

## Documentation

- **Complete Guide**: See `README.md` (650 lines)
- **Code Examples**: See `USAGE_EXAMPLES.md` (450 lines)
- **Implementation Details**: See `REDTEAM_VISUALIZATION_COMPLETE.md` (400 lines)

---

## Key Points

✅ **Zero dependencies** - Works immediately, no pip install
✅ **Self-contained** - Single HTML file, works offline
✅ **Tufte-designed** - Professional appearance
✅ **Production ready** - No beta features
✅ **Easy to use** - 2 lines of code to get started
✅ **Extensible** - Hook into CARTS systems

---

**Questions?** Check `USAGE_EXAMPLES.md` for 10+ complete examples.

**Created**: December 5, 2025
**Status**: ✅ Production Ready
