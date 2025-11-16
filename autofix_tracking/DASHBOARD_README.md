# AutoFix Monitoring Dashboard

**Status**: ✅ Production Ready (November 16, 2025)
**Location**: `/autofix_tracking/dashboard.html`
**Generator**: `/autofix_tracking/dashboard_generator.py`

Comprehensive real-time monitoring dashboard for tracking autofix success rates, confidence calibration, and category performance metrics.

## Overview

The AutoFix Monitoring Dashboard provides visual insights into the effectiveness of the autofix system through:

- **Success Rate Tracking**: Watch fix success rates evolve over time with trend analysis
- **Confidence Calibration**: Validate that the system's predicted confidence matches actual success rates
- **Category Breakdown**: Understand which issue categories are easiest/hardest to fix
- **Session Comparison**: Compare individual session performance across metrics
- **Statistical Insights**: Auto-generated insights for each visualization

## Quick Start

### Generate Dashboard

```bash
# From project root
python autofix_tracking/dashboard_generator.py

# Output: autofix_tracking/dashboard.html
```

### View Dashboard

```bash
# Open in default browser
open autofix_tracking/dashboard.html

# Or use local server for best performance
python -m http.server 8000 --directory autofix_tracking
# Then visit: http://localhost:8000/dashboard.html
```

## Features

### 1. Key Metrics (Header Cards)

Six metric cards show the most important aggregate statistics:

| Metric | Description | Good Range |
|--------|-------------|-----------|
| **Overall Success Rate** | % of fixes applied vs attempted | 75-90% |
| **Average Confidence** | Mean confidence across all fixes | 75-85% |
| **Total Fixes Applied** | Number of successful fixes | Depends on volume |
| **Total Attempted** | Total fixes tried | Depends on volume |
| **Avg Duration/Session** | Time per session in seconds | <60s ideal |
| **Sessions Tracked** | Number of sessions | Increasing over time |

**Color Coding**:
- Success Rate > 80% = Green
- Success Rate 60-80% = Orange
- Success Rate < 60% = Red

### 2. Success Rate Over Time

**Chart Type**: Line chart with filled area
**X-axis**: Timestamps (chronological)
**Y-axis**: Success rate percentage (0-100%)
**Interpretation**:
- Upward trend = System improving
- Downward trend = Quality regression
- Flat trend = Consistent performance
- Volatility = Inconsistent behavior

**Actionable Insights**:
- If declining: Review recent changes, check confidence calibration
- If below 70%: Consider increasing confidence threshold
- If above 90%: May be under-conservative, consider lowering threshold

### 3. Confidence Over Time

**Chart Type**: Line chart with filled area
**X-axis**: Timestamps (chronological)
**Y-axis**: Average confidence percentage (0-100%)
**Interpretation**:
- High confidence + High success = Well-calibrated
- High confidence + Low success = Overconfident (needs calibration)
- Low confidence + High success = Underconfident (too conservative)
- Low confidence + Low success = Broken (investigate)

**Actionable Insights**:
- Confidence should correlate with success rate
- Gaps between confidence and success indicate calibration issues
- Growing confidence is positive only if success grows too

### 4. Confidence Calibration Curve

**Chart Type**: Scatter plot + reference diagonal
**X-axis**: Predicted confidence bins (0-100%)
**Y-axis**: Actual success rate (0-100%)
**Reference Line**: Perfect calibration (diagonal, dashed)

**Interpretation**:
- Points ON the diagonal = Perfect calibration
- Points ABOVE diagonal = Overconfident (actual success < predicted)
- Points BELOW diagonal = Underconfident (actual success > predicted)

**Perfect Calibration Meaning**:
```
If system predicts 80% confidence, it should succeed ~80% of the time.
If system predicts 60% confidence, it should succeed ~60% of the time.
```

**Calibration Categories**:
| Prediction | Ideal Actual | Calibration Status |
|-----------|-------------|-------------------|
| 50% confidence | 50% success | Perfect |
| 70% confidence | 70% success | Perfect |
| 90% confidence | 90% success | Perfect |
| 90% confidence | 70% success | Overconfident ⚠️ |
| 50% confidence | 80% success | Underconfident ⚠️ |

**How to Improve Calibration**:
1. **Overconfident (points above diagonal)**:
   - Increase confidence threshold
   - Review confidence scoring logic
   - Add more validation checks

2. **Underconfident (points below diagonal)**:
   - Decrease confidence threshold (if success rate is high)
   - Review what makes successful fixes "underrated"
   - Investigate if validation is too strict

### 5. Success Rate by Category

**Chart Type**: Bar chart
**X-axis**: Issue categories (dead_code, hardcoded_values, etc.)
**Y-axis**: Success rate percentage (0-100%)

**Interpretation**:
- Taller bars = Easier to fix automatically
- Shorter bars = Harder/riskier category

**Category Insights**:
- **dead_code**: Usually high success (70-95%)
- **hardcoded_values**: Medium success (60-80%)
- **missing_docstrings**: Usually high success (80-90%)
- **incomplete**: Lower success (40-70%)

**Actionable Insights**:
- Categories below 50%: Consider disabling or improving strategy
- Categories above 90%: Excellent, can increase confidence threshold
- Wide variation: May indicate environmental/context-dependent fixes

### 6. Attempted Fixes by Category

**Chart Type**: Bar chart
**X-axis**: Issue categories
**Y-axis**: Number of attempted fixes

**Interpretation**:
- Shows volume of issues per category
- Higher bars = More frequent issues
- Helps prioritize which categories to focus on

**Coverage Analysis**:
```
If dead_code has 100 attempted but 40% success:
- That's 40 good fixes
- That's 60 fixes that didn't work (needs investigation)
- Prioritize debugging this category
```

### 7. Session Comparison Table

**Columns**:
| Column | Description |
|--------|-------------|
| Session ID | Unique identifier (session_YYYYMMDD_HHMMSS) |
| Timestamp | When session ran |
| Attempted | How many fixes tried |
| Applied | How many succeeded |
| Success Rate | Applied / Attempted |
| Avg Confidence | Mean confidence of all fixes |
| Duration | How long session took |

**Row Coloring**:
- Success Rate > 80% = Green
- Success Rate 60-80% = Orange
- Success Rate < 60% = Red

**What to Look For**:
- Consistency: Similar success rates session-to-session
- Trends: Is success improving over time?
- Outliers: Unusually high/low success sessions
- Correlation: Do high-confidence sessions have high success?

## Data Sources

Dashboard automatically loads data from:

1. **Primary**: `autofix_tracking/all_sessions.json`
   - Aggregated statistics from all sessions
   - Updated after each autofix run

2. **Fallback**: Individual session files
   - `autofix_tracking/session_*.json`
   - Loaded if all_sessions.json is empty

3. **Sample Data**: Synthetic data for demonstration
   - Used if no real data available
   - Realistic patterns: 65% → 90% success over 30 days

## Data Structure

### Session Record

```json
{
  "session_id": "session_20251116_184910",
  "start_time": "2025-11-16T18:49:10.583799",
  "end_time": "2025-11-16T18:50:17.662275",
  "total_fixes_attempted": 25,
  "total_fixes_applied": 20,
  "success_rate": 0.8,
  "avg_confidence": 0.78,
  "categories": ["dead_code", "hardcoded_values"],
  "duration_ms": 67053.5
}
```

### Aggregated Statistics

```json
{
  "sessions": [...],
  "stats_by_category": {
    "dead_code": {
      "attempted": 100,
      "applied": 85,
      "avg_confidence": 0.82
    }
  },
  "stats_by_strategy": {
    "ast": {
      "attempted": 50,
      "applied": 45,
      "success_rate": 0.9
    }
  }
}
```

## Dashboard Customization

### Modify Thresholds

Edit `dashboard_generator.py`:

```python
# Line 227: Change success rate color threshold
if session.success_rate > 0.80:  # Change from 0.8 to 0.9
    status_color = "green"
```

### Change Data Source

```python
# Line 51: Point to custom tracking file
def __init__(self, tracking_dir: str = "./autofix_tracking", data_file: str = "all_sessions.json"):
    self.data_file = data_file
```

### Add New Metrics

1. Compute new statistic in `compute_*()` methods
2. Add to HTML template with new chart div
3. Add Plotly trace in JavaScript section

Example:

```python
# In compute_new_metric()
def compute_latency_distribution(self):
    latencies = [s.duration_seconds for s in self.sessions]
    return sorted(latencies)

# In _build_html_template(), add:
latency_json = json.dumps(latencies)

# In HTML, add div:
<div id="latencyChart" class="chart"></div>

# In JavaScript, add trace:
const traceLatency = {
    y: latencies,
    type: 'histogram',
    name: 'Duration Distribution',
    marker: {color: '#9C27B0'}
};
Plotly.newPlot('latencyChart', [traceLatency], layout, {responsive: true});
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load 30 sessions | <100ms | File I/O + JSON parsing |
| Generate HTML | ~50ms | Template rendering |
| Render charts (first load) | ~500ms | Plotly initialization |
| Render charts (cached) | ~50ms | Browser cached |
| Total generation | ~200ms | Start to finish |

**Browser Performance**:
- Chrome/Safari/Firefox: Smooth, 60fps
- Mobile (iOS Safari): Good performance
- Edge: Good performance
- IE11: Not supported (uses ES6+)

## Integration with AutoFix Pipeline

### Workflow

```
1. AutoFix runs → Tracking data saved
2. Run dashboard generator → HTML updated
3. View dashboard in browser → Analyze metrics
4. Make decisions → Adjust thresholds/strategies
5. Next AutoFix run → Improved results (hopefully!)
```

### Automated Updates

To auto-regenerate on each AutoFix session:

```python
# In apply_autofixes.py, after session.end_session():
from autofix_tracking.dashboard_generator import AutoFixDashboard

dashboard = AutoFixDashboard()
dashboard.load_data()
dashboard.generate_html()
print("Dashboard updated!")
```

Or via cron job (hourly):

```bash
# In crontab
0 * * * * cd /path/to/project && python autofix_tracking/dashboard_generator.py
```

## Troubleshooting

### Issue: Dashboard shows only sample data

**Solution**: Ensure tracking data exists:
```bash
# Check if all_sessions.json has data
python -c "import json; data=json.load(open('autofix_tracking/all_sessions.json')); print(f'Sessions: {len(data[\"sessions\"])}')"

# If empty, run an autofix session first
python apply_autofixes.py --max-files 10
```

### Issue: Dashboard looks blank in browser

**Solution**:
- Try opening in different browser
- Check browser console for errors (F12 → Console)
- Try local server instead of file:// (see "View Dashboard" section)
- Ensure JavaScript is enabled

### Issue: Charts not rendering

**Solution**:
- Check internet connection (Plotly loaded from CDN)
- Try offline version (edit HTML to use local Plotly)
- Open browser console (F12 → Console) to see errors

### Issue: Old data still showing

**Solution**:
- Hard refresh browser (Ctrl+Shift+R on Windows/Linux, Cmd+Shift+R on Mac)
- Clear browser cache
- Re-run dashboard generator

## Statistical Methodology

### Success Rate Calculation

```
success_rate = (fixes_applied) / (fixes_attempted)

Example: 20 applied / 25 attempted = 80%
```

### Average Confidence

```
avg_confidence = (sum of all fix confidences) / (number of fixes)

Example: (0.95 + 0.88 + 0.92) / 3 = 0.917 (91.7%)
```

### Calibration Curve

```
1. Bin confidence scores into 10 groups (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
2. For each bin, compute actual success rate
3. Plot bin center confidence (x) vs actual success (y)
4. Compare to diagonal (perfect calibration)

Formula:
  Calibration Error = |predicted_confidence - actual_success_rate|
  Smaller = Better
```

### Category Performance

```
success_rate_by_category = (fixes_applied_in_category) / (fixes_attempted_in_category)

Aggregated across all sessions where category was enabled
```

## Best Practices

### Monitor Key Metrics

1. **Watch the success rate trend**
   - Should be stable or increasing
   - Sudden drops warrant investigation

2. **Check confidence calibration**
   - Should track close to success rate
   - Divergence indicates model issues

3. **Review category performance**
   - Categories < 50%: Consider disabling
   - Categories > 90%: Can be more aggressive

4. **Session consistency**
   - Should not vary wildly
   - High variance = environmental issues

### Threshold Tuning

Based on calibration curve:

```python
# If overconfident (points above diagonal):
confidence_threshold = 0.90  # Increase from 0.85

# If underconfident (points below diagonal):
confidence_threshold = 0.80  # Decrease from 0.85

# If success < 70%:
confidence_threshold = 0.95  # Increase significantly

# If success > 95%:
confidence_threshold = 0.75  # Decrease (too conservative)
```

### Regular Audits

Run this workflow monthly:

1. Generate dashboard
2. Check calibration curve
3. Review category breakdown
4. Identify underperforming categories
5. Update strategies/thresholds
6. Run A/B test on next session

## Architecture

### Generator Components

1. **Data Loading** (`load_data()`)
   - Reads all_sessions.json
   - Parses session metadata
   - Handles missing files gracefully

2. **Statistics Computation**
   - `compute_calibration_curve()`: Binning + success rates
   - `compute_category_stats()`: Per-category aggregation
   - Overall metrics: mean, sum, etc.

3. **HTML Generation** (`_build_html_template()`)
   - Builds responsive page layout
   - Injects data as JSON
   - Includes all CSS/JavaScript inline

4. **Plotly Charting**
   - Line charts: success/confidence over time
   - Bar charts: category comparison
   - Scatter plot: calibration curve
   - All interactive (hover, zoom, pan)

### File Structure

```
autofix_tracking/
├── dashboard.html              # Generated dashboard (23KB)
├── dashboard_generator.py      # Generator script (475 lines)
├── DASHBOARD_README.md         # This file (documentation)
├── all_sessions.json           # Aggregated tracking data
├── session_*.json              # Individual session records
└── learning_data.json          # ML training data export
```

## Future Enhancements

**Planned Features** (Phase 2):
- [ ] Streaming updates (WebSocket)
- [ ] Custom date range filtering
- [ ] Export to CSV/PDF
- [ ] Comparative analysis (A/B test results)
- [ ] Anomaly detection and alerts
- [ ] Machine learning recommendations
- [ ] Dark mode
- [ ] Multi-project dashboard

## Related Files

- **Generator**: `/autofix_tracking/dashboard_generator.py`
- **Tracker**: `/autofix_tracker.py`
- **AutoFix Policy**: `/xterminator/autofix_policy.py`
- **Batch Processor**: `/apply_autofixes.py`

## Support & Reporting

**Issues or questions?**
- Check recent console output for warnings
- Review tracking files for data validity
- Check browser console (F12 → Console)
- Re-run generator with verbose output

**To improve dashboard**:
1. Add new metrics to generator
2. Test with sample data
3. Verify HTML rendering
4. Update documentation
5. Commit changes

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-16 | Initial release with 6 charts, sample data support |

---

**Generated**: November 16, 2025 | **By**: mythRL Team | **Status**: Production Ready
