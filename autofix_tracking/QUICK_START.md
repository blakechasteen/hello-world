# AutoFix Dashboard - Quick Start Guide

## TL;DR

```bash
# Generate dashboard from tracking data
python autofix_tracking/dashboard_generator.py

# View in browser
open autofix_tracking/dashboard.html
```

## 5-Minute Guide

### 1. Generate Dashboard

```bash
python autofix_tracking/dashboard_generator.py
```

**Output**:
```
✓ Dashboard generated: autofix_tracking/dashboard.html

Key Metrics:
  • Overall Success Rate: 82.0%
  • Average Confidence: 77.4%
  • Total Fixes Applied: 712/875
  • Sessions Tracked: 30
```

### 2. Open Dashboard

**Option A**: Direct (requires local server for full features)
```bash
python -m http.server 8000 --directory autofix_tracking
# Then open: http://localhost:8000/dashboard.html
```

**Option B**: File browser
```bash
open autofix_tracking/dashboard.html  # macOS
xdg-open autofix_tracking/dashboard.html  # Linux
start autofix_tracking/dashboard.html  # Windows
```

### 3. Read the Dashboard

The dashboard shows 5 key sections:

#### Header Metrics (6 cards)
```
Overall Success Rate    Average Confidence    Total Fixes Applied
    82.0%                   77.4%                    712

Total Attempted         Avg Duration/Session   Sessions Tracked
     875                     45.3s                     30
```

**What it means**:
- **82% success rate** = Good! Most fixes are working
- **77.4% confidence** = System is reasonably sure about its fixes
- **712/875** = Applied 712 out of 875 attempted fixes

#### Chart 1: Success Rate Over Time
Shows if fixes are getting better or worse:
- ⬆️ Going up = Improving!
- ➡️ Flat = Consistent
- ⬇️ Going down = Something broke

#### Chart 2: Confidence Over Time
Shows how sure the system is:
- Should track together with success rate
- If they diverge = Calibration problem

#### Chart 3: Calibration Curve
**Most important chart for understanding the system!**

Shows: "Does the system's confidence match actual success?"

```
Perfect calibration: 80% confidence → 80% success
Overconfident: 80% confidence → 50% success (system too sure)
Underconfident: 50% confidence → 80% success (system too cautious)
```

**How to read it**:
- Points ON diagonal = Perfect ✅
- Points ABOVE = Overconfident (too sure) ⚠️
- Points BELOW = Underconfident (too cautious) ✅

#### Chart 4: Success by Category
Shows which types of fixes work best:
```
dead_code: 85% (easy to fix)
hardcoded_values: 72% (medium)
missing_docstrings: 88% (easy)
incomplete: 45% (hard, consider disabling)
```

#### Chart 5: Fixes by Category
Shows which issues are most common:
```
If you see many "incomplete" issues but they have low success:
→ Either improve the strategy or disable that category
```

### 4. Take Action

Based on what you see:

**If success rate is low (<70%)**:
```python
# Increase confidence threshold to be more selective
confidence_threshold = 0.95  # Was 0.85
```

**If success rate is very high (>95%)**:
```python
# Decrease threshold to be more aggressive
confidence_threshold = 0.75  # Was 0.85
```

**If confidence doesn't match success**:
```python
# System needs calibration
# Review confidence scoring in autofix_policy.py
```

**If a category has low success (<50%)**:
```python
# Disable that category
categories = ["dead_code", "hardcoded_values"]  # Remove low-success ones
```

## Common Questions

**Q: Why is my success rate low?**
A: Possible reasons (in order of likelihood):
1. Confidence threshold is too low (apply risky fixes)
2. Issue is complex (incomplete, ambiguous context)
3. Fix strategy is wrong for the context
4. Code patterns aren't in training examples

**Q: Why does confidence not match success?**
A: The system's confidence calculation might be wrong:
1. Missing important context features
2. Confidence formula doesn't match reality
3. Need to retrain confidence model

**Q: Can I see session-by-session details?**
A: Yes! Scroll to "Session Comparison" table. Click on Session ID to see details in `autofix_tracking/session_*.json`

**Q: How often should I regenerate?**
A: After each autofix run, or set up hourly cron job:
```bash
0 * * * * cd /project && python autofix_tracking/dashboard_generator.py
```

**Q: Can I customize the dashboard?**
A: Yes! Edit `dashboard_generator.py`:
- Add new charts
- Change colors/thresholds
- Modify layout
- Then regenerate

## Dashboard Files

```
autofix_tracking/
├── dashboard.html              # ← Open this in browser!
├── dashboard_generator.py      # ← Run this to update
├── DASHBOARD_README.md         # ← Full documentation
├── QUICK_START.md              # ← You are here
├── all_sessions.json           # ← Auto-tracked data
└── session_*.json              # ← Individual sessions
```

## Integration with AutoFix

To auto-update dashboard after each autofix run:

```python
# In apply_autofixes.py, add after tracker.end_session():

from autofix_tracking.dashboard_generator import AutoFixDashboard

dashboard = AutoFixDashboard()
dashboard.load_data()
dashboard.generate_html()
print("✅ Dashboard updated!")
```

## Keyboard Shortcuts (in charts)

| Action | Shortcut |
|--------|----------|
| Zoom | Click-drag on chart |
| Pan | Hold shift + click-drag |
| Reset view | Double-click |
| Download image | Camera icon (top-right) |

## Troubleshooting

**Dashboard is blank?**
```bash
# Make sure you have recent data
ls -l autofix_tracking/*.json

# If all_sessions.json is empty or missing, run an autofix session first
python apply_autofixes.py --max-files 10

# Then regenerate dashboard
python autofix_tracking/dashboard_generator.py
```

**Charts not showing?**
- Try opening in Chrome/Safari (not IE)
- Check browser console for errors (F12 → Console)
- Try using local server (see Option A above)

**Data looks old?**
- Hard refresh browser (Ctrl+Shift+R)
- Clear browser cache
- Regenerate dashboard

## Next Steps

1. **Generate the dashboard** (see TL;DR above)
2. **Open it in browser** and explore the charts
3. **Check your success rate** - is it going up or down?
4. **Look at calibration curve** - are you over/under confident?
5. **Review categories** - which ones work best?
6. **Make a decision** - should you adjust thresholds?
7. **Run next autofix** - with new settings
8. **Compare results** - did it improve?

## More Information

- Full documentation: `DASHBOARD_README.md`
- Tracking system: `autofix_tracker.py`
- AutoFix engine: `xterminator/autofix_policy.py`
- Batch processor: `apply_autofixes.py`

---

**Questions?** Check DASHBOARD_README.md for detailed explanations of every chart and metric.
