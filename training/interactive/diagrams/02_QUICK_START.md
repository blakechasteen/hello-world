# Thompson Sampling Interactive - Quick Start Guide

**File:** `02_thompson_sampling.html`
**No Installation Required:** Open directly in any web browser

---

## 🚀 Getting Started (30 seconds)

1. **Open the file** in your browser
2. **Move a slider** - see the curve change instantly
3. **Click "Sample Once"** - watch Thompson Sampling in action
4. **Compare tools** - notice how uncertainty affects selection

---

## 🎮 Main Controls

### **Left Panel: Tool Parameters**

Each of 3 tools (A, B, C) has:
- **α (Alpha) Slider** - Number of successes (1-100)
- **β (Beta) Slider** - Number of failures (1-100)
- **Real-Time Stats:**
  - Expected success rate
  - Confidence level (HIGH/MEDIUM/LOW)
  - Uncertainty bar (visual indicator)
- **Reset Button** - Return tool to defaults

### **Right Panel: Visualizations**

**Beta Distribution Curves**
- 3 side-by-side SVG graphs
- Shows probability distribution of success rate
- Peak indicates most likely value
- Width indicates uncertainty

**Thompson Sampling Simulation**
- **Sample Once** - Draw from each tool, pick winner
- **Sample 10 Times** - Run 10 iterations rapidly
- **Clear History** - Reset counters
- **Win Statistics** - See which tool wins most often

**Sample History Table**
- Last 20 samples in reverse order
- Shows iteration, winner, sample value, timestamp

---

## 💡 Interactive Features

### **Preset Scenarios** (One-Click)

| Preset | Description | When to Use |
|--------|-------------|------------|
| **Equal Uncertainty** | All tools β(5,5) | Learn baseline behavior |
| **One Dominant** | A strong, B&C weak | See exploitation bias |
| **High Exploration** | All β(2,1) | Maximize variety |
| **Well-Tested** | All β(50,10) | High confidence mode |

### **Header Controls**

- 🌙 **Dark Mode Toggle** - Switch color scheme
- **Reset All** - Return everything to defaults
- **Download State** - Save configuration as JSON

---

## 📊 Understanding the Display

### **Expected Success Rate**
```
Formula: α / (α + β)

Tool A: 50/(50+10) = 83.3%
Tool B: 10/(10+5) = 66.7%
Tool C: 2/(2+1) = 66.7%
```

### **Uncertainty Levels**

| Level | Indicator | Meaning | Example |
|-------|-----------|---------|---------|
| **LOW** 🟢 | Green bar, narrow curve | Confident in expected value | α=50, β=10 |
| **MEDIUM** 🟡 | Yellow bar, medium curve | Moderate confidence | α=10, β=5 |
| **HIGH** 🔴 | Red bar, wide curve | Very uncertain | α=2, β=1 |

### **Thompson Sampling Decision**

When you click "Sample":
1. System draws random value from each tool's Beta distribution
2. Whichever tool's sample is highest → that tool wins
3. Over many samples, you see the exploitation/exploration balance:
   - High confidence tools win often (exploitation)
   - Low confidence tools win sometimes (exploration)

---

## 🎯 Learning Experiments

### **Experiment 1: Effect of Confidence**

**Goal:** See how confidence affects selection frequency

**Steps:**
1. Set Tool A: α=80, β=5 (very confident)
2. Set Tool B: α=2, β=2 (very uncertain)
3. Click "Sample 20 Times"
4. Observe: Tool A wins ~90% of time

**Insight:** Thompson Sampling exploits confident tools

---

### **Experiment 2: Catching Up**

**Goal:** See exploration enable catching up to leader

**Steps:**
1. Set Tool A: α=50, β=10 (leader)
2. Set Tool B: α=5, β=1 (underdog)
3. Click "Sample 15 Times"
4. Observe: Tool B wins sometimes despite lower expected value

**Insight:** High uncertainty tools still get explored

---

### **Experiment 3: Equal Options**

**Goal:** See unbiased selection with equal uncertainty

**Steps:**
1. Click "Equal Uncertainty" preset
2. All tools at Beta(5,5)
3. Click "Sample 30 Times"
4. Observe: ~10/10/10 wins (roughly equal)

**Insight:** Equal uncertainty → unbiased selection

---

## 📱 Mobile & Responsive

- **Works on phones** - Touch-friendly sliders
- **Works on tablets** - Responsive grid layout
- **Works on desktop** - Full 2-column layout
- **Dark mode** - Automatically detects system preference

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Navigate between controls |
| `Enter` | Click focused button |
| `Space` | Toggle focused button |
| Arrow keys | Adjust focused slider |

---

## 🧠 Key Concepts Explained

### **What is Thompson Sampling?**

A **Bayesian bandit algorithm** that decides which of several options ("arms") to try. It:
1. Maintains a probability distribution for each option's success rate
2. Samples from each distribution
3. Tries the option with the highest sample

**Why this works:** Uncertainty naturally drives exploration, confidence drives exploitation.

### **Why Beta Distribution?**

Beta(α, β) is the **conjugate prior for Bernoulli distribution**, meaning:
- α = number of successes observed
- β = number of failures observed
- Distribution captures "state of knowledge" about success probability

### **Real-World Applications**

- **A/B Testing:** Compare webpage layouts
- **Drug Trials:** Allocate patients to more effective treatment
- **Recommendation Systems:** Explore new items while exploiting favorites
- **Portfolio Optimization:** Balance exploration and exploitation in trading

---

## 🐛 Troubleshooting

### **Curves not showing?**
- Check browser console (F12) for errors
- Try refreshing the page
- Modern browser required (Chrome 90+, Firefox 88+, Safari 14+)

### **Sliders feel slow?**
- This is normal for complex mathematical calculations
- Redrawing curves takes ~30-50ms
- If persistent, try closing other browser tabs

### **Numbers seem wrong?**
- Double-check the formula: E[X] = α/(α+β)
- Try preset scenarios to see expected behavior
- Sample more times (law of large numbers helps)

### **Dark mode not persisting?**
- Browser must allow localStorage
- Check if cookies/storage is blocked
- Try clearing cache and reloading

---

## 🔗 Learn More

**In This Document:**
- See "How Thompson Sampling Works" section (scroll down)
- Read key insight about uncertainty-exploration relationship

**Original Training Docs:**
- [TRAINING_PART_1_FOUNDATIONS.md](../../TRAINING_PART_1_FOUNDATIONS.md) - Full theoretical background
- [MULTIMEDIA_ENHANCEMENT_PLAN.md](../../MULTIMEDIA_ENHANCEMENT_PLAN.md) - Design specifications

**External Resources:**
- "The Bernoulli Bandit Problem" - Classic reference
- Bandit Algorithms for Website Optimization (O'Reilly)
- Thompson, W. R. (1933) - Original paper

---

## ✅ Checklist: What You Can Do

- [x] Adjust α and β sliders for each tool
- [x] Watch curves update in real-time
- [x] Run Thompson Sampling simulation
- [x] See win statistics and sample history
- [x] Load example presets
- [x] Reset individual tools or all tools
- [x] Toggle dark mode
- [x] Download current configuration
- [x] Use on mobile, tablet, desktop
- [x] Experiment with different values
- [x] Understand uncertainty-exploration link

---

## 💾 Saving Your Work

### **To Download Configuration**
1. Click "Download State" button (header)
2. JSON file saved with all current parameter values
3. Share with others or load later

### **To Save to Browser**
- Dark mode preference auto-saves
- Theme persists across sessions
- No account required

---

## ❓ Common Questions

**Q: Can I enter values directly?**
A: No, use sliders. This forces visual learning of relationships.

**Q: Why does Tool C win sometimes despite low α?**
A: High uncertainty = more exploration. Thompson Sampling is doing its job!

**Q: How many samples should I run?**
A: At least 10-20 to see patterns. 100+ for convergence to theoretical values.

**Q: Can I modify the code?**
A: Yes! File is plain HTML/CSS/JavaScript. Open in any text editor.

**Q: Works offline?**
A: Yes! No internet connection required. Pure client-side JavaScript.

---

## 🎓 Teaching Tips

**For Students:**
1. Start with "Equal Uncertainty" preset
2. Gradually increase confidence difference
3. Observe how wins correlate with uncertainty
4. Try to predict outcomes before sampling

**For Teachers:**
1. Project on screen for whole class discussion
2. Ask: "What happens if we make Tool C very confident?"
3. Have students make predictions, then test
4. Connect to real-world applications (A/B testing, drug trials)

---

## 📊 Interactive Diagram Reference

| Component | Purpose | Location |
|-----------|---------|----------|
| **Tool Cards** | Adjust α/β values | Left panel |
| **Beta Curves** | Visualize distributions | Top right |
| **Sample Buttons** | Run Thompson Sampling | Middle right |
| **Statistics** | Win counts and percentages | Lower right |
| **History Table** | View recent samples | Bottom right |
| **Explanation** | Educational content | Below main area |
| **Header Controls** | Theme, reset, download | Top |

---

**Version:** 1.0
**Status:** ✅ Ready to Use
**Created:** November 2025
