# Diagram #7: 9-Layer Architecture - Quick Start Guide

**File:** `07_9layer_architecture.html`
**Type:** Interactive HTML visualization (zero dependencies)
**Last Updated:** November 16, 2025

---

## 🚀 Quick Start (2 Minutes)

### Open the Diagram

**Local file (easiest):**
```bash
# On macOS/Linux
open 07_9layer_architecture.html

# On Windows
start 07_9layer_architecture.html
```

**Web server (recommended for best experience):**
```bash
cd training/interactive/
python3 -m http.server 8000
# Visit: http://localhost:8000/diagrams/07_9layer_architecture.html
```

### First Steps

1. **Read the header** - Understand what the 9-layer architecture is
2. **Click "Trace Query"** - Watch data animate through all 9 layers
3. **Click any layer card** - Expand details or open detailed modal
4. **Explore the sidebar** - See performance metrics, adjust parameters

---

## 🎯 Main Features (At a Glance)

| Feature | What It Does | How to Use |
|---------|-------------|-----------|
| **Layer Cards** | Shows each of 9 layers with timing | Click to expand, shows input/output |
| **Trace Query Button** | Animates data flowing through pipeline | Click once, watch animation run |
| **Example Query Selector** | 3 different query types | Change dropdown to switch queries |
| **Mode Selector** | BARE (fast) / FAST (balanced) / FUSED (quality) | Click buttons or use dropdown |
| **Data Viewer** | Shows data structure at each layer | See how data transforms through pipeline |
| **Performance Profiler** | Latency breakdown by stage | Shows bottlenecks in red |
| **Parameter Sliders** | Adjust retrieval limit, scales, graph depth | Drag to see real-time latency changes |
| **Modal Details** | Deep technical explanation per layer | Click layer card to open |

---

## 📚 Common Tasks

### Task 1: Understand How Data Flows
1. Expand all layer cards (click expand button ▼)
2. Read each layer's "Output" section
3. Notice size growing from 50B (input) to 10KB (final output)
4. Click "Trace Query" to see animation

**Time: 5 minutes**
**Result:** Understand complete pipeline

---

### Task 2: Identify Performance Bottleneck
1. Look at "Performance Profiler" in right sidebar
2. Find red bar (bottleneck)
3. Layer 4 (Memory Retrieval) is slowest: 50ms (34% of total)
4. Click Layer 4 modal to read optimization strategies

**Time: 3 minutes**
**Result:** Know why Memory Retrieval is slow

---

### Task 3: Compare Execution Modes
1. Click "BARE" mode button (top left)
2. Watch timing badges on all layers decrease
3. See Performance Profiler total time change to ~50ms
4. Click "FUSED" to see opposite (300ms+)
5. Back to "FAST" (balanced default at 150ms)

**Time: 3 minutes**
**Result:** Understand BARE/FAST/FUSED tradeoffs

---

### Task 4: Optimize Latency
1. Open "Adjust Parameters" panel (bottom right)
2. Drag "Memory Retrieval Limit" slider to 3 (from 6)
3. Watch total time drop to ~98ms
4. Drag "Embedding Scales" to 1 (from 3)
5. Watch features extraction time drop
6. Note: Total time ~70ms but quality decreases

**Time: 5 minutes**
**Result:** See speed vs. quality tradeoff

---

### Task 5: Learn About Specific Layer
1. Click any layer card (e.g., "Layer 7: Decision Collapse")
2. Modal opens with:
   - **Purpose:** What it does
   - **Algorithm:** Pseudocode showing how it works
   - **Example:** Sample input/output
   - **Performance:** Timing and data size
3. Read algorithm and example
4. Close modal (click ✕ or click outside)

**Time: 5 minutes per layer**
**Result:** Deep technical understanding

---

## 🔍 What Each Section Means

### Architecture Column (Left)
```
[Layer 1] Input Processing → 50B → 200B (3ms)
    ↓
[Layer 2] Pattern Selection → FAST mode selected (1ms)
    ↓
[Layer 3] Temporal Control → Time bounds set (<1ms)
    ↓
[Layer 4] Memory Retrieval → 6 memories retrieved ⚡ (50ms, bottleneck!)
    ↓
[Layer 5] Feature Extraction → Embeddings + motifs (35ms)
    ↓
[Layer 6] Warp Space → Continuous manifold (12ms)
    ↓
[Layer 7] Decision → Select "answer" tool (9ms)
    ↓
[Layer 8] Execution → Generate response (30ms)
    ↓
[Layer 9] Spacetime → Complete output with trace (7ms)
```

### Data Viewer (Top Right)
Shows actual data structure flowing through pipeline. Watch how `Query(text="...")` becomes `Spacetime(result="...", confidence=0.92, trace=[...])`.

### Performance Profiler (Middle Right)
Horizontal bars showing timing contribution of each layer. Red bar = bottleneck. Longer bar = more time spent.

### Adjust Parameters (Bottom Right)
Three sliders:
- **Retrieval Limit (1-10):** How many memories to fetch. More = slower but potentially higher quality
- **Embedding Scales (1-3):** Multi-scale embeddings. More scales = more features but slower
- **Graph Hop Depth (1-5):** How deep to search knowledge graph. More = more context but slower

---

## 💡 Tips & Tricks

### Tip 1: Watch the Animation
Click "Trace Query" a few times to see the visual flow. Helps internalize the pipeline order.

### Tip 2: Layer 4 is the Bottleneck
Memory Retrieval (Layer 4) takes 34% of total time. This is where optimization efforts should focus.

### Tip 3: BARE vs FUSED
- **BARE:** Use for mobile/edge devices (50ms, lower quality)
- **FAST:** Use for production web services (150ms, good quality)
- **FUSED:** Use for research/documentation (300ms, best quality)

### Tip 4: Slider Experiments
Reduce retrieval limit from 6 to 3, watch total time drop from 150ms to ~98ms. Still decent quality!

### Tip 5: Read the Modals
Each layer has unique algorithm explanation. Modals are the deepest source of understanding.

---

## ❓ Common Questions

**Q: Why is Layer 4 (Memory Retrieval) so slow?**
A: It does three things: BM25 search, semantic similarity, graph traversal. Databases are the bottleneck in most systems.

**Q: What's the difference between BARE and FAST?**
A: BARE skips features extraction (Layer 5). Faster but lower quality. FAST is the balanced default.

**Q: Can I reduce latency below 50ms?**
A: Yes! Use BARE mode (~50ms) or reduce retrieval limit + embedding scales. But quality suffers.

**Q: What's "Thompson Sampling" mentioned in Layer 7?**
A: A Bayesian algorithm that balances exploring new tools vs. using known-good tools. See Diagram #2 for details.

**Q: Why does data grow from 50B to 10KB?**
A: Each layer adds context: embeddings (2KB), spectral features (1KB), trace (2KB), etc. But final response is smaller (~1KB).

**Q: Can I use this for production?**
A: Yes! It's a working visualization. But the timing values are estimated. In production, profile against real queries.

---

## 🎓 Learning Paths

### Path 1: Visual Learner (15 minutes)
1. Click "Trace Query" several times (2 min)
2. Expand all layer cards (3 min)
3. Identify bottleneck in performance profiler (2 min)
4. Click Layer 4 modal to understand why (8 min)
5. Adjust sliders to experiment with latency (3 min)

**Result:** Intuitive understanding of pipeline and bottlenecks

---

### Path 2: Deep Technical (30 minutes)
1. Read header section (2 min)
2. Expand layers 1, 4, 5, 7, 9 (8 min)
3. Click each of those layers to read detailed modals (15 min)
4. Read algorithms in modals (3 min)
5. Experiment with different modes and sliders (2 min)

**Result:** Comprehensive technical understanding

---

### Path 3: Architecture Overview (10 minutes)
1. Read header (1 min)
2. Skim all layer titles and key operations (3 min)
3. Click "Trace Query" to see flow (2 min)
4. Look at Performance Profiler bottleneck (2 min)
5. Read data viewer to see transformations (2 min)

**Result:** High-level understanding of the system

---

## 🖥️ Technical Details

**Technology:** Pure HTML5 + CSS3 + JavaScript (ES6+)
**Dependencies:** None (works completely offline)
**File Size:** 57 KB
**Load Time:** <500ms
**Browser Support:** Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
**Mobile:** Fully responsive and touch-friendly
**Accessibility:** WCAG AA compliant

---

## 📖 Related Resources

| Resource | What It Is | Where To Find |
|----------|-----------|---------------|
| **Text Guide** | Full written documentation | TRAINING_PART_2_CORE_CONCEPTS.md |
| **Animated SVG** | Animated 9-layer flow | animated/svg/07_9layer_flow.svg |
| **PDF Version** | Printable PDF | pdf/PART_2_CORE_CONCEPTS.pdf |
| **Full Spec** | Implementation specification | MULTIMEDIA_ENHANCEMENT_PLAN.md (lines 61-97) |
| **Complete Summary** | Detailed feature list | INTERACTIVE_DIAGRAM_7_SUMMARY.md |

---

## 🚀 Next Steps

1. **Understand Thompson Sampling:** View Diagram #2 for deep dive on decision-making
2. **Learn Memory Systems:** Read about Knowledge Graphs and Vector DBs
3. **Study Feature Extraction:** Understand Matryoshka embeddings and spectral features
4. **Read Production Code:** Look at `HoloLoom/weaving_orchestrator.py` to see actual implementation

---

**Happy exploring! Questions? Open an issue or check the full documentation.**
