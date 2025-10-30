# 🎨 mythRL Narrative Depth Dashboard

## Interactive Real-Time Visualization

Beautiful Streamlit dashboard for exploring narrative intelligence with live depth analysis.

![Dashboard Preview](https://img.shields.io/badge/status-ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r dashboard_requirements.txt
```

This installs:
- `streamlit` - Web application framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation

### 2. Launch Dashboard

```bash
cd mythRL
$env:PYTHONPATH = "."; streamlit run demos/narrative_depth_dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

### 3. Analyze Narratives!

- Choose from pre-loaded examples (Odyssey, LOTR, Harry Potter)
- Or paste your own narrative text
- Click "Analyze Narrative Depth"
- Watch the magic happen! ✨

---

## 🎯 Features

### 📊 Real-Time Analysis
- **Progressive gate unlocking animation** - Watch Matryoshka gates unlock
- **Complexity gauge** - Live complexity score with color-coded zones
- **Depth progression chart** - See how complexity evolves through layers

### 🎭 Character & Archetype Detection
- **Archetypal resonance radar** - Multi-dimensional pattern visualization
- **Character detection** - Identifies 30+ universal characters
- **Campbell stage mapping** - Hero's Journey progress tracking

### 🌌 Cosmic Truth Revelation
- **Animated truth display** - Cosmic insights with pulsing effects
- **Transformation journey** - Step-by-step meaning deepening
- **Symbolic element table** - All detected symbols and interpretations

### ⚡ Performance Monitoring
- **Cache statistics** - Hit rate, size, evictions in real-time
- **Hot entries tracker** - Most frequently accessed analyses
- **Performance metrics** - Sub-millisecond analysis times

---

## 📸 Dashboard Sections

### Tab 1: Analysis
- Text input (custom or examples)
- Real-time depth analysis
- Matryoshka gate visualization
- Complexity gauge & progression charts
- Archetypal resonance radar
- Cosmic truth revelation
- Transformation journey timeline
- Symbolic elements table
- Mythic truths display

### Tab 2: Statistics
- Cache performance metrics
- Hit rate visualization
- Hot entries leaderboard
- Performance distribution pie chart

### Tab 3: Examples
- Pre-analyzed example texts
- Depth level comparisons
- Quick-start demonstrations

---

## 🎨 Visual Elements

### Gate Unlocking Animation
```
🔓 SURFACE    🔓 SYMBOLIC    🔓 ARCHETYPAL    🔓 MYTHIC    🔓 COSMIC
✓ UNLOCKED    ✓ UNLOCKED     ✓ UNLOCKED       ✓ UNLOCKED   ✓ UNLOCKED
```

### Complexity Gauge
- **Gray zone** (0-0.3): Surface level
- **Yellow zone** (0.3-0.5): Symbolic awakening
- **Green zone** (0.5-0.7): Archetypal resonance
- **Orange zone** (0.7-0.85): Mythic significance
- **Purple zone** (0.85-1.0): Cosmic revelation

### Archetypal Radar
Multi-dimensional visualization of archetypal patterns:
- Hero, Mentor, Shadow, Trickster, Sage
- Love, Fear, Hope, Wisdom, Courage
- And more...

---

## 📚 Example Texts Included

1. **Simple Observation** (SYMBOLIC) - Basic scene description
2. **Telemachus' Call** (COSMIC) - Hero's call to adventure
3. **Odysseus & Athena** (COSMIC) - Mythic encounter at crossroads
4. **Frodo's Sacrifice** (COSMIC) - Ultimate transformation
5. **Harry's Mentor** (ARCHETYPAL) - Wisdom transmission
6. **Land of the Dead** (COSMIC) - Death and rebirth

---

## 🛠️ Configuration

### Sidebar Settings
- **Example text selector** - Choose pre-loaded narratives
- **About section** - Quick reference guide
- **Depth levels guide** - Understanding the 5 levels

### Custom Text Analysis
- Paste any narrative (stories, myths, scripts, etc.)
- Works with any length (optimized for 100-1000 words)
- Supports mythological, fictional, and real-world narratives

---

## 🎯 Use Cases

### For Writers
- Analyze story depth and mythic resonance
- Ensure Hero's Journey structure
- Check archetypal balance

### For Researchers
- Study narrative patterns across cultures
- Compare mythological structures
- Analyze character archetypes

### For Educators
- Teach Hero's Journey concepts
- Demonstrate archetypal patterns
- Visualize narrative complexity

### For Developers
- Understand mythRL capabilities
- Debug depth analysis results
- Monitor cache performance

---

## 🔧 Troubleshooting

### Dashboard won't start
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit plotly pandas
```

### Import errors
```bash
# Ensure PYTHONPATH is set
$env:PYTHONPATH = "."

# Or run from mythRL root directory
cd mythRL
streamlit run demos/narrative_depth_dashboard.py
```

### Port already in use
```bash
# Use custom port
streamlit run demos/narrative_depth_dashboard.py --server.port 8502
```

---

## 🚀 Advanced Usage

### Embedding in Applications
```python
# The dashboard uses the same API as production code
from HoloLoom.narrative_cache import CachedMatryoshkaDepth

analyzer = CachedMatryoshkaDepth()
result = await analyzer.analyze_depth(text)
```

### Custom Visualizations
The dashboard components are modular - use them in your own apps:
- `create_gate_visualization()` - Gate unlocking display
- `create_complexity_gauge()` - Plotly gauge chart
- `create_archetypal_radar()` - Radar chart for archetypes
- `create_depth_progression_chart()` - Bar chart for progression

---

## 📊 Performance

- **Initial load**: ~2-3 seconds
- **Analysis time**: 0.5-1.5ms (cold cache)
- **Analysis time**: 0.03-0.1ms (hot cache)
- **Rendering**: <100ms for all visualizations
- **Memory usage**: ~50MB (base) + cache size

---

## 🎉 What Makes This Special

### Instant Visual Feedback
Watch AI discover meaning in real-time - no waiting for batch processing.

### Progressive Complexity
See exactly how the system scales from simple to profound analysis.

### Cache Transparency
Understand performance optimizations as they happen.

### Beautiful Design
Gradient headers, animated cosmic truths, smooth interactions.

### Production-Ready
Same caching and analysis engine as production API - no mocks or demos.

---

## 🌟 Future Enhancements

- [ ] WebSocket streaming for long texts
- [ ] Export results as PDF/PNG
- [ ] Comparison mode (analyze multiple texts side-by-side)
- [ ] Historical analysis timeline
- [ ] Custom archetype databases
- [ ] Multi-language support

---

## 📝 Technical Details

### Architecture
```
Streamlit Frontend
    ↓
CachedMatryoshkaDepth
    ↓
MatryoshkaNarrativeDepth
    ↓
NarrativeIntelligence
    ↓
Universal Character Database + Campbell Stages
```

### Data Flow
1. User inputs text
2. Dashboard shows progress
3. Analyzer runs (with caching)
4. Results visualized in real-time
5. Stats updated automatically

### Caching Strategy
- LRU eviction with TTL
- Content-based hashing (SHA256)
- Automatic cache warming with examples
- Statistics tracked per-session

---

## 🎓 Learn More

- **Main README**: `../README.md`
- **API Documentation**: `../CLAUDE.md`
- **Production Guide**: `narrative_depth_production.py`
- **Full Testing**: `../tests/test_full_odyssey_depth.py`

---

## 💡 Tips

1. **Start with examples** - Get familiar with depth levels
2. **Try your own text** - See how your writing measures up
3. **Watch the gates** - Notice when complexity triggers new levels
4. **Check cache stats** - See the 21.5x speedup in action
5. **Compare texts** - Run multiple analyses to understand patterns

---

## 🙏 Credits

Built with:
- **Streamlit** - Beautiful web apps in Python
- **Plotly** - Interactive visualizations
- **mythRL** - Revolutionary narrative intelligence

---

**Ready to explore narrative depth?** 

```bash
streamlit run demos/narrative_depth_dashboard.py
```

**Let the journey begin!** 🪆✨
