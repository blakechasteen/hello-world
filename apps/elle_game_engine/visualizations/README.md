# BigPlay Interactive Visualizations

**Version:** 1.0.0
**Philosophy:** "Elegance at Every Step"

---

## 📊 What's Here

This directory contains **interactive HTML visualizations** for the BigPlay Engine. Each visualization is:
- **Self-contained** - No build step, just open in browser
- **Responsive** - Works on desktop, tablet, and mobile
- **Accessible** - Keyboard navigation, screen reader support
- **Beautiful** - Consistent design system with dark mode

---

## 🎨 Available Visualizations

### 1. [**Architecture Diagram**](architecture.html) 🏗️
Interactive system architecture showing all 16 components across 4 layers.

**Features:**
- Click any component to see detailed information
- 4 layers: Client (Unity/Godot/Unreal/Web), API (FastAPI/WebSocket), Core Engine, Data Layer
- Color-coded data flow (API calls, data flow, LLM requests)
- Detailed panels explaining each component

**Best for:** Understanding how BigPlay works end-to-end

---

### 2. [**PAD Emotion Model**](emotion-model.html) 😊
3D interactive visualization of the Pleasure-Arousal-Dominance emotional space.

**Features:**
- Rotate and zoom the 3D emotion cube (Three.js WebGL)
- 5 sample NPCs plotted in emotion space
- Click NPCs to highlight and auto-focus camera
- Complete explanation of PAD + Trust dimensions

**Best for:** Understanding how NPC emotions work

**Requirements:** WebGL-capable browser (works best on desktop)

---

### 3. [**Performance Dashboard**](performance-dashboard.html) 📊
Real-time metrics dashboard with live charts and analytics.

**Features:**
- 6 KPI cards (uptime, latency, throughput, sessions, cache hit rate, token usage)
- 7 interactive charts (latency distribution, throughput, errors, LLM providers, memory, cost)
- Simulated real-time updates (every 3 seconds)
- Time range controls (1h, 6h, 24h, 7d)

**Best for:** Understanding BigPlay's performance characteristics

**Note:** Currently shows demo data. Connect to live backend for real metrics.

---

### 4. [**Visualizations Hub**](index.html) 🎮
Landing page and navigation hub for all visualizations.

**Features:**
- Beautiful card-based layout
- Links to all visualizations
- Dark mode toggle (top right)
- Responsive design

**Start here!**

---

## 🚀 Quick Start

### Option 1: Open in Browser (Recommended)
```bash
# From repository root
open apps/elle_game_engine/visualizations/index.html

# Or double-click index.html in file explorer
```

### Option 2: Local Server (for development)
```bash
# Python 3
python -m http.server 8080 --directory apps/elle_game_engine/visualizations

# Node.js
npx http-server apps/elle_game_engine/visualizations -p 8080

# Then open: http://localhost:8080
```

---

## 🎨 Design System

All visualizations use a **unified design system** for consistency.

### Files
- **[design-system.css](design-system.css)** - CSS variables, components, utilities
- **[bigplay-ui.js](bigplay-ui.js)** - Dark mode, tooltips, accessibility, analytics

### Features

#### 🌓 Dark Mode
- Toggle button (top right, moon/sun icon)
- Persists across sessions (localStorage)
- Smooth transitions
- Works on all visualizations

#### 💬 Tooltips
Add tooltips to any element:
```html
<span data-tooltip="This explains the feature">Hover me</span>
```

#### ♿ Accessibility
- Keyboard navigation (Tab, Enter, Space)
- Skip links for screen readers
- ARIA labels and live regions
- Focus indicators
- Reduced motion support

#### 📱 Responsive Design
- Mobile-first approach
- Touch-friendly interactions
- Adaptive layouts
- Optimized for all screen sizes

---

## 📦 Technology Stack

### Core Libraries
- **Three.js** (r128) - 3D graphics for emotion model
- **Chart.js** (4.4.0) - Charts for performance dashboard
- **Vanilla JavaScript** - No framework bloat

### Fonts
- **Space Grotesk** - Headings
- **Inter** - Body text
- **Fira Code** - Code snippets

### Browser Support
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile Safari 14+

---

## 🛠️ Customization

### Theming

Edit `design-system.css` to change colors:

```css
:root {
    --color-primary: #667eea;      /* Your brand color */
    --color-secondary: #764ba2;    /* Secondary color */
    /* ... */
}
```

### Adding New Visualizations

1. **Create HTML file** (e.g., `my-viz.html`)
2. **Include design system:**
   ```html
   <link rel="stylesheet" href="design-system.css">
   <script src="bigplay-ui.js"></script>
   ```
3. **Use design system classes:**
   ```html
   <div class="card">
       <h2>My Visualization</h2>
       <button class="btn btn-primary">Action</button>
   </div>
   ```
4. **Add to index.html:**
   ```html
   <div class="viz-card" onclick="window.location.href='my-viz.html'">
       <div class="viz-icon">🎨</div>
       <div class="viz-title">My Visualization</div>
       <!-- ... -->
   </div>
   ```

---

## 📈 Enhancement Roadmap

See **[ENHANCEMENT_ROADMAP.md](ENHANCEMENT_ROADMAP.md)** for the complete plan to take these visualizations from good to world-class.

### Coming Soon (Phase 2-6)

**Phase 1: Polish & Perfection** ⏳ In Progress
- ✅ Design system with dark mode
- ✅ Tooltips and micro-interactions
- ✅ Accessibility overhaul
- ⏳ Mobile optimization
- ⏳ Loading states and animations

**Phase 2: Interactive Learning**
- Guided tours with step-by-step walkthroughs
- Interactive playgrounds (live NPC conversations)
- Contextual help and tooltips

**Phase 3: Advanced Visualizations**
- Quest flow diagrams (branching narratives)
- Multiplayer architecture (WebSocket flows)
- NPC relationship graphs (force-directed networks)
- API sequence diagrams (time-based)

**Phase 4: Real-Time Integration**
- WebSocket connection to live BigPlay instance
- Real-time metrics streaming
- Live NPC conversations

**Phase 5: Developer Tools**
- Schema validator and editor
- Performance profiler
- Code generator from visual flows

**Phase 6: Community Features**
- Share & embed visualizations
- Community gallery and templates

---

## 🎯 Quick Wins Implemented

✅ **Design System** (2 hours)
- Unified color palette and spacing
- CSS custom properties for easy theming
- Dark mode toggle with persistence

✅ **Better Tooltips** (2 hours)
- Hover any badge to see helpful info
- Smart positioning (avoids viewport edges)
- Accessible (keyboard + screen reader support)

✅ **Improved Buttons** (1 hour)
- Design system button classes
- Hover effects and transitions
- Focus states for accessibility

---

## 📊 Performance

### File Sizes
- **design-system.css**: 12KB
- **bigplay-ui.js**: 8KB
- **architecture.html**: 23KB
- **emotion-model.html**: 21KB
- **performance-dashboard.html**: 17KB
- **index.html**: 11KB

**Total**: ~92KB (excluding external libraries)

### Load Times
- First load (cold cache): ~1-2s on 3G
- Subsequent loads (warm cache): <500ms
- Time to interactive: <1s

### Lighthouse Scores (Target)
- Performance: 95+
- Accessibility: 95+
- Best Practices: 100
- SEO: 100

---

## 🐛 Troubleshooting

### Dark mode not working
- Check browser console for errors
- Ensure `bigplay-ui.js` is loaded
- Clear localStorage: `localStorage.removeItem('bigplay-theme')`

### 3D emotion model not showing
- Check for WebGL support: chrome://gpu
- Try different browser (Chrome/Firefox recommended)
- Update graphics drivers

### Charts not rendering
- Check browser console for Chart.js errors
- Ensure internet connection (CDN libraries)
- Try hard refresh (Cmd/Ctrl + Shift + R)

### Tooltips not appearing
- Ensure element has `data-tooltip` attribute
- Check `bigplay-ui.js` is loaded
- Verify no JavaScript errors in console

---

## 📝 Contributing

Want to improve the visualizations? Here's how:

1. **Read the roadmap** - [ENHANCEMENT_ROADMAP.md](ENHANCEMENT_ROADMAP.md)
2. **Pick a task** - Start with "Quick Wins" for immediate impact
3. **Make changes** - Edit HTML/CSS/JS files
4. **Test** - Open in multiple browsers, test accessibility
5. **Document** - Update this README with changes
6. **Commit** - Use clear commit messages

### Code Style
- Use design system variables (not hard-coded values)
- Add comments for complex logic
- Follow existing patterns
- Test on mobile and desktop

---

## 📚 Resources

### Learn More
- [BigPlay Engine Docs](../BIGPLAY_ENGINE.md) - Platform overview
- [Getting Started](../GETTING_STARTED.md) - Quick start guide
- [Architecture Guide](../ARCHITECTURE.md) - Technical deep dive
- [Tutorials](../TUTORIALS.md) - Hands-on examples

### Design Inspiration
- [Observable](https://observablehq.com) - Interactive notebooks
- [Distill.pub](https://distill.pub) - ML visualizations
- [Nicky Case](https://ncase.me) - Explorable explanations

### Libraries
- [Three.js Docs](https://threejs.org/docs/) - 3D graphics
- [Chart.js Docs](https://www.chartjs.org/docs/) - Charts
- [D3.js](https://d3js.org) - Data visualization (future)

---

## 📄 License

MIT License - Same as BigPlay Engine

---

## 🙏 Acknowledgments

Built with:
- ❤️ By the BigPlay team
- 🎨 Inspired by Observable, Distill, and Nicky Case
- 🚀 Using Three.js, Chart.js, and vanilla JavaScript
- ♿ With accessibility in mind (WCAG 2.1 AA)

---

**Last Updated:** 2025-11-16
**Maintainers:** BigPlay Team

**Questions?** Open an issue or check the docs!
