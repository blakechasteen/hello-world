# Visualizer Expansion Roadmap

**Goal**: Expand HoloLoom's Edward Tufte Machine with advanced visualizations, interactivity, and analytics

**Current State**: Phase 2.4 Complete (1,790 lines, basic dashboards operational)
**Target State**: Production-ready visual analytics platform

---

## Phase 1: Enhanced Panel Renderers (Week 1) ⭐ HIGH PRIORITY

### 1.1 Force-Directed Network Graphs (D3.js)
**Purpose**: Make knowledge threads visually stunning
**Effort**: 4-6 hours
**Impact**: HIGH - transforms static lists into interactive graphs

**Tasks**:
- [ ] Add D3.js force simulation to HTMLRenderer
- [ ] Implement node/edge rendering with physics
- [ ] Add hover tooltips showing thread details
- [ ] Color-code nodes by type (entity, concept, relationship)
- [ ] Add zoom/pan controls
- [ ] Click nodes to expand related threads

**Files**:
- `HoloLoom/visualization/html_renderer.py` (enhance `_render_network()`)
- `HoloLoom/visualization/network_graph.js` (new, embedded in HTML)

---

### 1.2 True Heatmaps (Semantic Dimensions)
**Purpose**: Visualize 244D semantic space meaningfully
**Effort**: 3-4 hours
**Impact**: MEDIUM - shows semantic understanding

**Tasks**:
- [ ] Extract top N dimensions from semantic cache
- [ ] Create heatmap with Plotly
- [ ] Add dimension labels on axes
- [ ] Color scale for intensity
- [ ] Hover shows dimension name + score
- [ ] Compare query vs cached patterns

**Files**:
- `HoloLoom/visualization/html_renderer.py` (enhance `_render_heatmap()`)
- `HoloLoom/visualization/constructor.py` (extract semantic data)

---

### 1.3 Distribution Charts (Metrics Analysis)
**Purpose**: Show uncertainty and variance
**Effort**: 2-3 hours
**Impact**: MEDIUM - better statistical insights

**Tasks**:
- [ ] Implement histogram renderer
- [ ] Box plot renderer for distributions
- [ ] Violin plot for confidence distributions
- [ ] Compare current vs historical metrics

**Files**:
- `HoloLoom/visualization/html_renderer.py` (enhance `_render_distribution()`)

---

### 1.4 Comparison Panels (Side-by-Side)
**Purpose**: Compare multiple queries or approaches
**Effort**: 3-4 hours
**Impact**: MEDIUM - useful for optimization

**Tasks**:
- [ ] Side-by-side metric comparison
- [ ] Diff highlighting (better/worse)
- [ ] Before/after timeline comparison
- [ ] A/B test result visualization

**Files**:
- `HoloLoom/visualization/dashboard.py` (new PanelType.COMPARISON)
- `HoloLoom/visualization/html_renderer.py` (new `_render_comparison()`)

---

## Phase 2: Interactive Features (Week 2) ⭐ HIGH PRIORITY

### 2.1 Panel Collapse/Expand
**Purpose**: Reduce visual clutter
**Effort**: 2-3 hours
**Impact**: HIGH - better UX

**Tasks**:
- [ ] Add collapse button to each panel
- [ ] Smooth CSS transitions
- [ ] Remember collapsed state in localStorage
- [ ] Keyboard shortcuts (Ctrl+Arrow to collapse/expand)

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add collapse controls)
- `HoloLoom/visualization/dashboard_controls.js` (new, embedded)

---

### 2.2 Filter Panels by Type
**Purpose**: Focus on specific information
**Effort**: 2 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Add filter toolbar (All, Metrics, Charts, Text, Networks)
- [ ] Hide/show panels based on filter
- [ ] Highlight count per category
- [ ] URL parameter for default filter

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add filter toolbar)

---

### 2.3 Bottleneck Auto-Detection & Highlighting
**Purpose**: Instantly identify performance issues
**Effort**: 3-4 hours
**Impact**: VERY HIGH - actionable insights

**Tasks**:
- [ ] Detect slowest stage (>40% of total time)
- [ ] Highlight in timeline with red/orange
- [ ] Add warning icon + tooltip
- [ ] Suggest optimization (e.g., "Retrieval is slow - consider caching")
- [ ] Compare to historical averages

**Files**:
- `HoloLoom/visualization/constructor.py` (add bottleneck detection)
- `HoloLoom/visualization/html_renderer.py` (visual highlighting)

---

### 2.4 Click-to-Drill-Down
**Purpose**: Explore details on demand
**Effort**: 4-5 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Click timeline stage → show detailed breakdown
- [ ] Click network node → show thread details
- [ ] Click metric → show historical trend
- [ ] Modal overlay for details

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add click handlers)
- `HoloLoom/visualization/modal.js` (new)

---

## Phase 3: Export & Sharing (Week 3)

### 3.1 Export to PDF
**Purpose**: Shareability
**Effort**: 4-6 hours
**Impact**: HIGH

**Tasks**:
- [ ] Add "Export PDF" button
- [ ] Use browser print API with custom CSS
- [ ] Page breaks between sections
- [ ] Print-optimized styling
- [ ] Include metadata footer

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add print CSS)
- `HoloLoom/visualization/export.js` (new)

---

### 3.2 Export Individual Panels
**Purpose**: Embed charts in presentations
**Effort**: 2-3 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Right-click panel → "Save as PNG"
- [ ] Download SVG for vector graphics
- [ ] Copy to clipboard as image
- [ ] Use html2canvas or Plotly's export

**Files**:
- `HoloLoom/visualization/export.js` (enhance)

---

### 3.3 Shareable URLs
**Purpose**: Share dashboards without files
**Effort**: 3-4 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Encode dashboard data in URL (base64 + gzip)
- [ ] Decode and render from URL
- [ ] URL shortening service integration (optional)
- [ ] Copy shareable link button

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add share button)
- `HoloLoom/visualization/url_codec.js` (new)

---

## Phase 4: Advanced Visualizations (Week 4)

### 4.1 3D Semantic Space (Plotly 3D)
**Purpose**: Explore embeddings spatially
**Effort**: 4-5 hours
**Impact**: LOW (cool but not critical)

**Tasks**:
- [ ] PCA/t-SNE to reduce 384D → 3D
- [ ] Plotly 3D scatter plot
- [ ] Color by cluster/topic
- [ ] Rotate, zoom, pan controls

**Files**:
- `HoloLoom/visualization/html_renderer.py` (new panel type)
- Requires scikit-learn for dimensionality reduction

---

### 4.2 Sankey Diagrams (Data Flow)
**Purpose**: Show information flow through pipeline
**Effort**: 3-4 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Query → Features → Context → Decision → Response
- [ ] Width proportional to data volume
- [ ] Color by stage
- [ ] Plotly Sankey chart

**Files**:
- `HoloLoom/visualization/html_renderer.py` (new `_render_sankey()`)

---

### 4.3 Radar Charts (Multi-Dimensional Comparison)
**Purpose**: Compare metrics across dimensions
**Effort**: 2-3 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Show confidence, speed, coverage, etc. on axes
- [ ] Compare multiple queries
- [ ] Plotly radar chart

**Files**:
- `HoloLoom/visualization/html_renderer.py` (new `_render_radar()`)

---

## Phase 5: Performance & Analytics (Week 5)

### 5.1 Trend Analysis
**Purpose**: Compare to historical data
**Effort**: 4-6 hours
**Impact**: HIGH

**Tasks**:
- [ ] Store dashboard metrics in SQLite
- [ ] Calculate rolling averages
- [ ] Show trend arrows (↑↓) next to metrics
- [ ] Sparklines for 7-day trends
- [ ] Percentile rankings (top 10%, median, etc.)

**Files**:
- `HoloLoom/visualization/analytics.py` (new)
- `HoloLoom/visualization/metrics_store.db` (new SQLite)

---

### 5.2 Anomaly Detection
**Purpose**: Flag unusual patterns
**Effort**: 3-4 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Z-score based outlier detection
- [ ] Flag metrics >2σ from mean
- [ ] Visual warning indicators
- [ ] "Why is this unusual?" explanations

**Files**:
- `HoloLoom/visualization/analytics.py` (enhance)

---

### 5.3 Performance Recommendations
**Purpose**: Actionable optimization suggestions
**Effort**: 4-5 hours
**Impact**: VERY HIGH

**Tasks**:
- [ ] Rule-based recommendation engine
- [ ] "Enable semantic cache" if disabled + slow
- [ ] "Reduce embedding scales" if >300ms
- [ ] "Check network threads" if >100 activated
- [ ] Show potential speedup estimates

**Files**:
- `HoloLoom/visualization/recommendations.py` (new)
- `HoloLoom/visualization/html_renderer.py` (add recommendations panel)

---

## Phase 6: Accessibility & Polish (Week 6)

### 6.1 Full Dark Theme
**Purpose**: Aesthetic + eye strain reduction
**Effort**: 2-3 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Dark CSS variables
- [ ] Theme toggle button
- [ ] Save preference in localStorage
- [ ] Adjust chart colors for dark background
- [ ] Auto-detect system theme preference

**Files**:
- `HoloLoom/visualization/html_renderer.py` (enhance with dark CSS)

---

### 6.2 Keyboard Navigation
**Purpose**: Power user efficiency
**Effort**: 2-3 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Tab through panels
- [ ] Arrow keys to navigate
- [ ] Shortcuts: `e` export, `d` dark mode, `f` filter, `c` collapse all
- [ ] Show shortcut help (press `?`)

**Files**:
- `HoloLoom/visualization/keyboard.js` (new)

---

### 6.3 Screen Reader Support
**Purpose**: Accessibility compliance
**Effort**: 3-4 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] ARIA labels on all interactive elements
- [ ] Semantic HTML5 tags
- [ ] Alt text for charts (describe data)
- [ ] Keyboard-only navigation
- [ ] WCAG 2.1 AA compliance

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add ARIA)

---

### 6.4 Mobile-Optimized Layouts
**Purpose**: Responsive design
**Effort**: 3-4 hours
**Impact**: MEDIUM

**Tasks**:
- [ ] Single column on mobile (<768px)
- [ ] Touch-friendly buttons
- [ ] Swipe to navigate panels
- [ ] Simplified charts for small screens
- [ ] Bottom navigation bar

**Files**:
- `HoloLoom/visualization/html_renderer.py` (add mobile CSS)

---

## Summary by Priority

### ⭐ Immediate (Week 1-2) - Maximum Impact
1. **Force-directed network graphs** - stunning visuals
2. **Bottleneck auto-detection** - actionable insights
3. **Panel collapse/expand** - better UX
4. **True heatmaps** - semantic understanding

### 🎯 High Value (Week 3-4) - Production Features
5. **Export to PDF** - shareability
6. **Performance recommendations** - optimization
7. **Trend analysis** - historical context
8. **Sankey diagrams** - data flow visualization

### 🔧 Polish (Week 5-6) - Professional Quality
9. **Dark theme** - aesthetics
10. **Keyboard navigation** - power users
11. **Accessibility** - inclusivity
12. **Mobile optimization** - responsive

---

## Implementation Order (Recommended)

**Sprint 1 (This Week)**: Phase 1.1, 1.2, 2.3 (Graphs, Heatmaps, Bottlenecks)
**Sprint 2 (Next Week)**: Phase 2.1, 2.4, 3.1 (Interactivity, Export)
**Sprint 3**: Phase 5.1, 5.3 (Analytics, Recommendations)
**Sprint 4**: Phase 6 (Polish & Accessibility)

---

## Metrics for Success

- **Visual Appeal**: Dashboards look professional (Edward Tufte standard)
- **Performance**: No noticeable slowdown (<50ms overhead)
- **Usability**: Users find bottlenecks in <5 seconds
- **Adoption**: 80% of queries use dashboards (when enabled)
- **Shareability**: Users export/share dashboards regularly

---

**Status**: 📋 ROADMAP CREATED - Ready for execution
**Next Step**: Start with Phase 1.1 (Force-Directed Network Graphs)
