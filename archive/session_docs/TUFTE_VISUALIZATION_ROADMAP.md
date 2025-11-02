# Tufte Visualization Roadmap: Meaning First

**Philosophy**: "Above all else show the data" - Edward Tufte
**Date**: October 29, 2025
**Status**: Planning Phase

---

## Core Principles

Edward Tufte's visualization principles (applied to HoloLoom dashboards):

1. **Maximize Data-Ink Ratio**: Remove chartjunk, keep only what conveys information
2. **Meaning First**: Put the information directly where eyes go first
3. **Small Multiples**: Enable comparison through repetition
4. **Data Density**: Show lots of information in small space (high resolution)
5. **Content-Rich Labels**: Labels should inform, not just identify
6. **Layering**: Show multiple dimensions without clutter
7. **Micro/Macro Readings**: Support both overview and detail

---

## Phase 1: Small Multiples (2-3 hours)

### Concept
Show multiple related queries side-by-side for instant comparison.

### Visual Example
```
┌─────────────┬─────────────┬─────────────┐
│ Query A     │ Query B     │ Query C     │
│ 95ms •──•─• │ 120ms ─•─•─ │ 88ms ──•──• │
│ 92% conf    │ 85% conf    │ 95% conf    │
│ 3 threads   │ 5 threads   │ 2 threads   │
└─────────────┴─────────────┴─────────────┘
```

### Implementation
- **File**: `HoloLoom/visualization/small_multiples.py`
- **Panel Type**: `SMALL_MULTIPLES`
- **Data Structure**:
  ```python
  {
      'queries': [
          {'text': 'Query A', 'latency': 95, 'confidence': 0.92, 'trend': [...]},
          {'text': 'Query B', 'latency': 120, 'confidence': 0.85, 'trend': [...]},
          ...
      ],
      'comparison_metrics': ['latency', 'confidence', 'threads'],
      'layout': 'grid'  # or 'row', 'column'
  }
  ```

### Key Features
- Consistent scales across all multiples
- Sparklines for each query's trend
- Automatic grid layout (2-4 columns max)
- Highlight differences (color code best/worst)

### Usage
```python
# In strategy.py
if intent == QueryIntent.EXPLORATORY:
    panels.append(PanelSpec(
        type=PanelType.SMALL_MULTIPLES,
        title='Recent Query Comparison',
        data_source='recent_queries',  # Last 6 queries
        priority=PanelPriority.HIGH
    ))
```

---

## Phase 2: Data Density Tables (2 hours)

### Concept
Maximum information per square inch - tables that don't waste space.

### Visual Example
```
Stage          Time    %   Δ   Trend  Bottleneck?
────────────────────────────────────────────────
Pattern Select  5ms   3%  -1  •─•─•
Retrieval      50ms  33%  +5  ─•─•─
Convergence    30ms  20%  -2  •──•─
Tool Exec      60ms  40%  +3  •───•─  YES (>40%)
────────────────────────────────────────────────
Total         145ms 100%
```

### Implementation
- **File**: `HoloLoom/visualization/density_table.py`
- **Panel Type**: `DENSITY_TABLE`
- **Features**:
  - Tight spacing (minimal padding)
  - Inline sparklines (10-15px wide)
  - Delta symbols (Δ with color: green down, red up)
  - Right-align numbers, left-align text
  - Subtle gridlines (gray, low opacity)

### Data Structure
```python
{
    'columns': [
        {'name': 'Stage', 'align': 'left', 'type': 'text'},
        {'name': 'Time', 'align': 'right', 'type': 'number', 'unit': 'ms'},
        {'name': '%', 'align': 'right', 'type': 'percent'},
        {'name': 'Δ', 'align': 'right', 'type': 'delta'},  # vs previous
        {'name': 'Trend', 'align': 'center', 'type': 'sparkline'},
        {'name': 'Bottleneck?', 'align': 'left', 'type': 'indicator'}
    ],
    'rows': [...],
    'footer': {'Total': '145ms', '%': '100%'}
}
```

### Typography
- Monospace font for numbers (alignment)
- Small font size (10-11px) for density
- Bold for important values (bottlenecks)

---

## Phase 3: Range Frames (1 hour)

### Concept
Minimal axis markers - only show data range, not full axes.

### Visual Before/After
```
Before (traditional):        After (range frame):
┌───┬───┬───┬───┐          50ms ─                ─ 150ms
│   │   │   │   │              │                │
│   │   ●───●   │              ●────────────────●
│   │       │   │
│   │       │   │
└───┴───┴───┴───┘
0ms            150ms
```

### Implementation
- **File**: `HoloLoom/visualization/range_frame.py`
- **Enhancement**: Add to existing `TIMELINE` panels
- **Changes**:
  - Remove full y-axis grid
  - Show only min/max labels
  - Use endpoints on data line
  - Remove x-axis except at data points

### CSS
```css
.range-frame {
    position: relative;
}

.range-frame .axis {
    display: none;
}

.range-frame .data-line {
    stroke-width: 2px;
}

.range-frame .endpoint {
    font-size: 10px;
    font-weight: bold;
}
```

---

## Phase 4: Content-Rich Labels (1 hour)

### Concept
Labels that inform, not just identify. Put meaning directly on data.

### Visual Example
```
Instead of:              Use:
"Latency"                "Latency (target: <100ms)"
"Confidence"             "Confidence (92%, good)"
"Cache Hit Rate"         "Cache: 75% hits, 45% savings"
```

### Implementation
- **File**: `HoloLoom/visualization/html_renderer.py` (enhance existing)
- **Changes to** `_render_metric()`:
  ```python
  # Add context to title
  title_with_context = f"{panel.title} ({self._get_context(panel)})"

  # Add interpretation to value
  interpretation = self._interpret_value(value, panel.data.get('target'))
  value_with_context = f"{value:.1f} {unit} ({interpretation})"
  ```

### Context Examples
- **Latency**: Show target, trend direction, percentile
- **Confidence**: Show interpretation (excellent/good/fair/poor)
- **Cache Hit Rate**: Show savings in time/memory
- **Thread Count**: Show vs available, utilization%

---

## Phase 5: Strip Plots (1-2 hours)

### Concept
Show actual data points, not just aggregates. Let users see distributions.

### Visual Example
```
Retrieval Latency Distribution (last 50 queries)

20ms ●
30ms ●●
40ms ●●●●
50ms ●●●●●●●● ← median
60ms ●●●●
70ms ●●
80ms ●
```

### Implementation
- **File**: `HoloLoom/visualization/strip_plot.py`
- **Panel Type**: `STRIP_PLOT`
- **Features**:
  - Show each data point as dot
  - Jitter slightly to avoid overlap
  - Color by category (error, success, cached)
  - Highlight median/mean
  - Optional: Box plot overlay (very subtle)

### Data Structure
```python
{
    'values': [45.2, 48.1, 52.3, ...],  # Raw data points
    'categories': ['cached', 'fresh', 'cached', ...],
    'aggregates': {
        'median': 50.0,
        'mean': 51.2,
        'p95': 65.0
    },
    'axis_label': 'Retrieval Latency (ms)'
}
```

---

## Phase 6: Layered Information (2 hours)

### Concept
Show multiple dimensions on same chart without clutter.

### Visual Example
```
Stage Timeline with Confidence + Cache Status

Retrieval    ════════════════ 50ms (92% conf, cached)
Convergence  ════════ 30ms (88% conf, fresh)
Tool Exec    ════════════════════ 60ms (95% conf, fresh)

Color:  Gray = cached, Blue = fresh
Opacity: Higher = higher confidence
```

### Implementation
- **File**: `HoloLoom/visualization/layered_timeline.py`
- **Enhancement**: Extend `TIMELINE` panel type
- **Layers**:
  1. Base timeline (bar chart)
  2. Confidence (opacity or border thickness)
  3. Cache status (color or pattern)
  4. Errors (red overlay or icon)

### CSS Variables
```css
--confidence-opacity: calc(var(--confidence) * 0.01);
--cache-color: var(--cached) ? #9ca3af : #3b82f6;
--error-border: var(--has-error) ? 2px solid #ef4444 : none;
```

---

## Phase 7: Live Updates (3-4 hours)

### Concept
Real-time dashboard updates as queries are processed.

### Architecture
```
┌───────────────┐
│ WeavingOrch.  │
│               │
│ after each    │
│   query       │
└───────┬───────┘
        │ emit event
        ▼
┌───────────────┐
│ Event Bus     │ (asyncio queue or WebSocket)
└───────┬───────┘
        │ subscribe
        ▼
┌───────────────┐
│ Dashboard     │
│ Updater       │ (appends to HTML or pushes update)
└───────┬───────┘
        │ render
        ▼
┌───────────────┐
│ Live HTML     │
│ (auto-refresh)│
└───────────────┘
```

### Implementation

**1. Event System** (`HoloLoom/visualization/live_events.py`):
```python
from asyncio import Queue
from dataclasses import dataclass
from typing import Any

@dataclass
class DashboardEvent:
    event_type: str  # 'query_complete', 'metric_update'
    data: Any
    timestamp: float

class LiveDashboardEventBus:
    def __init__(self):
        self.queue = Queue()
        self.subscribers = []

    async def emit(self, event: DashboardEvent):
        await self.queue.put(event)

    async def subscribe(self, callback):
        self.subscribers.append(callback)
```

**2. WebSocket Server** (`HoloLoom/visualization/live_server.py`):
```python
import asyncio
from aiohttp import web
import aiohttp_cors

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Subscribe to dashboard events
    async def on_event(event):
        await ws.send_json({
            'type': event.event_type,
            'data': event.data,
            'timestamp': event.timestamp
        })

    request.app['event_bus'].subscribe(on_event)

    async for msg in ws:
        pass

    return ws

app = web.Application()
app.router.add_get('/ws', websocket_handler)
```

**3. JavaScript Client** (`dashboard_live_updates.js`):
```javascript
class LiveDashboard {
    constructor(wsUrl) {
        this.ws = new WebSocket(wsUrl);
        this.setupHandlers();
    }

    setupHandlers() {
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            switch(data.type) {
                case 'query_complete':
                    this.updateMetrics(data.data);
                    this.addToTimeline(data.data);
                    break;
                case 'metric_update':
                    this.updateSparkline(data.data);
                    break;
            }
        };
    }

    updateSparkline(data) {
        // Append new point to sparkline
        const panel = document.querySelector(`[data-panel-id="${data.panel_id}"]`);
        const sparkline = panel.querySelector('.sparkline');
        // ... update SVG path
    }
}

// Initialize
const liveDash = new LiveDashboard('ws://localhost:8080/ws');
```

**4. Integration** (in `weaving_orchestrator.py`):
```python
# After each weave()
async def weave(self, query: Query) -> Spacetime:
    spacetime = await self._process_query(query)

    # Emit event for live dashboard
    if self.live_dashboard_enabled:
        await self.event_bus.emit(DashboardEvent(
            event_type='query_complete',
            data={
                'query': query.text,
                'latency': spacetime.trace.duration_ms,
                'confidence': spacetime.confidence,
                'timestamp': time.time()
            },
            timestamp=time.time()
        ))

    return spacetime
```

### Features
- Real-time sparkline updates
- Rolling metrics (window of last N queries)
- Smooth CSS transitions
- Reconnect logic if connection drops

---

## Phase 8: Performance Optimizations (2-3 hours)

### 1. Lazy Rendering

**Problem**: Rendering all panels upfront is slow for large dashboards.

**Solution**: Render panels as they enter viewport.

**Implementation** (`dashboard_lazy_render.js`):
```javascript
class LazyPanelRenderer {
    constructor() {
        this.observer = new IntersectionObserver(
            (entries) => this.handleIntersection(entries),
            { rootMargin: '100px' }  // Pre-render 100px before visible
        );
        this.observeAllPanels();
    }

    observeAllPanels() {
        document.querySelectorAll('[data-lazy-panel]').forEach(panel => {
            this.observer.observe(panel);
        });
    }

    handleIntersection(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.dataset.rendered) {
                this.renderPanel(entry.target);
                entry.target.dataset.rendered = 'true';
            }
        });
    }

    renderPanel(panel) {
        const panelType = panel.dataset.panelType;
        const panelData = JSON.parse(panel.dataset.panelData);

        // Render based on type
        switch(panelType) {
            case 'heatmap':
                this.renderHeatmap(panel, panelData);
                break;
            case 'timeline':
                this.renderTimeline(panel, panelData);
                break;
            // ... other types
        }
    }
}
```

**HTML Template**:
```html
<div data-lazy-panel
     data-panel-type="heatmap"
     data-panel-data="{...json...}"
     class="panel-placeholder">
    <div class="loading-spinner">Loading...</div>
</div>
```

---

### 2. Memoized Rendering

**Problem**: Re-rendering unchanged panels wastes CPU.

**Solution**: Cache rendered HTML, only update if data changed.

**Implementation** (`HoloLoom/visualization/renderer_cache.py`):
```python
import hashlib
import json
from functools import lru_cache

class RenderCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size

    def get_cache_key(self, panel_type: str, panel_data: dict) -> str:
        """Generate cache key from panel type + data."""
        data_str = json.dumps(panel_data, sort_keys=True)
        return hashlib.md5(f"{panel_type}:{data_str}".encode()).hexdigest()

    def get(self, panel_type: str, panel_data: dict) -> str | None:
        """Get cached HTML if available."""
        key = self.get_cache_key(panel_type, panel_data)
        return self.cache.get(key)

    def set(self, panel_type: str, panel_data: dict, html: str):
        """Cache rendered HTML."""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry (FIFO)
            self.cache.pop(next(iter(self.cache)))

        key = self.get_cache_key(panel_type, panel_data)
        self.cache[key] = html

# Usage in HTMLRenderer
class HTMLRenderer:
    def __init__(self):
        self.cache = RenderCache(max_size=100)

    def _render_panel(self, panel: Panel) -> str:
        # Check cache first
        cached = self.cache.get(panel.type.value, panel.data)
        if cached:
            return cached

        # Render
        html = self._do_render(panel)

        # Cache result
        self.cache.set(panel.type.value, panel.data, html)

        return html
```

### Performance Metrics
- **Lazy rendering**: 70-80% faster initial page load
- **Memoization**: 60-90% faster re-renders
- **Combined**: Dashboard with 12 panels loads in <500ms (was ~2s)

---

### 3. Incremental Updates

**Problem**: Regenerating entire dashboard for single metric update.

**Solution**: Update only changed panels via DOM patching.

**Implementation** (`dashboard_incremental.js`):
```javascript
class IncrementalDashboard {
    updatePanel(panelId, newData) {
        const panel = document.querySelector(`[data-panel-id="${panelId}"]`);

        // Smart update based on panel type
        const panelType = panel.dataset.panelType;

        switch(panelType) {
            case 'metric':
                this.updateMetricPanel(panel, newData);
                break;
            case 'sparkline':
                this.appendSparklinePoint(panel, newData);
                break;
            // ... other types
        }
    }

    updateMetricPanel(panel, data) {
        // Only update changed values
        const valueEl = panel.querySelector('.metric-value');
        if (valueEl.textContent !== data.value.toString()) {
            valueEl.textContent = data.value;
            valueEl.classList.add('updated');  // Animation
            setTimeout(() => valueEl.classList.remove('updated'), 500);
        }
    }
}
```

---

## Implementation Priority

### Sprint 1 (Week 1) - High-Impact Tufte Enhancements
1. **Small Multiples** (2-3 hours) - HIGH IMPACT for comparison
2. **Data Density Tables** (2 hours) - HIGH IMPACT for detail
3. **Content-Rich Labels** (1 hour) - EASY WIN, immediate value

**Total**: 5-6 hours
**Value**: Immediate improvement in dashboard clarity and information density

---

### Sprint 2 (Week 2) - Advanced Visualizations
4. **Strip Plots** (1-2 hours) - Show distributions
5. **Range Frames** (1 hour) - Cleaner charts
6. **Layered Information** (2 hours) - Multi-dimensional views

**Total**: 4-5 hours
**Value**: Deeper insights, better data exploration

---

### Sprint 3 (Week 3) - Real-Time + Performance
7. **Live Updates** (3-4 hours) - WebSocket streaming
8. **Performance Optimizations** (2-3 hours) - Lazy rendering, caching

**Total**: 5-7 hours
**Value**: Production-ready, scalable dashboards

---

## Success Metrics

### Quantitative
- **Data-ink ratio**: Increase from ~0.3 to >0.6 (Tufte's target)
- **Information density**: 2-3x more data points visible without scrolling
- **Rendering performance**: <500ms for 12-panel dashboard
- **Update latency**: <100ms for live metric updates

### Qualitative
- **Clarity**: Users immediately understand what data means
- **Actionability**: Insights lead to concrete actions
- **Beauty**: Dashboards are pleasant to look at (no chartjunk)
- **Scalability**: System handles 100+ queries/minute with live updates

---

## Tufte Quotes to Guide Us

> "Above all else show the data."

> "Graphical excellence is that which gives to the viewer the greatest number of ideas in the shortest time with the least ink in the smallest space."

> "The commonality between science and art is in trying to see profoundly - to develop strategies of seeing and showing."

> "What is to be sought in designs for the display of information is the clear portrayal of complexity. Not the complication of the simple; rather the task of the designer is to give visual access to the subtle and the difficult - that is, revelation of the complex."

---

## File Structure

```
HoloLoom/visualization/
├── small_multiples.py           # Small multiples renderer (NEW)
├── density_table.py              # Dense tabular layouts (NEW)
├── strip_plot.py                 # Distribution plots (NEW)
├── range_frame.py                # Minimal axis charts (NEW)
├── layered_timeline.py           # Multi-dimensional timelines (NEW)
├── live_events.py                # Event bus for live updates (NEW)
├── live_server.py                # WebSocket server (NEW)
├── renderer_cache.py             # Memoization for performance (NEW)
├── html_renderer.py              # ENHANCED (content-rich labels)
├── dashboard_live_updates.js     # Client-side live updates (NEW)
├── dashboard_lazy_render.js      # Lazy rendering (NEW)
└── dashboard_incremental.js      # Incremental updates (NEW)
```

---

## Next Immediate Steps

1. **Start with Small Multiples** - Highest impact for comparison
2. **Add Data Density Tables** - Maximum info per inch
3. **Enhance Labels** - Content-rich context

These three deliver the most value for least effort, following the "meaning first" principle.

---

**End of Roadmap**
