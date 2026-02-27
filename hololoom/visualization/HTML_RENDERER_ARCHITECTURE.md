# HTML Dashboard Renderer Architecture

**Created:** October 28, 2025
**Purpose:** Self-contained HTML dashboards generated from Spacetime fabrics

---

## Design Principles

1. **Standalone** - Single HTML file, no server required
2. **Beautiful** - Polished, professional visualization (not debug output)
3. **Interactive** - Click to drill down, hover for tooltips
4. **Fast** - Render <100ms even with complex dashboards
5. **Shareable** - Email/upload the HTML file, works anywhere

---

## Technology Stack

### Core Libraries (CDN, no build step)

```html
<!-- Visualization -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

<!-- Styling -->
<script src="https://cdn.tailwindcss.com"></script>

<!-- Interactive components (optional) -->
<script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>

<!-- Icons -->
<link href="https://cdn.jsdelivr.net/npm/lucide-static@latest/font/lucide.css" rel="stylesheet">
```

**Why this stack:**
- **Plotly.js:** Already used in existing dashboards, excellent Python → JS bridge
- **Tailwind CSS:** Rapid styling without custom CSS files
- **Alpine.js:** Lightweight reactivity for collapsible panels, tooltips
- **Lucide:** Clean icon set (used in dashboard/src)

---

## Architecture Overview

```
Spacetime (Python)
    ↓
DashboardStrategy (Python)
    ↓
Dashboard object (Python)
    ↓
HTMLRenderer.render() (Python)
    ↓
HTML string
    ↓
Write to file / Return to client
```

---

## HTML Structure Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ dashboard.title }}</title>

    <!-- CDN Libraries -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>

    <!-- Custom Styles (inlined) -->
    <style>
        /* Custom theme colors */
        :root {
            --primary: #6366f1;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        /* Panel animations */
        .panel {
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .panel:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        /* Plotly customization */
        .js-plotly-plot .plotly .modebar {
            right: 10px !important;
        }
    </style>
</head>

<body class="bg-gray-50 min-h-screen">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-2xl font-bold text-gray-900">
                        {{ dashboard.title }}
                    </h1>
                    <p class="text-sm text-gray-500 mt-1">
                        Generated {{ timestamp }} • Complexity: {{ complexity }}
                    </p>
                </div>
                <div class="flex gap-2">
                    <!-- Action buttons -->
                    <button onclick="exportData()" class="btn-secondary">
                        <i class="lucide lucide-download"></i> Export JSON
                    </button>
                    <button onclick="window.print()" class="btn-secondary">
                        <i class="lucide lucide-printer"></i> Print
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Dashboard Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Summary Cards (Always at top) -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {{ summary_cards }}
        </div>

        <!-- Main Dashboard Grid -->
        <div class="{{ grid_layout_class }}">
            {{ panels }}
        </div>

        <!-- Expandable Provenance Section -->
        <div x-data="{ open: false }" class="mt-8">
            <button @click="open = !open" class="w-full bg-white border border-gray-200 rounded-lg p-4 flex justify-between items-center hover:bg-gray-50">
                <span class="font-semibold text-gray-900">Full Computational Trace</span>
                <i class="lucide" :class="open ? 'lucide-chevron-up' : 'lucide-chevron-down'"></i>
            </button>
            <div x-show="open" x-collapse class="mt-2 bg-white border border-gray-200 rounded-lg p-6">
                <pre class="text-xs text-gray-700 overflow-x-auto">{{ trace_json }}</pre>
            </div>
        </div>
    </main>

    <!-- Embedded Data (for export) -->
    <script id="dashboard-data" type="application/json">
        {{ spacetime_json }}
    </script>

    <!-- Dashboard Logic -->
    <script>
        // Spacetime data
        const spacetimeData = JSON.parse(document.getElementById('dashboard-data').textContent);

        // Export function
        function exportData() {
            const dataStr = JSON.stringify(spacetimeData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'spacetime_{{ timestamp }}.json';
            link.click();
        }

        // Initialize Plotly charts
        {{ plotly_init_scripts }}

        // Interactive features
        {{ custom_interactions }}
    </script>
</body>
</html>
```

---

## Panel Template System

### Base Panel Structure

```html
<div class="panel bg-white rounded-lg shadow-sm border border-gray-200 p-6 {{ size_class }}">
    <!-- Panel Header -->
    <div class="flex justify-between items-start mb-4">
        <div>
            <h3 class="text-lg font-semibold text-gray-900">{{ panel.title }}</h3>
            <p class="text-sm text-gray-500 mt-1">{{ panel.subtitle }}</p>
        </div>
        <button onclick="expandPanel('{{ panel.id }}')" class="text-gray-400 hover:text-gray-600">
            <i class="lucide lucide-maximize-2"></i>
        </button>
    </div>

    <!-- Panel Content (varies by type) -->
    <div class="panel-content">
        {{ panel_specific_content }}
    </div>

    <!-- Panel Footer (optional metadata) -->
    <div class="panel-footer text-xs text-gray-400 mt-4 pt-4 border-t border-gray-100">
        {{ panel.metadata }}
    </div>
</div>
```

### Panel Type Templates

#### 1. Metric Card Panel

```html
<div class="text-center">
    <div class="text-4xl font-bold text-{{ color }}-600 mb-2">
        {{ value }}
    </div>
    <div class="text-sm text-gray-500">
        {{ label }}
    </div>
    <!-- Optional trend indicator -->
    <div class="mt-2 text-xs">
        <span class="inline-flex items-center px-2 py-1 rounded-full bg-{{ trend_color }}-100 text-{{ trend_color }}-800">
            <i class="lucide lucide-trending-{{ trend_direction }} mr-1"></i>
            {{ trend_value }}
        </span>
    </div>
</div>
```

#### 2. Timeline Panel (Waterfall Chart)

```html
<div id="timeline-{{ panel.id }}"></div>

<script>
    // Plotly waterfall chart for stage durations
    const timelineData = [{
        type: 'waterfall',
        x: {{ stage_names }},
        y: {{ stage_durations }},
        connector: { line: { color: 'rgb(63, 63, 63)' } },
        marker: {
            color: {{ stage_colors }}
        },
        text: {{ stage_labels }},
        textposition: 'outside'
    }];

    const timelineLayout = {
        title: '',
        showlegend: false,
        xaxis: { title: 'Stage' },
        yaxis: { title: 'Duration (ms)' },
        margin: { t: 20, r: 20, b: 60, l: 60 }
    };

    Plotly.newPlot('timeline-{{ panel.id }}', timelineData, timelineLayout, {
        responsive: true,
        displayModeBar: true
    });
</script>
```

#### 3. Trajectory Panel (3D Semantic Path)

```html
<div id="trajectory-{{ panel.id }}" style="height: 400px;"></div>

<script>
    // 3D scatter plot of semantic trajectory
    const trajectoryData = [{
        type: 'scatter3d',
        mode: 'markers+lines',
        x: {{ coords_x }},
        y: {{ coords_y }},
        z: {{ coords_z }},
        marker: {
            size: 6,
            color: {{ velocities }},
            colorscale: 'Viridis',
            colorbar: { title: 'Velocity' },
            line: { color: 'rgba(0,0,0,0.1)', width: 0.5 }
        },
        line: {
            color: 'rgba(100,100,100,0.3)',
            width: 2
        },
        text: {{ word_labels }},
        hovertemplate: '<b>%{text}</b><br>Velocity: %{marker.color:.3f}<extra></extra>'
    }];

    const trajectoryLayout = {
        title: '',
        scene: {
            xaxis: { title: 'PC1' },
            yaxis: { title: 'PC2' },
            zaxis: { title: 'PC3' },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.5 }
            }
        },
        margin: { t: 20, r: 0, b: 0, l: 0 }
    };

    Plotly.newPlot('trajectory-{{ panel.id }}', trajectoryData, trajectoryLayout, {
        responsive: true
    });
</script>
```

#### 4. Heatmap Panel (Semantic Dimensions)

```html
<div id="heatmap-{{ panel.id }}" style="height: 400px;"></div>

<script>
    const heatmapData = [{
        type: 'heatmap',
        z: {{ values_matrix }},
        x: {{ dimension_names }},
        y: ['Query', 'Response'],
        colorscale: 'RdBu',
        zmid: 0,
        text: {{ text_annotations }},
        texttemplate: '%{text}',
        textfont: { size: 10 },
        hovertemplate: '%{y} - %{x}<br>Value: %{z:.3f}<extra></extra>'
    }];

    const heatmapLayout = {
        title: '',
        xaxis: { tickangle: -45 },
        yaxis: { autorange: 'reversed' },
        margin: { t: 20, r: 20, b: 100, l: 80 }
    };

    Plotly.newPlot('heatmap-{{ panel.id }}', heatmapData, heatmapLayout, {
        responsive: true
    });
</script>
```

#### 5. Network Panel (Knowledge Graph)

```html
<div id="network-{{ panel.id }}" style="height: 400px;"></div>

<script>
    // Force-directed graph using Plotly
    const networkData = [
        // Edges
        {
            type: 'scatter',
            mode: 'lines',
            x: {{ edge_x_coords }},
            y: {{ edge_y_coords }},
            line: { width: 1, color: 'rgba(150,150,150,0.3)' },
            hoverinfo: 'none'
        },
        // Nodes
        {
            type: 'scatter',
            mode: 'markers+text',
            x: {{ node_x_coords }},
            y: {{ node_y_coords }},
            marker: {
                size: {{ node_sizes }},
                color: {{ node_colors }},
                line: { width: 1, color: 'white' }
            },
            text: {{ node_labels }},
            textposition: 'top center',
            hovertemplate: '<b>%{text}</b><br>Connections: %{marker.size}<extra></extra>'
        }
    ];

    const networkLayout = {
        title: '',
        showlegend: false,
        xaxis: { visible: false },
        yaxis: { visible: false },
        margin: { t: 20, r: 20, b: 20, l: 20 },
        hovermode: 'closest'
    };

    Plotly.newPlot('network-{{ panel.id }}', networkData, networkLayout, {
        responsive: true
    });
</script>
```

#### 6. Distribution Panel (Bar Chart)

```html
<div id="distribution-{{ panel.id }}" style="height: 300px;"></div>

<script>
    const distributionData = [{
        type: 'bar',
        x: {{ labels }},
        y: {{ values }},
        marker: {
            color: {{ bar_colors }},
            line: { width: 1, color: 'white' }
        },
        text: {{ value_labels }},
        textposition: 'outside',
        hovertemplate: '<b>%{x}</b><br>%{y}<extra></extra>'
    }];

    const distributionLayout = {
        title: '',
        xaxis: { tickangle: -45 },
        yaxis: { title: '{{ y_axis_label }}' },
        margin: { t: 20, r: 20, b: 100, l: 60 }
    };

    Plotly.newPlot('distribution-{{ panel.id }}', distributionData, distributionLayout, {
        responsive: true
    });
</script>
```

---

## Grid Layout System

### Layout Classes (Tailwind Grid)

```python
LAYOUT_CONFIGS = {
    "metric": "grid grid-cols-1 gap-6",  # Single column, simple
    "flow": "grid grid-cols-1 md:grid-cols-2 gap-6",  # Two columns
    "research": "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",  # Three columns
    "adaptive": "grid grid-cols-1 md:grid-cols-2 gap-6"  # Responsive two columns
}

PANEL_SIZE_CLASSES = {
    "small": "md:col-span-1",  # 1/3 width on desktop
    "medium": "md:col-span-1 lg:col-span-1",  # 1/2 width
    "large": "md:col-span-2",  # 2/3 width
    "full-width": "md:col-span-2 lg:col-span-3"  # Full width
}
```

### Example Layout: Research Dashboard

```html
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    <!-- Row 1: Two medium panels -->
    <div class="md:col-span-1">{{ trajectory_panel }}</div>
    <div class="md:col-span-1">{{ network_panel }}</div>

    <!-- Row 2: Full-width heatmap -->
    <div class="md:col-span-2 lg:col-span-3">{{ heatmap_panel }}</div>

    <!-- Row 3: Full-width timeline -->
    <div class="md:col-span-2 lg:col-span-3">{{ timeline_panel }}</div>

    <!-- Row 4: Two medium panels -->
    <div class="md:col-span-1">{{ distribution_panel }}</div>
    <div class="md:col-span-1">{{ metrics_panel }}</div>
</div>
```

---

## Python Implementation Structure

```python
# File: HoloLoom/visualization/html_renderer.py

from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from jinja2 import Template

class HTMLRenderer:
    """Renders Dashboard objects as standalone HTML files."""

    def __init__(self):
        self.template_loader = TemplateLoader()
        self.panel_renderers = {
            'metric': self._render_metric_panel,
            'timeline': self._render_timeline_panel,
            'trajectory': self._render_trajectory_panel,
            'heatmap': self._render_heatmap_panel,
            'network': self._render_network_panel,
            'distribution': self._render_distribution_panel,
            'text': self._render_text_panel
        }

    def render(self, dashboard: Dashboard) -> str:
        """
        Render Dashboard to HTML string.

        Args:
            dashboard: Dashboard object to render

        Returns:
            Complete HTML string (ready to write to file)
        """
        # 1. Render summary cards
        summary_cards_html = self._render_summary_cards(dashboard)

        # 2. Render main panels
        panels_html = self._render_panels(dashboard.panels)

        # 3. Render provenance section
        trace_json = json.dumps(dashboard.spacetime.trace, indent=2, default=str)

        # 4. Generate Plotly init scripts
        plotly_scripts = self._generate_plotly_scripts(dashboard.panels)

        # 5. Get grid layout class
        grid_layout_class = LAYOUT_CONFIGS[dashboard.layout]

        # 6. Render main template
        template = self.template_loader.get_template('main.html')
        html = template.render(
            dashboard=dashboard,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            complexity=dashboard.spacetime.metadata.get('complexity', 'UNKNOWN'),
            summary_cards=summary_cards_html,
            panels=panels_html,
            grid_layout_class=grid_layout_class,
            trace_json=trace_json,
            spacetime_json=json.dumps(dashboard.spacetime.to_dict(), indent=2),
            plotly_init_scripts=plotly_scripts,
            custom_interactions=self._generate_interactions(dashboard)
        )

        return html

    def render_to_file(self, dashboard: Dashboard, path: str) -> Path:
        """
        Render Dashboard and save to HTML file.

        Args:
            dashboard: Dashboard object to render
            path: Output file path

        Returns:
            Path to saved file
        """
        html = self.render(dashboard)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')
        return output_path

    def _render_summary_cards(self, dashboard: Dashboard) -> str:
        """Render the 4 summary metric cards."""
        spacetime = dashboard.spacetime

        cards = [
            {
                'label': 'Confidence',
                'value': f"{spacetime.confidence:.2f}",
                'color': 'green' if spacetime.confidence >= 0.7 else 'yellow',
                'icon': 'check-circle'
            },
            {
                'label': 'Duration',
                'value': f"{spacetime.trace.duration_ms:.1f}ms",
                'color': 'blue',
                'icon': 'clock'
            },
            {
                'label': 'Tool',
                'value': spacetime.tool_used,
                'color': 'purple',
                'icon': 'wrench'
            },
            {
                'label': 'Threads',
                'value': str(len(spacetime.trace.threads_activated)),
                'color': 'indigo',
                'icon': 'git-branch'
            }
        ]

        template = self.template_loader.get_template('summary_card.html')
        return '\n'.join([template.render(card=card) for card in cards])

    def _render_panels(self, panels: List[Panel]) -> str:
        """Render all panels in dashboard."""
        rendered = []
        for panel in panels:
            renderer = self.panel_renderers.get(panel.type)
            if renderer:
                html = renderer(panel)
                rendered.append(html)
        return '\n'.join(rendered)

    def _render_metric_panel(self, panel: Panel) -> str:
        """Render a metric card panel."""
        template = self.template_loader.get_template('panel_metric.html')
        return template.render(panel=panel)

    def _render_timeline_panel(self, panel: Panel) -> str:
        """Render a timeline (waterfall) panel."""
        # Extract stage durations from panel data
        stage_durations = panel.data.get('stage_durations', {})

        # Convert to Plotly-friendly format
        stages = list(stage_durations.keys())
        durations = list(stage_durations.values())
        colors = self._get_stage_colors(stages)

        template = self.template_loader.get_template('panel_timeline.html')
        return template.render(
            panel=panel,
            stage_names=json.dumps(stages),
            stage_durations=json.dumps(durations),
            stage_colors=json.dumps(colors),
            stage_labels=json.dumps([f"{d:.1f}ms" for d in durations])
        )

    # ... other panel renderers ...

    def _generate_plotly_scripts(self, panels: List[Panel]) -> str:
        """Generate initialization scripts for all Plotly charts."""
        # Already embedded in panel templates
        return ""

    def _generate_interactions(self, dashboard: Dashboard) -> str:
        """Generate custom interaction JavaScript."""
        return """
        function expandPanel(panelId) {
            // Open panel in modal or new window
            console.log('Expanding panel:', panelId);
        }
        """

    def _get_stage_colors(self, stages: List[str]) -> List[str]:
        """Get color for each stage type."""
        color_map = {
            'features': '#6366f1',
            'retrieval': '#10b981',
            'decision': '#f59e0b',
            'execution': '#ef4444'
        }
        return [color_map.get(s.lower(), '#6b7280') for s in stages]


class TemplateLoader:
    """Loads HTML templates from files or returns inline templates."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        self.template_cache = {}

    def get_template(self, name: str) -> Template:
        """Get Jinja2 template by name."""
        if name in self.template_cache:
            return self.template_cache[name]

        # Try to load from file
        template_path = self.template_dir / name
        if template_path.exists():
            template_str = template_path.read_text()
        else:
            # Fall back to inline templates
            template_str = INLINE_TEMPLATES.get(name, "")

        template = Template(template_str)
        self.template_cache[name] = template
        return template


# Inline template fallbacks (when template files don't exist)
INLINE_TEMPLATES = {
    'summary_card.html': '''
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div class="flex items-center justify-between">
            <div>
                <p class="text-sm text-gray-500">{{ card.label }}</p>
                <p class="text-2xl font-bold text-{{ card.color }}-600 mt-1">{{ card.value }}</p>
            </div>
            <i class="lucide lucide-{{ card.icon }} text-{{ card.color }}-400 text-3xl"></i>
        </div>
    </div>
    ''',

    # ... more inline templates ...
}
```

---

## Data Flow: Spacetime → HTML

```python
# Example: Timeline panel generation

# 1. Spacetime contains trace with stage_durations
spacetime.trace.stage_durations = {
    'features': 25.3,
    'retrieval': 45.2,
    'decision': 15.1,
    'execution': 14.4
}

# 2. DashboardConstructor creates PanelSpec
panel_spec = PanelSpec(
    type='timeline',
    data_source='trace.stage_durations',
    size='full-width',
    priority=3
)

# 3. PanelGenerator creates Panel with data
panel = Panel(
    id='timeline_001',
    type='timeline',
    title='Execution Timeline',
    subtitle='Stage-by-stage breakdown',
    data={
        'stage_durations': spacetime.trace.stage_durations
    },
    size='full-width'
)

# 4. HTMLRenderer converts to HTML
html = renderer._render_timeline_panel(panel)

# Result: Plotly waterfall chart embedded in HTML
```

---

## Performance Optimizations

### 1. Template Caching
```python
# Cache compiled Jinja2 templates
self.template_cache[name] = Template(template_str)
```

### 2. Lazy Plotly Initialization
```python
# Only initialize visible charts, defer off-screen charts
<script>
    // Use Intersection Observer to init charts when scrolled into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.dataset.initialized) {
                initChart(entry.target.id);
                entry.target.dataset.initialized = 'true';
            }
        });
    });
</script>
```

### 3. Minify Output (Optional)
```python
# Use htmlmin to compress output
from htmlmin import minify
html = minify(html, remove_comments=True, remove_empty_space=True)
```

---

## Output Examples

### Metric Dashboard (LITE mode)
- File size: ~50KB
- Render time: ~20ms
- Panels: 4 summary cards + 1 text panel

### Flow Dashboard (FAST mode)
- File size: ~200KB
- Render time: ~50ms
- Panels: 4 summary cards + 2 medium charts + 1 full-width chart

### Research Dashboard (RESEARCH mode)
- File size: ~500KB
- Render time: ~100ms
- Panels: 4 summary cards + 6 detailed panels + full trace

---

## Testing Strategy

```python
# File: HoloLoom/tests/unit/test_html_renderer.py

def test_render_basic_dashboard():
    """Test rendering a minimal dashboard."""
    spacetime = create_test_spacetime()
    dashboard = Dashboard(
        title="Test Dashboard",
        layout="metric",
        panels=[Panel(type='metric', data={'value': 0.87})],
        spacetime=spacetime
    )

    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    assert '<html' in html
    assert 'Test Dashboard' in html
    assert 'plotly' in html.lower()

def test_render_timeline_panel():
    """Test timeline panel rendering."""
    panel = Panel(
        type='timeline',
        data={'stage_durations': {'features': 25.3, 'retrieval': 45.2}}
    )

    renderer = HTMLRenderer()
    html = renderer._render_timeline_panel(panel)

    assert 'waterfall' in html
    assert '25.3' in html
    assert '45.2' in html

def test_file_output():
    """Test rendering to file."""
    dashboard = create_test_dashboard()
    renderer = HTMLRenderer()

    output_path = renderer.render_to_file(dashboard, '/tmp/test_dashboard.html')

    assert output_path.exists()
    assert output_path.stat().st_size > 1000  # Non-trivial file
```

---

## Next Steps

1. ✅ **Architecture designed**
2. ⏳ **Implement HTMLRenderer core** (next task)
3. ⏳ **Create inline templates for 3 panel types**
4. ⏳ **Test with sample Spacetime**
5. ⏳ **Add remaining panel types**

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Templating | Jinja2 | Python-native, powerful, fast |
| Visualization | Plotly.js | Already used, excellent Python bridge |
| Styling | Tailwind CSS (CDN) | No build step, rapid development |
| Interactivity | Alpine.js | Lightweight (~15KB), no build step |
| Distribution | Standalone HTML | No server required, highly shareable |
| Data embedding | JSON in `<script>` tag | Enables client-side export, searchable |

---

Ready to implement Phase B: Working prototype with 2-3 panel types.