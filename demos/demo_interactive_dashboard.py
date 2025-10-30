"""
Interactive Dashboard Demo
===========================

Comprehensive showcase of all panel sizes with diverse interactive data.

Demonstrates:
- All panel sizes: 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, full-width, hero
- Interactive elements (charts, metrics, tables)
- Real-world data examples
- Modern CSS/HTML5 features
"""

import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.visualization.dashboard import (
    Dashboard, Panel, PanelType, PanelSize, LayoutType
)
from HoloLoom.visualization.html_renderer import HTMLRenderer


def create_interactive_dashboard() -> Dashboard:
    """Create comprehensive interactive dashboard with all panel sizes."""

    panels = []

    # ========================================================================
    # HERO Section - Tabbed Chart Showcase (5 charts in tabs)
    # ========================================================================
    # Use TEXT panel with custom tabbed HTML/JS for 5 charts
    panels.append(Panel(
        id="hero-tabbed-charts",
        type=PanelType.TEXT,
        title="HoloLoom Interactive Dashboard (HiD)",
        subtitle="Real-time system monitoring - 5 visualization modes",
        size=PanelSize.HERO,
        data={
            'truncate': False,  # DON'T truncate - we have full HTML/CSS/JS here!
            'content': """
                <div class="tabbed-hero">
                    <!-- Tab Navigation -->
                    <div class="tab-nav" role="tablist">
                        <button class="tab-button active" data-tab="trends" role="tab" aria-selected="true">
                            📈 Trends
                        </button>
                        <button class="tab-button" data-tab="timeline" role="tab" aria-selected="false">
                            ⏱️ Timeline
                        </button>
                        <button class="tab-button" data-tab="network" role="tab" aria-selected="false">
                            🕸️ Network
                        </button>
                        <button class="tab-button" data-tab="heatmap" role="tab" aria-selected="false">
                            🔥 Heatmap
                        </button>
                        <button class="tab-button" data-tab="resources" role="tab" aria-selected="false">
                            💾 Resources
                        </button>
                    </div>

                    <!-- Tab Content -->
                    <div class="tab-content active" id="tab-trends" role="tabpanel">
                        <div id="chart-trends" style="height: 600px;"></div>
                    </div>
                    <div class="tab-content" id="tab-timeline" role="tabpanel">
                        <div id="chart-timeline" style="height: 600px;"></div>
                    </div>
                    <div class="tab-content" id="tab-network" role="tabpanel">
                        <div id="chart-network" style="height: 600px;"></div>
                    </div>
                    <div class="tab-content" id="tab-heatmap" role="tabpanel">
                        <div id="chart-heatmap" style="height: 600px;"></div>
                    </div>
                    <div class="tab-content" id="tab-resources" role="tabpanel">
                        <div id="chart-resources" style="height: 600px;"></div>
                    </div>
                </div>

                <style>
                    .tabbed-hero {
                        margin-top: var(--space-4);
                    }

                    .tab-nav {
                        display: flex;
                        gap: var(--space-2);
                        margin-bottom: var(--space-6);
                        border-bottom: 2px solid var(--color-border-default);
                        padding-bottom: var(--space-2);
                    }

                    .tab-button {
                        padding: var(--space-3) var(--space-6);
                        background: transparent;
                        border: none;
                        border-bottom: 3px solid transparent;
                        color: var(--color-text-secondary);
                        font-size: var(--font-size-base);
                        font-weight: var(--font-weight-semibold);
                        cursor: pointer;
                        transition: all var(--transition-fast);
                        position: relative;
                        top: 2px;
                    }

                    .tab-button:hover {
                        color: var(--color-text-primary);
                        background: var(--color-surface-hover);
                        border-radius: var(--radius-md) var(--radius-md) 0 0;
                    }

                    .tab-button.active {
                        color: var(--color-accent-primary);
                        border-bottom-color: var(--color-accent-primary);
                        font-weight: var(--font-weight-bold);
                    }

                    .tab-content {
                        display: none;
                    }

                    .tab-content.active {
                        display: block;
                        animation: fadeIn 0.3s ease-in;
                    }

                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                </style>

                <script>
                (function() {
                    // Tab switching logic
                    const tabButtons = document.querySelectorAll('.tab-button');
                    const tabContents = document.querySelectorAll('.tab-content');

                    tabButtons.forEach(button => {
                        button.addEventListener('click', () => {
                            const tabId = button.dataset.tab;

                            // Update buttons
                            tabButtons.forEach(btn => {
                                btn.classList.remove('active');
                                btn.setAttribute('aria-selected', 'false');
                            });
                            button.classList.add('active');
                            button.setAttribute('aria-selected', 'true');

                            // Update content
                            tabContents.forEach(content => {
                                content.classList.remove('active');
                            });
                            document.getElementById('tab-' + tabId).classList.add('active');
                        });
                    });

                    // Initialize charts
                    const themeColors = getThemeColors();

                    // Chart 1: Multi-line Trends
                    const trendsData = [
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 120 + (i % 20) - 10 + Math.floor(i / 10) * 2),
                            name: 'Query Latency (ms)',
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#3b82f6', width: 3 }
                        },
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 850 + (i % 30) - 15 + Math.floor(i / 5) * 3),
                            name: 'Queries/Min',
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#22c55e', width: 3 }
                        },
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 15 - (i % 8) + Math.floor(i / 15)),
                            name: 'Errors/Min',
                            type: 'scatter',
                            mode: 'lines',
                            line: { color: '#ef4444', width: 3 }
                        }
                    ];

                    Plotly.newPlot('chart-trends', trendsData, {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: themeColors.text, family: 'Inter, sans-serif', size: 14 },
                        margin: { t: 20, r: 40, b: 60, l: 60 },
                        xaxis: { title: 'Time (seconds ago)', gridcolor: themeColors.grid },
                        yaxis: { title: 'Value', gridcolor: themeColors.grid },
                        showlegend: true,
                        legend: { orientation: 'h', y: -0.15 }
                    }, { responsive: true, displayModeBar: false });

                    // Chart 2: Timeline Waterfall
                    const timelineData = [{
                        x: [25, 30, 20, 20, 30],
                        y: ['Feature Extraction', 'Memory Retrieval', 'Warp Tensioning', 'Policy Decision', 'Tool Execution'],
                        type: 'bar',
                        orientation: 'h',
                        marker: {
                            color: ['#3b82f6', '#a855f7', '#22c55e', '#f97316', '#ef4444']
                        },
                        text: ['25ms (20%)', '30ms (24%)', '20ms (16%)', '20ms (16%)', '30ms (24%)'],
                        textposition: 'inside',
                        insidetextanchor: 'middle',
                        textfont: { color: 'white', size: 14 }
                    }];

                    Plotly.newPlot('chart-timeline', timelineData, {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: themeColors.text, family: 'Inter, sans-serif', size: 14 },
                        margin: { t: 20, r: 40, b: 40, l: 180 },
                        xaxis: { title: 'Duration (ms)', gridcolor: themeColors.grid },
                        yaxis: { gridcolor: themeColors.grid }
                    }, { responsive: true, displayModeBar: false });

                    // Chart 3: Network Graph (D3.js)
                    const networkSvg = d3.select('#chart-network')
                        .append('svg')
                        .attr('width', '100%')
                        .attr('height', '100%');

                    const networkWidth = document.getElementById('chart-network').clientWidth;
                    const networkHeight = 600;

                    const nodes = [
                        {id: 'Query', size: 20, color: '#3b82f6'},
                        {id: 'Memory', size: 18, color: '#22c55e'},
                        {id: 'Policy', size: 16, color: '#f97316'},
                        {id: 'Warp', size: 14, color: '#a855f7'},
                        {id: 'Tool', size: 12, color: '#ef4444'}
                    ];

                    const links = [
                        {source: 'Query', target: 'Memory'},
                        {source: 'Memory', target: 'Warp'},
                        {source: 'Warp', target: 'Policy'},
                        {source: 'Policy', target: 'Tool'}
                    ];

                    const simulation = d3.forceSimulation(nodes)
                        .force('link', d3.forceLink(links).id(d => d.id).distance(150))
                        .force('charge', d3.forceManyBody().strength(-400))
                        .force('center', d3.forceCenter(networkWidth / 2, networkHeight / 2));

                    const link = networkSvg.append('g')
                        .selectAll('line')
                        .data(links)
                        .enter().append('line')
                        .attr('stroke', '#cbd5e1')
                        .attr('stroke-width', 2);

                    const node = networkSvg.append('g')
                        .selectAll('g')
                        .data(nodes)
                        .enter().append('g')
                        .call(d3.drag()
                            .on('start', (event, d) => {
                                if (!event.active) simulation.alphaTarget(0.3).restart();
                                d.fx = d.x;
                                d.fy = d.y;
                            })
                            .on('drag', (event, d) => {
                                d.fx = event.x;
                                d.fy = event.y;
                            })
                            .on('end', (event, d) => {
                                if (!event.active) simulation.alphaTarget(0);
                                d.fx = null;
                                d.fy = null;
                            }));

                    node.append('circle')
                        .attr('r', d => d.size)
                        .attr('fill', d => d.color);

                    node.append('text')
                        .text(d => d.id)
                        .attr('x', 0)
                        .attr('y', d => d.size + 20)
                        .attr('text-anchor', 'middle')
                        .attr('fill', themeColors.text)
                        .style('font-size', '14px')
                        .style('font-weight', 'bold');

                    simulation.on('tick', () => {
                        link
                            .attr('x1', d => d.source.x)
                            .attr('y1', d => d.source.y)
                            .attr('x2', d => d.target.x)
                            .attr('y2', d => d.target.y);

                        node.attr('transform', d => `translate(${d.x},${d.y})`);
                    });

                    // Chart 4: Heatmap
                    const heatmapData = [{
                        z: [
                            [45, 95, 180, 450],
                            [42, 92, 175, 440],
                            [48, 98, 185, 460],
                            [52, 105, 195, 480],
                            [50, 102, 190, 470],
                            [46, 96, 182, 455]
                        ],
                        x: ['LITE', 'FAST', 'FULL', 'RESEARCH'],
                        y: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                        type: 'heatmap',
                        colorscale: [
                            [0, 'rgb(34, 197, 94)'],
                            [0.5, 'rgb(251, 191, 36)'],
                            [1, 'rgb(239, 68, 68)']
                        ],
                        hovertemplate: '<b>%{y}</b><br>%{x}: %{z}ms<extra></extra>',
                        colorbar: {
                            title: 'Latency (ms)',
                            titleside: 'right',
                            tickfont: { color: themeColors.text }
                        }
                    }];

                    Plotly.newPlot('chart-heatmap', heatmapData, {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: themeColors.text, family: 'Inter, sans-serif', size: 14 },
                        margin: { t: 20, r: 100, b: 60, l: 80 },
                        xaxis: { gridcolor: themeColors.grid },
                        yaxis: { title: 'Time of Day', gridcolor: themeColors.grid }
                    }, { responsive: true, displayModeBar: false });

                    // Chart 5: Stacked Area (Resources)
                    const resourcesData = [
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 2.1 + (i % 10) / 50),
                            name: 'Yarn Graph',
                            type: 'scatter',
                            mode: 'lines',
                            fill: 'tonexty',
                            line: { width: 0 },
                            fillcolor: 'rgba(59, 130, 246, 0.3)'
                        },
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 1.8 + (i % 8) / 40),
                            name: 'Warp Space',
                            type: 'scatter',
                            mode: 'lines',
                            fill: 'tonexty',
                            line: { width: 0 },
                            fillcolor: 'rgba(168, 85, 247, 0.3)'
                        },
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 1.4 - (i % 12) / 60),
                            name: 'Cache',
                            type: 'scatter',
                            mode: 'lines',
                            fill: 'tonexty',
                            line: { width: 0 },
                            fillcolor: 'rgba(34, 197, 94, 0.3)'
                        },
                        {
                            x: Array.from({length: 60}, (_, i) => i),
                            y: Array.from({length: 60}, (_, i) => 0.9 + (i % 6) / 100),
                            name: 'Features',
                            type: 'scatter',
                            mode: 'lines',
                            fill: 'tozeroy',
                            line: { width: 0 },
                            fillcolor: 'rgba(249, 115, 22, 0.3)'
                        }
                    ];

                    Plotly.newPlot('chart-resources', resourcesData, {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: themeColors.text, family: 'Inter, sans-serif', size: 14 },
                        margin: { t: 20, r: 40, b: 60, l: 60 },
                        xaxis: { title: 'Time (seconds ago)', gridcolor: themeColors.grid },
                        yaxis: { title: 'Memory (GB)', gridcolor: themeColors.grid },
                        showlegend: true,
                        legend: { orientation: 'h', y: -0.15 }
                    }, { responsive: true, displayModeBar: false });
                })();
                </script>
            """
        }
    ))

    # ========================================================================
    # TINY Panels - 1/6 width (6 per row on desktop)
    # ========================================================================
    tiny_metrics = [
        {'label': 'Active Users', 'value': '2.4K', 'raw': 2431, 'color': 'blue', 'icon': '👥'},
        {'label': 'Queries/Min', 'value': '847', 'raw': 847, 'color': 'green', 'icon': '⚡'},
        {'label': 'Errors', 'value': '12', 'raw': 12, 'color': 'red', 'icon': '⚠️'},
        {'label': 'CPU Usage', 'value': '42%', 'raw': 42, 'color': 'yellow', 'icon': '💻'},
        {'label': 'Memory', 'value': '6.2GB', 'raw': 6.2, 'color': 'purple', 'icon': '🧠'},
        {'label': 'Uptime', 'value': '99.9%', 'raw': 99.9, 'color': 'green', 'icon': '✅'},
    ]

    for i, metric in enumerate(tiny_metrics):
        panels.append(Panel(
            id=f"tiny-kpi-{i}",
            type=PanelType.METRIC,
            title=f"{metric['icon']} {metric['label']}",
            size=PanelSize.TINY,
            data={
                'value': metric['raw'],
                'formatted': metric['value'],
                'color': metric['color'],
                'label': metric['label']
            }
        ))

    # ========================================================================
    # COMPACT Panels - 1/4 width (4 per row on desktop) with sparklines
    # ========================================================================
    compact_metrics = [
        {
            'title': 'Query Latency',
            'value': 125.5,
            'formatted': '125.5ms',
            'trend': [145, 138, 132, 128, 125.5],
            'trend_direction': 'down',
            'color': 'green',
            'subtitle': 'Avg response time'
        },
        {
            'title': 'Confidence Score',
            'value': 0.98,
            'formatted': '98%',
            'trend': [0.94, 0.95, 0.96, 0.97, 0.98],
            'trend_direction': 'up',
            'color': 'blue',
            'subtitle': 'Model certainty'
        },
        {
            'title': 'Cache Hit Rate',
            'value': 0.83,
            'formatted': '83%',
            'trend': [0.78, 0.80, 0.81, 0.82, 0.83],
            'trend_direction': 'up',
            'color': 'purple',
            'subtitle': 'Memory efficiency'
        },
        {
            'title': 'Throughput',
            'value': 1847,
            'formatted': '1.8K/s',
            'trend': [1620, 1710, 1780, 1820, 1847],
            'trend_direction': 'up',
            'color': 'orange',
            'subtitle': 'Queries per second'
        },
    ]

    for i, metric in enumerate(compact_metrics):
        panels.append(Panel(
            id=f"compact-metric-{i}",
            type=PanelType.METRIC,
            title=metric['title'],
            subtitle=metric['subtitle'],
            size=PanelSize.COMPACT,
            data=metric
        ))

    # ========================================================================
    # SMALL Panels - 1/3 width (3 per row) - Line charts
    # ========================================================================

    # Latency over time
    panels.append(Panel(
        id="chart-latency-trend",
        type=PanelType.LINE,
        title="Latency Trend",
        subtitle="Last 24 hours",
        size=PanelSize.SMALL,
        data={
            'x': ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:59'],
            'y': [150, 145, 135, 128, 125, 122, 120],
            'label': 'Response Time (ms)'
        }
    ))

    # Query volume
    panels.append(Panel(
        id="chart-query-volume",
        type=PanelType.BAR,
        title="Query Volume",
        subtitle="By hour (today)",
        size=PanelSize.SMALL,
        data={
            'x': ['6am', '9am', '12pm', '3pm', '6pm', '9pm'],
            'y': [450, 820, 1240, 1680, 1420, 890],
            'label': 'Queries'
        }
    ))

    # Error distribution
    panels.append(Panel(
        id="chart-error-types",
        type=PanelType.BAR,
        title="Error Types",
        subtitle="Last 7 days",
        size=PanelSize.SMALL,
        data={
            'x': ['Timeout', 'Invalid', 'Auth', 'Rate Limit', 'Server'],
            'y': [45, 23, 12, 8, 5],
            'label': 'Count'
        }
    ))

    # ========================================================================
    # MEDIUM Panels - 1/2 width (2 per row) - Scatter plots
    # ========================================================================

    # Latency vs Confidence
    panels.append(Panel(
        id="scatter-latency-confidence",
        type=PanelType.SCATTER,
        title="Latency vs Confidence",
        subtitle="Query performance analysis",
        size=PanelSize.MEDIUM,
        data={
            'x': [95, 120, 145, 110, 130, 155, 100, 125, 140, 115],
            'y': [0.98, 0.96, 0.92, 0.97, 0.95, 0.90, 0.98, 0.96, 0.93, 0.97],
            'xlabel': 'Latency (ms)',
            'ylabel': 'Confidence',
            'labels': [f'Query {i+1}' for i in range(10)]
        }
    ))

    # Cache performance
    panels.append(Panel(
        id="scatter-cache-perf",
        type=PanelType.SCATTER,
        title="Cache Performance",
        subtitle="Hit rate vs speedup",
        size=PanelSize.MEDIUM,
        data={
            'x': [0.65, 0.72, 0.78, 0.83, 0.88, 0.91, 0.94, 0.96, 0.98],
            'y': [2.1, 2.8, 3.5, 4.2, 5.1, 6.8, 8.5, 11.2, 15.4],
            'xlabel': 'Hit Rate',
            'ylabel': 'Speedup (x)',
            'labels': [f'Shard {i+1}' for i in range(9)]
        }
    ))

    # ========================================================================
    # LARGE / TWO_THIRDS Panels - 2/3 width - Timeline
    # ========================================================================
    panels.append(Panel(
        id="timeline-processing",
        type=PanelType.TIMELINE,
        title="Query Processing Timeline",
        subtitle="Detailed stage breakdown for last query (125ms total)",
        size=PanelSize.LARGE,
        data={
            'stages': ['Feature Extraction', 'Memory Retrieval', 'Warp Tensioning', 'Policy Decision', 'Tool Execution'],
            'durations': [25, 30, 20, 20, 30],
            'percentages': [20.0, 24.0, 16.0, 16.0, 24.0],
            'colors': ['#3b82f6', '#a855f7', '#22c55e', '#f97316', '#ef4444']
        }
    ))

    # ========================================================================
    # THREE_QUARTERS Panels - 3/4 width - Heatmap
    # ========================================================================
    panels.append(Panel(
        id="heatmap-performance",
        type=PanelType.HEATMAP,
        title="Performance Heatmap",
        subtitle="Query latency by time and complexity",
        size=PanelSize.THREE_QUARTERS,
        data={
            'x_labels': ['LITE', 'FAST', 'FULL', 'RESEARCH'],
            'y_labels': ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            'values': [
                [45, 95, 180, 450],   # 00:00
                [42, 92, 175, 440],   # 04:00
                [48, 98, 185, 460],   # 08:00
                [52, 105, 195, 480],  # 12:00
                [50, 102, 190, 470],  # 16:00
                [46, 96, 182, 455],   # 20:00
            ],
            'colorscale': 'Viridis'
        }
    ))

    # ========================================================================
    # SMALL Panels - 1/3 width - Additional metrics with context
    # ========================================================================

    # Memory usage breakdown - ENHANCED WITH MORE CONTENT
    panels.append(Panel(
        id="metric-memory-breakdown",
        type=PanelType.TEXT,
        title="Memory Breakdown",
        subtitle="Current allocation across HoloLoom subsystems",
        size=PanelSize.SMALL,
        data={
            'content': """
                <div style="font-family: monospace;">
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem; background: var(--color-blue-50); border-radius: 0.5rem;">
                            <span style="font-weight: 600;">🧵 Yarn Graph</span>
                            <span style="color: var(--color-blue-600); font-weight: bold;">2.1 GB</span>
                        </div>
                        <div style="padding-left: 1rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Nodes:</span>
                                <span>142,847</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Edges:</span>
                                <span>1,024,331</span>
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem; background: var(--color-purple-50); border-radius: 0.5rem;">
                            <span style="font-weight: 600;">🌀 Warp Space</span>
                            <span style="color: var(--color-purple-600); font-weight: bold;">1.8 GB</span>
                        </div>
                        <div style="padding-left: 1rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Active tensors:</span>
                                <span>24</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Cached manifolds:</span>
                                <span>8,421</span>
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem; background: var(--color-green-50); border-radius: 0.5rem;">
                            <span style="font-weight: 600;">💾 Embeddings Cache</span>
                            <span style="color: var(--color-green-600); font-weight: bold;">1.4 GB</span>
                        </div>
                        <div style="padding-left: 1rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Vectors (96D):</span>
                                <span>1.2M</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Hit rate:</span>
                                <span style="color: var(--color-green-600);">94.2%</span>
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding: 0.5rem; background: var(--color-orange-50); border-radius: 0.5rem;">
                            <span style="font-weight: 600;">📊 Feature Buffers</span>
                            <span style="color: var(--color-orange-600); font-weight: bold;">0.9 GB</span>
                        </div>
                        <div style="padding-left: 1rem; margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Motifs:</span>
                                <span>47,231</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Spectral features:</span>
                                <span>12,847</span>
                            </div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-between; padding: 1rem; border-top: 2px solid var(--color-border); font-weight: bold; background: var(--color-bg-elevated); border-radius: 0.5rem;">
                        <span>TOTAL ALLOCATED</span>
                        <span style="color: var(--color-accent-primary);">6.2 GB / 16 GB</span>
                    </div>
                </div>
            """
        }
    ))

    # Tool usage stats
    panels.append(Panel(
        id="metric-tool-usage",
        type=PanelType.TEXT,
        title="Tool Usage Stats",
        subtitle="Last 1000 queries",
        size=PanelSize.SMALL,
        data={
            'content': """
                <div style="font-family: monospace; font-size: 0.875rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>🔍 Search:</span>
                        <span style="color: var(--color-blue-600);">487 (48.7%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>💬 Answer:</span>
                        <span style="color: var(--color-green-600);">312 (31.2%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>📊 Analyze:</span>
                        <span style="color: var(--color-purple-600);">142 (14.2%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>🔧 Execute:</span>
                        <span style="color: var(--color-orange-600);">59 (5.9%)</span>
                    </div>
                </div>
            """
        }
    ))

    # System health
    panels.append(Panel(
        id="metric-system-health",
        type=PanelType.TEXT,
        title="System Health",
        subtitle="Real-time status",
        size=PanelSize.SMALL,
        data={
            'content': """
                <div style="font-family: monospace; font-size: 0.875rem;">
                    <div style="display: flex; align-items: center; margin-bottom: 0.75rem;">
                        <span style="color: var(--color-green-600); font-size: 1.5rem; margin-right: 0.5rem;">●</span>
                        <div>
                            <div style="font-weight: bold;">All Systems Operational</div>
                            <div style="font-size: 0.75rem; color: var(--color-text-secondary);">Last check: 2 seconds ago</div>
                        </div>
                    </div>
                    <div style="padding-top: 0.5rem; border-top: 1px solid var(--color-border); font-size: 0.75rem; color: var(--color-text-secondary);">
                        <div>✅ Memory backends: Healthy</div>
                        <div>✅ Embedding service: Healthy</div>
                        <div>✅ Policy engine: Healthy</div>
                    </div>
                </div>
            """
        }
    ))

    # ========================================================================
    # COMPACT Panels - 1/4 width - Recent activity
    # ========================================================================

    recent_queries = [
        {'query': 'What is Thompson Sampling?', 'time': '2s ago', 'latency': '98ms', 'tool': '💬'},
        {'query': 'Explain matryoshka embeddings', 'time': '15s ago', 'latency': '145ms', 'tool': '🔍'},
        {'query': 'Show memory usage trends', 'time': '1m ago', 'latency': '210ms', 'tool': '📊'},
        {'query': 'Compare LITE vs FAST modes', 'time': '3m ago', 'latency': '185ms', 'tool': '📊'},
    ]

    for i, query in enumerate(recent_queries):
        panels.append(Panel(
            id=f"recent-query-{i}",
            type=PanelType.TEXT,
            title=f"{query['tool']} Recent Query",
            subtitle=query['time'],
            size=PanelSize.COMPACT,
            data={
                'content': f"""
                    <div style="font-size: 0.875rem;">
                        <div style="margin-bottom: 0.5rem; font-weight: 500;">
                            "{query['query']}"
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--color-text-secondary);">
                            <span>⚡ {query['latency']}</span>
                            <span>{query['time']}</span>
                        </div>
                    </div>
                """
            }
        ))

    # ========================================================================
    # FULL_WIDTH Panel - Network graph
    # ========================================================================
    panels.append(Panel(
        id="network-knowledge-graph",
        type=PanelType.NETWORK,
        title="Knowledge Graph Visualization",
        subtitle="Entity relationships (top 20 nodes by centrality)",
        size=PanelSize.FULL_WIDTH,
        data={
            'nodes': [
                {'id': 'Thompson Sampling', 'size': 20, 'color': 'blue'},
                {'id': 'Exploration', 'size': 15, 'color': 'green'},
                {'id': 'Exploitation', 'size': 15, 'color': 'green'},
                {'id': 'Multi-Armed Bandit', 'size': 18, 'color': 'blue'},
                {'id': 'Bayesian', 'size': 12, 'color': 'purple'},
                {'id': 'Policy Engine', 'size': 16, 'color': 'orange'},
                {'id': 'Neural Network', 'size': 14, 'color': 'orange'},
                {'id': 'Matryoshka', 'size': 13, 'color': 'purple'},
                {'id': 'Embeddings', 'size': 14, 'color': 'purple'},
                {'id': 'WarpSpace', 'size': 12, 'color': 'red'},
            ],
            'edges': [
                {'source': 'Thompson Sampling', 'target': 'Multi-Armed Bandit'},
                {'source': 'Thompson Sampling', 'target': 'Exploration'},
                {'source': 'Thompson Sampling', 'target': 'Exploitation'},
                {'source': 'Thompson Sampling', 'target': 'Bayesian'},
                {'source': 'Policy Engine', 'target': 'Thompson Sampling'},
                {'source': 'Policy Engine', 'target': 'Neural Network'},
                {'source': 'Matryoshka', 'target': 'Embeddings'},
                {'source': 'Policy Engine', 'target': 'Embeddings'},
                {'source': 'WarpSpace', 'target': 'Embeddings'},
            ]
        }
    ))

    # Create mock spacetime
    class MockSpacetime:
        query_text = "Interactive Dashboard Demo"
        response = "Comprehensive showcase of all panel sizes and data types"
        tool_used = "dashboard"
        confidence = 1.0
        trace = {}
        metadata = {}

        def to_dict(self):
            return {
                'query_text': self.query_text,
                'response': self.response,
                'tool_used': self.tool_used,
                'confidence': self.confidence
            }

    return Dashboard(
        title="HoloLoom Interactive Dashboard",
        layout=LayoutType.ADAPTIVE,
        panels=panels,
        spacetime=MockSpacetime(),
        metadata={
            'total_panels': len(panels),
            'panel_sizes_used': list(set(p.size for p in panels)),
            'demo_version': '2.0',
            'features': ['All panel sizes', 'Interactive charts', 'Modern CSS', 'Dark mode']
        }
    )


def main():
    """Generate interactive dashboard demo."""
    print("[Interactive Dashboard] Generating comprehensive demo...")

    dashboard = create_interactive_dashboard()
    renderer = HTMLRenderer()
    html = renderer.render(dashboard)

    output_path = Path(__file__).parent / "output" / "interactive_dashboard.html"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(html, encoding='utf-8')

    print(f"Dashboard saved to: {output_path}")
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         Interactive Dashboard Demo - COMPLETE                    ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║                                                                  ║")
    print(f"║  📊 Total Panels: {len(dashboard.panels):>2}                                            ║")
    print("║                                                                  ║")
    print("║  Panel Sizes Showcased:                                          ║")
    print("║    • TINY (1/6) - 6 KPI metrics                                  ║")
    print("║    • COMPACT (1/4) - 8 performance metrics + recent queries      ║")
    print("║    • SMALL (1/3) - 6 charts and text panels                      ║")
    print("║    • MEDIUM (1/2) - 2 scatter plots                              ║")
    print("║    • LARGE (2/3) - 1 timeline visualization                      ║")
    print("║    • THREE_QUARTERS (3/4) - 1 heatmap                            ║")
    print("║    • FULL_WIDTH - 1 network graph                                ║")
    print("║    • HERO - 1 welcome banner                                     ║")
    print("║                                                                  ║")
    print("║  🎯 Interactive Features:                                        ║")
    print("║     • Press 'T' to toggle dark mode                              ║")
    print("║     • Press '?' for keyboard shortcuts                           ║")
    print("║     • Hover over charts for tooltips                             ║")
    print("║     • Resize window for responsive layouts                       ║")
    print("║     • All charts adapt to theme changes                          ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📄 Output: {output_path}")
    print()
    print("Open in your browser to explore!")

    return output_path


if __name__ == "__main__":
    output_path = main()

    # Open in browser (Windows)
    import subprocess
    try:
        subprocess.run(['start', str(output_path)], shell=True, check=False)
    except Exception:
        pass
