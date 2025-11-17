#!/usr/bin/env python3
"""
Advanced Chain Visualization with Interactive HTML/D3.js

Provides enhanced visualization capabilities including:
- Interactive HTML dashboards with D3.js
- Real-time execution tracing
- Bottleneck highlighting
- Cost visualization
- Timeline/Gantt charts
- Error flow diagrams
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .chain_visualization import ChainVisualizer, GraphNode, GraphEdge, NodeType
from .chain_tracing import ExecutionTracer, StepTrace


@dataclass
class VisualizationMetrics:
    """Metrics for visualization"""
    total_duration: float = 0.0
    total_cost: float = 0.0
    step_durations: Dict[str, float] = field(default_factory=dict)
    step_costs: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    error_paths: List[List[str]] = field(default_factory=list)


class AdvancedChainVisualizer(ChainVisualizer):
    """Enhanced chain visualizer with interactive capabilities"""

    def __init__(self):
        super().__init__()
        self.metrics = VisualizationMetrics()
        self.execution_trace: Optional[List[StepTrace]] = None
        self.cost_model: Dict[str, float] = {
            "execute": 0.01,
            "parallel": 0.05,
            "retry": 0.02,
            "loop": 0.03,
            "transform": 0.001,
            "conditional": 0.002
        }

    def set_execution_trace(self, trace: List[StepTrace]) -> None:
        """Set execution trace for visualization"""
        self.execution_trace = trace
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate visualization metrics from trace"""
        if not self.execution_trace:
            return

        total_duration = 0.0
        total_cost = 0.0
        step_durations = {}
        step_costs = {}
        bottlenecks = []

        for trace in self.execution_trace:
            if trace.duration:
                step_durations[trace.step_name] = trace.duration
                total_duration += trace.duration

                # Calculate cost
                processor_type = trace.processor_type
                base_cost = self.cost_model.get(processor_type, 0.01)
                step_cost = base_cost * (trace.duration / 10.0)  # Scale by duration
                step_costs[trace.step_name] = step_cost
                total_cost += step_cost

                # Identify bottlenecks (steps taking > 20% of total time)
                if trace.duration > total_duration * 0.2:
                    bottlenecks.append({
                        "step": trace.step_name,
                        "duration": trace.duration,
                        "percentage": (trace.duration / total_duration) * 100
                    })

        self.metrics = VisualizationMetrics(
            total_duration=total_duration,
            total_cost=total_cost,
            step_durations=step_durations,
            step_costs=step_costs,
            bottlenecks=sorted(bottlenecks, key=lambda x: x["duration"], reverse=True)
        )

    def to_interactive_html(self, title: str = "Chain Visualization") -> str:
        """Generate interactive HTML with D3.js visualization"""
        graph_data = self.to_json()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px 8px 0 0;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #f9f9f9;
            border-bottom: 1px solid #e0e0e0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 14px;
            color: #999;
        }}
        .tabs {{
            display: flex;
            background: #f0f0f0;
            border-bottom: 2px solid #667eea;
        }}
        .tab {{
            padding: 15px 30px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
        }}
        .tab:hover {{
            background: rgba(102, 126, 234, 0.1);
        }}
        .tab.active {{
            background: #667eea;
            color: white;
        }}
        .tab-content {{
            padding: 30px;
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        #graph-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background: white;
        }}
        #timeline-container {{
            width: 100%;
            height: 400px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background: white;
        }}
        .node {{
            cursor: pointer;
            transition: all 0.3s;
        }}
        .node:hover {{
            filter: brightness(1.2);
        }}
        .node-input {{ fill: #e1f5e1; stroke: #4caf50; }}
        .node-output {{ fill: #fff3e0; stroke: #ff9800; }}
        .node-processor {{ fill: #e3f2fd; stroke: #2196f3; }}
        .node-decision {{ fill: #f3e5f5; stroke: #9c27b0; }}
        .node-parallel {{ fill: #fce4ec; stroke: #e91e63; }}
        .node-loop {{ fill: #e0f2f1; stroke: #009688; }}
        .node-bottleneck {{ fill: #ffebee; stroke: #f44336; stroke-width: 3px; }}
        .link {{
            fill: none;
            stroke: #999;
            stroke-width: 2px;
        }}
        .link-error {{
            stroke: #f44336;
            stroke-width: 3px;
            stroke-dasharray: 5, 5;
        }}
        .node-label {{
            font-size: 12px;
            font-weight: 500;
            text-anchor: middle;
            pointer-events: none;
        }}
        .bottleneck-list {{
            list-style: none;
            padding: 0;
        }}
        .bottleneck-item {{
            padding: 15px;
            margin: 10px 0;
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            border-radius: 4px;
        }}
        .bottleneck-name {{
            font-weight: bold;
            color: #333;
        }}
        .bottleneck-stats {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .cost-bar {{
            height: 30px;
            background: linear-gradient(90deg, #4caf50 0%, #ff9800 100%);
            border-radius: 4px;
            margin: 5px 0;
            position: relative;
        }}
        .cost-label {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-weight: bold;
            font-size: 12px;
        }}
        .legend {{
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            margin-top: 20px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            vertical-align: middle;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>Interactive chain execution visualization and performance analysis</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Total Duration</div>
                <div class="metric-value">{self.metrics.total_duration:.2f}<span class="metric-unit">s</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Cost</div>
                <div class="metric-value">${self.metrics.total_cost:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Steps</div>
                <div class="metric-value">{len(self.nodes)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Bottlenecks</div>
                <div class="metric-value">{len(self.metrics.bottlenecks)}</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('graph')">Dependency Graph</button>
            <button class="tab" onclick="showTab('timeline')">Timeline</button>
            <button class="tab" onclick="showTab('bottlenecks')">Bottlenecks</button>
            <button class="tab" onclick="showTab('costs')">Cost Analysis</button>
        </div>

        <div id="graph-tab" class="tab-content active">
            <div id="graph-container"></div>
            <div class="legend">
                <div class="legend-item">
                    <span class="legend-color" style="background: #e1f5e1; border: 2px solid #4caf50;"></span>
                    Input
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #e3f2fd; border: 2px solid #2196f3;"></span>
                    Processor
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #f3e5f5; border: 2px solid #9c27b0;"></span>
                    Decision
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #fce4ec; border: 2px solid #e91e63;"></span>
                    Parallel
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #e0f2f1; border: 2px solid #009688;"></span>
                    Loop
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background: #ffebee; border: 2px solid #f44336;"></span>
                    Bottleneck
                </div>
            </div>
        </div>

        <div id="timeline-tab" class="tab-content">
            <div id="timeline-container"></div>
        </div>

        <div id="bottlenecks-tab" class="tab-content">
            <h2>Performance Bottlenecks</h2>
            <ul class="bottleneck-list">
                {self._generate_bottleneck_html()}
            </ul>
        </div>

        <div id="costs-tab" class="tab-content">
            <h2>Cost Breakdown by Step</h2>
            {self._generate_cost_html()}
        </div>
    </div>

    <script>
        const graphData = {json.dumps(graph_data)};
        const metrics = {json.dumps({
            'step_durations': self.metrics.step_durations,
            'step_costs': self.metrics.step_costs,
            'bottlenecks': self.metrics.bottlenecks
        })};

        function showTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');

            // Render visualization if needed
            if (tabName === 'graph' && !window.graphRendered) {{
                renderGraph();
                window.graphRendered = true;
            }} else if (tabName === 'timeline' && !window.timelineRendered) {{
                renderTimeline();
                window.timelineRendered = true;
            }}
        }}

        function renderGraph() {{
            const container = d3.select('#graph-container');
            const width = container.node().clientWidth;
            const height = container.node().clientHeight;

            const svg = container.append('svg')
                .attr('width', width)
                .attr('height', height);

            // Create force simulation
            const simulation = d3.forceSimulation(graphData.nodes)
                .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(100))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(50));

            // Draw edges
            const link = svg.append('g')
                .selectAll('line')
                .data(graphData.edges)
                .join('line')
                .attr('class', 'link');

            // Draw nodes
            const node = svg.append('g')
                .selectAll('circle')
                .data(graphData.nodes)
                .join('circle')
                .attr('class', d => {{
                    let classes = 'node node-' + d.type;
                    if (metrics.bottlenecks.some(b => b.step === d.id)) {{
                        classes += ' node-bottleneck';
                    }}
                    return classes;
                }})
                .attr('r', 30)
                .call(drag(simulation));

            // Node labels
            const label = svg.append('g')
                .selectAll('text')
                .data(graphData.nodes)
                .join('text')
                .attr('class', 'node-label')
                .text(d => d.name.length > 15 ? d.name.substring(0, 12) + '...' : d.name);

            // Tooltips
            node.append('title')
                .text(d => {{
                    const duration = metrics.step_durations[d.id] || 0;
                    const cost = metrics.step_costs[d.id] || 0;
                    return `${{d.name}}\\nType: ${{d.type}}\\nDuration: ${{duration.toFixed(2)}}s\\nCost: $${{cost.toFixed(4)}}`;
                }});

            // Update positions
            simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y + 4);
            }});

            function drag(simulation) {{
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}

                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}

                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}

                return d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended);
            }}
        }}

        function renderTimeline() {{
            const container = d3.select('#timeline-container');
            const width = container.node().clientWidth;
            const height = container.node().clientHeight;
            const margin = {{top: 40, right: 30, bottom: 60, left: 150}};

            const svg = container.append('svg')
                .attr('width', width)
                .attr('height', height);

            const chartWidth = width - margin.left - margin.right;
            const chartHeight = height - margin.top - margin.bottom;

            const g = svg.append('g')
                .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

            // Prepare timeline data
            const steps = Object.keys(metrics.step_durations);
            let cumulative = 0;
            const timelineData = steps.map(step => {{
                const duration = metrics.step_durations[step];
                const start = cumulative;
                cumulative += duration;
                return {{ step, start, duration, end: cumulative }};
            }});

            // Scales
            const xScale = d3.scaleLinear()
                .domain([0, cumulative])
                .range([0, chartWidth]);

            const yScale = d3.scaleBand()
                .domain(steps)
                .range([0, chartHeight])
                .padding(0.2);

            // Draw bars
            g.selectAll('.bar')
                .data(timelineData)
                .join('rect')
                .attr('class', 'bar')
                .attr('x', d => xScale(d.start))
                .attr('y', d => yScale(d.step))
                .attr('width', d => xScale(d.duration))
                .attr('height', yScale.bandwidth())
                .attr('fill', '#667eea')
                .attr('opacity', 0.8);

            // Add axes
            g.append('g')
                .attr('transform', `translate(0,${{chartHeight}})`)
                .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => d.toFixed(1) + 's'));

            g.append('g')
                .call(d3.axisLeft(yScale));

            // Add title
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .text('Execution Timeline (Gantt Chart)');
        }}

        // Initial render
        window.graphRendered = false;
        window.timelineRendered = false;
        renderGraph();
        window.graphRendered = true;
    </script>
</body>
</html>"""
        return html

    def _generate_bottleneck_html(self) -> str:
        """Generate HTML for bottleneck list"""
        if not self.metrics.bottlenecks:
            return '<li style="padding: 20px; text-align: center; color: #666;">No bottlenecks detected</li>'

        html_parts = []
        for bottleneck in self.metrics.bottlenecks:
            html_parts.append(f"""
                <li class="bottleneck-item">
                    <div class="bottleneck-name">{bottleneck['step']}</div>
                    <div class="bottleneck-stats">
                        Duration: {bottleneck['duration']:.2f}s ({bottleneck['percentage']:.1f}% of total)
                    </div>
                </li>
            """)
        return ''.join(html_parts)

    def _generate_cost_html(self) -> str:
        """Generate HTML for cost breakdown"""
        if not self.metrics.step_costs:
            return '<p>No cost data available</p>'

        html_parts = []
        max_cost = max(self.metrics.step_costs.values()) if self.metrics.step_costs else 1.0

        for step, cost in sorted(self.metrics.step_costs.items(), key=lambda x: x[1], reverse=True):
            width_pct = (cost / max_cost) * 100
            html_parts.append(f"""
                <div style="margin: 15px 0;">
                    <div style="font-weight: bold; margin-bottom: 5px;">{step}</div>
                    <div class="cost-bar" style="width: {width_pct}%;">
                        <span class="cost-label">${cost:.4f}</span>
                    </div>
                </div>
            """)

        return ''.join(html_parts)

    def save_html(self, filepath: str, title: str = "Chain Visualization") -> None:
        """Save interactive HTML visualization to file"""
        html_content = self.to_interactive_html(title)
        Path(filepath).write_text(html_content, encoding='utf-8')


def visualize_chain_advanced(
    chain_def: Dict[str, Any],
    execution_trace: Optional[List[StepTrace]] = None,
    output_file: Optional[str] = None,
    title: str = "Chain Visualization"
) -> str:
    """
    Create advanced interactive visualization of a chain.

    Args:
        chain_def: Chain definition dictionary
        execution_trace: Optional execution trace for metrics
        output_file: Optional file path to save HTML
        title: Visualization title

    Returns:
        HTML string if output_file is None, else file path
    """
    visualizer = AdvancedChainVisualizer()
    visualizer.build_graph_from_chain(chain_def)

    if execution_trace:
        visualizer.set_execution_trace(execution_trace)

    html_content = visualizer.to_interactive_html(title)

    if output_file:
        visualizer.save_html(output_file, title)
        return output_file
    else:
        return html_content


__all__ = [
    'AdvancedChainVisualizer',
    'VisualizationMetrics',
    'visualize_chain_advanced'
]
