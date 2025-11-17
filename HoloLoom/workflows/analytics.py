#!/usr/bin/env python3
"""
Workflow Analytics Dashboard
============================

Real-time workflow performance monitoring and analytics.

Features:
- Execution time tracking
- Node performance metrics
- Bottleneck detection
- Success/failure rates
- Trend analysis
- Visual dashboards (HTML export)

Usage:
    from HoloLoom.workflows.analytics import WorkflowAnalytics

    analytics = WorkflowAnalytics()

    # Track execution
    analytics.track_execution(workflow_id, execution_data)

    # Get dashboard
    html = analytics.generate_dashboard(workflow_id)

    # Get metrics
    metrics = analytics.get_metrics(workflow_id)
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Single workflow execution record"""
    execution_id: str
    workflow_id: str
    workflow_name: str
    timestamp: datetime

    # Performance
    total_duration_ms: float
    nodes_executed: int
    nodes_failed: int

    # Node-level metrics
    node_durations: Dict[str, float] = field(default_factory=dict)
    node_statuses: Dict[str, str] = field(default_factory=dict)

    # Results
    status: str = "success"  # success, failure, partial
    error_message: Optional[str] = None

    # Input/output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeMetrics:
    """Aggregated metrics for a specific node"""
    node_id: str
    agent_type: str

    # Execution stats
    total_executions: int = 0
    successes: int = 0
    failures: int = 0

    # Performance
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0

    # History
    duration_history: List[float] = field(default_factory=list)


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for a workflow"""
    workflow_id: str
    workflow_name: str

    # Execution stats
    total_executions: int = 0
    successes: int = 0
    failures: int = 0
    partial_failures: int = 0

    # Performance
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0

    # Trends
    duration_trend: List[float] = field(default_factory=list)
    success_rate_trend: List[float] = field(default_factory=list)

    # Node-level
    node_metrics: Dict[str, NodeMetrics] = field(default_factory=dict)

    # Bottlenecks
    bottleneck_nodes: List[str] = field(default_factory=list)

    # Recent executions
    recent_executions: List[ExecutionRecord] = field(default_factory=list)


class WorkflowAnalytics:
    """
    Analytics engine for workflow performance monitoring.

    Tracks execution metrics, identifies bottlenecks, and generates dashboards.
    """

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self.executions: Dict[str, List[ExecutionRecord]] = defaultdict(list)
        self.metrics_cache: Dict[str, WorkflowMetrics] = {}

    def track_execution(self, execution_data: Dict[str, Any]):
        """
        Track a workflow execution.

        Args:
            execution_data: Execution data from workflow executor
        """
        record = ExecutionRecord(
            execution_id=execution_data.get('execution_id', self._generate_id()),
            workflow_id=execution_data.get('workflow_id', 'unknown'),
            workflow_name=execution_data.get('workflow_name', 'Unknown Workflow'),
            timestamp=datetime.now(),
            total_duration_ms=execution_data.get('total_duration_ms', 0.0),
            nodes_executed=execution_data.get('nodes_executed', 0),
            nodes_failed=execution_data.get('nodes_failed', 0),
            node_durations=execution_data.get('node_durations', {}),
            node_statuses=execution_data.get('node_statuses', {}),
            status=execution_data.get('status', 'success'),
            error_message=execution_data.get('error_message'),
            input_data=execution_data.get('input_data', {}),
            output_data=execution_data.get('output_data', {})
        )

        workflow_id = record.workflow_id
        self.executions[workflow_id].append(record)

        # Invalidate cache
        if workflow_id in self.metrics_cache:
            del self.metrics_cache[workflow_id]

        # Cleanup old records
        self._cleanup_old_records(workflow_id)

        logger.info(f"Tracked execution for {record.workflow_name}: {record.status}")

    def get_metrics(self, workflow_id: str, refresh: bool = False) -> Optional[WorkflowMetrics]:
        """
        Get aggregated metrics for a workflow.

        Args:
            workflow_id: Workflow ID
            refresh: Force refresh of cached metrics

        Returns:
            WorkflowMetrics or None if no data
        """
        if workflow_id not in self.executions:
            return None

        if not refresh and workflow_id in self.metrics_cache:
            return self.metrics_cache[workflow_id]

        # Compute metrics
        metrics = self._compute_metrics(workflow_id)
        self.metrics_cache[workflow_id] = metrics

        return metrics

    def _compute_metrics(self, workflow_id: str) -> WorkflowMetrics:
        """Compute aggregated metrics from execution records"""
        records = self.executions[workflow_id]

        if not records:
            return WorkflowMetrics(workflow_id=workflow_id, workflow_name='Unknown')

        # Workflow-level stats
        total_executions = len(records)
        successes = sum(1 for r in records if r.status == 'success')
        failures = sum(1 for r in records if r.status == 'failure')
        partial = sum(1 for r in records if r.status == 'partial')

        # Duration stats
        durations = [r.total_duration_ms for r in records if r.total_duration_ms > 0]
        avg_duration = statistics.mean(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        # P95 duration
        p95_duration = 0.0
        if durations:
            sorted_durations = sorted(durations)
            p95_idx = int(len(sorted_durations) * 0.95)
            p95_duration = sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else sorted_durations[-1]

        # Trends (last 20 executions)
        recent = records[-20:]
        duration_trend = [r.total_duration_ms for r in recent]
        success_rate_trend = [
            sum(1 for r in recent[max(0, i-5):i+1] if r.status == 'success') / min(i+1, 5)
            for i in range(len(recent))
        ]

        # Node-level metrics
        node_metrics = self._compute_node_metrics(records)

        # Identify bottlenecks (nodes taking >40% of total time)
        bottlenecks = []
        for node_id, node_m in node_metrics.items():
            if node_m.avg_duration_ms > avg_duration * 0.4:
                bottlenecks.append(node_id)

        return WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_name=records[0].workflow_name if records else 'Unknown',
            total_executions=total_executions,
            successes=successes,
            failures=failures,
            partial_failures=partial,
            avg_duration_ms=avg_duration,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            p95_duration_ms=p95_duration,
            duration_trend=duration_trend,
            success_rate_trend=success_rate_trend,
            node_metrics=node_metrics,
            bottleneck_nodes=bottlenecks,
            recent_executions=recent
        )

    def _compute_node_metrics(self, records: List[ExecutionRecord]) -> Dict[str, NodeMetrics]:
        """Compute node-level metrics"""
        node_data = defaultdict(lambda: {
            'durations': [],
            'successes': 0,
            'failures': 0,
            'agent_type': 'unknown'
        })

        # Aggregate data
        for record in records:
            for node_id, duration in record.node_durations.items():
                node_data[node_id]['durations'].append(duration)

                status = record.node_statuses.get(node_id, 'success')
                if status == 'success':
                    node_data[node_id]['successes'] += 1
                else:
                    node_data[node_id]['failures'] += 1

        # Compute metrics
        metrics = {}
        for node_id, data in node_data.items():
            durations = data['durations']

            p95_duration = 0.0
            if durations:
                sorted_durations = sorted(durations)
                p95_idx = int(len(sorted_durations) * 0.95)
                p95_duration = sorted_durations[p95_idx] if p95_idx < len(sorted_durations) else sorted_durations[-1]

            metrics[node_id] = NodeMetrics(
                node_id=node_id,
                agent_type=data['agent_type'],
                total_executions=len(durations),
                successes=data['successes'],
                failures=data['failures'],
                avg_duration_ms=statistics.mean(durations) if durations else 0.0,
                min_duration_ms=min(durations) if durations else 0.0,
                max_duration_ms=max(durations) if durations else 0.0,
                p95_duration_ms=p95_duration,
                duration_history=durations[-20:]  # Keep last 20
            )

        return metrics

    def generate_dashboard(self, workflow_id: str) -> str:
        """
        Generate HTML dashboard for workflow analytics.

        Args:
            workflow_id: Workflow ID

        Returns:
            HTML string
        """
        metrics = self.get_metrics(workflow_id)
        if not metrics:
            return "<html><body><h1>No data available</h1></body></html>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Workflow Analytics: {metrics.workflow_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            padding: 20px;
            border-radius: 6px;
            background: #f8f9fa;
        }}
        .metric-card.success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
        }}
        .metric-card.warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .metric-card.error {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .chart {{
            height: 200px;
            background: #f8f9fa;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }}
        .bottleneck {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            border-left: 4px solid #ffc107;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .success-rate {{
            color: #28a745;
            font-weight: bold;
        }}
        .failure-rate {{
            color: #dc3545;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Workflow Analytics</h1>
        <div class="subtitle">{metrics.workflow_name} (ID: {workflow_id})</div>

        <div class="metrics-grid">
            <div class="metric-card success">
                <div class="metric-label">Total Executions</div>
                <div class="metric-value">{metrics.total_executions}</div>
            </div>
            <div class="metric-card {"success" if metrics.successes / max(metrics.total_executions, 1) > 0.9 else "warning"}">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{metrics.successes / max(metrics.total_executions, 1) * 100:.1f}%</div>
                <div class="metric-label">{metrics.successes} / {metrics.total_executions}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Duration</div>
                <div class="metric-value">{metrics.avg_duration_ms:.0f}ms</div>
                <div class="metric-label">P95: {metrics.p95_duration_ms:.0f}ms</div>
            </div>
            <div class="metric-card {"error" if metrics.failures > 0 else ""}">
                <div class="metric-label">Failures</div>
                <div class="metric-value">{metrics.failures}</div>
            </div>
        </div>

        {self._render_bottlenecks(metrics)}

        <h2>Performance Trend</h2>
        <div class="chart">
            {self._render_sparkline(metrics.duration_trend, "Duration (ms)")}
        </div>

        <h2>Success Rate Trend</h2>
        <div class="chart">
            {self._render_sparkline(metrics.success_rate_trend, "Success Rate", is_percentage=True)}
        </div>

        <h2>Node Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Node ID</th>
                    <th>Agent Type</th>
                    <th>Executions</th>
                    <th>Success Rate</th>
                    <th>Avg Duration</th>
                    <th>P95 Duration</th>
                </tr>
            </thead>
            <tbody>
                {self._render_node_table(metrics.node_metrics)}
            </tbody>
        </table>

        <h2>Recent Executions</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Nodes Executed</th>
                </tr>
            </thead>
            <tbody>
                {self._render_recent_executions(metrics.recent_executions)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        return html

    def _render_bottlenecks(self, metrics: WorkflowMetrics) -> str:
        """Render bottleneck warnings"""
        if not metrics.bottleneck_nodes:
            return ""

        html = "<h2>⚠️ Performance Bottlenecks</h2>"
        for node_id in metrics.bottleneck_nodes:
            node_m = metrics.node_metrics.get(node_id)
            if node_m:
                html += f"""
                <div class="bottleneck">
                    <strong>{node_id}</strong> is taking {node_m.avg_duration_ms:.0f}ms
                    ({node_m.avg_duration_ms / max(metrics.avg_duration_ms, 1) * 100:.0f}% of total workflow time)
                </div>
                """

        return html

    def _render_sparkline(self, data: List[float], label: str, is_percentage: bool = False) -> str:
        """Render simple ASCII sparkline"""
        if not data:
            return "<p>No data</p>"

        if is_percentage:
            data_str = ", ".join(f"{v*100:.1f}%" for v in data[-10:])
        else:
            data_str = ", ".join(f"{v:.0f}" for v in data[-10:])

        return f"<p><strong>{label}:</strong> {data_str}</p>"

    def _render_node_table(self, node_metrics: Dict[str, NodeMetrics]) -> str:
        """Render node performance table rows"""
        html = ""
        for node_id, metrics in sorted(node_metrics.items(), key=lambda x: x[1].avg_duration_ms, reverse=True):
            success_rate = metrics.successes / max(metrics.total_executions, 1) * 100

            html += f"""
                <tr>
                    <td>{node_id}</td>
                    <td>{metrics.agent_type}</td>
                    <td>{metrics.total_executions}</td>
                    <td class="{"success-rate" if success_rate > 90 else "failure-rate"}">{success_rate:.1f}%</td>
                    <td>{metrics.avg_duration_ms:.0f}ms</td>
                    <td>{metrics.p95_duration_ms:.0f}ms</td>
                </tr>
            """

        return html

    def _render_recent_executions(self, records: List[ExecutionRecord]) -> str:
        """Render recent executions table"""
        html = ""
        for record in reversed(records[-10:]):  # Last 10
            status_class = "success-rate" if record.status == "success" else "failure-rate"
            html += f"""
                <tr>
                    <td>{record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    <td class="{status_class}">{record.status}</td>
                    <td>{record.total_duration_ms:.0f}ms</td>
                    <td>{record.nodes_executed}</td>
                </tr>
            """

        return html

    def _cleanup_old_records(self, workflow_id: str):
        """Remove records older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        self.executions[workflow_id] = [
            r for r in self.executions[workflow_id]
            if r.timestamp > cutoff
        ]

    def _generate_id(self) -> str:
        """Generate unique execution ID"""
        return f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def export_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Export metrics as JSON"""
        metrics = self.get_metrics(workflow_id)
        if not metrics:
            return {}

        return {
            'workflow_id': metrics.workflow_id,
            'workflow_name': metrics.workflow_name,
            'total_executions': metrics.total_executions,
            'successes': metrics.successes,
            'failures': metrics.failures,
            'success_rate': metrics.successes / max(metrics.total_executions, 1),
            'avg_duration_ms': metrics.avg_duration_ms,
            'p95_duration_ms': metrics.p95_duration_ms,
            'bottleneck_nodes': metrics.bottleneck_nodes,
            'node_metrics': {
                node_id: {
                    'avg_duration_ms': m.avg_duration_ms,
                    'success_rate': m.successes / max(m.total_executions, 1),
                    'p95_duration_ms': m.p95_duration_ms
                }
                for node_id, m in metrics.node_metrics.items()
            }
        }


if __name__ == "__main__":
    # Demo usage
    analytics = WorkflowAnalytics()

    # Simulate executions
    import random

    for i in range(50):
        analytics.track_execution({
            'execution_id': f'exec_{i}',
            'workflow_id': 'test_workflow_1',
            'workflow_name': 'Test Workflow',
            'total_duration_ms': random.uniform(100, 500),
            'nodes_executed': 5,
            'nodes_failed': random.choice([0, 0, 0, 0, 1]),  # 80% success
            'node_durations': {
                'node_1': random.uniform(20, 50),
                'node_2': random.uniform(30, 100),
                'node_3': random.uniform(200, 300),  # Bottleneck!
                'node_4': random.uniform(10, 30),
                'node_5': random.uniform(5, 20)
            },
            'node_statuses': {
                'node_1': 'success',
                'node_2': 'success',
                'node_3': 'success',
                'node_4': 'success',
                'node_5': random.choice(['success', 'success', 'success', 'failure'])
            },
            'status': random.choice(['success', 'success', 'success', 'success', 'failure'])
        })

    # Generate dashboard
    html = analytics.generate_dashboard('test_workflow_1')

    # Save to file
    with open('/tmp/workflow_analytics_demo.html', 'w') as f:
        f.write(html)

    print("=" * 80)
    print("Workflow Analytics Dashboard Generated")
    print("=" * 80)
    print(f"Saved to: /tmp/workflow_analytics_demo.html")

    # Export metrics
    metrics_json = analytics.export_metrics('test_workflow_1')
    print("\nMetrics Summary:")
    print(json.dumps(metrics_json, indent=2))

    # Get metrics object
    metrics = analytics.get_metrics('test_workflow_1')
    print(f"\nBottleneck nodes: {metrics.bottleneck_nodes}")
    print(f"Success rate: {metrics.successes / metrics.total_executions * 100:.1f}%")
