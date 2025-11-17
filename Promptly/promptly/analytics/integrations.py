"""
Integrations - Connect analytics to external monitoring systems

Provides:
- OpenTelemetry support
- Prometheus metrics export
- Grafana dashboards
- Logging integration (structured logs)
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

# Optional dependencies
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry, write_to_textfile
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Prometheus integration unavailable.")

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("Warning: opentelemetry not installed. OpenTelemetry integration unavailable.")


class PrometheusExporter:
    """Export Promptly metrics to Prometheus"""

    def __init__(self, port: int = 9090, registry=None):
        """
        Initialize Prometheus exporter

        Args:
            port: HTTP port for metrics endpoint
            registry: Custom Prometheus registry (optional)
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not installed")

        self.port = port
        self.registry = registry or CollectorRegistry()
        self.server_started = False

        # Define metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        # Operation metrics
        self.operation_counter = Counter(
            'promptly_operations_total',
            'Total number of operations',
            ['operation', 'status'],
            registry=self.registry
        )

        self.operation_duration = Histogram(
            'promptly_operation_duration_seconds',
            'Operation duration in seconds',
            ['operation'],
            registry=self.registry
        )

        # Quality metrics
        self.quality_score = Histogram(
            'promptly_quality_score',
            'Prompt quality scores',
            ['prompt', 'evaluator'],
            registry=self.registry
        )

        # Usage metrics
        self.prompt_accesses = Counter(
            'promptly_prompt_accesses_total',
            'Total prompt accesses',
            ['prompt', 'branch'],
            registry=self.registry
        )

        self.evaluations = Counter(
            'promptly_evaluations_total',
            'Total evaluations run',
            ['prompt', 'branch'],
            registry=self.registry
        )

        self.chain_executions = Counter(
            'promptly_chain_executions_total',
            'Total chain executions',
            ['chain', 'status'],
            registry=self.registry
        )

        # Resource metrics
        self.cpu_usage = Gauge(
            'promptly_cpu_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'promptly_memory_mb',
            'Memory usage in MB',
            registry=self.registry
        )

        # System info
        self.system_info = Info(
            'promptly_system',
            'Promptly system information',
            registry=self.registry
        )

    def start_http_server(self):
        """Start HTTP server for metrics endpoint"""
        if not self.server_started:
            start_http_server(self.port, registry=self.registry)
            self.server_started = True

    def record_operation(self, operation: str, duration_ms: float, status: str = 'success'):
        """Record an operation"""
        self.operation_counter.labels(operation=operation, status=status).inc()
        self.operation_duration.labels(operation=operation).observe(duration_ms / 1000.0)

    def record_quality_score(self, prompt: str, evaluator: str, score: float):
        """Record a quality score"""
        self.quality_score.labels(prompt=prompt, evaluator=evaluator).observe(score)

    def record_prompt_access(self, prompt: str, branch: str):
        """Record a prompt access"""
        self.prompt_accesses.labels(prompt=prompt, branch=branch).inc()

    def record_evaluation(self, prompt: str, branch: str):
        """Record an evaluation"""
        self.evaluations.labels(prompt=prompt, branch=branch).inc()

    def record_chain_execution(self, chain: str, status: str):
        """Record a chain execution"""
        self.chain_executions.labels(chain=chain, status=status).inc()

    def update_resource_usage(self, cpu_percent: float, memory_mb: float):
        """Update resource usage metrics"""
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_mb)

    def set_system_info(self, info: Dict):
        """Set system information"""
        self.system_info.info(info)

    def export_to_file(self, output_path: str):
        """Export metrics to a file (for node_exporter textfile collector)"""
        write_to_textfile(output_path, self.registry)

    def sync_from_analytics(self, performance_monitor=None, usage_analytics=None,
                           quality_metrics=None):
        """Sync metrics from analytics modules"""
        if performance_monitor:
            stats = performance_monitor.get_operation_stats()
            for op, data in stats.items():
                # Note: Counters can't be set directly, only incremented
                # This is a limitation of Prometheus
                pass

            resource_stats = performance_monitor.get_resource_stats(hours=1)
            if resource_stats:
                self.update_resource_usage(
                    resource_stats.get('avg_cpu_percent', 0),
                    resource_stats.get('avg_memory_mb', 0)
                )

        if usage_analytics:
            most_used = usage_analytics.get_most_used_prompts(days=1)
            # Record as info - actual metrics would be collected in real-time

        if quality_metrics:
            top = quality_metrics.get_top_performing_prompts(days=1)
            # Record as info


class OpenTelemetryIntegration:
    """Integration with OpenTelemetry for distributed tracing and metrics"""

    def __init__(self, service_name: str = "promptly",
                 enable_tracing: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize OpenTelemetry integration

        Args:
            service_name: Name of the service
            enable_tracing: Enable distributed tracing
            enable_metrics: Enable metrics collection
        """
        if not OTEL_AVAILABLE:
            raise ImportError("opentelemetry not installed")

        self.service_name = service_name
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics

        # Initialize providers
        if enable_tracing:
            resource = Resource.create({"service.name": service_name})
            self.tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.tracer_provider)
            self.tracer = trace.get_tracer(__name__)

        if enable_metrics:
            self.meter_provider = MeterProvider()
            metrics.set_meter_provider(self.meter_provider)
            self.meter = metrics.get_meter(__name__)

            # Initialize metrics
            self._init_metrics()

    def _init_metrics(self):
        """Initialize OpenTelemetry metrics"""
        # Operation counters
        self.operation_counter = self.meter.create_counter(
            "promptly.operations",
            description="Number of operations"
        )

        self.operation_duration = self.meter.create_histogram(
            "promptly.operation.duration",
            description="Operation duration in milliseconds"
        )

        # Quality metrics
        self.quality_score_histogram = self.meter.create_histogram(
            "promptly.quality.score",
            description="Quality scores"
        )

    def trace_operation(self, operation_name: str):
        """
        Context manager for tracing an operation

        Usage:
            with otel.trace_operation('add_prompt'):
                # do work
                pass
        """
        if not self.enable_tracing:
            return DummySpan()

        return self.tracer.start_as_current_span(operation_name)

    def record_operation(self, operation: str, duration_ms: float, attributes: Dict = None):
        """Record an operation with metrics"""
        if not self.enable_metrics:
            return

        attrs = attributes or {}
        attrs['operation'] = operation

        self.operation_counter.add(1, attrs)
        self.operation_duration.record(duration_ms, attrs)

    def record_quality_score(self, prompt: str, score: float, attributes: Dict = None):
        """Record a quality score"""
        if not self.enable_metrics:
            return

        attrs = attributes or {}
        attrs['prompt'] = prompt

        self.quality_score_histogram.record(score, attrs)


class DummySpan:
    """Dummy span for when tracing is disabled"""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class GrafanaExporter:
    """Export dashboards and data for Grafana"""

    def __init__(self, output_dir: str = "./grafana"):
        """
        Initialize Grafana exporter

        Args:
            output_dir: Directory to export dashboards to
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_dashboard(self, dashboard_name: str = "promptly-overview") -> str:
        """Export a Grafana dashboard JSON"""
        dashboard = self._create_overview_dashboard()

        output_path = self.output_dir / f"{dashboard_name}.json"
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)

        return str(output_path)

    def _create_overview_dashboard(self) -> Dict:
        """Create overview dashboard configuration"""
        return {
            "dashboard": {
                "title": "Promptly Analytics Overview",
                "tags": ["promptly", "analytics"],
                "timezone": "browser",
                "panels": [
                    self._create_panel(
                        title="Operations Per Second",
                        target="rate(promptly_operations_total[5m])",
                        panel_id=1,
                        x=0, y=0, w=12, h=8
                    ),
                    self._create_panel(
                        title="Operation Duration (p95)",
                        target='histogram_quantile(0.95, rate(promptly_operation_duration_seconds_bucket[5m]))',
                        panel_id=2,
                        x=12, y=0, w=12, h=8
                    ),
                    self._create_panel(
                        title="CPU Usage",
                        target="promptly_cpu_percent",
                        panel_id=3,
                        x=0, y=8, w=12, h=8
                    ),
                    self._create_panel(
                        title="Memory Usage",
                        target="promptly_memory_mb",
                        panel_id=4,
                        x=12, y=8, w=12, h=8
                    ),
                    self._create_panel(
                        title="Quality Scores",
                        target="promptly_quality_score",
                        panel_id=5,
                        x=0, y=16, w=24, h=8
                    ),
                ],
                "schemaVersion": 16,
                "version": 0
            }
        }

    def _create_panel(self, title: str, target: str, panel_id: int,
                     x: int, y: int, w: int, h: int) -> Dict:
        """Create a dashboard panel"""
        return {
            "id": panel_id,
            "title": title,
            "type": "graph",
            "gridPos": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            },
            "targets": [
                {
                    "expr": target,
                    "refId": "A"
                }
            ],
            "xaxis": {
                "mode": "time",
                "show": True
            },
            "yaxis": {
                "format": "short",
                "show": True
            }
        }

    def export_prometheus_config(self) -> str:
        """Export Prometheus scrape configuration"""
        config = {
            "scrape_configs": [
                {
                    "job_name": "promptly",
                    "static_configs": [
                        {
                            "targets": ["localhost:9090"]
                        }
                    ],
                    "scrape_interval": "15s"
                }
            ]
        }

        output_path = self.output_dir / "prometheus.yml"
        with open(output_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)

        return str(output_path)


class StructuredLogger:
    """Structured logging for analytics events"""

    def __init__(self, log_path: str = "./logs/analytics.jsonl"):
        """
        Initialize structured logger

        Args:
            log_path: Path to log file (JSONL format)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_event(self, event_type: str, data: Dict, level: str = "info"):
        """Log an analytics event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event_type": event_type,
            "data": data
        }

        with self._lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    def log_operation(self, operation: str, duration_ms: float, status: str = "success",
                     metadata: Dict = None):
        """Log an operation"""
        self.log_event("operation", {
            "operation": operation,
            "duration_ms": duration_ms,
            "status": status,
            "metadata": metadata or {}
        })

    def log_quality_score(self, prompt: str, version: int, score: float,
                         evaluator: str, metadata: Dict = None):
        """Log a quality score"""
        self.log_event("quality_score", {
            "prompt": prompt,
            "version": version,
            "score": score,
            "evaluator": evaluator,
            "metadata": metadata or {}
        })

    def log_usage(self, resource: str, action: str, user: str = None,
                 metadata: Dict = None):
        """Log a usage event"""
        self.log_event("usage", {
            "resource": resource,
            "action": action,
            "user": user,
            "metadata": metadata or {}
        })

    def log_error(self, error_type: str, message: str, metadata: Dict = None):
        """Log an error"""
        self.log_event("error", {
            "error_type": error_type,
            "message": message,
            "metadata": metadata or {}
        }, level="error")


class AnalyticsHub:
    """Central hub for all analytics integrations"""

    def __init__(self, config: Dict = None):
        """
        Initialize analytics hub

        Args:
            config: Configuration dictionary with integration settings
        """
        self.config = config or {}

        # Initialize integrations based on config
        self.prometheus = None
        self.otel = None
        self.grafana = None
        self.logger = None

        if self.config.get('prometheus', {}).get('enabled', False):
            try:
                self.prometheus = PrometheusExporter(
                    port=self.config['prometheus'].get('port', 9090)
                )
                if self.config['prometheus'].get('start_server', True):
                    self.prometheus.start_http_server()
            except ImportError:
                pass

        if self.config.get('opentelemetry', {}).get('enabled', False):
            try:
                self.otel = OpenTelemetryIntegration(
                    service_name=self.config['opentelemetry'].get('service_name', 'promptly')
                )
            except ImportError:
                pass

        if self.config.get('grafana', {}).get('enabled', False):
            self.grafana = GrafanaExporter(
                output_dir=self.config['grafana'].get('output_dir', './grafana')
            )

        if self.config.get('logging', {}).get('enabled', True):
            self.logger = StructuredLogger(
                log_path=self.config['logging'].get('path', './logs/analytics.jsonl')
            )

    def record_operation(self, operation: str, duration_ms: float,
                        status: str = 'success', metadata: Dict = None):
        """Record an operation across all integrations"""
        if self.prometheus:
            self.prometheus.record_operation(operation, duration_ms, status)

        if self.otel:
            self.otel.record_operation(operation, duration_ms, metadata)

        if self.logger:
            self.logger.log_operation(operation, duration_ms, status, metadata)

    def record_quality_score(self, prompt: str, version: int, score: float,
                            evaluator: str, metadata: Dict = None):
        """Record a quality score across all integrations"""
        if self.prometheus:
            self.prometheus.record_quality_score(prompt, evaluator, score)

        if self.otel:
            self.otel.record_quality_score(prompt, score, metadata)

        if self.logger:
            self.logger.log_quality_score(prompt, version, score, evaluator, metadata)

    def record_usage(self, resource: str, action: str, user: str = None,
                    metadata: Dict = None):
        """Record a usage event across all integrations"""
        if self.prometheus and action == 'access':
            branch = metadata.get('branch', 'main') if metadata else 'main'
            self.prometheus.record_prompt_access(resource, branch)

        if self.logger:
            self.logger.log_usage(resource, action, user, metadata)

    def export_grafana_dashboard(self) -> Optional[str]:
        """Export Grafana dashboard"""
        if self.grafana:
            return self.grafana.export_dashboard()
        return None

    def export_prometheus_config(self) -> Optional[str]:
        """Export Prometheus configuration"""
        if self.grafana:
            return self.grafana.export_prometheus_config()
        return None
