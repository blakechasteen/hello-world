"""
HoloLoom Telemetry - Observability for distributed systems.

Provides comprehensive telemetry capabilities:
- OpenTelemetry-compatible distributed tracing
- Prometheus-compatible metrics
- W3C Trace Context propagation
- Pre-defined collectors for HoloLoom systems

Quick Start:
    # Tracing
    from hololoom.telemetry import get_tracer, span

    tracer = get_tracer()
    with tracer.trace("my_operation") as s:
        # Your code here
        s.set_attribute("result", "success")

    # Or use decorator
    @span("process_query")
    async def process_query(query: str):
        return await do_work(query)

    # Metrics
    from hololoom.telemetry import get_registry, counter, histogram

    queries = counter("queries_total", "Total queries", ["mode"])
    queries.inc(labels={"mode": "fast"})

    latency = histogram("latency_seconds", "Query latency", ["mode"])
    with latency.time(labels={"mode": "fast"}):
        process_query()

    # Pre-defined collectors
    from hololoom.telemetry import get_collector

    collector = get_collector()
    collector.record_query(mode="fast", latency_seconds=0.15, confidence=0.92)

    # Start metrics server
    from hololoom.telemetry import start_metrics_server

    server = start_metrics_server(port=9090)
    # Metrics available at http://localhost:9090/metrics
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  PROTOCOL TYPES
# ═══════════════════════════════════════════════════════════════════════════════

from hololoom.telemetry.protocol import (
    # Protocols
    TracingProtocol,
    MetricsProtocol,
    SpanProcessorProtocol,
    # Span types
    SpanContext,
    SpanData,
    SpanKind,
    SpanStatus,
    # Metric types
    MetricPoint,
    MetricType,
    HistogramBuckets,
    # Factory functions
    create_span_context,
    generate_trace_id,
    generate_span_id,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

from hololoom.telemetry.config import (
    TelemetryConfig,
    TracingConfig,
    MetricsConfig,
    SamplingStrategy,
    ExporterType,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  TRACING
# ═══════════════════════════════════════════════════════════════════════════════

from hololoom.telemetry.tracing import (
    # Tracer
    OTelTracer,
    create_tracer,
    get_tracer,
    # Decorator
    span,
    # Processors
    ConsoleSpanProcessor,
    BatchSpanProcessor,
    # Context propagation
    TraceContextManager,
    W3CTraceContext,
    get_current_context,
    set_current_context,
    clear_context,
    run_with_context,
    run_with_context_async,
    # Baggage
    get_baggage,
    set_baggage,
    get_all_baggage,
    clear_baggage,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

from hololoom.telemetry.metrics import (
    # Registry
    PrometheusRegistry,
    create_registry,
    get_registry,
    # Metric types
    Counter,
    Gauge,
    Histogram,
    Summary,
    # Convenience functions
    counter,
    gauge,
    histogram,
    summary,
    # Collectors
    HoloLoomCollector,
    create_default_collector,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTERS
# ═══════════════════════════════════════════════════════════════════════════════

from hololoom.telemetry.exporters import (
    # Jaeger
    JaegerExporter,
    create_jaeger_exporter,
    # Prometheus HTTP
    PrometheusHTTPServer,
    create_prometheus_server,
    start_metrics_server,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: GLOBAL COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

from hololoom.telemetry.metrics.collectors import get_collector

# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Protocols
    "TracingProtocol",
    "MetricsProtocol",
    "SpanProcessorProtocol",
    # Span types
    "SpanContext",
    "SpanData",
    "SpanKind",
    "SpanStatus",
    # Metric types
    "MetricPoint",
    "MetricType",
    "HistogramBuckets",
    # Factory functions
    "create_span_context",
    "generate_trace_id",
    "generate_span_id",
    # Configuration
    "TelemetryConfig",
    "TracingConfig",
    "MetricsConfig",
    "SamplingStrategy",
    "ExporterType",
    # Tracer
    "OTelTracer",
    "create_tracer",
    "get_tracer",
    # Decorator
    "span",
    # Processors
    "ConsoleSpanProcessor",
    "BatchSpanProcessor",
    # Context propagation
    "TraceContextManager",
    "W3CTraceContext",
    "get_current_context",
    "set_current_context",
    "clear_context",
    "run_with_context",
    "run_with_context_async",
    # Baggage
    "get_baggage",
    "set_baggage",
    "get_all_baggage",
    "clear_baggage",
    # Registry
    "PrometheusRegistry",
    "create_registry",
    "get_registry",
    # Metric types
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    # Convenience functions
    "counter",
    "gauge",
    "histogram",
    "summary",
    # Collectors
    "HoloLoomCollector",
    "create_default_collector",
    "get_collector",
    # Jaeger
    "JaegerExporter",
    "create_jaeger_exporter",
    # Prometheus HTTP
    "PrometheusHTTPServer",
    "create_prometheus_server",
    "start_metrics_server",
]
