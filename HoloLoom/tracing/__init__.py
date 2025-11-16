"""
HoloLoom Distributed Tracing with OpenTelemetry
================================================

Provides comprehensive distributed tracing for HoloLoom's agentic reasoning system.

Features:
- Auto-instrumentation for FastAPI
- Manual instrumentation for HoloLoom components
- Multiple exporters (Jaeger, Zipkin, Console)
- Context propagation (HTTP, WebSocket)
- Performance analysis and bottleneck detection
- Graceful degradation without OpenTelemetry

Created: 2025-11-16
Integration: Phase 5 → Distributed Tracing
"""

from HoloLoom.tracing.opentelemetry_integration import (
    TracingConfig,
    init_tracing,
    get_tracer,
    TRACING_AVAILABLE,
)

from HoloLoom.tracing.instrumentation import (
    instrument_app,
    TracingMiddleware,
    instrument_hololoom_components,
)

from HoloLoom.tracing.trace_context import (
    TraceContext,
    propagate_trace_context,
    extract_trace_context,
    inject_trace_context,
)

from HoloLoom.tracing.performance_analyzer import (
    PerformanceAnalyzer,
    detect_bottlenecks,
    analyze_trace,
)

__all__ = [
    # Core
    "TracingConfig",
    "init_tracing",
    "get_tracer",
    "TRACING_AVAILABLE",
    # Instrumentation
    "instrument_app",
    "TracingMiddleware",
    "instrument_hololoom_components",
    # Context propagation
    "TraceContext",
    "propagate_trace_context",
    "extract_trace_context",
    "inject_trace_context",
    # Performance analysis
    "PerformanceAnalyzer",
    "detect_bottlenecks",
    "analyze_trace",
]
