"""
Tracing module - OpenTelemetry distributed tracing.

Provides trace context propagation and span management.
"""

from hololoom.telemetry.tracing.otel_tracer import (
    OTelTracer,
    create_tracer,
    get_tracer,
    span,
    ConsoleSpanProcessor,
    BatchSpanProcessor,
)
from hololoom.telemetry.tracing.context import (
    TraceContextManager,
    W3CTraceContext,
    get_current_context,
    set_current_context,
    clear_context,
    run_with_context,
    run_with_context_async,
    get_baggage,
    set_baggage,
    get_all_baggage,
    clear_baggage,
)

__all__ = [
    # Tracer
    "OTelTracer",
    "create_tracer",
    "get_tracer",
    "span",
    # Span processors
    "ConsoleSpanProcessor",
    "BatchSpanProcessor",
    # Context
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
]
