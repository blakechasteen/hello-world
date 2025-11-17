#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Tracing for Reasoning Engine
==========================================
OpenTelemetry integration for end-to-end observability.

Phase 4: Advanced Observability

Author: Claude Code
Date: 2025-11-17
"""

import logging
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

# Optional OpenTelemetry (graceful degradation)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    logger.info("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
    OTEL_AVAILABLE = False
    trace = None


# ============================================================================
# Tracer Setup
# ============================================================================

_tracer: Optional[Any] = None


def setup_tracing(
    service_name: str = "hololoom-reasoning",
    service_version: str = "1.0.0",
    exporter: Optional[Any] = None
) -> Optional[Any]:
    """
    Setup OpenTelemetry tracing.

    Args:
        service_name: Service name for traces
        service_version: Service version
        exporter: Custom span exporter (default: ConsoleSpanExporter)

    Returns:
        Tracer instance or None if OpenTelemetry not available

    Usage:
        # Basic setup (console export)
        setup_tracing(service_name="my-service")

        # With Jaeger exporter
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        setup_tracing(service_name="my-service", exporter=jaeger_exporter)
    """
    global _tracer

    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available, tracing disabled")
        return None

    # Create resource
    resource = Resource(attributes={
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_VERSION: service_version,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add exporter
    if exporter is None:
        exporter = ConsoleSpanExporter()

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer
    _tracer = trace.get_tracer(__name__)

    logger.info(f"OpenTelemetry tracing configured: {service_name} v{service_version}")

    return _tracer


def get_tracer() -> Optional[Any]:
    """Get global tracer instance."""
    global _tracer

    if not OTEL_AVAILABLE:
        return None

    if _tracer is None:
        # Auto-setup with defaults
        _tracer = setup_tracing()

    return _tracer


# ============================================================================
# Tracing Context Manager
# ============================================================================

@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None
):
    """
    Context manager for creating traced spans.

    Args:
        name: Span name
        attributes: Span attributes (metadata)
        kind: Span kind (SERVER, CLIENT, INTERNAL, etc.)

    Usage:
        with trace_span("reasoning_operation", attributes={"mode": "standard"}):
            # Do work
            result = await engine.reason(query, features, context)
    """
    tracer = get_tracer()

    if tracer is None:
        # No-op if tracing not available
        yield None
        return

    # Create span
    span_kind = kind or trace.SpanKind.INTERNAL
    with tracer.start_as_current_span(name, kind=span_kind) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            # Record exception
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# ============================================================================
# Tracing Decorators
# ============================================================================

def trace_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator for tracing functions.

    Args:
        span_name: Custom span name (default: function name)
        attributes: Static attributes to add

    Usage:
        @trace_function(span_name="custom_name", attributes={"component": "reasoning"})
        def my_function(arg1, arg2):
            # Function body
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            attrs = attributes or {}

            # Add function metadata
            attrs["function.name"] = func.__name__
            attrs["function.module"] = func.__module__

            with trace_span(name, attributes=attrs):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def trace_async_function(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator for tracing async functions.

    Usage:
        @trace_async_function(span_name="reasoning", attributes={"component": "engine"})
        async def my_async_function(arg1, arg2):
            # Async function body
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = span_name or func.__name__
            attrs = attributes or {}

            # Add function metadata
            attrs["function.name"] = func.__name__
            attrs["function.module"] = func.__module__

            with trace_span(name, attributes=attrs):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Reasoning-Specific Tracing
# ============================================================================

class ReasoningTracer:
    """
    High-level tracer for reasoning operations.

    Provides pre-configured tracing for reasoning engine with standard attributes.
    """

    def __init__(self, tracer: Optional[Any] = None):
        self.tracer = tracer or get_tracer()
        self.enabled = self.tracer is not None

    @contextmanager
    def trace_reasoning(
        self,
        mode: str,
        query_text: str,
        **extra_attributes
    ):
        """
        Trace a reasoning operation.

        Args:
            mode: Reasoning mode (fast/standard/deep)
            query_text: Query text (truncated for privacy)
            **extra_attributes: Additional span attributes

        Usage:
            tracer = ReasoningTracer()

            with tracer.trace_reasoning(mode="standard", query_text="What is..."):
                result = await engine.reason(query, features, context)
        """
        if not self.enabled:
            yield None
            return

        # Prepare attributes
        attributes = {
            "reasoning.mode": mode,
            "reasoning.query_length": len(query_text),
            "reasoning.query_preview": query_text[:100],  # Privacy: only first 100 chars
            **extra_attributes
        }

        with trace_span("reasoning_operation", attributes=attributes, kind=trace.SpanKind.INTERNAL) as span:
            start_time = time.time()

            try:
                yield span
            finally:
                # Add result attributes
                duration_ms = (time.time() - start_time) * 1000
                if span:
                    span.set_attribute("reasoning.duration_ms", duration_ms)

    @contextmanager
    def trace_component(
        self,
        component_name: str,
        **attributes
    ):
        """
        Trace a reasoning component (planner, reasoner, verifier, etc.).

        Args:
            component_name: Component name
            **attributes: Component-specific attributes

        Usage:
            with tracer.trace_component("planner", complexity=0.7):
                intent = planner.analyze_intent(query, features)
        """
        if not self.enabled:
            yield None
            return

        attrs = {
            "component.name": component_name,
            **attributes
        }

        with trace_span(f"reasoning.{component_name}", attributes=attrs):
            yield

    def add_event(self, span: Any, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Add event to current span.

        Args:
            span: Current span
            name: Event name
            attributes: Event attributes

        Usage:
            with tracer.trace_reasoning(...) as span:
                # ... do work ...
                tracer.add_event(span, "verification_failed", {"reason": "inconsistency"})
        """
        if span and self.enabled:
            span.add_event(name, attributes=attributes or {})

    def set_result(
        self,
        span: Any,
        confidence: float,
        chain_length: int,
        success: bool = True
    ):
        """
        Set result attributes on span.

        Args:
            span: Current span
            confidence: Result confidence
            chain_length: Number of reasoning steps
            success: Whether operation succeeded

        Usage:
            with tracer.trace_reasoning(...) as span:
                result = await engine.reason(...)
                tracer.set_result(span, result.total_confidence, len(result.chain))
        """
        if span and self.enabled:
            span.set_attribute("reasoning.confidence", confidence)
            span.set_attribute("reasoning.chain_length", chain_length)
            span.set_attribute("reasoning.success", success)

            if success:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, "Reasoning failed"))


# ============================================================================
# Convenience Functions
# ============================================================================

def get_reasoning_tracer() -> ReasoningTracer:
    """Get global reasoning tracer instance."""
    return ReasoningTracer()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup tracing
    setup_tracing(service_name="example-service")

    # Use high-level tracer
    tracer = ReasoningTracer()

    # Simulate reasoning operation
    with tracer.trace_reasoning(mode="standard", query_text="What is machine learning?") as span:
        # Simulate components
        with tracer.trace_component("planner"):
            time.sleep(0.01)  # Simulate work

        with tracer.trace_component("reasoner"):
            time.sleep(0.05)  # Simulate work

        with tracer.trace_component("verifier"):
            time.sleep(0.02)  # Simulate work

        # Set result
        tracer.set_result(span, confidence=0.85, chain_length=5, success=True)

    print("\nExample traces exported to console (see above)")
