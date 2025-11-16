#!/usr/bin/env python3
"""
Auto-Instrumentation for FastAPI and HoloLoom Components
==========================================================

Provides automatic and manual instrumentation for tracing.

Features:
- FastAPI auto-instrumentation
- Custom tracing middleware
- HoloLoom component instrumentation
- Request/response tracing
- WebSocket tracing
- Database query tracing

Usage:
    from fastapi import FastAPI
    from HoloLoom.tracing import instrument_app, TracingConfig, init_tracing

    app = FastAPI()

    # Initialize tracing
    config = TracingConfig(exporter="jaeger")
    init_tracing(config)

    # Auto-instrument FastAPI
    instrument_app(app)

Created: 2025-11-16
"""

import logging
import time
from typing import Callable, Optional, Dict, Any
from functools import wraps

# Try to import FastAPI and Starlette
try:
    from fastapi import FastAPI, Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    Request = None
    Response = None
    BaseHTTPMiddleware = None

# Try to import OpenTelemetry instrumentations
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OTEL_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    OTEL_INSTRUMENTATION_AVAILABLE = False

from HoloLoom.tracing.opentelemetry_integration import (
    TRACING_AVAILABLE,
    get_tracer,
    create_span_attributes,
)

if TRACING_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
else:
    trace = None
    Status = None
    StatusCode = None
    SpanKind = None

logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI Auto-Instrumentation
# ============================================================================

def instrument_app(app: Optional[Any] = None) -> None:
    """
    Auto-instrument FastAPI application with OpenTelemetry.

    This instruments:
    - All HTTP endpoints (automatic span creation)
    - Outgoing HTTP requests
    - Logging (correlates logs with traces)

    Args:
        app: FastAPI application instance (optional if already initialized)

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> instrument_app(app)
    """
    if not OTEL_INSTRUMENTATION_AVAILABLE:
        logger.warning(
            "OpenTelemetry instrumentation not available. "
            "Install with: pip install opentelemetry-instrumentation-fastapi "
            "opentelemetry-instrumentation-requests opentelemetry-instrumentation-logging"
        )
        return

    if not TRACING_AVAILABLE:
        logger.warning("OpenTelemetry SDK not available, skipping instrumentation")
        return

    # Instrument FastAPI if app provided
    if app is not None and FASTAPI_AVAILABLE:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")

    # Instrument outgoing requests
    RequestsInstrumentor().instrument()
    logger.info("HTTP requests instrumented with OpenTelemetry")

    # Instrument logging
    LoggingInstrumentor().instrument(set_logging_format=True)
    logger.info("Logging instrumented with OpenTelemetry")


def instrument_hololoom_components() -> None:
    """
    Instrument HoloLoom components (currently a placeholder).

    In the future, this could instrument:
    - Database drivers (Neo4j, Qdrant)
    - Message queues
    - Custom protocols
    """
    logger.info("HoloLoom component instrumentation placeholder")
    # Future: Add database, message queue instrumentation
    pass


# ============================================================================
# Custom Tracing Middleware
# ============================================================================

class TracingMiddleware(BaseHTTPMiddleware if FASTAPI_AVAILABLE else object):
    """
    Custom tracing middleware for additional context and control.

    This middleware adds:
    - Custom span attributes (user_id, request_id, etc.)
    - Error tracking and status codes
    - Request/response size tracking
    - Custom events
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing."""
        if not TRACING_AVAILABLE:
            # No tracing available, just pass through
            return await call_next(request)

        tracer = get_tracer(__name__)

        # Create span for request
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            kind=SpanKind.SERVER if TRACING_AVAILABLE else None,
        ) as span:
            # Add request attributes
            if span.is_recording():
                span.set_attributes(
                    create_span_attributes(
                        http_method=request.method,
                        http_url=str(request.url),
                        http_route=request.url.path,
                        http_scheme=request.url.scheme,
                        http_host=request.url.hostname,
                        http_user_agent=request.headers.get("user-agent", ""),
                        http_client_ip=request.client.host if request.client else "",
                    )
                )

                # Add custom attributes from request state
                if hasattr(request.state, "user"):
                    span.set_attribute("user.id", request.state.user.username)

                if hasattr(request.state, "request_id"):
                    span.set_attribute("request.id", request.state.request_id)

            start_time = time.time()

            try:
                # Process request
                response = await call_next(request)

                # Add response attributes
                if span.is_recording():
                    duration_ms = (time.time() - start_time) * 1000

                    span.set_attributes(
                        create_span_attributes(
                            http_status_code=response.status_code,
                            http_response_size=response.headers.get("content-length", 0),
                            duration_ms=duration_ms,
                        )
                    )

                    # Set status based on response code
                    if response.status_code >= 500:
                        span.set_status(Status(StatusCode.ERROR, "Server error"))
                    elif response.status_code >= 400:
                        span.set_status(Status(StatusCode.ERROR, "Client error"))
                    else:
                        span.set_status(Status(StatusCode.OK))

                return response

            except Exception as e:
                # Record exception
                if span.is_recording():
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


# ============================================================================
# Function Decorators
# ============================================================================

def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[Any] = None,
):
    """
    Decorator to trace a function.

    Args:
        name: Span name (uses function name if None)
        attributes: Additional span attributes
        kind: Span kind (INTERNAL, CLIENT, SERVER, etc.)

    Example:
        >>> @traced(attributes={"component": "memory"})
        ... async def recall_memories(query: str):
        ...     # function body
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not TRACING_AVAILABLE:
                return await func(*args, **kwargs)

            tracer = get_tracer(func.__module__)
            span_name = name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(
                span_name,
                kind=kind or (SpanKind.INTERNAL if TRACING_AVAILABLE else None),
            ) as span:
                # Add attributes
                if span.is_recording() and attributes:
                    span.set_attributes(create_span_attributes(**attributes))

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK) if TRACING_AVAILABLE else None)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)) if TRACING_AVAILABLE else None)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not TRACING_AVAILABLE:
                return func(*args, **kwargs)

            tracer = get_tracer(func.__module__)
            span_name = name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(
                span_name,
                kind=kind or (SpanKind.INTERNAL if TRACING_AVAILABLE else None),
            ) as span:
                # Add attributes
                if span.is_recording() and attributes:
                    span.set_attributes(create_span_attributes(**attributes))

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK) if TRACING_AVAILABLE else None)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)) if TRACING_AVAILABLE else None)
                    raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# WebSocket Tracing
# ============================================================================

class WebSocketTracing:
    """Helper for tracing WebSocket connections."""

    def __init__(self, websocket: Any, tracer: Any = None):
        """
        Initialize WebSocket tracing.

        Args:
            websocket: WebSocket connection
            tracer: OpenTelemetry tracer (uses global if None)
        """
        self.websocket = websocket
        self.tracer = tracer or get_tracer(__name__)
        self.connection_span = None

    async def __aenter__(self):
        """Start tracing the connection."""
        if TRACING_AVAILABLE:
            self.connection_span = self.tracer.start_as_current_span(
                "websocket.connection",
                kind=SpanKind.SERVER,
            ).__enter__()

            if self.connection_span.is_recording():
                self.connection_span.set_attribute("websocket.id", id(self.websocket))

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End tracing the connection."""
        if self.connection_span and TRACING_AVAILABLE:
            if exc_type is not None:
                self.connection_span.record_exception(exc_val)
                self.connection_span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self.connection_span.set_status(Status(StatusCode.OK))

            self.connection_span.__exit__(exc_type, exc_val, exc_tb)

        return False

    def trace_message(self, message_type: str, attributes: Optional[Dict] = None):
        """
        Create a span for a WebSocket message.

        Args:
            message_type: Type of message (e.g., "text", "binary")
            attributes: Additional attributes

        Returns:
            Context manager for the span
        """
        if not TRACING_AVAILABLE:
            import contextlib
            return contextlib.nullcontext()

        attrs = {"websocket.message.type": message_type}
        if attributes:
            attrs.update(attributes)

        return self.tracer.start_as_current_span(
            f"websocket.{message_type}",
            attributes=create_span_attributes(**attrs),
        )


# ============================================================================
# Database Query Tracing
# ============================================================================

def trace_database_query(
    operation: str,
    query: str,
    database: str = "unknown",
    attributes: Optional[Dict] = None,
):
    """
    Create a span for a database query.

    Args:
        operation: Operation type (SELECT, INSERT, etc.)
        query: SQL/Cypher query
        database: Database name
        attributes: Additional attributes

    Returns:
        Context manager for the span

    Example:
        >>> with trace_database_query("SELECT", "SELECT * FROM users", "postgres"):
        ...     cursor.execute(query)
    """
    if not TRACING_AVAILABLE:
        import contextlib
        return contextlib.nullcontext()

    tracer = get_tracer(__name__)

    # Truncate long queries
    query_preview = query[:200] + "..." if len(query) > 200 else query

    attrs = {
        "db.operation": operation,
        "db.statement": query_preview,
        "db.system": database,
    }

    if attributes:
        attrs.update(attributes)

    return tracer.start_as_current_span(
        f"db.{operation.lower()}",
        kind=SpanKind.CLIENT if TRACING_AVAILABLE else None,
        attributes=create_span_attributes(**attrs),
    )


# ============================================================================
# Manual Span Creation Helpers
# ============================================================================

def create_span(
    name: str,
    attributes: Optional[Dict] = None,
    kind: Optional[Any] = None,
):
    """
    Create a manual span.

    Args:
        name: Span name
        attributes: Span attributes
        kind: Span kind

    Returns:
        Context manager for the span

    Example:
        >>> with create_span("custom_operation", {"key": "value"}):
        ...     # do work
    """
    if not TRACING_AVAILABLE:
        import contextlib
        return contextlib.nullcontext()

    tracer = get_tracer(__name__)

    return tracer.start_as_current_span(
        name,
        kind=kind or (SpanKind.INTERNAL if TRACING_AVAILABLE else None),
        attributes=create_span_attributes(**(attributes or {})),
    )


def add_span_event(
    name: str,
    attributes: Optional[Dict] = None,
):
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Event attributes

    Example:
        >>> add_span_event("cache_hit", {"key": "query_123"})
    """
    if not TRACING_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.add_event(name, attributes=create_span_attributes(**(attributes or {})))


def set_span_attribute(key: str, value: Any):
    """
    Set an attribute on the current span.

    Args:
        key: Attribute key
        value: Attribute value

    Example:
        >>> set_span_attribute("user.id", "123")
    """
    if not TRACING_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        attrs = create_span_attributes(**{key: value})
        span.set_attributes(attrs)


def record_exception(exception: Exception):
    """
    Record an exception on the current span.

    Args:
        exception: Exception to record

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     record_exception(e)
        ...     raise
    """
    if not TRACING_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))
