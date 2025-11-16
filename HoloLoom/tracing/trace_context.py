#!/usr/bin/env python3
"""
Trace Context Propagation
==========================

Handles context propagation across service boundaries for distributed tracing.

Features:
- HTTP header propagation (W3C Trace Context)
- WebSocket context propagation
- Manual context injection/extraction
- Context carrier utilities

Usage:
    from HoloLoom.tracing import extract_trace_context, inject_trace_context

    # Server side (extract from incoming request)
    ctx = extract_trace_context(request.headers)

    # Client side (inject into outgoing request)
    headers = {}
    inject_trace_context(headers)
    requests.get("http://api.example.com", headers=headers)

Created: 2025-11-16
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

# Try to import OpenTelemetry propagation
try:
    from opentelemetry import trace, context
    from opentelemetry.propagate import extract, inject, set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    PROPAGATION_AVAILABLE = True
except ImportError:
    PROPAGATION_AVAILABLE = False
    trace = None
    context = None
    extract = None
    inject = None

from HoloLoom.tracing.opentelemetry_integration import TRACING_AVAILABLE

logger = logging.getLogger(__name__)


# ============================================================================
# Context Propagation
# ============================================================================

@dataclass
class TraceContext:
    """Container for trace context information."""

    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    trace_flags: Optional[str] = None
    trace_state: Optional[str] = None

    @classmethod
    def from_current_span(cls) -> "TraceContext":
        """Extract context from current span."""
        if not TRACING_AVAILABLE:
            return cls()

        span = trace.get_current_span()
        if not span or not span.is_recording():
            return cls()

        span_context = span.get_span_context()
        return cls(
            trace_id=format(span_context.trace_id, "032x") if span_context.trace_id else None,
            span_id=format(span_context.span_id, "016x") if span_context.span_id else None,
            trace_flags=format(span_context.trace_flags, "02x") if span_context.trace_flags else None,
            trace_state=str(span_context.trace_state) if span_context.trace_state else None,
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        result = {}
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.trace_flags:
            result["trace_flags"] = self.trace_flags
        if self.trace_state:
            result["trace_state"] = self.trace_state
        return result


def propagate_trace_context() -> TraceContext:
    """
    Get current trace context for propagation.

    Returns:
        TraceContext with current trace information

    Example:
        >>> ctx = propagate_trace_context()
        >>> print(f"Trace ID: {ctx.trace_id}")
    """
    return TraceContext.from_current_span()


def extract_trace_context(carrier: Dict[str, str]) -> Optional[Any]:
    """
    Extract trace context from carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary containing trace context (HTTP headers)

    Returns:
        OpenTelemetry context or None

    Example:
        >>> # In FastAPI endpoint
        >>> ctx = extract_trace_context(request.headers)
        >>> # Use context for child spans
    """
    if not PROPAGATION_AVAILABLE:
        logger.debug("Context propagation not available")
        return None

    try:
        # Extract context from carrier
        ctx = extract(carrier)
        return ctx
    except Exception as e:
        logger.warning(f"Failed to extract trace context: {e}")
        return None


def inject_trace_context(carrier: Dict[str, str]) -> Dict[str, str]:
    """
    Inject current trace context into carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary to inject context into

    Returns:
        Updated carrier with trace context

    Example:
        >>> headers = {}
        >>> inject_trace_context(headers)
        >>> response = requests.get("http://api.example.com", headers=headers)
    """
    if not PROPAGATION_AVAILABLE:
        logger.debug("Context propagation not available")
        return carrier

    try:
        # Inject current context into carrier
        inject(carrier)
        return carrier
    except Exception as e:
        logger.warning(f"Failed to inject trace context: {e}")
        return carrier


# ============================================================================
# WebSocket Context Propagation
# ============================================================================

def extract_websocket_context(
    query_params: Dict[str, str],
    headers: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """
    Extract trace context from WebSocket connection.

    WebSocket doesn't have headers in the same way as HTTP, so we support:
    1. Query parameters (traceparent, tracestate)
    2. Initial HTTP upgrade headers (if available)

    Args:
        query_params: WebSocket query parameters
        headers: Optional HTTP upgrade headers

    Returns:
        OpenTelemetry context or None

    Example:
        >>> # In WebSocket endpoint
        >>> @app.websocket("/ws")
        >>> async def websocket_endpoint(
        ...     websocket: WebSocket,
        ...     traceparent: Optional[str] = Query(None),
        ...     tracestate: Optional[str] = Query(None)
        ... ):
        ...     ctx = extract_websocket_context({
        ...         "traceparent": traceparent,
        ...         "tracestate": tracestate
        ...     })
    """
    if not PROPAGATION_AVAILABLE:
        return None

    # Try headers first (HTTP upgrade)
    if headers:
        ctx = extract_trace_context(headers)
        if ctx:
            return ctx

    # Try query parameters
    carrier = {}
    if "traceparent" in query_params and query_params["traceparent"]:
        carrier["traceparent"] = query_params["traceparent"]
    if "tracestate" in query_params and query_params["tracestate"]:
        carrier["tracestate"] = query_params["tracestate"]

    if carrier:
        return extract_trace_context(carrier)

    return None


def create_websocket_query_params(trace_ctx: Optional[TraceContext] = None) -> Dict[str, str]:
    """
    Create WebSocket query parameters with trace context.

    Args:
        trace_ctx: Trace context (uses current if None)

    Returns:
        Dictionary with traceparent and tracestate parameters

    Example:
        >>> params = create_websocket_query_params()
        >>> ws_url = f"ws://localhost:8000/ws?{urlencode(params)}"
    """
    if not PROPAGATION_AVAILABLE:
        return {}

    if trace_ctx is None:
        trace_ctx = propagate_trace_context()

    params = {}

    # Create W3C traceparent header
    if trace_ctx.trace_id and trace_ctx.span_id:
        # Format: version-trace_id-parent_id-trace_flags
        traceparent = f"00-{trace_ctx.trace_id}-{trace_ctx.span_id}-{trace_ctx.trace_flags or '01'}"
        params["traceparent"] = traceparent

    if trace_ctx.trace_state:
        params["tracestate"] = trace_ctx.trace_state

    return params


# ============================================================================
# Custom Context Carriers
# ============================================================================

class DictCarrier:
    """Dictionary-based carrier for context propagation."""

    def __init__(self, data: Optional[Dict[str, str]] = None):
        self.data = data or {}

    def get(self, key: str) -> Optional[str]:
        """Get value from carrier."""
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        """Set value in carrier."""
        self.data[key] = value

    def keys(self):
        """Get all keys."""
        return self.data.keys()

    def __getitem__(self, key: str) -> str:
        """Dictionary-style access."""
        return self.data[key]

    def __setitem__(self, key: str, value: str) -> None:
        """Dictionary-style assignment."""
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data

    def to_dict(self) -> Dict[str, str]:
        """Convert to plain dictionary."""
        return self.data.copy()


# ============================================================================
# Propagation Format Configuration
# ============================================================================

def configure_propagators(formats: list[str] = None):
    """
    Configure global propagation formats.

    Supported formats:
    - "w3c": W3C Trace Context (default)
    - "b3": Zipkin B3 format
    - "b3multi": B3 multi-header format

    Args:
        formats: List of format names (default: ["w3c"])

    Example:
        >>> configure_propagators(["w3c", "b3"])
    """
    if not PROPAGATION_AVAILABLE:
        logger.warning("Propagation not available")
        return

    if formats is None:
        formats = ["w3c"]

    propagators = []

    for fmt in formats:
        if fmt == "w3c":
            propagators.append(TraceContextTextMapPropagator())
        elif fmt in ("b3", "b3multi"):
            propagators.append(B3MultiFormat())
        else:
            logger.warning(f"Unknown propagation format: {fmt}")

    if propagators:
        from opentelemetry.propagators.composite import CompositePropagator
        composite = CompositePropagator(propagators)
        set_global_textmap(composite)
        logger.info(f"Configured propagators: {formats}")


# ============================================================================
# Utilities
# ============================================================================

def get_trace_id() -> Optional[str]:
    """
    Get current trace ID.

    Returns:
        Trace ID as hex string or None

    Example:
        >>> trace_id = get_trace_id()
        >>> logger.info(f"Trace ID: {trace_id}")
    """
    if not TRACING_AVAILABLE:
        return None

    span = trace.get_current_span()
    if not span or not span.is_recording():
        return None

    span_context = span.get_span_context()
    if span_context.trace_id:
        return format(span_context.trace_id, "032x")

    return None


def get_span_id() -> Optional[str]:
    """
    Get current span ID.

    Returns:
        Span ID as hex string or None

    Example:
        >>> span_id = get_span_id()
        >>> logger.info(f"Span ID: {span_id}")
    """
    if not TRACING_AVAILABLE:
        return None

    span = trace.get_current_span()
    if not span or not span.is_recording():
        return None

    span_context = span.get_span_context()
    if span_context.span_id:
        return format(span_context.span_id, "016x")

    return None


def is_sampled() -> bool:
    """
    Check if current trace is sampled.

    Returns:
        True if sampled, False otherwise

    Example:
        >>> if is_sampled():
        ...     # Add expensive attributes
    """
    if not TRACING_AVAILABLE:
        return False

    span = trace.get_current_span()
    if not span or not span.is_recording():
        return False

    span_context = span.get_span_context()
    return span_context.trace_flags.sampled if span_context.trace_flags else False


# ============================================================================
# Request ID Integration
# ============================================================================

def get_or_create_request_id(headers: Dict[str, str]) -> str:
    """
    Get request ID from headers or create from trace ID.

    Args:
        headers: HTTP headers

    Returns:
        Request ID (trace ID or generated)

    Example:
        >>> request_id = get_or_create_request_id(request.headers)
        >>> logger.info(f"Request ID: {request_id}", extra={"request_id": request_id})
    """
    # Check for existing request ID header
    if "x-request-id" in headers:
        return headers["x-request-id"]

    # Use trace ID if available
    trace_id = get_trace_id()
    if trace_id:
        return trace_id

    # Generate new ID
    import uuid
    return str(uuid.uuid4())


def add_request_id_to_span(request_id: str):
    """
    Add request ID as span attribute.

    Args:
        request_id: Request ID to add

    Example:
        >>> request_id = get_or_create_request_id(request.headers)
        >>> add_request_id_to_span(request_id)
    """
    if not TRACING_AVAILABLE:
        return

    span = trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute("request.id", request_id)
