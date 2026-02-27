"""
Trace Context Propagation - W3C Trace Context support.

Provides context management for distributed trace propagation across
service boundaries using W3C Trace Context standard.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from hololoom.telemetry.protocol import SpanContext, SpanData

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTEXT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

# Context variable for current span (async-safe)
_current_span: contextvars.ContextVar[Optional[SpanData]] = contextvars.ContextVar(
    "current_span", default=None
)

# Context variable for current context (async-safe)
_current_context: contextvars.ContextVar[Optional[SpanContext]] = contextvars.ContextVar(
    "current_context", default=None
)

# Baggage items (key-value pairs propagated across boundaries)
_baggage: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "baggage", default={}
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceContextManager:
    """Manages trace context propagation."""

    @staticmethod
    def get_current_span() -> Optional[SpanData]:
        """Get the current span from async-safe context."""
        return _current_span.get()

    @staticmethod
    def set_current_span(span: Optional[SpanData]) -> contextvars.Token:
        """Set the current span, returning a token for restoration."""
        return _current_span.set(span)

    @staticmethod
    def reset_span(token: contextvars.Token) -> None:
        """Reset span to previous value using token."""
        _current_span.reset(token)

    @staticmethod
    def get_current_context() -> Optional[SpanContext]:
        """Get the current span context."""
        span = _current_span.get()
        if span is not None:
            return span.context
        return _current_context.get()

    @staticmethod
    def set_current_context(context: Optional[SpanContext]) -> contextvars.Token:
        """Set the current context, returning a token for restoration."""
        return _current_context.set(context)

    @staticmethod
    def reset_context(token: contextvars.Token) -> None:
        """Reset context to previous value using token."""
        _current_context.reset(token)

    @staticmethod
    @contextmanager
    def span_scope(span: SpanData):
        """Context manager for setting current span."""
        token = _current_span.set(span)
        try:
            yield span
        finally:
            _current_span.reset(token)

    @staticmethod
    @contextmanager
    def context_scope(context: SpanContext):
        """Context manager for setting current context."""
        token = _current_context.set(context)
        try:
            yield context
        finally:
            _current_context.reset(token)


# ═══════════════════════════════════════════════════════════════════════════════
#  BAGGAGE
# ═══════════════════════════════════════════════════════════════════════════════


def get_baggage(key: str) -> Optional[str]:
    """Get a baggage item."""
    baggage = _baggage.get()
    return baggage.get(key)


def set_baggage(key: str, value: str) -> None:
    """Set a baggage item."""
    baggage = _baggage.get().copy()
    baggage[key] = value
    _baggage.set(baggage)


def get_all_baggage() -> Dict[str, str]:
    """Get all baggage items."""
    return _baggage.get().copy()


def clear_baggage() -> None:
    """Clear all baggage items."""
    _baggage.set({})


# ═══════════════════════════════════════════════════════════════════════════════
#  W3C TRACE CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════


class W3CTraceContext:
    """W3C Trace Context format handler."""

    TRACEPARENT_HEADER = "traceparent"
    TRACESTATE_HEADER = "tracestate"
    BAGGAGE_HEADER = "baggage"

    @classmethod
    def inject(cls, carrier: Dict[str, str]) -> None:
        """Inject trace context into HTTP headers."""
        context = get_current_context()
        if context is None:
            return

        # traceparent: version-traceid-spanid-flags
        traceparent = f"00-{context.trace_id}-{context.span_id}-{context.trace_flags:02x}"
        carrier[cls.TRACEPARENT_HEADER] = traceparent

        # tracestate (vendor-specific key-value pairs)
        if context.trace_state:
            carrier[cls.TRACESTATE_HEADER] = context.trace_state

        # baggage
        baggage = get_all_baggage()
        if baggage:
            carrier[cls.BAGGAGE_HEADER] = ",".join(
                f"{k}={v}" for k, v in baggage.items()
            )

    @classmethod
    def extract(cls, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from HTTP headers."""
        traceparent = carrier.get(cls.TRACEPARENT_HEADER)
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) != 4:
            return None

        try:
            version, trace_id, span_id, flags = parts
            if version != "00":
                return None  # Unknown version
            if len(trace_id) != 32 or len(span_id) != 16:
                return None  # Invalid format

            context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=int(flags, 16),
                trace_state=carrier.get(cls.TRACESTATE_HEADER, ""),
                is_remote=True,
            )

            # Extract baggage
            baggage_header = carrier.get(cls.BAGGAGE_HEADER, "")
            if baggage_header:
                for item in baggage_header.split(","):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        set_baggage(key.strip(), value.strip())

            return context
        except (ValueError, TypeError):
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def get_current_context() -> Optional[SpanContext]:
    """Get the current span context."""
    return TraceContextManager.get_current_context()


def set_current_context(context: SpanContext) -> contextvars.Token:
    """Set the current span context."""
    return TraceContextManager.set_current_context(context)


def clear_context() -> None:
    """Clear current context and baggage."""
    _current_span.set(None)
    _current_context.set(None)
    clear_baggage()


def run_with_context(
    context: SpanContext,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a function with the given trace context."""
    with TraceContextManager.context_scope(context):
        return func(*args, **kwargs)


async def run_with_context_async(
    context: SpanContext,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run an async function with the given trace context."""
    with TraceContextManager.context_scope(context):
        return await func(*args, **kwargs)
