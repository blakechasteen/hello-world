"""
OpenTelemetry Tracer - Distributed tracing implementation.

Provides span management, context propagation, and exporter integration.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from threading import local
from typing import Any, TypeVar

from hololoom.telemetry.config import SamplingStrategy, TracingConfig
from hololoom.telemetry.protocol import (
    SpanContext,
    SpanData,
    SpanKind,
    SpanProcessorProtocol,
    SpanStatus,
    TracingProtocol,
    create_span_context,
    generate_span_id,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Thread-local storage for current span
_context = local()


# ═══════════════════════════════════════════════════════════════════════════════
#  SPAN PROCESSORS
# ═══════════════════════════════════════════════════════════════════════════════


class ConsoleSpanProcessor:
    """Exports spans to console (development)."""

    def on_start(self, span: SpanData) -> None:
        """Called when span starts."""
        pass  # No output on start

    def on_end(self, span: SpanData) -> None:
        """Called when span ends - print to console."""
        duration = span.duration_ms
        status = "✓" if span.status == SpanStatus.OK else "✗" if span.status == SpanStatus.ERROR else "?"
        logger.info(
            f"[SPAN] {status} {span.name} ({duration:.2f}ms) "
            f"trace={span.context.trace_id[:8]}... span={span.context.span_id[:8]}..."
        )

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Force flush (no-op for console)."""
        return True


class BatchSpanProcessor:
    """Batches spans before export (production)."""

    def __init__(
        self,
        exporter: SpanProcessorProtocol,
        max_batch_size: int = 512,
        max_export_batch_size: int = 512,
        schedule_delay_ms: int = 5000,
    ):
        self.exporter = exporter
        self.max_batch_size = max_batch_size
        self.max_export_batch_size = max_export_batch_size
        self.schedule_delay_ms = schedule_delay_ms
        self._pending: list[SpanData] = []
        self._shutdown = False

    def on_start(self, span: SpanData) -> None:
        """Called when span starts."""
        pass

    def on_end(self, span: SpanData) -> None:
        """Called when span ends - add to batch."""
        if self._shutdown:
            return
        self._pending.append(span)
        if len(self._pending) >= self.max_batch_size:
            self._export_batch()

    def _export_batch(self) -> None:
        """Export current batch."""
        if not self._pending:
            return
        batch = self._pending[: self.max_export_batch_size]
        self._pending = self._pending[self.max_export_batch_size:]
        for span in batch:
            self.exporter.on_end(span)

    def shutdown(self) -> None:
        """Shutdown and flush remaining spans."""
        self._shutdown = True
        self.force_flush()

    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Flush all pending spans."""
        while self._pending:
            self._export_batch()
        return True


# ═══════════════════════════════════════════════════════════════════════════════
#  SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════


class Sampler:
    """Determines whether to sample a trace."""

    def __init__(self, config: TracingConfig):
        self.config = config
        self._ratio_threshold = int(config.sampling_ratio * 0xFFFFFFFF)

    def should_sample(
        self,
        trace_id: str,
        parent_context: SpanContext | None = None,
    ) -> bool:
        """Determine if this trace should be sampled."""
        strategy = self.config.sampling_strategy

        if strategy == SamplingStrategy.ALWAYS_ON:
            return True
        elif strategy == SamplingStrategy.ALWAYS_OFF:
            return False
        elif strategy == SamplingStrategy.PARENT_BASED:
            if parent_context is not None:
                return parent_context.trace_flags & 0x01 == 1
            return True  # No parent, sample
        elif strategy == SamplingStrategy.RATIO:
            # Use trace_id as deterministic random seed
            trace_hash = int(trace_id[:8], 16)
            return trace_hash <= self._ratio_threshold
        return True


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACER
# ═══════════════════════════════════════════════════════════════════════════════


class OTelTracer(TracingProtocol):
    """OpenTelemetry-compatible tracer implementation."""

    def __init__(
        self,
        config: TracingConfig | None = None,
        processors: list[SpanProcessorProtocol] | None = None,
    ):
        self.config = config or TracingConfig()
        self.sampler = Sampler(self.config)
        self.processors = processors or [ConsoleSpanProcessor()]
        self._enabled = self.config.enabled

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        parent: SpanContext | None = None,
    ) -> SpanData:
        """Start a new span."""
        # Get parent from argument or current context
        if parent is None:
            current = self.get_current_span()
            if current is not None:
                parent = current.context

        # Create context (inherit trace_id from parent or generate new)
        if parent is not None:
            context = SpanContext(
                trace_id=parent.trace_id,
                span_id=generate_span_id(),
                is_remote=False,
            )
            parent_id = parent.span_id
        else:
            context = create_span_context()
            parent_id = None

        # Check sampling
        if not self.sampler.should_sample(context.trace_id, parent):
            # Create a non-recording span
            return SpanData(
                name=name,
                context=SpanContext(
                    trace_id=context.trace_id,
                    span_id=context.span_id,
                    trace_flags=0,  # Not sampled
                ),
                kind=kind,
            )

        # Create recording span
        span = SpanData(
            name=name,
            context=SpanContext(
                trace_id=context.trace_id,
                span_id=context.span_id,
                trace_flags=1,  # Sampled
            ),
            parent_id=parent_id,
            kind=kind,
            attributes=attributes or {},
            start_time=datetime.utcnow(),
        )

        # Add service info
        span.set_attribute("service.name", self.config.service_name)
        span.set_attribute("service.version", self.config.service_version)

        # Notify processors
        if self._enabled:
            for processor in self.processors:
                try:
                    processor.on_start(span)
                except Exception as e:
                    logger.warning(f"Span processor on_start error: {e}")

        return span

    def end_span(self, span: SpanData) -> None:
        """End a span and export it."""
        span.end()

        # Set OK status if not already set
        if span.status == SpanStatus.UNSET:
            span.set_status(SpanStatus.OK)

        # Notify processors
        if self._enabled and span.context.trace_flags & 0x01:
            for processor in self.processors:
                try:
                    processor.on_end(span)
                except Exception as e:
                    logger.warning(f"Span processor on_end error: {e}")

    def get_current_span(self) -> SpanData | None:
        """Get the current span from thread-local context."""
        return getattr(_context, "current_span", None)

    def _set_current_span(self, span: SpanData | None) -> None:
        """Set the current span in thread-local context."""
        _context.current_span = span

    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject trace context into HTTP headers (W3C Trace Context)."""
        span = self.get_current_span()
        if span is None:
            return

        ctx = span.context
        # traceparent: version-traceid-spanid-flags
        traceparent = f"00-{ctx.trace_id}-{ctx.span_id}-{ctx.trace_flags:02x}"
        carrier["traceparent"] = traceparent

        if ctx.trace_state:
            carrier["tracestate"] = ctx.trace_state

    def extract_context(self, carrier: dict[str, str]) -> SpanContext | None:
        """Extract trace context from HTTP headers (W3C Trace Context)."""
        traceparent = carrier.get("traceparent")
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) != 4:
            return None

        try:
            version, trace_id, span_id, flags = parts
            return SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=int(flags, 16),
                trace_state=carrier.get("tracestate", ""),
                is_remote=True,
            )
        except (ValueError, TypeError):
            return None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Context manager for tracing a block of code."""
        span = self.start_span(name, kind=kind, attributes=attributes)
        previous = self.get_current_span()
        self._set_current_span(span)
        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("exception.type", type(e).__name__)
            span.set_attribute("exception.message", str(e))
            raise
        finally:
            self._set_current_span(previous)
            self.end_span(span)

    def shutdown(self) -> None:
        """Shutdown all processors."""
        for processor in self.processors:
            try:
                processor.shutdown()
            except Exception as e:
                logger.warning(f"Processor shutdown error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL TRACER
# ═══════════════════════════════════════════════════════════════════════════════


_global_tracer: OTelTracer | None = None


def create_tracer(config: TracingConfig | None = None) -> OTelTracer:
    """Create and set the global tracer."""
    global _global_tracer
    _global_tracer = OTelTracer(config)
    return _global_tracer


def get_tracer() -> OTelTracer:
    """Get the global tracer, creating one if needed."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = OTelTracer()
    return _global_tracer


# ═══════════════════════════════════════════════════════════════════════════════
#  DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════


def span(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            tracer = get_tracer()
            with tracer.trace(span_name, kind=kind, attributes=attributes):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            tracer = get_tracer()
            with tracer.trace(span_name, kind=kind, attributes=attributes):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper

    return decorator
