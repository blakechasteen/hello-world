"""
Telemetry Protocols - Unified interfaces for tracing and metrics.

Provides protocol definitions for OpenTelemetry tracing and Prometheus metrics
collection across all HoloLoom systems.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, runtime_checkable


# ═══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class SpanKind(Enum):
    """OpenTelemetry span kinds."""

    INTERNAL = auto()  # Internal operation
    SERVER = auto()  # Server-side of RPC
    CLIENT = auto()  # Client-side of RPC
    PRODUCER = auto()  # Message producer
    CONSUMER = auto()  # Message consumer


class SpanStatus(Enum):
    """Span status codes."""

    UNSET = auto()  # Default, not explicitly set
    OK = auto()  # Operation completed successfully
    ERROR = auto()  # Operation failed


class MetricType(Enum):
    """Prometheus metric types."""

    COUNTER = auto()  # Monotonically increasing
    GAUGE = auto()  # Can go up and down
    HISTOGRAM = auto()  # Bucketed observations
    SUMMARY = auto()  # Quantile observations


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SpanContext:
    """Immutable context for trace propagation."""

    trace_id: str
    span_id: str
    trace_flags: int = 0
    trace_state: str = ""
    is_remote: bool = False


@dataclass(slots=True)
class SpanData:
    """Data collected for a single span."""

    name: str
    context: SpanContext
    parent_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[SpanContext] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to this span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on this span."""
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self) -> None:
        """Mark span as ended."""
        self.end_time = datetime.utcnow()


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """Single metric observation."""

    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    unit: str = ""


@dataclass(slots=True)
class HistogramBuckets:
    """Histogram bucket configuration."""

    boundaries: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])
    counts: List[int] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        """Observe a value and update buckets."""
        self.sum_value += value
        self.count += 1
        if not self.counts:
            self.counts = [0] * (len(self.boundaries) + 1)
        for i, boundary in enumerate(self.boundaries):
            if value <= boundary:
                self.counts[i] += 1
                return
        self.counts[-1] += 1  # +Inf bucket


# ═══════════════════════════════════════════════════════════════════════════════
#  PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class TracingProtocol(Protocol):
    """Protocol for distributed tracing providers."""

    @abstractmethod
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent: Optional[SpanContext] = None,
    ) -> SpanData:
        """Start a new span."""
        ...

    @abstractmethod
    def end_span(self, span: SpanData) -> None:
        """End a span and export it."""
        ...

    @abstractmethod
    def get_current_span(self) -> Optional[SpanData]:
        """Get the currently active span."""
        ...

    @abstractmethod
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into a carrier (e.g., HTTP headers)."""
        ...

    @abstractmethod
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from a carrier."""
        ...


@runtime_checkable
class MetricsProtocol(Protocol):
    """Protocol for metrics collection providers."""

    @abstractmethod
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Increment a counter metric."""
        ...

    @abstractmethod
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        """Set a gauge metric value."""
        ...

    @abstractmethod
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """Observe a histogram metric value."""
        ...

    @abstractmethod
    def summary(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        quantiles: Optional[List[float]] = None,
        description: str = "",
    ) -> None:
        """Observe a summary metric value."""
        ...

    @abstractmethod
    def export(self) -> str:
        """Export metrics in Prometheus format."""
        ...


@runtime_checkable
class SpanProcessorProtocol(Protocol):
    """Protocol for span processors (exporters)."""

    @abstractmethod
    def on_start(self, span: SpanData) -> None:
        """Called when a span starts."""
        ...

    @abstractmethod
    def on_end(self, span: SpanData) -> None:
        """Called when a span ends."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the processor."""
        ...

    @abstractmethod
    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Force flush any pending spans."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════════


T = TypeVar("T")
SpanDecorator = Callable[[Callable[..., T]], Callable[..., T]]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_trace_id() -> str:
    """Generate a random trace ID (32 hex chars)."""
    import secrets
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generate a random span ID (16 hex chars)."""
    import secrets
    return secrets.token_hex(8)


def create_span_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> SpanContext:
    """Create a new span context with optional IDs."""
    return SpanContext(
        trace_id=trace_id or generate_trace_id(),
        span_id=span_id or generate_span_id(),
    )
