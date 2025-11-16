#!/usr/bin/env python3
"""
OpenTelemetry Integration for HoloLoom
========================================

Core tracing setup with support for multiple exporters and graceful degradation.

Features:
- Jaeger exporter (production)
- Zipkin exporter (alternative)
- Console exporter (development)
- OTLP exporter (generic)
- Configurable sampling
- Resource attributes
- Graceful degradation

Usage:
    from HoloLoom.tracing import init_tracing, TracingConfig

    config = TracingConfig(
        service_name="hololoom-dashboard",
        environment="production",
        exporter="jaeger",
        sample_rate=0.5  # 50% sampling
    )

    tracer = init_tracing(config)

    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # ... do work ...

Created: 2025-11-16
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import (
        TraceIdRatioBased,
        ParentBasedTraceIdRatio,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.semconv.trace import SpanAttributes

    # Try Jaeger exporter
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        JAEGER_AVAILABLE = True
    except ImportError:
        JAEGER_AVAILABLE = False

    # Try Zipkin exporter
    try:
        from opentelemetry.exporter.zipkin.json import ZipkinExporter
        ZIPKIN_AVAILABLE = True
    except ImportError:
        ZIPKIN_AVAILABLE = False

    # Try OTLP exporter (generic)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    trace = None
    TracerProvider = None
    JAEGER_AVAILABLE = False
    ZIPKIN_AVAILABLE = False
    OTLP_AVAILABLE = False

import contextlib

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    # Service identification
    service_name: str = "hololoom-dashboard"
    service_version: str = "1.0.0"
    environment: str = "development"

    # Exporter selection
    exporter: str = "console"  # console, jaeger, zipkin, otlp

    # Jaeger config
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    jaeger_endpoint: Optional[str] = None  # For Jaeger Collector HTTP

    # Zipkin config
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"

    # OTLP config
    otlp_endpoint: str = "http://localhost:4317"

    # Sampling
    sample_rate: float = 1.0  # 0.0-1.0, 1.0 = trace everything

    # Performance
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_ms: int = 30000
    schedule_delay_ms: int = 5000

    # Additional resource attributes
    resource_attributes: Dict[str, Any] = field(default_factory=dict)

    # Enable/disable
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.sample_rate < 0.0 or self.sample_rate > 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")

        if self.exporter not in ["console", "jaeger", "zipkin", "otlp"]:
            raise ValueError(
                f"Unknown exporter: {self.exporter}. "
                f"Must be one of: console, jaeger, zipkin, otlp"
            )


# ============================================================================
# No-Op Tracer (Fallback)
# ============================================================================

class NoOpSpan:
    """No-op span when tracing is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: Optional[Any] = None,
        attributes: Optional[Dict] = None,
        links: Optional[list] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> NoOpSpan:
        """Return a no-op context manager."""
        return NoOpSpan()

    def start_span(
        self,
        name: str,
        context: Optional[Any] = None,
        kind: Optional[Any] = None,
        attributes: Optional[Dict] = None,
        links: Optional[list] = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> NoOpSpan:
        """Return a no-op span."""
        return NoOpSpan()


# ============================================================================
# Core Functions
# ============================================================================

_global_tracer: Optional[Any] = None


def init_tracing(config: TracingConfig) -> Any:
    """
    Initialize OpenTelemetry tracing with the specified configuration.

    Args:
        config: Tracing configuration

    Returns:
        Configured tracer (or NoOpTracer if unavailable)

    Example:
        >>> config = TracingConfig(exporter="jaeger", sample_rate=0.5)
        >>> tracer = init_tracing(config)
        >>> with tracer.start_as_current_span("operation"):
        ...     # do work
    """
    global _global_tracer

    if not TRACING_AVAILABLE or not config.enabled:
        logger.warning(
            "OpenTelemetry not available or tracing disabled. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-jaeger opentelemetry-exporter-zipkin-json"
        )
        _global_tracer = NoOpTracer()
        return _global_tracer

    # Create resource with service identification
    resource_attrs = {
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "deployment.environment": config.environment,
    }
    resource_attrs.update(config.resource_attributes)
    resource = Resource.create(resource_attrs)

    # Configure sampler
    sampler = ParentBasedTraceIdRatio(config.sample_rate)

    # Create provider
    provider = TracerProvider(resource=resource, sampler=sampler)

    # Configure exporter
    exporter = _create_exporter(config)
    if exporter is None:
        logger.warning(f"Failed to create {config.exporter} exporter, using console")
        exporter = ConsoleSpanExporter()

    # Create batch processor
    processor = BatchSpanProcessor(
        exporter,
        max_queue_size=config.max_queue_size,
        max_export_batch_size=config.max_export_batch_size,
        export_timeout_millis=config.export_timeout_ms,
        schedule_delay_millis=config.schedule_delay_ms,
    )

    provider.add_span_processor(processor)

    # Set as global provider
    trace.set_tracer_provider(provider)

    # Get tracer
    _global_tracer = trace.get_tracer(__name__)

    logger.info(
        f"OpenTelemetry tracing initialized: "
        f"exporter={config.exporter}, "
        f"sample_rate={config.sample_rate}, "
        f"service={config.service_name}"
    )

    return _global_tracer


def _create_exporter(config: TracingConfig) -> Optional[SpanExporter]:
    """Create exporter based on configuration."""

    if config.exporter == "console":
        return ConsoleSpanExporter()

    elif config.exporter == "jaeger":
        if not JAEGER_AVAILABLE:
            logger.warning(
                "Jaeger exporter not available. "
                "Install with: pip install opentelemetry-exporter-jaeger"
            )
            return None

        # Use Thrift over UDP (agent)
        if config.jaeger_endpoint is None:
            return JaegerExporter(
                agent_host_name=config.jaeger_agent_host,
                agent_port=config.jaeger_agent_port,
            )
        # Use HTTP (collector)
        else:
            return JaegerExporter(
                collector_endpoint=config.jaeger_endpoint,
            )

    elif config.exporter == "zipkin":
        if not ZIPKIN_AVAILABLE:
            logger.warning(
                "Zipkin exporter not available. "
                "Install with: pip install opentelemetry-exporter-zipkin-json"
            )
            return None

        return ZipkinExporter(endpoint=config.zipkin_endpoint)

    elif config.exporter == "otlp":
        if not OTLP_AVAILABLE:
            logger.warning(
                "OTLP exporter not available. "
                "Install with: pip install opentelemetry-exporter-otlp"
            )
            return None

        return OTLPSpanExporter(endpoint=config.otlp_endpoint)

    return None


def get_tracer(name: Optional[str] = None) -> Any:
    """
    Get the global tracer or create a named tracer.

    Args:
        name: Optional tracer name (uses __name__ if None)

    Returns:
        Tracer instance (or NoOpTracer if unavailable)
    """
    global _global_tracer

    if not TRACING_AVAILABLE:
        return NoOpTracer()

    if _global_tracer is None:
        # Initialize with default config
        config = TracingConfig()
        _global_tracer = init_tracing(config)

    if name is not None and TRACING_AVAILABLE:
        return trace.get_tracer(name)

    return _global_tracer


def shutdown_tracing():
    """Gracefully shutdown tracing (flush pending spans)."""
    if TRACING_AVAILABLE and trace.get_tracer_provider():
        try:
            trace.get_tracer_provider().shutdown()
            logger.info("Tracing shutdown complete")
        except Exception as e:
            logger.warning(f"Error during tracing shutdown: {e}")


# ============================================================================
# Environment-Based Configuration
# ============================================================================

def config_from_env() -> TracingConfig:
    """
    Create tracing configuration from environment variables.

    Environment Variables:
        TRACING_ENABLED: Enable/disable tracing (default: false)
        TRACING_EXPORTER: Exporter type (console|jaeger|zipkin|otlp)
        TRACING_SERVICE_NAME: Service name (default: hololoom-dashboard)
        TRACING_ENVIRONMENT: Environment (dev|staging|prod)
        TRACING_SAMPLE_RATE: Sample rate 0.0-1.0 (default: 1.0)
        JAEGER_AGENT_HOST: Jaeger agent host (default: localhost)
        JAEGER_AGENT_PORT: Jaeger agent port (default: 6831)
        ZIPKIN_ENDPOINT: Zipkin endpoint URL
        OTLP_ENDPOINT: OTLP endpoint URL

    Returns:
        TracingConfig instance
    """
    return TracingConfig(
        enabled=os.getenv("TRACING_ENABLED", "false").lower() == "true",
        exporter=os.getenv("TRACING_EXPORTER", "console"),
        service_name=os.getenv("TRACING_SERVICE_NAME", "hololoom-dashboard"),
        environment=os.getenv("TRACING_ENVIRONMENT", "development"),
        sample_rate=float(os.getenv("TRACING_SAMPLE_RATE", "1.0")),
        jaeger_agent_host=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        jaeger_agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
        zipkin_endpoint=os.getenv(
            "ZIPKIN_ENDPOINT",
            "http://localhost:9411/api/v2/spans"
        ),
        otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
    )


# ============================================================================
# Utilities
# ============================================================================

def create_span_attributes(**kwargs) -> Dict[str, Any]:
    """
    Create span attributes with proper types.

    OpenTelemetry only supports: str, bool, int, float, list of these.
    This helper ensures proper typing.
    """
    attrs = {}

    for key, value in kwargs.items():
        if value is None:
            continue

        # Convert complex types to strings
        if isinstance(value, (dict, list, tuple)) and not isinstance(value, str):
            attrs[key] = str(value)
        elif isinstance(value, (str, bool, int, float)):
            attrs[key] = value
        else:
            attrs[key] = str(value)

    return attrs


# ============================================================================
# Span Helpers
# ============================================================================

class SpanContext:
    """Context manager for creating spans with automatic error handling."""

    def __init__(
        self,
        tracer: Any,
        name: str,
        attributes: Optional[Dict] = None,
        kind: Optional[Any] = None,
    ):
        self.tracer = tracer
        self.name = name
        self.attributes = attributes or {}
        self.kind = kind
        self.span = None

    def __enter__(self):
        if TRACING_AVAILABLE:
            self.span = self.tracer.start_as_current_span(
                self.name,
                kind=self.kind,
                attributes=self.attributes,
            ).__enter__()
        else:
            self.span = NoOpSpan()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span and TRACING_AVAILABLE:
            if exc_type is not None:
                # Record exception
                self.span.record_exception(exc_val)
                self.span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
            else:
                self.span.set_status(trace.Status(trace.StatusCode.OK))

            self.span.__exit__(exc_type, exc_val, exc_tb)

        return False  # Don't suppress exceptions
