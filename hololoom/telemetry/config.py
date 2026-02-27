"""
Telemetry Configuration - Settings for tracing and metrics.

Supports environment variable overrides for production deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class ExporterType(Enum):
    """Supported telemetry exporters."""

    CONSOLE = auto()  # Print to stdout (development)
    JAEGER = auto()  # Jaeger tracing backend
    OTLP = auto()  # OpenTelemetry Protocol
    PROMETHEUS = auto()  # Prometheus metrics endpoint


class SamplingStrategy(Enum):
    """Trace sampling strategies."""

    ALWAYS_ON = auto()  # Sample all traces
    ALWAYS_OFF = auto()  # Sample no traces
    RATIO = auto()  # Sample based on ratio
    PARENT_BASED = auto()  # Follow parent decision


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class TracingConfig:
    """Configuration for distributed tracing."""

    enabled: bool = True
    service_name: str = "hololoom"
    service_version: str = "1.0.0"
    exporter: ExporterType = ExporterType.CONSOLE
    sampling_strategy: SamplingStrategy = SamplingStrategy.ALWAYS_ON
    sampling_ratio: float = 1.0  # For RATIO strategy
    max_spans_per_trace: int = 1000
    max_attributes_per_span: int = 128
    max_events_per_span: int = 128

    # Jaeger-specific
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831

    # OTLP-specific
    otlp_endpoint: str = "http://localhost:4317"
    otlp_headers: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> TracingConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("HOLOLOOM_TRACING_ENABLED", "true").lower() == "true",
            service_name=os.getenv("HOLOLOOM_SERVICE_NAME", "hololoom"),
            service_version=os.getenv("HOLOLOOM_SERVICE_VERSION", "1.0.0"),
            exporter=ExporterType[os.getenv("HOLOLOOM_TRACE_EXPORTER", "CONSOLE").upper()],
            sampling_strategy=SamplingStrategy[
                os.getenv("HOLOLOOM_SAMPLING_STRATEGY", "ALWAYS_ON").upper()
            ],
            sampling_ratio=float(os.getenv("HOLOLOOM_SAMPLING_RATIO", "1.0")),
            jaeger_endpoint=os.getenv(
                "HOLOLOOM_JAEGER_ENDPOINT", "http://localhost:14268/api/traces"
            ),
            jaeger_agent_host=os.getenv("HOLOLOOM_JAEGER_AGENT_HOST", "localhost"),
            jaeger_agent_port=int(os.getenv("HOLOLOOM_JAEGER_AGENT_PORT", "6831")),
            otlp_endpoint=os.getenv("HOLOLOOM_OTLP_ENDPOINT", "http://localhost:4317"),
        )


@dataclass(slots=True)
class MetricsConfig:
    """Configuration for Prometheus metrics."""

    enabled: bool = True
    prefix: str = "hololoom"
    http_port: int = 9090
    http_path: str = "/metrics"

    # Histogram buckets for common metrics
    latency_buckets: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

    # Default labels added to all metrics
    default_labels: dict = field(default_factory=dict)

    # Push gateway (optional)
    push_gateway_url: Optional[str] = None
    push_interval_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> MetricsConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("HOLOLOOM_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("HOLOLOOM_METRICS_PREFIX", "hololoom"),
            http_port=int(os.getenv("HOLOLOOM_METRICS_PORT", "9090")),
            http_path=os.getenv("HOLOLOOM_METRICS_PATH", "/metrics"),
            push_gateway_url=os.getenv("HOLOLOOM_PUSHGATEWAY_URL"),
            push_interval_seconds=float(
                os.getenv("HOLOLOOM_PUSH_INTERVAL", "15.0")
            ),
        )


@dataclass(slots=True)
class TelemetryConfig:
    """Combined telemetry configuration."""

    tracing: TracingConfig = field(default_factory=TracingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Global settings
    enabled: bool = True
    environment: str = "development"  # development, staging, production

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        """Create config from environment variables."""
        return cls(
            tracing=TracingConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            enabled=os.getenv("HOLOLOOM_TELEMETRY_ENABLED", "true").lower() == "true",
            environment=os.getenv("HOLOLOOM_ENVIRONMENT", "development"),
        )

    @classmethod
    def development(cls) -> TelemetryConfig:
        """Development configuration (console output)."""
        return cls(
            tracing=TracingConfig(
                enabled=True,
                exporter=ExporterType.CONSOLE,
                sampling_strategy=SamplingStrategy.ALWAYS_ON,
            ),
            metrics=MetricsConfig(
                enabled=True,
                http_port=9090,
            ),
            environment="development",
        )

    @classmethod
    def production(cls) -> TelemetryConfig:
        """Production configuration (Jaeger + Prometheus)."""
        return cls(
            tracing=TracingConfig(
                enabled=True,
                exporter=ExporterType.JAEGER,
                sampling_strategy=SamplingStrategy.RATIO,
                sampling_ratio=0.1,  # Sample 10% in production
            ),
            metrics=MetricsConfig(
                enabled=True,
                http_port=9090,
            ),
            environment="production",
        )

    @classmethod
    def disabled(cls) -> TelemetryConfig:
        """Disabled configuration (for testing)."""
        return cls(
            tracing=TracingConfig(enabled=False),
            metrics=MetricsConfig(enabled=False),
            enabled=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT VARIABLE REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

ENV_VARS = """
Environment Variables:

General:
  HOLOLOOM_TELEMETRY_ENABLED  - Enable/disable all telemetry (default: true)
  HOLOLOOM_ENVIRONMENT        - Environment name (default: development)

Tracing:
  HOLOLOOM_TRACING_ENABLED    - Enable/disable tracing (default: true)
  HOLOLOOM_SERVICE_NAME       - Service name for spans (default: hololoom)
  HOLOLOOM_SERVICE_VERSION    - Service version (default: 1.0.0)
  HOLOLOOM_TRACE_EXPORTER     - Exporter type: CONSOLE, JAEGER, OTLP (default: CONSOLE)
  HOLOLOOM_SAMPLING_STRATEGY  - Sampling: ALWAYS_ON, ALWAYS_OFF, RATIO (default: ALWAYS_ON)
  HOLOLOOM_SAMPLING_RATIO     - Sampling ratio for RATIO strategy (default: 1.0)
  HOLOLOOM_JAEGER_ENDPOINT    - Jaeger collector endpoint
  HOLOLOOM_JAEGER_AGENT_HOST  - Jaeger agent host (default: localhost)
  HOLOLOOM_JAEGER_AGENT_PORT  - Jaeger agent UDP port (default: 6831)
  HOLOLOOM_OTLP_ENDPOINT      - OTLP collector endpoint (default: http://localhost:4317)

Metrics:
  HOLOLOOM_METRICS_ENABLED    - Enable/disable metrics (default: true)
  HOLOLOOM_METRICS_PREFIX     - Metric name prefix (default: hololoom)
  HOLOLOOM_METRICS_PORT       - HTTP port for /metrics (default: 9090)
  HOLOLOOM_METRICS_PATH       - HTTP path for metrics (default: /metrics)
  HOLOLOOM_PUSHGATEWAY_URL    - Prometheus Pushgateway URL (optional)
  HOLOLOOM_PUSH_INTERVAL      - Push interval in seconds (default: 15.0)
"""
