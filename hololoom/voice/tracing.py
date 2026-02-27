"""
Distributed Tracing for HoloLoom VoiceAgent
==========================================

OpenTelemetry + Jaeger integration for complete request flow visibility.

Features:
- Zero-config defaults (localhost:6831)
- Automatic span creation for voice commands
- TTS synthesis tracing
- Cache operation tracing
- Error recording and propagation
- Context propagation across async boundaries
- Performance metrics and attributes
- Graceful degradation if Jaeger unavailable

Architecture:
- OpenTelemetry SDK for instrumentation
- Jaeger exporter for trace collection
- Async batch export (no blocking)
- Configurable sampling rates
- Service mesh compatibility

Date: November 16, 2025
"""

import asyncio
import functools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.semconv.trace import SpanAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Mock classes for graceful degradation
    class Status:
        def __init__(self, code, message=""):
            pass

    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

# Logging
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class SamplingStrategy(Enum):
    """Trace sampling strategies."""
    ALWAYS_ON = "always_on"      # 100% sampling
    ALWAYS_OFF = "always_off"    # 0% sampling
    PROBABILISTIC = "probabilistic"  # Configurable rate


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""

    # Core settings
    enable_tracing: bool = True
    service_name: str = "hololoom-voice-agent"
    service_version: str = "1.0.0"
    deployment_environment: str = "production"

    # Jaeger settings
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831

    # Sampling
    sampling_strategy: SamplingStrategy = SamplingStrategy.ALWAYS_ON
    sample_rate: float = 1.0  # 100% for development, 0.1 for production

    # Export settings
    batch_export: bool = True
    export_timeout_ms: int = 30000
    max_export_batch_size: int = 512

    # Debug
    console_export: bool = False  # Also export to console
    verbose_spans: bool = True    # Include detailed attributes

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be 0.0-1.0, got {self.sample_rate}")

        if self.jaeger_port < 1 or self.jaeger_port > 65535:
            raise ValueError(f"jaeger_port must be 1-65535, got {self.jaeger_port}")


# ============================================================================
# Tracing Manager
# ============================================================================

class TracingManager:
    """
    Manages OpenTelemetry tracing for VoiceAgent.

    Features:
    - Automatic tracer provider initialization
    - Jaeger exporter configuration
    - Decorators for common operations
    - Context propagation
    - Error recording
    - Performance metrics

    Usage:
        >>> config = TracingConfig(jaeger_host="localhost")
        >>> tracing = TracingManager(config)
        >>>
        >>> @tracing.trace_voice_command()
        >>> async def process_command(transcript: str):
        >>>     return await hololoom.weave(transcript)
    """

    def __init__(self, config: Optional[TracingConfig] = None):
        """
        Initialize tracing manager.

        Args:
            config: Tracing configuration (defaults to TracingConfig())
        """
        self.config = config or TracingConfig()
        self.tracer = None
        self.provider = None
        self.propagator = None
        self._active_spans: Dict[str, Any] = {}

        if self.config.enable_tracing:
            if not OTEL_AVAILABLE:
                logger.warning(
                    "OpenTelemetry not available - tracing disabled. "
                    "Install with: pip install opentelemetry-api opentelemetry-sdk "
                    "opentelemetry-exporter-jaeger-thrift"
                )
                self.config.enable_tracing = False
            else:
                self.propagator = TraceContextTextMapPropagator()
                self._initialize_tracing()

    def _initialize_tracing(self):
        """Initialize OpenTelemetry with Jaeger exporter."""
        try:
            # Create resource
            resource = Resource.create({
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
                "deployment.environment": self.config.deployment_environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
            })

            # Create tracer provider
            self.provider = TracerProvider(resource=resource)

            # Add Jaeger exporter
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.config.jaeger_host,
                    agent_port=self.config.jaeger_port,
                )

                if self.config.batch_export:
                    processor = BatchSpanProcessor(
                        jaeger_exporter,
                        max_export_batch_size=self.config.max_export_batch_size,
                        export_timeout_millis=self.config.export_timeout_ms,
                    )
                else:
                    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
                    processor = SimpleSpanProcessor(jaeger_exporter)

                self.provider.add_span_processor(processor)

                logger.info(
                    "Jaeger exporter initialized",
                    host=self.config.jaeger_host,
                    port=self.config.jaeger_port
                )

            except Exception as e:
                logger.warning(
                    "Failed to initialize Jaeger exporter - using console export",
                    error=str(e)
                )
                # Fallback to console export
                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                self.provider.add_span_processor(console_processor)

            # Optional console export for debugging
            if self.config.console_export:
                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                self.provider.add_span_processor(console_processor)

            # Set global tracer provider
            trace.set_tracer_provider(self.provider)

            # Get tracer
            self.tracer = trace.get_tracer(
                __name__,
                self.config.service_version
            )

            logger.info(
                "Tracing initialized",
                service=self.config.service_name,
                sample_rate=self.config.sample_rate
            )

        except Exception as e:
            logger.error("Failed to initialize tracing", error=str(e))
            self.config.enable_tracing = False

    def shutdown(self, timeout_seconds: int = 30):
        """
        Shutdown tracing and flush remaining spans.

        Args:
            timeout_seconds: Maximum time to wait for flush
        """
        if self.provider:
            try:
                self.provider.shutdown()
                logger.info("Tracing shutdown complete")
            except Exception as e:
                logger.error("Error during tracing shutdown", error=str(e))

    # ========================================================================
    # Decorators
    # ========================================================================

    def trace_voice_command(self, operation_name: Optional[str] = None):
        """
        Decorator for tracing voice command processing.

        Records:
        - Transcript text (first 100 chars)
        - Processing time
        - Confidence score
        - Tool used
        - Intent classification
        - Success/error status

        Args:
            operation_name: Custom operation name (defaults to function name)

        Usage:
            >>> @tracing.trace_voice_command()
            >>> async def process_voice_input(self, transcript: str):
            >>>     return await self.orchestrator.weave(transcript)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.config.enable_tracing or not self.tracer:
                    return await func(*args, **kwargs)

                op_name = operation_name or f"voice_command.{func.__name__}"

                with self.tracer.start_as_current_span(
                    op_name,
                    kind=trace.SpanKind.SERVER
                ) as span:
                    start_time = time.time()

                    try:
                        # Extract transcript from args (assume second arg or kwarg)
                        transcript = None
                        if len(args) > 1 and isinstance(args[1], str):
                            transcript = args[1]
                        elif 'transcript' in kwargs:
                            transcript = kwargs['transcript']

                        if transcript and self.config.verbose_spans:
                            span.set_attribute("voice.transcript", transcript[:100])
                            span.set_attribute("voice.transcript_length", len(transcript))

                        # Execute function
                        result = await func(*args, **kwargs)

                        # Record result attributes
                        processing_time_ms = (time.time() - start_time) * 1000
                        span.set_attribute("processing_time_ms", processing_time_ms)

                        if hasattr(result, 'confidence'):
                            span.set_attribute("confidence", result.confidence)

                        if hasattr(result, 'tool_used'):
                            span.set_attribute("tool_used", result.tool_used)

                        if hasattr(result, 'intent'):
                            span.set_attribute("intent", result.intent)

                        if isinstance(result, str) and self.config.verbose_spans:
                            span.set_attribute("response", result[:200])
                            span.set_attribute("response_length", len(result))

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        span.set_attribute("error", True)
                        span.set_attribute("error_type", type(e).__name__)
                        raise

            return wrapper
        return decorator

    def trace_tts_synthesis(self, operation_name: Optional[str] = None):
        """
        Decorator for tracing TTS synthesis.

        Records:
        - Text to synthesize (first 100 chars)
        - Voice model used
        - Audio size (bytes)
        - Synthesis time
        - Cache hit/miss
        - Success/error status

        Args:
            operation_name: Custom operation name (defaults to "tts.synthesis")

        Usage:
            >>> @tracing.trace_tts_synthesis()
            >>> async def synthesize(self, text: str, voice: str = "nova"):
            >>>     return await openai.audio.speech.create(...)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.config.enable_tracing or not self.tracer:
                    return await func(*args, **kwargs)

                op_name = operation_name or "tts.synthesis"

                with self.tracer.start_as_current_span(
                    op_name,
                    kind=trace.SpanKind.INTERNAL
                ) as span:
                    start_time = time.time()

                    try:
                        # Extract parameters
                        text = None
                        voice = None

                        if len(args) > 1 and isinstance(args[1], str):
                            text = args[1]
                        elif 'text' in kwargs:
                            text = kwargs['text']

                        if len(args) > 2:
                            voice = args[2]
                        elif 'voice' in kwargs:
                            voice = kwargs['voice']

                        if text and self.config.verbose_spans:
                            span.set_attribute("tts.text", text[:100])
                            span.set_attribute("tts.text_length", len(text))

                        if voice:
                            span.set_attribute("tts.voice", voice)

                        # Execute synthesis
                        result = await func(*args, **kwargs)

                        # Record metrics
                        synthesis_time_ms = (time.time() - start_time) * 1000
                        span.set_attribute("tts.synthesis_time_ms", synthesis_time_ms)

                        if isinstance(result, bytes):
                            span.set_attribute("tts.audio_size_bytes", len(result))

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        span.set_attribute("error", True)
                        span.set_attribute("error_type", type(e).__name__)
                        raise

            return wrapper
        return decorator

    def trace_cache_operation(self, operation: str = "lookup"):
        """
        Decorator for tracing cache operations.

        Records:
        - Operation type (lookup/store/invalidate)
        - Cache key (hashed)
        - Hit/miss status
        - Cache size
        - Operation time

        Args:
            operation: Type of cache operation

        Usage:
            >>> @tracing.trace_cache_operation("lookup")
            >>> async def get_cached_audio(self, key: str):
            >>>     return self.cache.get(key)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.config.enable_tracing or not self.tracer:
                    return await func(*args, **kwargs)

                with self.tracer.start_as_current_span(
                    f"cache.{operation}",
                    kind=trace.SpanKind.INTERNAL
                ) as span:
                    start_time = time.time()

                    try:
                        span.set_attribute("cache.operation", operation)

                        # Extract cache key
                        key = None
                        if len(args) > 1:
                            key = str(args[1])
                        elif 'key' in kwargs:
                            key = str(kwargs['key'])

                        if key and self.config.verbose_spans:
                            # Hash key for privacy
                            import hashlib
                            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
                            span.set_attribute("cache.key_hash", key_hash)

                        result = await func(*args, **kwargs)

                        operation_time_ms = (time.time() - start_time) * 1000
                        span.set_attribute("cache.operation_time_ms", operation_time_ms)

                        # Record hit/miss for lookup
                        if operation == "lookup":
                            hit = result is not None
                            span.set_attribute("cache.hit", hit)

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return wrapper
        return decorator

    def trace_hololoom_weave(self, operation_name: Optional[str] = None):
        """
        Decorator for tracing HoloLoom weaving operations.

        Records:
        - Query text
        - Processing mode (BARE/FAST/FUSED)
        - Memory shards used
        - Weaving time
        - Confidence
        - Tool selection

        Args:
            operation_name: Custom operation name

        Usage:
            >>> @tracing.trace_hololoom_weave()
            >>> async def weave(self, query: Query):
            >>>     return await self.orchestrator.weave(query)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.config.enable_tracing or not self.tracer:
                    return await func(*args, **kwargs)

                op_name = operation_name or "hololoom.weave"

                with self.tracer.start_as_current_span(
                    op_name,
                    kind=trace.SpanKind.INTERNAL
                ) as span:
                    start_time = time.time()

                    try:
                        # Extract query
                        query = None
                        if len(args) > 1:
                            query = args[1]
                        elif 'query' in kwargs:
                            query = kwargs['query']

                        if query and self.config.verbose_spans:
                            query_text = getattr(query, 'text', str(query))
                            span.set_attribute("hololoom.query", str(query_text)[:100])

                        result = await func(*args, **kwargs)

                        weave_time_ms = (time.time() - start_time) * 1000
                        span.set_attribute("hololoom.weave_time_ms", weave_time_ms)

                        if hasattr(result, 'confidence'):
                            span.set_attribute("hololoom.confidence", result.confidence)

                        if hasattr(result, 'metadata'):
                            metadata = result.metadata
                            if isinstance(metadata, dict):
                                if 'tool_used' in metadata:
                                    span.set_attribute("hololoom.tool_used", metadata['tool_used'])
                                if 'mode' in metadata:
                                    span.set_attribute("hololoom.mode", metadata['mode'])

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return wrapper
        return decorator

    # ========================================================================
    # Manual Span Management
    # ========================================================================

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for manual span creation.

        Args:
            name: Span name
            kind: Span kind (SERVER, CLIENT, INTERNAL, etc.)
            attributes: Initial span attributes

        Usage:
            >>> async with tracing.span("custom_operation") as span:
            >>>     span.set_attribute("custom.key", "value")
            >>>     result = await do_work()
        """
        if not self.config.enable_tracing or not self.tracer:
            yield None
            return

        span_kind = kind or trace.SpanKind.INTERNAL

        with self.tracer.start_as_current_span(name, kind=span_kind) as span:
            try:
                # Set initial attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                yield span

                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def get_current_span(self) -> Optional[Any]:
        """
        Get the current active span.

        Returns:
            Current span or None if no active span
        """
        if not self.config.enable_tracing or not OTEL_AVAILABLE:
            return None

        return trace.get_current_span()

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Add an event to the current span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        if not self.config.enable_tracing:
            return

        span = self.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes=attributes or {})

    def set_attribute(self, key: str, value: Any):
        """
        Set attribute on current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.config.enable_tracing:
            return

        span = self.get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)


# ============================================================================
# Global Instance
# ============================================================================

# Global tracing manager (initialized on first use)
_global_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager(config: Optional[TracingConfig] = None) -> TracingManager:
    """
    Get or create global tracing manager.

    Args:
        config: Optional config (only used on first call)

    Returns:
        Global tracing manager instance
    """
    global _global_tracing_manager

    if _global_tracing_manager is None:
        _global_tracing_manager = TracingManager(config)

    return _global_tracing_manager


def shutdown_tracing(timeout_seconds: int = 30):
    """
    Shutdown global tracing manager.

    Args:
        timeout_seconds: Timeout for flush
    """
    global _global_tracing_manager

    if _global_tracing_manager:
        _global_tracing_manager.shutdown(timeout_seconds)
        _global_tracing_manager = None
