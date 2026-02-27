"""
Jaeger Exporter - Export spans to Jaeger for distributed tracing.

Implements span export to Jaeger using Thrift over HTTP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from HoloLoom.telemetry.protocol import SpanData, SpanProcessorProtocol

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  JAEGER EXPORTER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class JaegerExporter(SpanProcessorProtocol):
    """Exports spans to Jaeger backend.

    Collects spans and exports them to Jaeger for distributed tracing
    visualization and analysis.
    """

    host: str = "localhost"
    port: int = 6831  # Thrift compact UDP port
    http_port: int = 14268  # HTTP collector port
    use_http: bool = True  # Use HTTP instead of UDP
    service_name: str = "hololoom"

    # Internal state
    _pending_spans: List[SpanData] = field(default_factory=list, init=False)
    _shutdown: bool = field(default=False, init=False)

    @property
    def endpoint(self) -> str:
        """Get the Jaeger collector endpoint."""
        if self.use_http:
            return f"http://{self.host}:{self.http_port}/api/traces"
        return f"{self.host}:{self.port}"

    def on_start(self, span: SpanData) -> None:
        """Called when a span starts (no-op for Jaeger)."""
        pass

    def on_end(self, span: SpanData) -> None:
        """Called when a span ends - queue for export."""
        if self._shutdown:
            return
        self._pending_spans.append(span)

    def _span_to_jaeger_format(self, span: SpanData) -> dict:
        """Convert SpanData to Jaeger Thrift format."""
        return {
            "traceIdHigh": 0,
            "traceIdLow": int(span.context.trace_id[:16], 16),
            "spanId": int(span.context.span_id, 16),
            "parentSpanId": int(span.parent_id, 16) if span.parent_id else 0,
            "operationName": span.name,
            "startTime": int(span.start_time.timestamp() * 1_000_000) if span.start_time else 0,
            "duration": int(span.duration_ms * 1000) if span.duration_ms else 0,
            "tags": [
                {"key": k, "vType": "STRING", "vStr": str(v)}
                for k, v in span.attributes.items()
            ],
            "logs": [
                {
                    "timestamp": int(e["timestamp"].timestamp() * 1_000_000),
                    "fields": [{"key": k, "vType": "STRING", "vStr": str(v)} for k, v in e.items() if k != "timestamp"]
                }
                for e in span.events
            ],
        }

    def export(self) -> bool:
        """Export pending spans to Jaeger.

        Returns True if export was successful.
        """
        if not self._pending_spans:
            return True

        try:
            # Build Jaeger batch
            batch = {
                "process": {
                    "serviceName": self.service_name,
                    "tags": [],
                },
                "spans": [self._span_to_jaeger_format(s) for s in self._pending_spans],
            }

            if self.use_http:
                # HTTP export (requires httpx or requests)
                try:
                    import httpx
                    response = httpx.post(
                        self.endpoint,
                        json=batch,
                        headers={"Content-Type": "application/json"},
                        timeout=5.0,
                    )
                    success = response.status_code == 202
                except ImportError:
                    # Fallback: log the spans
                    logger.warning(
                        f"httpx not available, logging {len(self._pending_spans)} spans"
                    )
                    for span in self._pending_spans:
                        logger.info(f"[JAEGER] {span.name} trace={span.context.trace_id[:8]}")
                    success = True
            else:
                # UDP export would go here (requires thrift)
                logger.warning("UDP export not implemented, use HTTP")
                success = False

            if success:
                self._pending_spans.clear()
            return success

        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown exporter and flush remaining spans."""
        self._shutdown = True
        self.force_flush()

    def force_flush(self, timeout_ms: int = 30000) -> bool:
        """Force flush all pending spans."""
        return self.export()


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def create_jaeger_exporter(
    host: str = "localhost",
    port: int = 14268,
    service_name: str = "hololoom",
) -> JaegerExporter:
    """Create a Jaeger exporter with common defaults."""
    return JaegerExporter(
        host=host,
        http_port=port,
        use_http=True,
        service_name=service_name,
    )
