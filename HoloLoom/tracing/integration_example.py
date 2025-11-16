#!/usr/bin/env python3
"""
HoloLoom Dashboard Server with Distributed Tracing
===================================================

Example integration of OpenTelemetry tracing into the dashboard server.

This shows how to:
1. Initialize tracing with configuration
2. Auto-instrument FastAPI
3. Instrument HoloLoom components
4. Add custom spans
5. View traces in Jaeger

Usage:
    # 1. Start Jaeger
    docker-compose -f docker-compose.tracing.yml up jaeger -d

    # 2. Run this server
    TRACING_ENABLED=true python HoloLoom/tracing/integration_example.py

    # 3. Make requests
    curl http://localhost:8000/query?text=test

    # 4. View traces
    Open http://localhost:16686 (Jaeger UI)

Created: 2025-11-16
"""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse

# Import HoloLoom tracing
from HoloLoom.tracing import (
    TracingConfig,
    init_tracing,
    instrument_app,
    instrument_hololoom,
    get_tracer,
    create_span,
    shutdown_tracing,
    config_from_env,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HoloLoom Dashboard with Tracing",
    description="Example integration of OpenTelemetry distributed tracing",
    version="1.0.0"
)

# ============================================================================
# Tracing Initialization
# ============================================================================

def initialize_tracing():
    """Initialize OpenTelemetry tracing."""

    # Option 1: Load from environment variables
    config = config_from_env()

    # Option 2: Programmatic configuration (overrides environment)
    # config = TracingConfig(
    #     service_name="hololoom-dashboard",
    #     environment="development",
    #     exporter="jaeger",  # or "zipkin", "console", "otlp"
    #     sample_rate=1.0,  # 100% sampling for development
    #     jaeger_agent_host="localhost",
    #     jaeger_agent_port=6831,
    # )

    # Initialize tracing
    tracer = init_tracing(config)
    logger.info(f"Tracing initialized: exporter={config.exporter}, sampling={config.sample_rate}")

    # Auto-instrument FastAPI (creates spans for all endpoints)
    instrument_app(app)
    logger.info("FastAPI auto-instrumented")

    # Instrument HoloLoom components
    # This adds tracing to WeavingOrchestrator, MemoryManager, etc.
    instrument_hololoom(
        orchestrator=True,
        memory=True,
        analytics=True,
        agentic=True,
        recursive=True,
    )
    logger.info("HoloLoom components instrumented")

    return tracer


# Initialize on module load
tracer = initialize_tracing()

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Dashboard server starting...")
    logger.info("Tracing enabled - view traces at http://localhost:16686")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Dashboard server shutting down...")

    # Flush remaining spans
    shutdown_tracing()
    logger.info("Tracing shutdown complete")


# ============================================================================
# API Endpoints (Auto-traced by FastAPI instrumentation)
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - automatically traced."""
    return {
        "service": "HoloLoom Dashboard",
        "tracing": "enabled",
        "jaeger_ui": "http://localhost:16686",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - automatically traced."""
    return {
        "status": "healthy",
        "tracing": "active"
    }


@app.get("/query")
async def query_endpoint(
    text: str = Query(..., description="Query text"),
    mode: str = Query("fast", description="Execution mode (bare/fast/fused)")
):
    """
    Query endpoint with custom tracing.

    This endpoint is auto-traced by FastAPI instrumentation,
    but we add custom spans for fine-grained visibility.
    """
    logger.info(f"Processing query: {text}")

    # Add custom span for query processing
    with create_span(
        "process_query",
        attributes={
            "query.text": text[:100],
            "query.mode": mode,
        }
    ):
        # Simulate processing steps
        with create_span("step_1_extract_features"):
            await asyncio.sleep(0.05)  # Simulate feature extraction
            features = {"count": 10}

        with create_span("step_2_retrieve_context"):
            await asyncio.sleep(0.03)  # Simulate retrieval
            context = {"shards": 5}

        with create_span("step_3_make_decision"):
            await asyncio.sleep(0.02)  # Simulate decision
            decision = {"tool": "answer", "confidence": 0.85}

    return {
        "query": text,
        "mode": mode,
        "features": features,
        "context": context,
        "decision": decision,
        "tracing": "View this trace in Jaeger UI"
    }


@app.get("/simulate-error")
async def simulate_error():
    """
    Simulate an error to see error tracking in traces.

    This endpoint intentionally raises an exception to demonstrate
    how errors are captured in traces.
    """
    with create_span("error_operation"):
        # This will be recorded as an error in the trace
        raise ValueError("This is a simulated error for testing tracing")


@app.get("/slow-operation")
async def slow_operation(
    duration: float = Query(1.0, description="Duration in seconds")
):
    """
    Simulate a slow operation to test performance analysis.

    This endpoint creates a long-running span to demonstrate
    bottleneck detection in traces.
    """
    with create_span(
        "slow_operation",
        attributes={"duration_seconds": duration}
    ):
        # Simulate slow operation
        await asyncio.sleep(duration)

        # Add nested slow operations
        with create_span("slow_phase_1"):
            await asyncio.sleep(duration * 0.4)

        with create_span("slow_phase_2"):
            await asyncio.sleep(duration * 0.6)

    return {
        "status": "completed",
        "duration": duration,
        "message": "Check Jaeger for bottleneck analysis"
    }


# ============================================================================
# Tracing Utilities
# ============================================================================

@app.get("/trace-info")
async def trace_info(request: Request):
    """
    Get information about the current trace.

    Returns trace ID, span ID, and sampling status.
    """
    from HoloLoom.tracing import get_trace_id, get_span_id, is_sampled

    return {
        "trace_id": get_trace_id(),
        "span_id": get_span_id(),
        "sampled": is_sampled(),
        "request_id": request.headers.get("x-request-id"),
        "jaeger_link": f"http://localhost:16686/trace/{get_trace_id()}" if get_trace_id() else None
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("HoloLoom Dashboard with Distributed Tracing")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Quick Start:")
    logger.info("1. Start Jaeger: docker-compose -f docker-compose.tracing.yml up jaeger -d")
    logger.info("2. Access Jaeger UI: http://localhost:16686")
    logger.info("3. Make requests to this server")
    logger.info("4. View traces in Jaeger UI")
    logger.info("")
    logger.info("Example requests:")
    logger.info("  curl http://localhost:8000/query?text=test")
    logger.info("  curl http://localhost:8000/slow-operation?duration=2")
    logger.info("  curl http://localhost:8000/trace-info")
    logger.info("")
    logger.info("=" * 70)

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
