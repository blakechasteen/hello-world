"""
Health Router
=============

Health check, stats, safety stats, and audit trail endpoints.
Extracted from agentic_api.py as part of W2 (Monolithic Files) SWOT remediation.

Endpoints:
- GET /health - Server health check
- GET /stats - Server statistics
- GET /safety-stats - Safety guardrails statistics
- GET /audit-trail - Recent audit trail entries
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health & Stats"])


def get_server_state(request: Request):
    """
    Dependency to get server state from the app.

    The state is attached to the app during startup in agentic_api.py.
    """
    return request.app.state.server_state


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Used by VS Code extension to verify server is running.
    """
    return {
        "status": "ok",
        "service": "HoloLoom Agentic API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/stats")
async def get_stats(state=Depends(get_server_state)):
    """
    Get server statistics.

    Returns:
        Comprehensive statistics about server performance:
        - Uptime and query counts
        - Success/failure rates
        - Latency metrics (avg, p95)
        - Queries by mode
        - Error breakdown
        - Memory and orchestrator status
    """
    base_stats = {
        "orchestrator_ready": state.orchestrator is not None,
        "memory_shards": len(state.shards),
        "rate_limiter_enabled": state.rate_limiter is not None,
    }

    # Add comprehensive stats if available
    if state.stats:
        base_stats.update(state.stats.get_stats_dict())

    # Add audit trail info
    if state.audit_trail:
        base_stats["audit_trail_entries"] = len(state.audit_trail.logs)

    return base_stats


@router.get("/safety-stats")
async def get_safety_stats(state=Depends(get_server_state)):
    """
    Get safety guardrails statistics.

    Useful for monitoring and investor demo.

    Returns:
        Dict with safety metrics
    """
    if not state.safety_guardrails:
        return {
            "enabled": False,
            "message": "Safety guardrails not initialized"
        }

    # SafetyGuardrails stats with proper testing_mode reflection
    testing_mode = getattr(state.safety_guardrails, 'testing_mode', False)
    env_testing_mode = os.environ.get("HOLOLOOM_TESTING_MODE", "").lower() == "true"

    if testing_mode:
        mode_message = "TESTING MODE - approval requirements bypassed (development only)"
    else:
        mode_message = "PRODUCTION MODE - full safety gating active"

    return {
        "enabled": True,
        "testing_mode": testing_mode,
        "env_testing_mode_set": env_testing_mode,
        "adversarial_detection_enabled": state.safety_guardrails.adversarial_detector is not None,
        "deception_detection_enabled": state.deception_detector is not None,
        "decisions_logged": len(state.safety_guardrails.decisions) if hasattr(state.safety_guardrails, 'decisions') else 0,
        "message": mode_message
    }


@router.get("/audit-trail")
async def get_audit_trail(limit: int = 100, state=Depends(get_server_state)):
    """
    Get recent audit trail entries.

    Args:
        limit: Maximum number of entries to return

    Returns:
        Recent audit trail entries
    """
    if not state.audit_trail:
        return {"entries": []}

    # Get recent entries
    entries = state.audit_trail.logs[-limit:]

    return {
        "total": len(state.audit_trail.logs),
        "entries": [
            {
                "decision_id": log.decision_id,
                "decision_type": log.decision_type.value,
                "outcome": log.outcome.value,
                "reason": log.reason,
                "timestamp": log.timestamp.isoformat(),
                "confidence": log.confidence,
            }
            for log in entries
        ]
    }
