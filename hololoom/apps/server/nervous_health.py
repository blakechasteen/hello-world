"""
Nervous Health — unified vital signs across the nervous system.

Aggregates health status from:
  - Cortex (HoloLoom): Ollama models, memory bus, vault, model router
  - Peripheral (Femtoclaw): reads data/health.json if path configured
  - Autonomic (OpenClaw): HTTP probe if URL configured

Metaphor: the autonomic nervous system monitors vital signs across all organs.
"""
import json
import logging
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["nervous-health"])

# --- Config from env ---

FEMTOCLAW_HEALTH_PATH = os.environ.get("FEMTOCLAW_HEALTH_PATH", "")
OPENCLAW_STATUS_URL = os.environ.get("OPENCLAW_STATUS_URL", "")
OLLAMA_URL = os.environ.get("PROMPTLY_OLLAMA_URL", "http://127.0.0.1:11434").strip()


# --- Response models ---

class ServiceDetail(BaseModel):
    name: str
    status: str  # healthy | degraded | down | unchecked
    latency_ms: float | None = None
    error: str | None = None


class SystemHealth(BaseModel):
    name: str
    role: str  # cortex | peripheral | autonomic
    status: str  # healthy | degraded | down | unchecked
    latency_ms: float | None = None
    services: list[ServiceDetail] = []


class NervousHealthResponse(BaseModel):
    timestamp: str
    overall: str  # healthy | degraded | critical
    systems: dict[str, SystemHealth]


# --- Health checks ---

async def _check_ollama() -> ServiceDetail:
    """Check local Ollama instance."""
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            latency = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                data = resp.json()
                return ServiceDetail(
                    name="ollama",
                    status="healthy",
                    latency_ms=round(latency, 1),
                )
            return ServiceDetail(name="ollama", status="down", latency_ms=round(latency, 1),
                                 error=f"HTTP {resp.status_code}")
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return ServiceDetail(name="ollama", status="down", latency_ms=round(latency, 1),
                             error=str(e)[:200])


async def _check_memory_bus() -> ServiceDetail:
    """Check if memory bus SQLite is accessible."""
    try:
        from hololoom.apps.server.memory_bus import get_observation_store
        store = get_observation_store()
        # Just check if the store exists and is initialized
        status = "healthy" if store else "down"
        return ServiceDetail(name="memory_bus", status=status)
    except Exception as e:
        return ServiceDetail(name="memory_bus", status="down", error=str(e)[:200])


async def _check_model_router() -> ServiceDetail:
    """Check model router bandit state."""
    try:
        from hololoom.apps.server.model_router import get_router
        router_inst = get_router()
        stats = router_inst.bandit.stats()
        return ServiceDetail(
            name="model_router",
            status="healthy",
        )
    except Exception as e:
        return ServiceDetail(name="model_router", status="degraded", error=str(e)[:200])


def _check_femtoclaw() -> SystemHealth:
    """Read Femtoclaw's health.json if path is configured."""
    if not FEMTOCLAW_HEALTH_PATH:
        return SystemHealth(
            name="femtoclaw", role="peripheral", status="unchecked",
        )

    try:
        with open(FEMTOCLAW_HEALTH_PATH, "r") as f:
            data = json.load(f)

        services = []
        for svc in data.get("services", []):
            services.append(ServiceDetail(
                name=svc.get("name", "unknown"),
                status=svc.get("status", "unchecked"),
                latency_ms=svc.get("latencyMs"),
                error=svc.get("error"),
            ))

        return SystemHealth(
            name="femtoclaw",
            role="peripheral",
            status=data.get("overall", "unchecked"),
            services=services,
        )
    except FileNotFoundError:
        return SystemHealth(name="femtoclaw", role="peripheral", status="unchecked",
                            services=[ServiceDetail(name="health_file", status="down",
                                                     error="health.json not found")])
    except Exception as e:
        return SystemHealth(name="femtoclaw", role="peripheral", status="down",
                            services=[ServiceDetail(name="health_file", status="down",
                                                     error=str(e)[:200])])


async def _check_openclaw() -> SystemHealth:
    """Probe OpenClaw status endpoint if configured."""
    if not OPENCLAW_STATUS_URL:
        return SystemHealth(name="openclaw", role="autonomic", status="unchecked")

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OPENCLAW_STATUS_URL)
            latency = (time.monotonic() - start) * 1000
            if resp.status_code == 200:
                return SystemHealth(
                    name="openclaw", role="autonomic", status="healthy",
                    latency_ms=round(latency, 1),
                )
            return SystemHealth(
                name="openclaw", role="autonomic", status="down",
                latency_ms=round(latency, 1),
                services=[ServiceDetail(name="gateway", status="down",
                                         error=f"HTTP {resp.status_code}")],
            )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return SystemHealth(
            name="openclaw", role="autonomic", status="down",
            latency_ms=round(latency, 1),
            services=[ServiceDetail(name="gateway", status="down", error=str(e)[:200])],
        )


def _compute_overall(systems: dict[str, SystemHealth]) -> str:
    """Aggregate system health into overall status."""
    statuses = [s.status for s in systems.values()]
    # Cortex down = critical
    cortex = systems.get("hololoom")
    if cortex and cortex.status == "down":
        return "critical"
    if any(s == "down" for s in statuses):
        return "degraded"
    if any(s == "degraded" for s in statuses):
        return "degraded"
    return "healthy"


# --- Endpoint ---

@router.get("/nervous/health", response_model=NervousHealthResponse)
async def nervous_health() -> NervousHealthResponse:
    """Unified health pulse across the entire nervous system."""
    from datetime import datetime, timezone

    # Check cortex (HoloLoom) services
    ollama = await _check_ollama()
    memory_bus = await _check_memory_bus()
    model_router_svc = await _check_model_router()

    cortex_services = [ollama, memory_bus, model_router_svc]
    cortex_status = "healthy"
    if any(s.status == "down" for s in cortex_services):
        cortex_status = "degraded"

    hololoom = SystemHealth(
        name="hololoom",
        role="cortex",
        status=cortex_status,
        services=cortex_services,
    )

    # Check peripheral (Femtoclaw)
    femtoclaw = _check_femtoclaw()

    # Check autonomic (OpenClaw)
    openclaw = await _check_openclaw()

    systems = {
        "hololoom": hololoom,
        "femtoclaw": femtoclaw,
        "openclaw": openclaw,
    }

    return NervousHealthResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall=_compute_overall(systems),
        systems=systems,
    )


@router.get("/nervous/health/full")
async def nervous_health_full() -> dict:
    """Extended health: base health + identity heartbeat + runtime layers.

    Reads heartbeat.md written by IdentityLayer every 60s.
    Falls back gracefully when heartbeat data isn't available.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    base = await nervous_health()
    result = base.model_dump()

    # Read identity heartbeat (written by IdentityLayer)
    heartbeat = {}
    try:
        from hololoom.agents.eval.identity_loader import IdentityLoader
        loader = IdentityLoader()
        for agent_name in loader.available_agents():
            hb_path = loader.root / agent_name / "heartbeat.md"
            if hb_path.exists():
                heartbeat[agent_name] = hb_path.read_text(encoding="utf-8")
    except ImportError:
        pass  # Identity system not installed
    except Exception as e:
        logger.debug("Failed to read heartbeat files: %s", e)

    result["heartbeat"] = heartbeat
    return result
