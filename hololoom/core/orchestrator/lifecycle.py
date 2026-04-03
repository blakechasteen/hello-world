"""
Orchestrator Lifecycle & Production Methods
============================================
Extracted from WeavingOrchestrator (March 2026 Refactor).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.fabric.spacetime import Spacetime
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

logger = logging.getLogger(__name__)


async def close_orchestrator(orch: WeavingOrchestrator) -> None:
    """Clean up all orchestrator resources. Idempotent."""
    if orch._closed:
        return

    logger.info("Closing WeavingOrchestrator...")

    # Stop recursive learning background learner
    if orch.enable_recursive_learning and orch._recursive_components:
        if orch._recursive_components.get('background_learner'):
            logger.info("Stopping background learner...")
            await orch._recursive_components['background_learner'].stop()

    # Cancel background tasks (with lock protection)
    async with orch._bg_lock:
        if orch._background_tasks:
            logger.info(f"Cancelling {len(orch._background_tasks)} background tasks")
            for task in orch._background_tasks:
                if not task.done():
                    task.cancel()
            tasks_to_wait = list(orch._background_tasks)
        else:
            tasks_to_wait = []

    if tasks_to_wait:
        try:
            await asyncio.wait(
                tasks_to_wait,
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED
            )
        except asyncio.TimeoutError:
            logger.warning("Some background tasks did not complete within timeout")
        orch._background_tasks.clear()

    # Close reflection buffer
    if orch.enable_reflection and orch.reflection_buffer:
        logger.info("Closing reflection buffer...")
        await orch.reflection_buffer.flush()
        await orch.reflection_buffer.close()

    # Stop Jenny Generative UI Runtime
    if orch.enable_jenny and orch.jenny_runtime and orch._jenny_started:
        logger.info("Stopping Jenny UI Runtime...")
        try:
            await orch.jenny_runtime.stop()
            logger.info("Jenny UI Runtime stopped successfully")
        except Exception as e:
            logger.warning(f"Error stopping Jenny UI Runtime: {e}")

    # Close memory backend connections
    if orch.memory:
        logger.info("Closing memory backend connections...")
        try:
            if hasattr(orch.memory, 'close'):
                if asyncio.iscoroutinefunction(orch.memory.close):
                    await orch.memory.close()
                else:
                    orch.memory.close()

            if hasattr(orch.memory, 'neo4j') and orch.memory.neo4j:
                if hasattr(orch.memory.neo4j, 'close'):
                    logger.debug("Closing Neo4j connection...")
                    orch.memory.neo4j.close()

            if hasattr(orch.memory, 'qdrant') and orch.memory.qdrant:
                if hasattr(orch.memory.qdrant, 'close'):
                    logger.debug("Closing Qdrant connection...")
                    orch.memory.qdrant.close()

            logger.info("Memory backend connections closed")
        except Exception as e:
            logger.warning(f"Error closing memory backend: {e}")

    orch._closed = True
    logger.info("WeavingOrchestrator closed successfully")


async def get_health(orch: WeavingOrchestrator) -> dict[str, Any] | None:
    """Get production health check status."""
    if not orch.enable_production_hardening or not orch.health_checker:
        logger.warning("[PRODUCTION] Health checks not enabled")
        return None

    try:
        result = await orch.health_checker.check_health()
        return result.to_dict()
    except Exception as e:
        logger.error(f"[PRODUCTION] Health check failed: {e}")
        return {
            "healthy": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


def get_circuit_breaker_status(orch: WeavingOrchestrator) -> dict[str, Any] | None:
    """Get circuit breaker states for all registered backends."""
    if not orch.enable_production_hardening or not orch.breaker_registry:
        logger.warning("[PRODUCTION] Circuit breakers not enabled")
        return None

    try:
        breakers_status = {}
        for backend_name, breaker in orch.breaker_registry.breakers.items():
            breakers_status[backend_name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure_time": breaker.last_failure_time,
                "opened_at": breaker.opened_at
            }

        return {
            "breakers": breakers_status,
            "healthy": orch.breaker_registry.get_health_summary()["healthy"],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"[PRODUCTION] Failed to get breaker status: {e}")
        return {"error": str(e), "timestamp": time.time()}


def save_dashboard(orch: WeavingOrchestrator, spacetime: Spacetime, output_path: str) -> None:
    """Save dashboard from Spacetime to HTML file."""
    if not orch.enable_dashboards:
        raise ValueError("Dashboards are not enabled. Initialize with enable_dashboards=True")

    dashboard = spacetime.metadata.get('dashboard')
    if dashboard is None:
        raise ValueError("No dashboard found in Spacetime metadata")

    from hololoom.visualization.html_renderer import save_dashboard as _save
    _save(dashboard, output_path)
    logger.info(f"[DASHBOARD] Saved to {output_path}")


async def dream(orch: WeavingOrchestrator, num_epochs: int = 1, num_workers: int = 5) -> None:
    """Trigger EGGROLL evolutionary training on recent experiences."""
    logger.info(f"[DREAM] Starting EGGROLL dream cycle (epochs={num_epochs}, workers={num_workers})")

    try:
        from hololoom.eggroll.integration import EggrollIntegration
        integration = EggrollIntegration(num_workers=num_workers)
        await integration.run_evolution_loop(num_epochs=num_epochs)
        logger.info("[DREAM] Dream cycle complete.")
    except ImportError:
        logger.error("[DREAM] EGGROLL module not found. Cannot dream.")
    except Exception as e:
        logger.error(f"[DREAM] Dream cycle failed: {e}")
