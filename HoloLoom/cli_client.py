"""
HoloLoom CLI Client

HTTP client for connecting CLI to HoloLoom API servers:
- Agent Manager API (port 8002): Thread/swarm management
- Agentic API (port 8000): Query execution, monitoring

Created: 2025-12-17
"""

import asyncio
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class APIConfig:
    """API server configuration."""
    agentic_url: str = "http://localhost:8000"
    agent_manager_url: str = "http://localhost:8002"
    timeout: float = 30.0


class HypervisorClient:
    """
    Client for HoloLoom Hypervisor APIs.

    Connects to:
    - Agent Manager API (8002): Thread management, swarms, priority
    - Agentic API (8000): Queries, audit trail, monitoring
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        if HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def check_agentic_health(self) -> Dict[str, Any]:
        """Check Agentic API health."""
        try:
            if not self._client:
                return {"status": "error", "message": "Client not initialized"}
            resp = await self._client.get(f"{self.config.agentic_url}/health")
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def check_agent_manager_health(self) -> Dict[str, Any]:
        """Check Agent Manager API health."""
        try:
            if not self._client:
                return {"status": "error", "message": "Client not initialized"}
            resp = await self._client.get(f"{self.config.agent_manager_url}/health")
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def is_server_available(self) -> bool:
        """Check if at least one server is available."""
        agentic = await self.check_agentic_health()
        agent_mgr = await self.check_agent_manager_health()
        return agentic.get("status") == "ok" or agent_mgr.get("status") == "healthy"

    # =========================================================================
    # Agent Management (Agent Manager API - port 8002)
    # =========================================================================

    async def list_threads(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """List all agent threads."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            params = {"limit": limit}
            if status:
                params["status"] = status
            resp = await self._client.get(
                f"{self.config.agent_manager_url}/api/threads",
                params=params
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Get thread details."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.get(
                f"{self.config.agent_manager_url}/api/threads/{thread_id}"
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def create_thread(
        self,
        name: str,
        query: str,
        agent_type: str = "weaving",
        reasoning_mode: str = "direct",
        priority: int = 0
    ) -> Dict[str, Any]:
        """Create a new agent thread."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.post(
                f"{self.config.agent_manager_url}/api/threads",
                json={
                    "name": name,
                    "query": query,
                    "agent_type": agent_type,
                    "reasoning_mode": reasoning_mode,
                    "priority": priority
                }
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def cancel_thread(self, thread_id: str) -> Dict[str, Any]:
        """Cancel a running thread."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.delete(
                f"{self.config.agent_manager_url}/api/threads/{thread_id}"
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def start_thread(self, thread_id: str) -> Dict[str, Any]:
        """Start a queued thread."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.post(
                f"{self.config.agent_manager_url}/api/threads/{thread_id}/start"
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def pause_thread(self, thread_id: str) -> Dict[str, Any]:
        """Pause a running thread."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.post(
                f"{self.config.agent_manager_url}/api/threads/{thread_id}/pause"
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def resume_thread(self, thread_id: str) -> Dict[str, Any]:
        """Resume a paused thread."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.post(
                f"{self.config.agent_manager_url}/api/threads/{thread_id}/resume"
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def wait_for_completion(
        self,
        thread_id: str,
        poll_interval: float = 0.5,
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Wait for a thread to complete by polling."""
        import time
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            thread = await self.get_thread(thread_id)
            if "error" in thread:
                return thread

            status = thread.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return thread

            await asyncio.sleep(poll_interval)

        return {"error": "Timeout waiting for thread completion", "thread_id": thread_id}

    async def get_thread_steps(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get steps for a thread."""
        try:
            if not self._client:
                return []
            resp = await self._client.get(
                f"{self.config.agent_manager_url}/api/threads/{thread_id}/steps"
            )
            return resp.json()
        except Exception as e:
            return []

    # =========================================================================
    # Queries (Agentic API - port 8000)
    # =========================================================================

    async def query(
        self,
        text: str,
        mode: str = "verify",
        max_steps: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a query through agentic reasoning."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            payload = {
                "text": text,
                "mode": mode,
                "max_steps": max_steps
            }
            if context:
                payload["context"] = context
            resp = await self._client.post(
                f"{self.config.agentic_url}/query",
                json=payload
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Stats & Monitoring
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.get(f"{self.config.agentic_url}/stats")
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    async def get_agent_manager_stats(self) -> Dict[str, Any]:
        """Get agent manager statistics."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            # Get thread list for stats
            threads = await self.list_threads(limit=200)
            return {
                "active_threads": threads.get("active", 0),
                "completed": threads.get("completed", 0),
                "failed": threads.get("failed", 0),
                "total": threads.get("total", 0)
            }
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Audit Trail
    # =========================================================================

    async def get_audit_trail(self, limit: int = 100) -> Dict[str, Any]:
        """Get audit trail entries."""
        try:
            if not self._client:
                return {"error": "Client not initialized"}
            resp = await self._client.get(
                f"{self.config.agentic_url}/audit-trail",
                params={"limit": limit}
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Cluster (Federation)
    # =========================================================================

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status (requires federation)."""
        # Check agent manager health as proxy for cluster
        health = await self.check_agent_manager_health()

        if health.get("status") == "healthy":
            return {
                "status": "healthy",
                "nodes": 1,  # Single node for now
                "active_threads": health.get("active_threads", 0),
                "active_swarms": health.get("active_swarms", 0),
                "uptime_seconds": health.get("uptime_seconds", 0)
            }
        else:
            return {
                "status": "unavailable",
                "message": health.get("message", "Agent Manager not running")
            }

    async def get_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Get cluster nodes."""
        health = await self.check_agent_manager_health()

        if health.get("status") == "healthy":
            return [{
                "id": "local",
                "address": self.config.agent_manager_url,
                "status": "healthy",
                "active_threads": health.get("active_threads", 0),
                "uptime_seconds": health.get("uptime_seconds", 0)
            }]
        else:
            return []


# Convenience function
def create_client(config: Optional[APIConfig] = None) -> HypervisorClient:
    """Create a HypervisorClient instance."""
    return HypervisorClient(config)
