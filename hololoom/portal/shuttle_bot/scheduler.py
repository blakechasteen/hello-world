"""
Simple Scheduler - Picks healthy nodes for job dispatch with queuing support.

Combines intelligent load balancing with job queue fallback:
- Uses LoadBalancer for ROUND_ROBIN, LEAST_LOADED, WEIGHTED, or CAPABILITY strategies
- Queues jobs when no nodes available, dispatches via background task
- Automatic retry with priority demotion on failure

Methods:
    get_nodes() - Get nodes from Portal
    pick_node() - Pick a healthy node using load balancer
    dispatch_job() - Dispatch job synchronously (blocking)
    dispatch_job_async() - Dispatch job asynchronously (non-blocking, <100ms)
    dispatch_or_queue() - Dispatch immediately or queue if no nodes available
    dispatch_batch_async() - Submit multiple jobs in single request
    poll_job_status() - Poll job status from node
    poll_job_progress() - Poll detailed job progress
    cancel_job() - Cancel a running or pending job
    start_queue_dispatcher() - Start background queue processing
    stop_queue_dispatcher() - Stop background queue processing
    get_queue_stats() - Get queue and dispatcher statistics
    get_load_balancer_stats() - Get load balancer metrics
"""

import uuid
from typing import Any

import httpx

from ..portal_server.job_queue import JobQueue, QueueDispatcher, QueuedJob
from ..shared.load_balancer import LoadBalancer, LoadBalanceStrategy, SelectionResult
from ..shared.logging import get_logger
from ..shared.types import NodeRecord

logger = get_logger(__name__, component="shuttle")


class SimpleScheduler:
    """
    Simple job scheduler that picks healthy nodes.

    MVP behavior:
    - Queries Portal for online nodes
    - Verifies node health before selection
    - Returns first healthy node
    """

    def __init__(
        self,
        portal_url: str,
        shared_secret: str,
        timeout: float = 5.0,
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED,
    ):
        """
        Initialize scheduler.

        Args:
            portal_url: Portal Server URL
            shared_secret: Shared secret for authentication
            timeout: HTTP request timeout
            load_balance_strategy: Node selection strategy (default: LEAST_LOADED)
        """
        self.portal_url = portal_url.rstrip("/")
        self.shared_secret = shared_secret
        self.timeout = timeout

        # Load balancer for intelligent node selection
        self.load_balancer = LoadBalancer(strategy=load_balance_strategy)

        # Job queue for buffering when no nodes available
        self.job_queue = JobQueue(max_size=10000)
        self.queue_dispatcher = QueueDispatcher(self.job_queue, self)

    async def get_nodes(self, online_only: bool = True) -> list[NodeRecord]:
        """
        Get nodes from Portal.

        Args:
            online_only: Only return online nodes

        Returns:
            List of NodeRecord objects
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.portal_url}/nodes",
                    params={"online_only": str(online_only).lower()},
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    nodes_data = response.json()
                    return [NodeRecord(**n) for n in nodes_data]
                else:
                    logger.error(f"Failed to get nodes: {response.status_code}")
                    return []

        except Exception as e:
            logger.error(f"Failed to get nodes from Portal: {e}")
            return []

    async def check_node_health(self, node: NodeRecord) -> bool:
        """
        Verify a node is healthy by calling its status endpoint.

        Args:
            node: Node to check

        Returns:
            True if node is healthy
        """
        try:
            # Parse address (host:port)
            address = node.address
            if not address.startswith("http"):
                address = f"http://{address}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{address}/health",
                    timeout=self.timeout,
                )

                return response.status_code == 200

        except Exception as e:
            logger.warning(f"Node {node.node_id} health check failed: {e}")
            return False

    async def pick_node(
        self,
        job_requirements: dict[str, Any] | None = None,
    ) -> NodeRecord | None:
        """
        Pick a healthy node for job dispatch using load balancing.

        Args:
            job_requirements: Optional requirements (gpu, min_memory_mb, etc.)

        Returns:
            NodeRecord or None if no healthy nodes available
        """
        nodes = await self.get_nodes(online_only=True)

        if not nodes:
            logger.warning("No online nodes available")
            return None

        # Use load balancer for selection
        result: SelectionResult = self.load_balancer.select(
            nodes=nodes,
            job_requirements=job_requirements
        )

        if not result.node:
            logger.warning(f"Load balancer returned no node: {result.reason}")
            return None

        # Verify health of selected node
        if await self.check_node_health(result.node):
            logger.info(
                f"Selected node: {result.node.node_id} "
                f"(strategy={result.strategy_used.value}, {result.reason})"
            )
            return result.node

        # Health check failed - try next best node
        logger.warning(f"Health check failed for {result.node.node_id}, trying alternatives")
        remaining = [n for n in nodes if n.node_id != result.node.node_id]

        for node in remaining:
            if await self.check_node_health(node):
                logger.info(f"Fallback to node: {node.node_id}")
                return node

        logger.warning("No healthy nodes found after fallback")
        return None

    def get_load_balancer_stats(self) -> dict[str, Any]:
        """Get load balancer metrics."""
        return self.load_balancer.get_stats()

    async def dispatch_job(
        self,
        node: NodeRecord,
        module_id: str,
        input_json: dict,
        job_id: str | None = None,
        timeout_seconds: int = 60,
    ) -> dict:
        """
        Dispatch a job to a specific node.

        Args:
            node: Target node
            module_id: WASM module to execute
            input_json: Input data for the job
            job_id: Optional job ID (generated if not provided)
            timeout_seconds: Job timeout

        Returns:
            Job result dict
        """
        import uuid

        if not job_id:
            job_id = str(uuid.uuid4())[:8]

        address = node.address
        if not address.startswith("http"):
            address = f"http://{address}"

        job_request = {
            "job_id": job_id,
            "module_id": module_id,
            "entry_function": "run",
            "input_json": input_json,
            "timeout_seconds": timeout_seconds,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{address}/jobs",
                    json=job_request,
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=timeout_seconds + 5,  # Extra buffer
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "job_id": job_id,
                        "status": "failed",
                        "error": f"Node returned {response.status_code}: {response.text}",
                    }

        except Exception as e:
            logger.error(f"Job dispatch failed: {e}")
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }

    async def dispatch_job_async(
        self,
        node: NodeRecord,
        module_id: str,
        input_json: dict,
        job_id: str | None = None,
        timeout_seconds: int = 60,
    ) -> dict:
        """
        Submit a job for async execution - returns immediately with job_id.

        Unlike dispatch_job(), this doesn't wait for completion.
        Use poll_job_status() to check progress.

        Args:
            node: Target node
            module_id: WASM module to execute
            input_json: Input data for the job
            job_id: Optional job ID (generated if not provided)
            timeout_seconds: Job timeout (for execution, not submission)

        Returns:
            Submission result with job_id and status
        """
        import uuid

        if not job_id:
            job_id = str(uuid.uuid4())[:8]

        address = node.address
        if not address.startswith("http"):
            address = f"http://{address}"

        job_request = {
            "job_id": job_id,
            "module_id": module_id,
            "entry_function": "run",
            "input_json": input_json,
            "timeout_seconds": timeout_seconds,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{address}/jobs/async",
                    json=job_request,
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=5.0,  # Quick submission, not waiting for execution
                )

                if response.status_code in (200, 202):
                    result = response.json()
                    result["job_id"] = job_id  # Ensure job_id is in response
                    return result
                else:
                    return {
                        "job_id": job_id,
                        "status": "failed",
                        "error": f"Node returned {response.status_code}: {response.text}",
                    }

        except Exception as e:
            logger.error(f"Async job submission failed: {e}")
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }

    async def poll_job_status(self, node: NodeRecord, job_id: str) -> dict:
        """
        Poll job status from a node.

        Args:
            node: Node where job was submitted
            job_id: Job ID to check

        Returns:
            Job status dict with 'status' field (pending/running/completed/failed)
        """
        address = node.address
        if not address.startswith("http"):
            address = f"http://{address}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{address}/jobs/{job_id}",
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=5.0,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {"status": "not_found", "error": f"Job {job_id} not found"}
                else:
                    return {
                        "status": "error",
                        "error": f"Node returned {response.status_code}",
                    }

        except Exception as e:
            logger.warning(f"Job status poll failed: {e}")
            return {"status": "error", "error": str(e)}

    async def poll_job_progress(self, node: NodeRecord, job_id: str) -> dict:
        """
        Poll detailed job progress from a node.

        Args:
            node: Node where job was submitted
            job_id: Job ID to check

        Returns:
            Progress dict with step info, percentage, and message
        """
        address = node.address
        if not address.startswith("http"):
            address = f"http://{address}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{address}/jobs/{job_id}/progress",
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=5.0,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {"status": "not_found", "error": f"Job {job_id} not found"}
                else:
                    return {
                        "status": "error",
                        "error": f"Node returned {response.status_code}",
                    }

        except Exception as e:
            logger.warning(f"Job progress poll failed: {e}")
            return {"status": "error", "error": str(e)}

    async def cancel_job(self, node: NodeRecord, job_id: str) -> dict:
        """
        Cancel a running or pending job on a node.

        Args:
            node: Node where job was submitted
            job_id: Job ID to cancel

        Returns:
            Cancellation result with 'cancelled' boolean and message
        """
        address = node.address
        if not address.startswith("http"):
            address = f"http://{address}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{address}/jobs/{job_id}/cancel",
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=5.0,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return {
                        "cancelled": False,
                        "message": f"Job {job_id} not found",
                    }
                else:
                    return {
                        "cancelled": False,
                        "message": f"Node returned {response.status_code}: {response.text}",
                    }

        except Exception as e:
            logger.error(f"Job cancellation failed: {e}")
            return {"cancelled": False, "message": str(e)}

    async def dispatch_batch_async(
        self,
        node: NodeRecord,
        jobs: list[dict],
    ) -> dict:
        """
        Submit multiple jobs for async execution in a single request.

        Each job dict should have: module_id, input_json, and optionally job_id, timeout_seconds.

        Args:
            node: Target node
            jobs: List of job specifications

        Returns:
            Batch submission result with submitted count, failed count, and job IDs
        """
        import uuid

        address = node.address
        if not address.startswith("http"):
            address = f"http://{address}"

        # Ensure each job has a job_id
        job_requests = []
        for job in jobs:
            job_request = {
                "job_id": job.get("job_id") or str(uuid.uuid4())[:8],
                "module_id": job["module_id"],
                "entry_function": job.get("entry_function", "run"),
                "input_json": job["input_json"],
                "timeout_seconds": job.get("timeout_seconds", 60),
            }
            job_requests.append(job_request)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{address}/jobs/batch",
                    json={"jobs": job_requests},
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=10.0,  # Slightly longer for batch
                )

                if response.status_code in (200, 202):
                    return response.json()
                else:
                    return {
                        "submitted": 0,
                        "failed": len(jobs),
                        "job_ids": [],
                        "errors": [f"Node returned {response.status_code}: {response.text}"],
                    }

        except Exception as e:
            logger.error(f"Batch job submission failed: {e}")
            return {
                "submitted": 0,
                "failed": len(jobs),
                "job_ids": [],
                "errors": [str(e)],
            }

    async def dispatch_or_queue(
        self,
        module_id: str,
        input_json: dict,
        job_requirements: dict[str, Any] | None = None,
        job_id: str | None = None,
        timeout_seconds: int = 60,
        priority: int = 0,
    ) -> dict:
        """
        Dispatch job immediately or queue if no nodes available.

        Returns immediately with job_id. Client polls for status.
        Jobs are queued with retry logic if no nodes are available,
        and dispatched by background QueueDispatcher when capacity frees up.

        Args:
            module_id: WASM module to execute
            input_json: Input data for the job
            job_requirements: Optional requirements (gpu, min_memory_mb, etc.)
            job_id: Optional job ID (generated if not provided)
            timeout_seconds: Job execution timeout
            priority: Job priority (higher = more important)

        Returns:
            Result dict with job_id, status, and queue info if queued
        """
        if not job_id:
            job_id = str(uuid.uuid4())[:8]

        # Try to dispatch immediately
        node = await self.pick_node(job_requirements)

        if node:
            # Dispatch immediately to available node
            result = await self.dispatch_job_async(
                node=node,
                module_id=module_id,
                input_json=input_json,
                job_id=job_id,
                timeout_seconds=timeout_seconds,
            )
            result["dispatched_immediately"] = True
            return result
        else:
            # No nodes available - queue for later dispatch
            queued_job = QueuedJob(
                job_id=job_id,
                module_id=module_id,
                input_json=input_json,
                job_requirements=job_requirements,
                timeout_seconds=timeout_seconds,
                priority=priority,
            )

            if await self.job_queue.enqueue(queued_job):
                logger.info(f"Job {job_id} queued (priority={priority})")
                queue_stats = self.job_queue.get_stats()
                return {
                    "job_id": job_id,
                    "status": "queued",
                    "dispatched_immediately": False,
                    "queue_depth": queue_stats["current_depth"],
                    "queue_position": queue_stats["current_depth"],  # Approximate
                }
            else:
                logger.error(f"Job {job_id} rejected - queue full")
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": "Queue full - max capacity reached",
                    "dispatched_immediately": False,
                }

    async def start_queue_dispatcher(self) -> None:
        """Start the background queue dispatcher."""
        await self.queue_dispatcher.start()
        logger.info("Queue dispatcher started")

    async def stop_queue_dispatcher(self) -> None:
        """Stop the background queue dispatcher gracefully."""
        await self.queue_dispatcher.stop()
        logger.info("Queue dispatcher stopped")

    def get_queue_stats(self) -> dict[str, Any]:
        """Get combined queue and dispatcher statistics."""
        queue_stats = self.job_queue.get_stats()
        dispatcher_stats = self.queue_dispatcher.get_stats()
        return {
            "queue": queue_stats,
            "dispatcher": dispatcher_stats,
        }
