"""
Node Daemon - Per-device WASM job executor.

FastAPI application providing job execution and status endpoints.
Registers with Portal on startup and sends periodic heartbeats.

Endpoints:
    GET /status - Node status with resource metrics
    POST /jobs - Submit a job for execution (blocking)
    POST /jobs/async - Submit a job for async execution (non-blocking, <100ms)
    POST /jobs/batch - Submit multiple jobs for async execution
    GET /jobs/{job_id} - Get job status/result
    POST /jobs/{job_id}/cancel - Cancel a running job
    GET /modules - List available WASM modules
    GET /health - Health check

Background Tasks:
    - Heartbeat loop (periodic Portal registration)
    - Job cleanup loop (expires old completed/failed jobs)
    - Timeout monitor loop (cancels jobs exceeding timeout)
"""

import argparse
import asyncio
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from pydantic import BaseModel

from .config import NodeConfig, load_config
from .wasm_runner import WasmRunner
from .module_registry import ModuleRegistry
from ..shared.types import (
    NodeCapabilities,
    NodeRegistration,
    JobRequest,
    JobResult,
    JobStatus,
    ModuleManifest,
)
from ..shared.logging import configure_logging, get_logger

# Try to import psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Global instances
config: Optional[NodeConfig] = None
wasm_runner: Optional[WasmRunner] = None
module_registry: Optional[ModuleRegistry] = None
logger = get_logger(__name__, component="node")

# Job tracking
_jobs: Dict[str, JobResult] = {}
_job_semaphore: Optional[asyncio.Semaphore] = None
_heartbeat_task: Optional[asyncio.Task] = None


# Async job tracking for non-blocking execution
from dataclasses import dataclass, field


@dataclass
class JobProgress:
    """Progress tracking for multi-step operations."""
    current_step: int = 0
    total_steps: int = 1
    step_name: str = ""
    percentage: float = 0.0
    message: str = ""


@dataclass
class AsyncJobTracker:
    """Tracks async job execution state."""
    job_id: str
    status: JobStatus
    submitted_at: datetime
    timeout_seconds: int = 60
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[JobResult] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)
    progress: JobProgress = field(default_factory=JobProgress)
    cancelled: bool = False


_async_jobs: Dict[str, AsyncJobTracker] = {}
_cleanup_task: Optional[asyncio.Task] = None
_timeout_monitor_task: Optional[asyncio.Task] = None

# Configuration for job lifecycle
JOB_EXPIRY_SECONDS = 3600  # 1 hour - completed jobs expire after this
JOB_CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes - check for expired jobs
JOB_TIMEOUT_CHECK_INTERVAL_SECONDS = 5  # 5 seconds - check for timed-out jobs
JOB_PERSIST_INTERVAL_SECONDS = 30  # 30 seconds - persist job state to disk

# Persistence configuration
_persistence_enabled: bool = False
_persistence_path: Optional[str] = None
_persistence_task: Optional[asyncio.Task] = None


def enable_persistence(path: str = ".job_state.json") -> None:
    """Enable job state persistence to the specified path."""
    global _persistence_enabled, _persistence_path
    _persistence_enabled = True
    _persistence_path = path
    logger.info(f"Job persistence enabled: {path}")


async def save_job_state() -> bool:
    """Save current job state to disk for crash recovery."""
    if not _persistence_enabled or not _persistence_path:
        return False

    try:
        import json
        from pathlib import Path

        # Only persist non-running jobs (running jobs can't be resumed)
        state = {
            "saved_at": datetime.utcnow().isoformat(),
            "node_id": config.node_id if config else None,
            "jobs": []
        }

        for job_id, tracker in _async_jobs.items():
            # Skip running jobs (tasks can't be serialized)
            if tracker.status == JobStatus.RUNNING:
                continue

            job_data = {
                "job_id": tracker.job_id,
                "status": tracker.status.value,
                "submitted_at": tracker.submitted_at.isoformat(),
                "timeout_seconds": tracker.timeout_seconds,
                "started_at": tracker.started_at.isoformat() if tracker.started_at else None,
                "completed_at": tracker.completed_at.isoformat() if tracker.completed_at else None,
                "cancelled": tracker.cancelled,
                "progress": {
                    "current_step": tracker.progress.current_step,
                    "total_steps": tracker.progress.total_steps,
                    "step_name": tracker.progress.step_name,
                    "percentage": tracker.progress.percentage,
                    "message": tracker.progress.message,
                },
            }

            # Serialize result if available
            if tracker.result:
                job_data["result"] = {
                    "job_id": tracker.result.job_id,
                    "status": tracker.result.status.value,
                    "output_json": tracker.result.output_json,
                    "error": tracker.result.error,
                    "execution_time_ms": tracker.result.execution_time_ms,
                    "node_id": tracker.result.node_id,
                }

            state["jobs"].append(job_data)

        # Atomic write via temp file
        path = Path(_persistence_path)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(state, indent=2))
        temp_path.rename(path)

        logger.debug(f"Saved {len(state['jobs'])} jobs to {path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to save job state: {e}")
        return False


async def load_job_state() -> int:
    """Load job state from disk on startup. Returns number of jobs loaded."""
    if not _persistence_enabled or not _persistence_path:
        return 0

    try:
        import json
        from pathlib import Path

        path = Path(_persistence_path)
        if not path.exists():
            logger.debug(f"No persistence file found at {path}")
            return 0

        data = json.loads(path.read_text())
        loaded_count = 0

        for job_data in data.get("jobs", []):
            try:
                job_id = job_data["job_id"]

                # Parse dates
                submitted_at = datetime.fromisoformat(job_data["submitted_at"])
                started_at = datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None
                completed_at = datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None

                # Create progress
                prog_data = job_data.get("progress", {})
                progress = JobProgress(
                    current_step=prog_data.get("current_step", 0),
                    total_steps=prog_data.get("total_steps", 1),
                    step_name=prog_data.get("step_name", ""),
                    percentage=prog_data.get("percentage", 0.0),
                    message=prog_data.get("message", ""),
                )

                # Create tracker
                tracker = AsyncJobTracker(
                    job_id=job_id,
                    status=JobStatus(job_data["status"]),
                    submitted_at=submitted_at,
                    timeout_seconds=job_data.get("timeout_seconds", 60),
                    started_at=started_at,
                    completed_at=completed_at,
                    progress=progress,
                    cancelled=job_data.get("cancelled", False),
                )

                # Restore result if available
                if "result" in job_data:
                    res_data = job_data["result"]
                    tracker.result = JobResult(
                        job_id=res_data["job_id"],
                        status=JobStatus(res_data["status"]),
                        output_json=res_data.get("output_json"),
                        error=res_data.get("error"),
                        execution_time_ms=res_data.get("execution_time_ms"),
                        node_id=res_data.get("node_id"),
                    )
                    _jobs[job_id] = tracker.result

                _async_jobs[job_id] = tracker
                loaded_count += 1

            except Exception as e:
                logger.warning(f"Failed to load job {job_data.get('job_id', '?')}: {e}")
                continue

        logger.info(f"Loaded {loaded_count} jobs from {path}")
        return loaded_count

    except Exception as e:
        logger.warning(f"Failed to load job state: {e}")
        return 0


async def persistence_loop() -> None:
    """Background task that periodically saves job state."""
    while True:
        await asyncio.sleep(JOB_PERSIST_INTERVAL_SECONDS)
        await save_job_state()


async def job_cleanup_loop() -> None:
    """Background task that cleans up expired completed/failed jobs."""
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        try:
            now = datetime.utcnow()
            expired_ids = []

            for job_id, tracker in list(_async_jobs.items()):
                # Only clean up completed jobs (not running/pending)
                if tracker.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if tracker.completed_at:
                        age = (now - tracker.completed_at).total_seconds()
                        if age > JOB_EXPIRY_SECONDS:
                            expired_ids.append(job_id)

            # Remove expired jobs
            for job_id in expired_ids:
                del _async_jobs[job_id]
                if job_id in _jobs:
                    del _jobs[job_id]

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired jobs")

        except Exception as e:
            logger.error(f"Job cleanup error: {e}")


async def timeout_monitor_loop() -> None:
    """Background task that cancels jobs exceeding their timeout."""
    while True:
        await asyncio.sleep(JOB_TIMEOUT_CHECK_INTERVAL_SECONDS)
        try:
            now = datetime.utcnow()

            for job_id, tracker in list(_async_jobs.items()):
                # Only check running jobs
                if tracker.status == JobStatus.RUNNING and tracker.started_at:
                    elapsed = (now - tracker.started_at).total_seconds()
                    if elapsed > tracker.timeout_seconds:
                        # Cancel the task
                        if tracker.task and not tracker.task.done():
                            tracker.task.cancel()
                            tracker.cancelled = True
                            tracker.status = JobStatus.FAILED
                            tracker.completed_at = now
                            tracker.result = JobResult(
                                job_id=job_id,
                                status=JobStatus.FAILED,
                                error=f"Job timed out after {tracker.timeout_seconds}s",
                                node_id=config.node_id if config else None,
                            )
                            _jobs[job_id] = tracker.result
                            logger.warning(
                                f"Job {job_id} timed out after {elapsed:.1f}s",
                                extra={"job_id": job_id}
                            )

        except Exception as e:
            logger.error(f"Timeout monitor error: {e}")


class NodeStatusResponse(BaseModel):
    node_id: str
    status: str
    cpu_percent: float
    memory_percent: float
    memory_available_mb: int
    current_jobs: int
    max_jobs: int
    wasm_runtime_available: bool
    modules_loaded: int
    uptime_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    node_id: str
    timestamp: str


def get_capabilities() -> NodeCapabilities:
    """Get current node hardware capabilities."""
    if PSUTIL_AVAILABLE:
        return NodeCapabilities(
            cpu_cores=psutil.cpu_count() or 1,
            memory_mb=int(psutil.virtual_memory().total / 1024 / 1024),
            gpu=False,  # TODO: GPU detection
            wasm_runtime="wasmtime" if wasm_runner and wasm_runner.is_available() else "mock",
        )
    else:
        return NodeCapabilities(
            cpu_cores=1,
            memory_mb=1024,
            gpu=False,
            wasm_runtime="mock",
        )


def verify_secret(x_shared_secret: str = Header(...)) -> str:
    """Verify shared secret from header."""
    if config and x_shared_secret != config.shared_secret:
        raise HTTPException(status_code=401, detail="Invalid shared secret")
    return x_shared_secret


async def register_with_portal() -> bool:
    """Register this node with the Portal server."""
    if not config:
        return False

    try:
        # Get our address
        hostname = socket.gethostname()
        address = f"{hostname}:{config.port}"

        registration = NodeRegistration(
            node_id=config.node_id,
            address=address,
            capabilities=get_capabilities(),
            shared_secret=config.shared_secret,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.portal_url}/nodes/register",
                json=registration.model_dump(),
                headers={"X-Shared-Secret": config.shared_secret},
                timeout=10.0,
            )

            if response.status_code == 200:
                logger.info(f"Registered with Portal: {config.portal_url}")
                return True
            else:
                logger.error(f"Failed to register: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        logger.error(f"Failed to register with Portal: {e}")
        return False


async def send_heartbeat() -> bool:
    """Send heartbeat to Portal."""
    if not config:
        return False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.portal_url}/nodes/{config.node_id}/heartbeat",
                headers={"X-Shared-Secret": config.shared_secret},
                timeout=5.0,
            )
            return response.status_code == 200

    except Exception as e:
        logger.warning(f"Heartbeat failed: {e}")
        return False


async def heartbeat_loop() -> None:
    """Background task that sends periodic heartbeats."""
    while True:
        await asyncio.sleep(config.heartbeat_interval_seconds if config else 30)
        await send_heartbeat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup, cleanup on shutdown."""
    global config, wasm_runner, module_registry, _job_semaphore
    global _heartbeat_task, _cleanup_task, _timeout_monitor_task, _persistence_task

    # Load config if not already set
    if config is None:
        config = load_config()

    configure_logging(level=config.log_level, json_format=config.log_json)
    logger.info(f"Starting Node Daemon: {config.node_id}")

    # Initialize WASM runner
    wasm_runner = WasmRunner(module_dir=config.wasm_module_dir)
    logger.info(f"WASM runtime available: {wasm_runner.is_available()}")

    # Initialize module registry
    module_registry = ModuleRegistry(config.wasm_module_dir)
    logger.info(f"Loaded {len(module_registry.list())} modules")

    # Initialize job semaphore
    _job_semaphore = asyncio.Semaphore(config.max_concurrent_jobs)

    # Load persisted job state (if enabled)
    loaded_jobs = await load_job_state()
    if loaded_jobs > 0:
        logger.info(f"Restored {loaded_jobs} jobs from persistent storage")

    # Register with Portal
    await register_with_portal()

    # Start background tasks
    _heartbeat_task = asyncio.create_task(heartbeat_loop())
    _cleanup_task = asyncio.create_task(job_cleanup_loop())
    _timeout_monitor_task = asyncio.create_task(timeout_monitor_loop())

    # Start persistence task if enabled
    if _persistence_enabled:
        _persistence_task = asyncio.create_task(persistence_loop())
        logger.info("Background tasks started: heartbeat, cleanup, timeout monitor, persistence")
    else:
        logger.info("Background tasks started: heartbeat, cleanup, timeout monitor")

    yield

    # Save job state before shutdown (if enabled)
    if _persistence_enabled:
        await save_job_state()
        logger.info("Saved job state to persistent storage")

    # Cleanup all background tasks
    tasks_to_cancel = [
        (_heartbeat_task, "heartbeat"),
        (_cleanup_task, "cleanup"),
        (_timeout_monitor_task, "timeout_monitor"),
    ]

    if _persistence_task:
        tasks_to_cancel.append((_persistence_task, "persistence"))

    for task, name in tasks_to_cancel:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug(f"Stopped {name} task")

    # Cancel any running jobs
    for job_id, tracker in list(_async_jobs.items()):
        if tracker.task and not tracker.task.done():
            tracker.task.cancel()
            logger.debug(f"Cancelled running job {job_id}")

    logger.info("Node Daemon shutting down")


def create_app(node_config: Optional[NodeConfig] = None) -> FastAPI:
    """Create FastAPI application with optional config."""
    global config
    if node_config:
        config = node_config

    app = FastAPI(
        title="Node Daemon",
        description="HoloLoom distributed compute node",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)
    return app


# Router
from fastapi import APIRouter
router = APIRouter()

_start_time = datetime.utcnow()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        node_id=config.node_id if config else "unknown",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/status", response_model=NodeStatusResponse)
async def node_status():
    """Get detailed node status with resource metrics."""
    if PSUTIL_AVAILABLE:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_available_mb = int(mem.available / 1024 / 1024)
    else:
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_available_mb = 1024

    # Count active jobs
    active_jobs = sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING)

    uptime = (datetime.utcnow() - _start_time).total_seconds()

    return NodeStatusResponse(
        node_id=config.node_id if config else "unknown",
        status="online",
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        memory_available_mb=memory_available_mb,
        current_jobs=active_jobs,
        max_jobs=config.max_concurrent_jobs if config else 4,
        wasm_runtime_available=wasm_runner.is_available() if wasm_runner else False,
        modules_loaded=len(module_registry.list()) if module_registry else 0,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/jobs", response_model=JobResult)
async def submit_job(
    job: JobRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_secret),
):
    """
    Submit a job for execution.

    The job will be executed using the specified WASM module.
    For synchronous execution, waits for completion.
    """
    if not wasm_runner:
        raise HTTPException(status_code=503, detail="WASM runner not initialized")

    # Check if module exists (unless base64 provided)
    if not job.wasm_base64:
        if module_registry and not module_registry.exists(job.module_id):
            raise HTTPException(
                status_code=404,
                detail=f"Module {job.module_id} not found"
            )

    # Acquire semaphore for concurrency limit
    async with _job_semaphore:
        logger.info(f"Executing job {job.job_id} with module {job.module_id}")

        # Mark as running
        _jobs[job.job_id] = JobResult(
            job_id=job.job_id,
            status=JobStatus.RUNNING,
            node_id=config.node_id if config else None,
        )

        # Get WASM path if available
        wasm_path = None
        if module_registry:
            wasm_path = module_registry.get_wasm_path(job.module_id)

        # Execute
        result = wasm_runner.run(
            module_id=job.module_id,
            entry_function=job.entry_function,
            input_json=job.input_json,
            wasm_path=wasm_path,
            wasm_base64=job.wasm_base64,
            timeout_seconds=job.timeout_seconds,
        )

        # Update result with job info
        result.job_id = job.job_id
        result.node_id = config.node_id if config else None

        # Store result
        _jobs[job.job_id] = result

        logger.info(
            f"Job {job.job_id} completed: {result.status}",
            extra={"job_id": job.job_id}
        )

        return result


@router.get("/jobs/{job_id}", response_model=JobResult)
async def get_job(
    job_id: str,
    _: str = Depends(verify_secret),
):
    """Get job status and result."""
    # Check async jobs first (includes result when complete)
    if job_id in _async_jobs:
        tracker = _async_jobs[job_id]
        if tracker.result:
            return tracker.result
        # Return a pending/running result
        return JobResult(
            job_id=job_id,
            status=tracker.status,
            node_id=config.node_id if config else None,
        )

    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _jobs[job_id]


async def _execute_job_async(job: JobRequest, tracker: AsyncJobTracker) -> None:
    """
    Execute a job asynchronously in the background.

    Acquires semaphore inside the task (not blocking HTTP response),
    runs WASM in thread pool, and updates tracker with result.
    Supports cancellation checking throughout execution.
    """
    try:
        # Check for early cancellation before acquiring semaphore
        if tracker.cancelled:
            tracker.status = JobStatus.FAILED
            tracker.completed_at = datetime.utcnow()
            tracker.result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error="Job cancelled before execution",
                node_id=config.node_id if config else None,
            )
            _jobs[job.job_id] = tracker.result
            return

        async with _job_semaphore:
            # Check again after acquiring semaphore (may have been cancelled while waiting)
            if tracker.cancelled:
                tracker.status = JobStatus.FAILED
                tracker.completed_at = datetime.utcnow()
                tracker.result = JobResult(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error="Job cancelled while waiting in queue",
                    node_id=config.node_id if config else None,
                )
                _jobs[job.job_id] = tracker.result
                return

            tracker.status = JobStatus.RUNNING
            tracker.started_at = datetime.utcnow()
            tracker.progress = JobProgress(
                current_step=1,
                total_steps=3,
                step_name="Initializing",
                percentage=0.0,
                message="Starting WASM execution"
            )

            # Get WASM path if available
            wasm_path = None
            if module_registry:
                wasm_path = module_registry.get_wasm_path(job.module_id)

            tracker.progress.step_name = "Executing"
            tracker.progress.percentage = 33.0
            tracker.progress.message = f"Running module {job.module_id}"

            # Run in thread pool (CPU-bound WASM execution)
            result = await asyncio.to_thread(
                wasm_runner.run,
                module_id=job.module_id,
                entry_function=job.entry_function,
                input_json=job.input_json,
                wasm_path=wasm_path,
                wasm_base64=job.wasm_base64,
                timeout_seconds=job.timeout_seconds,
            )

            # Final cancellation check after execution
            if tracker.cancelled:
                tracker.status = JobStatus.FAILED
                tracker.completed_at = datetime.utcnow()
                tracker.result = JobResult(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error="Job cancelled during execution",
                    node_id=config.node_id if config else None,
                )
                _jobs[job.job_id] = tracker.result
                return

            # Update result with job info
            result.job_id = job.job_id
            result.node_id = config.node_id if config else None

            # Store result in tracker and legacy dict
            tracker.result = result
            tracker.status = result.status
            tracker.completed_at = datetime.utcnow()
            tracker.progress = JobProgress(
                current_step=3,
                total_steps=3,
                step_name="Complete",
                percentage=100.0,
                message="Job finished successfully"
            )
            _jobs[job.job_id] = result  # Backward compat with GET /jobs/{id}

            logger.info(
                f"Async job {job.job_id} completed: {result.status}",
                extra={"job_id": job.job_id}
            )

    except asyncio.CancelledError:
        # Handle task cancellation (from timeout or manual cancel)
        if not tracker.completed_at:  # Only if not already marked
            tracker.status = JobStatus.FAILED
            tracker.completed_at = datetime.utcnow()
            tracker.cancelled = True
            tracker.result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error="Job was cancelled",
                node_id=config.node_id if config else None,
            )
            _jobs[job.job_id] = tracker.result
            logger.info(f"Job {job.job_id} cancelled", extra={"job_id": job.job_id})
        raise  # Re-raise to properly complete the task

    except Exception as e:
        tracker.status = JobStatus.FAILED
        tracker.completed_at = datetime.utcnow()
        tracker.result = JobResult(
            job_id=job.job_id,
            status=JobStatus.FAILED,
            error=str(e),
            node_id=config.node_id if config else None,
        )
        _jobs[job.job_id] = tracker.result

        logger.error(
            f"Async job {job.job_id} failed: {e}",
            extra={"job_id": job.job_id}
        )


class AsyncJobResponse(BaseModel):
    """Response for async job submission."""
    job_id: str
    status: str
    message: str


class JobProgressResponse(BaseModel):
    """Progress information for a running job."""
    current_step: int
    total_steps: int
    step_name: str
    percentage: float
    message: str


class JobStatusWithProgress(BaseModel):
    """Extended job status with progress tracking."""
    job_id: str
    status: str
    progress: Optional[JobProgressResponse] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


class CancelJobResponse(BaseModel):
    """Response for job cancellation."""
    job_id: str
    cancelled: bool
    message: str
    previous_status: str


class BatchJobRequest(BaseModel):
    """Request for batch job submission."""
    jobs: List[JobRequest]


class BatchJobResponse(BaseModel):
    """Response for batch job submission."""
    submitted: int
    failed: int
    job_ids: List[str]
    errors: List[str]


@router.post("/jobs/async", response_model=AsyncJobResponse)
async def submit_job_async(
    job: JobRequest,
    _: str = Depends(verify_secret),
):
    """
    Submit a job for async execution.

    Returns immediately with job_id. Use GET /jobs/{job_id} to poll status.
    Target: <100ms response time.
    """
    if not wasm_runner:
        raise HTTPException(status_code=503, detail="WASM runner not initialized")

    # Check if module exists (unless base64 provided)
    if not job.wasm_base64:
        if module_registry and not module_registry.exists(job.module_id):
            raise HTTPException(
                status_code=404,
                detail=f"Module {job.module_id} not found"
            )

    # Create tracker with PENDING status
    tracker = AsyncJobTracker(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
    )
    _async_jobs[job.job_id] = tracker

    # Schedule background execution (non-blocking)
    task = asyncio.create_task(_execute_job_async(job, tracker))
    tracker.task = task

    logger.info(
        f"Async job {job.job_id} submitted for module {job.module_id}",
        extra={"job_id": job.job_id}
    )

    return AsyncJobResponse(
        job_id=job.job_id,
        status="pending",
        message="Job queued for execution"
    )


@router.post("/jobs/batch", response_model=BatchJobResponse)
async def submit_batch_jobs(
    batch: BatchJobRequest,
    _: str = Depends(verify_secret),
):
    """
    Submit multiple jobs for async execution.

    Returns immediately with job_ids for all submitted jobs.
    Use GET /jobs/{job_id} to poll individual job status.
    """
    if not wasm_runner:
        raise HTTPException(status_code=503, detail="WASM runner not initialized")

    submitted_ids = []
    errors = []

    for job in batch.jobs:
        try:
            # Validate module exists (unless base64 provided)
            if not job.wasm_base64:
                if module_registry and not module_registry.exists(job.module_id):
                    errors.append(f"Module {job.module_id} not found for job {job.job_id}")
                    continue

            # Create tracker with PENDING status
            tracker = AsyncJobTracker(
                job_id=job.job_id,
                status=JobStatus.PENDING,
                submitted_at=datetime.utcnow(),
                timeout_seconds=job.timeout_seconds,
            )
            _async_jobs[job.job_id] = tracker

            # Schedule background execution (non-blocking)
            task = asyncio.create_task(_execute_job_async(job, tracker))
            tracker.task = task

            submitted_ids.append(job.job_id)

        except Exception as e:
            errors.append(f"Failed to submit job {job.job_id}: {str(e)}")

    logger.info(f"Batch submitted: {len(submitted_ids)} jobs, {len(errors)} errors")

    return BatchJobResponse(
        submitted=len(submitted_ids),
        failed=len(errors),
        job_ids=submitted_ids,
        errors=errors
    )


@router.post("/jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(
    job_id: str,
    _: str = Depends(verify_secret),
):
    """
    Cancel a running or pending job.

    For running jobs, attempts to cancel the asyncio task.
    For pending jobs, marks as cancelled before execution starts.
    """
    if job_id not in _async_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    tracker = _async_jobs[job_id]
    previous_status = tracker.status.value if hasattr(tracker.status, 'value') else str(tracker.status)

    # Already completed/failed - can't cancel
    if tracker.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return CancelJobResponse(
            job_id=job_id,
            cancelled=False,
            message=f"Job already {previous_status}, cannot cancel",
            previous_status=previous_status
        )

    # Mark as cancelled
    tracker.cancelled = True

    # If running, cancel the task
    if tracker.task and not tracker.task.done():
        tracker.task.cancel()

    # Update status
    tracker.status = JobStatus.FAILED
    tracker.completed_at = datetime.utcnow()
    tracker.result = JobResult(
        job_id=job_id,
        status=JobStatus.FAILED,
        error="Job cancelled by user",
        node_id=config.node_id if config else None,
    )
    _jobs[job_id] = tracker.result

    logger.info(f"Job {job_id} cancelled by user", extra={"job_id": job_id})

    return CancelJobResponse(
        job_id=job_id,
        cancelled=True,
        message="Job cancelled successfully",
        previous_status=previous_status
    )


@router.get("/jobs/{job_id}/progress", response_model=JobStatusWithProgress)
async def get_job_progress(
    job_id: str,
    _: str = Depends(verify_secret),
):
    """
    Get detailed job progress including step information.

    Provides more detailed status than GET /jobs/{job_id}.
    """
    if job_id not in _async_jobs:
        # Check legacy jobs
        if job_id in _jobs:
            result = _jobs[job_id]
            return JobStatusWithProgress(
                job_id=job_id,
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                error=result.error if hasattr(result, 'error') else None
            )
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    tracker = _async_jobs[job_id]
    now = datetime.utcnow()

    # Calculate elapsed time
    elapsed = None
    if tracker.started_at:
        if tracker.completed_at:
            elapsed = (tracker.completed_at - tracker.started_at).total_seconds()
        else:
            elapsed = (now - tracker.started_at).total_seconds()

    # Build progress response
    progress = None
    if tracker.progress:
        progress = JobProgressResponse(
            current_step=tracker.progress.current_step,
            total_steps=tracker.progress.total_steps,
            step_name=tracker.progress.step_name,
            percentage=tracker.progress.percentage,
            message=tracker.progress.message
        )

    error = None
    if tracker.result and tracker.result.error:
        error = tracker.result.error

    return JobStatusWithProgress(
        job_id=job_id,
        status=tracker.status.value if hasattr(tracker.status, 'value') else str(tracker.status),
        progress=progress,
        elapsed_seconds=elapsed,
        error=error
    )


@router.get("/modules", response_model=List[ModuleManifest])
async def list_modules(_: str = Depends(verify_secret)):
    """List available WASM modules."""
    if not module_registry:
        raise HTTPException(status_code=503, detail="Module registry not initialized")

    return module_registry.list()


@router.post("/modules/reload")
async def reload_modules(_: str = Depends(verify_secret)):
    """Reload modules from disk."""
    if not module_registry:
        raise HTTPException(status_code=503, detail="Module registry not initialized")

    module_registry.reload()
    return {"message": f"Reloaded {len(module_registry.list())} modules"}


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Node Daemon")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    parser.add_argument("--node-id", help="Override node ID")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply overrides
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.node_id:
        cfg.node_id = args.node_id

    # Create app with config
    application = create_app(cfg)

    # Run server
    uvicorn.run(
        application,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )