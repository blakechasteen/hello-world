"""
Command Handlers - Parse and execute Shuttle bot commands.

Commands:
    !loom status - Show Loom overview
    !loom recall <query> - Search HoloLoom memory
    !loom experience <text> - Store text to memory
    !loom weave <query> - Run full reasoning cycle (async)
    !status - List pending jobs
    !status <job_id> - Check job status/result with progress
    !cancel <job_id> - Cancel a running or pending job
    !nodes - List all nodes
    !job run <module> <json> - Run a WASM job
    !modules - List available WASM modules
    !help - Show help

Async Job Execution:
    Long-running operations (like !loom weave) now execute asynchronously.
    Jobs return immediately with a job_id (<100ms response).
    Use !status <job_id> to poll for completion and retrieve results.
    Use !cancel <job_id> to cancel a job that's taking too long.

Features:
    - Job cleanup: Old completed/failed jobs expire after 1 hour
    - Timeout handling: Jobs exceeding timeout are auto-cancelled
    - Progress tracking: Status shows step-by-step progress bars
    - Cancellation: Cancel jobs at any point via !cancel command
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

from ..hololoom_bridge import HoloLoomBridge
from ..shared.logging import get_logger
from ..shared.types import LoomStatus, NodeCapabilities, NodeRecord, NodeStatus
from .scheduler import SimpleScheduler

logger = get_logger(__name__, component="shuttle")


@dataclass
class PendingJob:
    """Tracks a pending async job for status polling."""
    job_id: str
    node_id: str
    node_address: str
    query: str
    submitted_at: datetime
    command_type: str  # "weave", "recall", etc.


class CommandHandler:
    """
    Handles parsing and execution of bot commands.
    """

    def __init__(
        self,
        portal_url: str,
        shared_secret: str,
        scheduler: SimpleScheduler,
        job_timeout: int = 60,
        hololoom_url: str | None = None,
    ):
        """
        Initialize command handler.

        Args:
            portal_url: Portal Server URL
            shared_secret: Shared secret for authentication
            scheduler: SimpleScheduler instance
            job_timeout: Default job timeout
            hololoom_url: HoloLoom API URL (optional)
        """
        self.portal_url = portal_url.rstrip("/")
        self.shared_secret = shared_secret
        self.scheduler = scheduler
        self.job_timeout = job_timeout
        self.hololoom = HoloLoomBridge(hololoom_url) if hololoom_url else None
        self._pending_jobs: dict[str, PendingJob] = {}

    async def handle(self, message: str) -> str:
        """
        Parse and execute a command.

        Args:
            message: Full message text (including prefix)

        Returns:
            Response message
        """
        # Remove prefix and split
        parts = message.strip().split(maxsplit=2)
        if not parts:
            return "Unknown command. Try !help"

        cmd = parts[0].lower().lstrip("!")

        if cmd == "help":
            return self._help()

        elif cmd == "status":
            if len(parts) < 2:
                return self._list_pending_jobs()
            return await self._check_job_status(parts[1])

        elif cmd == "cancel":
            if len(parts) < 2:
                return "Usage: !cancel <job_id>"
            return await self._cancel_job(parts[1])

        elif cmd == "loom":
            if not self.hololoom:
                return "HoloLoom integration not configured. Contact admin."

            if len(parts) < 2:
                return "Usage: !loom <status|recall|experience|weave>"

            subcommand = parts[1].lower()

            if subcommand == "status":
                return await self._loom_status()

            elif subcommand == "recall":
                if len(parts) < 3:
                    return "Usage: !loom recall <query>"
                query = parts[2] if len(parts) == 3 else " ".join(parts[2:])
                return await self._handle_loom_recall(query)

            elif subcommand == "experience":
                if len(parts) < 3:
                    return "Usage: !loom experience <text>"
                text = parts[2] if len(parts) == 3 else " ".join(parts[2:])
                return await self._handle_loom_experience(text)

            elif subcommand == "weave":
                if len(parts) < 3:
                    return "Usage: !loom weave <query>"
                query = parts[2] if len(parts) == 3 else " ".join(parts[2:])
                return await self._handle_loom_weave(query)

            else:
                return f"Unknown loom command: {subcommand}. Usage: !loom <status|recall|experience|weave>"

        elif cmd == "nodes":
            return await self._list_nodes()

        elif cmd == "modules":
            return await self._list_modules()

        elif cmd == "job":
            if len(parts) >= 2 and parts[1].lower() == "run":
                # Parse: !job run <module> <json>
                if len(parts) < 3:
                    return "Usage: !job run <module_id> <json>\nExample: !job run add {\"a\":2,\"b\":3}"

                # Extract module and JSON from remaining text
                remainder = parts[2]
                match = re.match(r"(\S+)\s+(.+)", remainder)
                if match:
                    module_id = match.group(1)
                    json_str = match.group(2)
                    return await self._run_job(module_id, json_str)
                else:
                    return "Usage: !job run <module_id> <json>"

            return "Usage: !job run <module_id> <json>"

        else:
            return f"Unknown command: {cmd}. Try !help"

    def _help(self) -> str:
        """Return help text."""
        return """**Shuttle Bot Commands**

**HoloLoom Memory** (requires HoloLoom configured)
**!loom status** - Show HoloLoom system status
**!loom recall <query>** - Search memory for relevant information
**!loom experience <text>** - Store text to long-term memory
**!loom weave <query>** - Run full reasoning cycle (async, use !status to check)

**Job Status & Control**
**!status** - List all pending jobs
**!status <job_id>** - Check job status/result with progress
**!cancel <job_id>** - Cancel a running or pending job

**Distributed Compute**
**!nodes** - List all registered nodes
**!modules** - List available WASM modules
**!job run <module> <json>** - Execute a WASM job
  Example: `!job run add {"a":2,"b":3}`

**Meta**
**!help** - Show this help

**About**
Shuttle is the ChatOps interface for your HoloLoom distributed compute Loom.
Jobs are dispatched to healthy nodes and results are returned here.
Long-running jobs (like !loom weave) run asynchronously - use !status to poll."""

    async def _loom_status(self) -> str:
        """Get and format Loom status."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.portal_url}/loom/status",
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    status = LoomStatus(**response.json())
                    return self._format_loom_status(status)
                else:
                    return f"Failed to get status: {response.status_code}"

        except Exception as e:
            logger.error(f"Failed to get Loom status: {e}")
            return f"Error: {e}"

    def _format_loom_status(self, status: LoomStatus) -> str:
        """Format LoomStatus for display."""
        uptime_hours = status.uptime_seconds / 3600

        return f"""**Loom Status: {status.loom_id}**

🖥️ **Nodes**: {status.nodes_online} online / {status.total_nodes} total
💾 **Resources**: {status.total_cpu_cores} CPU cores, {status.total_memory_mb // 1024}GB RAM
⚙️ **Jobs**: {status.jobs_in_progress} in progress
⏱️ **Uptime**: {uptime_hours:.1f} hours"""

    async def _list_nodes(self) -> str:
        """List all nodes."""
        nodes = await self.scheduler.get_nodes(online_only=False)

        if not nodes:
            return "No nodes registered."

        lines = ["**Registered Nodes**\n"]
        for node in nodes:
            status_emoji = "🟢" if node.status.value == "online" else "🔴"
            lines.append(
                f"{status_emoji} **{node.node_id}** ({node.address})\n"
                f"   CPU: {node.capabilities.cpu_cores} cores, "
                f"RAM: {node.capabilities.memory_mb // 1024}GB, "
                f"Status: {node.status.value}"
            )

        return "\n".join(lines)

    async def _list_modules(self) -> str:
        """List available WASM modules from a node."""
        node = await self.scheduler.pick_node()
        if not node:
            return "No healthy nodes available to query modules."

        try:
            address = node.address
            if not address.startswith("http"):
                address = f"http://{address}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{address}/modules",
                    headers={"X-Shared-Secret": self.shared_secret},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    modules = response.json()
                    if not modules:
                        return "No WASM modules loaded."

                    lines = [f"**Available Modules** (from {node.node_id})\n"]
                    for mod in modules:
                        desc = mod.get("description", "No description")
                        lines.append(f"• **{mod['id']}** v{mod.get('version', '?')} - {desc}")

                    return "\n".join(lines)
                else:
                    return f"Failed to get modules: {response.status_code}"

        except Exception as e:
            logger.error(f"Failed to get modules: {e}")
            return f"Error: {e}"

    async def _run_job(self, module_id: str, json_str: str) -> str:
        """Run a WASM job."""
        # Parse JSON input
        try:
            input_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        # Pick a node
        node = await self.scheduler.pick_node()
        if not node:
            return "No healthy nodes available to run job."

        # Dispatch job
        logger.info(f"Dispatching job to {node.node_id}: {module_id}")

        result = await self.scheduler.dispatch_job(
            node=node,
            module_id=module_id,
            input_json=input_json,
            timeout_seconds=self.job_timeout,
        )

        return self._format_job_result(result, node.node_id)

    def _format_job_result(self, result: dict, node_id: str) -> str:
        """Format job result for display."""
        status = result.get("status", "unknown")
        job_id = result.get("job_id", "?")

        if status == "completed":
            output = result.get("output_json", {})
            exec_time = result.get("execution_time_ms", 0)

            output_str = json.dumps(output, indent=2)
            if len(output_str) > 500:
                output_str = output_str[:500] + "\n... (truncated)"

            return f"""**Job Completed** ✅

**ID**: {job_id}
**Node**: {node_id}
**Time**: {exec_time:.2f}ms

**Output**:
```json
{output_str}
```"""

        else:
            error = result.get("error", "Unknown error")
            return f"""**Job Failed** ❌

**ID**: {job_id}
**Node**: {node_id}
**Status**: {status}
**Error**: {error}"""

    async def _handle_loom_recall(self, query: str) -> str:
        """
        Search HoloLoom memory for relevant information.

        Args:
            query: Search query

        Returns:
            Formatted response with top results
        """
        try:
            results = await self.hololoom.recall(query, k=5)

            if not results:
                return f"No memories found for: {query}"

            lines = [f"**Memory Search Results** for '{query}'\n"]
            for i, result in enumerate(results, 1):
                relevance = result.get("relevance", 0.0)
                text = result.get("text", "")
                # Truncate long texts
                if len(text) > 200:
                    text = text[:200] + "..."
                lines.append(f"{i}. [{relevance:.0%}] {text}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return f"Error searching memory: {e}"

    async def _handle_loom_experience(self, text: str) -> str:
        """
        Store text to HoloLoom long-term memory.

        Args:
            text: Text to store

        Returns:
            Confirmation message
        """
        try:
            memory_id = await self.hololoom.experience(text)

            return f"Memory stored successfully.\nID: `{memory_id}`\nText: {text[:100]}..."

        except Exception as e:
            logger.error(f"Experience failed: {e}")
            return f"Error storing memory: {e}"

    async def _handle_loom_weave(self, query: str) -> str:
        """
        Submit HoloLoom reasoning job for async execution.

        Returns immediately with job ID. Use !status <job_id> to check progress.

        Args:
            query: Query to reason about

        Returns:
            Job submission confirmation with ID
        """
        import uuid

        # Pick a healthy node
        node = await self.scheduler.pick_node()
        if not node:
            return "No healthy nodes available to run reasoning job."

        # Generate job ID
        job_id = f"weave-{str(uuid.uuid4())[:6]}"

        # Submit job asynchronously
        result = await self.scheduler.dispatch_job_async(
            node=node,
            module_id="hololoom",
            input_json={"query": query, "mode": "verify", "operation": "weave"},
            job_id=job_id,
            timeout_seconds=self.job_timeout,
        )

        if result.get("status") in ("pending", "running", "queued"):
            # Track pending job for status polling
            self._pending_jobs[job_id] = PendingJob(
                job_id=job_id,
                node_id=node.node_id,
                node_address=node.address,
                query=query,
                submitted_at=datetime.utcnow(),
                command_type="weave"
            )

            query_preview = query[:50] + "..." if len(query) > 50 else query

            return f"""**Job Submitted** ⏳

**ID**: `{job_id}`
**Node**: {node.node_id}
**Query**: {query_preview}

Use `!status {job_id}` to check progress."""

        else:
            error = result.get("error", "Unknown error")
            return f"""**Job Submission Failed** ❌

**Error**: {error}"""

    async def _check_job_status(self, job_id: str) -> str:
        """
        Check status of a pending job with progress information.

        Args:
            job_id: Job ID to check

        Returns:
            Formatted status or result
        """
        if job_id not in self._pending_jobs:
            return f"Unknown job: `{job_id}`\n\nUse `!status` to list pending jobs."

        pending = self._pending_jobs[job_id]

        # Create a minimal node record for polling

        node = NodeRecord(
            node_id=pending.node_id,
            address=pending.node_address,
            capabilities=NodeCapabilities(cpu_cores=1, memory_mb=1024, gpu=False, wasm_runtime="unknown"),
            status=NodeStatus.ONLINE
        )

        result = await self.scheduler.poll_job_status(node, job_id)
        status = result.get("status", "unknown")

        if status == "completed":
            # Job finished - remove from pending and format result
            del self._pending_jobs[job_id]
            return self._format_weave_result(pending, result)

        elif status == "failed":
            del self._pending_jobs[job_id]
            error = result.get("error", "Unknown error")
            return f"""**Job Failed** ❌

**ID**: `{job_id}`
**Query**: {pending.query[:50]}...
**Error**: {error}"""

        elif status == "not_found":
            del self._pending_jobs[job_id]
            return f"""**Job Not Found** ⚠

Job `{job_id}` is no longer tracked by node {pending.node_id}.
It may have expired or the node restarted."""

        else:
            # Still running - try to get progress info
            elapsed = (datetime.utcnow() - pending.submitted_at).total_seconds()

            # Fetch detailed progress
            progress = await self.scheduler.poll_job_progress(node, job_id)
            progress_info = ""

            if progress.get("status") != "error" and progress.get("progress"):
                p = progress["progress"]
                pct = p.get("percentage", 0)
                step_name = p.get("step_name", "Processing")
                current = p.get("current_step", 0)
                total = p.get("total_steps", 1)
                msg = p.get("message", "")

                progress_bar = self._format_progress_bar(pct)
                progress_info = f"""
**Progress**: {progress_bar} {pct:.0f}%
**Step**: {current}/{total} - {step_name}
{f"**Details**: {msg}" if msg else ""}"""

            return f"""**Job Running** ⏳

**ID**: `{job_id}`
**Status**: {status}
**Node**: {pending.node_id}
**Elapsed**: {elapsed:.0f}s
**Query**: {pending.query[:50]}...{progress_info}

Check again with `!status {job_id}` or cancel with `!cancel {job_id}`"""

    def _format_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Format a simple text progress bar."""
        filled = int(width * percentage / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    async def _cancel_job(self, job_id: str) -> str:
        """
        Cancel a running or pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            Cancellation result message
        """
        if job_id not in self._pending_jobs:
            return f"Unknown job: `{job_id}`\n\nUse `!status` to list pending jobs."

        pending = self._pending_jobs[job_id]

        # Create a minimal node record for cancellation

        node = NodeRecord(
            node_id=pending.node_id,
            address=pending.node_address,
            capabilities=NodeCapabilities(cpu_cores=1, memory_mb=1024, gpu=False, wasm_runtime="unknown"),
            status=NodeStatus.ONLINE
        )

        result = await self.scheduler.cancel_job(node, job_id)

        if result.get("cancelled"):
            # Remove from pending jobs
            del self._pending_jobs[job_id]
            elapsed = (datetime.utcnow() - pending.submitted_at).total_seconds()

            return f"""**Job Cancelled** 🛑

**ID**: `{job_id}`
**Query**: {pending.query[:50]}...
**Runtime**: {elapsed:.1f}s before cancellation
**Previous Status**: {result.get('previous_status', 'unknown')}"""

        else:
            message = result.get("message", "Unknown error")
            return f"""**Cancellation Failed** ⚠

**ID**: `{job_id}`
**Reason**: {message}

The job may have already completed or failed."""

    def _format_weave_result(self, pending: PendingJob, result: dict) -> str:
        """Format completed weave job result."""
        output = result.get("output_json", {})
        response = output.get("response", "No response")
        confidence = output.get("confidence", 0.0)
        steps = output.get("steps_taken", 0)
        verified = output.get("verified", False)
        exec_time = result.get("execution_time_ms", 0)

        # Truncate long responses
        if len(response) > 500:
            response = response[:500] + "\n... (truncated)"

        verify_status = "✓ Verified" if verified else "⚠ Unverified"
        elapsed = (datetime.utcnow() - pending.submitted_at).total_seconds()

        return f"""**Reasoning Complete** ✅

**Query**: {pending.query}

**Response**:
{response}

**Metrics**:
Confidence: {confidence:.0%}
Steps: {steps}
Status: {verify_status}
Execution: {exec_time:.0f}ms
Total Time: {elapsed:.1f}s"""

    def _list_pending_jobs(self) -> str:
        """List all pending jobs."""
        if not self._pending_jobs:
            return "No pending jobs."

        lines = ["**Pending Jobs**\n"]
        for job_id, pending in self._pending_jobs.items():
            elapsed = (datetime.utcnow() - pending.submitted_at).total_seconds()
            query_preview = pending.query[:30] + "..." if len(pending.query) > 30 else pending.query
            lines.append(f"• `{job_id}` ({pending.command_type}) - {elapsed:.0f}s - {query_preview}")

        lines.append("\nUse `!status <job_id>` to check a specific job.")
        return "\n".join(lines)
