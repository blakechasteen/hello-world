#!/usr/bin/env python3
"""
HoloLoom Matrix Bot Handlers
=============================
Command handlers that integrate Matrix bot with HoloLoom weaving shuttle.

Commands:
- !weave <query> - Execute query through full weaving cycle with reflection
- !memory <action> - Manage HoloLoom memory
- !trace <query_id> - Show Spacetime trace for a query
- !learn - Trigger learning analysis
- !stats - Show system statistics including reflection metrics
- !analyze <text> - Analyze with convergence engine
- !help - Show command help
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    pass

# Import HoloLoom components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.loom.command import PatternCard

# Handler registry (decorator-based registration)
from HoloLoom.chatops.handlers.handler_registry import (
    HandlerRegistry, HandlerCategory, chatops_handler
)

# WebSocket progress streaming (optional, graceful degradation)
try:
    from HoloLoom.chatops.handlers.websocket_progress import (
        JobProgressBroadcaster, JobProgressManager, get_global_broadcaster
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    JobProgressBroadcaster = None
    JobProgressManager = None

# Prometheus metrics (optional, graceful degradation)
try:
    from HoloLoom.chatops.handlers.prometheus_metrics import (
        JobMetricsCollector, get_metrics_collector
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    JobMetricsCollector = None
    get_metrics_collector = None

# Alignment framework imports (graceful degradation if unavailable)
try:
    from HoloLoom.alignment import create_guardrails, create_audit_trail
    from HoloLoom.alignment.safety_guardrails import ActionRequest, ActionCategory
    from HoloLoom.alignment.audit_trail import DecisionType, OutcomeType
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
    create_guardrails = None
    create_audit_trail = None

# Scratchpad auto-storage (graceful degradation if unavailable)
try:
    from HoloLoom.chatops.handlers.conversation_handlers import record_weave_result
    SCRATCHPAD_RECORDING_AVAILABLE = True
except ImportError:
    SCRATCHPAD_RECORDING_AVAILABLE = False
    record_weave_result = None

logger = logging.getLogger(__name__)


# ============================================================================
# Async Job Tracking (Moonshot Pattern)
# ============================================================================

class WeaveJobStatus(Enum):
    """Status of an async weave job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PendingWeaveJob:
    """Tracks a pending async weave job."""
    job_id: str
    query: str
    status: WeaveJobStatus
    room_id: str
    user_id: str
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    spacetime: Optional[Any] = None  # Result when completed
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    progress_step: int = 0
    progress_total: int = 9  # 9-step weaving cycle
    progress_message: str = "Queued"


class HoloLoomMatrixHandlers:
    """
    Handler class for Matrix bot commands that use HoloLoom.

    Integrates Matrix chatops with:
    - Weaving shuttle (full 9-step weaving cycle)
    - Reflection loop (continuous learning)
    - Memory management (add, search, stats)
    - Analytics and monitoring
    """

    def __init__(self, bot, config_mode: str = "fast", memory_shards: Optional[list] = None, enable_alignment: bool = True):
        """
        Initialize handlers with HoloLoom weaving shuttle.

        Args:
            bot: MatrixBot instance
            config_mode: HoloLoom config (bare/fast/fused)
            memory_shards: Optional initial memory shards
            enable_alignment: Enable SafetyGuardrails and AuditTrail integration
        """
        self.bot = bot

        # Initialize HoloLoom weaving shuttle
        logger.info(f"Initializing HoloLoom WeavingShuttle (mode={config_mode})...")

        # Create config
        if config_mode == "bare":
            config = Config.bare()
        elif config_mode == "fused":
            config = Config.fused()
        else:
            config = Config.fast()

        # Create initial memory shards if not provided
        if memory_shards is None:
            memory_shards = [
                MemoryShard(
                    id="welcome",
                    text="HoloLoom is a neural decision-making system with a complete weaving architecture.",
                    episode="system",
                    entities=["HoloLoom", "weaving", "architecture"],
                    motifs=["SYSTEM", "INFO"]
                )
            ]

        # Initialize shuttle with reflection enabled
        self.shuttle = WeavingShuttle(
            cfg=config,
            shards=memory_shards,
            enable_reflection=True,
            reflection_capacity=1000
        )

        # Initialize alignment framework (SafetyGuardrails + AuditTrail)
        if enable_alignment and ALIGNMENT_AVAILABLE:
            self.guardrails = create_guardrails(enable_adversarial_detection=True)
            self.audit = create_audit_trail()
            logger.info("Alignment framework enabled (SafetyGuardrails + AuditTrail)")
        else:
            self.guardrails = None
            self.audit = None
            if enable_alignment and not ALIGNMENT_AVAILABLE:
                logger.warning("Alignment requested but not available - running without safety gating")

        # Track spacetime artifacts by query hash
        self.spacetime_history: Dict[str, Any] = {}

        # Async job tracking (Moonshot pattern)
        self._pending_jobs: Dict[str, PendingWeaveJob] = {}
        self._max_concurrent_jobs = 10  # Limit concurrent weave jobs

        # WebSocket progress broadcasting (optional)
        self._progress_broadcaster: Optional[JobProgressBroadcaster] = None
        if WEBSOCKET_AVAILABLE:
            try:
                self._progress_broadcaster = get_global_broadcaster()
                logger.info("WebSocket progress broadcasting enabled")
            except Exception as e:
                logger.debug(f"WebSocket broadcaster not initialized: {e}")

        # Prometheus metrics collection (optional)
        self._metrics_collector: Optional[JobMetricsCollector] = None
        if METRICS_AVAILABLE:
            try:
                self._metrics_collector = get_metrics_collector()
                logger.info("Prometheus metrics collection enabled")
            except Exception as e:
                logger.debug(f"Metrics collector not initialized: {e}")

        logger.info("HoloLoom handlers initialized with reflection loop (async jobs enabled)")

        # Run startup diagnostics (P0 fix: warn about missing components)
        self._startup_diagnostics = self._run_startup_diagnostics()

    def _run_startup_diagnostics(self) -> Dict[str, Any]:
        """
        Run startup diagnostics and warn about missing optional components.

        Returns:
            Dict with component status and warnings for display via !status
        """
        diagnostics = {
            "components": {},
            "warnings": [],
            "summary": ""
        }

        # Check Alignment Framework
        if ALIGNMENT_AVAILABLE and self.guardrails:
            diagnostics["components"]["alignment"] = {"status": "✓ ACTIVE", "description": "SafetyGuardrails + AuditTrail"}
        elif ALIGNMENT_AVAILABLE:
            diagnostics["components"]["alignment"] = {"status": "⚠ DISABLED", "description": "Available but disabled"}
            diagnostics["warnings"].append("Alignment framework available but disabled - actions not safety-gated")
        else:
            diagnostics["components"]["alignment"] = {"status": "✗ MISSING", "description": "Not installed"}
            diagnostics["warnings"].append("⚠ SAFETY: Alignment framework not available - running WITHOUT safety guardrails")
            logger.warning("STARTUP: Alignment framework not available - actions will NOT be safety-gated")

        # Check WebSocket Progress Streaming
        if WEBSOCKET_AVAILABLE and self._progress_broadcaster:
            diagnostics["components"]["websocket"] = {"status": "✓ ACTIVE", "description": "Real-time progress streaming"}
        elif WEBSOCKET_AVAILABLE:
            diagnostics["components"]["websocket"] = {"status": "⚠ PARTIAL", "description": "Available but broadcaster not initialized"}
            diagnostics["warnings"].append("WebSocket available but not initialized - no real-time job progress")
        else:
            diagnostics["components"]["websocket"] = {"status": "✗ MISSING", "description": "Not installed"}
            diagnostics["warnings"].append("WebSocket not available - job progress polling only via !status")
            logger.info("STARTUP: WebSocket progress not available - use !status <job_id> for progress")

        # Check Prometheus Metrics
        if METRICS_AVAILABLE and self._metrics_collector:
            diagnostics["components"]["metrics"] = {"status": "✓ ACTIVE", "description": "Prometheus metrics collection"}
        elif METRICS_AVAILABLE:
            diagnostics["components"]["metrics"] = {"status": "⚠ PARTIAL", "description": "Available but collector not initialized"}
            diagnostics["warnings"].append("Metrics available but not collecting - no observability")
        else:
            diagnostics["components"]["metrics"] = {"status": "✗ MISSING", "description": "Not installed"}
            diagnostics["warnings"].append("Prometheus metrics not available - limited observability")
            logger.info("STARTUP: Prometheus metrics not available - limited system observability")

        # Generate summary
        active_count = sum(1 for c in diagnostics["components"].values() if c["status"].startswith("✓"))
        total_count = len(diagnostics["components"])
        warning_count = len(diagnostics["warnings"])

        if warning_count == 0:
            diagnostics["summary"] = f"✓ All {total_count} optional components active"
            logger.info(f"STARTUP: All {total_count} optional components active")
        else:
            diagnostics["summary"] = f"⚠ {active_count}/{total_count} components active ({warning_count} warnings)"
            logger.warning(f"STARTUP: {active_count}/{total_count} optional components active - {warning_count} warnings")
            for warning in diagnostics["warnings"]:
                if "SAFETY" in warning:
                    logger.warning(f"  {warning}")

        return diagnostics

    def get_startup_diagnostics(self) -> Dict[str, Any]:
        """Get startup diagnostics for display via !status or similar."""
        return self._startup_diagnostics

    # ========================================================================
    # Safety Gating
    # ========================================================================

    async def _gate_action(self, room, event, action: str, category: str, args: str) -> bool:
        """
        Gate action through SafetyGuardrails. Returns True if allowed.

        Args:
            room: Matrix room
            event: Matrix event
            action: Action name (e.g., "weave", "memory_store")
            category: ActionCategory value (e.g., "QUERY", "STORAGE")
            args: User input text

        Returns:
            True if action is allowed, False if blocked
        """
        if not self.guardrails:
            return True  # No guardrails = allow all

        try:
            # Create action request
            request = ActionRequest(
                action=action,
                category=ActionCategory[category] if hasattr(ActionCategory, category) else ActionCategory.QUERY,
                context={"args": args, "room": room.room_id},
                user_id=event.sender
            )

            # Evaluate through guardrails
            decision = self.guardrails.evaluate(request, text_input=args)

            # Log to audit trail
            if self.audit:
                self.audit.log_decision(
                    decision_type=DecisionType.SAFETY_GATE,
                    outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
                    reason=decision.reason,
                    query_text=args,
                    risk_level=decision.risk_level.value if hasattr(decision.risk_level, 'value') else str(decision.risk_level),
                    metadata={"action": action, "user": event.sender, "room": room.room_id}
                )

            if not decision.allowed:
                await self.bot.send_message(
                    room.room_id,
                    f"⚠️ **Action blocked** ({decision.risk_level.value if hasattr(decision.risk_level, 'value') else decision.risk_level}): {decision.reason}",
                    markdown=True
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Safety gating error: {e}")
            # Fail open with warning (allow action but log)
            return True

    # ========================================================================
    # WebSocket Progress Broadcasting
    # ========================================================================

    def set_progress_broadcaster(self, broadcaster: "JobProgressBroadcaster") -> None:
        """
        Set the WebSocket progress broadcaster for real-time job updates.

        Args:
            broadcaster: JobProgressBroadcaster instance
        """
        self._progress_broadcaster = broadcaster
        logger.info("Custom WebSocket progress broadcaster configured")

    def set_metrics_collector(self, collector: "JobMetricsCollector") -> None:
        """
        Set the Prometheus metrics collector for job observability.

        Args:
            collector: JobMetricsCollector instance
        """
        self._metrics_collector = collector
        logger.info("Custom Prometheus metrics collector configured")

    async def _broadcast_progress(
        self,
        job: PendingWeaveJob,
        current_step: int,
        step_name: str,
        message: str,
        **kwargs
    ) -> None:
        """
        Broadcast progress update if broadcaster is available.

        Args:
            job: The pending weave job
            current_step: Current step number (1-9)
            step_name: Name of current step
            message: Human-readable progress message
            **kwargs: Additional metadata
        """
        # Update job progress
        job.progress_step = current_step
        job.progress_message = message

        # Broadcast via WebSocket if available
        if self._progress_broadcaster:
            try:
                await self._progress_broadcaster.job_progress(
                    job_id=job.job_id,
                    current_step=current_step,
                    step_name=step_name,
                    message=message,
                    room_id=job.room_id,
                    user_id=job.user_id,
                    query=job.query,
                    metadata=kwargs
                )
            except Exception as e:
                logger.debug(f"Progress broadcast failed: {e}")

    # ========================================================================
    # Async Weave Execution (Moonshot Pattern)
    # ========================================================================

    async def _execute_weave_async(self, job: PendingWeaveJob):
        """
        Execute weave in background. Updates job status and sends result to room.

        This runs as a background task, not blocking the command handler.
        Broadcasts progress updates via WebSocket at each step.
        """
        try:
            # Update status to running
            job.status = WeaveJobStatus.RUNNING
            job.started_at = datetime.now()

            # Step 1: Initializing
            await self._broadcast_progress(job, 1, "Initializing", "Starting weaving cycle...")

            # Broadcast job_started
            if self._progress_broadcaster:
                await self._progress_broadcaster.job_started(
                    job_id=job.job_id,
                    query=job.query,
                    room_id=job.room_id,
                    user_id=job.user_id
                )

            # Record metrics for job start
            if self._metrics_collector:
                self._metrics_collector.job_started(job.job_id)

            # Step 2: Pattern selection
            await self._broadcast_progress(job, 2, "Pattern Selection", "Selecting weaving pattern...")

            # Step 3: Feature extraction
            await self._broadcast_progress(job, 3, "Feature Extraction", "Extracting features...")

            # Execute weaving cycle with reflection
            query = Query(text=job.query)

            # Step 4: Memory retrieval
            await self._broadcast_progress(job, 4, "Memory Retrieval", "Searching knowledge graph...")

            # Step 5: Warp space tensioning
            await self._broadcast_progress(job, 5, "Warp Space", "Tensioning warp threads...")

            # Step 6: Decision making
            await self._broadcast_progress(job, 6, "Convergence", "Collapsing to decision...")

            spacetime = await self.shuttle.weave_and_reflect(
                query,
                feedback={"source": "matrix", "user": job.user_id, "room": job.room_id}
            )

            # Step 7: Tool execution
            await self._broadcast_progress(job, 7, "Execution", f"Executing {spacetime.tool_used}...")

            # Step 8: Spacetime weaving
            await self._broadcast_progress(job, 8, "Spacetime", "Weaving Spacetime fabric...")

            # Store result
            job.spacetime = spacetime
            job.status = WeaveJobStatus.COMPLETED
            job.completed_at = datetime.now()

            # Step 9: Complete
            await self._broadcast_progress(job, 9, "Complete", "Weaving cycle complete")

            # Store spacetime for trace command
            query_id = f"{job.user_id}_{datetime.now().timestamp()}"
            self.spacetime_history[query_id] = spacetime

            # Record weave result for scratch pad integration
            # This enables !scratch store and auto-storage of artifacts
            if SCRATCHPAD_RECORDING_AVAILABLE and record_weave_result:
                try:
                    await record_weave_result(
                        user_id=job.user_id,
                        room_id=job.room_id,
                        spacetime=spacetime,
                        auto_store=False  # User can explicitly store via !scratch store
                    )
                except Exception as e:
                    logger.warning(f"Failed to record weave result for scratchpad: {e}")

            # Broadcast job_completed via WebSocket
            duration = (job.completed_at - job.submitted_at).total_seconds()
            if self._progress_broadcaster:
                await self._progress_broadcaster.job_completed(
                    job_id=job.job_id,
                    tool_used=spacetime.tool_used,
                    confidence=spacetime.confidence,
                    duration_ms=spacetime.trace.duration_ms,
                    response_preview=spacetime.response[:500] if spacetime.response else "",
                    room_id=job.room_id,
                    user_id=job.user_id,
                    query=job.query,
                    metadata={
                        "context_shards": spacetime.trace.context_shards_count,
                        "motifs_count": len(spacetime.trace.motifs_detected),
                        "threads_activated": len(spacetime.trace.threads_activated)
                    }
                )

            # Record metrics for job completion
            if self._metrics_collector:
                self._metrics_collector.job_completed(
                    job_id=job.job_id,
                    duration_ms=spacetime.trace.duration_ms,
                    confidence=spacetime.confidence,
                    tool_used=spacetime.tool_used
                )

            # Send completion notification to room
            response = f"""
**✨ Weaving Complete** (`{job.job_id}`)

**Query:** {job.query[:100]}

**Decision:**
• Tool: `{spacetime.tool_used}`
• Confidence: {spacetime.confidence:.1%}
• Duration: {spacetime.trace.duration_ms:.0f}ms (total: {duration:.1f}s)

**Context:**
• Shards retrieved: {spacetime.trace.context_shards_count}
• Motifs detected: {len(spacetime.trace.motifs_detected)}
• Embedding scales: {spacetime.trace.embedding_scales_used}
• Threads activated: {len(spacetime.trace.threads_activated)}

**Result:**
{spacetime.response[:500] if spacetime.response else 'No output'}

*React with 👍/👎/⭐ to provide feedback*
*Use `!trace {query_id}` for full Spacetime trace*
"""

            await self.bot.send_message(job.room_id, response, markdown=True)

        except asyncio.CancelledError:
            job.status = WeaveJobStatus.CANCELLED
            job.completed_at = datetime.now()
            job.error = "Cancelled by user"
            job.progress_message = "Cancelled"

            # Broadcast job_cancelled via WebSocket
            if self._progress_broadcaster:
                await self._progress_broadcaster.job_cancelled(
                    job_id=job.job_id,
                    room_id=job.room_id,
                    user_id=job.user_id
                )

            # Record metrics for job cancellation
            if self._metrics_collector:
                self._metrics_collector.job_cancelled(job.job_id)

            await self.bot.send_message(
                job.room_id,
                f"🚫 **Job cancelled:** `{job.job_id}`",
                markdown=True
            )
            raise

        except Exception as e:
            logger.error(f"Weave job {job.job_id} failed: {e}", exc_info=True)
            job.status = WeaveJobStatus.FAILED
            job.completed_at = datetime.now()
            job.error = str(e)
            job.progress_message = f"Failed: {str(e)[:50]}"

            # Broadcast job_failed via WebSocket
            if self._progress_broadcaster:
                await self._progress_broadcaster.job_failed(
                    job_id=job.job_id,
                    error=str(e),
                    room_id=job.room_id,
                    user_id=job.user_id,
                    query=job.query
                )

            # Record metrics for job failure
            if self._metrics_collector:
                self._metrics_collector.job_failed(job.job_id, str(e))

            await self.bot.send_message(
                job.room_id,
                f"❌ **Job failed** (`{job.job_id}`): {str(e)}",
                markdown=True
            )

    def _format_progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Format a text-based progress bar."""
        filled = int(width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (width - filled)
        percent = int(100 * current / total) if total > 0 else 0
        return f"[{bar}] {percent}%"

    @chatops_handler(
        command="weave",
        description="Execute query through full weaving cycle with reflection (async)",
        usage="!weave <your query>",
        category=HandlerCategory.QUERY,
        aliases=["w"]
    )
    async def handle_weave(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Execute query through full weaving cycle with reflection.

        **Async Pattern:** Returns immediately with a job ID. Use `!status <job_id>`
        to check progress or wait for the completion notification.

        Usage: !weave What is Thompson Sampling?

        Supports reactions for feedback:
        - 👍 = helpful (positive feedback)
        - 👎 = not helpful (negative feedback)
        - ⭐ = excellent (high reward)
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                "Usage: `!weave <your query>`\n\nExample: `!weave Explain Thompson Sampling`",
                markdown=True
            )
            return

        # Safety gate check
        if not await self._gate_action(room, event, "weave", "QUERY", args):
            return

        # Check concurrent job limit
        running_jobs = sum(1 for j in self._pending_jobs.values()
                          if j.status in (WeaveJobStatus.PENDING, WeaveJobStatus.RUNNING))
        if running_jobs >= self._max_concurrent_jobs:
            await self.bot.send_message(
                room.room_id,
                f"⚠️ **Too many concurrent jobs** ({running_jobs}/{self._max_concurrent_jobs}). "
                f"Please wait for some to complete or use `!status` to check progress.",
                markdown=True
            )
            return

        # Generate job ID
        job_id = f"weave-{str(uuid.uuid4())[:6]}"

        # Create pending job
        job = PendingWeaveJob(
            job_id=job_id,
            query=args,
            status=WeaveJobStatus.PENDING,
            room_id=room.room_id,
            user_id=event.sender,
            submitted_at=datetime.now(),
            progress_message="Queued"
        )
        self._pending_jobs[job_id] = job

        # Broadcast job_queued via WebSocket
        if self._progress_broadcaster:
            await self._progress_broadcaster.job_queued(
                job_id=job_id,
                query=args,
                room_id=room.room_id,
                user_id=event.sender
            )

        # Record metrics for job submission
        if self._metrics_collector:
            self._metrics_collector.job_submitted(
                job_id=job_id,
                query=args,
                room_id=room.room_id,
                user_id=event.sender
            )

        # Start background execution
        task = asyncio.create_task(self._execute_weave_async(job))
        job.task = task

        # Return immediately with job ID (<100ms response)
        await self.bot.send_message(
            room.room_id,
            f"""
🔮 **Job Submitted** (`{job_id}`)

**Query:** `{args[:80]}{'...' if len(args) > 80 else ''}`

Processing in background. You'll receive a notification when complete.

• Use `!status {job_id}` to check progress
• Use `!cancel {job_id}` to cancel
• Use `!status` to list all pending jobs
""",
            markdown=True
        )

    @chatops_handler(
        command="status",
        description="Check status of async weave jobs",
        usage="!status [job_id]",
        category=HandlerCategory.SYSTEM,
        aliases=["s", "jobs"]
    )
    async def handle_status(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Check status of async weave jobs.

        Usage:
            !status           - List all pending/running jobs
            !status <job_id>  - Check specific job status
        """
        job_id = args.strip() if args else None

        if job_id:
            # Check specific job
            if job_id not in self._pending_jobs:
                await self.bot.send_message(
                    room.room_id,
                    f"❓ Unknown job: `{job_id}`\n\nUse `!status` to list all jobs.",
                    markdown=True
                )
                return

            job = self._pending_jobs[job_id]

            # Calculate elapsed time
            elapsed = (datetime.now() - job.submitted_at).total_seconds()

            # Status icon
            status_icons = {
                WeaveJobStatus.PENDING: "⏳",
                WeaveJobStatus.RUNNING: "🔄",
                WeaveJobStatus.COMPLETED: "✅",
                WeaveJobStatus.FAILED: "❌",
                WeaveJobStatus.CANCELLED: "🚫"
            }
            icon = status_icons.get(job.status, "❓")

            # Format progress bar
            progress_bar = self._format_progress_bar(job.progress_step, job.progress_total)

            response = f"""
**{icon} Job Status: `{job_id}`**

**Status:** {job.status.value.title()}
**Query:** `{job.query[:60]}{'...' if len(job.query) > 60 else ''}`

**Progress:** {progress_bar}
**Step:** {job.progress_step}/{job.progress_total} - {job.progress_message}

**Timing:**
• Submitted: {job.submitted_at.strftime('%H:%M:%S')}
• Elapsed: {elapsed:.1f}s
"""

            if job.status == WeaveJobStatus.COMPLETED and job.spacetime:
                response += f"""
**Result:**
• Tool: `{job.spacetime.tool_used}`
• Confidence: {job.spacetime.confidence:.1%}
• Duration: {job.spacetime.trace.duration_ms:.0f}ms
"""

            if job.status == WeaveJobStatus.FAILED and job.error:
                response += f"\n**Error:** {job.error}"

            await self.bot.send_message(room.room_id, response, markdown=True)

        else:
            # List all jobs (filter to user's room or show all for admins)
            active_jobs = [
                j for j in self._pending_jobs.values()
                if j.status in (WeaveJobStatus.PENDING, WeaveJobStatus.RUNNING)
            ]
            recent_completed = [
                j for j in self._pending_jobs.values()
                if j.status in (WeaveJobStatus.COMPLETED, WeaveJobStatus.FAILED, WeaveJobStatus.CANCELLED)
                and (datetime.now() - (j.completed_at or j.submitted_at)).total_seconds() < 300  # Last 5 minutes
            ]

            if not active_jobs and not recent_completed:
                await self.bot.send_message(
                    room.room_id,
                    "📋 **No active jobs.**\n\nUse `!weave <query>` to submit a job.",
                    markdown=True
                )
                return

            response = "**📋 Job Status**\n\n"

            if active_jobs:
                response += "**Active Jobs:**\n"
                for job in active_jobs:
                    elapsed = (datetime.now() - job.submitted_at).total_seconds()
                    icon = "⏳" if job.status == WeaveJobStatus.PENDING else "🔄"
                    response += f"• {icon} `{job.job_id}` ({elapsed:.0f}s) - {job.query[:40]}...\n"
                response += "\n"

            if recent_completed:
                response += "**Recently Completed:**\n"
                for job in recent_completed[-5:]:  # Last 5
                    icon = "✅" if job.status == WeaveJobStatus.COMPLETED else "❌" if job.status == WeaveJobStatus.FAILED else "🚫"
                    duration = (job.completed_at - job.submitted_at).total_seconds() if job.completed_at else 0
                    response += f"• {icon} `{job.job_id}` ({duration:.1f}s) - {job.query[:40]}...\n"

            response += f"\n*Use `!status <job_id>` for details*"

            await self.bot.send_message(room.room_id, response, markdown=True)

    @chatops_handler(
        command="cancel",
        description="Cancel a running weave job",
        usage="!cancel <job_id>",
        category=HandlerCategory.SYSTEM
    )
    async def handle_cancel(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Cancel a running or pending weave job.

        Usage: !cancel <job_id>
        """
        job_id = args.strip() if args else None

        if not job_id:
            await self.bot.send_message(
                room.room_id,
                "Usage: `!cancel <job_id>`\n\nUse `!status` to list active jobs.",
                markdown=True
            )
            return

        if job_id not in self._pending_jobs:
            await self.bot.send_message(
                room.room_id,
                f"❓ Unknown job: `{job_id}`",
                markdown=True
            )
            return

        job = self._pending_jobs[job_id]

        # Check if job can be cancelled
        if job.status not in (WeaveJobStatus.PENDING, WeaveJobStatus.RUNNING):
            await self.bot.send_message(
                room.room_id,
                f"⚠️ Job `{job_id}` cannot be cancelled (status: {job.status.value})",
                markdown=True
            )
            return

        # Cancel the task
        if job.task and not job.task.done():
            job.task.cancel()
            try:
                await job.task
            except asyncio.CancelledError:
                pass

        job.status = WeaveJobStatus.CANCELLED
        job.completed_at = datetime.now()
        job.error = "Cancelled by user"
        job.progress_message = "Cancelled"

        elapsed = (job.completed_at - job.submitted_at).total_seconds()
        await self.bot.send_message(
            room.room_id,
            f"🚫 **Job cancelled:** `{job_id}` (ran for {elapsed:.1f}s)",
            markdown=True
        )

    @chatops_handler(
        command="memory",
        description="Manage HoloLoom memory (add, search, stats)",
        usage="!memory <add|search|stats> [args]",
        category=HandlerCategory.QUERY,
        aliases=["mem", "m"]
    )
    async def handle_memory(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Manage HoloLoom memory.

        Usage:
            !memory add <text>
            !memory search <query>
            !memory stats
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                """
**Memory Commands:**

• `!memory add <text>` - Add knowledge to memory
• `!memory search <query>` - Search memory
• `!memory stats` - Show memory statistics

**Example:**
```
!memory add MCTS uses Thompson Sampling
!memory search What is MCTS?
```
""",
                markdown=True
            )
            return

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        content = parts[1] if len(parts) > 1 else ""

        # Safety gate check (STORAGE for add, RETRIEVAL for search)
        category = "STORAGE" if action == "add" else "RETRIEVAL"
        if not await self._gate_action(room, event, f"memory_{action}", category, args):
            return

        try:
            if action == "add":
                if not content:
                    await self.bot.send_message(room.room_id, "Usage: `!memory add <text>`")
                    return

                # Add to memory (create new shard and add to yarn graph)
                new_shard = MemoryShard(
                    id=f"matrix_{datetime.now().timestamp()}",
                    text=content,
                    episode="matrix_chat",
                    entities=[],  # Could extract entities here
                    motifs=[],
                    metadata={"source": "matrix", "user": event.sender, "room": room.room_id}
                )
                self.shuttle.yarn_graph.shards[new_shard.id] = new_shard

                await self.bot.send_message(
                    room.room_id,
                    f"✅ **Added to memory**\n\n`{content[:200]}`",
                    markdown=True
                )

            elif action == "search":
                if not content:
                    await self.bot.send_message(room.room_id, "Usage: `!memory search <query>`")
                    return

                # Search memory using retriever
                query_obj = Query(text=content)
                results = await self.shuttle.retriever.retrieve(query_obj, limit=5)

                if not results:
                    await self.bot.send_message(room.room_id, "No results found")
                    return

                response = f"**🔍 Search Results** ({len(results)} found)\n\n"
                for i, shard in enumerate(results[:3], 1):
                    text = shard.text[:150] + "..." if len(shard.text) > 150 else shard.text
                    response += f"{i}. `{text}`\n\n"

                await self.bot.send_message(room.room_id, response, markdown=True)

            elif action == "stats":
                # Get memory stats
                total_shards = len(self.shuttle.yarn_graph.shards)
                reflection_metrics = self.shuttle.get_reflection_metrics()

                success_rate = reflection_metrics.get('success_rate', 0) if reflection_metrics else 0
                response = f"""
**📊 Memory Statistics**

**Yarn Graph:**
• Total threads (shards): {total_shards}
• Backend: In-memory (Yarn Graph)

**Reflection:**
• Total cycles: {reflection_metrics.get('total_cycles', 0) if reflection_metrics else 0}
• Success rate: {success_rate:.1%}

**System:**
• Mode: {self.shuttle.cfg.mode.value}
• Uptime: {datetime.now().isoformat()}
"""

                await self.bot.send_message(room.room_id, response, markdown=True)

            else:
                await self.bot.send_message(
                    room.room_id,
                    f"Unknown action: `{action}`. Try `!memory` for help."
                )

        except Exception as e:
            logger.error(f"Memory error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    @chatops_handler(
        command="analyze",
        description="Analyze text with convergence engine",
        usage="!analyze <text to analyze>",
        category=HandlerCategory.QUERY,
        aliases=["a"]
    )
    async def handle_analyze(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Analyze text with convergence engine.

        Usage: !analyze <text>
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                "Usage: `!analyze <text to analyze>`"
            )
            return

        # Safety gate check
        if not await self._gate_action(room, event, "analyze", "ANALYSIS", args):
            return

        try:
            await self.bot.send_typing(room.room_id, typing=True)

            # Weave analysis query
            query = Query(text=f"Analyze the following: {args}")
            spacetime = await self.shuttle.weave(query)

            response = f"""
**🔬 Analysis Results**

**Input:** {args[:200]}

**Decision:** {spacetime.tool_used} ({spacetime.confidence:.1%} confidence)

**Weaving:**
• Duration: {spacetime.trace.duration_ms:.0f}ms
• Context: {spacetime.trace.context_shards_count} shards
• Motifs: {len(spacetime.trace.motifs_detected)}
• Scales: {spacetime.trace.embedding_scales_used}
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Analyze error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    @chatops_handler(
        command="stats",
        description="Show system statistics including reflection metrics",
        usage="!stats",
        category=HandlerCategory.SYSTEM
    )
    async def handle_stats(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show HoloLoom system statistics including reflection metrics.

        Usage: !stats
        """
        # Safety gate check
        if not await self._gate_action(room, event, "stats", "QUERY", args or ""):
            return

        try:
            # Get reflection metrics
            reflection_metrics = self.shuttle.get_reflection_metrics()

            # Build reflection info
            reflection_info = ""
            if reflection_metrics:
                tool_success = reflection_metrics.get('tool_success_rates', {})
                tool_recs = reflection_metrics.get('tool_recommendations', [])

                reflection_info = f"""
**Reflection Loop:**
• Total cycles: {reflection_metrics.get('total_cycles', 0)}
• Success rate: {reflection_metrics.get('success_rate', 0):.1%}
• Learning status: ✅ Active

**Tool Performance:**
"""
                for tool, rate in list(tool_success.items())[:3]:
                    reflection_info += f"• {tool}: {rate:.1%}\n"

                if tool_recs:
                    reflection_info += f"\n**Recommended:** {', '.join(tool_recs[:3])}"

            # Get startup diagnostics for optional components
            diagnostics = self.get_startup_diagnostics()
            components_info = f"\n**Optional Components:** {diagnostics['summary']}\n"
            for name, info in diagnostics["components"].items():
                components_info += f"• {name.title()}: {info['status']} - {info['description']}\n"

            # Show warnings if any
            warnings_info = ""
            if diagnostics["warnings"]:
                warnings_info = "\n**⚠ Warnings:**\n"
                for warning in diagnostics["warnings"]:
                    warnings_info += f"• {warning}\n"

            response = f"""
**📈 HoloLoom Statistics**

**System:**
• Status: ✅ Operational
• Mode: {self.shuttle.cfg.mode.value}
• Reflection: {'Enabled' if self.shuttle.enable_reflection else 'Disabled'}

{reflection_info}
{components_info}{warnings_info}
**Architecture:**
• Full 9-step weaving cycle
• Thompson Sampling exploration
• Multi-scale embeddings
• Spacetime provenance tracking
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    @chatops_handler(
        command="trace",
        description="Show full Spacetime trace for a query",
        usage="!trace [query_id]",
        category=HandlerCategory.LEARNING,
        aliases=["t"]
    )
    async def handle_trace(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show full Spacetime trace for a query.

        Usage: !trace [query_id]
        Shows the most recent trace if no ID provided.
        """
        # Safety gate check
        if not await self._gate_action(room, event, "trace", "QUERY", args or ""):
            return

        try:
            # Get trace - either latest or by ID
            if args and args in self.spacetime_history:
                spacetime = self.spacetime_history[args]
            elif self.spacetime_history:
                # Get most recent
                spacetime = list(self.spacetime_history.values())[-1]
            else:
                await self.bot.send_message(
                    room.room_id,
                    "No traces available. Use `!weave` first."
                )
                return

            trace = spacetime.trace

            # Format detailed trace
            response = f"""
**🔍 Spacetime Trace**

**Query:** {spacetime.query_text}

**Weaving Cycle (9 steps):**
1. Pattern: `{trace.pattern_used}` selected
2. Temporal window created
3. Threads: {len(trace.threads_activated)} activated
4. Features extracted:
   • Motifs: {', '.join(trace.motifs_detected[:5])}
   • Scales: {trace.embedding_scales_used}
5. Warp space: Tensioned
6. Context: {trace.context_shards_count} shards retrieved
7. Convergence: `{spacetime.tool_used}` (confidence {spacetime.confidence:.1%})
8. Tool executed
9. Spacetime woven

**Stage Timings:**
"""
            for stage, duration in list(trace.stage_durations.items())[:6]:
                response += f"• {stage}: {duration:.0f}ms\n"

            response += f"\n**Total Duration:** {trace.duration_ms:.0f}ms"

            # Add bandit stats if available
            if hasattr(trace, 'bandit_statistics') and trace.bandit_statistics:
                response += "\n\n**Bandit Statistics:**\n"
                for tool, stats in list(trace.bandit_statistics.items())[:3]:
                    response += f"• {tool}: {stats}\n"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Trace error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    @chatops_handler(
        command="learn",
        description="Trigger learning analysis and show insights",
        usage="!learn [force]",
        category=HandlerCategory.LEARNING,
        aliases=["l"]
    )
    async def handle_learn(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Trigger learning analysis and show insights.

        Usage: !learn [force]
        Use 'force' to analyze even if not enough cycles have passed.
        """
        # Safety gate check (MODIFICATION since learning modifies system state)
        if not await self._gate_action(room, event, "learn", "MODIFICATION", args or ""):
            return

        try:
            if not self.shuttle.enable_reflection:
                await self.bot.send_message(
                    room.room_id,
                    "⚠️ Reflection loop is disabled. Enable it to use learning."
                )
                return

            await self.bot.send_typing(room.room_id, typing=True)

            # Analyze and generate learning signals
            force = args.lower() == "force"
            signals = await self.shuttle.learn(force=force)

            if not signals:
                await self.bot.send_message(
                    room.room_id,
                    "📚 No new learning signals yet. Need more cycles for analysis."
                )
                return

            # Format learning insights
            response = f"""
**🧠 Learning Analysis**

Generated {len(signals)} learning signals:

"""

            for signal in signals[:5]:  # Show top 5
                response += f"**{signal.signal_type}**\n"
                if signal.tool:
                    response += f"• Tool: `{signal.tool}`\n"
                if hasattr(signal, 'reward') and signal.reward is not None:
                    response += f"• Reward: {signal.reward:.2f}\n"
                if signal.pattern:
                    response += f"• Pattern: {signal.pattern}\n"
                if signal.recommendation:
                    response += f"• Action: {signal.recommendation}\n"
                response += "\n"

            response += f"*Applying {len(signals)} learning signals to improve system...*"

            await self.bot.send_message(room.room_id, response, markdown=True)

            # Apply signals
            await self.shuttle.apply_learning_signals(signals)

            await self.bot.send_message(
                room.room_id,
                "✅ **Learning complete!** System has adapted based on past performance.",
                markdown=True
            )

        except Exception as e:
            logger.error(f"Learn error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    @chatops_handler(
        command="help",
        description="Show help for HoloLoom commands",
        usage="!help [command]",
        category=HandlerCategory.SYSTEM,
        aliases=["h", "?"]
    )
    async def handle_help(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show help for HoloLoom commands.

        Usage: !help [command]

        If a command is specified, shows detailed help for that command.
        Otherwise shows all available commands grouped by category.
        """
        # Safety gate check (safe command, but log for audit)
        if not await self._gate_action(room, event, "help", "QUERY", args or ""):
            return

        registry = self.get_registry()

        if args and args.strip():
            # Show help for specific command
            command = args.strip().lstrip("!")
            command_help = registry.generate_help_for_command(command)

            if command_help:
                help_text = f"**🔮 HoloLoom Command Help**\n\n{command_help}"
            else:
                # Command not found - show available commands
                all_commands = registry.get_all_commands()
                help_text = f"❌ Unknown command: `!{command}`\n\n"
                help_text += f"**Available commands:** {', '.join(f'`!{c}`' for c in sorted(all_commands))}\n\n"
                help_text += "Use `!help` to see all commands with descriptions."
        else:
            # Show all commands from registry (auto-generated from metadata)
            help_text = f"""**🔮 HoloLoom Bot - Commands**

{registry.generate_help(prefix="!", show_aliases=True, show_usage=True)}

**Examples:**
```
!weave Explain Thompson Sampling  # Returns job ID immediately
!status weave-abc123              # Check specific job
!cancel weave-abc123              # Cancel a running job
!help weave                       # Detailed help for weave command
```

**Powered by:**
• Full 9-step weaving architecture
• Reflection loop (continuous learning)
• Thompson Sampling exploration
• SafetyGuardrails + AuditTrail (alignment framework)
"""

        await self.bot.send_message(room.room_id, help_text, markdown=True)

    @chatops_handler(
        command="ping",
        description="Check bot responsiveness and status",
        usage="!ping",
        category=HandlerCategory.SYSTEM,
        aliases=["p"]
    )
    async def handle_ping(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Simple ping command.

        Usage: !ping
        """
        # Safety gate check (safe command, but log for audit)
        if not await self._gate_action(room, event, "ping", "QUERY", args or ""):
            return

        reflection_metrics = self.shuttle.get_reflection_metrics()
        total_cycles = reflection_metrics.get('total_cycles', 0) if reflection_metrics else 0

        response = f"""
**🏓 Pong!**

HoloLoom is operational.
• Weaving cycles: {total_cycles}
• Reflection: {'✅ Active' if self.shuttle.enable_reflection else '⚠️ Disabled'}
• Status: ✅ Ready
"""

        await self.bot.send_message(room.room_id, response, markdown=True)

    @chatops_handler(
        command="audit",
        description="Query safety audit trail for decisions",
        usage="!audit [recent|search|stats] [args]",
        category=HandlerCategory.ADMIN,
        requires_auth=True
    )
    async def handle_audit(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Query the audit trail for safety decisions.

        Usage:
            !audit - Show recent decisions
            !audit recent [count] - Show last N decisions (default 10)
            !audit search <term> - Search decisions by text
            !audit stats - Show audit statistics
        """
        # Safety gate check
        if not await self._gate_action(room, event, "audit", "QUERY", args or ""):
            return

        if not self.audit:
            await self.bot.send_message(
                room.room_id,
                "⚠️ Audit trail not enabled. Initialize handlers with `enable_alignment=True`.",
                markdown=True
            )
            return

        try:
            parts = args.split(maxsplit=1) if args else []
            subcommand = parts[0].lower() if parts else "recent"
            param = parts[1] if len(parts) > 1 else ""

            if subcommand == "recent":
                # Show recent decisions (access logs directly)
                count = int(param) if param.isdigit() else 10
                all_logs = self.audit.logs
                decisions = all_logs[-count:] if len(all_logs) > count else all_logs
                decisions = list(reversed(decisions))  # Most recent first

                if not decisions:
                    await self.bot.send_message(room.room_id, "📝 No audit decisions yet.")
                    return

                response = f"**📋 Last {len(decisions)} Audit Decisions**\n\n"
                for i, d in enumerate(decisions, 1):
                    outcome_icon = "✅" if d.outcome.value == "approved" else "❌"
                    action = d.metadata.get('action', 'unknown') if d.metadata else 'unknown'
                    user = d.metadata.get('user', 'N/A')[:20] if d.metadata else 'N/A'
                    response += f"{i}. {outcome_icon} `{action}` - {d.reason[:50] if d.reason else 'N/A'}\n"
                    response += f"   Risk: {d.risk_level or 'N/A'} | User: {user}\n"

                await self.bot.send_message(room.room_id, response, markdown=True)

            elif subcommand == "search":
                if not param:
                    await self.bot.send_message(room.room_id, "Usage: `!audit search <term>`")
                    return

                # Filter logs by query text containing search term
                search_lower = param.lower()
                decisions = [
                    d for d in self.audit.logs
                    if (d.query_text and search_lower in d.query_text.lower()) or
                       (d.reason and search_lower in d.reason.lower())
                ][:10]

                if not decisions:
                    await self.bot.send_message(room.room_id, f"No decisions found for '{param}'.")
                    return

                response = f"**🔍 Search Results for '{param}'**\n\n"
                for i, d in enumerate(decisions, 1):
                    outcome_icon = "✅" if d.outcome.value == "approved" else "❌"
                    action = d.metadata.get('action', 'unknown') if d.metadata else 'unknown'
                    response += f"{i}. {outcome_icon} `{action}` - {d.query_text[:40] if d.query_text else 'N/A'}...\n"

                await self.bot.send_message(room.room_id, response, markdown=True)

            elif subcommand == "stats":
                # Calculate statistics from logs
                all_logs = self.audit.logs
                total = len(all_logs)
                approved = sum(1 for d in all_logs if d.outcome.value == "approved")
                rejected = sum(1 for d in all_logs if d.outcome.value == "rejected")
                rejection_rate = rejected / total if total > 0 else 0

                # Count by risk level
                by_risk = {}
                for d in all_logs:
                    risk = d.risk_level or "UNKNOWN"
                    by_risk[risk] = by_risk.get(risk, 0) + 1

                response = f"""
**📊 Audit Trail Statistics**

• Total decisions: {total}
• Approved: {approved}
• Rejected: {rejected}
• Rejection rate: {rejection_rate:.1%}

**By Risk Level:**
• SAFE: {by_risk.get('SAFE', by_risk.get('safe', 0))}
• LOW: {by_risk.get('LOW', by_risk.get('low', 0))}
• MEDIUM: {by_risk.get('MEDIUM', by_risk.get('medium', 0))}
• HIGH: {by_risk.get('HIGH', by_risk.get('high', 0))}
"""
                await self.bot.send_message(room.room_id, response, markdown=True)

            else:
                await self.bot.send_message(
                    room.room_id,
                    "Usage: `!audit [recent|search <term>|stats]`\n\nExample: `!audit recent 5`",
                    markdown=True
                )

        except Exception as e:
            logger.error(f"Audit error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    # ========================================================================
    # Utilities
    # ========================================================================

    def register_all(self, code_department=None):
        """
        Register all handlers with the bot using the registry.

        Args:
            code_department: Optional ClaudeCodeDepartment for code handlers
        """
        # Create registry and register this instance
        self._registry = HandlerRegistry()
        count = self._registry.register_instance(self)

        # Register TestHandlers (no dependencies)
        try:
            from HoloLoom.chatops.handlers.test_handlers import TestHandlers
            test_handlers = TestHandlers()
            test_count = self._registry.register_instance(test_handlers)
            count += test_count
            logger.info(f"Registered {test_count} test handlers")
        except ImportError as e:
            logger.warning(f"Could not register test handlers: {e}")

        # Register CodeHandlers (requires ClaudeCodeDepartment)
        if code_department is not None:
            try:
                from HoloLoom.chatops.handlers.code_handlers import CodeHandlers
                code_handlers = CodeHandlers(department=code_department)
                code_count = self._registry.register_instance(code_handlers)
                count += code_count
                logger.info(f"Registered {code_count} code handlers")
            except ImportError as e:
                logger.warning(f"Could not register code handlers: {e}")
        else:
            logger.debug("Code handlers not registered (no department provided)")

        # Also register with bot's handler dict for backward compatibility
        for command in self._registry.get_all_commands():
            spec = self._registry.get_handler(command)
            if spec:
                self.bot.register_handler(command, spec.handler)
                # Register aliases too
                for alias in spec.aliases:
                    self.bot.register_handler(alias, spec.handler)

        logger.info(f"Registered {count} total handlers via registry (with aliases)")

    def get_registry(self) -> HandlerRegistry:
        """Get the handler registry for introspection."""
        if not hasattr(self, '_registry'):
            self._registry = HandlerRegistry()
            self._registry.register_instance(self)
        return self._registry

    def generate_help_text(self) -> str:
        """Generate help text from registry metadata."""
        return self.get_registry().generate_help(prefix="!")

    async def shutdown(self):
        """Clean shutdown with proper lifecycle management."""
        logger.info("Shutting down HoloLoom WeavingShuttle...")

        # Close shuttle (cancels tasks, flushes reflection buffer)
        await self.shuttle.close()

        logger.info("HoloLoom handlers shut down successfully")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
HoloLoom Matrix Bot Handlers
=============================

This module provides command handlers that integrate
Matrix chatops with HoloLoom WeavingShuttle.

Commands available:
- !weave <query> - Full weaving cycle with reflection
- !trace [id] - Show Spacetime trace with full provenance
- !learn [force] - Trigger learning analysis
- !memory add/search/stats - Memory management
- !analyze <text> - Convergence engine analysis
- !stats - System statistics with reflection metrics
- !help - Command help
- !ping - Health check

Features:
- Full 9-step weaving architecture
- Reflection loop for continuous learning
- Thompson Sampling exploration
- Spacetime provenance tracking
- Multi-scale embeddings (Matryoshka)

To use:
    from HoloLoom.chatops.core.matrix_bot import MatrixBot
    from HoloLoom.chatops.handlers.hololoom_handlers import HoloLoomMatrixHandlers

    bot = MatrixBot(config)
    handlers = HoloLoomMatrixHandlers(bot, config_mode="fast")
    handlers.register_all()
    await bot.start()
""")
