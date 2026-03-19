"""
Agent Manager Integration Layer

Connects AgentManagerHub events to WebSocket broadcasts and bridges
with the existing AgentMonitor for backward compatibility.

This module provides:
1. Event-to-WebSocket bridge - Hub events trigger WebSocket broadcasts
2. AgentMonitor bridge - Existing monitor receives events for backward compat
3. Unified startup/shutdown - Single point of control for all systems

Status: Production Ready (December 2025)
"""

import asyncio
import logging
from typing import Any

from hololoom.agentic.monitoring import (
    AgentMonitor,
    AgentStatus,
    get_monitor,
    start_monitoring,
    stop_monitoring,
)
from hololoom.apps.chatops.handlers.agent_manager_ws import (
    AgentManagerMessage,
    AgentManagerMessageType,
    AgentManagerWSManager,
)
from hololoom.apps.chatops.handlers.agent_manager_ws import (
    create_agent_manager_router as create_ws_router,
)
from hololoom.apps.server.agent_manager_hub import AgentManagerHub

logger = logging.getLogger(__name__)


class AgentManagerIntegration:
    """
    Integration layer connecting AgentManagerHub, WebSocket handlers,
    and existing AgentMonitor.

    Responsibilities:
    - Register event handlers on AgentManagerHub
    - Broadcast events via AgentManagerWSManager
    - Bridge to AgentMonitor for backward compatibility
    - Provide unified lifecycle management
    """

    def __init__(
        self,
        hub: AgentManagerHub | None = None,
        ws_manager: AgentManagerWSManager | None = None,
        monitor: AgentMonitor | None = None,
        enable_legacy_monitor: bool = True,
    ):
        """
        Initialize integration layer.

        Args:
            hub: AgentManagerHub instance (created if None)
            ws_manager: WebSocket manager (created if None)
            monitor: AgentMonitor instance (uses global if None)
            enable_legacy_monitor: Bridge events to legacy AgentMonitor
        """
        self.hub = hub or AgentManagerHub()
        self.ws_manager = ws_manager or AgentManagerWSManager()
        self.monitor = monitor
        self.enable_legacy_monitor = enable_legacy_monitor

        # Track if we created the monitor
        self._created_monitor = False

        # Event handlers registered
        self._handlers_registered = False

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

    async def start(self):
        """Start all integrated systems."""
        logger.info("Starting Agent Manager Integration")

        # Start legacy monitor if enabled
        if self.enable_legacy_monitor:
            if self.monitor is None:
                self.monitor = get_monitor()
                self._created_monitor = True
            await start_monitoring()

            # Connect hub to monitor
            self.hub.monitor = self.monitor

        # Register event handlers
        self._register_event_handlers()

        logger.info("Agent Manager Integration started")

    async def stop(self):
        """Stop all integrated systems."""
        logger.info("Stopping Agent Manager Integration")

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._background_tasks.clear()

        # Stop legacy monitor if we created it
        if self.enable_legacy_monitor and self._created_monitor:
            await stop_monitoring()

        logger.info("Agent Manager Integration stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def _register_event_handlers(self):
        """Register handlers for all hub events."""
        if self._handlers_registered:
            return

        # Thread lifecycle events
        self.hub.register_event_handler("thread_created", self._on_thread_created)
        self.hub.register_event_handler("thread_started", self._on_thread_started)
        self.hub.register_event_handler("thread_paused", self._on_thread_paused)
        self.hub.register_event_handler("thread_resumed", self._on_thread_resumed)
        self.hub.register_event_handler("thread_completed", self._on_thread_completed)
        self.hub.register_event_handler("thread_cancelled", self._on_thread_cancelled)
        self.hub.register_event_handler("thread_failed", self._on_thread_failed)

        # Progress events
        self.hub.register_event_handler("step_started", self._on_step_started)
        self.hub.register_event_handler("step_progress", self._on_step_progress)
        self.hub.register_event_handler("step_completed", self._on_step_completed)

        # Priority events
        self.hub.register_event_handler("priority_changed", self._on_priority_changed)

        # Injection events
        self.hub.register_event_handler("mrf_injected", self._on_mrf_injected)
        self.hub.register_event_handler("mcts_injected", self._on_mcts_injected)

        # Swarm events
        self.hub.register_event_handler("swarm_created", self._on_swarm_created)
        self.hub.register_event_handler("swarm_child_completed", self._on_swarm_child_completed)
        self.hub.register_event_handler("swarm_merged", self._on_swarm_merged)

        # Git/temporal events
        self.hub.register_event_handler("commit_created", self._on_commit_created)
        self.hub.register_event_handler("branch_created", self._on_branch_created)
        self.hub.register_event_handler("checkout_completed", self._on_checkout_completed)

        self._handlers_registered = True
        logger.debug("Event handlers registered")

    # =========================================================================
    # Thread Lifecycle Event Handlers
    # =========================================================================

    async def _on_thread_created(self, event: dict[str, Any]):
        """Handle thread created event."""
        thread_id = event.get("thread_id")
        project = event.get("project", "default")

        # Create message with proper fields (no data dict)
        message = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_CREATED,
            thread_id=thread_id,
            thread_name=event.get("name"),
            thread_status="idle",
            priority=event.get("priority", 0),
            message=event.get("reasoning_mode", "DIRECT"),
        )

        # Broadcast via WebSocket to project subscribers
        await self.ws_manager.broadcast_to_pattern(f"project:{project}", message)

        # Also broadcast to thread-specific subscribers
        await self.ws_manager.broadcast_to_pattern(f"thread:{thread_id}", message)

        # Legacy monitor bridge
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_started(
                agent_id=thread_id,
                project=project,
                query=event.get("query", ""),
                mode=event.get("reasoning_mode", "DIRECT"),
                files=event.get("files", []),
            )

    async def _on_thread_started(self, event: dict[str, Any]):
        """Handle thread started event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.THREAD_STARTED,
            event,
        )

        # Legacy monitor
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_status(
                agent_id=thread_id,
                status=AgentStatus.RUNNING,
            )

    async def _on_thread_paused(self, event: dict[str, Any]):
        """Handle thread paused event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.THREAD_PAUSED,
            event,
        )

        # Legacy monitor
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_status(
                agent_id=thread_id,
                status=AgentStatus.WAITING,
            )

    async def _on_thread_resumed(self, event: dict[str, Any]):
        """Handle thread resumed event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.THREAD_RESUMED,
            event,
        )

        # Legacy monitor
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_status(
                agent_id=thread_id,
                status=AgentStatus.RUNNING,
            )

    async def _on_thread_completed(self, event: dict[str, Any]):
        """Handle thread completed event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.THREAD_COMPLETED,
            event,
        )

        # Legacy monitor
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_completed(
                agent_id=thread_id,
                steps=event.get("steps", []),
                total_duration_ms=event.get("total_duration_ms", 0),
            )

    async def _on_thread_cancelled(self, event: dict[str, Any]):
        """Handle thread cancelled event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.THREAD_CANCELLED,
            event,
        )

        # Legacy monitor - use failed status
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_failed(
                agent_id=thread_id,
                error="Thread cancelled by user",
            )

    async def _on_thread_failed(self, event: dict[str, Any]):
        """Handle thread failed event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.THREAD_ERROR,
            event,
        )

        # Legacy monitor
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_failed(
                agent_id=thread_id,
                error=event.get("error", "Unknown error"),
            )

    # =========================================================================
    # Step Progress Event Handlers
    # =========================================================================

    async def _on_step_started(self, event: dict[str, Any]):
        """Handle step started event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.STEP_STARTED,
            event,
        )

    async def _on_step_progress(self, event: dict[str, Any]):
        """Handle step progress event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.STEP_PROGRESS,
            event,
        )

        # Legacy monitor - update step
        if self.enable_legacy_monitor and self.monitor:
            await self.monitor.agent_step(
                agent_id=thread_id,
                step=event.get("current_step", 0),
                total_steps=event.get("total_steps", 1),
                step_type=event.get("step_type", "execution"),
                confidence=event.get("confidence"),
                epistemic=event.get("epistemic_confidence"),
                finding=event.get("finding"),
                latency_ms=event.get("latency_ms"),
            )

            # Update feed lines
            await self.monitor.agent_feed(
                agent_id=thread_id,
                line1=event.get("status_line", "Processing..."),
                line2=event.get("detail_line", ""),
            )

    async def _on_step_completed(self, event: dict[str, Any]):
        """Handle step completed event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.STEP_COMPLETED,
            event,
        )

    # =========================================================================
    # Priority Event Handlers
    # =========================================================================

    async def _on_priority_changed(self, event: dict[str, Any]):
        """Handle priority changed event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.PRIORITY_CHANGED,
            event,
        )

    # =========================================================================
    # Injection Event Handlers
    # =========================================================================

    async def _on_mrf_injected(self, event: dict[str, Any]):
        """Handle MRF injection event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.MRF_INJECTED,
            event,
        )

    async def _on_mcts_injected(self, event: dict[str, Any]):
        """Handle MCTS injection event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.MCTS_INJECTED,
            event,
        )

    # =========================================================================
    # Swarm Event Handlers
    # =========================================================================

    async def _on_swarm_created(self, event: dict[str, Any]):
        """Handle swarm created event."""
        swarm_id = event.get("swarm_id")

        # Create message with proper fields (no data dict)
        message = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_SPAWNED,
            swarm_id=swarm_id,
            thread_id=event.get("parent_thread_id"),
            child_thread_ids=event.get("child_thread_ids", []),
            merge_strategy=event.get("merge_strategy"),
            message=event.get("message", "Swarm created"),
        )

        # Broadcast to swarm subscribers
        await self.ws_manager.broadcast_to_pattern(f"swarm:{swarm_id}", message)

        # Also broadcast to wildcard
        await self.ws_manager.broadcast_to_pattern("*", message)

    async def _on_swarm_child_completed(self, event: dict[str, Any]):
        """Handle swarm child completed event."""
        swarm_id = event.get("swarm_id")

        message = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_CHILD_COMPLETED,
            swarm_id=swarm_id,
            thread_id=event.get("thread_id"),
            confidence=event.get("confidence", 0.0),
            message=event.get("result", ""),
        )

        await self.ws_manager.broadcast_to_pattern(f"swarm:{swarm_id}", message)

    async def _on_swarm_merged(self, event: dict[str, Any]):
        """Handle swarm merged event."""
        swarm_id = event.get("swarm_id")

        message = AgentManagerMessage(
            type=AgentManagerMessageType.SWARM_MERGED,
            swarm_id=swarm_id,
            merge_strategy=event.get("merge_strategy"),
            confidence=event.get("final_confidence", 0.0),
            message=event.get("final_result", ""),
        )

        await self.ws_manager.broadcast_to_pattern(f"swarm:{swarm_id}", message)

    # =========================================================================
    # Git/Temporal Event Handlers
    # =========================================================================

    async def _on_commit_created(self, event: dict[str, Any]):
        """Handle git commit created event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.SNAPSHOT_CREATED,
            event,
        )

    async def _on_branch_created(self, event: dict[str, Any]):
        """Handle git branch created event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.BRANCH_CREATED,
            event,
        )

    async def _on_checkout_completed(self, event: dict[str, Any]):
        """Handle git checkout completed event."""
        thread_id = event.get("thread_id")

        await self._broadcast_thread_event(
            thread_id,
            AgentManagerMessageType.ROLLBACK_COMPLETED,
            event,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _broadcast_thread_event(
        self,
        thread_id: str,
        message_type: AgentManagerMessageType,
        event: dict[str, Any],
    ):
        """Broadcast event to thread subscribers and wildcard."""
        # Extract specific fields from event for AgentManagerMessage
        message = AgentManagerMessage(
            type=message_type,
            thread_id=thread_id,
            thread_name=event.get("name") or event.get("thread_name"),
            thread_status=event.get("status") or event.get("thread_status"),
            current_step=event.get("current_step", 0),
            total_steps=event.get("total_steps", 0),
            step_name=event.get("step_name", ""),
            step_type=event.get("step_type", ""),
            percentage=event.get("percentage", 0.0),
            message=event.get("message") or event.get("error") or "",
            confidence=event.get("confidence", 0.0),
            epistemic_confidence=event.get("epistemic_confidence", 0.0),
            priority=event.get("priority", 0),
            old_priority=event.get("old_priority"),
            swarm_id=event.get("swarm_id"),
        )

        # Thread-specific subscribers
        await self.ws_manager.broadcast_to_pattern(
            f"thread:{thread_id}",
            message,
        )

        # Wildcard subscribers
        await self.ws_manager.broadcast_to_pattern("*", message)

        # Project subscribers if thread has project
        thread = self.hub.get_thread(thread_id)
        if thread and hasattr(thread, 'project') and thread.project:
            await self.ws_manager.broadcast_to_pattern(
                f"project:{thread.project}",
                message,
            )


# =============================================================================
# Factory Functions
# =============================================================================

_global_integration: AgentManagerIntegration | None = None


def get_integration() -> AgentManagerIntegration | None:
    """Get the global integration instance."""
    return _global_integration


async def create_integration(
    hub: AgentManagerHub | None = None,
    ws_manager: AgentManagerWSManager | None = None,
    enable_legacy_monitor: bool = True,
) -> AgentManagerIntegration:
    """
    Create and start the global integration instance.

    Args:
        hub: Optional AgentManagerHub (created if None)
        ws_manager: Optional WebSocket manager (created if None)
        enable_legacy_monitor: Enable legacy AgentMonitor bridge

    Returns:
        Started AgentManagerIntegration instance
    """
    global _global_integration

    if _global_integration is not None:
        logger.warning("Global integration already exists, returning existing")
        return _global_integration

    _global_integration = AgentManagerIntegration(
        hub=hub,
        ws_manager=ws_manager,
        enable_legacy_monitor=enable_legacy_monitor,
    )
    await _global_integration.start()

    return _global_integration


async def shutdown_integration():
    """Shutdown the global integration instance."""
    global _global_integration

    if _global_integration is not None:
        await _global_integration.stop()
        _global_integration = None


# =============================================================================
# FastAPI Integration
# =============================================================================

def create_integrated_app():
    """
    Create a fully integrated FastAPI app with all Agent Manager components.

    This uses the existing agent_manager_api.app (which has all REST endpoints)
    and adds WebSocket support plus the integration layer.

    Returns:
        FastAPI application with REST API, WebSocket, and integration layer
    """
    # Import the existing app which has all REST endpoints
    from hololoom.apps.server.agent_manager_api import app, get_hub

    # Get the hub from the API module
    hub = get_hub()

    # Create WebSocket manager
    ws_manager = AgentManagerWSManager()

    # Create integration (will be started on app startup)
    integration = AgentManagerIntegration(
        hub=hub,
        ws_manager=ws_manager,
        enable_legacy_monitor=True,
    )

    # Add WebSocket router to existing app
    ws_router = create_ws_router(ws_manager, hub)
    app.include_router(ws_router)

    # Store original startup/shutdown handlers
    original_startup = app.router.on_startup.copy() if app.router.on_startup else []
    original_shutdown = app.router.on_shutdown.copy() if app.router.on_shutdown else []

    # Clear and add combined handlers
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()

    @app.on_event("startup")
    async def startup():
        # Run original startup events
        for handler in original_startup:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()
        # Start integration
        await integration.start()

    @app.on_event("shutdown")
    async def shutdown():
        # Stop integration first
        await integration.stop()
        # Run original shutdown events
        for handler in original_shutdown:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()

    # Store references for access
    app.state.hub = hub
    app.state.ws_manager = ws_manager
    app.state.integration = integration

    return app
