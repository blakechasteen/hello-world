"""
Jenny Action Handler System
============================
Execute panel actions (PIN, DISMISS, WHY, etc.) with full provenance tracking.

Week 3 MVP - November 2025

Philosophy:
> "Every click is a decision. Every decision is logged."

This module handles user interactions with Jenny panels, bridging:
- User clicks → Action execution
- Action outcomes → Lifecycle transitions
- Complete audit trail via SpecLedger

Action Flow:
    User clicks "Pin" → ActionHandler.execute()
                      → LifecycleManager.transition(STABLE)
                      → SpecLedger.log_transition()
                      → Return ActionResult

References:
- jenny_spec.py (ACTION_PIN, ACTION_DISMISS, etc.)
- jenny_lifecycle.py (state transitions)
- spec_ledger.py (provenance tracking)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable, Protocol, Union
from datetime import datetime
from uuid import uuid4
import asyncio
import functools

from .jenny_enums import LifecycleStage, DissolutionTrigger, ActionStatus
from .jenny_spec import (
    JennySpec,
    create_action,
)
from .jenny_lifecycle import (
    JennyLifecycleManagerBase,
    PanelState,
    PanelNotFoundError,
    InvalidTransitionError,
)
from .spec_ledger import SpecLedger, SpecLedgerEntry


# ============================================================================
# Action Result Types
# ============================================================================

@dataclass(frozen=True)
class ActionResult:
    """
    Immutable result of an action execution.

    Provides complete provenance of what happened when user
    interacted with a panel.
    """
    action_id: str
    handler: str
    status: ActionStatus
    spec_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # State changes
    previous_lifecycle: Optional[LifecycleStage] = None
    new_lifecycle: Optional[LifecycleStage] = None

    # Output
    message: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    # Error info (if failed)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/API."""
        return {
            "action_id": self.action_id,
            "handler": self.handler,
            "status": self.status.value,
            "spec_id": self.spec_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_lifecycle": self.previous_lifecycle.value if self.previous_lifecycle else None,
            "new_lifecycle": self.new_lifecycle.value if self.new_lifecycle else None,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }


# ============================================================================
# Action Handler Protocol
# ============================================================================

class ActionHandlerProtocol(Protocol):
    """Protocol for action handler implementations."""

    async def execute(
        self,
        action: Dict[str, Any],
        spec: JennySpec,
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """Execute an action on a panel."""
        ...

    def register_handler(
        self,
        handler_name: str,
        handler_fn: Callable[[JennySpec, Dict[str, Any]], Awaitable[ActionResult]],
    ) -> None:
        """Register a custom action handler."""
        ...


# ============================================================================
# Action Handler Decorator (Phase 2.2 Elegance Refactor)
# ============================================================================

# Type alias for handler return: either ActionResult or success dict
HandlerReturn = Union[ActionResult, Dict[str, Any]]


def action_handler(handler_name: str, tracks_lifecycle: bool = False):
    """
    Decorator for action handlers that extracts common boilerplate.

    Standardizes:
    - action_id generation (UUID)
    - Error handling (exceptions → FAILED ActionResult)
    - ActionResult creation from success dict

    The decorated function can return:
    - ActionResult: Passed through (for custom error handling)
    - Dict: Auto-wrapped in SUCCESS ActionResult with keys:
        - message: Success message
        - data: Additional data (optional)
        - new_lifecycle: New lifecycle stage (optional)

    Args:
        handler_name: Name for the handler (used in ActionResult.handler)
        tracks_lifecycle: If True, captures spec.lifecycle as previous_lifecycle

    Usage:
        @action_handler("expand_panel")
        async def _handle_expand_panel(spec, params, lifecycle_manager, ledger):
            return {"message": "Panel expanded", "data": {"new_size": "xlarge"}}
    """
    def decorator(handler_fn: Callable) -> Callable:
        @functools.wraps(handler_fn)
        async def wrapper(
            spec: JennySpec,
            params: Dict[str, Any],
            lifecycle_manager: JennyLifecycleManagerBase,
            ledger: Optional[SpecLedger],
        ) -> ActionResult:
            action_id = str(uuid4())
            previous_lifecycle = spec.lifecycle if tracks_lifecycle else None

            try:
                result = await handler_fn(spec, params, lifecycle_manager, ledger)

                # If handler returns ActionResult directly, pass through
                if isinstance(result, ActionResult):
                    return result

                # Handler returned success dict - wrap in ActionResult
                return ActionResult(
                    action_id=action_id,
                    handler=handler_name,
                    status=ActionStatus.SUCCESS,
                    spec_id=spec.spec_id,
                    previous_lifecycle=previous_lifecycle,
                    new_lifecycle=result.get("new_lifecycle"),
                    message=result.get("message"),
                    data=result.get("data", {}),
                )

            except (InvalidTransitionError, PanelNotFoundError) as e:
                # Known lifecycle errors - provide context
                return ActionResult(
                    action_id=action_id,
                    handler=handler_name,
                    status=ActionStatus.FAILED,
                    spec_id=spec.spec_id,
                    previous_lifecycle=previous_lifecycle,
                    error=str(e),
                    message=f"Lifecycle error: {type(e).__name__}",
                )

            except Exception as e:
                # Generic error handling
                return ActionResult(
                    action_id=action_id,
                    handler=handler_name,
                    status=ActionStatus.FAILED,
                    spec_id=spec.spec_id,
                    previous_lifecycle=previous_lifecycle,
                    error=str(e),
                )

        return wrapper
    return decorator


# ============================================================================
# Built-in Action Handlers
# ============================================================================

@action_handler("pin_panel", tracks_lifecycle=True)
async def _handle_pin_panel(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """
    Pin panel: NASCENT → STABLE

    Makes a temporary panel persistent.
    Decorator handles: action_id, error handling, ActionResult creation.
    """
    # Core logic: pin the panel
    await lifecycle_manager.pin(spec.spec_id)

    # Log to ledger
    if ledger:
        await ledger.log_lifecycle_transition(
            spec_id=spec.spec_id,
            new_stage=LifecycleStage.STABLE,
        )

    return {
        "new_lifecycle": LifecycleStage.STABLE,
        "message": "Panel pinned successfully",
    }


@action_handler("dismiss_panel", tracks_lifecycle=True)
async def _handle_dismiss_panel(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """
    Dismiss panel: * → DISSOLVING → ARCHIVED

    User manually closes a panel.
    Decorator handles: action_id, error handling, ActionResult creation.
    """
    # Dissolve the panel (trigger: MANUAL)
    await lifecycle_manager.dissolve(spec.spec_id, DissolutionTrigger.MANUAL)

    # Log to ledger
    if ledger:
        await ledger.log_lifecycle_transition(
            spec_id=spec.spec_id,
            new_stage=LifecycleStage.DISSOLVING,
            trigger=DissolutionTrigger.MANUAL,
        )

    return {
        "new_lifecycle": LifecycleStage.DISSOLVING,
        "message": "Panel dismissed",
    }


@action_handler("show_why_panel")
async def _handle_show_why_panel(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """
    Show "Why this UI?" meta-panel.

    Generates a WHY panel explaining the reasoning behind current panel.
    This is where Jenny explains itself (breaks the strange loop).
    Decorator handles: action_id, error handling, ActionResult creation.
    """
    # Generate explanation data
    why_data = {
        "original_spec_id": spec.spec_id,
        "panel_type": spec.panel_type.value,
        "reasoning": [
            f"Panel type '{spec.panel_type.value}' was selected based on query content",
            f"Confidence level determined appropriate visualization",
            f"Content structure matched {spec.panel_type.value} template",
        ],
        "alternatives_considered": [
            "text", "graph", "table"
        ],
        "provenance": {
            "spacetime_id": spec.spacetime_id,
            "compiler_version": spec.compiler_version,
            "timestamp": spec.timestamp.isoformat(),
        },
    }

    return {
        "message": "Why panel generated",
        "data": {
            "why_panel_data": why_data,
            "should_spawn_panel": True,
        },
    }


@action_handler("expand_panel")
async def _handle_expand_panel(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """Expand panel to larger size."""
    return {
        "message": "Panel expanded",
        "data": {
            "new_size": "xlarge",
            "animation": "scale_up",
        },
    }


@action_handler("collapse_panel")
async def _handle_collapse_panel(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """Collapse panel to smaller size."""
    return {
        "message": "Panel collapsed",
        "data": {
            "new_size": "small",
            "animation": "scale_down",
        },
    }


@action_handler("copy_content")
async def _handle_copy_content(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """Copy panel content to clipboard."""
    # Extract copyable content based on panel type
    content = spec.content
    if spec.panel_type.value == "text":
        copy_text = content.get("text", "")
    elif spec.panel_type.value == "code":
        copy_text = content.get("code", "")
    else:
        copy_text = str(content)

    return {
        "message": "Content copied to clipboard",
        "data": {
            "copied_text": copy_text,
            "content_type": spec.panel_type.value,
        },
    }


@action_handler("export_panel")
async def _handle_export_panel(
    spec: JennySpec,
    params: Dict[str, Any],
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger],
) -> HandlerReturn:
    """
    Export panel data.

    This action requires confirmation (marked in jenny_spec.py).
    Note: Returns ActionResult directly for PENDING status (decorator passes through).
    """
    # Check if confirmation was provided
    confirmed = params.get("confirmed", False)

    if not confirmed:
        # Return ActionResult directly for non-success status
        return ActionResult(
            action_id=str(uuid4()),
            handler="export_panel",
            status=ActionStatus.PENDING,
            spec_id=spec.spec_id,
            message="Export requires confirmation",
            data={
                "requires_confirmation": True,
                "confirmation_message": "Are you sure you want to export this panel?",
            },
        )

    # Perform export
    export_data = {
        "spec": spec.to_dict(),
        "exported_at": datetime.now().isoformat(),
        "format": params.get("format", "json"),
    }

    return {
        "message": "Panel exported successfully",
        "data": {
            "export_data": export_data,
        },
    }


# ============================================================================
# Main Action Handler
# ============================================================================

class JennyActionHandler:
    """
    Main action handler for Jenny panels.

    Coordinates action execution, lifecycle transitions, and provenance logging.

    Usage:
        handler = JennyActionHandler(lifecycle_manager, ledger)

        # Execute built-in action
        result = await handler.execute(
            action=ACTION_PIN,
            spec=my_spec,
        )

        # Register custom handler
        handler.register_handler("custom_action", my_custom_handler)
    """

    def __init__(
        self,
        lifecycle_manager: JennyLifecycleManagerBase,
        ledger: Optional[SpecLedger] = None,
        learning_callback: Optional[Callable[[ActionResult, JennySpec], Awaitable[None]]] = None,
    ):
        """
        Initialize action handler.

        Args:
            lifecycle_manager: Manages panel lifecycle state
            ledger: Optional SpecLedger for provenance tracking
            learning_callback: Optional callback for Thompson Sampling learning updates.
                              Called with (ActionResult, JennySpec) after successful actions.
                              Used by Phase 2.1-2.2 MRF integration to update panel type priors.
        """
        self._lifecycle_manager = lifecycle_manager
        self._ledger = ledger
        self._learning_callback = learning_callback

        # Built-in handlers
        self._handlers: Dict[str, Callable] = {
            "pin_panel": _handle_pin_panel,
            "dismiss_panel": _handle_dismiss_panel,
            "show_why_panel": _handle_show_why_panel,
            "expand_panel": _handle_expand_panel,
            "collapse_panel": _handle_collapse_panel,
            "copy_content": _handle_copy_content,
            "export_panel": _handle_export_panel,
        }

        # Action history for debugging
        self._history: List[ActionResult] = []
        self._max_history = 1000

    async def execute(
        self,
        action: Dict[str, Any],
        spec: JennySpec,
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """
        Execute an action on a panel.

        Args:
            action: Action definition dict (from jenny_spec.py)
            spec: The JennySpec to act upon
            context: Optional additional context

        Returns:
            ActionResult with outcome details
        """
        handler_name = action.get("handler", "")
        params = action.get("params", {})

        # Merge context into params
        if context:
            params = {**params, **context}

        # Check for confirmation requirement
        requires_confirmation = action.get("requires_confirmation", False)
        if requires_confirmation and not params.get("confirmed", False):
            return ActionResult(
                action_id=str(uuid4()),
                handler=handler_name,
                status=ActionStatus.PENDING,
                spec_id=spec.spec_id,
                message=f"Action '{handler_name}' requires confirmation",
                data={
                    "requires_confirmation": True,
                    "action": action,
                },
            )

        # Get handler
        handler_fn = self._handlers.get(handler_name)

        if not handler_fn:
            result = ActionResult(
                action_id=str(uuid4()),
                handler=handler_name,
                status=ActionStatus.FAILED,
                spec_id=spec.spec_id,
                error=f"Unknown handler: {handler_name}",
            )
        else:
            # Execute handler
            result = await handler_fn(
                spec,
                params,
                self._lifecycle_manager,
                self._ledger,
            )

        # Record in history
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Phase 2.1-2.2: Emit learning event for Thompson Sampling updates
        # Only emit for successful actions to provide learning signal
        if self._learning_callback and result.status == ActionStatus.SUCCESS:
            try:
                await self._learning_callback(result, spec)
            except Exception as e:
                # Don't fail the action due to learning callback error
                # Just log and continue (graceful degradation)
                import logging
                logging.getLogger(__name__).warning(
                    f"Learning callback failed for {handler_name}: {e}"
                )

        return result

    def register_handler(
        self,
        handler_name: str,
        handler_fn: Callable,
    ) -> None:
        """
        Register a custom action handler.

        Args:
            handler_name: Name for the handler (matches action["handler"])
            handler_fn: Async function (spec, params, lifecycle, ledger) -> ActionResult
        """
        self._handlers[handler_name] = handler_fn

    def get_available_handlers(self) -> List[str]:
        """Get list of registered handler names."""
        return list(self._handlers.keys())

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent action history."""
        return [r.to_dict() for r in self._history[-limit:]]

    def clear_history(self) -> None:
        """Clear action history."""
        self._history.clear()


# ============================================================================
# Factory Functions
# ============================================================================

def create_action_handler(
    lifecycle_manager: JennyLifecycleManagerBase,
    ledger: Optional[SpecLedger] = None,
    learning_callback: Optional[Callable[[ActionResult, JennySpec], Awaitable[None]]] = None,
) -> JennyActionHandler:
    """
    Create a Jenny action handler.

    Args:
        lifecycle_manager: Panel lifecycle manager
        ledger: Optional spec ledger for provenance
        learning_callback: Optional callback for Thompson Sampling learning updates

    Returns:
        Configured JennyActionHandler
    """
    return JennyActionHandler(lifecycle_manager, ledger, learning_callback)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Types
    "ActionStatus",
    "ActionResult",
    "ActionHandlerProtocol",

    # Main class
    "JennyActionHandler",

    # Factory
    "create_action_handler",
]