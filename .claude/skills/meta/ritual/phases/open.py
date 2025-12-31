"""
/ritual open - Session Opening Phase
=====================================

Initializes a new coding ritual session with:
- Task description
- Complexity level
- Success criteria

Maps to: PREFLIGHT → COUNTDOWN lifecycle states
Emits: ritual.session.opened event

HoloLoom Integration:
- Gets complexity recommendations from Thompson Sampling
- Retrieves similar past sessions from memory
- Collects relevant lessons from RAG
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.session_manager import (
    SessionManager,
    SessionState,
    ComplexityLevel,
    RitualPhase,
)
from events import (
    EVENT_BUS_AVAILABLE,
    RitualContext,
    RitualEventType,
    create_ritual_event,
    ritual_started,
)

# Optional HoloLoom integration
if TYPE_CHECKING:
    from hololoom import HoloLoomIntegration


class OpenPhaseHandler:
    """
    Handles /ritual open command.

    Initializes a new coding session with:
    - Task description (required)
    - Complexity level (optional, default: fast)
    - Success criteria (prompted if not provided)
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[Any] = None,
        hololoom_integration: Optional["HoloLoomIntegration"] = None,
    ):
        """
        Initialize open phase handler.

        Args:
            session_manager: SessionManager instance (created if not provided)
            event_bus: Optional EventBus for emitting events
            hololoom_integration: Optional HoloLoomIntegration for recommendations
        """
        self.session_manager = session_manager or SessionManager()
        self.event_bus = event_bus
        self.hololoom = hololoom_integration

    async def execute(
        self,
        task: str,
        complexity: str = "fast",
        success_criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the open phase.

        Args:
            task: Description of what you're building
            complexity: lite, fast, full, or research
            success_criteria: List of success criteria (prompts if empty)

        Returns:
            Structured result with session info and next steps
        """
        start_time = datetime.now()

        # Get HoloLoom recommendations if available
        hololoom_recommendations = None
        if self.hololoom:
            try:
                hololoom_recommendations = await self.hololoom.get_open_recommendations(task)
                # Use recommended complexity if user didn't specify
                if complexity == "fast" and hololoom_recommendations.get("recommended_complexity"):
                    complexity = hololoom_recommendations["recommended_complexity"]
            except Exception:
                # Graceful degradation - continue without recommendations
                pass

        # Check for existing active session
        if self.session_manager.has_active_session():
            return self._error_response(
                "Active session already exists. Use /ritual close first, "
                "or /ritual status to view current session."
            )

        # Validate complexity
        try:
            complexity_level = ComplexityLevel(complexity.lower())
        except ValueError:
            return self._error_response(
                f"Invalid complexity '{complexity}'. "
                f"Valid options: lite, fast, full, research"
            )

        # Generate correlation ID for event chain
        correlation_id = f"ritual-{uuid.uuid4().hex[:8]}"

        # Create session
        try:
            session = self.session_manager.create_session(
                task=task,
                complexity=complexity_level,
                success_criteria=success_criteria or [],
                correlation_id=correlation_id,
            )
        except Exception as e:
            return self._error_response(f"Failed to create session: {e}")

        # Emit event if EventBus available
        events_emitted = []
        if self.event_bus:
            event_id = await self._emit_session_opened(session)
            if event_id:
                events_emitted.append(event_id)

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Build response
        response = self._build_success_response(
            session=session,
            needs_criteria=not success_criteria,
            execution_time_ms=execution_time_ms,
            events_emitted=events_emitted,
            hololoom_recommendations=hololoom_recommendations,
        )

        return response

    def _build_success_response(
        self,
        session: SessionState,
        needs_criteria: bool,
        execution_time_ms: float,
        events_emitted: List[str],
        hololoom_recommendations: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build success response for session opening."""

        # Determine next steps based on whether criteria provided
        if needs_criteria:
            next_steps = [
                "Define success criteria for this task",
                "Use /ritual design to define interfaces",
                "Use /ritual status to check progress",
            ]
            message = (
                f"Ritual session opened. "
                f"Let's define what 'done' looks like for: {session.task}"
            )
        else:
            next_steps = [
                "Use /ritual design to define interfaces",
                "Use /ritual build to start implementation",
                "Use /ritual status to check progress",
            ]
            message = "Ritual session opened. Ready to begin."

        response = {
            "phase": "open",
            "session_id": session.session_id,
            "status": "success",
            "message": message,
            "next_steps": next_steps,
            "data": {
                "task": session.task,
                "complexity": session.complexity.value,
                "success_criteria": session.success_criteria,
                "needs_criteria": needs_criteria,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

        # Add HoloLoom recommendations if available
        if hololoom_recommendations:
            response["data"]["hololoom"] = {
                "recommended_complexity": hololoom_recommendations.get("recommended_complexity"),
                "similar_sessions": hololoom_recommendations.get("similar_sessions", []),
                "relevant_lessons": hololoom_recommendations.get("relevant_lessons", []),
                "warnings": hololoom_recommendations.get("warnings", []),
            }
            # Add warnings from past sessions
            if hololoom_recommendations.get("warnings"):
                response["warnings"] = hololoom_recommendations["warnings"]

        return response

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Build error response."""
        return {
            "phase": "open",
            "session_id": None,
            "status": "error",
            "message": message,
            "next_steps": [],
            "data": {},
            "metadata": {},
        }

    async def _emit_session_opened(self, session: SessionState) -> Optional[str]:
        """Emit ritual.session.opened event."""
        if not self.event_bus or not EVENT_BUS_AVAILABLE:
            return None

        try:
            # Create RitualContext from session state
            context = RitualContext(
                ritual_id=session.correlation_id or session.session_id,
                feature_name=session.task,
                started_at=session.started_at,
            )

            # Create event using ritual_started factory
            event = ritual_started(context)

            # Add session-specific payload
            event["payload"].update({
                "session_id": session.session_id,
                "complexity": session.complexity.value,
                "success_criteria": session.success_criteria,
            })

            # Update topic to be more specific
            event["topic"] = "ritual.session.opened"

            # Record event in session
            session.record_event(event.get("event_id", ""))

            # Emit via EventBus
            await self.event_bus.emit(event)
            return event.get("event_id")

        except Exception:
            # Event emission failed, but don't break the flow
            return None


# Convenience function for direct invocation
async def open_ritual_session(
    task: str,
    complexity: str = "fast",
    success_criteria: Optional[List[str]] = None,
    event_bus: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Open a new ritual session.

    Args:
        task: Description of what you're building
        complexity: lite, fast, full, or research
        success_criteria: List of success criteria
        event_bus: Optional EventBus for events

    Returns:
        Session opening result
    """
    handler = OpenPhaseHandler(event_bus=event_bus)
    return await handler.execute(
        task=task,
        complexity=complexity,
        success_criteria=success_criteria,
    )


# Example usage and formatting for Claude
def format_open_response(result: Dict[str, Any]) -> str:
    """
    Format open response for display.

    Returns formatted string suitable for terminal output.
    """
    if result["status"] == "error":
        return f"Error: {result['message']}"

    lines = [
        "",
        "Ritual Session Opened",
        "",
        f"**Task**: {result['data']['task']}",
        f"**Complexity**: {result['data']['complexity']}",
        f"**Session ID**: {result['session_id']}",
        "",
    ]

    if result['data'].get('needs_criteria'):
        lines.extend([
            "Let's define success criteria. What needs to be true when you're done?",
            "",
            "Suggestions based on your task:",
            "- Core functionality works",
            "- Tests pass",
            "- Code is documented",
            "",
            "Enter your criteria (one per line, empty line when done):",
        ])
    else:
        criteria = result['data']['success_criteria']
        lines.append("**Success Criteria**:")
        for i, criterion in enumerate(criteria, 1):
            lines.append(f"  {i}. {criterion}")
        lines.append("")
        lines.append("**Next Steps**:")
        for step in result['next_steps']:
            lines.append(f"  - {step}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    import asyncio

    async def demo():
        result = await open_ritual_session(
            task="Add dark mode toggle to settings",
            complexity="fast",
            success_criteria=[
                "Toggle appears in settings page",
                "Theme persists across sessions",
                "All components respect theme",
            ],
        )
        print(format_open_response(result))

    asyncio.run(demo())