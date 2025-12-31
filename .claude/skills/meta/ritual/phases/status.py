"""
/ritual status - Session Status Display
=======================================

Shows current session status including:
- Phase and progress
- Success criteria completion
- Time elapsed
- Wave progress (if in build phase)
- Safety checks (if completed)

Provides a snapshot of where you are in the ritual.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.session_manager import (
    SessionManager,
    SessionState,
    RitualPhase,
)


class StatusPhaseHandler:
    """
    Handles /ritual status command.

    Shows current session state:
    - Phase and sub-phase
    - Success criteria status
    - Time elapsed
    - Wave progress
    - Recent activity
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
    ):
        """
        Initialize status phase handler.

        Args:
            session_manager: SessionManager instance (created if not provided)
        """
        self.session_manager = session_manager or SessionManager()

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the status phase.

        Returns:
            Structured result with session status
        """
        start_time = datetime.now()

        # Check for active session
        if not self.session_manager.has_active_session():
            return self._no_session_response()

        # Get current session
        session = self.session_manager.current_session

        # Get session summary from manager
        summary = self.session_manager.get_session_summary()

        # Build phase progress
        phase_progress = self._get_phase_progress(session)

        # Build success criteria status
        criteria_status = self._get_criteria_status(session)

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return self._build_success_response(
            session=session,
            summary=summary,
            phase_progress=phase_progress,
            criteria_status=criteria_status,
            execution_time_ms=execution_time_ms,
        )

    def _get_phase_progress(self, session: SessionState) -> Dict[str, Any]:
        """Calculate progress through ritual phases."""
        phase_order = [
            RitualPhase.OPENING,
            RitualPhase.OPEN,
            RitualPhase.DESIGNING,
            RitualPhase.BUILDING,
            RitualPhase.CHECKING,
            RitualPhase.CLOSING,
            RitualPhase.CLOSED,
        ]

        current_phase = session.phase
        current_idx = 0

        for i, phase in enumerate(phase_order):
            if phase == current_phase:
                current_idx = i
                break

        # Calculate percentage (exclude NONE and CLOSED from progress)
        total_phases = len(phase_order) - 1  # Exclude CLOSED
        progress_percent = (current_idx / total_phases) * 100 if total_phases > 0 else 0

        return {
            "current_phase": current_phase.value,
            "phase_index": current_idx,
            "total_phases": total_phases,
            "progress_percent": round(progress_percent, 1),
            "phases_completed": [p.value for p in phase_order[:current_idx]],
        }

    def _get_criteria_status(self, session: SessionState) -> Dict[str, Any]:
        """Get success criteria completion status."""
        criteria = session.success_criteria
        total = len(criteria)

        # In a full implementation, we'd track which criteria are met
        # For now, we just show the list
        return {
            "total": total,
            "completed": 0,  # Would be tracked in full implementation
            "pending": total,
            "criteria": [
                {"criterion": c, "status": "pending"}
                for c in criteria
            ],
        }

    def _build_success_response(
        self,
        session: SessionState,
        summary: Dict[str, Any],
        phase_progress: Dict[str, Any],
        criteria_status: Dict[str, Any],
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Build success response for status display."""

        # Determine next steps based on current phase
        next_steps = self._get_next_steps(session.phase)

        # Build message based on phase
        message = self._get_phase_message(session.phase, session.task)

        return {
            "phase": "status",
            "session_id": session.session_id,
            "status": "success",
            "message": message,
            "next_steps": next_steps,
            "data": {
                "task": session.task,
                "complexity": session.complexity.value,
                "current_phase": session.phase.value,
                "phase_progress": phase_progress,
                "duration": summary.get("duration", "Unknown"),
                "started_at": session.started_at,
                "waves_completed": summary.get("waves_completed", 0),
                "waves_total": 3,
                "checks_passed": summary.get("checks_passed", 0),
                "checks_total": summary.get("checks_total", 0),
                "designs_count": len(session.designs),
                "builds_count": len(session.builds),
                "success_criteria": criteria_status,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "correlation_id": session.correlation_id,
            },
        }

    def _get_phase_message(self, phase: RitualPhase, task: str) -> str:
        """Get a contextual message for the current phase."""
        messages = {
            RitualPhase.NONE: "No active phase.",
            RitualPhase.OPENING: f"Opening session for: {task}",
            RitualPhase.OPEN: f"Session open. Ready to design for: {task}",
            RitualPhase.DESIGNING: f"Designing interfaces for: {task}",
            RitualPhase.BUILDING: f"Building in waves for: {task}",
            RitualPhase.CHECKING: f"Running safety checks for: {task}",
            RitualPhase.CLOSING: f"Closing session for: {task}",
            RitualPhase.CLOSED: f"Session completed for: {task}",
        }
        return messages.get(phase, f"Session in progress: {task}")

    def _get_next_steps(self, phase: RitualPhase) -> List[str]:
        """Get recommended next steps based on current phase."""
        steps = {
            RitualPhase.NONE: [
                "Use /ritual open to start a new session",
            ],
            RitualPhase.OPENING: [
                "Define success criteria",
                "Use /ritual design to begin",
            ],
            RitualPhase.OPEN: [
                "Use /ritual design to define interfaces",
                "Use /ritual build to start implementation",
            ],
            RitualPhase.DESIGNING: [
                "Complete interface design",
                "Use /ritual build when ready",
            ],
            RitualPhase.BUILDING: [
                "Complete current wave",
                "Use /ritual check when all waves pass",
            ],
            RitualPhase.CHECKING: [
                "Review safety checklist",
                "Use /ritual close when checks pass",
            ],
            RitualPhase.CLOSING: [
                "Add lessons learned",
                "Complete session close",
            ],
            RitualPhase.CLOSED: [
                "Review archived session",
                "Use /ritual open for new session",
            ],
        }
        return steps.get(phase, ["Use /ritual help for commands"])

    def _no_session_response(self) -> Dict[str, Any]:
        """Build response when no session is active."""
        return {
            "phase": "status",
            "session_id": None,
            "status": "no_session",
            "message": "No active session. Use /ritual open to start a new session.",
            "next_steps": [
                "Use /ritual open <task> to start a session",
                "Use /ritual help for command reference",
            ],
            "data": {
                "has_session": False,
            },
            "metadata": {},
        }


# Convenience function for direct invocation
async def get_ritual_status() -> Dict[str, Any]:
    """
    Get current ritual session status.

    Returns:
        Session status result
    """
    handler = StatusPhaseHandler()
    return await handler.execute()


def format_status_response(result: Dict[str, Any]) -> str:
    """
    Format status response for display.

    Returns formatted string suitable for terminal output.
    """
    if result["status"] == "no_session":
        return f"{result['message']}"

    lines = [
        "",
        "Ritual Session Status",
        "",
        f"**Task**: {result['data']['task']}",
        f"**Phase**: {result['data']['current_phase']}",
        f"**Duration**: {result['data']['duration']}",
        f"**Complexity**: {result['data']['complexity']}",
        "",
    ]

    # Phase progress
    progress = result['data']['phase_progress']
    lines.append(f"**Progress**: {progress['progress_percent']}%")

    # Visual progress bar
    filled = int(progress['progress_percent'] / 10)
    bar = "=" * filled + "-" * (10 - filled)
    lines.append(f"  [{bar}]")
    lines.append("")

    # Wave progress (if building)
    if result['data']['current_phase'] == 'building':
        waves = result['data']['waves_completed']
        lines.extend([
            "**Wave Progress**:",
            f"  Wave 1 (Core): {'Done' if waves >= 1 else 'Pending'}",
            f"  Wave 2 (Edge): {'Done' if waves >= 2 else 'Pending'}",
            f"  Wave 3 (Integration): {'Done' if waves >= 3 else 'Pending'}",
            "",
        ])

    # Check progress (if checking or later)
    checks_total = result['data']['checks_total']
    if checks_total > 0:
        checks_passed = result['data']['checks_passed']
        lines.extend([
            "**Safety Checks**:",
            f"  Passed: {checks_passed}/{checks_total}",
            "",
        ])

    # Success criteria
    criteria = result['data']['success_criteria']
    if criteria['total'] > 0:
        lines.append("**Success Criteria**:")
        for item in criteria['criteria']:
            status_icon = "[x]" if item['status'] == 'completed' else "[ ]"
            lines.append(f"  {status_icon} {item['criterion']}")
        lines.append("")

    # Next steps
    lines.append("**Next Steps**:")
    for step in result['next_steps']:
        lines.append(f"  - {step}")

    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    async def demo():
        # First open a session
        from open import open_ritual_session

        await open_ritual_session(
            task="Demo task for status",
            complexity="fast",
            success_criteria=["Works correctly", "Tests pass"],
        )

        # Then check status
        result = await get_ritual_status()
        print(format_status_response(result))

    asyncio.run(demo())