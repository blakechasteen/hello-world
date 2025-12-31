"""
/ritual build - Wave Validation Build Phase
============================================

Implements a 3-wave validation build process:
- Wave 1: Core functionality (does it work?)
- Wave 2: Edge cases (what breaks it?)
- Wave 3: Integration (does it play nice?)

Maps to: ORBIT lifecycle state
Emits: ritual.build.wave_1, ritual.build.wave_2, ritual.build.wave_3 events
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.session_manager import (
    SessionManager,
    SessionState,
    RitualPhase,
)
from events import (
    EVENT_BUS_AVAILABLE,
    RitualContext,
    RitualEventType,
    create_ritual_event,
    phase_completed,
)


class WaveDefinition:
    """Definition of a build wave with validation criteria."""

    WAVE_1_CORE = {
        "name": "Core",
        "number": 1,
        "focus": "Does it work?",
        "description": "Verify core functionality works for the happy path",
        "checklist": [
            "Primary function executes without errors",
            "Expected output is produced",
            "Basic inputs are handled correctly",
            "No crashes or exceptions on normal use",
        ],
    }

    WAVE_2_EDGE = {
        "name": "Edge",
        "number": 2,
        "focus": "What breaks it?",
        "description": "Test edge cases and error conditions",
        "checklist": [
            "Empty inputs handled gracefully",
            "Invalid inputs produce clear errors",
            "Boundary conditions work correctly",
            "Resource limits are respected",
            "Concurrent access (if applicable) is safe",
        ],
    }

    WAVE_3_INTEGRATION = {
        "name": "Integration",
        "number": 3,
        "focus": "Does it play nice?",
        "description": "Verify integration with other components",
        "checklist": [
            "Works with existing components",
            "APIs are consistent with conventions",
            "No regression in related functionality",
            "Performance is acceptable",
            "Documentation is accurate",
        ],
    }

    @classmethod
    def get_wave(cls, wave_number: int) -> Dict[str, Any]:
        """Get wave definition by number."""
        waves = {
            1: cls.WAVE_1_CORE,
            2: cls.WAVE_2_EDGE,
            3: cls.WAVE_3_INTEGRATION,
        }
        return waves.get(wave_number, cls.WAVE_1_CORE)

    @classmethod
    def all_waves(cls) -> List[Dict[str, Any]]:
        """Get all wave definitions."""
        return [cls.WAVE_1_CORE, cls.WAVE_2_EDGE, cls.WAVE_3_INTEGRATION]


class BuildPhaseHandler:
    """
    Handles /ritual build command.

    Implements 3-wave validation build process:
    - Wave 1: Core functionality (happy path)
    - Wave 2: Edge cases (error handling)
    - Wave 3: Integration (compatibility)

    Each wave must pass before proceeding to the next.
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize build phase handler.

        Args:
            session_manager: SessionManager instance (created if not provided)
            event_bus: Optional EventBus for emitting events
        """
        self.session_manager = session_manager or SessionManager()
        self.event_bus = event_bus

    async def execute(
        self,
        wave: Optional[int] = None,
        wave_status: Optional[str] = None,
        wave_notes: Optional[str] = None,
        checklist_results: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the build phase.

        Args:
            wave: Wave number (1, 2, or 3). If None, shows current wave status.
            wave_status: Status of wave completion ("pass", "fail", "skip")
            wave_notes: Notes about wave completion
            checklist_results: Results of checklist items {item: passed}

        Returns:
            Structured result with wave status and next steps
        """
        start_time = datetime.now()

        # Check for active session
        if not self.session_manager.has_active_session():
            return self._error_response(
                "No active session. Use /ritual open to start a session."
            )

        # Get current session
        session = self.session_manager.current_session

        # Transition to building phase if not already there
        if session.phase not in [RitualPhase.DESIGNING, RitualPhase.BUILDING]:
            if session.phase == RitualPhase.OPEN:
                # Can start building directly from open
                self.session_manager.update_phase(RitualPhase.BUILDING)
                session = self.session_manager.current_session
            else:
                return self._error_response(
                    f"Cannot build from phase '{session.phase.value}'. "
                    f"Use /ritual status to check current phase."
                )
        elif session.phase == RitualPhase.DESIGNING:
            # Transition from designing to building
            self.session_manager.update_phase(RitualPhase.BUILDING)
            session = self.session_manager.current_session

        # Determine current wave progress
        waves_completed = self._get_completed_waves(session)
        current_wave = len(waves_completed) + 1

        # If no wave specified, show current wave status
        if wave is None:
            return self._show_wave_status(
                session=session,
                waves_completed=waves_completed,
                current_wave=current_wave,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        # Validate wave number
        if wave < 1 or wave > 3:
            return self._error_response(
                f"Invalid wave number '{wave}'. Valid waves: 1, 2, 3"
            )

        # Check if trying to skip waves
        if wave > current_wave:
            return self._error_response(
                f"Cannot start Wave {wave} before completing Wave {current_wave}. "
                f"Waves must be completed in order."
            )

        # If wave already completed, show its results
        if wave < current_wave:
            return self._show_completed_wave(
                session=session,
                wave=wave,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        # Process wave status update
        if wave_status:
            return await self._process_wave_result(
                session=session,
                wave=wave,
                status=wave_status,
                notes=wave_notes,
                checklist_results=checklist_results,
                start_time=start_time,
            )

        # Show wave details and checklist
        return self._show_wave_details(
            session=session,
            wave=wave,
            execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )

    def _get_completed_waves(self, session: SessionState) -> List[Dict[str, Any]]:
        """Get list of completed waves from session builds."""
        return [
            b for b in session.builds
            if b.get("status") in ["pass", "passed", "complete"]
        ]

    def _show_wave_status(
        self,
        session: SessionState,
        waves_completed: List[Dict[str, Any]],
        current_wave: int,
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Show overall wave progress."""
        wave_summary = []
        for w in WaveDefinition.all_waves():
            completed = any(
                b.get("wave") == w["number"]
                for b in waves_completed
            )
            wave_summary.append({
                "wave": w["number"],
                "name": w["name"],
                "focus": w["focus"],
                "status": "completed" if completed else (
                    "current" if w["number"] == current_wave else "pending"
                ),
            })

        if current_wave > 3:
            message = "All waves completed! Ready for /ritual check."
            next_steps = [
                "Use /ritual check to run safety checklist",
                "Use /ritual close to complete session",
            ]
        else:
            wave_def = WaveDefinition.get_wave(current_wave)
            message = f"Wave {current_wave} ({wave_def['name']}): {wave_def['focus']}"
            next_steps = [
                f"Use /ritual build {current_wave} to see Wave {current_wave} details",
                f"Use /ritual build {current_wave} pass to mark wave complete",
            ]

        return {
            "phase": "build",
            "session_id": session.session_id,
            "status": "success",
            "message": message,
            "next_steps": next_steps,
            "data": {
                "task": session.task,
                "current_wave": current_wave if current_wave <= 3 else None,
                "waves_completed": len(waves_completed),
                "waves_total": 3,
                "wave_summary": wave_summary,
                "all_waves_complete": current_wave > 3,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "correlation_id": session.correlation_id,
            },
        }

    def _show_wave_details(
        self,
        session: SessionState,
        wave: int,
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Show details for a specific wave."""
        wave_def = WaveDefinition.get_wave(wave)

        return {
            "phase": "build",
            "session_id": session.session_id,
            "status": "wave_details",
            "message": f"Wave {wave} ({wave_def['name']}): {wave_def['focus']}",
            "next_steps": [
                "Complete the checklist items below",
                f"Use /ritual build {wave} pass when all items pass",
                f"Use /ritual build {wave} fail if issues found",
            ],
            "data": {
                "task": session.task,
                "wave": wave,
                "wave_name": wave_def["name"],
                "wave_focus": wave_def["focus"],
                "wave_description": wave_def["description"],
                "checklist": [
                    {"item": item, "status": "pending"}
                    for item in wave_def["checklist"]
                ],
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "correlation_id": session.correlation_id,
            },
        }

    def _show_completed_wave(
        self,
        session: SessionState,
        wave: int,
        execution_time_ms: float,
    ) -> Dict[str, Any]:
        """Show results of a completed wave."""
        wave_def = WaveDefinition.get_wave(wave)

        # Find the build record for this wave
        build_record = next(
            (b for b in session.builds if b.get("wave") == wave),
            None
        )

        return {
            "phase": "build",
            "session_id": session.session_id,
            "status": "wave_completed",
            "message": f"Wave {wave} ({wave_def['name']}) already completed.",
            "next_steps": [
                "View other waves with /ritual build",
                "Continue to next wave",
            ],
            "data": {
                "wave": wave,
                "wave_name": wave_def["name"],
                "build_record": build_record,
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "correlation_id": session.correlation_id,
            },
        }

    async def _process_wave_result(
        self,
        session: SessionState,
        wave: int,
        status: str,
        notes: Optional[str],
        checklist_results: Optional[Dict[str, bool]],
        start_time: datetime,
    ) -> Dict[str, Any]:
        """Process and record wave completion result."""
        wave_def = WaveDefinition.get_wave(wave)

        # Normalize status
        status_normalized = status.lower()
        if status_normalized in ["pass", "passed", "complete", "done"]:
            status_normalized = "pass"
        elif status_normalized in ["fail", "failed", "error"]:
            status_normalized = "fail"
        elif status_normalized in ["skip", "skipped"]:
            status_normalized = "skip"
        else:
            return self._error_response(
                f"Invalid wave status '{status}'. Use: pass, fail, or skip"
            )

        # Create build record
        build_record = {
            "wave": wave,
            "wave_name": wave_def["name"],
            "status": status_normalized,
            "notes": notes,
            "checklist_results": checklist_results or {},
            "completed_at": datetime.now().isoformat(),
        }

        # Add build to session
        self.session_manager.add_build(build_record)

        # Emit event if EventBus available
        events_emitted = []
        if self.event_bus:
            event_id = await self._emit_wave_event(session, wave, status_normalized)
            if event_id:
                events_emitted.append(event_id)

        # Determine next steps
        if status_normalized == "pass":
            if wave < 3:
                next_wave = wave + 1
                next_wave_def = WaveDefinition.get_wave(next_wave)
                message = (
                    f"Wave {wave} passed! "
                    f"Next: Wave {next_wave} ({next_wave_def['name']})"
                )
                next_steps = [
                    f"Use /ritual build {next_wave} to start Wave {next_wave}",
                    f"Wave {next_wave} focus: {next_wave_def['focus']}",
                ]
            else:
                message = "All waves completed! Ready for safety checks."
                next_steps = [
                    "Use /ritual check to run safety checklist",
                    "Use /ritual close to complete session",
                ]
        elif status_normalized == "fail":
            message = f"Wave {wave} failed. Fix issues before proceeding."
            next_steps = [
                "Address the failing checklist items",
                f"Use /ritual build {wave} pass when fixed",
                "Use /ritual build to see wave status",
            ]
        else:  # skip
            if wave < 3:
                next_wave = wave + 1
                message = f"Wave {wave} skipped. Proceeding to Wave {next_wave}."
                next_steps = [
                    f"Use /ritual build {next_wave} to start Wave {next_wave}",
                    "Note: Skipped waves may indicate technical debt",
                ]
            else:
                message = "All waves processed. Ready for safety checks."
                next_steps = [
                    "Use /ritual check to run safety checklist",
                    "Consider revisiting skipped waves",
                ]

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "phase": "build",
            "session_id": session.session_id,
            "status": "success",
            "message": message,
            "next_steps": next_steps,
            "data": {
                "task": session.task,
                "wave": wave,
                "wave_name": wave_def["name"],
                "wave_status": status_normalized,
                "build_record": build_record,
                "all_waves_complete": wave >= 3 and status_normalized in ["pass", "skip"],
            },
            "metadata": {
                "execution_time_ms": round(execution_time_ms, 2),
                "events_emitted": events_emitted,
                "correlation_id": session.correlation_id,
            },
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Build error response."""
        return {
            "phase": "build",
            "session_id": None,
            "status": "error",
            "message": message,
            "next_steps": [],
            "data": {},
            "metadata": {},
        }

    async def _emit_wave_event(
        self,
        session: SessionState,
        wave: int,
        status: str,
    ) -> Optional[str]:
        """Emit ritual.build.wave_N event."""
        if not self.event_bus or not EVENT_BUS_AVAILABLE:
            return None

        try:
            # Create RitualContext from session state
            context = RitualContext(
                ritual_id=session.correlation_id or session.session_id,
                feature_name=session.task,
                started_at=session.started_at,
                current_phase="build",
            )

            # Create event using phase_completed factory (wave completion)
            event = phase_completed(context, phase_name=f"build.wave_{wave}")

            # Add wave-specific payload
            event["payload"].update({
                "session_id": session.session_id,
                "wave": wave,
                "status": status,
            })

            # Update topic to be wave-specific
            event["topic"] = f"ritual.build.wave_{wave}"

            # Record event in session
            session.record_event(event.get("event_id", ""))

            # Emit via EventBus
            await self.event_bus.emit(event)
            return event.get("event_id")

        except Exception:
            # Event emission failed, but don't break the flow
            return None


# Convenience function for direct invocation
async def start_build_wave(
    wave: Optional[int] = None,
    wave_status: Optional[str] = None,
    wave_notes: Optional[str] = None,
    checklist_results: Optional[Dict[str, bool]] = None,
    event_bus: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Start or update a build wave.

    Args:
        wave: Wave number (1, 2, or 3)
        wave_status: Status ("pass", "fail", "skip")
        wave_notes: Notes about completion
        checklist_results: Checklist item results
        event_bus: Optional EventBus for events

    Returns:
        Build phase result
    """
    handler = BuildPhaseHandler(event_bus=event_bus)
    return await handler.execute(
        wave=wave,
        wave_status=wave_status,
        wave_notes=wave_notes,
        checklist_results=checklist_results,
    )


def format_build_response(result: Dict[str, Any]) -> str:
    """
    Format build response for display.

    Returns formatted string suitable for terminal output.
    """
    if result["status"] == "error":
        return f"Error: {result['message']}"

    lines = [
        "",
        "Build Phase",
        "",
        f"**Status**: {result['message']}",
        "",
    ]

    # Show wave summary if available
    wave_summary = result["data"].get("wave_summary", [])
    if wave_summary:
        lines.append("**Wave Progress**:")
        for wave in wave_summary:
            status_icon = {
                "completed": "[x]",
                "current": "[>]",
                "pending": "[ ]",
            }.get(wave["status"], "[ ]")
            lines.append(
                f"  {status_icon} Wave {wave['wave']} ({wave['name']}): "
                f"{wave['focus']}"
            )
        lines.append("")

    # Show checklist if showing wave details
    checklist = result["data"].get("checklist", [])
    if checklist:
        lines.append("**Checklist**:")
        for item in checklist:
            status_icon = "[x]" if item.get("status") == "passed" else "[ ]"
            lines.append(f"  {status_icon} {item['item']}")
        lines.append("")

    # Show build record if wave was processed
    build_record = result["data"].get("build_record")
    if build_record:
        lines.extend([
            "**Build Record**:",
            f"  Wave: {build_record['wave']} ({build_record['wave_name']})",
            f"  Status: {build_record['status']}",
        ])
        if build_record.get("notes"):
            lines.append(f"  Notes: {build_record['notes']}")
        lines.append("")

    # Next steps
    lines.append("**Next Steps**:")
    for step in result["next_steps"]:
        lines.append(f"  - {step}")

    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    async def demo():
        # First open a session
        from open import open_ritual_session

        await open_ritual_session(
            task="Demo build task",
            complexity="fast",
            success_criteria=["Core works", "Handles errors"],
        )

        # Check wave status
        result = await start_build_wave()
        print(format_build_response(result))
        print("\n" + "=" * 50 + "\n")

        # Show wave 1 details
        result = await start_build_wave(wave=1)
        print(format_build_response(result))
        print("\n" + "=" * 50 + "\n")

        # Pass wave 1
        result = await start_build_wave(
            wave=1,
            wave_status="pass",
            wave_notes="All core tests passing",
        )
        print(format_build_response(result))

    asyncio.run(demo())