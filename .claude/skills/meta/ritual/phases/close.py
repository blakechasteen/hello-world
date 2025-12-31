"""
/ritual close - Session Closing Phase
======================================

Wraps up the coding ritual session with:
- Git diff metrics (if available)
- Lessons learned capture
- Session summary generation
- Archive to .ritual/sessions/

Maps to: REENTRY → LANDED lifecycle states
Emits: ritual.session.closed event
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess
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
    ritual_completed,
)


class ClosePhaseHandler:
    """
    Handles /ritual close command.

    Wraps up session with:
    - Git diff metrics
    - Lessons learned
    - Session archival
    - Summary generation
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize close phase handler.

        Args:
            session_manager: SessionManager instance (created if not provided)
            event_bus: Optional EventBus for emitting events
        """
        self.session_manager = session_manager or SessionManager()
        self.event_bus = event_bus

    async def execute(
        self,
        lessons_learned: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the close phase.

        Args:
            lessons_learned: List of lessons from this session
            notes: Additional notes to include

        Returns:
            Structured result with session summary
        """
        start_time = datetime.now()

        # Check for active session
        if not self.session_manager.has_active_session():
            return self._error_response(
                "No active session to close. Use /ritual open to start a session."
            )

        # Get current session
        session = self.session_manager.current_session

        # Gather git metrics
        git_metrics = self._gather_git_metrics()

        # Add notes if provided
        if notes:
            session.notes = notes

        # Close session with lessons
        try:
            closed_session = self.session_manager.close_session(
                lessons=lessons_learned or []
            )
        except Exception as e:
            return self._error_response(f"Failed to close session: {e}")

        # Get session summary
        summary = self.session_manager.get_session_summary()

        # Archive session
        archive_path = self._archive_session(closed_session, summary, git_metrics)

        # Update lessons.md
        self._update_lessons_file(closed_session.lessons_learned)

        # Emit event if EventBus available
        events_emitted = []
        if self.event_bus:
            event_id = await self._emit_session_closed(closed_session, summary)
            if event_id:
                events_emitted.append(event_id)

        # Calculate execution time
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Build response
        return self._build_success_response(
            session=closed_session,
            summary=summary,
            git_metrics=git_metrics,
            archive_path=archive_path,
            execution_time_ms=execution_time_ms,
            events_emitted=events_emitted,
            needs_lessons=not lessons_learned,
        )

    def _gather_git_metrics(self) -> Dict[str, Any]:
        """Gather git diff metrics if available."""
        metrics = {
            "available": False,
            "files_changed": 0,
            "lines_added": 0,
            "lines_removed": 0,
        }

        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return metrics

            # Get diff stats
            diff_result = subprocess.run(
                ["git", "diff", "--stat", "--cached", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if diff_result.returncode == 0 and diff_result.stdout:
                metrics["available"] = True
                lines = diff_result.stdout.strip().split("\n")

                if lines:
                    # Parse summary line: "X files changed, Y insertions(+), Z deletions(-)"
                    summary_line = lines[-1] if lines else ""

                    import re

                    files_match = re.search(r"(\d+) files? changed", summary_line)
                    added_match = re.search(r"(\d+) insertions?\(\+\)", summary_line)
                    removed_match = re.search(r"(\d+) deletions?\(-\)", summary_line)

                    if files_match:
                        metrics["files_changed"] = int(files_match.group(1))
                    if added_match:
                        metrics["lines_added"] = int(added_match.group(1))
                    if removed_match:
                        metrics["lines_removed"] = int(removed_match.group(1))

            # Also get unstaged changes
            unstaged_result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if unstaged_result.returncode == 0 and unstaged_result.stdout:
                metrics["available"] = True
                # Add to existing metrics (simple approach)

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # Git not available or error - continue without metrics
            pass

        return metrics

    def _archive_session(
        self,
        session: SessionState,
        summary: Dict[str, Any],
        git_metrics: Dict[str, Any],
    ) -> Optional[Path]:
        """Archive session to .ritual/sessions/."""
        try:
            # Create sessions directory
            sessions_dir = Path(".ritual/sessions")
            sessions_dir.mkdir(parents=True, exist_ok=True)

            # Generate archive filename
            filename = f"{session.session_id}.md"
            archive_path = sessions_dir / filename

            # Build markdown content
            content = self._build_archive_markdown(session, summary, git_metrics)

            # Write file
            archive_path.write_text(content, encoding="utf-8")

            return archive_path

        except Exception:
            return None

    def _build_archive_markdown(
        self,
        session: SessionState,
        summary: Dict[str, Any],
        git_metrics: Dict[str, Any],
    ) -> str:
        """Build markdown content for session archive."""
        lines = [
            f"# Ritual Session: {session.task}",
            "",
            "## Session Info",
            "",
            f"- **Session ID**: {session.session_id}",
            f"- **Started**: {session.started_at}",
            f"- **Completed**: {session.completed_at}",
            f"- **Duration**: {summary.get('duration', 'Unknown')}",
            f"- **Complexity**: {session.complexity.value}",
            "",
            "## Success Criteria",
            "",
        ]

        for criterion in session.success_criteria:
            lines.append(f"- [ ] {criterion}")

        lines.extend([
            "",
            "## Phases Completed",
            "",
        ])

        # Add design records
        if session.designs:
            lines.append("### Designs")
            for design in session.designs:
                lines.append(f"- **{design.get('component', 'Unknown')}**: {design.get('rationale', '')}")
            lines.append("")

        # Add build records
        if session.builds:
            lines.append("### Builds")
            for build in session.builds:
                wave = build.get('wave', 'Unknown')
                status = build.get('status', 'Unknown')
                lines.append(f"- Wave {wave}: {status}")
            lines.append("")

        # Add check records
        if session.checks:
            lines.append("### Safety Checks")
            passed = sum(1 for c in session.checks if c.get('passed', False))
            lines.append(f"- Passed: {passed}/{len(session.checks)}")
            lines.append("")

        # Add git metrics
        if git_metrics.get("available"):
            lines.extend([
                "## Git Metrics",
                "",
                f"- Files changed: {git_metrics.get('files_changed', 0)}",
                f"- Lines added: {git_metrics.get('lines_added', 0)}",
                f"- Lines removed: {git_metrics.get('lines_removed', 0)}",
                "",
            ])

        # Add lessons learned
        if session.lessons_learned:
            lines.extend([
                "## Lessons Learned",
                "",
            ])
            for lesson in session.lessons_learned:
                lines.append(f"- {lesson}")
            lines.append("")

        # Add notes
        if session.notes:
            lines.extend([
                "## Notes",
                "",
                session.notes,
                "",
            ])

        return "\n".join(lines)

    def _update_lessons_file(self, lessons: List[str]) -> None:
        """Append lessons to .ritual/lessons.md."""
        if not lessons:
            return

        try:
            lessons_path = Path(".ritual/lessons.md")
            lessons_path.parent.mkdir(parents=True, exist_ok=True)

            # Read existing content
            existing = ""
            if lessons_path.exists():
                existing = lessons_path.read_text(encoding="utf-8")
            else:
                # Create header for new file
                existing = "# Lessons Learned\n\nCaptured from ritual sessions.\n\n"

            # Append new lessons with date
            date_str = datetime.now().strftime("%Y-%m-%d")
            new_content = f"\n## {date_str}\n\n"
            for lesson in lessons:
                new_content += f"- {lesson}\n"

            # Write updated file
            lessons_path.write_text(existing + new_content, encoding="utf-8")

        except Exception:
            pass  # Lessons file update is not critical

    def _build_success_response(
        self,
        session: SessionState,
        summary: Dict[str, Any],
        git_metrics: Dict[str, Any],
        archive_path: Optional[Path],
        execution_time_ms: float,
        events_emitted: List[str],
        needs_lessons: bool,
    ) -> Dict[str, Any]:
        """Build success response for session closing."""

        if needs_lessons:
            message = (
                f"Session closing. What did you learn from: {session.task}?"
            )
            next_steps = [
                "Share lessons learned from this session",
                "Use /ritual open to start a new session",
            ]
        else:
            message = "Ritual complete. Well done!"
            next_steps = [
                "Review archived session for future reference",
                "Use /ritual open to start a new session",
            ]

        return {
            "phase": "close",
            "session_id": session.session_id,
            "status": "success",
            "message": message,
            "next_steps": next_steps,
            "data": {
                "task": session.task,
                "duration": summary.get("duration", "Unknown"),
                "phases_completed": summary.get("phases_completed", []),
                "waves_completed": summary.get("waves_completed", 0),
                "checks_passed": summary.get("checks_passed", 0),
                "checks_total": summary.get("checks_total", 0),
                "git_metrics": git_metrics,
                "lessons_learned": session.lessons_learned,
                "archive_path": str(archive_path) if archive_path else None,
                "needs_lessons": needs_lessons,
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
            "phase": "close",
            "session_id": None,
            "status": "error",
            "message": message,
            "next_steps": [],
            "data": {},
            "metadata": {},
        }

    async def _emit_session_closed(
        self,
        session: SessionState,
        summary: Dict[str, Any],
    ) -> Optional[str]:
        """Emit ritual.session.closed event."""
        if not self.event_bus or not EVENT_BUS_AVAILABLE:
            return None

        try:
            # Create RitualContext from session state
            context = RitualContext(
                ritual_id=session.correlation_id or session.session_id,
                feature_name=session.task,
                started_at=session.started_at,
                current_phase="close",
            )

            # Create event using ritual_completed factory
            event = ritual_completed(context)

            # Add close-specific payload
            event["payload"].update({
                "session_id": session.session_id,
                "duration": summary.get("duration"),
                "lessons_learned": session.lessons_learned,
                "completed_at": session.completed_at,
            })

            # Update topic to be specific
            event["topic"] = "ritual.session.closed"

            # Record event in session
            session.record_event(event.get("event_id", ""))

            # Emit via EventBus
            await self.event_bus.emit(event)
            return event.get("event_id")

        except Exception:
            # Event emission failed, but don't break the flow
            return None


# Convenience function for direct invocation
async def close_ritual_session(
    lessons_learned: Optional[List[str]] = None,
    notes: Optional[str] = None,
    event_bus: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Close the current ritual session.

    Args:
        lessons_learned: List of lessons from this session
        notes: Additional notes to include
        event_bus: Optional EventBus for events

    Returns:
        Session closing result
    """
    handler = ClosePhaseHandler(event_bus=event_bus)
    return await handler.execute(
        lessons_learned=lessons_learned,
        notes=notes,
    )


def format_close_response(result: Dict[str, Any]) -> str:
    """
    Format close response for display.

    Returns formatted string suitable for terminal output.
    """
    if result["status"] == "error":
        return f"Error: {result['message']}"

    lines = [
        "",
        "Closing Ritual Session",
        "",
        "**Session Summary**:",
        f"- Task: {result['data']['task']}",
        f"- Duration: {result['data']['duration']}",
    ]

    if result['data'].get('waves_completed'):
        lines.append(f"- Waves completed: {result['data']['waves_completed']}/3")

    checks_passed = result['data'].get('checks_passed', 0)
    checks_total = result['data'].get('checks_total', 0)
    if checks_total:
        lines.append(f"- Safety checks: {checks_passed}/{checks_total} passed")

    git = result['data'].get('git_metrics', {})
    if git.get('available'):
        lines.extend([
            "",
            "**Git Metrics**:",
            f"- Files changed: {git.get('files_changed', 0)}",
            f"- Lines added: {git.get('lines_added', 0)}",
            f"- Lines removed: {git.get('lines_removed', 0)}",
        ])

    if result['data'].get('needs_lessons'):
        lines.extend([
            "",
            "**Lessons Learned**:",
            "What did you learn from this session?",
            "",
            "> (Enter your lessons, one per line)",
        ])
    else:
        lessons = result['data'].get('lessons_learned', [])
        if lessons:
            lines.extend([
                "",
                "**Lessons Captured**:",
            ])
            for lesson in lessons:
                lines.append(f"  - {lesson}")

    if result['data'].get('archive_path'):
        lines.extend([
            "",
            f"Session archived to: {result['data']['archive_path']}",
        ])

    if not result['data'].get('needs_lessons'):
        lines.extend([
            "",
            "Ritual complete. Well done!",
        ])

    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio

    async def demo():
        # First open a session
        from open import open_ritual_session

        await open_ritual_session(
            task="Demo task",
            complexity="fast",
            success_criteria=["Works correctly"],
        )

        # Then close it
        result = await close_ritual_session(
            lessons_learned=[
                "Always validate file content, not just extension",
                "S3 pre-signed URLs simplify upload flow",
            ],
            notes="Demo session for testing close functionality.",
        )
        print(format_close_response(result))

    asyncio.run(demo())