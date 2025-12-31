"""
Session State Manager for Coding Ritual
========================================

Created: 2025-12-30
Purpose: Manage ritual session state with file-based persistence

This module handles:
- Creating and loading sessions
- Persisting state to .ritual/ directory
- Tracking phases, timestamps, and metrics
- Managing session lifecycle

File Structure:
    .ritual/
    ├── sessions/                   # Archived session documents
    │   └── 2025-12-30_feature-x.md
    ├── current_session.json        # Active session state
    ├── lessons.md                  # Captured lessons
    └── metrics/                    # Historical metrics
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import os


class RitualPhase(Enum):
    """Phases of a coding ritual session."""
    NONE = "none"                # No active session
    OPENING = "opening"          # Session being initialized
    OPEN = "open"                # Session active, awaiting work
    DESIGNING = "designing"      # Protocol-first design phase
    BUILDING = "building"        # Wave validation in progress
    CHECKING = "checking"        # Safety checklist review
    CLOSING = "closing"          # Session being wrapped up
    CLOSED = "closed"            # Session completed


class BuildWave(Enum):
    """Wave validation stages."""
    WAVE_1 = "wave_1"  # Core functionality
    WAVE_2 = "wave_2"  # Edge cases
    WAVE_3 = "wave_3"  # Integration


class ComplexityLevel(Enum):
    """Session complexity levels (maps to HoloLoom modes)."""
    LITE = "lite"        # Quick tasks, minimal ceremony
    FAST = "fast"        # Standard tasks, balanced approach
    FULL = "full"        # Complex tasks, full ritual
    RESEARCH = "research"  # Open-ended exploration


@dataclass
class DesignRecord:
    """Record of a design decision."""
    component: str
    interface: str
    rationale: str
    dependencies: List[str] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BuildRecord:
    """Record of a build wave."""
    wave: BuildWave
    status: str  # pending, in_progress, passed, failed
    notes: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CheckRecord:
    """Record of a safety check."""
    item: str
    passed: bool
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionState:
    """
    Complete state of a ritual session.

    Persisted to .ritual/current_session.json
    """
    # Identity
    session_id: str
    task: str
    complexity: ComplexityLevel

    # Lifecycle
    phase: RitualPhase = RitualPhase.OPENING
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    closed_at: Optional[str] = None

    # Success criteria
    success_criteria: List[str] = field(default_factory=list)

    # Design records
    designs: List[Dict[str, Any]] = field(default_factory=list)

    # Build records
    current_wave: Optional[str] = None
    builds: List[Dict[str, Any]] = field(default_factory=list)

    # Check records
    checks: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    tests_passed: int = 0
    tests_failed: int = 0

    # Lessons and notes
    lessons_learned: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Event tracking (for EventBus integration)
    correlation_id: Optional[str] = None
    event_sequence: int = 0
    events_emitted: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums to strings
        data['phase'] = self.phase.value
        data['complexity'] = self.complexity.value
        if self.current_wave:
            data['current_wave'] = self.current_wave
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary."""
        # Convert string enums back
        data['phase'] = RitualPhase(data['phase'])
        data['complexity'] = ComplexityLevel(data['complexity'])
        return cls(**data)


class SessionManager:
    """
    Manages ritual session state with file-based persistence.

    Usage:
        manager = SessionManager()

        # Start a new session
        session = manager.create_session(
            task="Implement feature X",
            complexity=ComplexityLevel.FAST,
            success_criteria=["Criterion 1", "Criterion 2"]
        )

        # Update phase
        manager.transition_phase(RitualPhase.DESIGNING)

        # Add records
        manager.add_design(DesignRecord(...))
        manager.add_build(BuildRecord(...))

        # Close session
        manager.close_session(lessons=["Learned X", "Learned Y"])
    """

    # Default ritual directory (relative to project root)
    DEFAULT_RITUAL_DIR = ".ritual"

    def __init__(self, ritual_dir: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            ritual_dir: Path to .ritual directory. Defaults to .ritual in cwd.
        """
        self.ritual_dir = Path(ritual_dir) if ritual_dir else Path.cwd() / self.DEFAULT_RITUAL_DIR
        self._current_session: Optional[SessionState] = None
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create ritual directories if they don't exist."""
        (self.ritual_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (self.ritual_dir / "metrics").mkdir(parents=True, exist_ok=True)

        # Create .gitignore to exclude current_session.json (has runtime state)
        gitignore = self.ritual_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("current_session.json\n")

        # Create lessons.md if it doesn't exist
        lessons_file = self.ritual_dir / "lessons.md"
        if not lessons_file.exists():
            lessons_file.write_text("# Lessons Learned\n\nCapturing insights from coding ritual sessions.\n\n")

    @property
    def current_session(self) -> Optional[SessionState]:
        """Get current active session."""
        if self._current_session is None:
            self._current_session = self._load_session()
        return self._current_session

    @property
    def session_file(self) -> Path:
        """Path to current session state file."""
        return self.ritual_dir / "current_session.json"

    @property
    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        session = self.current_session
        if session is None:
            return False
        return session.phase not in [RitualPhase.NONE, RitualPhase.CLOSED]

    def _load_session(self) -> Optional[SessionState]:
        """Load session from file."""
        if not self.session_file.exists():
            return None
        try:
            data = json.loads(self.session_file.read_text(encoding='utf-8'))
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted session file - archive it and return None
            if self.session_file.exists():
                backup = self.ritual_dir / f"corrupted_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.session_file.rename(backup)
            return None

    def _save_session(self) -> None:
        """Save current session to file."""
        if self._current_session is None:
            return
        self._current_session.updated_at = datetime.now().isoformat()
        self.session_file.write_text(
            json.dumps(self._current_session.to_dict(), indent=2),
            encoding='utf-8'
        )

    def create_session(
        self,
        task: str,
        complexity: ComplexityLevel = ComplexityLevel.FAST,
        success_criteria: Optional[List[str]] = None,
        correlation_id: Optional[str] = None
    ) -> SessionState:
        """
        Create a new ritual session.

        Args:
            task: Description of the task
            complexity: Session complexity level
            success_criteria: List of success criteria
            correlation_id: Optional correlation ID for event tracking

        Returns:
            New SessionState

        Raises:
            RuntimeError: If there's already an active session
        """
        if self.has_active_session:
            raise RuntimeError(
                f"Active session already exists: {self.current_session.session_id}. "
                f"Use close_session() first or force_close_session() to abandon it."
            )

        # Generate session ID from date and task slug
        date_str = datetime.now().strftime("%Y-%m-%d")
        task_slug = task.lower().replace(" ", "-")[:30]
        session_id = f"{date_str}_{task_slug}"

        # Generate correlation_id if not provided
        if correlation_id is None:
            import uuid
            correlation_id = f"ritual-{uuid.uuid4().hex[:8]}"

        self._current_session = SessionState(
            session_id=session_id,
            task=task,
            complexity=complexity,
            success_criteria=success_criteria or [],
            correlation_id=correlation_id,
            phase=RitualPhase.OPEN
        )

        self._save_session()
        return self._current_session

    def transition_phase(self, new_phase: RitualPhase) -> SessionState:
        """
        Transition to a new phase.

        Args:
            new_phase: The phase to transition to

        Returns:
            Updated SessionState

        Raises:
            RuntimeError: If no active session
        """
        if not self.has_active_session:
            raise RuntimeError("No active session. Use create_session() first.")

        self._current_session.phase = new_phase
        self._current_session.event_sequence += 1
        self._save_session()
        return self._current_session

    def add_design(self, design: DesignRecord) -> SessionState:
        """Add a design decision record."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        self._current_session.designs.append(asdict(design))
        self._save_session()
        return self._current_session

    def add_build(self, build: BuildRecord) -> SessionState:
        """Add a build wave record."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        self._current_session.builds.append({
            'wave': build.wave.value,
            'status': build.status,
            'notes': build.notes,
            'timestamp': build.timestamp
        })
        self._current_session.current_wave = build.wave.value
        self._save_session()
        return self._current_session

    def add_check(self, check: CheckRecord) -> SessionState:
        """Add a safety check record."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        self._current_session.checks.append(asdict(check))
        self._save_session()
        return self._current_session

    def update_metrics(
        self,
        files_changed: Optional[int] = None,
        lines_added: Optional[int] = None,
        lines_removed: Optional[int] = None,
        tests_passed: Optional[int] = None,
        tests_failed: Optional[int] = None
    ) -> SessionState:
        """Update session metrics."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        if files_changed is not None:
            self._current_session.files_changed = files_changed
        if lines_added is not None:
            self._current_session.lines_added = lines_added
        if lines_removed is not None:
            self._current_session.lines_removed = lines_removed
        if tests_passed is not None:
            self._current_session.tests_passed = tests_passed
        if tests_failed is not None:
            self._current_session.tests_failed = tests_failed

        self._save_session()
        return self._current_session

    def add_note(self, note: str) -> SessionState:
        """Add a note to the session."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        self._current_session.notes.append(note)
        self._save_session()
        return self._current_session

    def add_success_criterion(self, criterion: str) -> SessionState:
        """Add a success criterion."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        self._current_session.success_criteria.append(criterion)
        self._save_session()
        return self._current_session

    def record_event(self, event_id: str) -> SessionState:
        """Record an emitted event ID."""
        if not self.has_active_session:
            raise RuntimeError("No active session.")

        self._current_session.events_emitted.append(event_id)
        self._current_session.event_sequence += 1
        self._save_session()
        return self._current_session

    def close_session(
        self,
        lessons: Optional[List[str]] = None
    ) -> SessionState:
        """
        Close the current session.

        Args:
            lessons: Lessons learned to capture

        Returns:
            Final SessionState
        """
        if not self.has_active_session:
            raise RuntimeError("No active session to close.")

        # Record lessons
        if lessons:
            self._current_session.lessons_learned.extend(lessons)
            self._append_lessons(lessons)

        # Update state
        self._current_session.phase = RitualPhase.CLOSED
        self._current_session.closed_at = datetime.now().isoformat()

        # Archive session
        self._archive_session()

        # Clear current session file
        if self.session_file.exists():
            self.session_file.unlink()

        final_session = self._current_session
        self._current_session = None

        return final_session

    def force_close_session(self) -> Optional[SessionState]:
        """
        Force close session without archiving (abandon).

        Returns:
            The abandoned session state, or None
        """
        if self.session_file.exists():
            self.session_file.unlink()

        abandoned = self._current_session
        self._current_session = None
        return abandoned

    def _append_lessons(self, lessons: List[str]) -> None:
        """Append lessons to the lessons file."""
        lessons_file = self.ritual_dir / "lessons.md"

        with open(lessons_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## {datetime.now().strftime('%Y-%m-%d')} - {self._current_session.task}\n\n")
            for lesson in lessons:
                f.write(f"- {lesson}\n")
            f.write("\n")

    def _archive_session(self) -> Path:
        """Archive the session to a markdown file."""
        session = self._current_session
        archive_path = self.ritual_dir / "sessions" / f"{session.session_id}.md"

        # Generate markdown
        md_lines = [
            f"# Ritual Session: {session.task}",
            f"",
            f"**Session ID**: {session.session_id}",
            f"**Complexity**: {session.complexity.value}",
            f"**Started**: {session.started_at}",
            f"**Closed**: {session.closed_at}",
            f"",
            f"## Success Criteria",
            f"",
        ]

        for criterion in session.success_criteria:
            md_lines.append(f"- [ ] {criterion}")

        md_lines.extend([
            f"",
            f"## Metrics",
            f"",
            f"- Files changed: {session.files_changed}",
            f"- Lines added: {session.lines_added}",
            f"- Lines removed: {session.lines_removed}",
            f"- Tests passed: {session.tests_passed}",
            f"- Tests failed: {session.tests_failed}",
            f"",
        ])

        if session.designs:
            md_lines.extend([
                f"## Design Decisions",
                f"",
            ])
            for design in session.designs:
                md_lines.append(f"### {design.get('component', 'Unknown')}")
                md_lines.append(f"")
                md_lines.append(f"**Interface**: {design.get('interface', 'N/A')}")
                md_lines.append(f"")
                md_lines.append(f"**Rationale**: {design.get('rationale', 'N/A')}")
                md_lines.append(f"")

        if session.builds:
            md_lines.extend([
                f"## Build Waves",
                f"",
            ])
            for build in session.builds:
                wave = build.get('wave', 'Unknown')
                status = build.get('status', 'unknown')
                md_lines.append(f"- **{wave}**: {status}")
                for note in build.get('notes', []):
                    md_lines.append(f"  - {note}")
            md_lines.append(f"")

        if session.checks:
            md_lines.extend([
                f"## Safety Checks",
                f"",
            ])
            for check in session.checks:
                passed = "✓" if check.get('passed') else "✗"
                item = check.get('item', 'Unknown')
                md_lines.append(f"- [{passed}] {item}")
            md_lines.append(f"")

        if session.lessons_learned:
            md_lines.extend([
                f"## Lessons Learned",
                f"",
            ])
            for lesson in session.lessons_learned:
                md_lines.append(f"- {lesson}")
            md_lines.append(f"")

        if session.notes:
            md_lines.extend([
                f"## Notes",
                f"",
            ])
            for note in session.notes:
                md_lines.append(f"- {note}")

        archive_path.write_text("\n".join(md_lines), encoding='utf-8')
        return archive_path

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session for display."""
        if not self.has_active_session:
            return {
                "active": False,
                "message": "No active ritual session"
            }

        session = self._current_session
        return {
            "active": True,
            "session_id": session.session_id,
            "task": session.task,
            "phase": session.phase.value,
            "complexity": session.complexity.value,
            "started_at": session.started_at,
            "success_criteria_count": len(session.success_criteria),
            "designs_count": len(session.designs),
            "current_wave": session.current_wave,
            "checks_passed": sum(1 for c in session.checks if c.get('passed')),
            "checks_total": len(session.checks),
            "correlation_id": session.correlation_id,
            "event_sequence": session.event_sequence
        }

    def list_past_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List past archived sessions."""
        sessions_dir = self.ritual_dir / "sessions"
        sessions = []

        for session_file in sorted(sessions_dir.glob("*.md"), reverse=True)[:limit]:
            # Extract basic info from filename
            session_id = session_file.stem
            sessions.append({
                "session_id": session_id,
                "file": str(session_file),
                "modified": datetime.fromtimestamp(session_file.stat().st_mtime).isoformat()
            })

        return sessions


# Convenience functions for skill usage
_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def reset_session_manager() -> None:
    """Reset the global session manager (for testing)."""
    global _manager
    _manager = None