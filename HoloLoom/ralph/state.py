# HoloLoom/ralph/state.py
"""
Ralph Loop Engine - State Management.

Handles state persistence, checkpoints, and handoff summaries.
State lives in files/memory, not in the context window - this is
the core insight of the Ralph technique.

"Progress doesn't persist in the LLM's context window -
it lives in your files and git history."

Created: 2026-01-28
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import hashlib
import subprocess
import asyncio


class CheckpointType(Enum):
    """Type of state checkpoint."""

    ITERATION = "iteration"       # Regular iteration checkpoint
    RESET = "reset"               # Created before context reset
    ERROR = "error"               # Created after error
    MANUAL = "manual"             # User-requested checkpoint
    COMPLETE = "complete"         # Task completed successfully


@dataclass
class TaskProgress:
    """Tracks progress on a specific task/subtask."""

    task_id: str
    description: str
    status: str  # pending, in_progress, completed, failed, blocked
    created_at: str
    updated_at: str
    completion_percent: float = 0.0
    notes: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)  # Files created/modified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completion_percent": self.completion_percent,
            "notes": self.notes,
            "blockers": self.blockers,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskProgress":
        return cls(**data)


@dataclass
class IterationRecord:
    """Record of a single iteration."""

    iteration_number: int
    started_at: str
    completed_at: Optional[str]
    duration_ms: float
    status: str  # success, error, timeout, skipped
    actions_taken: List[str]
    files_modified: List[str]
    memories_created: int = 0
    memories_recalled: int = 0
    confidence: float = 0.0
    error_message: Optional[str] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "actions_taken": self.actions_taken,
            "files_modified": self.files_modified,
            "memories_created": self.memories_created,
            "memories_recalled": self.memories_recalled,
            "confidence": self.confidence,
            "error_message": self.error_message,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        return cls(**data)


@dataclass
class RalphState:
    """
    Complete state of a Ralph loop execution.

    This is what gets persisted between iterations and across context resets.
    The key insight: everything needed to continue work is serializable.
    """

    # --- Identity ---
    loop_id: str
    task: str
    template: str
    created_at: str

    # --- Progress ---
    current_iteration: int = 0
    total_iterations: int = 0
    status: str = "pending"  # pending, running, paused, completed, failed
    completion_percent: float = 0.0

    # --- Task Tracking ---
    tasks: List[TaskProgress] = field(default_factory=list)
    current_task_id: Optional[str] = None

    # --- Iteration History ---
    iterations: List[IterationRecord] = field(default_factory=list)
    consecutive_errors: int = 0
    total_errors: int = 0

    # --- Context Resets ---
    reset_count: int = 0
    last_reset_at: Optional[str] = None
    reset_reasons: List[str] = field(default_factory=list)

    # --- Memory State ---
    hot_patterns: List[str] = field(default_factory=list)
    key_memories: List[str] = field(default_factory=list)
    learned_patterns: Dict[str, float] = field(default_factory=dict)

    # --- Artifacts ---
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    git_commits: List[str] = field(default_factory=list)

    # --- Metrics ---
    total_duration_ms: float = 0.0
    avg_iteration_ms: float = 0.0
    avg_confidence: float = 0.0

    # --- Metadata ---
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()

    def update_metrics(self):
        """Recalculate aggregate metrics from iteration history."""
        if not self.iterations:
            return

        successful = [i for i in self.iterations if i.status == "success"]
        if successful:
            self.total_duration_ms = sum(i.duration_ms for i in self.iterations)
            self.avg_iteration_ms = self.total_duration_ms / len(self.iterations)
            self.avg_confidence = sum(i.confidence for i in successful) / len(successful)

        self.total_iterations = len(self.iterations)
        self.total_errors = sum(1 for i in self.iterations if i.status == "error")

        # Update completion percent based on tasks
        if self.tasks:
            completed = sum(1 for t in self.tasks if t.status == "completed")
            self.completion_percent = completed / len(self.tasks) * 100

        self.updated_at = datetime.utcnow().isoformat()

    def add_iteration(self, record: IterationRecord):
        """Add an iteration record and update metrics."""
        self.iterations.append(record)
        self.current_iteration = record.iteration_number

        if record.status == "error":
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        # Track file changes
        for f in record.files_modified:
            if f not in self.files_modified:
                self.files_modified.append(f)

        self.update_metrics()

    def record_reset(self, reason: str):
        """Record a context reset."""
        self.reset_count += 1
        self.last_reset_at = datetime.utcnow().isoformat()
        self.reset_reasons.append(f"[{self.last_reset_at}] {reason}")
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "loop_id": self.loop_id,
            "task": self.task,
            "template": self.template,
            "created_at": self.created_at,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "status": self.status,
            "completion_percent": self.completion_percent,
            "tasks": [t.to_dict() for t in self.tasks],
            "current_task_id": self.current_task_id,
            "iterations": [i.to_dict() for i in self.iterations],
            "consecutive_errors": self.consecutive_errors,
            "total_errors": self.total_errors,
            "reset_count": self.reset_count,
            "last_reset_at": self.last_reset_at,
            "reset_reasons": self.reset_reasons,
            "hot_patterns": self.hot_patterns,
            "key_memories": self.key_memories,
            "learned_patterns": self.learned_patterns,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "git_commits": self.git_commits,
            "total_duration_ms": self.total_duration_ms,
            "avg_iteration_ms": self.avg_iteration_ms,
            "avg_confidence": self.avg_confidence,
            "metadata": self.metadata,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RalphState":
        """Deserialize from dictionary."""
        tasks = [TaskProgress.from_dict(t) for t in data.pop("tasks", [])]
        iterations = [IterationRecord.from_dict(i) for i in data.pop("iterations", [])]
        return cls(tasks=tasks, iterations=iterations, **data)

    def get_summary(self) -> str:
        """Get a human-readable summary of current state."""
        lines = [
            f"# Ralph Loop: {self.loop_id}",
            f"Task: {self.task}",
            f"Status: {self.status}",
            f"Progress: {self.completion_percent:.1f}% ({self.current_iteration} iterations)",
            f"Resets: {self.reset_count}",
            f"Confidence: {self.avg_confidence:.2f}",
            "",
            "## Recent Activity:",
        ]

        for record in self.iterations[-5:]:
            lines.append(f"  - Iteration {record.iteration_number}: {record.status} ({record.duration_ms:.0f}ms)")
            if record.summary:
                lines.append(f"    {record.summary}")

        if self.tasks:
            lines.append("")
            lines.append("## Tasks:")
            for task in self.tasks:
                status_icon = {"completed": "✅", "in_progress": "🔄", "pending": "⏳", "failed": "❌"}.get(task.status, "❓")
                lines.append(f"  {status_icon} {task.description} ({task.completion_percent:.0f}%)")

        return "\n".join(lines)


@dataclass
class StateCheckpoint:
    """
    A snapshot of Ralph state at a specific point in time.

    Used for recovery, debugging, and handoff between context windows.
    """

    checkpoint_id: str
    checkpoint_type: CheckpointType
    created_at: str
    state: RalphState
    reason: str = ""
    context_usage_percent: float = 0.0
    handoff_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_type": self.checkpoint_type.value,
            "created_at": self.created_at,
            "state": self.state.to_dict(),
            "reason": self.reason,
            "context_usage_percent": self.context_usage_percent,
            "handoff_summary": self.handoff_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateCheckpoint":
        state = RalphState.from_dict(data.pop("state"))
        checkpoint_type = CheckpointType(data.pop("checkpoint_type"))
        return cls(state=state, checkpoint_type=checkpoint_type, **data)


def generate_checkpoint_id(state: RalphState, checkpoint_type: CheckpointType) -> str:
    """Generate a unique checkpoint ID."""
    content = f"{state.loop_id}:{state.current_iteration}:{checkpoint_type.value}:{datetime.utcnow().isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def create_handoff_summary(state: RalphState, max_length: int = 2000) -> str:
    """
    Create a concise handoff summary for the next context window.

    This is the "bridge" between context resets - it captures the essential
    information needed for a fresh agent to continue work effectively.
    """
    lines = [
        "# RALPH HANDOFF SUMMARY",
        f"Loop ID: {state.loop_id}",
        f"Task: {state.task}",
        f"Template: {state.template}",
        f"Iteration: {state.current_iteration} | Resets: {state.reset_count}",
        f"Progress: {state.completion_percent:.1f}%",
        "",
        "## CURRENT STATE",
        f"Status: {state.status}",
    ]

    # Current task
    if state.current_task_id:
        current_task = next((t for t in state.tasks if t.task_id == state.current_task_id), None)
        if current_task:
            lines.append(f"Working on: {current_task.description}")
            if current_task.blockers:
                lines.append(f"Blockers: {', '.join(current_task.blockers)}")

    # Recent progress
    lines.append("")
    lines.append("## RECENT PROGRESS (last 3 iterations)")
    for record in state.iterations[-3:]:
        lines.append(f"- Iteration {record.iteration_number}: {record.summary or record.status}")
        if record.actions_taken:
            lines.append(f"  Actions: {', '.join(record.actions_taken[:3])}")

    # Key files modified
    if state.files_modified:
        lines.append("")
        lines.append("## KEY FILES MODIFIED")
        for f in state.files_modified[-10:]:
            lines.append(f"- {f}")

    # Remaining tasks
    pending_tasks = [t for t in state.tasks if t.status in ("pending", "in_progress")]
    if pending_tasks:
        lines.append("")
        lines.append("## REMAINING TASKS")
        for task in pending_tasks[:5]:
            status = "🔄" if task.status == "in_progress" else "⏳"
            lines.append(f"{status} {task.description}")

    # Key memories/patterns (for memory system integration)
    if state.hot_patterns:
        lines.append("")
        lines.append("## HOT PATTERNS (carry forward)")
        for pattern in state.hot_patterns[:5]:
            lines.append(f"- {pattern}")

    # Truncate if too long
    summary = "\n".join(lines)
    if len(summary) > max_length:
        summary = summary[:max_length - 50] + "\n\n[TRUNCATED - see full state file]"

    return summary


async def save_state(
    state: RalphState,
    path: Path,
    checkpoint_type: CheckpointType = CheckpointType.ITERATION,
    reason: str = "",
    context_usage_percent: float = 0.0,
    create_git_commit: bool = False,
) -> StateCheckpoint:
    """
    Save state to disk and optionally create git commit.

    Returns the checkpoint that was created.
    """
    # Ensure directory exists
    path.mkdir(parents=True, exist_ok=True)

    # Create checkpoint
    checkpoint_id = generate_checkpoint_id(state, checkpoint_type)
    checkpoint = StateCheckpoint(
        checkpoint_id=checkpoint_id,
        checkpoint_type=checkpoint_type,
        created_at=datetime.utcnow().isoformat(),
        state=state,
        reason=reason,
        context_usage_percent=context_usage_percent,
        handoff_summary=create_handoff_summary(state),
    )

    # Save current state
    state_file = path / "state.json"
    with open(state_file, "w") as f:
        json.dump(state.to_dict(), f, indent=2)

    # Save checkpoint
    checkpoint_file = path / f"checkpoint_{checkpoint_id}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)

    # Save handoff summary as markdown (easy to read)
    handoff_file = path / "HANDOFF.md"
    with open(handoff_file, "w") as f:
        f.write(checkpoint.handoff_summary)

    # Optionally create git commit
    if create_git_commit:
        await _create_git_commit(
            files=[str(state_file), str(checkpoint_file), str(handoff_file)],
            message=f"ralph: {checkpoint_type.value} checkpoint - iteration {state.current_iteration}",
        )

    return checkpoint


async def load_state(path: Path) -> Optional[RalphState]:
    """Load state from disk."""
    state_file = path / "state.json"
    if not state_file.exists():
        return None

    with open(state_file, "r") as f:
        data = json.load(f)

    return RalphState.from_dict(data)


async def load_checkpoint(path: Path, checkpoint_id: str) -> Optional[StateCheckpoint]:
    """Load a specific checkpoint."""
    checkpoint_file = path / f"checkpoint_{checkpoint_id}.json"
    if not checkpoint_file.exists():
        return None

    with open(checkpoint_file, "r") as f:
        data = json.load(f)

    return StateCheckpoint.from_dict(data)


async def list_checkpoints(path: Path) -> List[Dict[str, Any]]:
    """List all checkpoints for a loop."""
    if not path.exists():
        return []

    checkpoints = []
    for file in path.glob("checkpoint_*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                checkpoints.append({
                    "checkpoint_id": data["checkpoint_id"],
                    "checkpoint_type": data["checkpoint_type"],
                    "created_at": data["created_at"],
                    "reason": data.get("reason", ""),
                    "iteration": data["state"]["current_iteration"],
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)


async def _create_git_commit(files: List[str], message: str):
    """Create a git commit with the specified files."""
    try:
        # Stage files
        for file in files:
            subprocess.run(["git", "add", file], check=True, capture_output=True)

        # Commit
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        # Git commit failed (maybe not a git repo, or no changes)
        pass
