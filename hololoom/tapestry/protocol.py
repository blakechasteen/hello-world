"""
Tapestry Protocols and Data Classes

Core abstractions for session continuity:
- ThreadStatus: Weaving state enum
- Thread: Single task (immutable)
- Tapestry: Session state container
- SignalResult: Single verification outcome
- FabricCheckResult: Aggregated verification
- TapestryBackend: Storage protocol
- FabricSignal: Verification signal protocol

Design Principles:
- Protocol-first: Define WHAT, not HOW
- Immutable records: Thread is frozen for safety
- Factory methods: Tapestry.create() for valid state
- Runtime checkable: Protocols validate implementations

Created: December 2025
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ─────────────────────────────────────────────────────────────────
# Enums (Weaving Status)
# ─────────────────────────────────────────────────────────────────

class ThreadStatus(Enum):
    """
    Status of a single thread in the tapestry.

    Follows weaving metaphor:
    - UNWOVEN: Thread not yet woven into fabric
    - WEAVING: Currently being woven
    - WOVEN: Successfully integrated into fabric
    - TANGLED: Blocked or failed (needs attention)
    - UNRAVELED: Rolled back (removed from fabric)
    """
    UNWOVEN = "unwoven"      # Not yet started
    WEAVING = "weaving"      # Currently in progress
    WOVEN = "woven"          # Successfully completed
    TANGLED = "tangled"      # Blocked/failed
    UNRAVELED = "unraveled"  # Rolled back


# ─────────────────────────────────────────────────────────────────
# Verification Results
# ─────────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """
    Result from a single FabricSignal verification.

    Attributes:
        signal_name: Which signal produced this result
        passed: Whether this signal passed
        score: Quality score (0.0-1.0)
        details: Additional context for debugging
        is_blocker: If True and passed=False, blocks regardless of weight
    """
    signal_name: str
    passed: bool
    score: float  # 0.0-1.0
    details: dict[str, Any] = field(default_factory=dict)
    is_blocker: bool = False  # Hard failure, blocks regardless of weight


@dataclass
class FabricCheckResult:
    """
    Aggregated result from all verification signals.

    Systems thinking: holistic verification, not just "tests pass".

    Attributes:
        passed: Overall pass/fail
        confidence: Weighted confidence score (0.0-1.0)
        signals: Individual signal results
        blockers: List of blocking failures
        recommendations: Suggested improvements
    """
    passed: bool
    confidence: float  # Weighted aggregate
    signals: dict[str, SignalResult] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# Core Data Classes
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Thread:
    """
    Single task in the tapestry. Immutable for safety.

    Each thread represents one discrete unit of work that can be
    woven into the fabric. Threads may depend on other threads.

    Attributes:
        id: Unique identifier
        description: Human-readable task description
        status: Current weaving status
        created_at: When thread was created
        updated_at: Last status change
        commit_hash: Git commit when woven (if successful)
        fabric_check: Verification result (if checked)
        dependencies: IDs of threads that must be woven first
    """
    id: str
    description: str
    status: ThreadStatus
    created_at: datetime
    updated_at: datetime
    commit_hash: str | None = None
    fabric_check: FabricCheckResult | None = None
    dependencies: tuple = field(default_factory=tuple)  # Immutable

    def with_status(
        self,
        status: ThreadStatus,
        commit_hash: str | None = None,
        fabric_check: FabricCheckResult | None = None
    ) -> 'Thread':
        """Create new Thread with updated status (immutable update)."""
        return Thread(
            id=self.id,
            description=self.description,
            status=status,
            created_at=self.created_at,
            updated_at=datetime.utcnow(),
            commit_hash=commit_hash or self.commit_hash,
            fabric_check=fabric_check or self.fabric_check,
            dependencies=self.dependencies
        )


@dataclass
class Tapestry:
    """
    The woven record of work. Persisted to .hololoom/tapestry.json

    Represents a complete session with:
    - A goal (what we're trying to achieve)
    - Multiple threads (individual tasks)
    - Git tracking (initial and current commits)
    - Metadata (extensible context)

    Factory method Tapestry.create() ensures valid initial state.
    """
    loom_id: str
    goal: str
    threads: list[Thread]
    created_at: datetime
    updated_at: datetime
    initial_commit: str
    current_commit: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _generate_loom_id(cls) -> str:
        """Generate unique loom identifier."""
        return f"loom_{uuid.uuid4().hex[:12]}"

    @classmethod
    def create(
        cls,
        goal: str,
        thread_descriptions: list[str],
        dependencies: dict[str, list[str]] | None = None
    ) -> 'Tapestry':
        """
        Factory: Create tapestry with unwoven threads.

        Args:
            goal: Overall goal for this session
            thread_descriptions: List of task descriptions
            dependencies: Optional mapping of thread_id -> [dependency_ids]

        Returns:
            New Tapestry with all threads in UNWOVEN status
        """
        now = datetime.utcnow()
        dependencies = dependencies or {}

        threads = []
        for i, desc in enumerate(thread_descriptions):
            thread_id = str(i + 1)
            deps = tuple(dependencies.get(thread_id, []))
            threads.append(Thread(
                id=thread_id,
                description=desc,
                status=ThreadStatus.UNWOVEN,
                created_at=now,
                updated_at=now,
                dependencies=deps
            ))

        return cls(
            loom_id=cls._generate_loom_id(),
            goal=goal,
            threads=threads,
            created_at=now,
            updated_at=now,
            initial_commit="",
            current_commit=""
        )

    def _get_thread(self, thread_id: str) -> Thread | None:
        """Get thread by ID."""
        for thread in self.threads:
            if thread.id == thread_id:
                return thread
        return None

    def next_unwoven(self) -> Thread | None:
        """
        Get next unwoven thread (respecting dependencies).

        Returns the first UNWOVEN thread whose dependencies
        are all WOVEN. Returns None if no eligible threads.
        """
        for thread in self.threads:
            if thread.status == ThreadStatus.UNWOVEN:
                # Check dependencies are woven
                deps_met = all(
                    (dep_thread := self._get_thread(dep)) is not None
                    and dep_thread.status == ThreadStatus.WOVEN
                    for dep in thread.dependencies
                )
                if deps_met:
                    return thread
        return None

    def update_thread(
        self,
        thread_id: str,
        status: ThreadStatus,
        commit_hash: str | None = None,
        fabric_check: FabricCheckResult | None = None
    ) -> 'Tapestry':
        """
        Update a thread's status, returning new Tapestry.

        Tapestry is mutable but we return self for fluent API.
        """
        for i, thread in enumerate(self.threads):
            if thread.id == thread_id:
                self.threads[i] = thread.with_status(
                    status=status,
                    commit_hash=commit_hash,
                    fabric_check=fabric_check
                )
                break
        self.updated_at = datetime.utcnow()
        return self

    def get_status_summary(self) -> dict[str, int]:
        """Get count of threads by status."""
        summary = {status.value: 0 for status in ThreadStatus}
        for thread in self.threads:
            summary[thread.status.value] += 1
        return summary

    def is_complete(self) -> bool:
        """Check if all threads are woven."""
        return all(t.status == ThreadStatus.WOVEN for t in self.threads)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        return {
            "loom_id": self.loom_id,
            "goal": self.goal,
            "threads": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat(),
                    "updated_at": t.updated_at.isoformat(),
                    "commit_hash": t.commit_hash,
                    "fabric_check": {
                        "passed": t.fabric_check.passed,
                        "confidence": t.fabric_check.confidence,
                        "signals": {
                            name: {
                                "signal_name": sr.signal_name,
                                "passed": sr.passed,
                                "score": sr.score,
                                "details": sr.details,
                                "is_blocker": sr.is_blocker
                            }
                            for name, sr in t.fabric_check.signals.items()
                        },
                        "blockers": t.fabric_check.blockers,
                        "recommendations": t.fabric_check.recommendations
                    } if t.fabric_check else None,
                    "dependencies": list(t.dependencies)
                }
                for t in self.threads
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "initial_commit": self.initial_commit,
            "current_commit": self.current_commit,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Tapestry':
        """Deserialize from dictionary."""
        threads = []
        for t in data["threads"]:
            fabric_check = None
            if t.get("fabric_check"):
                fc = t["fabric_check"]
                fabric_check = FabricCheckResult(
                    passed=fc["passed"],
                    confidence=fc["confidence"],
                    signals={
                        name: SignalResult(
                            signal_name=sr["signal_name"],
                            passed=sr["passed"],
                            score=sr["score"],
                            details=sr.get("details", {}),
                            is_blocker=sr.get("is_blocker", False)
                        )
                        for name, sr in fc.get("signals", {}).items()
                    },
                    blockers=fc.get("blockers", []),
                    recommendations=fc.get("recommendations", [])
                )

            threads.append(Thread(
                id=t["id"],
                description=t["description"],
                status=ThreadStatus(t["status"]),
                created_at=datetime.fromisoformat(t["created_at"]),
                updated_at=datetime.fromisoformat(t["updated_at"]),
                commit_hash=t.get("commit_hash"),
                fabric_check=fabric_check,
                dependencies=tuple(t.get("dependencies", []))
            ))

        return cls(
            loom_id=data["loom_id"],
            goal=data["goal"],
            threads=threads,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            initial_commit=data.get("initial_commit", ""),
            current_commit=data.get("current_commit", ""),
            metadata=data.get("metadata", {})
        )


# ─────────────────────────────────────────────────────────────────
# Protocols (Define WHAT, not HOW)
# ─────────────────────────────────────────────────────────────────

@runtime_checkable
class TapestryBackend(Protocol):
    """
    Backend for tapestry persistence. Pluggable.

    Implementations:
    - JsonTapestryBackend: Atomic JSON file storage
    - SqliteTapestryBackend: SQLite for durability (optional)

    All methods are async for consistency with HoloLoom patterns.
    """

    async def load(self) -> Tapestry | None:
        """Load tapestry from storage. Returns None if not exists."""
        ...

    async def save(self, tapestry: Tapestry) -> None:
        """Save tapestry atomically."""
        ...

    async def exists(self) -> bool:
        """Check if tapestry exists in storage."""
        ...

    async def delete(self) -> None:
        """Delete tapestry from storage."""
        ...


@runtime_checkable
class FabricSignal(Protocol):
    """
    Single verification signal. Pluggable.

    Follows HoloLoom's Department pattern:
    each signal is independent, composable.

    Built-in signals (with weights):
    - TestSignal (0.20): Run pytest, check pass/fail
    - TroughSignal (0.20): AI slop detection
    - AlignmentSignal (0.15): Safety guardrails
    - ArchitectureSignal (0.15): Pattern adherence
    - DocumentationSignal (0.15): Docstring coverage
    - IntegrationSignal (0.15): Cross-system coherence

    Attributes:
        name: Unique signal identifier
        weight: 0.0-1.0, used in weighted aggregation
    """
    name: str
    weight: float  # 0.0-1.0

    async def check(self, thread: Thread, context: dict[str, Any]) -> SignalResult:
        """
        Run this verification signal.

        Args:
            thread: The thread being verified
            context: Additional context (files, test paths, etc.)

        Returns:
            SignalResult with pass/fail, score, and details
        """
        ...


# ─────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────

class TapestryError(Exception):
    """Base exception for Tapestry system."""
    pass


class NoTapestryError(TapestryError):
    """Raised when no tapestry exists to resume."""
    pass


class ThreadNotFoundError(TapestryError):
    """Raised when a thread ID is not found."""
    pass


class DirtyWorkingDirectoryError(TapestryError):
    """Raised when git working directory has uncommitted changes."""
    pass


class VerificationFailedError(TapestryError):
    """Raised when fabric verification fails with blockers."""
    pass
