# HoloLoom/ralph/config.py
"""
Ralph Loop Engine - Configuration.

Provides configuration for context thresholds, iteration limits,
state persistence, and automatic reset behavior.

Created: 2026-01-28
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import os


class ResetStrategy(Enum):
    """Strategy for handling context resets."""

    IMMEDIATE = "immediate"       # Reset immediately when threshold hit
    GRACEFUL = "graceful"         # Finish current operation, then reset
    CHECKPOINT = "checkpoint"     # Create checkpoint, then reset
    CONSOLIDATE = "consolidate"   # Run memory consolidation, then reset


class StateBackend(Enum):
    """Where to persist Ralph state between iterations."""

    FILE = "file"                 # JSON file on disk
    MEMORY = "memory"             # HoloLoom memory systems
    GIT = "git"                   # Git commits (Huntley's original approach)
    HYBRID = "hybrid"             # File + Memory (recommended)


@dataclass
class ContextThresholds:
    """
    Thresholds for automatic context reset.

    When context usage exceeds these thresholds, different actions trigger.
    """

    # Warning threshold (default 60%) - emit warning, prepare for reset
    warning_percent: float = 0.60

    # Consolidation threshold (default 75%) - start consolidating memories
    consolidation_percent: float = 0.75

    # Reset threshold (default 85%) - trigger context reset
    reset_percent: float = 0.85

    # Critical threshold (default 95%) - force immediate reset
    critical_percent: float = 0.95

    # Estimated context window size (tokens)
    # Claude: ~200k, but we're conservative
    estimated_context_size: int = 150_000

    # Minimum tokens to preserve for response generation
    response_reserve: int = 10_000

    def __post_init__(self):
        """Validate thresholds are in order."""
        assert 0 < self.warning_percent < self.consolidation_percent
        assert self.consolidation_percent < self.reset_percent
        assert self.reset_percent < self.critical_percent <= 1.0

    @classmethod
    def conservative(cls) -> "ContextThresholds":
        """Conservative thresholds - reset early, stay safe."""
        return cls(
            warning_percent=0.50,
            consolidation_percent=0.60,
            reset_percent=0.70,
            critical_percent=0.80,
        )

    @classmethod
    def aggressive(cls) -> "ContextThresholds":
        """Aggressive thresholds - maximize context usage."""
        return cls(
            warning_percent=0.70,
            consolidation_percent=0.80,
            reset_percent=0.90,
            critical_percent=0.95,
        )

    @classmethod
    def default(cls) -> "ContextThresholds":
        """Default balanced thresholds."""
        return cls()


@dataclass
class RalphConfig:
    """
    Configuration for the Ralph Loop Engine.

    Controls iteration behavior, state persistence, context monitoring,
    and reset strategies.
    """

    # --- Iteration Settings ---

    # Maximum iterations before forced stop (0 = unlimited)
    max_iterations: int = 100

    # Delay between iterations (seconds)
    iteration_delay: float = 0.5

    # Timeout per iteration (seconds, 0 = no timeout)
    iteration_timeout: float = 300.0

    # Whether to continue on iteration errors
    continue_on_error: bool = True

    # Maximum consecutive errors before stopping
    max_consecutive_errors: int = 3

    # --- State Persistence ---

    # Where to persist state
    state_backend: StateBackend = StateBackend.HYBRID

    # Path for file-based state (relative to working directory)
    state_path: Path = field(default_factory=lambda: Path(".ralph"))

    # Whether to auto-save state after each iteration
    auto_save: bool = True

    # Whether to create git commits for state changes
    git_commits: bool = False

    # Git commit message template
    git_commit_template: str = "ralph: iteration {iteration} - {summary}"

    # --- Context Management ---

    # Context thresholds for automatic reset
    context_thresholds: ContextThresholds = field(
        default_factory=ContextThresholds.default
    )

    # Reset strategy when threshold hit
    reset_strategy: ResetStrategy = ResetStrategy.CONSOLIDATE

    # Whether to enable automatic context monitoring
    auto_monitor: bool = True

    # How often to check context usage (iterations)
    monitor_interval: int = 1

    # --- Memory Integration ---

    # Whether to use HoloLoom memory systems
    use_memory_systems: bool = True

    # Whether to consolidate memories on reset
    consolidate_on_reset: bool = True

    # Whether to preserve hot patterns across resets
    preserve_hot_patterns: bool = True

    # Maximum memories to carry forward in handoff summary
    handoff_memory_limit: int = 50

    # --- Logging & Debugging ---

    # Enable verbose logging
    verbose: bool = False

    # Log file path (None = no file logging)
    log_path: Optional[Path] = None

    # Whether to save iteration traces for debugging
    save_traces: bool = True

    # --- Hooks ---

    # Enable hook system
    enable_hooks: bool = True

    # Built-in hooks to enable
    builtin_hooks: List[str] = field(default_factory=lambda: [
        "state_checkpoint",
        "memory_consolidation",
        "progress_logging",
    ])

    # --- Template Settings ---

    # Default loop template
    default_template: str = "iterative_refinement"

    # Custom template directory
    template_dir: Optional[Path] = None

    # --- Metadata ---

    # Additional metadata to include in state
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize paths and validate config."""
        if isinstance(self.state_path, str):
            self.state_path = Path(self.state_path)
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)
        if isinstance(self.template_dir, str):
            self.template_dir = Path(self.template_dir)

    @classmethod
    def default(cls) -> "RalphConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def minimal(cls) -> "RalphConfig":
        """Minimal configuration - fast, low overhead."""
        return cls(
            max_iterations=10,
            auto_save=False,
            auto_monitor=False,
            use_memory_systems=False,
            save_traces=False,
            enable_hooks=False,
        )

    @classmethod
    def production(cls) -> "RalphConfig":
        """Production configuration - full features, safety first."""
        return cls(
            max_iterations=100,
            iteration_timeout=600.0,
            continue_on_error=True,
            max_consecutive_errors=5,
            state_backend=StateBackend.HYBRID,
            auto_save=True,
            git_commits=True,
            context_thresholds=ContextThresholds.conservative(),
            reset_strategy=ResetStrategy.CONSOLIDATE,
            consolidate_on_reset=True,
            preserve_hot_patterns=True,
            save_traces=True,
        )

    @classmethod
    def development(cls) -> "RalphConfig":
        """Development configuration - verbose, debugging enabled."""
        return cls(
            max_iterations=20,
            verbose=True,
            save_traces=True,
            log_path=Path(".ralph/debug.log"),
            context_thresholds=ContextThresholds.aggressive(),
        )

    @classmethod
    def from_env(cls) -> "RalphConfig":
        """Create configuration from environment variables."""
        return cls(
            max_iterations=int(os.getenv("RALPH_MAX_ITERATIONS", "100")),
            iteration_timeout=float(os.getenv("RALPH_TIMEOUT", "300")),
            state_path=Path(os.getenv("RALPH_STATE_PATH", ".ralph")),
            verbose=os.getenv("RALPH_VERBOSE", "").lower() in ("1", "true", "yes"),
            git_commits=os.getenv("RALPH_GIT_COMMITS", "").lower() in ("1", "true", "yes"),
            auto_monitor=os.getenv("RALPH_AUTO_MONITOR", "1").lower() in ("1", "true", "yes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_iterations": self.max_iterations,
            "iteration_delay": self.iteration_delay,
            "iteration_timeout": self.iteration_timeout,
            "continue_on_error": self.continue_on_error,
            "max_consecutive_errors": self.max_consecutive_errors,
            "state_backend": self.state_backend.value,
            "state_path": str(self.state_path),
            "auto_save": self.auto_save,
            "git_commits": self.git_commits,
            "context_thresholds": {
                "warning_percent": self.context_thresholds.warning_percent,
                "consolidation_percent": self.context_thresholds.consolidation_percent,
                "reset_percent": self.context_thresholds.reset_percent,
                "critical_percent": self.context_thresholds.critical_percent,
            },
            "reset_strategy": self.reset_strategy.value,
            "auto_monitor": self.auto_monitor,
            "use_memory_systems": self.use_memory_systems,
            "consolidate_on_reset": self.consolidate_on_reset,
            "preserve_hot_patterns": self.preserve_hot_patterns,
            "handoff_memory_limit": self.handoff_memory_limit,
            "verbose": self.verbose,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RalphConfig":
        """Create from dictionary."""
        thresholds_data = data.pop("context_thresholds", {})
        thresholds = ContextThresholds(**thresholds_data) if thresholds_data else ContextThresholds.default()

        state_backend = data.pop("state_backend", "hybrid")
        if isinstance(state_backend, str):
            state_backend = StateBackend(state_backend)

        reset_strategy = data.pop("reset_strategy", "consolidate")
        if isinstance(reset_strategy, str):
            reset_strategy = ResetStrategy(reset_strategy)

        return cls(
            context_thresholds=thresholds,
            state_backend=state_backend,
            reset_strategy=reset_strategy,
            **data
        )
