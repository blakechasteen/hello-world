#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hofstadter Scratchpad System
=============================

Persistent internal dialogue loops for recursive self-reflection.

Created: 2025-01-20
Updated: 2025-12-30 - Added backward-compatibility classes

Features:
- Persistent working memory (SQLite backend)
- Internal dialogue loops (system reasoning about its reasoning)
- Hofstadter-style strange loops (recursive self-reference)
- DS-STAR verification integration
- Complete thought provenance
- Tree visualization

Components:
- RecursiveScratchpad: Main entry point
- InternalDialogue: Dialogue loop engine
- StrangeLoop: Level-crossing detection
- ThoughtPersistence: SQLite backend

Backward-Compatible Components (for scratchpad_integration.py):
- Scratchpad: Simple entry-based scratchpad
- ScratchpadEntry: Single reasoning step

Usage:
    from hololoom.scratchpad import RecursiveScratchpad

    async with RecursiveScratchpad() as scratchpad:
        # Initial thought
        thought = await scratchpad.think("What is Thompson Sampling?")

        # Internal dialogue loop
        dialogue = await scratchpad.dialogue_loop(
            initial_thought=thought,
            max_depth=5
        )

        # Visualize
        print(dialogue.tree_visualization())

        # Persist
        await scratchpad.save_session("my_exploration")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from hololoom.core.recursive.scratchpad.recursive_scratchpad import (
    RecursiveScratchpad,
    Thought,
    ThoughtType,
    DialogueTree,
)

from hololoom.core.recursive.scratchpad.internal_dialogue import (
    InternalDialogue,
    DialogueStep,
    DialogueMode,
)

from hololoom.core.recursive.scratchpad.strange_loops import (
    StrangeLoop,
    LoopDetector,
    LevelCrossing,
)

from hololoom.core.recursive.scratchpad.persistence import (
    ThoughtPersistence,
    SessionManager,
)


# ============================================================================
# Backward-Compatible Simple Scratchpad (for scratchpad_integration.py)
# ============================================================================

@dataclass
class ScratchpadEntry:
    """
    Simple scratchpad entry for backward compatibility.

    Maps the thought -> action -> observation -> score pattern
    used by scratchpad_integration.py and test_scratchpad.py.

    Supports both positional args: ScratchpadEntry("T", "A", "O", 0.8)
    and keyword args with optional fields.
    """
    thought: str
    action: str
    observation: str
    score: Optional[float] = None
    iteration: Optional[int] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Scratchpad:
    """
    Simple entry-based scratchpad for backward compatibility.

    Provides the interface expected by scratchpad_integration.py and test_scratchpad.py:
    - .entries list
    - .add_entry() method
    - .get_history() method (returns List[ScratchpadEntry])
    - .get_last_n() method
    - .capacity property
    """

    def __init__(self, capacity: Optional[int] = None):
        self.entries: List[ScratchpadEntry] = []
        self.capacity = capacity

    def add_entry(
        self,
        thought: str,
        action: str,
        observation: str,
        score: Optional[float] = None,
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScratchpadEntry:
        """Add an entry to the scratchpad."""
        entry = ScratchpadEntry(
            thought=thought,
            action=action,
            observation=observation,
            score=score,
            iteration=iteration,
            metadata=metadata or {}
        )
        self.entries.append(entry)

        # Enforce capacity limit - keep only last N entries
        if self.capacity is not None and len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity:]

        return entry

    def get_history(self, max_entries: Optional[int] = None) -> List[ScratchpadEntry]:
        """Get copy of entries list."""
        entries_to_show = self.entries
        if max_entries:
            entries_to_show = self.entries[-max_entries:]
        # Return a copy, not the original
        return list(entries_to_show)

    def get_last_n(self, n: int) -> List[ScratchpadEntry]:
        """Get last N entries."""
        return list(self.entries[-n:])

    def clear(self):
        """Clear all entries."""
        self.entries = []

    def __len__(self) -> int:
        return len(self.entries)


# ============================================================================
# Stub Classes for Unused Imports (backward compatibility)
# ============================================================================

class LoopType(Enum):
    """Loop types for recursive refinement (stub for backward compatibility)."""
    REFINE = "refine"
    CRITIQUE = "critique"
    VERIFY = "verify"
    ELEGANCE = "elegance"
    HOFSTADTER = "hofstadter"


@dataclass
class LoopConfig:
    """Configuration for recursive loops (stub for backward compatibility)."""
    max_iterations: int = 3
    quality_threshold: float = 0.85
    loop_type: LoopType = LoopType.REFINE
    enable_learning: bool = True


@dataclass
class LoopResult:
    """Result of a recursive loop (stub for backward compatibility)."""
    success: bool = False
    iterations: int = 0
    final_quality: float = 0.0
    improvement: float = 0.0
    history: List[ScratchpadEntry] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a summary string of the loop result."""
        sign = "+" if self.improvement >= 0 else ""
        return (
            f"{self.iterations} iterations, "
            f"quality: {self.final_quality:.3f}, "
            f"improvement: {sign}{self.improvement:.3f}"
        )


class RecursiveEngine:
    """Recursive loop engine (stub for backward compatibility)."""

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.scratchpad = Scratchpad()

    def get_scratchpad(self) -> Scratchpad:
        """Get the scratchpad instance."""
        return self.scratchpad

    async def run_loop(
        self,
        initial_input: str,
        initial_quality: float
    ) -> LoopResult:
        """Run a recursive refinement loop (stub that simulates improvement)."""
        quality = initial_quality
        iterations = 0

        # Simulate improvement until reaching threshold or max iterations
        while quality < self.config.quality_threshold and iterations < self.config.max_iterations:
            iterations += 1
            # Simulate quality improvement
            quality = min(quality + 0.15, 1.0)

            # Add entry to scratchpad
            self.scratchpad.add_entry(
                thought=f"Iteration {iterations}: improving quality",
                action=self.config.loop_type.value,
                observation=f"Quality: {quality:.2f}",
                score=quality,
                iteration=iterations
            )

        success = quality >= self.config.quality_threshold
        improvement = quality - initial_quality

        return LoopResult(
            success=success,
            iterations=iterations,
            final_quality=quality,
            improvement=improvement,
            history=list(self.scratchpad.entries)
        )


__all__ = [
    # Main API
    "RecursiveScratchpad",
    "Thought",
    "ThoughtType",
    "DialogueTree",
    # Internal Dialogue
    "InternalDialogue",
    "DialogueStep",
    "DialogueMode",
    # Strange Loops
    "StrangeLoop",
    "LoopDetector",
    "LevelCrossing",
    # Persistence
    "ThoughtPersistence",
    "SessionManager",
    # Backward-Compatible Simple Scratchpad
    "Scratchpad",
    "ScratchpadEntry",
    # Backward-Compatible Stubs
    "RecursiveEngine",
    "LoopConfig",
    "LoopType",
    "LoopResult",
]
