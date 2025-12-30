#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hofstadter Scratchpad System
=============================

Persistent internal dialogue loops for recursive self-reflection.

Created: 2025-01-20

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

Usage:
    from HoloLoom.scratchpad import RecursiveScratchpad

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

from HoloLoom.recursive.scratchpad.recursive_scratchpad import (
    RecursiveScratchpad,
    Thought,
    ThoughtType,
    DialogueTree,
)

from HoloLoom.recursive.scratchpad.internal_dialogue import (
    InternalDialogue,
    DialogueStep,
    DialogueMode,
)

from HoloLoom.recursive.scratchpad.strange_loops import (
    StrangeLoop,
    LoopDetector,
    LevelCrossing,
)

from HoloLoom.recursive.scratchpad.persistence import (
    ThoughtPersistence,
    SessionManager,
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
]
