#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strange Loops Detector
======================

Hofstadter-style recursive self-reference and level-crossing detection.

Created: 2025-01-20

Philosophy:
"A strange loop is when hierarchical levels, each of which is normally
'above' or 'below' one another, loop back to create a tangled hierarchy."
— Douglas Hofstadter, "Gödel, Escher, Bach"

Types of Loops:
1. **Direct Self-Reference**: Thought references itself
2. **Cyclic Reference**: A→B→C→A pattern
3. **Level-Crossing**: Meta-thought affects object-level thought
4. **Strange Loop**: Upward and downward causality simultaneously
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
import re

from hololoom.recursive.scratchpad.recursive_scratchpad import (
    Thought,
    ThoughtType,
    DialogueTree,
)


class LoopType(Enum):
    """Type of detected loop."""
    DIRECT_SELF_REFERENCE = "direct_self_reference"
    CYCLIC_REFERENCE = "cyclic_reference"
    LEVEL_CROSSING = "level_crossing"
    STRANGE_LOOP = "strange_loop"
    META_REASONING = "meta_reasoning"


@dataclass
class LevelCrossing:
    """Level-crossing event."""
    upper_thought: Thought    # Meta-level thought
    lower_thought: Thought    # Object-level thought
    crossing_type: str        # How they interact
    strength: float           # Strength of crossing (0-1)


@dataclass
class StrangeLoop:
    """Detected strange loop."""
    loop_type: LoopType
    thoughts: List[Thought]
    description: str
    strength: float  # How "strange" (0-1)
    level_crossings: List[LevelCrossing]


class LoopDetector:
    """
    Detect strange loops in dialogue trees.

    Identifies:
    - Direct self-reference
    - Cyclic reasoning patterns
    - Level-crossing between meta and object levels
    - True strange loops (tangled hierarchies)
    """

    def __init__(
        self,
        min_loop_size: int = 2,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize loop detector.

        Args:
            min_loop_size: Minimum cycle size
            similarity_threshold: Text similarity threshold
        """
        self.min_loop_size = min_loop_size
        self.similarity_threshold = similarity_threshold

    def detect_loops(self, tree: DialogueTree) -> Dict[str, StrangeLoop]:
        """
        Detect all loops in dialogue tree.

        Args:
            tree: Dialogue tree

        Returns:
            Dict of thought_id → StrangeLoop
        """
        loops = {}

        # 1. Direct self-reference
        for thought in tree.thoughts.values():
            if self._is_self_referential(thought):
                loops[thought.id] = StrangeLoop(
                    loop_type=LoopType.DIRECT_SELF_REFERENCE,
                    thoughts=[thought],
                    description=f"Thought directly references itself",
                    strength=0.9,
                    level_crossings=[]
                )

        # 2. Cyclic references
        cycles = self._find_cycles(tree)
        for cycle in cycles:
            loop_id = cycle[0].id
            loops[loop_id] = StrangeLoop(
                loop_type=LoopType.CYCLIC_REFERENCE,
                thoughts=cycle,
                description=f"Cyclic reasoning: {len(cycle)} thoughts",
                strength=0.7,
                level_crossings=[]
            )

        # 3. Level crossings
        crossings = self._detect_level_crossings(tree)
        for crossing in crossings:
            loop_id = crossing.upper_thought.id
            if loop_id not in loops:
                loops[loop_id] = StrangeLoop(
                    loop_type=LoopType.LEVEL_CROSSING,
                    thoughts=[crossing.upper_thought, crossing.lower_thought],
                    description=f"Meta-thought affects object-thought",
                    strength=crossing.strength,
                    level_crossings=[crossing]
                )

        # 4. Strange loops (level crossing + cycle)
        for loop in list(loops.values()):
            if loop.loop_type == LoopType.CYCLIC_REFERENCE:
                # Check if cycle involves level crossing
                has_crossing = any(
                    self._involves_level_crossing(t, tree)
                    for t in loop.thoughts
                )
                if has_crossing:
                    loop.loop_type = LoopType.STRANGE_LOOP
                    loop.description = "True strange loop: cycle with level-crossing"
                    loop.strength = 1.0

        # 5. Meta-reasoning loops
        meta_loops = self._detect_meta_reasoning(tree)
        for thought_id, meta_loop in meta_loops.items():
            if thought_id not in loops:
                loops[thought_id] = meta_loop

        return loops

    def _is_self_referential(self, thought: Thought) -> bool:
        """Check if thought references itself."""
        text_lower = thought.text.lower()

        # Patterns of self-reference
        patterns = [
            r'\bmy (thinking|reasoning|understanding|assumption)',
            r'\bi (think|assume|believe|reason)',
            r'this (thought|reasoning|assumption)',
            r'in thinking (this|that)',
        ]

        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _find_cycles(self, tree: DialogueTree) -> List[List[Thought]]:
        """Find cyclic patterns in semantic content."""
        cycles = []
        thoughts = list(tree.thoughts.values())

        # Look for semantic similarity cycles
        for i, t1 in enumerate(thoughts):
            for j in range(i + self.min_loop_size, min(i + 10, len(thoughts))):
                t2 = thoughts[j]

                # Check if semantically similar
                if self._semantic_similarity(t1.text, t2.text) > self.similarity_threshold:
                    # Found potential cycle
                    cycle_thoughts = thoughts[i:j+1]
                    if len(cycle_thoughts) >= self.min_loop_size:
                        cycles.append(cycle_thoughts)

        return cycles

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity (simple Jaccard for now)."""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _detect_level_crossings(self, tree: DialogueTree) -> List[LevelCrossing]:
        """Detect level crossings (meta ↔ object)."""
        crossings = []

        for thought in tree.thoughts.values():
            # Is this a meta-thought?
            if self._is_meta_thought(thought):
                # Find object-thoughts it might affect
                path = tree.get_path(thought.id)
                for ancestor in path[:-1]:
                    if not self._is_meta_thought(ancestor):
                        # Found level crossing
                        crossing = LevelCrossing(
                            upper_thought=thought,
                            lower_thought=ancestor,
                            crossing_type="meta_to_object",
                            strength=0.8
                        )
                        crossings.append(crossing)

        return crossings

    def _is_meta_thought(self, thought: Thought) -> bool:
        """Check if thought is meta-level (about thinking itself)."""
        # Reflections are meta by definition
        if thought.type == ThoughtType.REFLECTION:
            return True

        text_lower = thought.text.lower()

        # Meta-reasoning patterns
        meta_patterns = [
            r'\b(thinking|reasoning|assuming|believing)',
            r'\bmy (process|approach|method|reasoning)',
            r'\b(meta|about my)',
            r'\breflect',
            r'\bconsider how',
            r'\bif i think',
        ]

        for pattern in meta_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def _involves_level_crossing(self, thought: Thought, tree: DialogueTree) -> bool:
        """Check if thought involves level crossing."""
        # Get path to root
        path = tree.get_path(thought.id)

        # Check if path has both meta and object thoughts
        has_meta = any(self._is_meta_thought(t) for t in path)
        has_object = any(not self._is_meta_thought(t) for t in path)

        return has_meta and has_object

    def _detect_meta_reasoning(self, tree: DialogueTree) -> Dict[str, StrangeLoop]:
        """Detect meta-reasoning patterns."""
        meta_loops = {}

        for thought in tree.thoughts.values():
            # Check if thought questions its own reasoning
            if self._questions_own_reasoning(thought):
                # Get context
                path = tree.get_path(thought.id)

                meta_loops[thought.id] = StrangeLoop(
                    loop_type=LoopType.META_REASONING,
                    thoughts=path,
                    description="Thought questions its own reasoning process",
                    strength=0.85,
                    level_crossings=[]
                )

        return meta_loops

    def _questions_own_reasoning(self, thought: Thought) -> bool:
        """Check if thought questions its own reasoning."""
        text_lower = thought.text.lower()

        patterns = [
            r'is my reasoning',
            r'am i (thinking|reasoning|assuming)',
            r'what if (my|i\'m)',
            r'is this (reasoning|thinking) (correct|valid|sound)',
            r'flawed',
            r'wrong about',
        ]

        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def visualize_loop(self, loop: StrangeLoop) -> str:
        """Visualize a strange loop."""
        lines = [
            f"{'='*60}",
            f"Strange Loop: {loop.loop_type.value}",
            f"Strength: {loop.strength:.2f}",
            f"Description: {loop.description}",
            f"{'='*60}",
            "",
            "Thoughts in Loop:",
        ]

        for i, thought in enumerate(loop.thoughts):
            marker = "↻" if i == len(loop.thoughts) - 1 else "→"
            lines.append(f"  {marker} [{thought.type.value}] {thought.text[:60]}...")

        if loop.level_crossings:
            lines.append("")
            lines.append("Level Crossings:")
            for crossing in loop.level_crossings:
                lines.append(f"  ↑↓ {crossing.upper_thought.text[:40]}... ↔ {crossing.lower_thought.text[:40]}...")

        return "\n".join(lines)


class StrangeLoopAnalyzer:
    """Analyze strange loops for insights."""

    @staticmethod
    def analyze_loop_density(tree: DialogueTree, loops: Dict[str, StrangeLoop]) -> float:
        """
        Compute loop density (loops per thought).

        High density = lots of self-reference and meta-reasoning
        """
        if not tree.thoughts:
            return 0.0

        return len(loops) / len(tree.thoughts)

    @staticmethod
    def get_strongest_loops(
        loops: Dict[str, StrangeLoop],
        n: int = 5
    ) -> List[StrangeLoop]:
        """Get N strongest loops."""
        return sorted(
            loops.values(),
            key=lambda l: l.strength,
            reverse=True
        )[:n]

    @staticmethod
    def classify_loop_complexity(loop: StrangeLoop) -> str:
        """
        Classify loop complexity.

        Returns:
            "simple", "moderate", "complex", "tangled"
        """
        if loop.loop_type == LoopType.DIRECT_SELF_REFERENCE:
            return "simple"
        elif loop.loop_type == LoopType.CYCLIC_REFERENCE:
            return "moderate"
        elif loop.loop_type == LoopType.LEVEL_CROSSING:
            return "complex"
        elif loop.loop_type == LoopType.STRANGE_LOOP:
            return "tangled"
        else:
            return "moderate"
