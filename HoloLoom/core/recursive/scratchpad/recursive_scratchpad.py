#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive Scratchpad
====================

Persistent working memory for internal dialogue and self-reflection.

Created: 2025-01-20

Architecture:
- Thought: Individual unit of reasoning
- DialogueTree: Hierarchical thought structure
- RecursiveScratchpad: Main orchestrator
- DS-STAR verification at each thought step

Philosophy:
"Thinking is not a straight line. It's a recursive exploration where each
thought spawns questions, which spawn deeper thoughts, forming a tree of
understanding."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import uuid
import asyncio
from pathlib import Path


class ThoughtType(Enum):
    """Type of thought in scratchpad."""
    INITIAL = "initial"           # Starting thought
    QUESTION = "question"         # Self-posed question
    ANSWER = "answer"             # Answer to question
    REFLECTION = "reflection"     # Meta-reflection
    VERIFICATION = "verification" # DS-STAR check
    INSIGHT = "insight"           # Emergent understanding
    CONTRADICTION = "contradiction"  # Detected inconsistency
    SYNTHESIS = "synthesis"       # Integration of thoughts


@dataclass
class Thought:
    """Individual unit of reasoning."""
    id: str
    text: str
    type: ThoughtType
    level: int  # Depth in dialogue tree
    timestamp: datetime
    parent_id: Optional[str] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification: Optional[Dict[str, Any]] = None  # DS-STAR results

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.type, str):
            self.type = ThoughtType(self.type)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thought."""
        return {
            'id': self.id,
            'text': self.text,
            'type': self.type.value,
            'level': self.level,
            'timestamp': self.timestamp.isoformat(),
            'parent_id': self.parent_id,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'verification': self.verification,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thought':
        """Deserialize thought."""
        return cls(
            id=data['id'],
            text=data['text'],
            type=ThoughtType(data['type']),
            level=data['level'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            parent_id=data.get('parent_id'),
            confidence=data.get('confidence', 0.5),
            metadata=data.get('metadata', {}),
            verification=data.get('verification'),
        )


@dataclass
class DialogueTree:
    """Hierarchical structure of thoughts."""
    root: Thought
    thoughts: Dict[str, Thought] = field(default_factory=dict)

    def __post_init__(self):
        """Index root thought."""
        self.thoughts[self.root.id] = self.root

    def add_thought(self, thought: Thought) -> None:
        """Add thought to tree."""
        self.thoughts[thought.id] = thought

    def get_children(self, thought_id: str) -> List[Thought]:
        """Get child thoughts."""
        return [
            t for t in self.thoughts.values()
            if t.parent_id == thought_id
        ]

    def get_path(self, thought_id: str) -> List[Thought]:
        """Get path from root to thought."""
        path = []
        current_id = thought_id

        while current_id:
            thought = self.thoughts.get(current_id)
            if not thought:
                break
            path.insert(0, thought)
            current_id = thought.parent_id

        return path

    def get_depth(self) -> int:
        """Get maximum depth of tree."""
        if not self.thoughts:
            return 0
        return max(t.level for t in self.thoughts.values())

    def tree_visualization(self, max_width: int = 80) -> str:
        """ASCII tree visualization."""
        lines = []

        def visit(thought: Thought, prefix: str = "", is_last: bool = True):
            # Format thought text
            marker = "└─" if is_last else "├─"
            thought_text = thought.text[:max_width-len(prefix)-10]
            if len(thought.text) > len(thought_text):
                thought_text += "..."

            # Add type and confidence
            type_marker = {
                ThoughtType.INITIAL: "🌱",
                ThoughtType.QUESTION: "❓",
                ThoughtType.ANSWER: "💡",
                ThoughtType.REFLECTION: "🤔",
                ThoughtType.VERIFICATION: "✓",
                ThoughtType.INSIGHT: "⚡",
                ThoughtType.CONTRADICTION: "⚠️",
                ThoughtType.SYNTHESIS: "🔗",
            }.get(thought.type, "•")

            conf_str = f"[{thought.confidence:.2f}]"
            lines.append(f"{prefix}{marker} {type_marker} {thought_text} {conf_str}")

            # Visit children
            children = self.get_children(thought.id)
            for i, child in enumerate(children):
                extension = "  " if is_last else "│ "
                visit(child, prefix + extension, i == len(children) - 1)

        visit(self.root)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree."""
        return {
            'root_id': self.root.id,
            'thoughts': {
                tid: t.to_dict() for tid, t in self.thoughts.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTree':
        """Deserialize tree."""
        thoughts = {
            tid: Thought.from_dict(tdata)
            for tid, tdata in data['thoughts'].items()
        }
        root = thoughts[data['root_id']]
        tree = cls(root=root, thoughts=thoughts)
        return tree


class RecursiveScratchpad:
    """
    Main scratchpad orchestrator.

    Enables:
    - Persistent working memory
    - Internal dialogue loops
    - Hofstadter strange loops
    - DS-STAR verification
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_verification: bool = True,
        max_dialogue_depth: int = 10,
    ):
        """
        Initialize scratchpad.

        Args:
            storage_path: Path to SQLite database (None = in-memory)
            enable_verification: Enable DS-STAR verification
            max_dialogue_depth: Maximum dialogue tree depth
        """
        self.storage_path = storage_path or Path("scratchpad.db")
        self.enable_verification = enable_verification
        self.max_dialogue_depth = max_dialogue_depth

        # Current session
        self.current_tree: Optional[DialogueTree] = None
        self.session_id: Optional[str] = None

        # Persistence (lazy loaded)
        self._persistence = None

        # Internal dialogue engine (lazy loaded)
        self._dialogue_engine = None

        # Strange loop detector (lazy loaded)
        self._loop_detector = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize scratchpad components."""
        # Lazy load persistence
        from HoloLoom.recursive.scratchpad.persistence import ThoughtPersistence
        self._persistence = ThoughtPersistence(self.storage_path)
        await self._persistence.initialize()

        # Lazy load dialogue engine
        from HoloLoom.recursive.scratchpad.internal_dialogue import InternalDialogue
        self._dialogue_engine = InternalDialogue(
            max_depth=self.max_dialogue_depth,
            enable_verification=self.enable_verification
        )

        # Lazy load loop detector
        from HoloLoom.recursive.scratchpad.strange_loops import LoopDetector
        self._loop_detector = LoopDetector()

        # Start new session
        self.session_id = str(uuid.uuid4())

    async def close(self) -> None:
        """Close scratchpad and persist state."""
        if self.current_tree and self.session_id:
            await self.save_session(self.session_id)

        if self._persistence:
            await self._persistence.close()

    async def think(
        self,
        text: str,
        thought_type: ThoughtType = ThoughtType.INITIAL,
        parent_id: Optional[str] = None
    ) -> Thought:
        """
        Create a single thought.

        Args:
            text: Thought content
            thought_type: Type of thought
            parent_id: Parent thought ID (None = root)

        Returns:
            Created thought
        """
        # Determine level
        if parent_id and self.current_tree:
            parent = self.current_tree.thoughts.get(parent_id)
            level = parent.level + 1 if parent else 0
        else:
            level = 0

        # Create thought
        thought = Thought(
            id=str(uuid.uuid4()),
            text=text,
            type=thought_type,
            level=level,
            timestamp=datetime.now(),
            parent_id=parent_id
        )

        # Add verification if enabled
        if self.enable_verification:
            thought.verification = await self._verify_thought(thought)
            thought.confidence = thought.verification.get('overall_score', 0.5)

        # Add to tree
        if not self.current_tree:
            self.current_tree = DialogueTree(root=thought)
        else:
            self.current_tree.add_thought(thought)

        return thought

    async def dialogue_loop(
        self,
        initial_thought: Thought,
        max_depth: Optional[int] = None,
        mode: str = "exploratory"
    ) -> DialogueTree:
        """
        Start internal dialogue loop from initial thought.

        Args:
            initial_thought: Starting thought
            max_depth: Maximum dialogue depth (None = use default)
            mode: Dialogue mode (exploratory, verification, synthesis)

        Returns:
            Complete dialogue tree
        """
        if not self._dialogue_engine:
            raise RuntimeError("Scratchpad not initialized")

        # Start dialogue
        dialogue_tree = await self._dialogue_engine.start_dialogue(
            initial_thought=initial_thought,
            max_depth=max_depth or self.max_dialogue_depth,
            mode=mode
        )

        # Detect strange loops
        if self._loop_detector:
            loops = self._loop_detector.detect_loops(dialogue_tree)
            if loops:
                # Add metadata about detected loops
                for thought_id, loop_info in loops.items():
                    if thought_id in dialogue_tree.thoughts:
                        dialogue_tree.thoughts[thought_id].metadata['strange_loop'] = loop_info

        # Update current tree
        self.current_tree = dialogue_tree

        return dialogue_tree

    async def save_session(self, session_name: str) -> None:
        """
        Save current session to persistent storage.

        Args:
            session_name: Session identifier
        """
        if not self.current_tree or not self._persistence:
            return

        await self._persistence.save_session(
            session_name=session_name,
            tree=self.current_tree
        )

    async def load_session(self, session_name: str) -> Optional[DialogueTree]:
        """
        Load session from persistent storage.

        Args:
            session_name: Session identifier

        Returns:
            Loaded dialogue tree or None
        """
        if not self._persistence:
            return None

        tree = await self._persistence.load_session(session_name)
        if tree:
            self.current_tree = tree
        return tree

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.

        Returns:
            List of session metadata
        """
        if not self._persistence:
            return []

        return await self._persistence.list_sessions()

    async def _verify_thought(self, thought: Thought) -> Dict[str, Any]:
        """
        Verify thought using DS-STAR framework.

        Args:
            thought: Thought to verify

        Returns:
            Verification results
        """
        # Placeholder for DS-STAR verification
        # In production, this would call RAG department's verification

        results = {
            'domain': 0.8,      # D: Domain relevance
            'sensibility': 0.9, # S: Logical consistency
            'temporal': 0.7,    # T: Information freshness
            'argument': 0.8,    # A: Evidence support
            'reference': 0.6,   # R: Source credibility
        }

        results['overall_score'] = sum(results.values()) / len(results)
        return results

    def get_current_tree(self) -> Optional[DialogueTree]:
        """Get current dialogue tree."""
        return self.current_tree

    def visualize(self) -> str:
        """Visualize current dialogue tree."""
        if not self.current_tree:
            return "No active dialogue tree"
        return self.current_tree.tree_visualization()
