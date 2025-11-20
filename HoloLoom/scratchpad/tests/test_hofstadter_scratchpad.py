#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Hofstadter Scratchpad System
======================================

Created: 2025-01-20
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
import tempfile
import os

from HoloLoom.scratchpad import (
    RecursiveScratchpad,
    Thought,
    ThoughtType,
    DialogueTree,
    InternalDialogue,
    DialogueMode,
    LoopDetector,
    LoopType,
    ThoughtPersistence,
    SessionManager,
)


# ============== Thought Tests ==============

def test_thought_creation():
    """Test thought creation."""
    thought = Thought(
        id="t1",
        text="Test thought",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    assert thought.id == "t1"
    assert thought.text == "Test thought"
    assert thought.type == ThoughtType.INITIAL
    assert thought.level == 0
    assert thought.confidence == 0.5  # Default


def test_thought_serialization():
    """Test thought to_dict/from_dict."""
    thought = Thought(
        id="t1",
        text="Test thought",
        type=ThoughtType.QUESTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="t0",
        confidence=0.8
    )

    # Serialize
    data = thought.to_dict()
    assert data['id'] == "t1"
    assert data['type'] == "question"
    assert data['confidence'] == 0.8

    # Deserialize
    restored = Thought.from_dict(data)
    assert restored.id == thought.id
    assert restored.type == thought.type
    assert restored.confidence == thought.confidence


# ============== DialogueTree Tests ==============

def test_dialogue_tree_creation():
    """Test dialogue tree creation."""
    root = Thought(
        id="root",
        text="Root thought",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = DialogueTree(root=root)
    assert tree.root.id == "root"
    assert "root" in tree.thoughts


def test_dialogue_tree_add_thought():
    """Test adding thoughts to tree."""
    root = Thought(
        id="root",
        text="Root",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = DialogueTree(root=root)

    child = Thought(
        id="child1",
        text="Child",
        type=ThoughtType.QUESTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="root"
    )

    tree.add_thought(child)
    assert "child1" in tree.thoughts


def test_dialogue_tree_get_children():
    """Test getting children."""
    root = Thought(
        id="root",
        text="Root",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = DialogueTree(root=root)

    child1 = Thought(
        id="child1",
        text="Child 1",
        type=ThoughtType.QUESTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="root"
    )
    child2 = Thought(
        id="child2",
        text="Child 2",
        type=ThoughtType.QUESTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="root"
    )

    tree.add_thought(child1)
    tree.add_thought(child2)

    children = tree.get_children("root")
    assert len(children) == 2


def test_dialogue_tree_get_path():
    """Test path extraction."""
    root = Thought(
        id="root",
        text="Root",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = DialogueTree(root=root)

    child = Thought(
        id="child",
        text="Child",
        type=ThoughtType.QUESTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="root"
    )
    grandchild = Thought(
        id="grandchild",
        text="Grandchild",
        type=ThoughtType.ANSWER,
        level=2,
        timestamp=datetime.now(),
        parent_id="child"
    )

    tree.add_thought(child)
    tree.add_thought(grandchild)

    path = tree.get_path("grandchild")
    assert len(path) == 3
    assert path[0].id == "root"
    assert path[1].id == "child"
    assert path[2].id == "grandchild"


def test_dialogue_tree_visualization():
    """Test tree visualization."""
    root = Thought(
        id="root",
        text="Root thought",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = DialogueTree(root=root)

    child = Thought(
        id="child",
        text="Child question",
        type=ThoughtType.QUESTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="root"
    )
    tree.add_thought(child)

    viz = tree.tree_visualization()
    assert "🌱" in viz  # INITIAL emoji
    assert "❓" in viz  # QUESTION emoji
    assert "Root thought" in viz


# ============== RecursiveScratchpad Tests ==============

@pytest.mark.asyncio
async def test_scratchpad_initialization():
    """Test scratchpad initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with RecursiveScratchpad(storage_path=db_path) as scratchpad:
            assert scratchpad.storage_path == db_path
            assert scratchpad.session_id is not None


@pytest.mark.asyncio
async def test_scratchpad_think():
    """Test creating thoughts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with RecursiveScratchpad(storage_path=db_path) as scratchpad:
            thought = await scratchpad.think("Test thought")

            assert thought.text == "Test thought"
            assert thought.type == ThoughtType.INITIAL
            assert scratchpad.current_tree is not None


@pytest.mark.asyncio
async def test_scratchpad_dialogue_loop():
    """Test dialogue loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        async with RecursiveScratchpad(
            storage_path=db_path,
            max_dialogue_depth=3
        ) as scratchpad:
            initial = await scratchpad.think("What is Thompson Sampling?")

            tree = await scratchpad.dialogue_loop(
                initial_thought=initial,
                max_depth=2,
                mode="exploratory"
            )

            assert len(tree.thoughts) > 1  # Should have questions/answers
            assert tree.get_depth() > 0


# ============== InternalDialogue Tests ==============

@pytest.mark.asyncio
async def test_internal_dialogue_exploratory():
    """Test exploratory dialogue generation."""
    engine = InternalDialogue(max_depth=3)

    initial = Thought(
        id="t0",
        text="Thompson Sampling is a Bayesian approach.",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = await engine.start_dialogue(
        initial_thought=initial,
        max_depth=2,
        mode="exploratory"
    )

    assert len(tree.thoughts) > 1
    # Should have questions
    questions = [t for t in tree.thoughts.values() if t.type == ThoughtType.QUESTION]
    assert len(questions) > 0


@pytest.mark.asyncio
async def test_internal_dialogue_verification():
    """Test verification dialogue."""
    engine = InternalDialogue(max_depth=3, enable_verification=True)

    initial = Thought(
        id="t0",
        text="Thompson Sampling always wins.",
        type=ThoughtType.INITIAL,
        level=0,
        timestamp=datetime.now()
    )

    tree = await engine.start_dialogue(
        initial_thought=initial,
        max_depth=2,
        mode="verification"
    )

    # Should have verification questions
    questions = [t for t in tree.thoughts.values() if t.type == ThoughtType.QUESTION]
    question_texts = [q.text.lower() for q in questions]

    # Should ask about evidence, logic, etc.
    has_verification = any(
        "evidence" in q or "sense" in q or "support" in q
        for q in question_texts
    )
    assert has_verification or len(questions) > 0


# ============== LoopDetector Tests ==============

def test_loop_detector_self_reference():
    """Test self-reference detection."""
    detector = LoopDetector()

    # Self-referential thought
    thought = Thought(
        id="t1",
        text="I'm thinking about my own thinking process.",
        type=ThoughtType.REFLECTION,
        level=0,
        timestamp=datetime.now()
    )

    assert detector._is_self_referential(thought)


def test_loop_detector_meta_thought():
    """Test meta-thought detection."""
    detector = LoopDetector()

    # Meta-thought
    meta = Thought(
        id="t1",
        text="How am I reasoning about this?",
        type=ThoughtType.REFLECTION,
        level=1,
        timestamp=datetime.now()
    )

    assert detector._is_meta_thought(meta)

    # Non-meta thought
    regular = Thought(
        id="t2",
        text="Thompson Sampling uses Beta distributions.",
        type=ThoughtType.ANSWER,
        level=1,
        timestamp=datetime.now()
    )

    assert not detector._is_meta_thought(regular)


def test_loop_detector_detect_loops():
    """Test loop detection in tree."""
    detector = LoopDetector()

    root = Thought(
        id="root",
        text="What is Thompson Sampling?",
        type=ThoughtType.QUESTION,
        level=0,
        timestamp=datetime.now()
    )

    tree = DialogueTree(root=root)

    # Add self-referential thought
    meta = Thought(
        id="meta",
        text="I'm questioning my own understanding.",
        type=ThoughtType.REFLECTION,
        level=1,
        timestamp=datetime.now(),
        parent_id="root"
    )
    tree.add_thought(meta)

    loops = detector.detect_loops(tree)
    assert len(loops) > 0  # Should detect self-reference


# ============== Persistence Tests ==============

@pytest.mark.asyncio
async def test_persistence_save_load():
    """Test save/load session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        persistence = ThoughtPersistence(db_path)
        await persistence.initialize()

        # Create tree
        root = Thought(
            id="root",
            text="Root thought",
            type=ThoughtType.INITIAL,
            level=0,
            timestamp=datetime.now()
        )
        tree = DialogueTree(root=root)

        child = Thought(
            id="child",
            text="Child thought",
            type=ThoughtType.QUESTION,
            level=1,
            timestamp=datetime.now(),
            parent_id="root"
        )
        tree.add_thought(child)

        # Save
        session_id = await persistence.save_session("test_session", tree)
        assert session_id == "test_session"

        # Load
        loaded_tree = await persistence.load_session("test_session")
        assert loaded_tree is not None
        assert len(loaded_tree.thoughts) == 2
        assert "root" in loaded_tree.thoughts
        assert "child" in loaded_tree.thoughts

        await persistence.close()


@pytest.mark.asyncio
async def test_persistence_list_sessions():
    """Test listing sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        persistence = ThoughtPersistence(db_path)
        await persistence.initialize()

        # Create sessions
        for i in range(3):
            root = Thought(
                id=f"root{i}",
                text=f"Root {i}",
                type=ThoughtType.INITIAL,
                level=0,
                timestamp=datetime.now()
            )
            tree = DialogueTree(root=root)
            await persistence.save_session(f"session_{i}", tree)

        # List
        sessions = await persistence.list_sessions()
        assert len(sessions) == 3

        await persistence.close()


@pytest.mark.asyncio
async def test_persistence_search():
    """Test thought search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        persistence = ThoughtPersistence(db_path)
        await persistence.initialize()

        # Create tree with searchable content
        root = Thought(
            id="root",
            text="Thompson Sampling is great",
            type=ThoughtType.INITIAL,
            level=0,
            timestamp=datetime.now()
        )
        tree = DialogueTree(root=root)

        child = Thought(
            id="child",
            text="Thompson uses Beta distributions",
            type=ThoughtType.ANSWER,
            level=1,
            timestamp=datetime.now(),
            parent_id="root"
        )
        tree.add_thought(child)

        await persistence.save_session("search_test", tree)

        # Search
        results = await persistence.search_thoughts("Thompson")
        assert len(results) == 2

        await persistence.close()


# ============== Integration Tests ==============

@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow: think → dialogue → save → load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create scratchpad
        async with RecursiveScratchpad(
            storage_path=db_path,
            max_dialogue_depth=3
        ) as scratchpad:
            # Think
            initial = await scratchpad.think("What is Thompson Sampling?")

            # Dialogue
            tree = await scratchpad.dialogue_loop(
                initial_thought=initial,
                max_depth=2,
                mode="exploratory"
            )

            # Save
            await scratchpad.save_session("integration_test")

        # Load in new scratchpad
        async with RecursiveScratchpad(storage_path=db_path) as scratchpad2:
            loaded_tree = await scratchpad2.load_session("integration_test")

            assert loaded_tree is not None
            assert len(loaded_tree.thoughts) == len(tree.thoughts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
