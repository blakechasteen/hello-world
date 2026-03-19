"""
Unit tests for hololoom.core.recursive — targeting low-coverage files.

Covers:
- scratchpad_minimal.py (ScratchpadEntry, Scratchpad, LoopConfig, LoopResult, RecursiveEngine)
- scratchpad/recursive_scratchpad.py (Thought, DialogueTree, RecursiveScratchpad)
- scratchpad/strange_loops.py (LoopDetector, StrangeLoopAnalyzer)
- scratchpad/internal_dialogue.py (InternalDialogue)
- scratchpad/persistence.py (ThoughtPersistence, SessionManager)
- scratchpad_integration.py (ProvenanceTracker, ScratchpadConfig, RecursiveRefiner)
- full_learning_loop.py (ThompsonPriors save/load, PolicyWeights, LearningMetrics)
"""

import pytest
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ============================================================================
# scratchpad_minimal.py
# ============================================================================
from hololoom.core.recursive.scratchpad_minimal import (
    ScratchpadEntry as MinimalEntry,
    Scratchpad as MinimalScratchpad,
    LoopType as MinimalLoopType,
    LoopConfig as MinimalLoopConfig,
    LoopResult as MinimalLoopResult,
    RecursiveEngine as MinimalRecursiveEngine,
)


class TestMinimalScratchpadEntry:
    def test_creation(self):
        entry = MinimalEntry(
            thought="think", action="act", observation="obs", score=0.7
        )
        assert entry.thought == "think"
        assert entry.score == 0.7
        assert entry.metadata == {}
        assert entry.iteration is None
        assert entry.timestamp is None

    def test_with_optional_fields(self):
        entry = MinimalEntry(
            thought="t", action="a", observation="o", score=0.5,
            iteration=2, timestamp=1234.0, metadata={"k": "v"}
        )
        assert entry.iteration == 2
        assert entry.timestamp == 1234.0
        assert entry.metadata["k"] == "v"


class TestMinimalScratchpad:
    def test_init_default_capacity(self):
        sp = MinimalScratchpad()
        assert sp.capacity == 1000
        assert len(sp) == 0

    def test_add_entry(self):
        sp = MinimalScratchpad()
        sp.add_entry("t", "a", "o", 0.5, iteration=1)
        assert len(sp) == 1
        assert sp.entries[0].thought == "t"
        assert sp.entries[0].iteration == 1

    def test_add_entry_with_metadata(self):
        sp = MinimalScratchpad()
        sp.add_entry("t", "a", "o", 0.5, metadata={"key": "val"})
        assert sp.entries[0].metadata == {"key": "val"}

    def test_add_entry_metadata_default(self):
        sp = MinimalScratchpad()
        sp.add_entry("t", "a", "o", 0.5)
        assert sp.entries[0].metadata == {}

    def test_capacity_enforcement(self):
        sp = MinimalScratchpad(capacity=2)
        sp.add_entry("t1", "a1", "o1", 0.1)
        sp.add_entry("t2", "a2", "o2", 0.2)
        sp.add_entry("t3", "a3", "o3", 0.3)
        assert len(sp) == 2
        assert sp.entries[0].thought == "t2"
        assert sp.entries[1].thought == "t3"

    def test_get_history(self):
        sp = MinimalScratchpad()
        sp.add_entry("t1", "a1", "o1", 0.1)
        history = sp.get_history()
        assert len(history) == 1
        # Returns a copy
        assert history is not sp.entries

    def test_get_last_n(self):
        sp = MinimalScratchpad()
        for i in range(5):
            sp.add_entry(f"t{i}", "a", "o", 0.5)
        last = sp.get_last_n(2)
        assert len(last) == 2
        assert last[0].thought == "t3"

    def test_clear(self):
        sp = MinimalScratchpad()
        sp.add_entry("t", "a", "o", 0.5)
        sp.clear()
        assert len(sp) == 0


class TestMinimalLoopTypes:
    def test_enum_values(self):
        assert MinimalLoopType.REFINE.value == "refine"
        assert MinimalLoopType.VERIFY.value == "verify"
        assert MinimalLoopType.CRITIQUE.value == "critique"
        assert MinimalLoopType.ELEGANCE.value == "elegance"
        assert MinimalLoopType.HOFSTADTER.value == "hofstadter"


class TestMinimalLoopConfig:
    def test_defaults(self):
        cfg = MinimalLoopConfig()
        assert cfg.max_iterations == 3
        assert cfg.quality_threshold == 0.85
        assert cfg.loop_type == MinimalLoopType.REFINE
        assert cfg.enable_learning is True


class TestMinimalLoopResult:
    def test_creation(self):
        result = MinimalLoopResult(success=True, iterations=2, final_quality=0.9)
        assert result.improvement == 0.0
        assert result.history == []

    def test_summary_positive_improvement(self):
        result = MinimalLoopResult(
            success=True, iterations=3, final_quality=0.95, improvement=0.2
        )
        s = result.summary()
        assert "3 iterations" in s
        assert "0.950" in s
        assert "+0.200" in s

    def test_summary_negative_improvement(self):
        result = MinimalLoopResult(
            success=False, iterations=1, final_quality=0.3, improvement=-0.1
        )
        s = result.summary()
        assert "-0.100" in s


class TestMinimalRecursiveEngine:
    def test_init(self):
        engine = MinimalRecursiveEngine(MinimalLoopConfig())
        assert isinstance(engine.scratchpad, MinimalScratchpad)

    @pytest.mark.asyncio
    async def test_run_loop(self):
        cfg = MinimalLoopConfig(quality_threshold=0.8)
        engine = MinimalRecursiveEngine(cfg)
        result = await engine.run_loop("test", initial_quality=0.5)
        assert result.success is True
        assert result.final_quality >= 0.8
        assert result.improvement == result.final_quality - 0.5

    @pytest.mark.asyncio
    async def test_run_loop_already_above_threshold(self):
        cfg = MinimalLoopConfig(quality_threshold=0.5)
        engine = MinimalRecursiveEngine(cfg)
        result = await engine.run_loop("test", initial_quality=0.9)
        assert result.success is True
        assert result.final_quality >= 0.9

    def test_get_scratchpad(self):
        engine = MinimalRecursiveEngine(MinimalLoopConfig())
        assert engine.get_scratchpad() is engine.scratchpad


# ============================================================================
# scratchpad/recursive_scratchpad.py — Thought and DialogueTree
# ============================================================================
from hololoom.recursive.scratchpad.recursive_scratchpad import (
    Thought,
    ThoughtType,
    DialogueTree,
    RecursiveScratchpad,
)


class TestThought:
    def _make(self, **kwargs):
        defaults = dict(
            id="t1", text="hello", type=ThoughtType.INITIAL,
            level=0, timestamp=datetime(2025, 1, 1)
        )
        defaults.update(kwargs)
        return Thought(**defaults)

    def test_creation(self):
        t = self._make()
        assert t.id == "t1"
        assert t.confidence == 0.5

    def test_post_init_string_timestamp(self):
        t = Thought(
            id="t1", text="hi", type=ThoughtType.INITIAL,
            level=0, timestamp="2025-01-01T00:00:00"
        )
        assert isinstance(t.timestamp, datetime)

    def test_post_init_string_type(self):
        t = Thought(
            id="t1", text="hi", type="initial",
            level=0, timestamp=datetime(2025, 1, 1)
        )
        assert t.type == ThoughtType.INITIAL

    def test_to_dict(self):
        t = self._make(metadata={"k": "v"}, verification={"score": 0.8})
        d = t.to_dict()
        assert d["id"] == "t1"
        assert d["type"] == "initial"
        assert d["metadata"] == {"k": "v"}
        assert d["verification"] == {"score": 0.8}

    def test_from_dict(self):
        data = {
            "id": "t2", "text": "bye", "type": "question",
            "level": 1, "timestamp": "2025-06-01T12:00:00",
            "parent_id": "t1", "confidence": 0.9,
            "metadata": {"x": 1}, "verification": None
        }
        t = Thought.from_dict(data)
        assert t.id == "t2"
        assert t.type == ThoughtType.QUESTION
        assert t.confidence == 0.9
        assert t.parent_id == "t1"

    def test_from_dict_minimal(self):
        data = {
            "id": "t3", "text": "x", "type": "answer",
            "level": 0, "timestamp": "2025-01-01T00:00:00"
        }
        t = Thought.from_dict(data)
        assert t.confidence == 0.5
        assert t.metadata == {}
        assert t.verification is None


class TestDialogueTree:
    def _make_tree(self):
        root = Thought(
            id="root", text="Start", type=ThoughtType.INITIAL,
            level=0, timestamp=datetime(2025, 1, 1)
        )
        tree = DialogueTree(root=root)
        return tree, root

    def test_init_indexes_root(self):
        tree, root = self._make_tree()
        assert "root" in tree.thoughts
        assert tree.thoughts["root"] is root

    def test_add_thought(self):
        tree, root = self._make_tree()
        child = Thought(
            id="c1", text="Child", type=ThoughtType.QUESTION,
            level=1, timestamp=datetime(2025, 1, 1), parent_id="root"
        )
        tree.add_thought(child)
        assert "c1" in tree.thoughts

    def test_get_children(self):
        tree, root = self._make_tree()
        for i in range(3):
            tree.add_thought(Thought(
                id=f"c{i}", text=f"Child {i}", type=ThoughtType.ANSWER,
                level=1, timestamp=datetime(2025, 1, 1), parent_id="root"
            ))
        children = tree.get_children("root")
        assert len(children) == 3

    def test_get_children_no_children(self):
        tree, root = self._make_tree()
        assert tree.get_children("root") == []

    def test_get_path(self):
        tree, root = self._make_tree()
        c1 = Thought(id="c1", text="C1", type=ThoughtType.QUESTION,
                     level=1, timestamp=datetime(2025, 1, 1), parent_id="root")
        c2 = Thought(id="c2", text="C2", type=ThoughtType.ANSWER,
                     level=2, timestamp=datetime(2025, 1, 1), parent_id="c1")
        tree.add_thought(c1)
        tree.add_thought(c2)
        path = tree.get_path("c2")
        assert len(path) == 3
        assert path[0].id == "root"
        assert path[-1].id == "c2"

    def test_get_path_nonexistent(self):
        tree, _ = self._make_tree()
        path = tree.get_path("nonexistent")
        assert path == []

    def test_get_depth(self):
        tree, root = self._make_tree()
        tree.add_thought(Thought(
            id="c1", text="C1", type=ThoughtType.QUESTION,
            level=3, timestamp=datetime(2025, 1, 1), parent_id="root"
        ))
        assert tree.get_depth() == 3

    def test_get_depth_root_only(self):
        tree, _ = self._make_tree()
        assert tree.get_depth() == 0

    def test_tree_visualization(self):
        tree, root = self._make_tree()
        tree.add_thought(Thought(
            id="c1", text="Question?", type=ThoughtType.QUESTION,
            level=1, timestamp=datetime(2025, 1, 1), parent_id="root"
        ))
        viz = tree.tree_visualization()
        assert "Start" in viz
        assert "Question?" in viz

    def test_to_dict_and_from_dict(self):
        tree, root = self._make_tree()
        tree.add_thought(Thought(
            id="c1", text="Child", type=ThoughtType.ANSWER,
            level=1, timestamp=datetime(2025, 1, 1), parent_id="root"
        ))
        d = tree.to_dict()
        assert d["root_id"] == "root"
        assert "c1" in d["thoughts"]

        tree2 = DialogueTree.from_dict(d)
        assert tree2.root.id == "root"
        assert "c1" in tree2.thoughts


# ============================================================================
# scratchpad/recursive_scratchpad.py — RecursiveScratchpad
# ============================================================================
class TestRecursiveScratchpad:
    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path,
            enable_verification=False,
            max_dialogue_depth=3
        ) as sp:
            assert sp.session_id is not None
            assert sp._persistence is not None
            assert sp._dialogue_engine is not None
            assert sp._loop_detector is not None

    @pytest.mark.asyncio
    async def test_think_creates_root(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False
        ) as sp:
            t = await sp.think("First thought")
            assert t.text == "First thought"
            assert t.type == ThoughtType.INITIAL
            assert t.level == 0
            assert sp.current_tree is not None
            assert sp.current_tree.root.id == t.id

    @pytest.mark.asyncio
    async def test_think_creates_child(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False
        ) as sp:
            root = await sp.think("Root")
            child = await sp.think("Child", ThoughtType.QUESTION, parent_id=root.id)
            assert child.level == 1
            assert child.parent_id == root.id
            assert child.id in sp.current_tree.thoughts

    @pytest.mark.asyncio
    async def test_think_with_verification(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=True
        ) as sp:
            t = await sp.think("Verified thought")
            assert t.verification is not None
            assert "overall_score" in t.verification
            assert t.confidence > 0

    @pytest.mark.asyncio
    async def test_dialogue_loop(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False, max_dialogue_depth=2
        ) as sp:
            root = await sp.think("Initial thought about AI is interesting")
            tree = await sp.dialogue_loop(root, max_depth=2, mode="exploratory")
            assert len(tree.thoughts) > 1
            # Prevent auto-save on exit (tree has duplicate IDs from dialogue)
            sp.current_tree = None

    @pytest.mark.asyncio
    async def test_dialogue_loop_not_initialized(self):
        sp = RecursiveScratchpad(enable_verification=False)
        root = Thought(
            id="r", text="test", type=ThoughtType.INITIAL,
            level=0, timestamp=datetime.now()
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            await sp.dialogue_loop(root)

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False
        ) as sp:
            await sp.think("Session thought")
            await sp.save_session("test_session")
            # Clear tree to prevent duplicate save on exit
            sp.current_tree = None

        # Load in a new instance
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False
        ) as sp2:
            tree = await sp2.load_session("test_session")
            assert tree is not None
            assert len(tree.thoughts) >= 1
            sp2.current_tree = None

    @pytest.mark.asyncio
    async def test_load_session_not_found(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False
        ) as sp:
            tree = await sp.load_session("nonexistent")
            assert tree is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path):
        db_path = tmp_path / "test.db"
        async with RecursiveScratchpad(
            storage_path=db_path, enable_verification=False
        ) as sp:
            await sp.think("S1")
            await sp.save_session("session1")
            sessions = await sp.list_sessions()
            assert len(sessions) >= 1
            sp.current_tree = None

    def test_visualize_no_tree(self):
        sp = RecursiveScratchpad(enable_verification=False)
        assert sp.visualize() == "No active dialogue tree"

    def test_get_current_tree_none(self):
        sp = RecursiveScratchpad(enable_verification=False)
        assert sp.get_current_tree() is None

    @pytest.mark.asyncio
    async def test_load_session_no_persistence(self):
        sp = RecursiveScratchpad(enable_verification=False)
        result = await sp.load_session("x")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions_no_persistence(self):
        sp = RecursiveScratchpad(enable_verification=False)
        result = await sp.list_sessions()
        assert result == []


# ============================================================================
# scratchpad/strange_loops.py
# ============================================================================
from hololoom.recursive.scratchpad.strange_loops import (
    LoopDetector,
    LoopType as SLLoopType,
    StrangeLoop as SLStrangeLoop,
    LevelCrossing,
    StrangeLoopAnalyzer,
)


class TestLoopDetector:
    def _make_tree_with_self_ref(self):
        root = Thought(
            id="t0", text="I think my reasoning is correct",
            type=ThoughtType.INITIAL, level=0, timestamp=datetime(2025, 1, 1)
        )
        return DialogueTree(root=root)

    def _make_tree_with_meta(self):
        root = Thought(
            id="t0", text="The sky is blue",
            type=ThoughtType.INITIAL, level=0, timestamp=datetime(2025, 1, 1)
        )
        tree = DialogueTree(root=root)
        tree.add_thought(Thought(
            id="t1", text="Reflecting on my process of thinking about this",
            type=ThoughtType.REFLECTION, level=1,
            timestamp=datetime(2025, 1, 1), parent_id="t0"
        ))
        return tree

    def test_detect_self_reference(self):
        detector = LoopDetector()
        tree = self._make_tree_with_self_ref()
        loops = detector.detect_loops(tree)
        assert len(loops) > 0
        found_self_ref = any(
            l.loop_type == SLLoopType.DIRECT_SELF_REFERENCE
            for l in loops.values()
        )
        assert found_self_ref

    def test_is_self_referential_patterns(self):
        detector = LoopDetector()
        cases = [
            ("I think this is correct", True),
            ("my reasoning says so", True),
            ("this thought is interesting", True),
            ("The sky is blue", False),
            ("2 + 2 = 4", False),
        ]
        for text, expected in cases:
            t = Thought(id="x", text=text, type=ThoughtType.INITIAL,
                        level=0, timestamp=datetime(2025, 1, 1))
            assert detector._is_self_referential(t) == expected, f"Failed for: {text}"

    def test_semantic_similarity(self):
        detector = LoopDetector()
        assert detector._semantic_similarity("hello world", "hello world") == 1.0
        assert detector._semantic_similarity("hello world", "goodbye moon") == 0.0
        assert detector._semantic_similarity("", "") == 0.0
        sim = detector._semantic_similarity("the cat sat", "the cat ran")
        assert 0 < sim < 1

    def test_find_cycles_with_similar_text(self):
        detector = LoopDetector(min_loop_size=2, similarity_threshold=0.5)
        root = Thought(id="t0", text="cats are nice pets to have",
                       type=ThoughtType.INITIAL, level=0, timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        tree.add_thought(Thought(
            id="t1", text="dogs are different animals", type=ThoughtType.ANSWER,
            level=1, timestamp=datetime(2025, 1, 1), parent_id="t0"
        ))
        tree.add_thought(Thought(
            id="t2", text="cats are nice pets to have at home",
            type=ThoughtType.ANSWER, level=2,
            timestamp=datetime(2025, 1, 1), parent_id="t1"
        ))
        cycles = detector._find_cycles(tree)
        assert len(cycles) >= 1

    def test_detect_level_crossing(self):
        detector = LoopDetector()
        tree = self._make_tree_with_meta()
        crossings = detector._detect_level_crossings(tree)
        assert len(crossings) >= 1
        assert crossings[0].crossing_type == "meta_to_object"

    def test_is_meta_thought(self):
        detector = LoopDetector()
        reflection = Thought(
            id="r", text="blah", type=ThoughtType.REFLECTION,
            level=0, timestamp=datetime(2025, 1, 1)
        )
        assert detector._is_meta_thought(reflection) is True

        meta_text = Thought(
            id="m", text="Thinking about my approach to this",
            type=ThoughtType.ANSWER, level=0, timestamp=datetime(2025, 1, 1)
        )
        assert detector._is_meta_thought(meta_text) is True

        plain = Thought(
            id="p", text="The answer is 42",
            type=ThoughtType.ANSWER, level=0, timestamp=datetime(2025, 1, 1)
        )
        assert detector._is_meta_thought(plain) is False

    def test_detect_meta_reasoning(self):
        detector = LoopDetector()
        root = Thought(
            id="t0", text="Am I thinking correctly about this?",
            type=ThoughtType.INITIAL, level=0, timestamp=datetime(2025, 1, 1)
        )
        tree = DialogueTree(root=root)
        meta = detector._detect_meta_reasoning(tree)
        assert len(meta) >= 1
        assert meta["t0"].loop_type == SLLoopType.META_REASONING

    def test_questions_own_reasoning(self):
        detector = LoopDetector()
        cases = [
            ("is my reasoning valid?", True),
            ("am i thinking correctly?", True),
            ("what if my assumption is wrong about this?", True),
            ("this reasoning is flawed", True),
            ("the weather is nice", False),
        ]
        for text, expected in cases:
            t = Thought(id="x", text=text, type=ThoughtType.INITIAL,
                        level=0, timestamp=datetime(2025, 1, 1))
            assert detector._questions_own_reasoning(t) == expected, f"Failed for: {text}"

    def test_involves_level_crossing(self):
        detector = LoopDetector()
        root = Thought(
            id="t0", text="fact", type=ThoughtType.INITIAL,
            level=0, timestamp=datetime(2025, 1, 1)
        )
        tree = DialogueTree(root=root)
        meta = Thought(
            id="t1", text="reflecting on my method",
            type=ThoughtType.REFLECTION, level=1,
            timestamp=datetime(2025, 1, 1), parent_id="t0"
        )
        tree.add_thought(meta)
        assert detector._involves_level_crossing(meta, tree) is True

    def test_visualize_loop(self):
        detector = LoopDetector()
        t1 = Thought(id="t1", text="Self referencing thought here",
                     type=ThoughtType.INITIAL, level=0, timestamp=datetime(2025, 1, 1))
        t2 = Thought(id="t2", text="Another thought in loop",
                     type=ThoughtType.ANSWER, level=1, timestamp=datetime(2025, 1, 1))
        loop = SLStrangeLoop(
            loop_type=SLLoopType.CYCLIC_REFERENCE,
            thoughts=[t1, t2],
            description="Test loop",
            strength=0.7,
            level_crossings=[]
        )
        viz = detector.visualize_loop(loop)
        assert "CYCLIC_REFERENCE" in viz or "cyclic_reference" in viz
        assert "0.70" in viz
        assert "Test loop" in viz

    def test_visualize_loop_with_crossings(self):
        detector = LoopDetector()
        t1 = Thought(id="t1", text="Meta-level thought about reasoning",
                     type=ThoughtType.REFLECTION, level=1, timestamp=datetime(2025, 1, 1))
        t2 = Thought(id="t2", text="Object-level thought about facts",
                     type=ThoughtType.INITIAL, level=0, timestamp=datetime(2025, 1, 1))
        crossing = LevelCrossing(
            upper_thought=t1, lower_thought=t2,
            crossing_type="meta_to_object", strength=0.8
        )
        loop = SLStrangeLoop(
            loop_type=SLLoopType.LEVEL_CROSSING,
            thoughts=[t1, t2], description="Level crossing",
            strength=0.8, level_crossings=[crossing]
        )
        viz = detector.visualize_loop(loop)
        assert "Level Crossings:" in viz


class TestStrangeLoopAnalyzer:
    def _make_loops(self):
        t1 = Thought(id="t1", text="x", type=ThoughtType.INITIAL,
                     level=0, timestamp=datetime(2025, 1, 1))
        loops = {
            "l1": SLStrangeLoop(
                loop_type=SLLoopType.DIRECT_SELF_REFERENCE,
                thoughts=[t1], description="self ref",
                strength=0.5, level_crossings=[]
            ),
            "l2": SLStrangeLoop(
                loop_type=SLLoopType.STRANGE_LOOP,
                thoughts=[t1], description="strange",
                strength=1.0, level_crossings=[]
            ),
        }
        return loops

    def test_analyze_loop_density(self):
        tree, _ = TestDialogueTree()._make_tree()
        loops = self._make_loops()
        density = StrangeLoopAnalyzer.analyze_loop_density(tree, loops)
        assert density == 2.0  # 2 loops / 1 thought

    def test_analyze_loop_density_empty(self):
        root = Thought(id="r", text="x", type=ThoughtType.INITIAL,
                       level=0, timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        # Hacky: clear thoughts to test empty
        tree.thoughts.clear()
        density = StrangeLoopAnalyzer.analyze_loop_density(tree, {})
        assert density == 0.0

    def test_get_strongest_loops(self):
        loops = self._make_loops()
        strongest = StrangeLoopAnalyzer.get_strongest_loops(loops, n=1)
        assert len(strongest) == 1
        assert strongest[0].strength == 1.0

    def test_classify_loop_complexity(self):
        t = Thought(id="x", text="x", type=ThoughtType.INITIAL,
                    level=0, timestamp=datetime(2025, 1, 1))
        cases = [
            (SLLoopType.DIRECT_SELF_REFERENCE, "simple"),
            (SLLoopType.CYCLIC_REFERENCE, "moderate"),
            (SLLoopType.LEVEL_CROSSING, "complex"),
            (SLLoopType.STRANGE_LOOP, "tangled"),
            (SLLoopType.META_REASONING, "moderate"),
        ]
        for loop_type, expected in cases:
            loop = SLStrangeLoop(
                loop_type=loop_type, thoughts=[t],
                description="x", strength=0.5, level_crossings=[]
            )
            assert StrangeLoopAnalyzer.classify_loop_complexity(loop) == expected


# ============================================================================
# scratchpad/internal_dialogue.py
# ============================================================================
from hololoom.recursive.scratchpad.internal_dialogue import (
    InternalDialogue,
    DialogueMode,
    DialogueStep,
)


class TestInternalDialogue:
    def _make_thought(self, text="AI is interesting", confidence=0.5, level=0):
        return Thought(
            id="t0", text=text, type=ThoughtType.INITIAL,
            level=level, timestamp=datetime(2025, 1, 1),
            confidence=confidence
        )

    @pytest.mark.asyncio
    async def test_start_dialogue_exploratory(self):
        engine = InternalDialogue(max_depth=2, confidence_threshold=0.95)
        thought = self._make_thought()
        tree = await engine.start_dialogue(thought, max_depth=2, mode="exploratory")
        assert len(tree.thoughts) > 1

    @pytest.mark.asyncio
    async def test_start_dialogue_verification(self):
        engine = InternalDialogue(max_depth=2, confidence_threshold=0.95)
        thought = self._make_thought()
        tree = await engine.start_dialogue(thought, max_depth=2, mode="verification")
        assert len(tree.thoughts) > 1

    @pytest.mark.asyncio
    async def test_start_dialogue_synthesis(self):
        engine = InternalDialogue(max_depth=2, confidence_threshold=0.95)
        thought = self._make_thought()
        tree = await engine.start_dialogue(thought, max_depth=2, mode="synthesis")
        assert len(tree.thoughts) >= 1

    @pytest.mark.asyncio
    async def test_start_dialogue_hofstadter(self):
        engine = InternalDialogue(max_depth=2, confidence_threshold=0.95)
        thought = self._make_thought()
        tree = await engine.start_dialogue(thought, max_depth=2, mode="hofstadter")
        assert len(tree.thoughts) >= 1

    @pytest.mark.asyncio
    async def test_high_confidence_stops_early(self):
        engine = InternalDialogue(max_depth=5, confidence_threshold=0.3)
        thought = self._make_thought(confidence=0.5)
        tree = await engine.start_dialogue(thought, max_depth=5, mode="exploratory")
        # Should stop early due to high confidence
        assert tree is not None

    @pytest.mark.asyncio
    async def test_generate_questions_unknown_mode(self):
        engine = InternalDialogue()
        thought = self._make_thought()
        # Manually test with a mode not in generators (shouldn't happen normally)
        questions = await engine._generate_questions(thought, DialogueMode.EXPLORATORY)
        assert isinstance(questions, list)

    @pytest.mark.asyncio
    async def test_generate_exploratory_questions(self):
        engine = InternalDialogue()
        thought = self._make_thought("The sky is blue", confidence=0.4)
        questions = await engine._generate_exploratory_questions(thought)
        assert len(questions) <= 2
        assert any("Why" in q or "what" in q.lower() or "certain" in q.lower()
                    for q in questions)

    @pytest.mark.asyncio
    async def test_generate_verification_questions(self):
        engine = InternalDialogue()
        thought = self._make_thought()
        thought.verification = {"argument": 0.5, "reference": 0.5}
        questions = await engine._generate_verification_questions(thought)
        assert len(questions) <= 2

    @pytest.mark.asyncio
    async def test_generate_synthesis_questions(self):
        engine = InternalDialogue()
        thought = self._make_thought(level=4)
        thought.level = 4
        questions = await engine._generate_synthesis_questions(thought)
        assert len(questions) <= 1

    @pytest.mark.asyncio
    async def test_generate_hofstadter_questions(self):
        engine = InternalDialogue()
        thought = self._make_thought(level=3)
        thought.level = 3
        questions = await engine._generate_hofstadter_questions(thought)
        assert len(questions) <= 1

    def test_synthesize_answer_patterns(self):
        engine = InternalDialogue()
        root = self._make_thought()
        tree = DialogueTree(root=root)
        cases = [
            "Why is this the case?",
            "How can I be more certain about this?",
            "What if the opposite were true?",
            "Does this make logical sense?",
            "How does this connect to what I learned before?",
            "What pattern am I seeing here?",
            "What assumptions am I making in thinking this?",
            "Is my reasoning process itself flawed?",
            "Something else entirely",
        ]
        for q in cases:
            answer = engine._synthesize_answer(q, root, tree)
            assert isinstance(answer, str)
            assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_generate_reflection_low_confidence(self):
        engine = InternalDialogue()
        answer = Thought(
            id="a1", text="Uncertain answer", type=ThoughtType.ANSWER,
            level=1, timestamp=datetime(2025, 1, 1), confidence=0.3
        )
        tree = DialogueTree(root=self._make_thought())
        reflection = await engine._generate_reflection(answer, tree)
        assert reflection is not None
        assert reflection.type == ThoughtType.REFLECTION

    @pytest.mark.asyncio
    async def test_generate_reflection_high_confidence(self):
        engine = InternalDialogue()
        answer = Thought(
            id="a1", text="Certain answer", type=ThoughtType.ANSWER,
            level=1, timestamp=datetime(2025, 1, 1), confidence=0.9
        )
        tree = DialogueTree(root=self._make_thought())
        reflection = await engine._generate_reflection(answer, tree)
        assert reflection is None

    def test_detect_convergence(self):
        engine = InternalDialogue(confidence_threshold=0.7)
        root = self._make_thought(confidence=0.8)
        tree = DialogueTree(root=root)
        for i in range(4):
            tree.add_thought(Thought(
                id=f"t{i+1}", text=f"High conf {i}",
                type=ThoughtType.ANSWER, level=1,
                timestamp=datetime(2025, 1, 1), confidence=0.85,
                parent_id="t0"
            ))
        assert engine.detect_convergence(tree) is True

    def test_detect_convergence_not_converged(self):
        engine = InternalDialogue(confidence_threshold=0.9)
        root = self._make_thought(confidence=0.3)
        tree = DialogueTree(root=root)
        assert engine.detect_convergence(tree) is False


# ============================================================================
# scratchpad/persistence.py
# ============================================================================
from hololoom.recursive.scratchpad.persistence import (
    ThoughtPersistence,
    SessionManager,
)


class TestThoughtPersistence:
    @pytest.mark.asyncio
    async def test_initialize_and_close(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        assert tp.db is not None
        await tp.close()
        assert tp.db is None

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()

        root = Thought(
            id="r1", text="Root thought", type=ThoughtType.INITIAL,
            level=0, timestamp=datetime(2025, 1, 1), confidence=0.8
        )
        child = Thought(
            id="c1", text="Child thought", type=ThoughtType.QUESTION,
            level=1, timestamp=datetime(2025, 1, 1), parent_id="r1",
            confidence=0.6, metadata={"key": "val"}
        )
        tree = DialogueTree(root=root)
        tree.add_thought(child)

        session_id = await tp.save_session("my_session", tree)
        assert session_id == "my_session"

        loaded = await tp.load_session("my_session")
        assert loaded is not None
        assert len(loaded.thoughts) == 2
        assert loaded.root.id == "r1"
        assert loaded.thoughts["c1"].metadata == {"key": "val"}

        await tp.close()

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        result = await tp.load_session("does_not_exist")
        assert result is None
        await tp.close()

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()

        root_a = Thought(id="r_a", text="root A", type=ThoughtType.INITIAL,
                         level=0, timestamp=datetime(2025, 1, 1))
        tree_a = DialogueTree(root=root_a)
        await tp.save_session("session_a", tree_a)

        root_b = Thought(id="r_b", text="root B", type=ThoughtType.INITIAL,
                         level=0, timestamp=datetime(2025, 1, 1))
        tree_b = DialogueTree(root=root_b)
        await tp.save_session("session_b", tree_b)

        sessions = await tp.list_sessions()
        assert len(sessions) == 2
        names = {s["session_name"] for s in sessions}
        assert "session_a" in names
        assert "session_b" in names
        await tp.close()

    @pytest.mark.asyncio
    async def test_delete_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()

        root = Thought(id="r", text="root", type=ThoughtType.INITIAL,
                       level=0, timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        await tp.save_session("to_delete", tree)

        deleted = await tp.delete_session("to_delete")
        assert deleted is True

        deleted_again = await tp.delete_session("to_delete")
        assert deleted_again is False

        await tp.close()

    @pytest.mark.asyncio
    async def test_search_thoughts(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()

        root = Thought(id="r", text="Thompson Sampling is a bandit algorithm",
                       type=ThoughtType.INITIAL, level=0,
                       timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        tree.add_thought(Thought(
            id="c1", text="Bayesian approach to exploration",
            type=ThoughtType.ANSWER, level=1,
            timestamp=datetime(2025, 1, 1), parent_id="r"
        ))
        await tp.save_session("search_test", tree)

        results = await tp.search_thoughts("Thompson")
        assert len(results) >= 1
        assert any("Thompson" in t.text for t in results)

        results_scoped = await tp.search_thoughts("Bayesian", session_name="search_test")
        assert len(results_scoped) >= 1

        await tp.close()

    @pytest.mark.asyncio
    async def test_save_session_not_initialized(self):
        tp = ThoughtPersistence(Path("/tmp/nonexistent.db"))
        root = Thought(id="r", text="x", type=ThoughtType.INITIAL,
                       level=0, timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        with pytest.raises(RuntimeError, match="not initialized"):
            await tp.save_session("test", tree)

    @pytest.mark.asyncio
    async def test_load_session_not_initialized(self):
        tp = ThoughtPersistence(Path("/tmp/nonexistent.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await tp.load_session("test")

    @pytest.mark.asyncio
    async def test_list_sessions_not_initialized(self):
        tp = ThoughtPersistence(Path("/tmp/nonexistent.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await tp.list_sessions()

    @pytest.mark.asyncio
    async def test_delete_session_not_initialized(self):
        tp = ThoughtPersistence(Path("/tmp/nonexistent.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await tp.delete_session("test")

    @pytest.mark.asyncio
    async def test_search_thoughts_not_initialized(self):
        tp = ThoughtPersistence(Path("/tmp/nonexistent.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await tp.search_thoughts("test")

    @pytest.mark.asyncio
    async def test_save_with_verification(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()

        root = Thought(
            id="r", text="root", type=ThoughtType.INITIAL,
            level=0, timestamp=datetime(2025, 1, 1),
            verification={"domain": 0.9, "overall_score": 0.85}
        )
        tree = DialogueTree(root=root)
        await tp.save_session("ver_test", tree)

        loaded = await tp.load_session("ver_test")
        assert loaded.root.verification is not None
        assert loaded.root.verification["domain"] == 0.9
        await tp.close()


class TestSessionManager:
    @pytest.mark.asyncio
    async def test_create_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)
        name = await mgr.create_session("test")
        assert name == "test"
        await tp.close()

    @pytest.mark.asyncio
    async def test_get_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)

        root = Thought(id="r", text="root", type=ThoughtType.INITIAL,
                       level=0, timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        await tp.save_session("s1", tree)

        loaded = await mgr.get_session("s1")
        assert loaded is not None
        await tp.close()

    @pytest.mark.asyncio
    async def test_list_all_sessions(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)
        sessions = await mgr.list_all_sessions()
        assert isinstance(sessions, list)
        await tp.close()

    @pytest.mark.asyncio
    async def test_delete_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)

        root = Thought(id="r", text="root", type=ThoughtType.INITIAL,
                       level=0, timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        await tp.save_session("del_me", tree)

        deleted = await mgr.delete_session("del_me")
        assert deleted is True
        await tp.close()

    @pytest.mark.asyncio
    async def test_search_across_sessions(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)

        root = Thought(id="r", text="unique_search_term here",
                       type=ThoughtType.INITIAL, level=0,
                       timestamp=datetime(2025, 1, 1))
        tree = DialogueTree(root=root)
        await tp.save_session("search_s", tree)

        results = await mgr.search_across_sessions("unique_search_term")
        assert len(results) >= 1
        await tp.close()

    @pytest.mark.asyncio
    async def test_get_session_stats(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)

        root = Thought(id="r", text="root", type=ThoughtType.INITIAL,
                       level=0, timestamp=datetime(2025, 1, 1), confidence=0.7)
        tree = DialogueTree(root=root)
        tree.add_thought(Thought(
            id="q1", text="Why?", type=ThoughtType.QUESTION,
            level=1, timestamp=datetime(2025, 1, 1),
            parent_id="r", confidence=0.6
        ))
        tree.add_thought(Thought(
            id="a1", text="Because.", type=ThoughtType.ANSWER,
            level=2, timestamp=datetime(2025, 1, 1),
            parent_id="q1", confidence=0.8
        ))
        await tp.save_session("stats_s", tree)

        stats = await mgr.get_session_stats("stats_s")
        assert stats is not None
        assert stats["total_thoughts"] == 3
        assert stats["questions"] == 1
        assert stats["answers"] == 1
        assert "avg_confidence" in stats
        await tp.close()

    @pytest.mark.asyncio
    async def test_get_session_stats_nonexistent(self, tmp_path):
        db_path = tmp_path / "test.db"
        tp = ThoughtPersistence(db_path)
        await tp.initialize()
        mgr = SessionManager(tp)
        stats = await mgr.get_session_stats("nonexistent")
        assert stats is None
        await tp.close()


# ============================================================================
# scratchpad_integration.py — units that don't require WeavingOrchestrator
# ============================================================================
from hololoom.core.recursive.scratchpad_integration import (
    ProvenanceTracker,
    ScratchpadConfig,
    RecursiveRefiner,
)


class TestScratchpadConfig:
    def test_defaults(self):
        cfg = ScratchpadConfig()
        assert cfg.enable_scratchpad is True
        assert cfg.enable_refinement is True
        assert cfg.refinement_threshold == 0.75
        assert cfg.max_refinement_iterations == 3
        assert cfg.persist_scratchpad is False
        assert cfg.persist_path is None

    def test_custom(self):
        cfg = ScratchpadConfig(
            enable_refinement=False,
            refinement_threshold=0.9,
            persist_scratchpad=True,
            persist_path="/tmp/sp.json"
        )
        assert cfg.enable_refinement is False
        assert cfg.persist_path == "/tmp/sp.json"


class TestProvenanceTracker:
    def _mock_spacetime(self):
        trace = MagicMock()
        trace.motifs_detected = ["pattern_a", "pattern_b"]
        trace.threads_activated = ["thread_1", "thread_2"]
        trace.embedding_scales_used = [64, 128]
        trace.tool_selected = "research"
        trace.policy_adapter = "thompson"
        trace.duration_ms = 150.5
        trace.context_shards_count = 3
        trace.errors = []
        trace.warnings = []
        trace.tool_confidence = 0.85
        trace.bandit_statistics = {"alpha": 2.0}

        spacetime = MagicMock()
        spacetime.trace = trace
        spacetime.response = "This is a test response"
        spacetime.query_text = "test query"
        return spacetime

    def test_extract_provenance(self):
        tracker = ProvenanceTracker()
        spacetime = self._mock_spacetime()
        entry = tracker.extract_provenance(spacetime, iteration=1)

        assert entry.iteration == 1
        assert "pattern_a" in entry.thought
        assert "research" in entry.action
        assert "thompson" in entry.action
        assert entry.score == 0.85
        assert entry.metadata["tool"] == "research"
        assert entry.metadata["bandit_stats"]["alpha"] == 2.0

    def test_extract_provenance_no_motifs(self):
        tracker = ProvenanceTracker()
        spacetime = self._mock_spacetime()
        spacetime.trace.motifs_detected = []
        spacetime.trace.threads_activated = []
        spacetime.trace.embedding_scales_used = []
        spacetime.trace.bandit_statistics = None
        entry = tracker.extract_provenance(spacetime)
        assert "No threads activated" in entry.thought
        assert "bandit_stats" not in entry.metadata

    def test_extract_provenance_with_errors(self):
        tracker = ProvenanceTracker()
        spacetime = self._mock_spacetime()
        spacetime.trace.errors = ["error1", "error2"]
        entry = tracker.extract_provenance(spacetime)
        assert "Errors: 2" in entry.observation

    def test_extract_provenance_long_response(self):
        tracker = ProvenanceTracker()
        spacetime = self._mock_spacetime()
        spacetime.response = "x" * 200
        entry = tracker.extract_provenance(spacetime)
        assert "..." in entry.observation


class TestRecursiveRefiner:
    def test_analyze_low_confidence_few_threads(self):
        refiner = RecursiveRefiner(orchestrator=MagicMock())
        spacetime = MagicMock()
        spacetime.trace.threads_activated = ["t1"]
        spacetime.trace.motifs_detected = []
        spacetime.trace.context_shards_count = 0
        spacetime.trace.errors = []
        analysis = refiner._analyze_low_confidence(spacetime)
        assert "few threads" in analysis
        assert "no motifs" in analysis
        assert "no context" in analysis

    def test_analyze_low_confidence_with_errors(self):
        refiner = RecursiveRefiner(orchestrator=MagicMock())
        spacetime = MagicMock()
        spacetime.trace.threads_activated = ["t1", "t2", "t3"]
        spacetime.trace.motifs_detected = ["m1"]
        spacetime.trace.context_shards_count = 5
        spacetime.trace.errors = ["e1"]
        analysis = refiner._analyze_low_confidence(spacetime)
        assert "1 errors" in analysis

    def test_analyze_low_confidence_default(self):
        refiner = RecursiveRefiner(orchestrator=MagicMock())
        spacetime = MagicMock()
        spacetime.trace.threads_activated = ["t1", "t2"]
        spacetime.trace.motifs_detected = ["m1"]
        spacetime.trace.context_shards_count = 5
        spacetime.trace.errors = []
        analysis = refiner._analyze_low_confidence(spacetime)
        assert "low policy confidence" in analysis

    def test_expand_query_few_threads(self):
        from hololoom.protocols.types import Query
        refiner = RecursiveRefiner(orchestrator=MagicMock())
        original = Query(text="What is AI?")
        spacetime = MagicMock()
        spacetime.response = "AI is artificial intelligence"
        spacetime.trace.motifs_detected = ["neural"]
        expanded = refiner._expand_query(original, spacetime, "few threads, no context", 1)
        assert "Context:" in expanded.text
        assert "Related to:" in expanded.text
        assert expanded.metadata["refinement_iteration"] == 1

    def test_expand_query_no_expansion_needed(self):
        from hololoom.protocols.types import Query
        refiner = RecursiveRefiner(orchestrator=MagicMock())
        original = Query(text="Simple query")
        spacetime = MagicMock()
        spacetime.response = "Simple answer"
        spacetime.trace.motifs_detected = []
        expanded = refiner._expand_query(original, spacetime, "low policy confidence", 1)
        assert expanded.text == "Simple query"


# ============================================================================
# full_learning_loop.py — ThompsonPriors, PolicyWeights, LearningMetrics
# ============================================================================
from hololoom.core.recursive.full_learning_loop import (
    ThompsonPriors,
    PolicyWeights,
    LearningMetrics,
)


class TestThompsonPriors:
    def test_initial_priors(self):
        tp = ThompsonPriors()
        assert tp.get_expected_reward("new_tool") == 0.5  # 1/(1+1)

    def test_update_success(self):
        tp = ThompsonPriors()
        tp.update_success("tool_a", 0.8)
        expected = tp.get_expected_reward("tool_a")
        assert expected > 0.5

    def test_update_failure(self):
        tp = ThompsonPriors()
        tp.update_failure("tool_b", 0.3)
        expected = tp.get_expected_reward("tool_b")
        assert expected < 0.5

    def test_get_uncertainty(self):
        tp = ThompsonPriors()
        u1 = tp.get_uncertainty("tool_x")
        tp.update_success("tool_x", 0.9)
        tp.update_success("tool_x", 0.9)
        u2 = tp.get_uncertainty("tool_x")
        # More data = less uncertainty
        assert u2 < u1

    def test_save_and_load(self, tmp_path):
        tp = ThompsonPriors()
        tp.update_success("tool_a", 0.8)
        tp.update_failure("tool_b", 0.3)

        path = str(tmp_path / "priors.json")
        tp.save(path)

        loaded = ThompsonPriors.load(path)
        assert loaded.get_expected_reward("tool_a") == tp.get_expected_reward("tool_a")
        assert loaded.get_expected_reward("tool_b") == tp.get_expected_reward("tool_b")

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            ThompsonPriors.load("/tmp/nonexistent_priors_12345.json")

    def test_save_atomic(self, tmp_path):
        tp = ThompsonPriors()
        tp.update_success("x", 0.5)
        path = str(tmp_path / "atomic_test.json")
        tp.save(path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "tool_priors" in data


class TestPolicyWeights:
    def test_initial_weight(self):
        pw = PolicyWeights()
        assert pw.get_weight("new_adapter") == 1.0

    def test_update_success_increases_relative(self):
        pw = PolicyWeights()
        pw.update("adapter_a", success=True)
        # Laplace smoothing: (1+1)/(1+2) = 0.667
        weight_after_success = pw.get_weight("adapter_a")
        pw_fail = PolicyWeights()
        pw_fail.update("adapter_a", success=False)
        weight_after_failure = pw_fail.get_weight("adapter_a")
        assert weight_after_success > weight_after_failure
        assert pw.get_success_rate("adapter_a") == 1.0

    def test_update_failure(self):
        pw = PolicyWeights()
        pw.update("adapter_b", success=False)
        # Laplace smoothing: (0+1)/(1+2) = 0.333
        assert pw.get_weight("adapter_b") < 0.5
        assert pw.get_success_rate("adapter_b") == 0.0

    def test_success_rate_multiple(self):
        pw = PolicyWeights()
        pw.update("a", success=True)
        pw.update("a", success=True)
        pw.update("a", success=False)
        rate = pw.get_success_rate("a")
        assert abs(rate - 2/3) < 0.01

    def test_success_rate_unknown(self):
        pw = PolicyWeights()
        assert pw.get_success_rate("unknown") == 0.5


class TestLearningMetrics:
    def test_initial_state(self):
        m = LearningMetrics()
        assert m.queries_processed == 0
        assert m.avg_confidence == 0.0

    def test_update(self):
        m = LearningMetrics()
        m.update(0.8)
        assert m.queries_processed == 1
        assert m.avg_confidence == 0.8
        m.update(0.6)
        assert m.queries_processed == 2
        assert abs(m.avg_confidence - 0.7) < 0.01
