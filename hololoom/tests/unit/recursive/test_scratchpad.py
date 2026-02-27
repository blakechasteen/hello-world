"""
Unit tests for HoloLoom.recursive.scratchpad

Tests scratchpad provenance tracking, entry management, and history retrieval.
"""

import pytest
import time
from hololoom.recursive.scratchpad import (
    Scratchpad,
    ScratchpadEntry,
    LoopType,
    LoopConfig,
    LoopResult,
    RecursiveEngine
)


class TestScratchpadEntry:
    """Test ScratchpadEntry dataclass"""

    def test_entry_creation(self):
        """Test creating a basic entry"""
        entry = ScratchpadEntry(
            thought="Analyzing query complexity",
            action="extract_features",
            observation="Found 3 motifs, 5 threads",
            score=0.85
        )

        assert entry.thought == "Analyzing query complexity"
        assert entry.action == "extract_features"
        assert entry.observation == "Found 3 motifs, 5 threads"
        assert entry.score == 0.85
        assert entry.iteration is None
        assert entry.timestamp is None

    def test_entry_with_metadata(self):
        """Test entry with optional metadata"""
        metadata = {"complexity": "high", "motifs": ["AI", "ML", "neural"]}
        entry = ScratchpadEntry(
            thought="Complex query detected",
            action="research",
            observation="Deep analysis required",
            score=0.75,
            iteration=1,
            timestamp=time.time(),
            metadata=metadata
        )

        assert entry.iteration == 1
        assert entry.timestamp is not None
        assert entry.metadata == metadata
        assert "complexity" in entry.metadata


class TestScratchpad:
    """Test Scratchpad class"""

    def test_initialization(self):
        """Test scratchpad initialization"""
        scratchpad = Scratchpad(capacity=100)

        assert len(scratchpad) == 0
        assert scratchpad.capacity == 100
        assert scratchpad.entries == []

    def test_add_entry(self):
        """Test adding entries to scratchpad"""
        scratchpad = Scratchpad()

        scratchpad.add_entry(
            thought="Test thought",
            action="test_action",
            observation="Test observation",
            score=0.9
        )

        assert len(scratchpad) == 1
        assert scratchpad.entries[0].thought == "Test thought"
        assert scratchpad.entries[0].score == 0.9

    def test_add_multiple_entries(self):
        """Test adding multiple entries"""
        scratchpad = Scratchpad()

        for i in range(5):
            scratchpad.add_entry(
                thought=f"Thought {i}",
                action=f"action_{i}",
                observation=f"Observation {i}",
                score=0.8 + i * 0.02
            )

        assert len(scratchpad) == 5
        assert scratchpad.entries[0].thought == "Thought 0"
        assert scratchpad.entries[4].thought == "Thought 4"
        assert scratchpad.entries[4].score == 0.88

    def test_capacity_limit(self):
        """Test scratchpad respects capacity limit"""
        scratchpad = Scratchpad(capacity=3)

        # Add 5 entries
        for i in range(5):
            scratchpad.add_entry(
                thought=f"Thought {i}",
                action=f"action_{i}",
                observation=f"Observation {i}",
                score=0.8
            )

        # Should only keep last 3
        assert len(scratchpad) == 3
        assert scratchpad.entries[0].thought == "Thought 2"
        assert scratchpad.entries[2].thought == "Thought 4"

    def test_get_history(self):
        """Test getting complete history"""
        scratchpad = Scratchpad()

        scratchpad.add_entry("T1", "A1", "O1", 0.8)
        scratchpad.add_entry("T2", "A2", "O2", 0.9)

        history = scratchpad.get_history()

        assert len(history) == 2
        assert history[0].thought == "T1"
        assert history[1].thought == "T2"
        # Should be a copy, not the original
        assert history is not scratchpad.entries

    def test_get_last_n(self):
        """Test getting last N entries"""
        scratchpad = Scratchpad()

        for i in range(10):
            scratchpad.add_entry(f"T{i}", f"A{i}", f"O{i}", 0.8)

        last_3 = scratchpad.get_last_n(3)

        assert len(last_3) == 3
        assert last_3[0].thought == "T7"
        assert last_3[2].thought == "T9"

    def test_get_last_n_more_than_available(self):
        """Test get_last_n when requesting more than available"""
        scratchpad = Scratchpad()

        scratchpad.add_entry("T1", "A1", "O1", 0.8)
        scratchpad.add_entry("T2", "A2", "O2", 0.9)

        last_5 = scratchpad.get_last_n(5)

        # Should return all available entries
        assert len(last_5) == 2

    def test_clear(self):
        """Test clearing scratchpad"""
        scratchpad = Scratchpad()

        scratchpad.add_entry("T1", "A1", "O1", 0.8)
        scratchpad.add_entry("T2", "A2", "O2", 0.9)

        assert len(scratchpad) == 2

        scratchpad.clear()

        assert len(scratchpad) == 0
        assert scratchpad.entries == []

    def test_add_entry_with_iteration(self):
        """Test tracking iteration numbers"""
        scratchpad = Scratchpad()

        for i in range(3):
            scratchpad.add_entry(
                thought=f"Iteration {i}",
                action="refine",
                observation=f"Quality: {0.7 + i*0.1}",
                score=0.7 + i*0.1,
                iteration=i
            )

        assert scratchpad.entries[0].iteration == 0
        assert scratchpad.entries[1].iteration == 1
        assert scratchpad.entries[2].iteration == 2

    def test_metadata_preservation(self):
        """Test metadata is preserved correctly"""
        scratchpad = Scratchpad()

        metadata = {
            "tool": "research",
            "confidence": 0.85,
            "threads": ["thread1", "thread2"]
        }

        scratchpad.add_entry(
            thought="Complex query",
            action="deep_search",
            observation="Found relevant info",
            score=0.85,
            metadata=metadata
        )

        retrieved = scratchpad.get_history()[0]
        assert retrieved.metadata == metadata
        assert retrieved.metadata["tool"] == "research"


class TestLoopConfig:
    """Test LoopConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = LoopConfig()

        assert config.max_iterations == 3
        assert config.quality_threshold == 0.85
        assert config.loop_type == LoopType.REFINE
        assert config.enable_learning is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = LoopConfig(
            max_iterations=5,
            quality_threshold=0.9,
            loop_type=LoopType.CRITIQUE,
            enable_learning=False
        )

        assert config.max_iterations == 5
        assert config.quality_threshold == 0.9
        assert config.loop_type == LoopType.CRITIQUE
        assert config.enable_learning is False


class TestLoopResult:
    """Test LoopResult dataclass"""

    def test_result_creation(self):
        """Test creating loop result"""
        result = LoopResult(
            success=True,
            iterations=3,
            final_quality=0.92,
            improvement=0.15
        )

        assert result.success is True
        assert result.iterations == 3
        assert result.final_quality == 0.92
        assert result.improvement == 0.15
        assert result.history == []

    def test_result_with_history(self):
        """Test result with scratchpad history"""
        entry1 = ScratchpadEntry("T1", "A1", "O1", 0.8)
        entry2 = ScratchpadEntry("T2", "A2", "O2", 0.9)

        result = LoopResult(
            success=True,
            iterations=2,
            final_quality=0.9,
            improvement=0.1,
            history=[entry1, entry2]
        )

        assert len(result.history) == 2
        assert result.history[0].score == 0.8
        assert result.history[1].score == 0.9

    def test_summary_successful(self):
        """Test summary generation for successful loop"""
        result = LoopResult(
            success=True,
            iterations=3,
            final_quality=0.92,
            improvement=0.15
        )

        summary = result.summary()

        assert "3 iterations" in summary
        assert "0.920" in summary
        assert "+0.150" in summary

    def test_summary_unsuccessful(self):
        """Test summary with negative improvement"""
        result = LoopResult(
            success=False,
            iterations=2,
            final_quality=0.65,
            improvement=-0.05
        )

        summary = result.summary()

        assert "2 iterations" in summary
        assert "0.650" in summary
        assert "-0.050" in summary


class TestRecursiveEngine:
    """Test RecursiveEngine class"""

    def test_initialization(self):
        """Test engine initialization"""
        config = LoopConfig(max_iterations=5)
        engine = RecursiveEngine(config)

        assert engine.config.max_iterations == 5
        assert isinstance(engine.scratchpad, Scratchpad)

    @pytest.mark.asyncio
    async def test_run_loop_basic(self):
        """Test basic loop execution"""
        config = LoopConfig(
            max_iterations=3,
            quality_threshold=0.8
        )
        engine = RecursiveEngine(config)

        result = await engine.run_loop(
            initial_input="Test query",
            initial_quality=0.5
        )

        assert isinstance(result, LoopResult)
        assert result.iterations >= 1
        assert result.final_quality >= result.improvement

    @pytest.mark.asyncio
    async def test_run_loop_reaches_threshold(self):
        """Test loop reaches quality threshold"""
        config = LoopConfig(
            max_iterations=5,
            quality_threshold=0.85
        )
        engine = RecursiveEngine(config)

        result = await engine.run_loop(
            initial_input="Test query",
            initial_quality=0.6
        )

        # Simple stub should reach or exceed threshold
        assert result.final_quality >= config.quality_threshold
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_loop_different_types(self):
        """Test different loop types"""
        for loop_type in LoopType:
            config = LoopConfig(loop_type=loop_type)
            engine = RecursiveEngine(config)

            result = await engine.run_loop("Test", 0.5)

            assert result is not None
            assert isinstance(result, LoopResult)

    def test_get_scratchpad(self):
        """Test retrieving scratchpad"""
        config = LoopConfig()
        engine = RecursiveEngine(config)

        scratchpad = engine.get_scratchpad()

        assert isinstance(scratchpad, Scratchpad)
        assert scratchpad is engine.scratchpad


class TestLoopType:
    """Test LoopType enum"""

    def test_loop_types_exist(self):
        """Test all expected loop types exist"""
        assert LoopType.REFINE.value == "refine"
        assert LoopType.VERIFY.value == "verify"
        assert LoopType.CRITIQUE.value == "critique"
        assert LoopType.ELEGANCE.value == "elegance"
        assert LoopType.HOFSTADTER.value == "hofstadter"

    def test_loop_type_iteration(self):
        """Test iterating over loop types"""
        types = list(LoopType)

        assert len(types) == 5
        assert LoopType.REFINE in types
        assert LoopType.HOFSTADTER in types


# Integration tests
class TestScratchpadIntegration:
    """Integration tests for scratchpad workflow"""

    def test_complete_refinement_workflow(self):
        """Test complete refinement workflow with scratchpad"""
        scratchpad = Scratchpad(capacity=10)

        # Initial query
        scratchpad.add_entry(
            thought="Low confidence result",
            action="initial_weave",
            observation="Confidence: 0.65",
            score=0.65,
            iteration=0
        )

        # Refinement iteration 1
        scratchpad.add_entry(
            thought="Expanding context",
            action="refine_query",
            observation="Added 2 more threads",
            score=0.75,
            iteration=1
        )

        # Refinement iteration 2
        scratchpad.add_entry(
            thought="Final refinement",
            action="critique_and_improve",
            observation="Confidence: 0.92",
            score=0.92,
            iteration=2
        )

        history = scratchpad.get_history()

        assert len(history) == 3
        assert history[0].score == 0.65
        assert history[-1].score == 0.92
        # Quality improved
        assert history[-1].score > history[0].score

    def test_multi_query_session(self):
        """Test tracking multiple queries in one session"""
        scratchpad = Scratchpad(capacity=100)

        queries = [
            ("What is AI?", 0.95),
            ("How does ML work?", 0.88),
            ("Explain neural networks", 0.82)
        ]

        for i, (query, score) in enumerate(queries):
            scratchpad.add_entry(
                thought=f"Query: {query}",
                action="weave",
                observation=f"Confidence: {score}",
                score=score,
                metadata={"query_index": i, "query_text": query}
            )

        assert len(scratchpad) == 3

        # Check metadata preservation
        for i, entry in enumerate(scratchpad.get_history()):
            assert entry.metadata["query_index"] == i
            assert queries[i][0] in entry.metadata["query_text"]
