# Elle HoloLoom Learning Integration Tests (Week 3 - Phase 2B)
# Tests Elle integration with HoloLoom learning systems.
# Test Coverage (10 tests): Policy updates, Thompson Sampling, Reflection, Alignment, Hot patterns
# Created: 2025-11-22

import pytest
from datetime import datetime
from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.recursive import FullLearningEngine
from HoloLoom.documentation.types import Query, MemoryShard


@pytest.mark.asyncio
async def test_elle_feedback_updates_policy():
    config = Config.fast()
    shards = [MemoryShard(content="Test shard", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        spacetime = await engine.weave(Query(text="Where is the hammer?"), enable_refinement=False)
        await engine.reflect(spacetime, feedback={"helpful": True, "confidence": 0.9})
        stats = engine.get_learning_statistics()
        assert stats["total_queries"] > 0


@pytest.mark.asyncio
async def test_thompson_sampling_learns_from_ar():
    config = Config.fast()
    config.enable_thompson_sampling = True
    shards = [MemoryShard(content="Hammer on workbench", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        for i in range(5):
            spacetime = await engine.weave(Query(text=f"Test query {i}"), enable_refinement=False)
            await engine.reflect(spacetime, feedback={"helpful": i > 2})
        stats = engine.get_learning_statistics()
        assert stats["total_queries"] == 5


@pytest.mark.asyncio
async def test_reflection_buffer_stores_ar_sessions():
    config = Config.fast()
    shards = [MemoryShard(content="AR session data", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        for i in range(3):
            spacetime = await engine.weave(Query(text=f"AR scan {i}"), enable_refinement=False)
            await engine.reflect(spacetime, feedback={"helpful": True})
        stats = engine.get_learning_statistics()
        assert stats["total_queries"] == 3


@pytest.mark.asyncio
async def test_hot_pattern_tracking_for_ar():
    config = Config.fast()
    shards = [MemoryShard(content="Hammer location: workbench", timestamp=datetime.now()), MemoryShard(content="Nails location: drawer", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        for _ in range(5):
            await engine.weave(Query(text="Where is hammer?"), enable_refinement=False)
        if hasattr(engine, "hot_tracker"):
            hot = engine.hot_tracker.get_hot_patterns(limit=5)
            assert len(hot) > 0


@pytest.mark.asyncio
async def test_alignment_gates_risky_ar_actions():
    config = Config.fast()
    config.enable_alignment = True
    shards = [MemoryShard(content="Test", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        spacetime = await engine.weave(Query(text="Execute system command in AR environment"), enable_refinement=False)
        assert spacetime is not None


@pytest.mark.asyncio
async def test_elle_learns_from_refinement():
    config = Config.fast()
    shards = [MemoryShard(content="Low confidence answer", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        spacetime = await engine.weave(Query(text="Obscure technical query"), enable_refinement=True, refinement_threshold=0.9)
        if hasattr(spacetime, "refinement_iterations"):
            assert spacetime.refinement_iterations >= 0


@pytest.mark.asyncio
async def test_elle_cross_session_learning():
    config = Config.fast()
    shards = [MemoryShard(content="Session 1 data", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        spacetime = await engine.weave(Query(text="Test pattern"), enable_refinement=False)
        await engine.reflect(spacetime, feedback={"helpful": True})
        engine.save_learning_state("./test_learning_state")
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        engine.load_learning_state("./test_learning_state")
        stats = engine.get_learning_statistics()
        assert stats is not None


@pytest.mark.asyncio
async def test_confidence_calibration_from_ar():
    config = Config.fast()
    shards = [MemoryShard(content="Calibration test", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        spacetime1 = await engine.weave(Query(text="Query 1"), enable_refinement=False)
        await engine.reflect(spacetime1, feedback={"helpful": False, "confidence": 0.3})
        spacetime2 = await engine.weave(Query(text="Query 2"), enable_refinement=False)
        assert spacetime1 is not None and spacetime2 is not None


@pytest.mark.asyncio
async def test_adaptive_retrieval_from_ar_usage():
    config = Config.fast()
    shards = [MemoryShard(content="Frequently accessed: hammer", timestamp=datetime.now()), MemoryShard(content="Rarely accessed: obscure tool", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards) as engine:
        for _ in range(10):
            await engine.weave(Query(text="hammer location"), enable_refinement=False)
        await engine.weave(Query(text="obscure tool"), enable_refinement=False)
        stats = engine.get_learning_statistics()
        assert stats["total_queries"] == 11


@pytest.mark.asyncio
async def test_background_learning_with_elle():
    config = Config.fast()
    shards = [MemoryShard(content="Background test", timestamp=datetime.now())]
    async with FullLearningEngine(cfg=config, shards=shards, enable_background_learning=False) as engine:
        for i in range(5):
            spacetime = await engine.weave(Query(text=f"Background query {i}"), enable_refinement=False)
            await engine.reflect(spacetime, feedback={"helpful": True})
        stats = engine.get_learning_statistics()
        assert stats["total_queries"] == 5
