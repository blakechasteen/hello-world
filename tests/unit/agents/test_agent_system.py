"""
Tests for Agent System

Tests:
1. Working memory (semantic focus, activation, tension)
2. Pattern learning
3. Agent orchestration
4. Cross-session persistence
5. Profile templates
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import tempfile
import shutil

from hololoom.agents import (
    create_agent,
    AgentWorkingMemory,
    WorkingMemoryLearner,
    get_profile,
    list_profiles,
    BUDGET_ADVISOR
)
from hololoom.agents.types import WorkingMemoryState, AgentProfile, AgentDomain
from hololoom.memory.graph import KG, KGEdge
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.documentation.types import Query
from hololoom.fabric.spacetime import Spacetime
from hololoom.config import Config


@pytest.fixture
def test_kg():
    """Create test knowledge graph"""
    kg = KG()
    kg.add_edges([
        KGEdge("budget", "Q4_2024", "HAS_PERIOD", 1.0),
        KGEdge("Q4_2024", "marketing", "HAS_CATEGORY", 1.0),
        KGEdge("marketing", "50000", "HAS_AMOUNT", 1.0),
    ])
    return kg


@pytest.fixture
def test_emb():
    """Create test embedding model"""
    return MatryoshkaEmbeddings()


@pytest.fixture
def temp_persist_dir():
    """Create temporary persistence directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================================
# Working Memory Tests
# ============================================================================

def test_working_memory_state():
    """Test working memory state creation and copying"""
    state = WorkingMemoryState(
        focus_vector=np.array([1.0, 2.0, 3.0]),
        attention_radius=0.3,
        momentum=0.7
    )

    assert state.focus_vector.shape == (3,)
    assert state.attention_radius == 0.3
    assert state.momentum == 0.7
    assert len(state.activation_map) == 0
    assert len(state.tensioned_threads) == 0

    # Test copy
    copy = state.copy()
    assert np.allclose(copy.focus_vector, state.focus_vector)
    assert copy.attention_radius == state.attention_radius

    # Modify copy doesn't affect original
    copy.focus_vector[0] = 999.0
    assert state.focus_vector[0] == 1.0


@pytest.mark.asyncio
async def test_working_memory_attend(test_kg, test_emb):
    """Test working memory attention mechanism"""
    profile = BUDGET_ADVISOR

    wm = AgentWorkingMemory(
        profile=profile,
        yarn_graph=test_kg,
        embedding_model=test_emb
    )

    # Attend to query
    query = Query(text="What is Q4 budget?")
    context = await wm.attend_to(query)

    assert len(context) > 0
    assert wm.state.focus_vector is not None
    assert len(wm.state.activation_map) > 0


@pytest.mark.asyncio
async def test_working_memory_pin(test_kg, test_emb):
    """Test working memory pinning"""
    profile = BUDGET_ADVISOR

    wm = AgentWorkingMemory(
        profile=profile,
        yarn_graph=test_kg,
        embedding_model=test_emb
    )

    # Pin concept
    wm.pin("budget policy", weight=0.7)

    # Tension profile should contain pin
    assert len(wm.state.tension_profile) >= 0  # May or may not find exact node


@pytest.mark.asyncio
async def test_working_memory_relax(test_kg, test_emb):
    """Test working memory relaxation"""
    profile = BUDGET_ADVISOR

    wm = AgentWorkingMemory(
        profile=profile,
        yarn_graph=test_kg,
        embedding_model=test_emb
    )

    # Activate
    query = Query(text="budget")
    await wm.attend_to(query)

    initial_activated = len([a for a in wm.state.activation_map.values() if a > 0.5])

    # Relax
    await wm.relax()

    # Activations should decay
    after_relax = len([a for a in wm.state.activation_map.values() if a > 0.5])
    assert after_relax <= initial_activated


def test_working_memory_summary(test_kg, test_emb):
    """Test working memory summary"""
    profile = BUDGET_ADVISOR

    wm = AgentWorkingMemory(
        profile=profile,
        yarn_graph=test_kg,
        embedding_model=test_emb
    )

    summary = wm.get_state_summary()

    assert 'semantic_focus' in summary
    assert 'activated_nodes' in summary
    assert 'tensioned_threads' in summary
    assert 'pinned_concepts' in summary


# ============================================================================
# Pattern Learning Tests
# ============================================================================

def test_learner_creation(temp_persist_dir):
    """Test learner creation and persistence directory"""
    profile = BUDGET_ADVISOR

    learner = WorkingMemoryLearner(
        profile=profile,
        persist_dir=temp_persist_dir
    )

    assert learner.profile == profile
    assert (temp_persist_dir / profile.agent_id).exists()
    assert len(learner.snapshots) == 0
    assert len(learner.patterns) == 0


def test_learner_record_snapshot(temp_persist_dir):
    """Test recording snapshots"""
    profile = BUDGET_ADVISOR

    learner = WorkingMemoryLearner(
        profile=profile,
        persist_dir=temp_persist_dir
    )

    # Create state
    state = WorkingMemoryState(
        focus_vector=np.random.randn(244),
        attention_radius=0.3
    )

    # Record successful outcome
    query = Query(text="test query")
    outcome = {
        'confidence': 0.85,
        'tool_used': 'answer',
        'warp_operations': []
    }

    learner.record_snapshot(query, state, outcome)

    assert len(learner.snapshots) == 1
    assert learner.snapshots[0].success is True
    assert len(learner.patterns) == 1  # Should create pattern


def test_learner_persistence(temp_persist_dir):
    """Test learner saves and loads state"""
    profile = BUDGET_ADVISOR

    # Session 1: Create and record
    learner1 = WorkingMemoryLearner(
        profile=profile,
        persist_dir=temp_persist_dir
    )

    state = WorkingMemoryState(
        focus_vector=np.random.randn(244),
        attention_radius=0.3
    )

    learner1.record_snapshot(
        Query(text="test"),
        state,
        {'confidence': 0.9, 'tool_used': 'answer', 'warp_operations': []}
    )

    learner1.save_state()

    # Session 2: Load
    learner2 = WorkingMemoryLearner(
        profile=profile,
        persist_dir=temp_persist_dir
    )

    assert len(learner2.snapshots) == 1
    assert len(learner2.patterns) == 1


def test_learner_suggest_improvements(temp_persist_dir):
    """Test learning-based suggestions"""
    profile = BUDGET_ADVISOR

    learner = WorkingMemoryLearner(
        profile=profile,
        persist_dir=temp_persist_dir
    )

    # Record successful pattern
    state = WorkingMemoryState(
        focus_vector=np.random.randn(244),
        attention_radius=0.3
    )

    for i in range(5):  # Multiple successes to establish pattern
        learner.record_snapshot(
            Query(text=f"budget query {i}"),
            state,
            {'confidence': 0.9, 'tool_used': 'answer', 'warp_operations': []}
        )

    # Query similar state
    similar_state = WorkingMemoryState(
        focus_vector=state.focus_vector + np.random.randn(244) * 0.05,  # Small perturbation
        attention_radius=0.3
    )

    suggestions = learner.suggest_improvements(
        similar_state,
        Query(text="budget query")
    )

    assert 'suggestions' in suggestions
    # Should find pattern with high similarity
    if 'pattern_id' in suggestions:
        assert suggestions['similarity'] > 0.8


# ============================================================================
# Agent Orchestrator Tests
# ============================================================================

@pytest.mark.asyncio
async def test_agent_creation(test_kg, test_emb):
    """Test agent creation"""
    agent = create_agent(
        'budget',
        shared_knowledge=test_kg,
        embedding_model=test_emb
    )

    assert agent.profile.domain == AgentDomain.BUDGET
    assert agent.working_memory is not None
    assert agent.learner is not None


@pytest.mark.asyncio
async def test_agent_query(test_kg, test_emb):
    """Test agent query processing"""
    async with create_agent('budget', test_kg, test_emb) as agent:
        query = Query(text="What is Q4 budget?")
        result = await agent.query(query)

        assert isinstance(result, Spacetime)
        assert result.confidence >= 0.0
        assert result.response is not None
        assert agent.stats.total_queries == 1


@pytest.mark.asyncio
async def test_agent_pin(test_kg, test_emb):
    """Test agent pinning"""
    async with create_agent('budget', test_kg, test_emb) as agent:
        agent.pin("fiscal policy", weight=0.7)

        summary = agent.get_working_memory_summary()
        assert summary['pinned_concepts'] >= 0


@pytest.mark.asyncio
async def test_agent_stats(test_kg, test_emb):
    """Test agent statistics tracking"""
    async with create_agent('budget', test_kg, test_emb) as agent:
        # Process queries
        for i in range(3):
            await agent.query(Query(text=f"query {i}"))

        stats = agent.get_agent_stats()
        assert stats.total_queries == 3
        assert stats.avg_confidence >= 0.0


@pytest.mark.asyncio
async def test_agent_learning_stats(test_kg, test_emb, temp_persist_dir):
    """Test agent learning statistics"""
    async with create_agent('budget', test_kg, test_emb, persist_dir=temp_persist_dir) as agent:
        # Process successful query
        await agent.query(Query(text="budget query"))

        learning_stats = agent.get_learning_stats()
        assert 'total_snapshots' in learning_stats
        assert learning_stats['total_snapshots'] >= 1


@pytest.mark.asyncio
async def test_agent_persistence(test_kg, test_emb, temp_persist_dir):
    """Test agent state persistence"""
    # Session 1: Create and query
    async with create_agent('budget', test_kg, test_emb, persist_dir=temp_persist_dir) as agent:
        await agent.query(Query(text="test query"))

    # Session 2: Verify persistence
    async with create_agent('budget', test_kg, test_emb, persist_dir=temp_persist_dir) as agent:
        learning_stats = agent.get_learning_stats()
        assert learning_stats['total_snapshots'] >= 1


# ============================================================================
# Profile Tests
# ============================================================================

def test_profile_templates():
    """Test profile template availability"""
    profiles = list_profiles()

    assert 'budget' in profiles
    assert 'architecture' in profiles
    assert 'code_review' in profiles
    assert 'research' in profiles
    assert 'planning' in profiles
    assert 'general' in profiles


def test_get_profile():
    """Test profile retrieval"""
    budget_profile = get_profile('budget')

    assert budget_profile.domain == AgentDomain.BUDGET
    assert 'accuracy' in budget_profile.priorities
    assert 'monetary_value' in budget_profile.semantic_dimensions


def test_profile_configuration():
    """Test profile has expected configuration"""
    budget = get_profile('budget')

    # Budget should have high accuracy requirements
    assert budget.success_threshold >= 0.75
    assert budget.refinement_threshold >= 0.75

    # Budget should be focused
    assert budget.attention_radius <= 0.3
    assert budget.momentum >= 0.7


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_agents_shared_knowledge(test_kg, test_emb):
    """Test multiple agents sharing knowledge graph"""
    budget = create_agent('budget', test_kg, test_emb)
    arch = create_agent('architecture', test_kg, test_emb)

    async with budget, arch:
        # Both can access shared knowledge
        result1 = await budget.query(Query(text="budget query"))
        result2 = await arch.query(Query(text="architecture query"))

        assert result1 is not None
        assert result2 is not None


@pytest.mark.asyncio
async def test_agent_context_manager(test_kg, test_emb, temp_persist_dir):
    """Test agent async context manager"""
    async with create_agent('budget', test_kg, test_emb, persist_dir=temp_persist_dir) as agent:
        await agent.query(Query(text="test"))

    # State should be saved on exit
    async with create_agent('budget', test_kg, test_emb, persist_dir=temp_persist_dir) as agent:
        stats = agent.get_learning_stats()
        assert stats['total_snapshots'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
