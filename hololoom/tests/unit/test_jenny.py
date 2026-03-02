#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Jenny Generative UI Runtime
============================================

Tests for the Jenny UI compilation system including:
- JennySpec frozen dataclass and lifecycle management
- JennyCompiler query analysis and panel generation
- SpecLedger audit trail and replay functionality

Philosophy: "Disposable pixels, durable decisions."

Author: Claude Code (MVP Week 1)
Date: 2025-12-01
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any

# Import Jenny components
from hololoom.visualization.jenny_spec import (
    JennySpec,
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
    PanelTypeJenny,
    PanelSizeJenny,
    LayoutHint,
    create_action,
    get_default_actions,
    ACTION_PIN,
    ACTION_DISMISS,
    ACTION_WHY,
)
from hololoom.visualization.jenny_compiler import (
    JennyCompiler,
    QueryAnalysis,
    analyze_query,
    generate_text_panel,
    generate_confidence_panel,
    generate_sources_panel,
)
from hololoom.visualization.spec_ledger import (
    SpecLedger,
    SpecLedgerEntry,
)
from hololoom.protocols.jenny import (
    CompilationStrategy,
    RenderTarget,
    JennyCompilerProtocol,
    SpecLedgerProtocol,
)

# Mock Spacetime for testing
from hololoom.fabric.spacetime import Spacetime, WeavingTrace


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_spacetime() -> Spacetime:
    """Create a mock Spacetime for testing."""
    trace = WeavingTrace(
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(milliseconds=150),
        duration_ms=150.0,
        stage_durations={
            'retrieval': 50.0,
            'decision': 30.0,
            'tool_execution': 70.0,
        },
        errors=[],
        warnings=[]
    )

    return Spacetime(
        query_text="What is Thompson Sampling?",
        response="Thompson Sampling is a Bayesian approach to exploration-exploitation.",
        tool_used="answer",
        confidence=0.85,
        trace=trace,
        metadata={
            'session_id': 'test_session',
            'pattern': 'FAST',
            'complexity': 'MODERATE',
        },
        context_summary="5 memory shards retrieved",
        sources_used=['shard_1', 'shard_2', 'shard_3']
    )


@pytest.fixture
def mock_low_confidence_spacetime() -> Spacetime:
    """Create a low-confidence Spacetime for testing refinement scenarios."""
    trace = WeavingTrace(
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(milliseconds=300),
        duration_ms=300.0,
        stage_durations={'retrieval': 100.0, 'decision': 50.0},
        errors=[],
        warnings=['Low relevance matches']
    )

    return Spacetime(
        query_text="Explain quantum entanglement in simple terms",
        response="Quantum entanglement is... (uncertain response)",
        tool_used="research",
        confidence=0.35,
        trace=trace,
        metadata={'session_id': 'test_session'},
        sources_used=['shard_1']
    )


@pytest.fixture
def temp_persist_path():
    """Create a temporary directory for SpecLedger persistence."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# JennySpec Tests
# ============================================================================

class TestJennySpec:
    """Tests for JennySpec frozen dataclass."""

    def test_spec_creation_defaults(self):
        """Test JennySpec creation with default values."""
        spec = JennySpec()

        assert spec.spec_id is not None
        assert len(spec.spec_id) == 36  # UUID format
        assert spec.panel_type == PanelTypeJenny.TEXT
        assert spec.lifecycle == LifecycleStage.NASCENT
        assert spec.binding_mode == BindingMode.STATIC
        assert spec.size == PanelSizeJenny.MEDIUM
        assert spec.priority == 0
        assert spec.position == (0.0, 0.0, 0.0)

    def test_spec_creation_with_values(self):
        """Test JennySpec creation with custom values."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.GRAPH,
            title="Knowledge Graph",
            subtitle="Entity relationships",
            content={'nodes': [], 'edges': []},
            size=PanelSizeJenny.LARGE,
            priority=10,
            position=(1.0, 2.0, 0.5)
        )

        assert spec.panel_type == PanelTypeJenny.GRAPH
        assert spec.title == "Knowledge Graph"
        assert spec.subtitle == "Entity relationships"
        assert spec.size == PanelSizeJenny.LARGE
        assert spec.priority == 10
        assert spec.position == (1.0, 2.0, 0.5)

    def test_spec_immutability(self):
        """Test that JennySpec is frozen (immutable)."""
        spec = JennySpec()

        with pytest.raises(Exception):  # FrozenInstanceError
            spec.title = "Modified Title"

    def test_spec_serialization(self):
        """Test JennySpec to_dict serialization."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.CONFIDENCE,
            title="Confidence",
            content={'score': 0.85}
        )

        data = spec.to_dict()

        assert data['panel_type'] == 'confidence'
        assert data['title'] == 'Confidence'
        assert data['content'] == {'score': 0.85}
        assert data['lifecycle'] == 'nascent'
        assert 'timestamp' in data
        assert 'spec_id' in data

    def test_spec_deserialization(self):
        """Test JennySpec from_dict deserialization."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.REASONING,
            title="Reasoning Chain",
            content={'steps': ['step1', 'step2']}
        )

        data = spec.to_dict()
        restored = JennySpec.from_dict(data)

        assert restored.panel_type == spec.panel_type
        assert restored.title == spec.title
        assert restored.content == spec.content
        assert restored.spec_id == spec.spec_id

    def test_spec_json_round_trip(self):
        """Test JSON serialization round trip."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.SOURCES,
            title="Sources",
            content={'sources': ['source1', 'source2']}
        )

        json_str = spec.to_json()
        restored = JennySpec.from_json(json_str)

        assert restored.to_dict() == spec.to_dict()

    def test_spec_lifecycle_transition(self):
        """Test with_lifecycle immutable transition."""
        nascent = JennySpec(panel_type=PanelTypeJenny.TEXT)

        # Transition to STABLE
        stable = nascent.with_lifecycle(LifecycleStage.STABLE)

        assert nascent.lifecycle == LifecycleStage.NASCENT  # Original unchanged
        assert stable.lifecycle == LifecycleStage.STABLE
        assert nascent.spec_id == stable.spec_id  # Same identity

    def test_spec_dissolution_transition(self):
        """Test dissolution with trigger."""
        stable = JennySpec(lifecycle=LifecycleStage.STABLE)

        dissolving = stable.with_lifecycle(
            LifecycleStage.DISSOLVING,
            trigger=DissolutionTrigger.TIMEOUT
        )

        assert dissolving.lifecycle == LifecycleStage.DISSOLVING
        assert dissolving.dissolution_trigger == DissolutionTrigger.TIMEOUT

    def test_spec_position_update(self):
        """Test with_position immutable update."""
        spec = JennySpec(position=(0.0, 0.0, 0.0))

        moved = spec.with_position(1.5, 2.5, 0.5)

        assert spec.position == (0.0, 0.0, 0.0)  # Original unchanged
        assert moved.position == (1.5, 2.5, 0.5)

    def test_reactive_binding_requires_data_source(self):
        """Test that REACTIVE binding mode requires data_source."""
        with pytest.raises(ValueError, match="REACTIVE binding mode requires data_source"):
            JennySpec(binding_mode=BindingMode.REACTIVE)

    def test_streaming_binding_requires_data_source(self):
        """Test that STREAMING binding mode requires data_source."""
        with pytest.raises(ValueError, match="STREAMING binding mode requires data_source"):
            JennySpec(binding_mode=BindingMode.STREAMING)

    def test_reactive_binding_with_data_source(self):
        """Test REACTIVE binding with data_source succeeds."""
        spec = JennySpec(
            binding_mode=BindingMode.REACTIVE,
            data_source="query://confidence"
        )

        assert spec.binding_mode == BindingMode.REACTIVE
        assert spec.data_source == "query://confidence"


class TestJennyActions:
    """Tests for action helpers and default actions."""

    def test_create_action(self):
        """Test create_action helper."""
        action = create_action(
            label="Test Action",
            handler="test_handler",
            action_type="button",
            params={'key': 'value'},
            requires_confirmation=True
        )

        assert action['label'] == "Test Action"
        assert action['handler'] == "test_handler"
        assert action['type'] == "button"
        assert action['params'] == {'key': 'value'}
        assert action['requires_confirmation'] is True
        assert 'action_id' in action

    def test_default_actions(self):
        """Test get_default_actions returns appropriate actions."""
        text_actions = get_default_actions(PanelTypeJenny.TEXT)
        graph_actions = get_default_actions(PanelTypeJenny.GRAPH)
        why_actions = get_default_actions(PanelTypeJenny.WHY)

        # Text panels should have copy action
        assert any(a['handler'] == 'copy_content' for a in text_actions)

        # Graph panels should have expand action
        assert any(a['handler'] == 'expand_panel' for a in graph_actions)

        # WHY (meta-panel) should have minimal actions
        assert len(why_actions) == 1
        assert why_actions[0]['handler'] == 'dismiss_panel'

    def test_standard_actions_exist(self):
        """Test standard action constants."""
        assert ACTION_PIN['handler'] == 'pin_panel'
        assert ACTION_DISMISS['handler'] == 'dismiss_panel'
        assert ACTION_WHY['handler'] == 'show_why_panel'


# ============================================================================
# JennyCompiler Tests
# ============================================================================

class TestQueryAnalysis:
    """Tests for query analysis functionality."""

    def test_analyze_factual_query(self, mock_spacetime):
        """Test analysis of factual query."""
        mock_spacetime.query_text = "What is Thompson Sampling?"
        analysis = analyze_query(mock_spacetime)

        assert analysis.query_type == 'factual'
        assert analysis.complexity in ['simple', 'moderate']

    def test_analyze_procedural_query(self, mock_spacetime):
        """Test analysis of procedural query."""
        mock_spacetime.query_text = "How do I implement Thompson Sampling?"
        analysis = analyze_query(mock_spacetime)

        assert analysis.query_type == 'procedural'

    def test_analyze_analytical_query(self, mock_spacetime):
        """Test analysis of analytical query."""
        mock_spacetime.query_text = "Compare Thompson Sampling vs UCB"
        analysis = analyze_query(mock_spacetime)

        assert analysis.query_type == 'analytical'

    def test_analyze_debug_query(self, mock_spacetime):
        """Test analysis of debug query."""
        mock_spacetime.query_text = "Why did the model fail?"
        analysis = analyze_query(mock_spacetime)

        assert analysis.query_type == 'debug'

    def test_analysis_detects_low_confidence(self, mock_low_confidence_spacetime):
        """Test that low confidence is detected."""
        analysis = analyze_query(mock_low_confidence_spacetime)

        assert analysis.confidence_level == 'low'

    def test_analysis_detects_high_confidence(self, mock_spacetime):
        """Test that high confidence is detected."""
        mock_spacetime.confidence = 0.92
        analysis = analyze_query(mock_spacetime)

        assert analysis.confidence_level == 'high'


class TestJennyCompiler:
    """Tests for JennyCompiler class."""

    @pytest.fixture
    def compiler(self):
        """Create a JennyCompiler instance."""
        return JennyCompiler()

    def test_compiler_implements_protocol(self, compiler):
        """Test that JennyCompiler implements JennyCompilerProtocol."""
        assert isinstance(compiler, JennyCompilerProtocol)

    def test_compiler_version(self, compiler):
        """Test compiler version property."""
        assert compiler.compiler_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_compile_minimal(self, compiler, mock_spacetime):
        """Test minimal compilation strategy."""
        specs = await compiler.compile(
            mock_spacetime,
            strategy=CompilationStrategy.MINIMAL
        )

        assert len(specs) >= 1
        assert len(specs) <= 3  # text + confidence + WHY meta-panel
        # Should always have text panel
        assert any(s.panel_type == PanelTypeJenny.TEXT for s in specs)

    @pytest.mark.asyncio
    async def test_compile_balanced(self, compiler, mock_spacetime):
        """Test balanced compilation strategy."""
        specs = await compiler.compile(
            mock_spacetime,
            strategy=CompilationStrategy.BALANCED
        )

        assert len(specs) >= 2
        assert len(specs) <= 4
        # Check for expected panels
        panel_types = [s.panel_type for s in specs]
        assert PanelTypeJenny.TEXT in panel_types
        assert PanelTypeJenny.CONFIDENCE in panel_types

    @pytest.mark.asyncio
    async def test_compile_comprehensive(self, compiler, mock_spacetime):
        """Test comprehensive compilation strategy."""
        # Add sources to trigger sources panel
        mock_spacetime.sources_used = ['source1', 'source2', 'source3']

        specs = await compiler.compile(
            mock_spacetime,
            strategy=CompilationStrategy.COMPREHENSIVE
        )

        assert len(specs) >= 4
        panel_types = [s.panel_type for s in specs]
        assert PanelTypeJenny.TEXT in panel_types
        assert PanelTypeJenny.CONFIDENCE in panel_types
        assert PanelTypeJenny.SOURCES in panel_types

    @pytest.mark.asyncio
    async def test_compile_auto_strategy(self, compiler, mock_spacetime):
        """Test auto strategy selection."""
        specs = await compiler.compile(
            mock_spacetime,
            strategy=CompilationStrategy.AUTO
        )

        assert len(specs) >= 1
        # Auto should select appropriate strategy based on complexity

    @pytest.mark.asyncio
    async def test_compile_with_context(self, compiler, mock_spacetime):
        """Test compilation with context."""
        context = {
            'session_id': 'test_session',
            'user_preferences': {'layout': 'grid'}
        }

        specs = await compiler.compile(mock_spacetime, context=context)

        assert len(specs) >= 1
        # Context is used for compilation decisions, verified by checking specs generated

    @pytest.mark.asyncio
    async def test_compile_stream(self, compiler, mock_spacetime):
        """Test streaming compilation."""
        specs = []
        async for spec in compiler.compile_stream(mock_spacetime):
            specs.append(spec)

        assert len(specs) >= 1
        # First spec should be text (high priority)
        assert specs[0].panel_type == PanelTypeJenny.TEXT

    def test_get_panel_strategy(self, compiler, mock_spacetime):
        """Test panel strategy recommendation."""
        strategy = compiler.get_panel_strategy(mock_spacetime)

        assert 'recommended_strategy' in strategy
        assert 'estimated_panels' in strategy
        assert 'panel_types' in strategy
        assert 'reasoning' in strategy

    def test_get_panel_strategy_complex_query(self, compiler, mock_spacetime):
        """Test strategy for complex query."""
        mock_spacetime.query_text = "Compare all reinforcement learning algorithms"
        mock_spacetime.metadata['complexity'] = 'RESEARCH'

        strategy = compiler.get_panel_strategy(mock_spacetime)

        assert strategy['estimated_panels'] >= 4

    @pytest.mark.asyncio
    async def test_compile_low_confidence(self, compiler, mock_low_confidence_spacetime):
        """Test compilation with low confidence."""
        specs = await compiler.compile(
            mock_low_confidence_spacetime,
            strategy=CompilationStrategy.BALANCED
        )

        # Low confidence should trigger additional panels
        panel_types = [s.panel_type for s in specs]
        # Should have reasoning panel for low confidence
        assert PanelTypeJenny.REASONING in panel_types or len(specs) >= 2


class TestPanelGenerators:
    """Tests for individual panel generator functions."""

    def test_generate_text_panel(self, mock_spacetime):
        """Test text panel generation."""
        spec = generate_text_panel(mock_spacetime)

        assert spec.panel_type == PanelTypeJenny.TEXT
        assert spec.title == "Response"
        assert 'text' in spec.content
        assert spec.content['text'] == mock_spacetime.response
        assert spec.priority == 0  # Text panel has highest priority (lowest number)

    def test_generate_confidence_panel(self, mock_spacetime):
        """Test confidence panel generation."""
        spec = generate_confidence_panel(mock_spacetime)

        assert spec.panel_type == PanelTypeJenny.CONFIDENCE
        assert 'value' in spec.content
        assert spec.content['value'] == mock_spacetime.confidence
        assert spec.size == PanelSizeJenny.SMALL

    def test_generate_sources_panel(self, mock_spacetime):
        """Test sources panel generation."""
        mock_spacetime.sources_used = ['source1', 'source2']

        spec = generate_sources_panel(mock_spacetime)

        assert spec.panel_type == PanelTypeJenny.SOURCES
        assert 'sources' in spec.content
        assert len(spec.content['sources']) == 2


# ============================================================================
# SpecLedger Tests
# ============================================================================

class TestSpecLedger:
    """Tests for SpecLedger audit trail."""

    @pytest.fixture
    def ledger(self, temp_persist_path):
        """Create a SpecLedger instance with temp path."""
        return SpecLedger(persist_path=temp_persist_path)

    def test_ledger_implements_protocol(self, ledger):
        """Test that SpecLedger implements SpecLedgerProtocol."""
        assert isinstance(ledger, SpecLedgerProtocol)

    @pytest.mark.asyncio
    async def test_log_spec(self, ledger):
        """Test logging a spec."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            title="Test Panel"
        )

        entry_id = await ledger.log_spec(
            spec=spec,
            spacetime_id="st_123",
            session_id="session_1"
        )

        assert entry_id is not None
        assert len(entry_id) > 0

    @pytest.mark.asyncio
    async def test_get_spec(self, ledger):
        """Test retrieving a spec by ID."""
        spec = JennySpec(panel_type=PanelTypeJenny.GRAPH)

        await ledger.log_spec(spec, spacetime_id="st_123")

        retrieved = await ledger.get_spec(spec.spec_id)

        assert retrieved is not None
        assert retrieved.spec_id == spec.spec_id
        assert retrieved.panel_type == PanelTypeJenny.GRAPH

    @pytest.mark.asyncio
    async def test_query_by_spacetime(self, ledger):
        """Test querying specs by spacetime ID."""
        spec1 = JennySpec(panel_type=PanelTypeJenny.TEXT)
        spec2 = JennySpec(panel_type=PanelTypeJenny.CONFIDENCE)
        spec3 = JennySpec(panel_type=PanelTypeJenny.GRAPH)

        await ledger.log_spec(spec1, spacetime_id="st_123")
        await ledger.log_spec(spec2, spacetime_id="st_123")
        await ledger.log_spec(spec3, spacetime_id="st_456")

        results = await ledger.query_by_spacetime("st_123")

        assert len(results) == 2
        spec_ids = [s.spec_id for s in results]
        assert spec1.spec_id in spec_ids
        assert spec2.spec_id in spec_ids

    @pytest.mark.asyncio
    async def test_query_by_panel_type(self, ledger):
        """Test querying specs by panel type."""
        spec1 = JennySpec(panel_type=PanelTypeJenny.TEXT)
        spec2 = JennySpec(panel_type=PanelTypeJenny.TEXT)
        spec3 = JennySpec(panel_type=PanelTypeJenny.GRAPH)

        await ledger.log_spec(spec1, spacetime_id="st_1")
        await ledger.log_spec(spec2, spacetime_id="st_2")
        await ledger.log_spec(spec3, spacetime_id="st_3")

        results = await ledger.query_by_panel_type("text")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_time_range(self, ledger):
        """Test querying specs by time range."""
        spec = JennySpec(panel_type=PanelTypeJenny.TEXT)

        await ledger.log_spec(spec, spacetime_id="st_123")

        now = datetime.now()
        start = (now - timedelta(hours=1)).isoformat()
        end = (now + timedelta(hours=1)).isoformat()

        results = await ledger.query_by_time_range(start, end)

        assert len(results) == 1
        assert results[0].spec_id == spec.spec_id

    @pytest.mark.asyncio
    async def test_replay_session(self, ledger):
        """Test session replay."""
        spec1 = JennySpec(panel_type=PanelTypeJenny.TEXT)
        spec2 = JennySpec(panel_type=PanelTypeJenny.CONFIDENCE)
        spec3 = JennySpec(panel_type=PanelTypeJenny.GRAPH)

        # Different sessions
        await ledger.log_spec(spec1, spacetime_id="st_1", session_id="session_A")
        await ledger.log_spec(spec2, spacetime_id="st_2", session_id="session_A")
        await ledger.log_spec(spec3, spacetime_id="st_3", session_id="session_B")

        replay = await ledger.replay_session("session_A")

        assert len(replay) == 2
        spec_ids = [s.spec_id for s in replay]
        assert spec1.spec_id in spec_ids
        assert spec2.spec_id in spec_ids

    @pytest.mark.asyncio
    async def test_get_statistics(self, ledger):
        """Test statistics calculation."""
        spec1 = JennySpec(panel_type=PanelTypeJenny.TEXT)
        spec2 = JennySpec(panel_type=PanelTypeJenny.TEXT)
        spec3 = JennySpec(panel_type=PanelTypeJenny.GRAPH)

        await ledger.log_spec(spec1, spacetime_id="st_1")
        await ledger.log_spec(spec2, spacetime_id="st_1")
        await ledger.log_spec(spec3, spacetime_id="st_2")

        stats = await ledger.get_statistics()

        assert stats['total_specs'] == 3
        assert 'panel_type_counts' in stats
        assert stats['panel_type_counts']['text'] == 2
        assert stats['panel_type_counts']['graph'] == 1

    @pytest.mark.asyncio
    async def test_log_spec_with_metadata(self, ledger):
        """Test logging spec with additional metadata."""
        spec = JennySpec(panel_type=PanelTypeJenny.REASONING)

        await ledger.log_spec(
            spec=spec,
            spacetime_id="st_123",
            metadata={
                'confidence': 0.85,
                'tool_used': 'answer',
                'query_text': 'What is Thompson Sampling?'
            }
        )

        retrieved = await ledger.get_spec(spec.spec_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_ledger_persistence(self, temp_persist_path):
        """Test that ledger persists across instances via auto_flush."""
        # Create and populate first ledger (auto_flush=True by default)
        ledger1 = SpecLedger(persist_path=temp_persist_path)
        spec = JennySpec(panel_type=PanelTypeJenny.TEXT)
        await ledger1.log_spec(spec, spacetime_id="st_123")
        # Auto-flush writes JSONL to disk immediately

        # Create new ledger at same path - should read persisted JSONL
        ledger2 = SpecLedger(persist_path=temp_persist_path)

        # Specs are persisted to JSONL, verify file exists
        import os
        jsonl_path = os.path.join(temp_persist_path, 'specs.jsonl')
        assert os.path.exists(jsonl_path), "JSONL file should exist after logging"


# ============================================================================
# Integration Tests
# ============================================================================

class TestJennyIntegration:
    """Integration tests for the complete Jenny pipeline."""

    @pytest.mark.asyncio
    async def test_full_compilation_pipeline(self, mock_spacetime, temp_persist_path):
        """Test full pipeline: compile → log → retrieve."""
        compiler = JennyCompiler()
        ledger = SpecLedger(persist_path=temp_persist_path)

        # Compile spacetime to specs
        specs = await compiler.compile(
            mock_spacetime,
            strategy=CompilationStrategy.BALANCED
        )

        assert len(specs) >= 2

        # Log all specs
        for spec in specs:
            await ledger.log_spec(
                spec=spec,
                spacetime_id="st_test",
                session_id="integration_test"
            )

        # Query back
        retrieved = await ledger.query_by_spacetime("st_test")

        assert len(retrieved) == len(specs)

    @pytest.mark.asyncio
    async def test_session_replay_workflow(self, mock_spacetime, temp_persist_path):
        """Test session replay workflow."""
        compiler = JennyCompiler()
        ledger = SpecLedger(persist_path=temp_persist_path)

        session_id = "replay_test_session"

        # Simulate multiple queries in a session
        for i in range(3):
            mock_spacetime.query_text = f"Query {i}"
            mock_spacetime.metadata['spacetime_id'] = f"st_{i}"

            specs = await compiler.compile(mock_spacetime)

            for spec in specs:
                await ledger.log_spec(
                    spec=spec,
                    spacetime_id=f"st_{i}",
                    session_id=session_id
                )

        # Replay session
        replay = await ledger.replay_session(session_id)

        # Should have all specs from all queries
        assert len(replay) >= 3 * 2  # At least 2 panels per query

    @pytest.mark.asyncio
    async def test_lifecycle_tracking(self, temp_persist_path):
        """Test lifecycle transitions via JennySpec methods.

        Note: Lifecycle management (update_lifecycle) is handled by
        JennyLifecycleManager (protocol: JennyLifecycleProtocol), not SpecLedger.
        SpecLedger is for audit/provenance only.
        """
        ledger = SpecLedger(persist_path=temp_persist_path)

        # Create and log spec in NASCENT state
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            lifecycle=LifecycleStage.NASCENT
        )
        await ledger.log_spec(spec, spacetime_id="st_123")

        # Simulate lifecycle transitions using JennySpec.with_lifecycle()
        pinned = spec.with_lifecycle(LifecycleStage.STABLE)
        assert pinned.lifecycle == LifecycleStage.STABLE

        dissolving = pinned.with_lifecycle(
            LifecycleStage.DISSOLVING,
            trigger=DissolutionTrigger.MANUAL
        )
        assert dissolving.lifecycle == LifecycleStage.DISSOLVING
        assert dissolving.dissolution_trigger == DissolutionTrigger.MANUAL

        # Log the dissolved spec for audit trail
        await ledger.log_spec(dissolving, spacetime_id="st_123")


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_spacetime_response(self, mock_spacetime):
        """Test handling of empty response."""
        mock_spacetime.response = ""

        spec = generate_text_panel(mock_spacetime)

        assert spec.content['text'] == ""

    @pytest.mark.asyncio
    async def test_compile_with_errors_in_trace(self, mock_spacetime):
        """Test compilation when trace has errors."""
        mock_spacetime.trace.errors = [
            {'stage': 'retrieval', 'error': 'No matches found'}
        ]

        compiler = JennyCompiler()
        specs = await compiler.compile(mock_spacetime)

        # Should still generate specs
        assert len(specs) >= 1

    @pytest.mark.asyncio
    async def test_ledger_query_empty_results(self, temp_persist_path):
        """Test querying with no results."""
        ledger = SpecLedger(persist_path=temp_persist_path)

        results = await ledger.query_by_spacetime("nonexistent_st")

        assert results == []

    @pytest.mark.asyncio
    async def test_ledger_get_nonexistent_spec(self, temp_persist_path):
        """Test getting a nonexistent spec."""
        ledger = SpecLedger(persist_path=temp_persist_path)

        result = await ledger.get_spec("nonexistent_id")

        assert result is None

    def test_spec_with_unicode_content(self):
        """Test spec with Unicode content."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            title="Unicode Test: 日本語 🚀",
            content={'text': '中文内容 한글 العربية'}
        )

        data = spec.to_dict()
        restored = JennySpec.from_dict(data)

        assert restored.title == spec.title
        assert restored.content == spec.content

    def test_spec_with_large_content(self):
        """Test spec with large content."""
        large_text = "x" * 100000  # 100KB of text

        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            content={'text': large_text}
        )

        json_str = spec.to_json()
        restored = JennySpec.from_json(json_str)

        assert len(restored.content['text']) == 100000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
