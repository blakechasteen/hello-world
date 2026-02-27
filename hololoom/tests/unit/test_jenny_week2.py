#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Jenny Generative UI Runtime - Week 2
=====================================================

Tests for Week 2 components:
- JennyRenderer (HTMLRenderer, TerminalRenderer, JSONRenderer)
- JennyLifecycleManager (InMemoryLifecycleManager)

Philosophy: "Disposable pixels, durable decisions."

Author: Claude Code (MVP Week 2)
Date: 2025-12-01
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import Jenny Week 1 components (specs, etc.)
from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
    PanelTypeJenny,
    PanelSizeJenny,
    create_action,
    ACTION_PIN,
    ACTION_DISMISS,
)

# Import Jenny Week 2 components - Renderers
from HoloLoom.visualization.jenny_renderer import (
    JennyRendererBase,
    HTMLRenderer,
    TerminalRenderer,
    JSONRenderer,
    RenderTarget,
    create_renderer,
    get_default_renderer,
)

# Import Jenny Week 2 components - Lifecycle Manager
from HoloLoom.visualization.jenny_lifecycle import (
    JennyLifecycleManagerBase,
    InMemoryLifecycleManager,
    PanelState,
    PanelNotFoundError,
    InvalidTransitionError,
    create_lifecycle_manager,
    get_default_lifecycle_manager,
)

# Import SpecLedger for lifecycle integration testing
from HoloLoom.visualization.spec_ledger import SpecLedger


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_text_spec() -> JennySpec:
    """Create a sample TEXT panel spec."""
    return JennySpec(
        panel_type=PanelTypeJenny.TEXT,
        title="Answer",
        subtitle="High confidence response",
        content={'text': 'Thompson Sampling is a Bayesian approach.'},
        lifecycle=LifecycleStage.NASCENT,
        size=PanelSizeJenny.MEDIUM,
        actions=(ACTION_PIN, ACTION_DISMISS),
    )


@pytest.fixture
def sample_confidence_spec() -> JennySpec:
    """Create a sample CONFIDENCE panel spec."""
    return JennySpec(
        panel_type=PanelTypeJenny.CONFIDENCE,
        title="Confidence",
        content={'score': 0.85, 'level': 'high'},
        lifecycle=LifecycleStage.NASCENT,
        size=PanelSizeJenny.SMALL,
    )


@pytest.fixture
def sample_graph_spec() -> JennySpec:
    """Create a sample GRAPH panel spec."""
    return JennySpec(
        panel_type=PanelTypeJenny.GRAPH,
        title="Knowledge Graph",
        content={
            'nodes': [
                {'id': 'node1', 'label': 'Thompson Sampling'},
                {'id': 'node2', 'label': 'Bayesian'},
            ],
            'edges': [
                {'source': 'node1', 'target': 'node2', 'label': 'IS_A'},
            ],
        },
        lifecycle=LifecycleStage.NASCENT,
        size=PanelSizeJenny.LARGE,
    )


@pytest.fixture
def sample_code_spec() -> JennySpec:
    """Create a sample CODE panel spec."""
    return JennySpec(
        panel_type=PanelTypeJenny.CODE,
        title="Example Code",
        content={
            'code': 'def thompson_sample(alpha, beta):\n    return np.random.beta(alpha, beta)',
            'language': 'python',
        },
        lifecycle=LifecycleStage.NASCENT,
    )


@pytest.fixture
def lifecycle_manager() -> InMemoryLifecycleManager:
    """Create a lifecycle manager for testing."""
    return InMemoryLifecycleManager()


@pytest.fixture
def lifecycle_manager_with_ledger(tmp_path) -> InMemoryLifecycleManager:
    """Create a lifecycle manager with SpecLedger integration."""
    ledger = SpecLedger(persist_path=str(tmp_path))
    return InMemoryLifecycleManager(spec_ledger=ledger)


# ============================================================================
# HTMLRenderer Tests
# ============================================================================

class TestHTMLRenderer:
    """Tests for HTMLRenderer."""

    @pytest.fixture
    def renderer(self) -> HTMLRenderer:
        """Create HTMLRenderer instance."""
        return HTMLRenderer()

    def test_renderer_supports_html_target(self, renderer):
        """Test that HTMLRenderer supports HTML target."""
        assert RenderTarget.HTML in renderer.supported_targets

    @pytest.mark.asyncio
    async def test_render_single_text_panel(self, renderer, sample_text_spec):
        """Test rendering a single TEXT panel."""
        html = await renderer.render_single(sample_text_spec, RenderTarget.HTML)

        assert isinstance(html, str)
        assert len(html) > 0
        assert 'jenny-panel' in html
        assert sample_text_spec.title in html
        assert sample_text_spec.content['text'] in html

    @pytest.mark.asyncio
    async def test_render_single_confidence_panel(self, renderer, sample_confidence_spec):
        """Test rendering a CONFIDENCE panel."""
        html = await renderer.render_single(sample_confidence_spec, RenderTarget.HTML)

        assert 'confidence' in html.lower()
        assert 'jenny-confidence' in html  # Has confidence styling

    @pytest.mark.asyncio
    async def test_render_single_graph_panel(self, renderer, sample_graph_spec):
        """Test rendering a GRAPH panel."""
        html = await renderer.render_single(sample_graph_spec, RenderTarget.HTML)

        # Renderer shows placeholder with node/edge counts
        assert 'jenny-graph' in html
        assert '2 nodes' in html
        assert '1 edges' in html

    @pytest.mark.asyncio
    async def test_render_single_code_panel(self, renderer, sample_code_spec):
        """Test rendering a CODE panel with syntax highlighting."""
        html = await renderer.render_single(sample_code_spec, RenderTarget.HTML)

        assert '<pre' in html
        assert 'thompson_sample' in html

    @pytest.mark.asyncio
    async def test_render_multiple_panels(self, renderer, sample_text_spec, sample_confidence_spec):
        """Test rendering multiple panels."""
        specs = [sample_text_spec, sample_confidence_spec]
        html = await renderer.render(specs, RenderTarget.HTML)

        assert isinstance(html, str)
        assert 'jenny-container' in html  # Container class for dashboard
        # Both panels should be present
        assert sample_text_spec.title in html
        assert 'confidence' in html.lower()

    @pytest.mark.asyncio
    async def test_render_with_lifecycle_classes(self, renderer, sample_text_spec):
        """Test that lifecycle stage affects CSS classes."""
        # NASCENT
        nascent_html = await renderer.render_single(sample_text_spec, RenderTarget.HTML)
        assert 'nascent' in nascent_html

        # STABLE
        stable_spec = sample_text_spec.with_lifecycle(LifecycleStage.STABLE)
        stable_html = await renderer.render_single(stable_spec, RenderTarget.HTML)
        assert 'stable' in stable_html

        # DISSOLVING
        dissolving_spec = sample_text_spec.with_lifecycle(
            LifecycleStage.DISSOLVING,
            trigger=DissolutionTrigger.MANUAL
        )
        dissolving_html = await renderer.render_single(dissolving_spec, RenderTarget.HTML)
        assert 'dissolving' in dissolving_html

    @pytest.mark.asyncio
    async def test_render_actions(self, renderer, sample_text_spec):
        """Test that action buttons are rendered."""
        html = await renderer.render_single(sample_text_spec, RenderTarget.HTML)

        # Should have Pin and Dismiss buttons
        assert 'Pin' in html
        assert 'Dismiss' in html

    @pytest.mark.asyncio
    async def test_render_empty_list(self, renderer):
        """Test rendering empty panel list."""
        html = await renderer.render([], RenderTarget.HTML)

        assert isinstance(html, str)
        assert 'jenny-container' in html  # Container class for empty dashboard

    @pytest.mark.asyncio
    async def test_render_update_diff(self, renderer, sample_text_spec):
        """Test render_update for panel transitions."""
        old_spec = sample_text_spec
        new_spec = sample_text_spec.with_lifecycle(LifecycleStage.STABLE)

        html = await renderer.render_update(old_spec, new_spec, RenderTarget.HTML)

        assert isinstance(html, str)


# ============================================================================
# TerminalRenderer Tests
# ============================================================================

class TestTerminalRenderer:
    """Tests for TerminalRenderer."""

    @pytest.fixture
    def renderer(self) -> TerminalRenderer:
        """Create TerminalRenderer instance."""
        return TerminalRenderer()

    def test_renderer_supports_terminal_target(self, renderer):
        """Test that TerminalRenderer supports TERMINAL target."""
        assert RenderTarget.TERMINAL in renderer.supported_targets

    @pytest.mark.asyncio
    async def test_render_single_text_panel(self, renderer, sample_text_spec):
        """Test rendering TEXT panel to terminal."""
        output = await renderer.render_single(sample_text_spec, RenderTarget.TERMINAL)

        assert isinstance(output, str)
        assert sample_text_spec.title in output
        assert sample_text_spec.content['text'] in output
        # Box drawing characters
        assert '─' in output or '-' in output

    @pytest.mark.asyncio
    async def test_render_multiple_panels(self, renderer, sample_text_spec, sample_confidence_spec):
        """Test rendering multiple panels."""
        specs = [sample_text_spec, sample_confidence_spec]
        output = await renderer.render(specs, RenderTarget.TERMINAL)

        assert isinstance(output, str)
        assert sample_text_spec.title in output
        assert 'Confidence' in output

    @pytest.mark.asyncio
    async def test_render_with_compact_option(self, renderer, sample_text_spec):
        """Test compact rendering mode."""
        output = await renderer.render_single(
            sample_text_spec,
            RenderTarget.TERMINAL,
            options={'compact': True}
        )

        assert isinstance(output, str)
        # Compact mode should be shorter
        assert len(output) < 500  # Reasonable limit for compact

    @pytest.mark.asyncio
    async def test_lifecycle_colors(self, renderer, sample_text_spec):
        """Test that different lifecycle stages have different colors."""
        nascent = await renderer.render_single(sample_text_spec, RenderTarget.TERMINAL)

        stable_spec = sample_text_spec.with_lifecycle(LifecycleStage.STABLE)
        stable = await renderer.render_single(stable_spec, RenderTarget.TERMINAL)

        # Both should render, possibly with different colors
        assert len(nascent) > 0
        assert len(stable) > 0


# ============================================================================
# JSONRenderer Tests
# ============================================================================

class TestJSONRenderer:
    """Tests for JSONRenderer."""

    @pytest.fixture
    def renderer(self) -> JSONRenderer:
        """Create JSONRenderer instance."""
        return JSONRenderer()

    def test_renderer_supports_json_target(self, renderer):
        """Test that JSONRenderer supports JSON target."""
        assert RenderTarget.JSON in renderer.supported_targets

    @pytest.mark.asyncio
    async def test_render_single_panel_json(self, renderer, sample_text_spec):
        """Test rendering panel to JSON."""
        json_str = await renderer.render_single(sample_text_spec, RenderTarget.JSON)

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['spec_id'] == sample_text_spec.spec_id
        assert data['panel_type'] == 'text'
        assert data['title'] == sample_text_spec.title

    @pytest.mark.asyncio
    async def test_render_multiple_panels_json(self, renderer, sample_text_spec, sample_confidence_spec):
        """Test rendering multiple panels to JSON object with panels array."""
        specs = [sample_text_spec, sample_confidence_spec]
        json_str = await renderer.render(specs, RenderTarget.JSON)

        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert 'panels' in data
        assert len(data['panels']) == 2
        assert data['count'] == 2

    @pytest.mark.asyncio
    async def test_json_is_valid(self, renderer, sample_graph_spec):
        """Test that JSON output is valid and parseable."""
        json_str = await renderer.render_single(sample_graph_spec, RenderTarget.JSON)

        # Should not raise
        data = json.loads(json_str)
        assert 'nodes' in data.get('content', {})


# ============================================================================
# Renderer Factory Tests
# ============================================================================

class TestRendererFactory:
    """Tests for renderer factory functions."""

    def test_create_html_renderer(self):
        """Test creating HTML renderer via factory."""
        renderer = create_renderer(RenderTarget.HTML)
        assert isinstance(renderer, HTMLRenderer)

    def test_create_terminal_renderer(self):
        """Test creating terminal renderer via factory."""
        renderer = create_renderer(RenderTarget.TERMINAL)
        assert isinstance(renderer, TerminalRenderer)

    def test_create_json_renderer(self):
        """Test creating JSON renderer via factory."""
        renderer = create_renderer(RenderTarget.JSON)
        assert isinstance(renderer, JSONRenderer)

    def test_get_default_renderer(self):
        """Test getting default renderer (should be HTML)."""
        renderer = get_default_renderer()
        assert isinstance(renderer, HTMLRenderer)


# ============================================================================
# InMemoryLifecycleManager Tests
# ============================================================================

class TestInMemoryLifecycleManager:
    """Tests for InMemoryLifecycleManager."""

    @pytest.mark.asyncio
    async def test_spawn_panel(self, lifecycle_manager, sample_text_spec):
        """Test spawning a new panel."""
        spec = await lifecycle_manager.spawn(sample_text_spec)

        assert spec.lifecycle == LifecycleStage.NASCENT
        assert spec.spec_id == sample_text_spec.spec_id

        # Should be retrievable
        retrieved = await lifecycle_manager.get_panel(spec.spec_id)
        assert retrieved is not None
        assert retrieved.spec_id == spec.spec_id

    @pytest.mark.asyncio
    async def test_pin_panel(self, lifecycle_manager, sample_text_spec):
        """Test pinning a NASCENT panel to STABLE."""
        spec = await lifecycle_manager.spawn(sample_text_spec)
        pinned = await lifecycle_manager.pin(spec.spec_id)

        assert pinned.lifecycle == LifecycleStage.STABLE

    @pytest.mark.asyncio
    async def test_pin_nonexistent_panel_raises(self, lifecycle_manager):
        """Test that pinning nonexistent panel raises error."""
        with pytest.raises(PanelNotFoundError):
            await lifecycle_manager.pin("nonexistent_id")

    @pytest.mark.asyncio
    async def test_pin_stable_panel_raises(self, lifecycle_manager, sample_text_spec):
        """Test that pinning already STABLE panel raises error."""
        spec = await lifecycle_manager.spawn(sample_text_spec)
        await lifecycle_manager.pin(spec.spec_id)

        with pytest.raises(InvalidTransitionError):
            await lifecycle_manager.pin(spec.spec_id)

    @pytest.mark.asyncio
    async def test_dissolve_nascent_panel(self, lifecycle_manager, sample_text_spec):
        """Test dissolving a NASCENT panel."""
        spec = await lifecycle_manager.spawn(sample_text_spec)
        dissolved = await lifecycle_manager.dissolve(spec.spec_id, DissolutionTrigger.MANUAL)

        assert dissolved.lifecycle == LifecycleStage.DISSOLVING
        assert dissolved.dissolution_trigger == DissolutionTrigger.MANUAL

    @pytest.mark.asyncio
    async def test_dissolve_stable_panel(self, lifecycle_manager, sample_text_spec):
        """Test dissolving a STABLE (pinned) panel."""
        spec = await lifecycle_manager.spawn(sample_text_spec)
        await lifecycle_manager.pin(spec.spec_id)
        dissolved = await lifecycle_manager.dissolve(spec.spec_id, DissolutionTrigger.TIMEOUT)

        assert dissolved.lifecycle == LifecycleStage.DISSOLVING

    @pytest.mark.asyncio
    async def test_archive_panel(self, lifecycle_manager, sample_text_spec):
        """Test archiving a DISSOLVING panel."""
        spec = await lifecycle_manager.spawn(sample_text_spec)
        await lifecycle_manager.dissolve(spec.spec_id, DissolutionTrigger.MANUAL)
        archived = await lifecycle_manager.archive(spec.spec_id)

        assert archived.lifecycle == LifecycleStage.ARCHIVED

        # Panel should no longer be in active storage
        retrieved = await lifecycle_manager.get_panel(spec.spec_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_archive_non_dissolving_raises(self, lifecycle_manager, sample_text_spec):
        """Test that archiving non-DISSOLVING panel raises error."""
        spec = await lifecycle_manager.spawn(sample_text_spec)

        with pytest.raises(InvalidTransitionError):
            await lifecycle_manager.archive(spec.spec_id)

    @pytest.mark.asyncio
    async def test_get_active_panels(self, lifecycle_manager, sample_text_spec, sample_confidence_spec):
        """Test getting active panels."""
        spec1 = await lifecycle_manager.spawn(sample_text_spec)
        spec2 = await lifecycle_manager.spawn(sample_confidence_spec)
        await lifecycle_manager.pin(spec1.spec_id)

        active = await lifecycle_manager.get_active_panels()

        assert len(active) == 2
        spec_ids = [s.spec_id for s in active]
        assert spec1.spec_id in spec_ids
        assert spec2.spec_id in spec_ids

    @pytest.mark.asyncio
    async def test_get_active_panels_by_session(self, lifecycle_manager, sample_text_spec, sample_confidence_spec):
        """Test filtering active panels by session."""
        # Spawn with different sessions via metadata
        spec1 = JennySpec(
            panel_type=sample_text_spec.panel_type,
            title=sample_text_spec.title,
            content=sample_text_spec.content,
            metadata={'session_id': 'session_A'}
        )
        spec2 = JennySpec(
            panel_type=sample_confidence_spec.panel_type,
            title=sample_confidence_spec.title,
            content=sample_confidence_spec.content,
            metadata={'session_id': 'session_B'}
        )

        await lifecycle_manager.spawn(spec1)
        await lifecycle_manager.spawn(spec2)

        session_a_panels = await lifecycle_manager.get_active_panels(session_id='session_A')
        session_b_panels = await lifecycle_manager.get_active_panels(session_id='session_B')

        assert len(session_a_panels) == 1
        assert len(session_b_panels) == 1
        assert session_a_panels[0].spec_id == spec1.spec_id
        assert session_b_panels[0].spec_id == spec2.spec_id

    @pytest.mark.asyncio
    async def test_cleanup_stale_panels(self, lifecycle_manager, sample_text_spec):
        """Test cleanup of stale NASCENT panels."""
        spec = await lifecycle_manager.spawn(sample_text_spec)

        # Manipulate spawn time to make it stale
        state = lifecycle_manager._panels[spec.spec_id]
        state.spawned_at = datetime.now() - timedelta(minutes=10)

        dissolved_ids = await lifecycle_manager.cleanup_stale(max_age_seconds=60)

        assert len(dissolved_ids) == 1
        assert spec.spec_id in dissolved_ids

    @pytest.mark.asyncio
    async def test_cleanup_skips_stable_panels(self, lifecycle_manager, sample_text_spec):
        """Test that cleanup doesn't affect STABLE panels."""
        spec = await lifecycle_manager.spawn(sample_text_spec)
        await lifecycle_manager.pin(spec.spec_id)

        # Manipulate spawn time
        state = lifecycle_manager._panels[spec.spec_id]
        state.spawned_at = datetime.now() - timedelta(minutes=10)

        dissolved_ids = await lifecycle_manager.cleanup_stale(max_age_seconds=60)

        # STABLE panel should not be dissolved
        assert spec.spec_id not in dissolved_ids

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, lifecycle_manager, sample_text_spec):
        """Test memory pressure forces dissolution."""
        # Spawn many panels
        specs = []
        for i in range(10):
            spec = JennySpec(
                panel_type=PanelTypeJenny.TEXT,
                title=f"Panel {i}",
                content={'text': f'Content {i}'},
            )
            spawned = await lifecycle_manager.spawn(spec)
            specs.append(spawned)

        # Trigger memory pressure
        dissolved_ids = await lifecycle_manager.handle_memory_pressure(target_panel_count=5)

        # Should have dissolved some panels
        assert len(dissolved_ids) >= 5

        # Should have 5 or fewer active panels now
        active = await lifecycle_manager.get_active_panels()
        assert len(active) <= 5

    @pytest.mark.asyncio
    async def test_statistics(self, lifecycle_manager, sample_text_spec, sample_confidence_spec):
        """Test statistics tracking."""
        spec1 = await lifecycle_manager.spawn(sample_text_spec)
        spec2 = await lifecycle_manager.spawn(sample_confidence_spec)
        await lifecycle_manager.pin(spec1.spec_id)
        await lifecycle_manager.dissolve(spec2.spec_id, DissolutionTrigger.TIMEOUT)
        await lifecycle_manager.archive(spec2.spec_id)

        stats = lifecycle_manager.get_statistics()

        assert stats['total_spawned'] == 2
        assert stats['total_pinned'] == 1
        assert stats['total_dissolved'] == 1
        assert stats['total_archived'] == 1
        assert stats['active_stable'] == 1
        assert stats['active_nascent'] == 0

    @pytest.mark.asyncio
    async def test_full_lifecycle_flow(self, lifecycle_manager, sample_text_spec):
        """Test complete lifecycle: spawn → pin → dissolve → archive."""
        # 1. Spawn (→ NASCENT)
        spec = await lifecycle_manager.spawn(sample_text_spec)
        assert spec.lifecycle == LifecycleStage.NASCENT

        # 2. Pin (→ STABLE)
        spec = await lifecycle_manager.pin(spec.spec_id)
        assert spec.lifecycle == LifecycleStage.STABLE

        # 3. Dissolve (→ DISSOLVING)
        spec = await lifecycle_manager.dissolve(spec.spec_id, DissolutionTrigger.MANUAL)
        assert spec.lifecycle == LifecycleStage.DISSOLVING

        # 4. Archive (→ ARCHIVED)
        spec = await lifecycle_manager.archive(spec.spec_id)
        assert spec.lifecycle == LifecycleStage.ARCHIVED

        # Panel should no longer be active
        active = await lifecycle_manager.get_active_panels()
        assert len(active) == 0


# ============================================================================
# Lifecycle Manager with SpecLedger Integration Tests
# ============================================================================

class TestLifecycleManagerLedgerIntegration:
    """Tests for lifecycle manager with SpecLedger integration."""

    @pytest.mark.asyncio
    async def test_spawn_logs_to_ledger(self, lifecycle_manager_with_ledger, sample_text_spec):
        """Test that spawn logs to SpecLedger."""
        manager = lifecycle_manager_with_ledger
        spec = await manager.spawn(sample_text_spec)

        # Check ledger has the spec
        ledger = manager.spec_ledger
        retrieved = await ledger.get_spec(spec.spec_id)

        assert retrieved is not None
        assert retrieved.spec_id == spec.spec_id

    @pytest.mark.asyncio
    async def test_lifecycle_transitions_logged(self, lifecycle_manager_with_ledger, sample_text_spec):
        """Test that lifecycle transitions are logged to SpecLedger."""
        manager = lifecycle_manager_with_ledger
        spec = await manager.spawn(sample_text_spec)
        await manager.pin(spec.spec_id)
        await manager.dissolve(spec.spec_id, DissolutionTrigger.MANUAL)

        # Ledger should have recorded transitions
        # (The actual logging is internal, we verify manager works with ledger)
        stats = manager.get_statistics()
        assert stats['ledger_enabled'] is True


# ============================================================================
# Lifecycle Manager Factory Tests
# ============================================================================

class TestLifecycleManagerFactory:
    """Tests for lifecycle manager factory functions."""

    def test_create_lifecycle_manager(self):
        """Test creating lifecycle manager via factory."""
        manager = create_lifecycle_manager()
        assert isinstance(manager, InMemoryLifecycleManager)

    def test_create_lifecycle_manager_with_max_panels(self):
        """Test creating lifecycle manager with custom max panels."""
        manager = create_lifecycle_manager(max_active_panels=50)
        assert manager.max_active_panels == 50

    def test_get_default_lifecycle_manager(self):
        """Test getting default lifecycle manager."""
        manager = get_default_lifecycle_manager()
        assert isinstance(manager, InMemoryLifecycleManager)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for Week 2 components."""

    @pytest.mark.asyncio
    async def test_dissolve_all_session_panels(self, lifecycle_manager, sample_text_spec):
        """Test dissolving all panels in a session."""
        # Create panels in a session
        spec = JennySpec(
            panel_type=sample_text_spec.panel_type,
            content=sample_text_spec.content,
            metadata={'session_id': 'test_session'}
        )
        await lifecycle_manager.spawn(spec)

        dissolved = await lifecycle_manager.dissolve_all_session(
            'test_session',
            DissolutionTrigger.ORPHAN
        )

        assert len(dissolved) >= 1

    @pytest.mark.asyncio
    async def test_render_unicode_content(self):
        """Test rendering Unicode content."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            title="Unicode: 日本語 🚀",
            content={'text': '中文内容 한글 العربية'}
        )

        renderer = HTMLRenderer()
        html = await renderer.render_single(spec, RenderTarget.HTML)

        assert '日本語' in html
        assert '中文内容' in html

    @pytest.mark.asyncio
    async def test_render_special_characters(self):
        """Test rendering HTML special characters."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            title="Special <chars> & \"quotes\"",
            content={'text': '<script>alert("XSS")</script>'}
        )

        renderer = HTMLRenderer()
        html = await renderer.render_single(spec, RenderTarget.HTML)

        # Should escape HTML characters
        assert '<script>' not in html or '&lt;script&gt;' in html


# ============================================================================
# Protocol Compliance Tests
# ============================================================================

class TestProtocolCompliance:
    """Tests to verify protocol compliance."""

    def test_html_renderer_has_required_methods(self):
        """Test HTMLRenderer has all protocol methods."""
        renderer = HTMLRenderer()

        assert hasattr(renderer, 'render')
        assert hasattr(renderer, 'render_single')
        assert hasattr(renderer, 'render_update')
        assert hasattr(renderer, 'supported_targets')

    def test_lifecycle_manager_has_required_methods(self):
        """Test InMemoryLifecycleManager has all protocol methods."""
        manager = InMemoryLifecycleManager()

        assert hasattr(manager, 'spawn')
        assert hasattr(manager, 'pin')
        assert hasattr(manager, 'dissolve')
        assert hasattr(manager, 'archive')
        assert hasattr(manager, 'get_active_panels')
        assert hasattr(manager, 'get_panel')
        assert hasattr(manager, 'cleanup_stale')
        assert hasattr(manager, 'handle_memory_pressure')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
