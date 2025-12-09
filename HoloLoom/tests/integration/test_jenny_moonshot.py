"""
Integration Tests: Jenny Moonshot Complete System
==================================================

Tests for all 6 Moonshot phases working together:
- M1: Renderer Registry (jenny_renderer_registry.py)
- M2: Async LLM Client (jenny_llm_client.py)
- M3: WCAG 2.1 AA Accessibility (jenny_accessibility.py)
- M4: React Props Rendering (jenny_renderer.py - ReactRenderer)
- M5: WebXR AR Rendering (jenny_renderer.py - ARRenderer)
- M6: Analytics Collection (jenny_analytics.py)

Run with:
    pytest HoloLoom/tests/integration/test_jenny_moonshot.py -v

Date: December 2025
Author: HoloLoom Team
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

# Jenny core imports
from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
    create_spec,
)

from HoloLoom.protocols.jenny import RenderTarget


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_spacetime():
    """Mock Spacetime with realistic data for moonshot testing."""
    spacetime = MagicMock()
    spacetime.query_text = "Explain Thompson Sampling and its applications in AI"
    spacetime.response = """Thompson Sampling is a Bayesian approach to the
    exploration-exploitation tradeoff. It balances exploring new options with
    exploiting known good options by sampling from posterior distributions."""
    spacetime.confidence = 0.85
    spacetime.trace = MagicMock()
    spacetime.trace.threads_activated = ["memory", "reasoning", "decision"]
    spacetime.trace.stage_durations = {"retrieve": 45, "decision": 30, "generate": 75}
    spacetime.trace.duration_ms = 150
    spacetime.metadata = {"cache_hit": False, "complexity": "moderate"}
    return spacetime


@pytest.fixture
def sample_specs() -> List[JennySpec]:
    """Generate sample JennySpecs for testing."""
    return [
        create_spec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.TEXT,
            title="Response",
            content={"text": "Thompson Sampling is a Bayesian approach..."},
            size=PanelSizeJenny.LARGE,
            priority=1,
        ),
        create_spec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.CONFIDENCE,
            title="Confidence",
            content={"value": 0.85},
            size=PanelSizeJenny.SMALL,
            priority=2,
        ),
        create_spec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.SOURCES,
            title="Sources",
            content={"sources": [{"title": "Wikipedia", "url": "#"}]},
            size=PanelSizeJenny.MEDIUM,
            priority=3,
        ),
    ]


@pytest.fixture
def fresh_registry():
    """Fresh renderer registry for test isolation."""
    from HoloLoom.visualization.jenny_renderer_registry import RendererRegistry
    registry = RendererRegistry()
    registry.clear()
    return registry


@pytest.fixture
def analytics_collector():
    """Fresh analytics collector for test isolation."""
    from HoloLoom.visualization.jenny_analytics import JennyAnalyticsCollector
    collector = JennyAnalyticsCollector()
    collector.clear()
    return collector


# ============================================================================
# M1: Renderer Registry Tests
# ============================================================================

class TestRendererRegistryIntegration:
    """M1: Renderer Registry integration tests."""

    def test_registry_singleton_pattern(self):
        """Registry should be a singleton."""
        from HoloLoom.visualization.jenny_renderer_registry import RendererRegistry

        registry1 = RendererRegistry()
        registry2 = RendererRegistry()

        assert registry1 is registry2, "Registry should be singleton"

    def test_register_all_renderers(self, fresh_registry):
        """All 5 renderers should be registerable."""
        from HoloLoom.visualization.jenny_renderer import (
            HTMLRenderer,
            TerminalRenderer,
            JSONRenderer,
            ReactRenderer,
            ARRenderer,
        )

        # Register all renderers
        fresh_registry.register(HTMLRenderer, priority=10)
        fresh_registry.register(TerminalRenderer, priority=5)
        fresh_registry.register(JSONRenderer, priority=3)
        fresh_registry.register(ReactRenderer, priority=8)
        fresh_registry.register(ARRenderer, priority=7)

        renderers = fresh_registry.list_renderers()
        assert len(renderers) == 5

        # Verify each is registered
        names = [r['name'] for r in renderers]
        assert 'html' in names
        assert 'terminal' in names
        assert 'json' in names
        assert 'react' in names
        assert 'ar' in names

    def test_priority_based_selection(self, fresh_registry):
        """Higher priority renderer should be selected."""
        from HoloLoom.visualization.jenny_renderer import HTMLRenderer

        # Register with specific priority
        fresh_registry.register(HTMLRenderer, priority=100)

        renderer = fresh_registry.get_for_target(RenderTarget.HTML)
        assert renderer is not None
        assert renderer.name == "html"

    def test_get_renderer_for_each_target(self, fresh_registry):
        """Should find renderer for each target type."""
        from HoloLoom.visualization.jenny_renderer import (
            HTMLRenderer,
            JSONRenderer,
            ReactRenderer,
            ARRenderer,
        )

        fresh_registry.register(HTMLRenderer, priority=10)
        fresh_registry.register(JSONRenderer, priority=5)
        fresh_registry.register(ReactRenderer, priority=8)
        fresh_registry.register(ARRenderer, priority=7)

        assert fresh_registry.get_for_target(RenderTarget.HTML) is not None
        assert fresh_registry.get_for_target(RenderTarget.JSON) is not None
        assert fresh_registry.get_for_target(RenderTarget.REACT) is not None
        assert fresh_registry.get_for_target(RenderTarget.AR) is not None

    @pytest.mark.asyncio
    async def test_concurrent_rendering(self, fresh_registry, sample_specs):
        """Should render to multiple targets concurrently."""
        from HoloLoom.visualization.jenny_renderer import (
            HTMLRenderer,
            JSONRenderer,
        )

        fresh_registry.register(HTMLRenderer, priority=10)
        fresh_registry.register(JSONRenderer, priority=5)

        results = await fresh_registry.render_concurrent(
            sample_specs,
            targets=[RenderTarget.HTML, RenderTarget.JSON]
        )

        assert RenderTarget.HTML in results
        assert RenderTarget.JSON in results
        assert len(results[RenderTarget.HTML]) > 0
        assert len(results[RenderTarget.JSON]) > 0


# ============================================================================
# M2: Async LLM Client Tests
# ============================================================================

class TestAsyncLLMClientIntegration:
    """M2: Async LLM Client integration tests."""

    def test_config_creation(self):
        """Should create LLM client config."""
        from HoloLoom.visualization.jenny_llm_client import LLMClientConfig

        config = LLMClientConfig.fast()
        assert config.timeout_seconds == 5.0
        assert config.max_retries == 1

        config = LLMClientConfig.reliable()
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3

    def test_response_dataclass(self):
        """LLMResponse should hold response data."""
        from HoloLoom.visualization.jenny_llm_client import LLMResponse

        response = LLMResponse(
            content="Test response",
            model="llama3.2:3b",
            provider="ollama",
            latency_ms=150.5,
            tokens_used=42,
        )

        assert response.content == "Test response"
        assert response.model == "llama3.2:3b"
        assert response.latency_ms == 150.5

    def test_error_dataclass(self):
        """LLMError should hold error details."""
        from HoloLoom.visualization.jenny_llm_client import LLMError

        error = LLMError(
            error_type="timeout",
            message="Request timed out",
            provider="ollama",
            retriable=True,
            status_code=504,
        )

        assert error.error_type == "timeout"
        assert error.retriable is True

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_llm(self):
        """Should handle LLM unavailability gracefully."""
        from HoloLoom.visualization.jenny_llm_client import LLMClientConfig

        # Config with invalid endpoint should not crash on creation
        config = LLMClientConfig(
            ollama_base_url="http://invalid:99999"
        )

        assert config.ollama_base_url == "http://invalid:99999"


# ============================================================================
# M3: Accessibility Integration Tests
# ============================================================================

class TestAccessibilityIntegration:
    """M3: WCAG 2.1 AA Accessibility integration tests."""

    def test_aria_roles_enum(self):
        """Should have all required ARIA roles."""
        from HoloLoom.visualization.jenny_accessibility import AriaRole

        # Landmark roles
        assert AriaRole.MAIN.value == "main"
        assert AriaRole.REGION.value == "region"

        # Widget roles
        assert AriaRole.BUTTON.value == "button"
        assert AriaRole.DIALOG.value == "dialog"

        # Live region roles
        assert AriaRole.ALERT.value == "alert"
        assert AriaRole.STATUS.value == "status"

    def test_aria_live_settings(self):
        """Should have ARIA live region settings."""
        from HoloLoom.visualization.jenny_accessibility import AriaLive

        assert AriaLive.OFF.value == "off"
        assert AriaLive.POLITE.value == "polite"
        assert AriaLive.ASSERTIVE.value == "assertive"

    def test_aria_attributes_to_dict(self):
        """Should convert AriaAttributes to dict."""
        from HoloLoom.visualization.jenny_accessibility import (
            AriaAttributes,
            AriaRole,
            AriaLive,
        )

        attrs = AriaAttributes(
            role=AriaRole.MAIN,
            label="Jenny Dashboard",
            live=AriaLive.POLITE,
        )

        attr_dict = attrs.to_dict()

        assert attr_dict["role"] == "main"
        assert attr_dict["aria-label"] == "Jenny Dashboard"
        assert attr_dict["aria-live"] == "polite"

    def test_aria_attributes_boolean_handling(self):
        """Should handle boolean attributes correctly."""
        from HoloLoom.visualization.jenny_accessibility import AriaAttributes

        attrs = AriaAttributes(
            expanded=True,
            hidden=False,
            disabled=True,
        )

        attr_dict = attrs.to_dict()

        assert attr_dict["aria-expanded"] == "true"
        assert attr_dict["aria-hidden"] == "false"
        assert attr_dict["aria-disabled"] == "true"


# ============================================================================
# M4: React Renderer Integration Tests
# ============================================================================

class TestReactRendererIntegration:
    """M4: React Props rendering integration tests."""

    @pytest.mark.asyncio
    async def test_render_to_react_props(self, sample_specs):
        """Should render specs to React component props."""
        from HoloLoom.visualization.jenny_renderer import ReactRenderer

        renderer = ReactRenderer()
        output = await renderer.render(sample_specs, target=RenderTarget.REACT)

        # Should be valid JSON
        props = json.loads(output)

        assert props['component'] == 'JennyDashboard'
        assert 'panels' in props['props']
        assert len(props['props']['panels']) == 3

    @pytest.mark.asyncio
    async def test_accessibility_props_included(self, sample_specs):
        """Should include accessibility props in React output."""
        from HoloLoom.visualization.jenny_renderer import ReactRenderer

        renderer = ReactRenderer()
        output = await renderer.render(
            sample_specs,
            target=RenderTarget.REACT,
            options={'include_accessibility': True}
        )

        props = json.loads(output)

        # Dashboard-level accessibility
        assert 'accessibility' in props['props']
        assert props['props']['accessibility']['role'] == 'main'

        # Panel-level accessibility
        for panel in props['props']['panels']:
            assert 'accessibility' in panel['props']

    @pytest.mark.asyncio
    async def test_event_handler_props(self, sample_specs):
        """Should include event handler props."""
        from HoloLoom.visualization.jenny_renderer import ReactRenderer

        renderer = ReactRenderer()
        output = await renderer.render(
            sample_specs,
            target=RenderTarget.REACT,
            options={'include_handlers': True}
        )

        props = json.loads(output)

        # Dashboard handlers
        assert 'handlers' in props['props']
        assert 'onPanelFocus' in props['props']['handlers']

        # Panel handlers
        for panel in props['props']['panels']:
            assert 'handlers' in panel['props']

    @pytest.mark.asyncio
    async def test_content_props_by_panel_type(self, sample_specs):
        """Should generate type-specific content props."""
        from HoloLoom.visualization.jenny_renderer import ReactRenderer

        renderer = ReactRenderer()
        output = await renderer.render(sample_specs, target=RenderTarget.REACT)

        props = json.loads(output)

        for panel in props['props']['panels']:
            panel_type = panel['props']['panelType']
            content_props = panel['props']['contentProps']

            if panel_type == 'text':
                assert 'text' in content_props
            elif panel_type == 'confidence':
                assert 'value' in content_props
                assert 'percentage' in content_props
            elif panel_type == 'sources':
                assert 'sources' in content_props


# ============================================================================
# M5: AR Renderer Integration Tests
# ============================================================================

class TestARRendererIntegration:
    """M5: WebXR AR rendering integration tests."""

    @pytest.mark.asyncio
    async def test_render_to_ar_json(self, sample_specs):
        """Should render specs to AR JSON."""
        from HoloLoom.visualization.jenny_renderer import ARRenderer

        renderer = ARRenderer()
        output = await renderer.render(sample_specs, target=RenderTarget.AR)

        # Should be valid JSON
        ar_scene = json.loads(output)

        assert ar_scene['ar_version'] == '1.0'
        assert 'panels' in ar_scene
        assert len(ar_scene['panels']) == 3

    @pytest.mark.asyncio
    async def test_transform_calculations(self, sample_specs):
        """Should calculate 3D transforms correctly."""
        from HoloLoom.visualization.jenny_renderer import ARRenderer

        renderer = ARRenderer()
        output = await renderer.render(sample_specs, target=RenderTarget.AR)

        ar_scene = json.loads(output)

        for panel in ar_scene['panels']:
            transform = panel['transform']

            # Position should have x, y, z
            assert 'position' in transform
            assert 'x' in transform['position']
            assert 'y' in transform['position']
            assert 'z' in transform['position']

            # Scale should have x, y, z
            assert 'scale' in transform
            assert transform['scale']['z'] == 0.01  # Panel depth

    @pytest.mark.asyncio
    async def test_coordinate_systems(self, sample_specs):
        """Should support different coordinate systems."""
        from HoloLoom.visualization.jenny_renderer import ARRenderer

        renderer = ARRenderer()

        # World coordinates (default)
        output = await renderer.render(
            sample_specs,
            target=RenderTarget.AR,
            options={'coordinate_system': 'world'}
        )
        ar_scene = json.loads(output)
        assert ar_scene['coordinate_system'] == 'world'

        # Device coordinates
        output = await renderer.render(
            sample_specs,
            target=RenderTarget.AR,
            options={'coordinate_system': 'device'}
        )
        ar_scene = json.loads(output)
        assert ar_scene['coordinate_system'] == 'device'

    @pytest.mark.asyncio
    async def test_size_to_meters_mapping(self, sample_specs):
        """Should map panel sizes to physical meters."""
        from HoloLoom.visualization.jenny_renderer import ARRenderer

        renderer = ARRenderer()

        # Check SIZE_TO_METERS constant
        assert PanelSizeJenny.SMALL in renderer.SIZE_TO_METERS
        assert PanelSizeJenny.LARGE in renderer.SIZE_TO_METERS

        # Large panels should be bigger than small
        small_scale = renderer.SIZE_TO_METERS[PanelSizeJenny.SMALL]
        large_scale = renderer.SIZE_TO_METERS[PanelSizeJenny.LARGE]

        assert large_scale['x'] > small_scale['x']
        assert large_scale['y'] > small_scale['y']


# ============================================================================
# M6: Analytics Integration Tests
# ============================================================================

class TestAnalyticsIntegration:
    """M6: Analytics collection integration tests."""

    def test_record_render_event(self, analytics_collector):
        """Should record render events."""
        analytics_collector.record_render(
            latency_ms=45.2,
            target='html',
            panel_count=3,
            cache_hit=True
        )

        metrics = analytics_collector.get_render_metrics()

        assert metrics['total_renders'] == 1
        assert metrics['avg_latency_ms'] == 45.2
        assert metrics['cache_hit_rate'] == 1.0

    def test_record_compile_event(self, analytics_collector):
        """Should record compile events."""
        analytics_collector.record_compile(
            latency_ms=120.5,
            model='llama3.2:3b',
            query_type='factual',
            panels_generated=5,
            token_count=150
        )

        metrics = analytics_collector.get_compile_metrics()

        assert metrics['total_compiles'] == 1
        assert metrics['avg_latency_ms'] == 120.5
        assert metrics['total_tokens'] == 150

    def test_render_event_property_pattern(self):
        """RenderEvent should use @property pattern."""
        from HoloLoom.visualization.jenny_analytics import RenderEvent, MetricType

        event = RenderEvent(
            value=50.0,
            target='react',
            panel_count=4,
            cache_hit=False,
        )

        # Properties should work correctly
        assert event.event_type == MetricType.RENDER_LATENCY
        assert event.metadata['target'] == 'react'
        assert event.metadata['panel_count'] == 4
        assert event.metadata['cache_hit'] is False

    def test_compile_event_property_pattern(self):
        """CompileEvent should use @property pattern."""
        from HoloLoom.visualization.jenny_analytics import CompileEvent, MetricType

        event = CompileEvent(
            value=100.0,
            model='gpt-4',
            query_type='analytical',
            panels_generated=3,
            token_count=200,
        )

        assert event.event_type == MetricType.COMPILE_LATENCY
        assert event.metadata['model'] == 'gpt-4'
        assert event.metadata['token_count'] == 200

    def test_analytics_dashboard_render(self, analytics_collector):
        """Should render analytics dashboard HTML."""
        from HoloLoom.visualization.jenny_analytics import JennyAnalyticsDashboard

        # Record some events
        analytics_collector.record_render(latency_ms=50, target='html', panel_count=3)
        analytics_collector.record_compile(latency_ms=100, model='ollama')
        analytics_collector.record_confidence(0.85)

        dashboard = JennyAnalyticsDashboard(analytics_collector)
        html = dashboard.render()

        assert '<!DOCTYPE html>' in html
        assert 'Jenny Learning Analytics' in html
        assert 'Render Performance' in html

    def test_aggregation_metrics(self, analytics_collector):
        """Should aggregate metrics correctly."""
        # Record multiple events
        for i in range(10):
            analytics_collector.record_render(
                latency_ms=40 + i * 2,  # 40-58ms
                target='html' if i % 2 == 0 else 'react',
                panel_count=3,
                cache_hit=(i % 3 == 0)
            )

        metrics = analytics_collector.get_render_metrics()

        assert metrics['total_renders'] == 10
        assert 40 <= metrics['avg_latency_ms'] <= 58
        # 4 cache hits out of 10 (i=0,3,6,9)
        assert 0.3 <= metrics['cache_hit_rate'] <= 0.5

    def test_summary_includes_all_sections(self, analytics_collector):
        """Summary should include all metric sections."""
        analytics_collector.record_render(latency_ms=50)
        analytics_collector.record_compile(latency_ms=100)
        analytics_collector.record_confidence(0.85)

        summary = analytics_collector.get_summary()

        assert 'uptime_seconds' in summary
        assert 'total_events' in summary
        assert 'render' in summary
        assert 'compile' in summary
        assert 'confidence' in summary
        assert 'panel_types' in summary
        assert 'accessibility' in summary


# ============================================================================
# Full Moonshot Pipeline Tests
# ============================================================================

class TestFullMoonshotPipeline:
    """Full pipeline integration: All 6 moonshot phases together."""

    @pytest.mark.asyncio
    async def test_complete_moonshot_pipeline(
        self,
        sample_specs,
        fresh_registry,
        analytics_collector
    ):
        """
        Complete pipeline test:
        1. Register renderers (M1)
        2. Render to HTML (M1)
        3. Render to React with accessibility (M3, M4)
        4. Render to AR (M5)
        5. Record analytics (M6)
        """
        from HoloLoom.visualization.jenny_renderer import (
            HTMLRenderer,
            ReactRenderer,
            ARRenderer,
        )

        # M1: Register renderers
        fresh_registry.register(HTMLRenderer, priority=10)
        fresh_registry.register(ReactRenderer, priority=8)
        fresh_registry.register(ARRenderer, priority=7)

        # M1: Render to HTML
        html_output = await fresh_registry.render(
            sample_specs,
            target=RenderTarget.HTML
        )
        assert len(html_output) > 0
        assert 'jenny-dashboard' in html_output.lower()

        # M6: Record render analytics
        analytics_collector.record_render(
            latency_ms=45.0,
            target='html',
            panel_count=len(sample_specs)
        )

        # M3 + M4: Render to React with accessibility
        react_output = await fresh_registry.render(
            sample_specs,
            target=RenderTarget.REACT,
            options={'include_accessibility': True}
        )
        react_props = json.loads(react_output)

        # Verify accessibility included
        assert 'accessibility' in react_props['props']

        # M6: Record React render
        analytics_collector.record_render(
            latency_ms=35.0,
            target='react',
            panel_count=len(sample_specs)
        )

        # M5: Render to AR
        ar_output = await fresh_registry.render(
            sample_specs,
            target=RenderTarget.AR,
            options={'coordinate_system': 'world'}
        )
        ar_scene = json.loads(ar_output)

        # Verify AR structure
        assert ar_scene['ar_version'] == '1.0'
        assert len(ar_scene['panels']) == 3

        # M6: Record AR render
        analytics_collector.record_render(
            latency_ms=55.0,
            target='ar',
            panel_count=len(sample_specs)
        )

        # M6: Verify analytics collected
        summary = analytics_collector.get_summary()
        assert summary['total_events'] == 3
        assert summary['render']['total_renders'] == 3

        # Verify by-target breakdown
        by_target = summary['render']['by_target']
        assert 'html' in by_target
        assert 'react' in by_target
        assert 'ar' in by_target

    @pytest.mark.asyncio
    async def test_accessibility_in_all_render_targets(self, sample_specs):
        """Accessibility props should be available in React output."""
        from HoloLoom.visualization.jenny_renderer import ReactRenderer

        renderer = ReactRenderer()

        # React with accessibility
        output = await renderer.render(
            sample_specs,
            target=RenderTarget.REACT,
            options={'include_accessibility': True}
        )
        props = json.loads(output)

        # Dashboard accessibility
        assert props['props']['accessibility']['role'] == 'main'
        assert props['props']['accessibility']['ariaLive'] == 'polite'

        # Panel accessibility
        for i, panel in enumerate(props['props']['panels']):
            a11y = panel['props']['accessibility']
            assert a11y['role'] == 'article'
            assert a11y['ariaSetsize'] == 3  # Total panels

    @pytest.mark.asyncio
    async def test_concurrent_multi_target_render(
        self,
        sample_specs,
        fresh_registry,
        analytics_collector
    ):
        """Should render to multiple targets concurrently."""
        from HoloLoom.visualization.jenny_renderer import (
            HTMLRenderer,
            JSONRenderer,
            ReactRenderer,
        )

        fresh_registry.register(HTMLRenderer, priority=10)
        fresh_registry.register(JSONRenderer, priority=5)
        fresh_registry.register(ReactRenderer, priority=8)

        # Concurrent render
        results = await fresh_registry.render_concurrent(
            sample_specs,
            targets=[RenderTarget.HTML, RenderTarget.JSON, RenderTarget.REACT]
        )

        # All targets should have output
        assert len(results) == 3

        # Record analytics for all
        for target, output in results.items():
            analytics_collector.record_render(
                latency_ms=30.0,
                target=target.value,
                panel_count=len(sample_specs)
            )

        # Verify analytics
        metrics = analytics_collector.get_render_metrics()
        assert metrics['total_renders'] == 3


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_render_jenny_function(self, fresh_registry, sample_specs):
        """render_jenny should use global registry."""
        from HoloLoom.visualization.jenny_renderer import HTMLRenderer
        from HoloLoom.visualization.jenny_renderer_registry import render_jenny

        fresh_registry.register(HTMLRenderer, priority=10)

        # This uses the global singleton
        output = asyncio.get_event_loop().run_until_complete(
            render_jenny(sample_specs, target=RenderTarget.HTML)
        )

        assert len(output) > 0

    def test_analytics_convenience_functions(self):
        """Module-level analytics functions should work."""
        from HoloLoom.visualization.jenny_analytics import (
            record_render,
            record_compile,
            record_confidence,
            get_analytics_summary,
            get_analytics_collector,
        )

        # Clear first
        get_analytics_collector().clear()

        # Use convenience functions
        record_render(latency_ms=50, target='html')
        record_compile(latency_ms=100, model='test')
        record_confidence(0.9)

        summary = get_analytics_summary()

        assert summary['render']['total_renders'] == 1
        assert summary['compile']['total_compiles'] == 1


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_specs_list(self, fresh_registry):
        """Should handle empty specs gracefully."""
        from HoloLoom.visualization.jenny_renderer import HTMLRenderer

        fresh_registry.register(HTMLRenderer, priority=10)

        output = asyncio.get_event_loop().run_until_complete(
            fresh_registry.render([], target=RenderTarget.HTML)
        )

        # Should return valid but minimal output
        assert output is not None

    def test_no_renderer_for_target(self, fresh_registry, sample_specs):
        """Should raise error when no renderer available."""
        # Registry is empty
        with pytest.raises(ValueError, match="No renderer available"):
            asyncio.get_event_loop().run_until_complete(
                fresh_registry.render(sample_specs, target=RenderTarget.HTML)
            )

    def test_analytics_with_no_events(self):
        """Should handle empty analytics gracefully."""
        from HoloLoom.visualization.jenny_analytics import JennyAnalyticsCollector

        collector = JennyAnalyticsCollector()
        collector.clear()

        metrics = collector.get_render_metrics()

        assert metrics['total_renders'] == 0
        assert metrics['avg_latency_ms'] == 0
