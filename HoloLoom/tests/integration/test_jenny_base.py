"""
Jenny Renderer Test Base Class
==============================
Abstract base class for testing Jenny renderers with minimal boilerplate.

Philosophy:
> "Adding a new renderer test should require ~20 lines of code, not 200."

The base class provides:
- Standard fixtures for JennySpec test data
- Common test methods inherited by all renderer tests
- Consistent assertion patterns
- Performance benchmarking helpers

Usage:
    class TestHTMLRenderer(RendererTestBase):
        @property
        def renderer_class(self):
            return HTMLRenderer

        @property
        def supported_target(self):
            return RenderTarget.HTML

        # Inherits 10+ standard tests automatically!
        # Add renderer-specific tests as needed.

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot Extensibility)
"""

import pytest
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Type, List, Dict, Any, Optional
from dataclasses import dataclass

# Jenny imports
from HoloLoom.protocols.jenny import (
    JennyRendererProtocol,
    RenderTarget,
)

try:
    from HoloLoom.visualization import (
        JennySpec,
        PanelTypeJenny,
        BindingMode,
        LifecycleStage,
    )
    JENNY_AVAILABLE = True
except ImportError:
    JENNY_AVAILABLE = False


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def jenny_specs_empty() -> List:
    """Empty list of specs."""
    return []


@pytest.fixture
def jenny_specs_single():
    """Single JennySpec for basic tests."""
    if not JENNY_AVAILABLE:
        pytest.skip("Jenny visualization not available")

    return [
        JennySpec(
            id="test-single-001",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.PINNED,
            confidence=0.85,
            raw_content="Test content for single spec",
            rendered_content=None,
            metadata={"source": "test"},
            actions=["pin_panel", "dismiss_panel"],
            accessibility={"aria-label": "Test panel"},
        )
    ]


@pytest.fixture
def jenny_specs_multiple():
    """Multiple JennySpecs for comprehensive tests."""
    if not JENNY_AVAILABLE:
        pytest.skip("Jenny visualization not available")

    return [
        JennySpec(
            id="test-multi-001",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=0.92,
            raw_content="First panel content",
            rendered_content=None,
            metadata={"source": "test", "order": 1},
            actions=["pin_panel", "dismiss_panel"],
        ),
        JennySpec(
            id="test-multi-002",
            panel_type=PanelTypeJenny.CODE,
            binding_mode=BindingMode.CONTEXT,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=0.78,
            raw_content="def hello(): return 'world'",
            rendered_content=None,
            metadata={"source": "test", "order": 2, "language": "python"},
            actions=["copy_code", "dismiss_panel"],
        ),
        JennySpec(
            id="test-multi-003",
            panel_type=PanelTypeJenny.GRAPH,
            binding_mode=BindingMode.MEMORY,
            lifecycle=LifecycleStage.PINNED,
            confidence=0.65,
            raw_content='{"nodes": [{"id": "A"}, {"id": "B"}], "edges": []}',
            rendered_content=None,
            metadata={"source": "test", "order": 3},
            actions=["zoom", "dismiss_panel"],
        ),
    ]


@pytest.fixture
def jenny_specs_all_panel_types():
    """One spec for each panel type."""
    if not JENNY_AVAILABLE:
        pytest.skip("Jenny visualization not available")

    specs = []
    for i, panel_type in enumerate(PanelTypeJenny):
        specs.append(
            JennySpec(
                id=f"test-type-{i:03d}",
                panel_type=panel_type,
                binding_mode=BindingMode.QUERY,
                lifecycle=LifecycleStage.ACTIVE,
                confidence=0.80,
                raw_content=f"Content for {panel_type.value} panel",
                rendered_content=None,
                metadata={"panel_type": panel_type.value},
                actions=["dismiss_panel"],
            )
        )
    return specs


@pytest.fixture
def jenny_specs_edge_cases():
    """Edge case specs for robustness testing."""
    if not JENNY_AVAILABLE:
        pytest.skip("Jenny visualization not available")

    return [
        # Empty content
        JennySpec(
            id="test-edge-001",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=0.5,
            raw_content="",
            rendered_content=None,
            metadata={},
            actions=[],
        ),
        # Very long content
        JennySpec(
            id="test-edge-002",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=0.5,
            raw_content="x" * 10000,
            rendered_content=None,
            metadata={},
            actions=[],
        ),
        # Unicode content
        JennySpec(
            id="test-edge-003",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=0.5,
            raw_content="Unicode: 你好世界 🌍 αβγδ",
            rendered_content=None,
            metadata={},
            actions=[],
        ),
        # Low confidence
        JennySpec(
            id="test-edge-004",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=0.01,
            raw_content="Low confidence content",
            rendered_content=None,
            metadata={},
            actions=[],
        ),
        # Max confidence
        JennySpec(
            id="test-edge-005",
            panel_type=PanelTypeJenny.TEXT,
            binding_mode=BindingMode.QUERY,
            lifecycle=LifecycleStage.ACTIVE,
            confidence=1.0,
            raw_content="Max confidence content",
            rendered_content=None,
            metadata={},
            actions=[],
        ),
    ]


# ============================================================================
# Renderer Test Base Class
# ============================================================================

class RendererTestBase(ABC):
    """
    Abstract base class for Jenny renderer tests.

    Subclasses must implement:
    - renderer_class: The renderer class to test
    - supported_target: The RenderTarget this renderer supports

    Inherited tests:
    - test_render_empty_specs: Empty input handling
    - test_render_single_spec: Basic rendering
    - test_render_multiple_specs: Multiple panels
    - test_render_all_panel_types: All panel types supported
    - test_render_edge_cases: Robustness testing
    - test_render_options: Custom options handling
    - test_render_concurrent: Concurrent rendering safety
    - test_render_performance: Performance benchmarks
    - test_renderer_properties: Property validation
    - test_supports_concurrent: Concurrency flag
    """

    @property
    @abstractmethod
    def renderer_class(self) -> Type[JennyRendererProtocol]:
        """Return the renderer class to test."""
        pass

    @property
    @abstractmethod
    def supported_target(self) -> RenderTarget:
        """Return the RenderTarget this renderer supports."""
        pass

    # ========================================================================
    # Setup / Helpers
    # ========================================================================

    def create_renderer(self) -> JennyRendererProtocol:
        """Create an instance of the renderer."""
        return self.renderer_class()

    # ========================================================================
    # Standard Tests (Inherited by All Subclasses)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_render_empty_specs(self, jenny_specs_empty):
        """Test rendering with empty spec list."""
        renderer = self.create_renderer()
        result = await renderer.render(jenny_specs_empty, target=self.supported_target)

        assert result is not None
        assert isinstance(result, str)
        # Empty specs should produce minimal output
        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_render_single_spec(self, jenny_specs_single):
        """Test rendering a single spec."""
        renderer = self.create_renderer()
        result = await renderer.render(jenny_specs_single, target=self.supported_target)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Content should appear in output (implementation-specific)

    @pytest.mark.asyncio
    async def test_render_multiple_specs(self, jenny_specs_multiple):
        """Test rendering multiple specs."""
        renderer = self.create_renderer()
        result = await renderer.render(jenny_specs_multiple, target=self.supported_target)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_render_all_panel_types(self, jenny_specs_all_panel_types):
        """Test rendering all panel types."""
        renderer = self.create_renderer()
        result = await renderer.render(
            jenny_specs_all_panel_types,
            target=self.supported_target
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_render_edge_cases(self, jenny_specs_edge_cases):
        """Test rendering edge case specs."""
        renderer = self.create_renderer()

        # Should not raise exceptions
        result = await renderer.render(
            jenny_specs_edge_cases,
            target=self.supported_target
        )

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_render_with_options(self, jenny_specs_single):
        """Test rendering with custom options."""
        renderer = self.create_renderer()

        # Basic options that most renderers should handle
        options = {
            "indent": 2,
            "pretty": True,
        }

        result = await renderer.render(
            jenny_specs_single,
            target=self.supported_target,
            options=options
        )

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_render_concurrent(self, jenny_specs_multiple):
        """Test concurrent rendering safety."""
        renderer = self.create_renderer()

        if not renderer.supports_concurrent():
            pytest.skip("Renderer does not support concurrent rendering")

        # Run multiple renders concurrently
        tasks = [
            renderer.render(jenny_specs_multiple, target=self.supported_target)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert result is not None
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_render_performance(self, jenny_specs_multiple):
        """Test render performance (basic benchmark)."""
        renderer = self.create_renderer()

        # Warm up
        await renderer.render(jenny_specs_multiple, target=self.supported_target)

        # Benchmark
        iterations = 10
        start = time.perf_counter()

        for _ in range(iterations):
            await renderer.render(jenny_specs_multiple, target=self.supported_target)

        duration_ms = (time.perf_counter() - start) * 1000
        avg_ms = duration_ms / iterations

        # Basic performance assertion (adjust threshold as needed)
        assert avg_ms < 1000, f"Render too slow: {avg_ms:.2f}ms average"

        # Record for reporting
        print(f"\n  Performance: {avg_ms:.2f}ms avg over {iterations} iterations")

    def test_renderer_properties(self):
        """Test renderer properties are correctly defined."""
        renderer = self.create_renderer()

        # Name should be non-empty string
        assert isinstance(renderer.name, str)
        assert len(renderer.name) > 0

        # Supported targets should include our target
        assert isinstance(renderer.supported_targets, list)
        assert self.supported_target in renderer.supported_targets

    def test_supports_concurrent(self):
        """Test supports_concurrent returns boolean."""
        renderer = self.create_renderer()

        result = renderer.supports_concurrent()
        assert isinstance(result, bool)


# ============================================================================
# Performance Benchmark Helper
# ============================================================================

@dataclass
class RenderBenchmark:
    """Results from render performance benchmarking."""
    renderer_name: str
    target: RenderTarget
    iterations: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    specs_count: int


async def benchmark_renderer(
    renderer: JennyRendererProtocol,
    specs: List,
    target: RenderTarget,
    iterations: int = 100
) -> RenderBenchmark:
    """
    Run performance benchmark on a renderer.

    Args:
        renderer: Renderer instance to benchmark
        specs: List of JennySpecs to render
        target: Render target
        iterations: Number of iterations

    Returns:
        RenderBenchmark with timing results
    """
    timings = []

    # Warm up
    await renderer.render(specs, target=target)

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        await renderer.render(specs, target=target)
        timings.append((time.perf_counter() - start) * 1000)

    return RenderBenchmark(
        renderer_name=renderer.name,
        target=target,
        iterations=iterations,
        total_ms=sum(timings),
        avg_ms=sum(timings) / len(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        specs_count=len(specs),
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'RendererTestBase',
    'RenderBenchmark',
    'benchmark_renderer',

    # Fixtures (accessed via pytest)
    'jenny_specs_empty',
    'jenny_specs_single',
    'jenny_specs_multiple',
    'jenny_specs_all_panel_types',
    'jenny_specs_edge_cases',
]
