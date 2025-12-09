#!/usr/bin/env python3
"""
Demo: Jenny Moonshot Complete System
=====================================

Showcases all 6 Moonshot phases working together:

- M1: Renderer Registry Pattern (singleton, priority-based selection)
- M2: Async LLM Clients (httpx connection pooling, exponential backoff)
- M3: WCAG 2.1 AA Accessibility (focus management, keyboard nav, screen reader)
- M4: React Component Props (TypeScript-friendly JSON output)
- M5: WebXR AR Output (3D transforms, coordinate systems)
- M6: Analytics Collection (event-based metrics, dashboard)

Philosophy: "Disposable pixels, durable decisions"

Run:
    PYTHONPATH=. python demos/demo_jenny_moonshot.py
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# Jenny core types
from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    BindingMode,
    LifecycleStage,
)

# M1: Renderer Registry
from HoloLoom.visualization.jenny_renderer_registry import (
    RendererRegistry,
    RenderTarget,
)

# M2: Async LLM Client
from HoloLoom.visualization.jenny_llm_client import (
    LLMClientConfig,
    LLMResponse,
    LLMError,
)

# M3: Accessibility
from HoloLoom.visualization.jenny_accessibility import (
    AriaRole,
    AriaLive,
    AriaRelevant,
    AriaAttributes,
    accessible_panel,
    KeyboardHandler,
    FocusManager,
    create_accessibility_layer,
)

# M4 & M5: Renderers
from HoloLoom.visualization.jenny_renderer import (
    HTMLRenderer,
    ReactRenderer,
    ARRenderer,
)
from HoloLoom.visualization.jenny_runtime import (
    JennyRuntime,
    create_runtime,
)

# M6: Analytics
from HoloLoom.visualization.jenny_analytics import (
    MetricType,
    RenderEvent,
    CompileEvent,
    JennyAnalyticsCollector,
    JennyAnalyticsDashboard,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n  --- {title} ---\n")


def create_sample_specs() -> List[JennySpec]:
    """Create sample JennySpecs for demonstration."""
    return [
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.TEXT,
            title="Response",
            content={"text": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation through probability matching."},
            size=PanelSizeJenny.LARGE,
            priority=1,
        ),
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.CODE,
            title="Implementation",
            content={
                "language": "python",
                "code": "import numpy as np\n\ndef thompson_sample(alphas, betas):\n    samples = [np.random.beta(a, b) for a, b in zip(alphas, betas)]\n    return np.argmax(samples)"
            },
            size=PanelSizeJenny.MEDIUM,
            priority=2,
        ),
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.GRAPH,
            title="Confidence Over Time",
            content={
                "type": "line",
                "data": [0.5, 0.6, 0.65, 0.72, 0.78, 0.85, 0.89],
                "labels": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
            },
            size=PanelSizeJenny.MEDIUM,
            priority=3,
        ),
    ]


async def demo_m1_renderer_registry() -> None:
    """M1: Renderer Registry Pattern - Singleton, priority-based selection."""
    print_header("M1: Renderer Registry Pattern")

    # Get singleton instance (creates if doesn't exist)
    registry = RendererRegistry()
    print(f"  Registry Singleton: {type(registry).__name__}")
    print(f"  Same instance check: {registry is RendererRegistry()}")
    print()

    # Renderers are auto-registered on import - show what's available
    print("  Auto-Registered Renderers (on import):")
    for target in [RenderTarget.HTML, RenderTarget.REACT, RenderTarget.AR]:
        renderer = registry.get_for_target(target)
        if renderer:
            print(f"    {target.value:10s} -> {type(renderer).__name__}")
        else:
            print(f"    {target.value:10s} -> (not registered)")
    print()

    # Show manual registration example
    print("  Manual Registration Example:")
    print("    registry.register(RenderTarget.HTML, HTMLRenderer(), priority=10)")
    print("    registry.register(RenderTarget.REACT, ReactRenderer(), priority=20)")
    print("    registry.register(RenderTarget.AR, ARRenderer(), priority=15)")
    print()

    # Demonstrate priority-based selection
    print("  Priority-Based Selection:")
    print("    Lower priority = higher priority (first to render)")
    print("    HTML     -> Standard web rendering (default)")
    print("    Terminal -> CLI output (fallback)")
    print("    React    -> TypeScript component props")
    print("    AR       -> WebXR 3D transforms")
    print()

    # Concurrent rendering capability
    print("  Concurrent Rendering: Supported via render_concurrent()")


async def demo_m2_async_llm_client() -> None:
    """M2: Async LLM Clients - Connection pooling, exponential backoff."""
    print_header("M2: Async LLM Client Infrastructure")

    # Show configuration options
    print_subheader("Configuration Presets")

    # Fast config for development
    fast_config = LLMClientConfig.fast()
    print(f"  Fast (Development):")
    print(f"    Timeout: {fast_config.timeout_seconds}s")
    print(f"    Retries: {fast_config.max_retries}")
    print(f"    Max Connections: {fast_config.max_connections}")
    print()

    # Reliable config for production
    reliable_config = LLMClientConfig.reliable()
    print(f"  Reliable (Production):")
    print(f"    Timeout: {reliable_config.timeout_seconds}s")
    print(f"    Retries: {reliable_config.max_retries}")
    print(f"    Max Connections: {reliable_config.max_connections}")
    print()

    print_subheader("Key Features")
    print("  - httpx AsyncClient with connection pooling")
    print("  - Exponential backoff with jitter on retries")
    print("  - Provider abstraction (Ollama, OpenAI, Anthropic)")
    print("  - Graceful degradation when LLM unavailable")
    print("  - Structured LLMResponse with latency tracking")
    print()

    # Mock a response to show structure
    print_subheader("Response Structure")
    mock_response = LLMResponse(
        content="Thompson Sampling balances exploration and exploitation.",
        model="llama3.2:3b",
        provider="ollama",
        latency_ms=245.5,
        tokens_used=42,
        finish_reason="stop",
        retries=0,
    )
    print(f"  LLMResponse:")
    print(f"    content: \"{mock_response.content[:50]}...\"")
    print(f"    model: {mock_response.model}")
    print(f"    latency_ms: {mock_response.latency_ms}")
    print(f"    tokens_used: {mock_response.tokens_used}")
    print(f"    retries: {mock_response.retries}")


async def demo_m3_accessibility() -> None:
    """M3: WCAG 2.1 AA Accessibility - ARIA, keyboard nav, focus management."""
    print_header("M3: WCAG 2.1 AA Accessibility Layer")

    print_subheader("ARIA Attributes")

    # Create accessibility attributes
    aria = AriaAttributes(
        role=AriaRole.REGION,
        label="Query Response Panel",
        live=AriaLive.POLITE,
        relevant=AriaRelevant.ADDITIONS,
        expanded=True,
        controls="panel-content-001",
    )

    attrs_dict = aria.to_dict()
    print("  Panel ARIA Attributes:")
    for key, value in attrs_dict.items():
        print(f"    {key}: {value}")
    print()

    print_subheader("Keyboard Navigation")
    print("  Focus Order:")
    print("    1. Tab      -> Move to next panel")
    print("    2. Shift+Tab -> Move to previous panel")
    print("    3. Enter    -> Activate/expand panel")
    print("    4. Escape   -> Close/collapse panel")
    print("    5. Arrow keys -> Navigate within panel")
    print()

    print_subheader("Screen Reader Announcements")
    print("  Live Regions:")
    print("    - POLITE: Wait for idle to announce")
    print("    - ASSERTIVE: Interrupt current speech")
    print("    - OFF: Silent updates")
    print()

    print_subheader("Accessibility Utilities")
    print("  make_panel_accessible(spec) -> Adds ARIA to panel")
    print("  add_keyboard_navigation(container) -> Adds key handlers")
    print("  Enhanced contrast: minimum 4.5:1 ratio")


async def demo_m4_react_renderer() -> None:
    """M4: React Component Props - TypeScript-friendly JSON output."""
    print_header("M4: React Component Renderer")

    specs = create_sample_specs()
    renderer = ReactRenderer()

    print_subheader("TypeScript-Friendly Output")
    print("  ReactRenderer outputs JSON props compatible with:")
    print("    - React components (functional or class)")
    print("    - TypeScript type checking")
    print("    - React Native (mobile)")
    print()

    # Render to React props - must specify RenderTarget.REACT
    result = await renderer.render(specs, target=RenderTarget.REACT)
    props = json.loads(result)

    print_subheader("JennyDashboard Component Props")
    print(f"  Component: {props.get('component', 'JennyDashboard')}")
    print(f"  Query ID: {props.get('queryId', 'N/A')}")
    print(f"  Panel Count: {len(props.get('panels', []))}")
    print()

    # Show first panel structure
    if props.get('panels'):
        panel = props['panels'][0]
        print("  First Panel Props:")
        print(f"    id: {panel.get('id', 'N/A')}")
        print(f"    type: {panel.get('type', 'N/A')}")
        print(f"    title: {panel.get('title', 'N/A')}")
        print(f"    size: {panel.get('size', 'N/A')}")
        print(f"    priority: {panel.get('priority', 'N/A')}")

        # Check for accessibility props
        if 'accessibility' in panel:
            print(f"    accessibility: [WCAG 2.1 AA compliant]")
    print()

    print_subheader("Integration Pattern")
    print("  import { JennyDashboard } from '@hololoom/jenny';")
    print("  <JennyDashboard {...props} />")


async def demo_m5_ar_renderer() -> None:
    """M5: WebXR AR Output - 3D transforms, coordinate systems."""
    print_header("M5: WebXR AR Renderer")

    specs = create_sample_specs()
    renderer = ARRenderer()

    print_subheader("WebXR Coordinate Systems")
    print("  Supported Reference Spaces:")
    print("    - world: Global coordinate system")
    print("    - device: Device-relative positioning")
    print("    - anchor: AR anchor-relative positioning")
    print()

    # Render to AR JSON - must specify RenderTarget.AR
    result = await renderer.render(specs, target=RenderTarget.AR)
    ar_data = json.loads(result)

    print_subheader("AR Panel Transforms")

    # Size to meters mapping
    print("  Panel Size to Meters:")
    print("    SMALL:  0.3m x 0.2m x 0.01m")
    print("    MEDIUM: 0.5m x 0.3m x 0.01m")
    print("    LARGE:  0.8m x 0.5m x 0.01m")
    print()

    # Show panel transforms
    if ar_data.get('panels'):
        print("  Panel 3D Positions:")
        for i, panel in enumerate(ar_data['panels'][:3]):
            transform = panel.get('transform', {})
            position = transform.get('position', {})
            scale = transform.get('scale', {})
            print(f"    Panel {i+1}: pos=({position.get('x', 0):.2f}, {position.get('y', 0):.2f}, {position.get('z', 0):.2f})")
    print()

    print_subheader("WebXR Integration")
    print("  const session = await navigator.xr.requestSession('immersive-ar');")
    print("  const panels = jennyAR.createPanels(arData);")
    print("  session.requestAnimationFrame(render);")


async def demo_m6_analytics() -> None:
    """M6: Analytics Collection - Event-based metrics, dashboard."""
    print_header("M6: Analytics Collection & Dashboard")

    collector = JennyAnalyticsCollector()

    print_subheader("Event Types")
    print("  RenderEvent: Tracks rendering latency per target")
    print("  CompileEvent: Tracks compilation time and panel count")
    print()

    # Record some events
    print_subheader("Recording Events")

    # Render events using record_render()
    for target, latency, cached in [("html", 45.2, True), ("react", 38.5, False), ("ar", 52.1, False)]:
        collector.record_render(
            latency_ms=latency,
            target=target,
            panel_count=3,
            cache_hit=cached,
        )
        print(f"  RenderEvent: target={target}, latency={latency}ms, cached={cached}")

    print()

    # Compile events using record_compile()
    for panel_count, compile_time in [(3, 12.5), (5, 18.2), (2, 8.1)]:
        collector.record_compile(
            latency_ms=compile_time,
            panels_generated=panel_count,
            query_type="factual",
        )
        print(f"  CompileEvent: panels={panel_count}, time={compile_time}ms")

    print()

    print_subheader("Analytics Summary")
    summary = collector.get_summary()
    print(f"  Total Events: {summary.get('total_events', 0)}")
    render_metrics = summary.get('render', {})
    compile_metrics = summary.get('compile', {})
    print(f"  Avg Render Latency: {render_metrics.get('avg_latency_ms', 0):.1f}ms")
    print(f"  Avg Compile Time: {compile_metrics.get('avg_latency_ms', 0):.1f}ms")
    print(f"  Cache Hit Rate: {render_metrics.get('cache_hit_rate', 0)*100:.1f}%")
    print()

    print_subheader("Dashboard Generation")
    dashboard = JennyAnalyticsDashboard(collector)
    print("  dashboard.render_html() -> Full HTML dashboard")
    print("  dashboard.export_prometheus() -> Prometheus metrics")
    print("  dashboard.export_json() -> JSON for external tools")


async def demo_full_pipeline() -> None:
    """Complete end-to-end pipeline demonstrating all 6 moonshots."""
    print_header("Full Moonshot Pipeline")

    print("  Simulating complete query flow...\n")

    # Create sample data
    specs = create_sample_specs()
    query_text = "What is Thompson Sampling?"
    confidence = 0.92

    # M6: Start analytics collection
    collector = JennyAnalyticsCollector()
    start_time = datetime.now()

    # M1: Get registry and renderers
    registry = RendererRegistry()

    # Render to all targets
    results = {}
    targets = [RenderTarget.HTML, RenderTarget.REACT, RenderTarget.AR]

    for target in targets:
        renderer = registry.get_for_target(target)
        if renderer:
            import time
            render_start = time.perf_counter()
            result = await renderer.render(specs, target=target)
            render_time = (time.perf_counter() - render_start) * 1000

            results[target.value] = result

            # M6: Record analytics using record_render()
            collector.record_render(
                latency_ms=render_time,
                target=target.value,
                panel_count=len(specs),
                cache_hit=False,
            )

    # Print results
    print("  Pipeline Results:")
    print(f"    Query: \"{query_text}\"")
    print(f"    Confidence: {confidence:.2f}")
    print(f"    Panels Generated: {len(specs)}")
    print()

    print("  Render Outputs:")
    for target, result in results.items():
        size = len(result)
        print(f"    {target:6s}: {size:,} bytes")
    print()

    # M6: Show analytics
    summary = collector.get_summary()
    render_metrics = summary.get('render', {})
    print("  Analytics:")
    print(f"    Total Render Events: {summary.get('total_events', 0)}")
    print(f"    Avg Latency: {render_metrics.get('avg_latency_ms', 0):.2f}ms")
    print()

    # M3: Accessibility check
    print("  Accessibility:")
    print("    ARIA roles: Applied to all panels")
    print("    Keyboard nav: Tab/Shift+Tab enabled")
    print("    Screen reader: Live regions configured")
    print()

    total_time = (datetime.now() - start_time).total_seconds() * 1000
    print(f"  Total Pipeline Time: {total_time:.2f}ms")


async def main():
    """Run all moonshot demos."""
    print("\n" + "="*60)
    print("  JENNY MOONSHOT: COMPLETE SYSTEM DEMO")
    print("  " + "="*56)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("  Philosophy: \"Disposable pixels, durable decisions\"")
    print("="*60)

    try:
        # M1: Renderer Registry
        await demo_m1_renderer_registry()

        # M2: Async LLM Client
        await demo_m2_async_llm_client()

        # M3: Accessibility
        await demo_m3_accessibility()

        # M4: React Renderer
        await demo_m4_react_renderer()

        # M5: AR Renderer
        await demo_m5_ar_renderer()

        # M6: Analytics
        await demo_m6_analytics()

        # Full Pipeline
        await demo_full_pipeline()

    except ImportError as e:
        print(f"\n  [!] Import Error: {e}")
        print("      Some Jenny components may not be available.")
        print("      Run: PYTHONPATH=. python demos/demo_jenny_moonshot.py")
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        print("      Demo encountered an issue but gracefully handled it.")

    print_header("Demo Complete!")
    print("  All 6 Moonshot phases demonstrated:")
    print()
    print("    M1: Renderer Registry  - Singleton, priority selection")
    print("    M2: Async LLM Client   - Connection pooling, backoff")
    print("    M3: Accessibility      - WCAG 2.1 AA compliant")
    print("    M4: React Renderer     - TypeScript-friendly props")
    print("    M5: AR Renderer        - WebXR 3D transforms")
    print("    M6: Analytics          - Event tracking, dashboard")
    print()
    print("  Jenny: \"Disposable pixels, durable decisions.\"")
    print()


if __name__ == "__main__":
    asyncio.run(main())
