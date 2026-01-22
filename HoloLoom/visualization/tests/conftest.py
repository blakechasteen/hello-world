"""
Jenny Visualization Runtime Test Fixtures
==========================================
Shared pytest fixtures for Jenny visualization tests.

Date: January 2026
Author: Claude Code

Provides:
- JennySpec fixtures (various panel types, lifecycle states)
- Mock Spacetime objects
- Renderer instances
- Registry helpers
- Thompson Sampling learner fixtures
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Import Jenny modules
from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LayoutHint,
    create_action,
    create_spec,
    get_default_actions,
    ACTION_PIN,
    ACTION_DISMISS,
    ACTION_WHY,
    ACTION_EXPAND,
    ACTION_COPY,
    DEFAULT_ACTIONS,
)
from HoloLoom.visualization.jenny_enums import (
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
    ActionStatus,
)


# ============================================================================
# Event Loop Fixture
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Spacetime Fixtures
# ============================================================================

@dataclass
class MockWeavingTrace:
    """Mock WeavingTrace for testing."""
    threads_activated: List[str]
    stage_durations: Dict[str, float]
    duration_ms: float

    def __init__(
        self,
        threads_activated: Optional[List[str]] = None,
        stage_durations: Optional[Dict[str, float]] = None,
        duration_ms: float = 150.0,
    ):
        self.threads_activated = threads_activated or ["thread1", "thread2"]
        self.stage_durations = stage_durations or {
            "retrieval": 50.0,
            "decision": 30.0,
            "execution": 70.0,
        }
        self.duration_ms = duration_ms


@dataclass
class MockSpacetime:
    """Mock Spacetime for testing."""
    query_text: str
    response: str
    confidence: float
    trace: Optional[MockWeavingTrace]
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def __init__(
        self,
        query_text: str = "What is Thompson Sampling?",
        response: str = "Thompson Sampling is a Bayesian approach...",
        confidence: float = 0.85,
        trace: Optional[MockWeavingTrace] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.query_text = query_text
        self.response = response
        self.confidence = confidence
        self.trace = trace or MockWeavingTrace()
        self.sources = sources or [
            {"id": "src1", "text": "Source 1 text", "relevance": 0.9},
            {"id": "src2", "text": "Source 2 text", "relevance": 0.8},
        ]
        self.metadata = metadata or {"tool_used": "answer"}


@pytest.fixture
def mock_spacetime() -> MockSpacetime:
    """Create a basic mock Spacetime."""
    return MockSpacetime()


@pytest.fixture
def mock_spacetime_high_confidence() -> MockSpacetime:
    """Create a mock Spacetime with high confidence."""
    return MockSpacetime(confidence=0.95)


@pytest.fixture
def mock_spacetime_low_confidence() -> MockSpacetime:
    """Create a mock Spacetime with low confidence."""
    return MockSpacetime(confidence=0.45)


@pytest.fixture
def mock_spacetime_with_sources() -> MockSpacetime:
    """Create a mock Spacetime with multiple sources."""
    return MockSpacetime(
        sources=[
            {"id": f"src{i}", "text": f"Source {i} content", "relevance": 0.9 - i * 0.1}
            for i in range(5)
        ]
    )


@pytest.fixture
def mock_spacetime_no_trace() -> MockSpacetime:
    """Create a mock Spacetime without trace."""
    return MockSpacetime(trace=None)


# ============================================================================
# JennySpec Fixtures
# ============================================================================

@pytest.fixture
def text_spec() -> JennySpec:
    """Create a TEXT panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.TEXT,
        title="Response",
        content={"text": "This is the response text."},
        size=PanelSizeJenny.MEDIUM,
        priority=0,
    )


@pytest.fixture
def graph_spec() -> JennySpec:
    """Create a GRAPH panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.GRAPH,
        title="Knowledge Graph",
        content={
            "nodes": [
                {"id": "n1", "label": "Node 1"},
                {"id": "n2", "label": "Node 2"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "type": "RELATES_TO"},
            ],
        },
        size=PanelSizeJenny.LARGE,
        priority=1,
    )


@pytest.fixture
def confidence_spec() -> JennySpec:
    """Create a CONFIDENCE panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.CONFIDENCE,
        title="Confidence",
        content={"value": 0.85, "label": "High confidence"},
        size=PanelSizeJenny.SMALL,
        priority=2,
    )


@pytest.fixture
def sources_spec() -> JennySpec:
    """Create a SOURCES panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.SOURCES,
        title="Sources",
        content={
            "sources": [
                {"id": "src1", "text": "Source 1", "relevance": 0.9},
                {"id": "src2", "text": "Source 2", "relevance": 0.8},
            ]
        },
        size=PanelSizeJenny.MEDIUM,
        priority=3,
    )


@pytest.fixture
def code_spec() -> JennySpec:
    """Create a CODE panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.CODE,
        title="Code Example",
        content={
            "code": "def hello():\n    return 'world'",
            "language": "python",
        },
        size=PanelSizeJenny.MEDIUM,
        priority=1,
    )


@pytest.fixture
def table_spec() -> JennySpec:
    """Create a TABLE panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.TABLE,
        title="Data Table",
        content={
            "headers": ["Name", "Value", "Score"],
            "rows": [
                ["Item 1", "A", 0.9],
                ["Item 2", "B", 0.8],
            ],
        },
        size=PanelSizeJenny.LARGE,
        priority=1,
    )


@pytest.fixture
def metric_spec() -> JennySpec:
    """Create a METRIC panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.METRIC,
        title="Latency",
        content={"value": "150ms", "delta": "-10%", "trend": "down"},
        size=PanelSizeJenny.SMALL,
        priority=2,
    )


@pytest.fixture
def why_spec() -> JennySpec:
    """Create a WHY panel spec."""
    return create_spec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.WHY,
        title="Why This UI?",
        content={
            "explanation": "These panels were selected based on your query type.",
            "query_type": "factual",
        },
        size=PanelSizeJenny.SMALL,
        priority=99,
        lifecycle=LifecycleStage.SYSTEM,
    )


@pytest.fixture
def reactive_spec() -> JennySpec:
    """Create a reactive binding spec."""
    return JennySpec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.METRIC,
        title="Live Metric",
        content={"value": "0"},
        size=PanelSizeJenny.SMALL,
        binding_mode=BindingMode.REACTIVE,
        data_source="metrics://cpu_usage",
        refresh_interval_ms=1000,
    )


@pytest.fixture
def streaming_spec() -> JennySpec:
    """Create a streaming binding spec."""
    return JennySpec(
        spacetime_id="st-001",
        panel_type=PanelTypeJenny.TEXT,
        title="Live Log",
        content={"text": ""},
        size=PanelSizeJenny.MEDIUM,
        binding_mode=BindingMode.STREAMING,
        data_source="ws://localhost:8080/log-stream",
    )


@pytest.fixture
def all_panel_types() -> List[PanelTypeJenny]:
    """List of all panel types."""
    return list(PanelTypeJenny)


@pytest.fixture
def all_lifecycle_stages() -> List[LifecycleStage]:
    """List of all lifecycle stages."""
    return list(LifecycleStage)


@pytest.fixture
def all_binding_modes() -> List[BindingMode]:
    """List of all binding modes."""
    return list(BindingMode)


@pytest.fixture
def spec_list(text_spec, graph_spec, confidence_spec) -> List[JennySpec]:
    """Create a list of specs for batch operations."""
    return [text_spec, graph_spec, confidence_spec]


# ============================================================================
# Renderer Fixtures
# ============================================================================

@pytest.fixture
def html_renderer():
    """Create HTMLRenderer instance."""
    from HoloLoom.visualization.jenny_renderer import HTMLRenderer
    return HTMLRenderer()


@pytest.fixture
def terminal_renderer():
    """Create TerminalRenderer instance."""
    from HoloLoom.visualization.jenny_renderer import TerminalRenderer
    return TerminalRenderer()


@pytest.fixture
def json_renderer():
    """Create JSONRenderer instance."""
    from HoloLoom.visualization.jenny_renderer import JSONRenderer
    return JSONRenderer()


@pytest.fixture
def react_renderer():
    """Create ReactRenderer instance."""
    from HoloLoom.visualization.jenny_renderer import ReactRenderer
    return ReactRenderer()


@pytest.fixture
def ar_renderer():
    """Create ARRenderer instance."""
    from HoloLoom.visualization.jenny_renderer import ARRenderer
    return ARRenderer()


# ============================================================================
# Registry Fixtures
# ============================================================================

@pytest.fixture
def fresh_registry():
    """Create a fresh registry instance by clearing singleton."""
    from HoloLoom.visualization.jenny_renderer_registry import RendererRegistry, RendererRegistryMeta

    # Clear the singleton
    RendererRegistryMeta._instance = None

    # Create fresh registry
    registry = RendererRegistry()
    yield registry

    # Clean up - clear again for other tests
    RendererRegistryMeta._instance = None


@pytest.fixture
def populated_registry(fresh_registry, html_renderer):
    """Create a registry with some renderers registered."""
    from HoloLoom.visualization.jenny_renderer import (
        HTMLRenderer,
        TerminalRenderer,
        JSONRenderer,
    )

    fresh_registry.register(HTMLRenderer, priority=10)
    fresh_registry.register(TerminalRenderer, priority=5)
    fresh_registry.register(JSONRenderer, priority=8)

    return fresh_registry


# ============================================================================
# Learner Fixtures
# ============================================================================

@pytest.fixture
def panel_learner():
    """Create a PanelTypeLearner instance."""
    from HoloLoom.visualization.jenny_mrf import PanelTypeLearner
    return PanelTypeLearner()


@pytest.fixture
def panel_learner_with_history():
    """Create a learner with some history."""
    from HoloLoom.visualization.jenny_mrf import PanelTypeLearner

    learner = PanelTypeLearner()

    # Simulate some learning history
    learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)
    learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.85)
    learner.update("factual", PanelTypeJenny.CONFIDENCE, success=True, confidence=0.7)
    learner.update("analytical", PanelTypeJenny.GRAPH, success=True, confidence=0.8)
    learner.update("analytical", PanelTypeJenny.GRAPH, success=False, confidence=0.3)

    return learner


@pytest.fixture
def temp_persist_path(tmp_path):
    """Create a temporary path for learner persistence."""
    return str(tmp_path / "learner_state.json")


# ============================================================================
# Runtime Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value="Generated response")
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_compiler():
    """Create a mock Jenny compiler."""
    compiler = AsyncMock()
    compiler.compile = AsyncMock()
    return compiler


# ============================================================================
# Action Fixtures
# ============================================================================

@pytest.fixture
def sample_actions() -> tuple:
    """Create sample actions for testing."""
    return (
        create_action("Pin", "pin_panel", action_type="toggle"),
        create_action("Dismiss", "dismiss_panel"),
        create_action("Copy", "copy_content"),
    )


@pytest.fixture
def action_with_confirmation():
    """Create an action that requires confirmation."""
    return create_action(
        "Delete",
        "delete_panel",
        requires_confirmation=True,
        params={"permanent": True},
    )


# ============================================================================
# Helper Functions
# ============================================================================

def assert_valid_html(html: str) -> None:
    """Assert that string is valid HTML."""
    assert html is not None
    assert isinstance(html, str)
    assert len(html) > 0
    # Basic structure checks
    assert "<" in html
    assert ">" in html


def assert_valid_json(json_str: str) -> Dict[str, Any]:
    """Assert that string is valid JSON and return parsed."""
    import json
    assert json_str is not None
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    return data


def assert_aria_compliant(html: str) -> None:
    """Assert that HTML has basic ARIA compliance."""
    # Check for role attribute
    assert 'role=' in html or 'aria-' in html, "HTML should have ARIA attributes"


def create_test_spec(
    panel_type: PanelTypeJenny = PanelTypeJenny.TEXT,
    lifecycle: LifecycleStage = LifecycleStage.NASCENT,
    binding_mode: BindingMode = BindingMode.STATIC,
    **kwargs
) -> JennySpec:
    """Helper to create test specs with defaults."""
    defaults = {
        "spacetime_id": "test-st-001",
        "title": f"Test {panel_type.value}",
        "content": {"text": "Test content"},
        "size": PanelSizeJenny.MEDIUM,
        "priority": 0,
    }
    defaults.update(kwargs)

    return JennySpec(
        panel_type=panel_type,
        lifecycle=lifecycle,
        binding_mode=binding_mode,
        **defaults,
    )
