"""
HoloLoom Visualization
======================
Self-constructing dashboards from Spacetime fabrics.

Like Wolfram Alpha: Every query auto-generates its optimal visualization.

Ruthlessly Elegant API:
    from hololoom.visualization import auto

    # One function. Zero configuration. Perfect dashboard.
    dashboard = auto(spacetime)
    dashboard = auto({'month': [...], 'value': [...]})
    dashboard = auto(memory_backend)
"""

# Ruthlessly Elegant Primary API
from .auto import auto, render, save

# Core Types (for advanced usage)
from .dashboard import (
    Panel, Dashboard, PanelType, PanelSize, LayoutType, ComplexityLevel
)

# Intelligent Builder (auto uses this internally)
from .widget_builder import WidgetBuilder, DataAnalyzer, InsightGenerator

# Legacy/Advanced APIs (prefer auto() instead)
from .html_renderer import HTMLRenderer
from .orchestrator_integration import DashboardOrchestrator, weave_and_visualize

# Jenny Generative UI Runtime (December 2025 - MVP Week 1)
# Philosophy: "Disposable pixels, durable decisions"
from .jenny_spec import (
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
from .jenny_compiler import JennyCompiler, QueryAnalysis
from .spec_ledger import SpecLedger, SpecLedgerEntry

# Jenny Generative UI Runtime (December 2025 - MVP Week 2)
# Renderers: Convert JennySpec → output (HTML, Terminal, JSON)
from .jenny_renderer import (
    JennyRendererBase,
    HTMLRenderer as JennyHTMLRenderer,
    TerminalRenderer,
    JSONRenderer,
    RenderTarget,
    create_renderer,
    get_default_renderer,
)

# Lifecycle Manager: Track panel state machine (NASCENT → STABLE → DISSOLVING → ARCHIVED)
from .jenny_lifecycle import (
    JennyLifecycleManagerBase,
    InMemoryLifecycleManager,
    PanelState,
    PanelNotFoundError,
    InvalidTransitionError,
    create_lifecycle_manager,
    get_default_lifecycle_manager,
    run_cleanup_loop,
)

# Jenny Generative UI Runtime (December 2025 - MVP Week 3)
# Action Handler: Execute panel actions (PIN, DISMISS, WHY, etc.) with provenance
from .jenny_actions import (
    ActionStatus,
    ActionResult,
    ActionHandlerProtocol,
    JennyActionHandler,
    create_action_handler,
)

# Streaming Manager: Real-time data updates for reactive/streaming bindings
from .jenny_streaming import (
    StreamStatus,
    UpdateType,
    StreamUpdate,
    Subscription,
    DataSourceProtocol,
    MockDataSource,
    StreamingManager,
    create_streaming_manager,
)

# Jenny Generative UI Runtime (December 2025 - MVP Week 4)
# JennyRuntime: Unified orchestrator bringing all components together
from .jenny_runtime import (
    JennyConfig,
    JennyPanel,
    JennyRuntime,
    create_runtime,
    jenny_session,
)

# Jenny MRF Integration (December 2025 - Phase B+C)
# MRF-enhanced compilation with Thompson Sampling learning
from .jenny_mrf import (
    JennyMRFCompiler,
    PanelTypeLearner,
    PanelTypePrior,
    ACTION_LEARNING_MAP,
    create_mrf_compiler,
    create_learning_callback,
)

# Jenny LLM Integration (December 2025 - Phase A)
# LLM-based panel compilation with graceful fallback chain
from .jenny_llm_compiler import (
    LLMJennyCompiler,
    SemanticProfile,
    LLMPanelSelection,
    LLM_OUTPUT_SCHEMA,
    create_llm_compiler,
    create_compiler_with_fallback,
    parse_llm_response,
    analyze_semantic_axes,
)

__all__ = [
    # Primary API (ruthlessly simple)
    'auto',           # THE function - visualize anything
    'render',         # Dashboard -> HTML
    'save',           # Dashboard -> file

    # Core types
    'Panel',
    'Dashboard',
    'PanelType',
    'PanelSize',
    'LayoutType',
    'ComplexityLevel',

    # Intelligent engine
    'WidgetBuilder',
    'DataAnalyzer',
    'InsightGenerator',

    # Advanced/Legacy
    'HTMLRenderer',
    'DashboardOrchestrator',
    'weave_and_visualize',

    # Jenny Generative UI Runtime
    'JennySpec',
    'LifecycleStage',
    'BindingMode',
    'DissolutionTrigger',
    'PanelTypeJenny',
    'PanelSizeJenny',
    'LayoutHint',
    'create_action',
    'get_default_actions',
    'ACTION_PIN',
    'ACTION_DISMISS',
    'ACTION_WHY',
    'JennyCompiler',
    'QueryAnalysis',
    'SpecLedger',
    'SpecLedgerEntry',

    # Jenny Week 2: Renderers
    'JennyRendererBase',
    'JennyHTMLRenderer',
    'TerminalRenderer',
    'JSONRenderer',
    'RenderTarget',
    'create_renderer',
    'get_default_renderer',

    # Jenny Week 2: Lifecycle Manager
    'JennyLifecycleManagerBase',
    'InMemoryLifecycleManager',
    'PanelState',
    'PanelNotFoundError',
    'InvalidTransitionError',
    'create_lifecycle_manager',
    'get_default_lifecycle_manager',
    'run_cleanup_loop',

    # Jenny Week 3: Action Handler
    'ActionStatus',
    'ActionResult',
    'ActionHandlerProtocol',
    'JennyActionHandler',
    'create_action_handler',

    # Jenny Week 3: Streaming Manager
    'StreamStatus',
    'UpdateType',
    'StreamUpdate',
    'Subscription',
    'DataSourceProtocol',
    'MockDataSource',
    'StreamingManager',
    'create_streaming_manager',

    # Jenny Week 4: Unified Runtime
    'JennyConfig',
    'JennyPanel',
    'JennyRuntime',
    'create_runtime',
    'jenny_session',

    # Jenny MRF Integration (Phase B+C)
    'JennyMRFCompiler',
    'PanelTypeLearner',
    'PanelTypePrior',
    'ACTION_LEARNING_MAP',
    'create_mrf_compiler',
    'create_learning_callback',

    # Jenny LLM Integration (Phase A)
    'LLMJennyCompiler',
    'SemanticProfile',
    'LLMPanelSelection',
    'LLM_OUTPUT_SCHEMA',
    'create_llm_compiler',
    'create_compiler_with_fallback',
    'parse_llm_response',
    'analyze_semantic_axes',
]
