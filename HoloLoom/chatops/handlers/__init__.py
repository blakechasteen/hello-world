"""
HoloLoom ChatOps - Advanced Handlers
=====================================

Core components:
- handler_registry: Decorator-based handler registration with auto-help

Optional advanced features:
- multimodal_handler: Image and file processing
- thread_handler: Thread-aware responses
- proactive_agent: Proactive suggestions
- hololoom_handlers: HoloLoom-specific handlers
- redteam_handlers: Red team and adversarial testing handlers
- pattern_tuning: Pattern optimization
- visualization_handlers: Dashboard and chart commands
- memory_symphony_handlers: Memory coordination and strategy commands
- temporal_handlers: Time-travel and temporal pattern queries
- department_handlers: Multi-department coordination and routing
- websocket_progress: WebSocket-based job progress streaming
- prometheus_metrics: Prometheus metrics for job observability
"""

# Core handler registry - always available
from HoloLoom.chatops.handlers.handler_registry import (
    HandlerRegistry,
    HandlerCategory,
    HandlerSpec,
    chatops_handler,
    get_global_registry,
    register_handlers_from
)

# Graceful imports - these are optional
__all__ = [
    "HandlerRegistry",
    "HandlerCategory",
    "HandlerSpec",
    "chatops_handler",
    "get_global_registry",
    "register_handlers_from"
]

try:
    from HoloLoom.chatops.handlers.multimodal_handler import MultimodalHandler
    __all__.append("MultimodalHandler")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.thread_handler import ThreadHandler
    __all__.append("ThreadHandler")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.proactive_agent import ProactiveAgent
    __all__.append("ProactiveAgent")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.hololoom_handlers import HoloLoomMatrixHandlers
    __all__.append("HoloLoomMatrixHandlers")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.redteam_handlers import RedTeamMatrixHandlers
    __all__.append("RedTeamMatrixHandlers")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.pattern_tuning import PatternTuner
    __all__.append("PatternTuner")
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.test_handlers import (
        register_test_handlers,
        TestRunner,
        TestStatusTracker,
        TestHandlers,
        handle_test_run,
        handle_test_status,
        handle_test_coverage,
        handle_test_benchmark,
        handle_test_ci,
        handle_test_help
    )
    __all__.extend([
        "register_test_handlers",
        "TestRunner",
        "TestStatusTracker",
        "TestHandlers",
        "handle_test_run",
        "handle_test_status",
        "handle_test_coverage",
        "handle_test_benchmark",
        "handle_test_ci",
        "handle_test_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.code_handlers import (
        register_code_handlers,
        CodeHandlers,
        handle_code_query,
        handle_code_refactor,
        handle_code_explain,
        handle_code_test,
        handle_code_fix,
        handle_code_context,
        handle_code_status,
        handle_code_help
    )
    __all__.extend([
        "register_code_handlers",
        "CodeHandlers",
        "handle_code_query",
        "handle_code_refactor",
        "handle_code_explain",
        "handle_code_test",
        "handle_code_fix",
        "handle_code_context",
        "handle_code_status",
        "handle_code_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.rag_handlers import (
        register_rag_handlers,
        RAGHandlers,
        handle_rag_query,
        handle_rag_ingest,
        handle_rag_search,
        handle_rag_stats,
        handle_rag_help
    )
    __all__.extend([
        "register_rag_handlers",
        "RAGHandlers",
        "handle_rag_query",
        "handle_rag_ingest",
        "handle_rag_search",
        "handle_rag_stats",
        "handle_rag_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.agentic_handlers import (
        register_agentic_handlers,
        AgenticHandlers,
        handle_research,
        handle_verify,
        handle_plan,
        handle_reason,
        handle_agentic_help
    )
    __all__.extend([
        "register_agentic_handlers",
        "AgenticHandlers",
        "handle_research",
        "handle_verify",
        "handle_plan",
        "handle_reason",
        "handle_agentic_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.visualization_handlers import (
        register_visualization_handlers,
        VisualizationHandlers,
        SessionMetrics,
        get_session_metrics,
        handle_dashboard_confidence,
        handle_dashboard_cache,
        handle_dashboard_waterfall,
        handle_dashboard_knowledge,
        handle_dashboard_rag,
        handle_dashboard_help,
        handle_dashboard_reset
    )
    __all__.extend([
        "register_visualization_handlers",
        "VisualizationHandlers",
        "SessionMetrics",
        "get_session_metrics",
        "handle_dashboard_confidence",
        "handle_dashboard_cache",
        "handle_dashboard_waterfall",
        "handle_dashboard_knowledge",
        "handle_dashboard_rag",
        "handle_dashboard_help",
        "handle_dashboard_reset"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.memory_symphony_handlers import (
        register_memory_symphony_handlers,
        MemorySymphonyHandlers,
        handle_memory_strategy,
        handle_memory_metrics,
        handle_memory_systems,
        handle_memory_history,
        handle_memory_help
    )
    __all__.extend([
        "register_memory_symphony_handlers",
        "MemorySymphonyHandlers",
        "handle_memory_strategy",
        "handle_memory_metrics",
        "handle_memory_systems",
        "handle_memory_history",
        "handle_memory_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.temporal_handlers import (
        register_temporal_handlers,
        TemporalHandlers,
        handle_temporal_travel,
        handle_temporal_between,
        handle_temporal_patterns,
        handle_temporal_help
    )
    __all__.extend([
        "register_temporal_handlers",
        "TemporalHandlers",
        "handle_temporal_travel",
        "handle_temporal_between",
        "handle_temporal_patterns",
        "handle_temporal_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.department_handlers import (
        register_department_handlers,
        DepartmentHandlers,
        handle_dept_list,
        handle_dept_status,
        handle_dept_process,
        handle_dept_capabilities,
        handle_dept_help
    )
    __all__.extend([
        "register_department_handlers",
        "DepartmentHandlers",
        "handle_dept_list",
        "handle_dept_status",
        "handle_dept_process",
        "handle_dept_capabilities",
        "handle_dept_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.websocket_progress import (
        JobMessageType,
        JobProgressMessage,
        JobProgressManager,
        JobProgressBroadcaster,
        create_progress_router
    )
    __all__.extend([
        "JobMessageType",
        "JobProgressMessage",
        "JobProgressManager",
        "JobProgressBroadcaster",
        "create_progress_router"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.prometheus_metrics import (
        JobMetricsCollector,
        MetricType,
        MetricValue,
        LatencyHistogram,
        create_metrics_router,
        get_metrics_collector,
        set_metrics_collector
    )
    __all__.extend([
        "JobMetricsCollector",
        "MetricType",
        "MetricValue",
        "LatencyHistogram",
        "create_metrics_router",
        "get_metrics_collector",
        "set_metrics_collector"
    ])
except ImportError:
    pass
