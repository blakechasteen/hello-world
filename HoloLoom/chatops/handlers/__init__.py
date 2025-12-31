"""
HoloLoom ChatOps - Advanced Handlers
=====================================

**113 Total Commands** (as of December 2025)

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
- feedback_handler: Reaction-based Thompson Sampling feedback
- ingestion_handlers: SpinningWheel data ingestion commands
- websocket_progress: WebSocket-based job progress streaming
- prometheus_metrics: Prometheus metrics for job observability
- conversation_handlers: Multi-turn conversation support (!continue, !context)
- cluster_handlers: Eggroll distributed cluster management (!cluster)
- alignment_handlers: Safety guardrails and audit trail commands (!safety, !audit)
- agent_manager_ws: WebSocket handlers for Agent Manager UI (thread lifecycle, progress, priority)
- learning_handlers: Thompson Sampling, hot patterns, refinement, reflection (!learn)
- awareness_handlers: Activation, spring dynamics, brain waves, meta-awareness (!aware)
- context_handlers: Context packing, adaptive expansion, streaming (!context pack/expand/stream)
- semantic_handlers: Phase 8 semantic state -> policy integration
    - !semantic <query> - Analyze semantic state (244D -> 8D)
    - !semantic shift - Show topic shift detection
    - !semantic tools [q] - Show semantic tool suggestions
    - !semantic bandit [q] - Show Thompson Sampling adjustments
    - !semantic compare <q1> | <q2> - Compare two queries semantically
    - !semantic replay [limit] - Replay conversation semantic trajectory
    - !semantic suggest - Suggest next action based on context
    - !semantic help - Show semantic commands help
- multi_agent_semantic: Phase 7 multi-agent semantic coordination
    - !agents list - Show registered agents with status
    - !agents topics - Show cross-room topic threads
    - !agents handoff <agent_id> - Initiate context handoff
    - !agents link <thread_id> - Link room to topic thread
    - !agents discover [query] - Discover related topics
    - !agents stats - Show coordination statistics
    - !agents help - Show multi-agent help
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
    from HoloLoom.chatops.handlers.feedback_handler import (
        FeedbackProcessor,
        BotMessageTracker,
        ThompsonFeedbackUpdater,
        get_feedback_processor,
        set_feedback_processor,
        handle_reaction_event,
        FeedbackHandlers,
        handle_feedback_stats,
        handle_feedback_process,
        handle_feedback_help,
        register_feedback_handlers
    )
    __all__.extend([
        "FeedbackProcessor",
        "BotMessageTracker",
        "ThompsonFeedbackUpdater",
        "get_feedback_processor",
        "set_feedback_processor",
        "handle_reaction_event",
        "FeedbackHandlers",
        "handle_feedback_stats",
        "handle_feedback_process",
        "handle_feedback_help",
        "register_feedback_handlers"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.ingestion_handlers import (
        register_ingestion_handlers,
        IngestionHandlers,
        SourceDetection,
        detect_source_type,
        handle_ingest,
        handle_ingest_youtube,
        handle_ingest_pdf,
        handle_ingest_url,
        handle_ingest_git,
        handle_ingest_image,
        handle_ingest_status,
        handle_ingest_help
    )
    __all__.extend([
        "register_ingestion_handlers",
        "IngestionHandlers",
        "SourceDetection",
        "detect_source_type",
        "handle_ingest",
        "handle_ingest_youtube",
        "handle_ingest_pdf",
        "handle_ingest_url",
        "handle_ingest_git",
        "handle_ingest_image",
        "handle_ingest_status",
        "handle_ingest_help"
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

try:
    from HoloLoom.chatops.handlers.conversation_handlers import (
        register_conversation_handlers,
        ConversationHandlers,
        SessionManager,
        UserSessionState,
        get_session_manager,
        handle_continue,
        handle_context,
        handle_conversation_help
    )
    __all__.extend([
        "register_conversation_handlers",
        "ConversationHandlers",
        "SessionManager",
        "UserSessionState",
        "get_session_manager",
        "handle_continue",
        "handle_context",
        "handle_conversation_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.cluster_handlers import (
        register_cluster_handlers,
        ClusterHandlers,
        ClusterManager,
        NodeInfo,
        ClusterInfo,
        NodeStatus,
        get_cluster_manager,
        set_cluster_manager,
        handle_cluster_status,
        handle_cluster_nodes,
        handle_cluster_balance,
        handle_cluster_help
    )
    __all__.extend([
        "register_cluster_handlers",
        "ClusterHandlers",
        "ClusterManager",
        "NodeInfo",
        "ClusterInfo",
        "NodeStatus",
        "get_cluster_manager",
        "set_cluster_manager",
        "handle_cluster_status",
        "handle_cluster_nodes",
        "handle_cluster_balance",
        "handle_cluster_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.alignment_handlers import (
        register_alignment_handlers,
        AlignmentHandlers,
        get_guardrails,
        set_guardrails,
        get_audit_trail,
        set_audit_trail,
        handle_safety_check,
        handle_safety_stats,
        handle_safety_history,
        handle_audit_log,
        handle_audit_trace,
        handle_audit_search,
        handle_alignment_help
    )
    __all__.extend([
        "register_alignment_handlers",
        "AlignmentHandlers",
        "get_guardrails",
        "set_guardrails",
        "get_audit_trail",
        "set_audit_trail",
        "handle_safety_check",
        "handle_safety_stats",
        "handle_safety_history",
        "handle_audit_log",
        "handle_audit_trace",
        "handle_audit_search",
        "handle_alignment_help"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.agent_manager_ws import (
        AgentManagerMessageType,
        AgentManagerMessage,
        AgentManagerWSManager,
        AgentManagerBroadcaster,
        create_agent_manager_router,
        get_global_manager as get_global_ws_manager,
        get_global_broadcaster
    )
    __all__.extend([
        "AgentManagerMessageType",
        "AgentManagerMessage",
        "AgentManagerWSManager",
        "AgentManagerBroadcaster",
        "create_agent_manager_router",
        "get_global_ws_manager",
        "get_global_broadcaster"
    ])
except ImportError:
    pass

try:
    from HoloLoom.chatops.handlers.workflow_handlers import (
        register_workflow_handlers,
        WorkflowHandlers,
        get_executor_url,
        set_executor_url,
        get_generator,
        set_generator,
        handle_workflow_list,
        handle_workflow_run,
        handle_workflow_status,
        handle_workflow_validate,
        handle_workflow_generate,
        handle_workflow_agents,
        handle_workflow_optimize,
        handle_workflow_help
    )
    __all__.extend([
        "register_workflow_handlers",
        "WorkflowHandlers",
        "get_executor_url",
        "set_executor_url",
        "get_generator",
        "set_generator",
        "handle_workflow_list",
        "handle_workflow_run",
        "handle_workflow_status",
        "handle_workflow_validate",
        "handle_workflow_generate",
        "handle_workflow_agents",
        "handle_workflow_optimize",
        "handle_workflow_help"
    ])
except ImportError:
    pass

# Learning handlers (December 2025) - Thompson Sampling, hot patterns, refinement
try:
    from HoloLoom.chatops.handlers.learning_handlers import (
        register_learning_handlers,
        LearningHandlers,
        get_learning_engine,
        set_learning_engine,
        get_hot_tracker,
        set_hot_tracker,
        get_reflection_buffer,
        set_reflection_buffer,
        handle_learn_stats,
        handle_learn_tool,
        handle_learn_hot,
        handle_learn_cold,
        handle_learn_refine,
        handle_learn_patterns,
        handle_learn_reflect,
        handle_learn_help
    )
    __all__.extend([
        "register_learning_handlers",
        "LearningHandlers",
        "get_learning_engine",
        "set_learning_engine",
        "get_hot_tracker",
        "set_hot_tracker",
        "get_reflection_buffer",
        "set_reflection_buffer",
        "handle_learn_stats",
        "handle_learn_tool",
        "handle_learn_hot",
        "handle_learn_cold",
        "handle_learn_refine",
        "handle_learn_patterns",
        "handle_learn_reflect",
        "handle_learn_help"
    ])
except ImportError:
    pass

# Awareness handlers (December 2025) - Activation, spring dynamics, brain waves
try:
    from HoloLoom.chatops.handlers.awareness_handlers import (
        register_awareness_handlers,
        AwarenessHandlers,
        get_awareness_graph,
        set_awareness_graph,
        get_spring_dynamics,
        set_spring_dynamics,
        get_wave_engine,
        set_wave_engine,
        handle_aware_metrics,
        handle_aware_activate,
        handle_aware_spring,
        handle_aware_wave_mode,
        handle_aware_consolidate,
        handle_aware_confidence,
        handle_aware_gaps,
        handle_aware_help
    )
    __all__.extend([
        "register_awareness_handlers",
        "AwarenessHandlers",
        "get_awareness_graph",
        "set_awareness_graph",
        "get_spring_dynamics",
        "set_spring_dynamics",
        "get_wave_engine",
        "set_wave_engine",
        "handle_aware_metrics",
        "handle_aware_activate",
        "handle_aware_spring",
        "handle_aware_wave_mode",
        "handle_aware_consolidate",
        "handle_aware_confidence",
        "handle_aware_gaps",
        "handle_aware_help"
    ])
except ImportError:
    pass

# Context handlers (December 2025) - Context packing, adaptive expansion, streaming
try:
    from HoloLoom.chatops.handlers.context_handlers import (
        register_context_handlers,
        ContextHandlers,
        get_context_packer,
        set_context_packer,
        get_activation_spreader,
        set_activation_spreader,
        get_importance_scorer,
        set_importance_scorer,
        get_knowledge_graph,
        set_knowledge_graph,
        handle_context_pack,
        handle_context_budget,
        handle_context_expand,
        handle_context_stream,
        handle_context_spread,
        handle_context_importance,
        handle_context_stats,
        handle_context_help
    )
    __all__.extend([
        "register_context_handlers",
        "ContextHandlers",
        "get_context_packer",
        "set_context_packer",
        "get_activation_spreader",
        "set_activation_spreader",
        "get_importance_scorer",
        "set_importance_scorer",
        "get_knowledge_graph",
        "set_knowledge_graph",
        "handle_context_pack",
        "handle_context_budget",
        "handle_context_expand",
        "handle_context_stream",
        "handle_context_spread",
        "handle_context_importance",
        "handle_context_stats",
        "handle_context_help"
    ])
except ImportError:
    pass

# Semantic handlers (December 2025) - Phase 8: Semantic State -> Policy integration
try:
    from HoloLoom.chatops.handlers.semantic_handlers import (
        SemanticMatrixHandlers,
        ConversationSemanticContext,
        create_semantic_handlers,
        get_semantic_handlers,
        set_semantic_handlers
    )
    __all__.extend([
        "SemanticMatrixHandlers",
        "ConversationSemanticContext",
        "create_semantic_handlers",
        "get_semantic_handlers",
        "set_semantic_handlers"
    ])
except ImportError:
    pass

# Multi-agent semantic coordination (December 2025) - Phase 7
try:
    from HoloLoom.chatops.handlers.multi_agent_semantic import (
        SharedSemanticRegistry,
        AgentInfo,
        AgentStatus,
        TopicThread,
        HandoffContext,
        MultiAgentHandlers,
        get_semantic_registry,
        register_multi_agent_handlers,
        notify_semantic_update
    )
    __all__.extend([
        "SharedSemanticRegistry",
        "AgentInfo",
        "AgentStatus",
        "TopicThread",
        "HandoffContext",
        "MultiAgentHandlers",
        "get_semantic_registry",
        "register_multi_agent_handlers",
        "notify_semantic_update"
    ])
except ImportError:
    pass

# Voice handlers (December 2025) - Voice-to-code via Matrix voice messages
try:
    from HoloLoom.chatops.handlers.voice_handler import (
        VoiceHandler,
        TranscriptionResult,
        VoiceCommandResult
    )
    __all__.extend([
        "VoiceHandler",
        "TranscriptionResult",
        "VoiceCommandResult"
    ])
except ImportError:
    pass

# Code voice grammar (December 2025) - Patterns for voice-to-code commands
try:
    from HoloLoom.chatops.handlers.code_voice_grammar import (
        CodeVoiceGrammar,
        CodeIntent,
        CodeCommandIntent
    )
    __all__.extend([
        "CodeVoiceGrammar",
        "CodeIntent",
        "CodeCommandIntent"
    ])
except ImportError:
    pass