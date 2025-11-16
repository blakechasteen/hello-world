"""
HoloLoom Promptly Integration

Extends Promptly with workflow-specific storage and management capabilities.
Includes DSPy integration for systematic prompt engineering.
"""


# DSPy integration (optional, graceful degradation if not installed)
try:
    from .dspy_bridge import (
        DSPyHoloLoom,
        DSPySignature,
        DSPyProgram,
        create_signature,
        QuestionAnswering,
        MemorySynthesis,
        ReasoningChain
    )
    from .dspy_workflow_adapter import (
        DSPyWorkflowAdapter,
        DSPyWorkflow,
        DSPyWorkflowStep,
        create_qa_workflow,
        create_research_workflow
    )

    __all__ = [
        "WorkflowStore",
        # DSPy Bridge
        "DSPyHoloLoom",
        "DSPySignature",
        "DSPyProgram",
        "create_signature",
        "QuestionAnswering",
        "MemorySynthesis",
        "ReasoningChain",
        # Workflow Adapter
        "DSPyWorkflowAdapter",
        "DSPyWorkflow",
        "DSPyWorkflowStep",
        "create_qa_workflow",
        "create_research_workflow"
    ]

    DSPY_AVAILABLE = True

except ImportError:
    __all__ = ["WorkflowStore"]
    DSPY_AVAILABLE = False
