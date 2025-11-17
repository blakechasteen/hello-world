#!/usr/bin/env python3
"""
HoloLoom Workflows Package
===========================

Complete workflow system for building complex multi-agent pipelines.

Features:
- Custom agent registry (extensible)
- Domain-specific templates (RAG, code review, data processing, etc.)
- AI-powered workflow generation (natural language → workflow)
- Real-time analytics dashboard
- Collaborative editing (multi-user)
- Git-like versioning

Usage:
    from HoloLoom.workflows import (
        get_registry,
        WorkflowTemplates,
        AIWorkflowGenerator,
        WorkflowAnalytics,
        CollaborationManager
    )

    # Get agent registry
    registry = get_registry()

    # Create workflow from template
    templates = WorkflowTemplates()
    workflow = templates.get('rag_research')

    # Generate workflow from natural language
    generator = AIWorkflowGenerator()
    workflow = await generator.generate(
        "Create a workflow that analyzes code for security issues"
    )

    # Track analytics
    analytics = WorkflowAnalytics()
    analytics.track_execution(execution_data)
    html = analytics.generate_dashboard(workflow_id)

    # Collaborative editing
    manager = CollaborationManager()
    session = manager.get_or_create_session(workflow_id)
"""

__version__ = "1.0.0"

# Core exports
from HoloLoom.workflows.agent_registry import (
    AgentRegistry,
    AgentCategory,
    AgentSchema,
    AgentConfig,
    get_registry
)

# Import from templates.py module (not the templates/ package directory)
import sys
import os
# Temporarily add the workflows directory to import the templates.py file directly
_templates_file = os.path.join(os.path.dirname(__file__), 'templates.py')
if os.path.exists(_templates_file):
    import importlib.util
    spec = importlib.util.spec_from_file_location("workflow_templates_module", _templates_file)
    templates_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(templates_module)
    WorkflowTemplates = templates_module.WorkflowTemplates
    WorkflowTemplate = templates_module.WorkflowTemplate
    TemplateCategory = templates_module.TemplateCategory
else:
    # Fallback if file doesn't exist
    WorkflowTemplates = None
    WorkflowTemplate = None
    TemplateCategory = None

from HoloLoom.workflows.ai_generator import (
    AIWorkflowGenerator,
    WorkflowIntent
)

from HoloLoom.workflows.analytics import (
    WorkflowAnalytics,
    WorkflowMetrics,
    NodeMetrics,
    ExecutionRecord
)

from HoloLoom.workflows.collaborative import (
    CollaborativeSession,
    CollaborationManager,
    Operation,
    OperationType,
    UserPresence
)

__all__ = [
    # Registry
    'AgentRegistry',
    'AgentCategory',
    'AgentSchema',
    'AgentConfig',
    'get_registry',

    # Templates
    'WorkflowTemplates',
    'WorkflowTemplate',
    'TemplateCategory',

    # AI Generation
    'AIWorkflowGenerator',
    'WorkflowIntent',

    # Analytics
    'WorkflowAnalytics',
    'WorkflowMetrics',
    'NodeMetrics',
    'ExecutionRecord',

    # Collaboration
    'CollaborativeSession',
    'CollaborationManager',
    'Operation',
    'OperationType',
    'UserPresence',
]


# Initialize global instances
_registry = None
_templates = None
_collaboration_manager = None
_analytics = None


def init_workflows():
    """Initialize workflow system with all components"""
    global _registry, _templates, _collaboration_manager, _analytics

    if _registry is None:
        _registry = get_registry()

    if _templates is None:
        _templates = WorkflowTemplates()

    if _collaboration_manager is None:
        _collaboration_manager = CollaborationManager()

    if _analytics is None:
        _analytics = WorkflowAnalytics()

    return {
        'registry': _registry,
        'templates': _templates,
        'collaboration': _collaboration_manager,
        'analytics': _analytics
    }


def get_workflow_system():
    """Get initialized workflow system components"""
    return init_workflows()
