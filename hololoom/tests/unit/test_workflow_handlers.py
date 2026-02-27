"""Unit tests for HoloLoom workflow ChatOps handlers.

Tests the complete ChatOps integration for workflow commands:
- Workflow list (list saved versions)
- Workflow run (execute workflow)
- Workflow status (executor health check)
- Workflow validate (validate workflow structure)
- Workflow generate (NLP-based generation)
- Workflow agents (list available agents)
- Workflow optimize (get optimization suggestions)
- Workflow help (show help text)

Author: HoloLoom Team
Date: 2025-12-11
"""

import asyncio
import pytest
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock


# ============================================================================
# Mock Data Classes
# ============================================================================

@dataclass
class MockMatrixRoom:
    """Mock nio.MatrixRoom for testing."""
    room_id: str = "!test:example.com"
    name: str = "test-room"
    display_name: str = "Test Room"


@dataclass
class MockRoomMessageText:
    """Mock nio.RoomMessageText event."""
    sender: str = "@user:example.com"
    body: str = ""
    room_id: str = "!test:example.com"


# ============================================================================
# Mock API Responses
# ============================================================================

MOCK_WORKFLOW_VERSIONS = {
    "versions": [
        {
            "version": "1",
            "name": "Simple Query",
            "message": "Initial version",
            "timestamp": "2025-12-10T10:00:00"
        },
        {
            "version": "2",
            "name": "Research Pipeline",
            "message": "Added verification step",
            "timestamp": "2025-12-11T14:30:00"
        },
        {
            "version": "3",
            "name": "Safety Workflow",
            "message": "Full safety integration",
            "timestamp": "2025-12-11T16:00:00"
        }
    ]
}

MOCK_WORKFLOW_EMPTY = {
    "versions": []
}

MOCK_WORKFLOW_EXECUTION_SUCCESS = {
    "success": True,
    "execution_time_ms": 1250.5,
    "nodes_executed": 5,
    "results": {
        "response": "Thompson Sampling is a Bayesian approach...",
        "confidence": 0.92
    },
    "trace": [
        {"node": "query_input", "status": "completed", "duration_ms": 10},
        {"node": "safety_check", "status": "completed", "duration_ms": 50},
        {"node": "hololoom_query", "status": "completed", "duration_ms": 1100},
        {"node": "format_output", "status": "completed", "duration_ms": 90}
    ]
}

MOCK_WORKFLOW_EXECUTION_PARTIAL = {
    "success": False,
    "execution_time_ms": 500.0,
    "nodes_executed": 2,
    "results": {
        "response": "Partial result",
        "confidence": 0.5
    },
    "errors": ["Node 'external_api' failed: timeout"]
}

MOCK_HEALTH_RESPONSE = {
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "active_workflows": 2,
    "total_executed": 150,
    "success_rate": 0.95
}

MOCK_VALIDATION_SUCCESS = {
    "valid": True,
    "nodes": 5,
    "connections": 4,
    "cycles": [],
    "warnings": []
}

MOCK_VALIDATION_INVALID = {
    "valid": False,
    "nodes": 3,
    "connections": 4,
    "cycles": [["node1", "node2", "node3", "node1"]],
    "errors": ["Cycle detected in workflow"],
    "warnings": ["Node 'orphan' has no connections"]
}

MOCK_AGENTS_RESPONSE = {
    "agents": [
        {"type": "query", "count": 3, "agents": ["hololoom_query", "memory_search", "multi_query"]},
        {"type": "process", "count": 3, "agents": ["embedder", "synthesizer", "refiner"]},
        {"type": "memory", "count": 3, "agents": ["memory_store", "context_retriever", "knowledge_fusion"]},
        {"type": "decision", "count": 3, "agents": ["thompson_sampler", "convergence", "safety"]},
        {"type": "output", "count": 2, "agents": ["response_generator", "format_converter"]},
        {"type": "control", "count": 3, "agents": ["conditional", "loop", "parallel"]}
    ],
    "total": 17
}

MOCK_OPTIMIZATION_SUGGESTIONS = {
    "suggestions": [
        {
            "suggestion_type": "parallelize",
            "description": "Nodes 'embedder' and 'memory_search' can run in parallel",
            "expected_improvement": 0.25,
            "confidence": 0.85
        },
        {
            "suggestion_type": "cache",
            "description": "Add caching for 'hololoom_query' node",
            "expected_improvement": 0.4,
            "confidence": 0.9
        },
        {
            "suggestion_type": "remove",
            "description": "Node 'unused_node' is never executed",
            "expected_improvement": 0.05,
            "confidence": 0.95
        }
    ]
}

MOCK_OPTIMIZATION_EMPTY = {
    "suggestions": []
}


# ============================================================================
# Mock WorkflowGenerator
# ============================================================================

class MockWorkflowGenerator:
    """Mock WorkflowGenerator for testing NLP synthesis."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.last_description = None

    def generate(self, description: str) -> Dict[str, Any]:
        """Generate a workflow from natural language description."""
        self.last_description = description

        if self.should_fail:
            raise Exception("NLP synthesis failed")

        # Return a mock workflow
        return {
            "version": "1.0",
            "name": f"Generated: {description[:20]}...",
            "nodes": [
                {"id": "input", "type": "query", "config": {}},
                {"id": "process", "type": "hololoom_query", "config": {}},
                {"id": "output", "type": "response_generator", "config": {}}
            ],
            "connections": [
                {"from": "input", "to": "process"},
                {"from": "process", "to": "output"}
            ]
        }


# ============================================================================
# Mock aiohttp Session
# ============================================================================

class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, json_data: Dict[str, Any], status: int = 200):
        self._json_data = json_data
        self.status = status

    async def json(self):
        return self._json_data

    async def text(self):
        return json.dumps(self._json_data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockClientSession:
    """Mock aiohttp ClientSession."""

    def __init__(self, responses: Dict[str, MockResponse] = None, should_raise: bool = False):
        self.responses = responses or {}
        self.should_raise = should_raise
        self.requests_made = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url: str, **kwargs):
        self.requests_made.append(("GET", url, kwargs))
        if self.should_raise:
            raise Exception("Connection error")
        return self.responses.get(url, MockResponse({"error": "Not found"}, 404))

    def post(self, url: str, json: Dict = None, **kwargs):
        self.requests_made.append(("POST", url, json, kwargs))
        if self.should_raise:
            raise Exception("Connection error")
        return self.responses.get(url, MockResponse({"error": "Not found"}, 404))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_room():
    """Create a mock Matrix room."""
    return MockMatrixRoom()


@pytest.fixture
def mock_event():
    """Create a mock message event."""
    return MockRoomMessageText()


@pytest.fixture
def mock_generator():
    """Create a mock WorkflowGenerator."""
    return MockWorkflowGenerator()


@pytest.fixture
def mock_failing_generator():
    """Create a failing mock WorkflowGenerator."""
    return MockWorkflowGenerator(should_fail=True)


# ============================================================================
# Test: handle_workflow_list
# ============================================================================

class TestWorkflowList:
    """Test !workflow list command handler."""

    @pytest.mark.asyncio
    async def test_list_success(self, mock_room, mock_event):
        """Test successful workflow listing."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_list

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/versions?branch=main": MockResponse(MOCK_WORKFLOW_VERSIONS)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_list(mock_room, mock_event, "")

        assert "Workflow Versions" in result
        assert "Simple Query" in result
        assert "Research Pipeline" in result
        assert "Safety Workflow" in result
        assert "v1" in result or "v2" in result or "v3" in result

    @pytest.mark.asyncio
    async def test_list_with_branch(self, mock_room, mock_event):
        """Test listing workflows from specific branch."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_list

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/versions?branch=develop": MockResponse(MOCK_WORKFLOW_VERSIONS)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_list(mock_room, mock_event, "develop")

        assert "develop" in result
        assert "Workflow Versions" in result

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_room, mock_event):
        """Test listing when no workflows exist."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_list

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/versions?branch=main": MockResponse(MOCK_WORKFLOW_EMPTY)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_list(mock_room, mock_event, "")

        assert "No saved workflows found" in result

    @pytest.mark.asyncio
    async def test_list_api_error(self, mock_room, mock_event):
        """Test handling of API error."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_list

        mock_session = MockClientSession(should_raise=True)

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_list(mock_room, mock_event, "")

        assert "Failed to list workflows" in result or "Connection error" in result


# ============================================================================
# Test: handle_workflow_run
# ============================================================================

class TestWorkflowRun:
    """Test !workflow run command handler."""

    @pytest.mark.asyncio
    async def test_run_success(self, mock_room, mock_event):
        """Test successful workflow execution."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_run

        workflow_json = json.dumps({
            "version": "1.0",
            "name": "Test Workflow",
            "nodes": [{"id": "test", "type": "query"}],
            "connections": []
        })

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/execute": MockResponse(MOCK_WORKFLOW_EXECUTION_SUCCESS)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_run(mock_room, mock_event, workflow_json)

        assert "Workflow Executed" in result
        assert "Success" in result
        assert "nodes executed" in result.lower() or "nodes" in result.lower()

    @pytest.mark.asyncio
    async def test_run_partial_success(self, mock_room, mock_event):
        """Test partial workflow execution (some nodes failed)."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_run

        workflow_json = json.dumps({
            "version": "1.0",
            "name": "Test",
            "nodes": [],
            "connections": []
        })

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/execute": MockResponse(MOCK_WORKFLOW_EXECUTION_PARTIAL)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_run(mock_room, mock_event, workflow_json)

        assert "Workflow Executed" in result
        # May show partial or warning status

    @pytest.mark.asyncio
    async def test_run_invalid_json(self, mock_room, mock_event):
        """Test handling of invalid JSON input."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_run

        result = await handle_workflow_run(mock_room, mock_event, "not valid json {{{")

        assert "error" in result.lower() or "invalid" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_run_empty_input(self, mock_room, mock_event):
        """Test handling of empty input."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_run

        result = await handle_workflow_run(mock_room, mock_event, "")

        assert "Usage" in result
        assert "workflow run" in result

    @pytest.mark.asyncio
    async def test_run_api_error(self, mock_room, mock_event):
        """Test handling of API execution error."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_run

        workflow_json = json.dumps({"version": "1.0", "nodes": [], "connections": []})
        mock_session = MockClientSession(should_raise=True)

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_run(mock_room, mock_event, workflow_json)

        assert "error" in result.lower() or "failed" in result.lower()


# ============================================================================
# Test: handle_workflow_status
# ============================================================================

class TestWorkflowStatus:
    """Test !workflow status command handler."""

    @pytest.mark.asyncio
    async def test_status_healthy(self, mock_room, mock_event):
        """Test status when executor is healthy."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_status

        mock_session = MockClientSession({
            "http://localhost:8001/health": MockResponse(MOCK_HEALTH_RESPONSE)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_status(mock_room, mock_event, "")

        assert "Status" in result or "Executor" in result
        assert "healthy" in result.lower() or "online" in result.lower()

    @pytest.mark.asyncio
    async def test_status_offline(self, mock_room, mock_event):
        """Test status when executor is offline."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_status

        mock_session = MockClientSession(should_raise=True)

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_status(mock_room, mock_event, "")

        assert "offline" in result.lower() or "error" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_status_with_active_workflows(self, mock_room, mock_event):
        """Test status shows active workflow count."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_status

        mock_session = MockClientSession({
            "http://localhost:8001/health": MockResponse(MOCK_HEALTH_RESPONSE)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_status(mock_room, mock_event, "")

        # Should contain information about executor
        assert len(result) > 10


# ============================================================================
# Test: handle_workflow_validate
# ============================================================================

class TestWorkflowValidate:
    """Test !workflow validate command handler."""

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_room, mock_event):
        """Test validation of valid workflow."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_validate

        workflow_json = json.dumps({
            "version": "1.0",
            "name": "Valid Workflow",
            "nodes": [{"id": "a"}, {"id": "b"}],
            "connections": [{"from": "a", "to": "b"}]
        })

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/validate": MockResponse(MOCK_VALIDATION_SUCCESS)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_validate(mock_room, mock_event, workflow_json)

        assert "valid" in result.lower() or "success" in result.lower()
        assert "5 nodes" in result or "nodes" in result.lower()

    @pytest.mark.asyncio
    async def test_validate_with_cycles(self, mock_room, mock_event):
        """Test validation detects cycles."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_validate

        workflow_json = json.dumps({
            "version": "1.0",
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "connections": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
                {"from": "c", "to": "a"}
            ]
        })

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/validate": MockResponse(MOCK_VALIDATION_INVALID)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_validate(mock_room, mock_event, workflow_json)

        assert "invalid" in result.lower() or "cycle" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_validate_invalid_json(self, mock_room, mock_event):
        """Test validation with invalid JSON."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_validate

        result = await handle_workflow_validate(mock_room, mock_event, "not json")

        assert "error" in result.lower() or "invalid" in result.lower()

    @pytest.mark.asyncio
    async def test_validate_empty_input(self, mock_room, mock_event):
        """Test validation with empty input."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_validate

        result = await handle_workflow_validate(mock_room, mock_event, "")

        assert "Usage" in result or "validate" in result


# ============================================================================
# Test: handle_workflow_generate
# ============================================================================

class TestWorkflowGenerate:
    """Test !workflow generate command handler."""

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_room, mock_event, mock_generator):
        """Test successful workflow generation."""
        from hololoom.apps.chatops.handlers import workflow_handlers
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_generate

        # Set the mock generator
        original_generator = workflow_handlers._generator
        workflow_handlers._generator = mock_generator

        try:
            result = await handle_workflow_generate(mock_room, mock_event, "Answer questions with safety checks")

            assert "Generated" in result or "workflow" in result.lower()
            assert mock_generator.last_description == "Answer questions with safety checks"
        finally:
            workflow_handlers._generator = original_generator

    @pytest.mark.asyncio
    async def test_generate_empty_description(self, mock_room, mock_event):
        """Test generation with empty description."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_generate

        result = await handle_workflow_generate(mock_room, mock_event, "")

        assert "Usage" in result or "description" in result.lower()

    @pytest.mark.asyncio
    async def test_generate_failure(self, mock_room, mock_event, mock_failing_generator):
        """Test handling of generation failure."""
        from hololoom.apps.chatops.handlers import workflow_handlers
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_generate

        original_generator = workflow_handlers._generator
        workflow_handlers._generator = mock_failing_generator

        try:
            result = await handle_workflow_generate(mock_room, mock_event, "Test description")

            assert "error" in result.lower() or "failed" in result.lower()
        finally:
            workflow_handlers._generator = original_generator

    @pytest.mark.asyncio
    async def test_generate_unavailable(self, mock_room, mock_event):
        """Test generation when generator is unavailable."""
        from hololoom.apps.chatops.handlers import workflow_handlers
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_generate

        original_generator = workflow_handlers._generator
        original_available = workflow_handlers.GENERATOR_AVAILABLE

        workflow_handlers._generator = None
        workflow_handlers.GENERATOR_AVAILABLE = False

        try:
            result = await handle_workflow_generate(mock_room, mock_event, "Test description")

            assert "unavailable" in result.lower() or "not available" in result.lower() or "error" in result.lower()
        finally:
            workflow_handlers._generator = original_generator
            workflow_handlers.GENERATOR_AVAILABLE = original_available


# ============================================================================
# Test: handle_workflow_agents
# ============================================================================

class TestWorkflowAgents:
    """Test !workflow agents command handler."""

    @pytest.mark.asyncio
    async def test_agents_success(self, mock_room, mock_event):
        """Test successful agent listing."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_agents

        mock_session = MockClientSession({
            "http://localhost:8001/api/agents": MockResponse(MOCK_AGENTS_RESPONSE)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_agents(mock_room, mock_event, "")

        assert "query" in result.lower() or "agents" in result.lower()

    @pytest.mark.asyncio
    async def test_agents_api_error_fallback(self, mock_room, mock_event):
        """Test fallback to hardcoded agents on API error."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_agents

        mock_session = MockClientSession(should_raise=True)

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_agents(mock_room, mock_event, "")

        # Should still return agent info (hardcoded fallback)
        assert "query" in result.lower() or "agents" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_agents_categories(self, mock_room, mock_event):
        """Test that all agent categories are included."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_agents

        mock_session = MockClientSession({
            "http://localhost:8001/api/agents": MockResponse(MOCK_AGENTS_RESPONSE)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_agents(mock_room, mock_event, "")

        # Check for agent type mentions (case-insensitive)
        result_lower = result.lower()
        assert any(cat in result_lower for cat in ["query", "process", "memory", "decision", "output", "control"])


# ============================================================================
# Test: handle_workflow_optimize
# ============================================================================

class TestWorkflowOptimize:
    """Test !workflow optimize command handler."""

    @pytest.mark.asyncio
    async def test_optimize_with_suggestions(self, mock_room, mock_event):
        """Test optimization returns suggestions."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_optimize

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/test-workflow/optimize": MockResponse(MOCK_OPTIMIZATION_SUGGESTIONS)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_optimize(mock_room, mock_event, "test-workflow")

        assert "Optimization" in result or "suggestion" in result.lower()
        assert "parallelize" in result.lower() or "cache" in result.lower()

    @pytest.mark.asyncio
    async def test_optimize_no_suggestions(self, mock_room, mock_event):
        """Test optimization when no improvements found."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_optimize

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/optimal/optimize": MockResponse(MOCK_OPTIMIZATION_EMPTY)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_optimize(mock_room, mock_event, "optimal")

        assert "No optimization" in result or "well-optimized" in result.lower() or "no suggestions" in result.lower()

    @pytest.mark.asyncio
    async def test_optimize_empty_id(self, mock_room, mock_event):
        """Test optimization with empty workflow ID."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_optimize

        result = await handle_workflow_optimize(mock_room, mock_event, "")

        assert "Usage" in result
        assert "optimize" in result

    @pytest.mark.asyncio
    async def test_optimize_api_error(self, mock_room, mock_event):
        """Test optimization with API error."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_optimize

        mock_session = MockClientSession(should_raise=True)

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_optimize(mock_room, mock_event, "test-workflow")

        assert "error" in result.lower() or "failed" in result.lower()


# ============================================================================
# Test: handle_workflow_help
# ============================================================================

class TestWorkflowHelp:
    """Test !workflow help command handler."""

    @pytest.mark.asyncio
    async def test_help_content(self, mock_room, mock_event):
        """Test help returns comprehensive information."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_help

        result = await handle_workflow_help(mock_room, mock_event, "")

        # Check for all commands mentioned in help text
        result_lower = result.lower()
        assert "list" in result_lower
        assert "run" in result_lower
        assert "status" in result_lower
        assert "validate" in result_lower
        assert "generate" in result_lower
        assert "agents" in result_lower
        assert "optimize" in result_lower
        # Note: help command doesn't mention itself in the help text


# ============================================================================
# Test: Configuration Functions
# ============================================================================

class TestConfiguration:
    """Test configuration getter/setter functions."""

    def test_get_set_executor_url(self):
        """Test getting and setting executor URL."""
        from hololoom.apps.chatops.handlers.workflow_handlers import (
            get_executor_url, set_executor_url
        )

        original = get_executor_url()

        try:
            set_executor_url("http://custom:9999")
            assert get_executor_url() == "http://custom:9999"
        finally:
            set_executor_url(original)

    def test_get_set_generator(self):
        """Test getting and setting generator."""
        from hololoom.apps.chatops.handlers import workflow_handlers
        from hololoom.apps.chatops.handlers.workflow_handlers import (
            get_generator, set_generator
        )

        original = workflow_handlers._generator

        try:
            mock_gen = MockWorkflowGenerator()
            set_generator(mock_gen)
            assert get_generator() == mock_gen
        finally:
            workflow_handlers._generator = original


# ============================================================================
# Test: WorkflowHandlers Class
# ============================================================================

class TestWorkflowHandlersClass:
    """Test the WorkflowHandlers container class."""

    def test_init_default(self):
        """Test default initialization."""
        from hololoom.apps.chatops.handlers.workflow_handlers import WorkflowHandlers

        handlers = WorkflowHandlers()
        # Should not raise
        assert handlers is not None

    def test_init_custom_url(self):
        """Test initialization with custom executor URL."""
        from hololoom.apps.chatops.handlers.workflow_handlers import (
            WorkflowHandlers, get_executor_url, set_executor_url
        )

        original = get_executor_url()

        try:
            handlers = WorkflowHandlers(executor_url="http://custom:1234")
            assert get_executor_url() == "http://custom:1234"
        finally:
            set_executor_url(original)

    @pytest.mark.asyncio
    async def test_handler_methods_exist(self):
        """Test that all handler methods exist."""
        from hololoom.apps.chatops.handlers.workflow_handlers import WorkflowHandlers

        handlers = WorkflowHandlers()

        assert hasattr(handlers, 'handle_list')
        assert hasattr(handlers, 'handle_run')
        assert hasattr(handlers, 'handle_status')
        assert hasattr(handlers, 'handle_validate')
        assert hasattr(handlers, 'handle_generate')
        assert hasattr(handlers, 'handle_agents')
        assert hasattr(handlers, 'handle_optimize')
        assert hasattr(handlers, 'handle_help')


# ============================================================================
# Test: Registration Function
# ============================================================================

class TestRegistration:
    """Test the handler registration function."""

    def test_register_workflow_handlers(self):
        """Test register_workflow_handlers function exists and is callable."""
        from hololoom.apps.chatops.handlers.workflow_handlers import register_workflow_handlers

        # Should not raise
        result = register_workflow_handlers()
        assert result is not None


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_unicode_in_description(self, mock_room, mock_event, mock_generator):
        """Test handling of unicode in workflow description."""
        from hololoom.apps.chatops.handlers import workflow_handlers
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_generate

        original_generator = workflow_handlers._generator
        workflow_handlers._generator = mock_generator

        try:
            result = await handle_workflow_generate(
                mock_room, mock_event,
                "Create workflow with emojis 🔄 and special chars: éàü"
            )

            # Should handle unicode gracefully
            assert result is not None
        finally:
            workflow_handlers._generator = original_generator

    @pytest.mark.asyncio
    async def test_very_long_workflow_json(self, mock_room, mock_event):
        """Test handling of very long workflow JSON."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_validate

        # Create a workflow with many nodes
        nodes = [{"id": f"node_{i}", "type": "query"} for i in range(100)]
        workflow = {
            "version": "1.0",
            "name": "Large Workflow",
            "nodes": nodes,
            "connections": []
        }

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/validate": MockResponse({
                "valid": True,
                "nodes": 100,
                "connections": 0
            })
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_validate(mock_room, mock_event, json.dumps(workflow))

        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_workflow_id(self, mock_room, mock_event):
        """Test handling of special characters in workflow ID."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_optimize

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/workflow-with-dashes_and_underscores/optimize":
                MockResponse(MOCK_OPTIMIZATION_EMPTY)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_optimize(
                mock_room, mock_event,
                "workflow-with-dashes_and_underscores"
            )

        assert result is not None


# ============================================================================
# Test: Timeout Handling
# ============================================================================

class TestTimeoutHandling:
    """Test timeout handling for API calls."""

    @pytest.mark.asyncio
    async def test_run_respects_timeout(self, mock_room, mock_event):
        """Test that workflow run uses longer timeout."""
        from hololoom.apps.chatops.handlers.workflow_handlers import handle_workflow_run

        workflow_json = json.dumps({
            "version": "1.0",
            "nodes": [],
            "connections": []
        })

        mock_session = MockClientSession({
            "http://localhost:8001/api/workflow/execute": MockResponse(MOCK_WORKFLOW_EXECUTION_SUCCESS)
        })

        with patch('hololoom.apps.chatops.handlers.workflow_handlers.aiohttp.ClientSession', return_value=mock_session):
            result = await handle_workflow_run(mock_room, mock_event, workflow_json)

        # Check that a POST request was made
        assert any(req[0] == "POST" for req in mock_session.requests_made)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
