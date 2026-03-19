#!/usr/bin/env python3
"""
Tests for MCP Federation — adapter + router.

Proves:
1. Adapter wraps a department and handles tool_call/tool_list/health_check
2. Router routes to correct department, enforces permissions
3. Inter-department calls work (Verification → MasterWeaver)
4. HARD permission denials block, SOFT denials log and pass
5. Audit log captures cross-department calls
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from hololoom.protocols.department import (
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
    VerificationResult,
)
from hololoom.infrastructure.mcp.protocol import (
    MCPRequest,
    RequestType,
    ResponseStatus,
    ErrorCode,
    generate_request_id,
    generate_session_id,
)
from hololoom.alignment.mcp_department_registry import (
    DepartmentCharter,
    MCPTool,
    PermissionLevel,
    PermissionMode,
)
from hololoom.apps.departments.base import BaseDepartment
from hololoom.apps.departments.mcp.mcp_adapter import DepartmentMCPAdapter
from hololoom.apps.departments.mcp.mcp_router import MCPRouter


# ============================================================================
# Test fixtures
# ============================================================================


class StubDepartment(BaseDepartment):
    """Minimal department for testing."""

    def __init__(self, name: str = "TestDept"):
        config = DepartmentConfig(
            name=name,
            domain="testing",
            version="1.0.0",
            supported_tasks=["test_task", "extract_entities"],
            confidence_range=(0.5, 0.9),
        )
        super().__init__(
            name=name,
            domain="testing",
            version="1.0.0",
            supported_tasks=["test_task", "extract_entities"],
            confidence_range=(0.5, 0.9),
            config=config,
        )
        self.last_request = None

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        self.last_request = request
        return DepartmentResponse(
            task_id=request.task_id,
            result={"echo": request.parameters, "task_type": request.task_type},
            confidence=ConfidenceMetadata.from_score(0.85),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, summary="ok", confidence=0.9)

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return prior


def make_charter(
    name: str = "TestDept",
    tools: list | None = None,
    permissions: list | None = None,
) -> DepartmentCharter:
    """Build a minimal charter for testing."""
    if tools is None:
        tools = [
            MCPTool(
                name="test_task",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
                required_permissions=[PermissionLevel.READ],
                permission_mode=PermissionMode.HARD,
                returns={"type": "object"},
            ),
            MCPTool(
                name="extract_entities",
                description="Extract entities",
                parameters={"type": "object", "properties": {}},
                required_permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
                permission_mode=PermissionMode.HARD,
                returns={"type": "object"},
            ),
        ]
    return DepartmentCharter(
        name=name,
        role="Test department",
        capabilities=["testing"],
        tools=tools,
        permissions=permissions or [PermissionLevel.READ, PermissionLevel.WRITE],
        dependencies=[],
        session_state="session_id",
        context_budget=10000,
    )


def make_request(
    tool_name: str = "test_task",
    request_type: RequestType = RequestType.TOOL_CALL,
    parameters: dict | None = None,
) -> MCPRequest:
    return MCPRequest(
        request_id=generate_request_id(),
        session_id=generate_session_id(),
        request_type=request_type,
        tool_name=tool_name,
        parameters=parameters or {"key": "value"},
    )


# ============================================================================
# Adapter tests
# ============================================================================


class TestDepartmentMCPAdapter:
    @pytest.fixture
    def adapter(self):
        dept = StubDepartment()
        charter = make_charter()
        return DepartmentMCPAdapter(dept, charter)

    @pytest.mark.asyncio
    async def test_tool_call_success(self, adapter):
        req = make_request("test_task", parameters={"input": "hello"})
        resp = await adapter.handle(req)

        assert resp.status == ResponseStatus.SUCCESS
        assert resp.result["confidence"] == pytest.approx(0.85, abs=0.01)
        assert resp.result["result"]["echo"]["input"] == "hello"
        assert resp.metadata["department"] == "TestDept"

    @pytest.mark.asyncio
    async def test_tool_call_unknown_tool(self, adapter):
        req = make_request("nonexistent_tool")
        resp = await adapter.handle(req)

        assert resp.status == ResponseStatus.ERROR
        assert resp.error["code"] == ErrorCode.TOOL_NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_tool_list(self, adapter):
        req = make_request(request_type=RequestType.TOOL_LIST)
        resp = await adapter.handle(req)

        assert resp.status == ResponseStatus.SUCCESS
        tools = resp.result["tools"]
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert "test_task" in names
        assert "extract_entities" in names

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        req = make_request(request_type=RequestType.HEALTH_CHECK)
        resp = await adapter.handle(req)

        assert resp.status == ResponseStatus.SUCCESS
        assert resp.result["status"] == "healthy"
        assert resp.result["name"] == "TestDept"

    @pytest.mark.asyncio
    async def test_tool_call_records_metrics(self, adapter):
        req = make_request("test_task")
        await adapter.handle(req)

        dept = adapter.department
        assert dept._metrics["total_requests"] == 1
        assert dept._metrics["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_tool_call_exception_escalates(self):
        dept = StubDepartment()
        dept.execute = AsyncMock(side_effect=RuntimeError("boom"))
        charter = make_charter()
        adapter = DepartmentMCPAdapter(dept, charter)

        req = make_request("test_task")
        resp = await adapter.handle(req)

        assert resp.status == ResponseStatus.ESCALATE
        assert "boom" in resp.error["message"]
        assert resp.error["escalate_to"] == "Orchestration"


# ============================================================================
# Router tests
# ============================================================================


class TestMCPRouter:
    @pytest.fixture
    def router(self):
        router = MCPRouter()

        # Register two departments
        dept_a = StubDepartment("Alpha")
        charter_a = make_charter("Alpha")
        router.register(DepartmentMCPAdapter(dept_a, charter_a))

        dept_b = StubDepartment("Beta")
        charter_b = make_charter(
            "Beta",
            permissions=[PermissionLevel.READ],  # Beta only has READ
        )
        router.register(DepartmentMCPAdapter(dept_b, charter_b))

        return router

    @pytest.mark.asyncio
    async def test_route_to_named_department(self, router):
        req = make_request("test_task")
        resp = await router.handle(req, target="Alpha")

        assert resp.status == ResponseStatus.SUCCESS
        assert resp.metadata["department"] == "Alpha"

    @pytest.mark.asyncio
    async def test_route_unknown_department(self, router):
        req = make_request("test_task")
        resp = await router.handle(req, target="Nonexistent")

        assert resp.status == ResponseStatus.ERROR
        assert resp.error["code"] == ErrorCode.TOOL_NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_route_no_target(self, router):
        req = make_request("test_task")
        resp = await router.handle(req)

        assert resp.status == ResponseStatus.ERROR
        assert resp.error["code"] == ErrorCode.INVALID_REQUEST.value

    @pytest.mark.asyncio
    async def test_route_via_metadata(self, router):
        req = make_request("test_task")
        req.metadata["target_department"] = "Beta"
        resp = await router.handle(req)

        assert resp.status == ResponseStatus.SUCCESS
        assert resp.metadata["department"] == "Beta"

    @pytest.mark.asyncio
    async def test_call_tool_convenience(self, router):
        resp = await router.call_tool(
            target="Alpha",
            tool_name="test_task",
            parameters={"x": 1},
            session_id="sess_123",
        )

        assert resp.status == ResponseStatus.SUCCESS
        assert resp.result["result"]["echo"]["x"] == 1

    @pytest.mark.asyncio
    async def test_hard_permission_denied(self, router):
        """Beta only has READ. extract_entities requires READ + WRITE."""
        resp = await router.call_tool(
            target="Alpha",
            tool_name="extract_entities",
            parameters={},
            session_id="sess_123",
            caller="Beta",  # Beta lacks WRITE
        )

        assert resp.status == ResponseStatus.ERROR
        assert resp.error["code"] == ErrorCode.PERMISSION_DENIED.value
        assert "WRITE" in resp.error["message"].lower() or "write" in resp.error["message"]

    @pytest.mark.asyncio
    async def test_permission_allowed(self, router):
        """Alpha has READ + WRITE, so extract_entities should succeed."""
        resp = await router.call_tool(
            target="Beta",
            tool_name="extract_entities",
            parameters={},
            session_id="sess_123",
            caller="Alpha",  # Alpha has WRITE
        )

        assert resp.status == ResponseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_external_caller_gets_full_access(self, router):
        """External callers (None) = app layer, get all permissions."""
        # Can call tools requiring WRITE (extract_entities needs READ+WRITE)
        resp = await router.call_tool(
            target="Alpha",
            tool_name="extract_entities",
            parameters={},
            session_id="sess_123",
            caller=None,
        )
        assert resp.status == ResponseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_audit_log(self, router):
        await router.call_tool(
            target="Alpha",
            tool_name="test_task",
            parameters={},
            session_id="sess_123",
            caller="Beta",
        )

        log = router.get_audit_log()
        assert len(log) == 1
        assert log[0]["caller"] == "Beta"
        assert log[0]["target"] == "Alpha"
        assert log[0]["tool"] == "test_task"

    @pytest.mark.asyncio
    async def test_audit_log_filter(self, router):
        await router.call_tool(
            target="Alpha", tool_name="test_task",
            parameters={}, session_id="s1", caller="Beta",
        )
        await router.call_tool(
            target="Beta", tool_name="test_task",
            parameters={}, session_id="s2", caller="Alpha",
        )

        beta_log = router.get_audit_log(caller="Beta")
        assert len(beta_log) == 1
        assert beta_log[0]["target"] == "Alpha"

    @pytest.mark.asyncio
    async def test_broadcast_health(self, router):
        health = await router.broadcast_health("sess_health")

        assert "Alpha" in health
        assert "Beta" in health
        assert health["Alpha"]["status"] == "healthy"
        assert health["Beta"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_departments_property(self, router):
        assert set(router.departments) == {"Alpha", "Beta"}


# ============================================================================
# Soft permission tests
# ============================================================================


class TestSoftPermissions:
    @pytest.mark.asyncio
    async def test_soft_permission_logs_but_allows(self):
        """SOFT permission mode should log a warning but not block."""
        router = MCPRouter()

        dept = StubDepartment("Soft")
        charter = make_charter(
            "Soft",
            tools=[
                MCPTool(
                    name="soft_tool",
                    description="Tool with soft permissions",
                    parameters={"type": "object", "properties": {}},
                    required_permissions=[PermissionLevel.ADMIN],
                    permission_mode=PermissionMode.SOFT,  # SOFT = log, don't block
                    returns={"type": "object"},
                ),
            ],
            permissions=[PermissionLevel.READ],  # Only READ
        )
        router.register(DepartmentMCPAdapter(dept, charter))

        # Register a caller department that only has READ (no ADMIN)
        caller_dept = StubDepartment("LimitedCaller")
        caller_charter = make_charter(
            "LimitedCaller",
            permissions=[PermissionLevel.READ],  # Only READ, no ADMIN
        )
        router.register(DepartmentMCPAdapter(caller_dept, caller_charter))

        # LimitedCaller lacks ADMIN, but tool is SOFT — should log and allow
        resp = await router.call_tool(
            target="Soft",
            tool_name="soft_tool",
            parameters={},
            session_id="sess_soft",
            caller="LimitedCaller",
        )

        # Should succeed despite lacking ADMIN (SOFT mode)
        assert resp.status == ResponseStatus.SUCCESS


# ============================================================================
# Inter-department workflow test
# ============================================================================


class TestInterDepartmentWorkflow:
    @pytest.mark.asyncio
    async def test_verification_calls_masterweaver(self):
        """
        End-to-end: Verification department calls MasterWeaver to re-extract
        entities with stricter parameters. This is the canonical cross-
        department flow from the architecture doc.
        """
        router = MCPRouter()

        # Set up MasterWeaver
        mw_dept = StubDepartment("MasterWeaver")
        mw_charter = make_charter(
            "MasterWeaver",
            permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
        )
        router.register(DepartmentMCPAdapter(mw_dept, mw_charter))

        # Set up Verification
        ver_dept = StubDepartment("Verification")
        ver_charter = make_charter(
            "Verification",
            permissions=[PermissionLevel.READ, PermissionLevel.EXECUTE],
        )
        router.register(DepartmentMCPAdapter(ver_dept, ver_charter))

        # Verification calls MasterWeaver.extract_entities
        session_id = "session_rerun_001"
        resp = await router.call_tool(
            target="MasterWeaver",
            tool_name="extract_entities",
            parameters={
                "input_data": "Queen spotted near entrance",
                "domain": "beekeeping",
                "confidence_threshold": 0.9,
            },
            session_id=session_id,
            caller="Verification",
        )

        # Verification has READ + EXECUTE, extract_entities needs READ + WRITE
        # Verification lacks WRITE → HARD deny
        assert resp.status == ResponseStatus.ERROR
        assert resp.error["code"] == ErrorCode.PERMISSION_DENIED.value

        # Now give Verification WRITE permission and retry
        ver_charter_v2 = make_charter(
            "Verification",
            permissions=[
                PermissionLevel.READ,
                PermissionLevel.EXECUTE,
                PermissionLevel.WRITE,
            ],
        )
        router.register(DepartmentMCPAdapter(ver_dept, ver_charter_v2))

        resp2 = await router.call_tool(
            target="MasterWeaver",
            tool_name="extract_entities",
            parameters={
                "input_data": "Queen spotted near entrance",
                "domain": "beekeeping",
                "confidence_threshold": 0.9,
            },
            session_id=session_id,
            caller="Verification",
        )

        assert resp2.status == ResponseStatus.SUCCESS
        assert resp2.result["confidence"] == pytest.approx(0.85, abs=0.01)

        # MasterWeaver received the right parameters
        assert mw_dept.last_request.parameters["domain"] == "beekeeping"

        # Audit log shows only the successful attempt
        # (denied requests are blocked before dispatch/audit)
        log = router.get_audit_log(caller="Verification")
        assert len(log) == 1
        assert log[0]["target"] == "MasterWeaver"
