"""
Integration tests for real department implementations.

Tests the RealInfrastructure, RealMasterWeaver, and RealVerification
departments through the MCP federation, verifying that charter tool
names route correctly to the real subsystem backends.

These tests run without external dependencies (Neo4j, Qdrant, Ollama)
by relying on graceful degradation — departments return lower-confidence
results when backends are unavailable, but never crash.
"""

import pytest

pytest.importorskip("fastapi")

import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from hololoom.apps.server.department_api import router, _get_router
from hololoom.apps.departments.mcp.bootstrap import (
    create_federation,
    create_real_federation,
)
from hololoom.apps.departments.mcp.real_departments import (
    RealInfrastructure,
    RealMasterWeaver,
    RealVerification,
    RealOrchestration,
    RealContext,
    RealExecution,
)


# ============================================================================
# Direct department tests (no HTTP, no adapter)
# ============================================================================


@pytest.fixture
def infra():
    return RealInfrastructure()


@pytest.fixture
def weaver():
    return RealMasterWeaver(enable_llm=False)


@pytest.fixture
def verifier():
    return RealVerification()


@pytest.fixture
def orchestrator():
    return RealOrchestration()


@pytest.fixture
def context():
    return RealContext()


@pytest.fixture
def execution():
    return RealExecution()


@pytest.mark.asyncio
async def test_infra_diagnose_performance(infra):
    """Infrastructure diagnose_performance returns health data."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="diagnose_performance",
        parameters={},
    )
    resp = await infra.execute(req)
    assert resp.confidence.score > 0
    # Should have store_stats or health_score in result
    assert "health_score" in resp.result or "error" not in resp.result


@pytest.mark.asyncio
async def test_infra_neo4j_degraded(infra):
    """Infrastructure query_neo4j degrades gracefully without Neo4j."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="query_neo4j",
        parameters={"cypher_query": "MATCH (n) RETURN n LIMIT 5"},
    )
    resp = await infra.execute(req)
    # Should return empty results with degradation note, not crash
    assert "results" in resp.result
    assert resp.result["count"] == 0
    assert "note" in resp.result  # degradation notice
    assert resp.confidence.score < 0.8  # reduced confidence


@pytest.mark.asyncio
async def test_infra_qdrant_degraded(infra):
    """Infrastructure query_qdrant degrades gracefully without backends."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="query_qdrant",
        parameters={
            "query_vector": [0.1, 0.2, 0.3],
            "collection": "test",
            "limit": 5,
        },
    )
    resp = await infra.execute(req)
    assert "results" in resp.result


@pytest.mark.asyncio
async def test_weaver_extract_entities(weaver):
    """MasterWeaver extracts entities from text."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="extract_entities",
        parameters={
            "input_data": "Queen Bee produces Royal Jelly in the Hive",
            "domain": "beekeeping",
        },
    )
    resp = await weaver.execute(req)
    assert resp.result["count"] > 0
    assert resp.result["domain"] == "beekeeping"
    entities = resp.result["entities"]
    entity_texts = [e["text"] for e in entities]
    assert "Queen" in entity_texts or "Hive" in entity_texts


@pytest.mark.asyncio
async def test_weaver_validate_consistency(weaver):
    """MasterWeaver validates entity consistency."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="validate_entity_consistency",
        parameters={
            "entities": [
                {"text": "Queen", "type": "insect"},
                {"text": "Hive", "type": "structure"},
            ],
        },
    )
    resp = await weaver.execute(req)
    assert "is_consistent" in resp.result
    assert resp.result["is_consistent"] is True


@pytest.mark.asyncio
async def test_weaver_validate_catches_inconsistency(weaver):
    """MasterWeaver detects inconsistent entity types."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="validate_entity_consistency",
        parameters={
            "entities": [
                {"text": "Queen", "type": "insect"},
                {"text": "Queen", "type": "person"},  # contradiction
            ],
        },
    )
    resp = await weaver.execute(req)
    assert resp.result["is_consistent"] is False
    assert len(resp.result["inconsistencies"]) > 0


@pytest.mark.asyncio
async def test_weaver_query_ontology(weaver):
    """MasterWeaver returns ontology stub when no taxonomy loaded."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="query_domain_ontology",
        parameters={"domain": "beekeeping", "concept": "queen"},
    )
    resp = await weaver.execute(req)
    assert "definition" in resp.result


@pytest.mark.asyncio
async def test_verifier_validate_confidence(verifier):
    """Verification validates confidence claims."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="validate_confidence_claim",
        parameters={
            "claim": {"confidence": 0.85, "entities": 5},
            "evidence": {"source": "pattern_match", "coverage": 0.9},
        },
    )
    resp = await verifier.execute(req)
    assert "actual_confidence" in resp.result
    assert "claimed_confidence" in resp.result
    assert resp.result["claimed_confidence"] == 0.85
    assert resp.result["validation_result"] is True  # within threshold


@pytest.mark.asyncio
async def test_verifier_detects_overconfidence(verifier):
    """Verification penalizes overconfident claims without evidence."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="validate_confidence_claim",
        parameters={
            "claim": {"confidence": 0.99},
            "evidence": {},
        },
    )
    resp = await verifier.execute(req)
    actual = resp.result["actual_confidence"]
    claimed = resp.result["claimed_confidence"]
    assert actual < claimed  # should be penalized
    assert len(resp.result["penalties"]) > 0


@pytest.mark.asyncio
async def test_verifier_request_rerun(verifier):
    """Verification accepts rerun requests."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="request_rerun",
        parameters={
            "department": "MasterWeaver",
            "task_id": "extract_001",
            "new_params": {"confidence_threshold": 0.9},
            "reason": "Low confidence on first pass",
        },
    )
    resp = await verifier.execute(req)
    assert resp.result["accepted"] is True
    assert "new_task_id" in resp.result


@pytest.mark.asyncio
async def test_verifier_cross_check(verifier):
    """Verification cross-checks departments."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="cross_check_departments",
        parameters={
            "task_id": "task_001",
            "departments": ["Infrastructure", "MasterWeaver"],
        },
    )
    resp = await verifier.execute(req)
    assert "consistent" in resp.result
    assert "departments_checked" in resp.result


# ============================================================================
# Direct Orchestration tests
# ============================================================================


@pytest.mark.asyncio
async def test_orchestration_route_task(orchestrator):
    """Orchestration routes tasks to best department."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="route_task",
        parameters={
            "task_spec": {"description": "Extract entities from beekeeping text"},
            "candidate_departments": ["Infrastructure", "MasterWeaver", "Verification"],
        },
    )
    resp = await orchestrator.execute(req)
    assert "assigned_department" in resp.result
    assert resp.result["assigned_department"] == "MasterWeaver"
    assert resp.confidence.score > 0


@pytest.mark.asyncio
async def test_orchestration_route_no_candidates(orchestrator):
    """Orchestration routes without explicit candidates."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="route_task",
        parameters={
            "task_spec": "diagnose backend performance issues",
        },
    )
    resp = await orchestrator.execute(req)
    assert "assigned_department" in resp.result
    assert resp.confidence.score > 0


@pytest.mark.asyncio
async def test_orchestration_session_context(orchestrator):
    """Orchestration retrieves session context."""
    from hololoom.protocols.department import DepartmentRequest

    # Seed some context
    orchestrator.update_session_context("sess_001", {
        "roadmap_phase": "beta",
        "decisions": [{"id": "d1", "choice": "use_inmemory"}],
    })

    req = DepartmentRequest(
        task_type="get_session_context",
        parameters={"session_id": "sess_001", "department": "MasterWeaver"},
    )
    resp = await orchestrator.execute(req)
    assert resp.result["roadmap_phase"] == "beta"
    assert len(resp.result["relevant_decisions"]) == 1
    assert resp.confidence.score > 0.5


@pytest.mark.asyncio
async def test_orchestration_escalate(orchestrator):
    """Orchestration logs escalation requests."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="escalate_decision",
        parameters={
            "issue": "Conflicting entity types across departments",
            "reason": "MasterWeaver and Verification disagree on entity classification",
            "affected_departments": ["MasterWeaver", "Verification"],
        },
    )
    resp = await orchestrator.execute(req)
    assert "escalation_id" in resp.result
    assert resp.result["status"] == "pending"
    assert resp.confidence.score > 0.9


# ============================================================================
# Direct Context tests
# ============================================================================


@pytest.mark.asyncio
async def test_context_enrich(context):
    """Context enriches raw data with structural analysis."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="enrich_context",
        parameters={
            "raw_data": "Queen Bees produce Royal Jelly in the Hive during Spring",
            "context_request": {"user_id": "test_user"},
            "passes": 2,
        },
    )
    resp = await context.execute(req)
    assert "enriched_context" in resp.result
    assert resp.result["passes_completed"] >= 1
    assert resp.confidence.score > 0


@pytest.mark.asyncio
async def test_context_detect_missing(context):
    """Context detects missing context fields."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="detect_missing_context",
        parameters={
            "decision": {"action": "deploy_model", "model": "qwen3.5:9b"},
            "available_context": {"user_id": "blake"},
        },
    )
    resp = await context.execute(req)
    assert "missing_context" in resp.result
    assert "impact_severity" in resp.result
    assert len(resp.result["recommendations"]) > 0
    # Should detect missing domain/temporal/historical context
    missing_fields = [m["field"] for m in resp.result["missing_context"]]
    assert any("domain" in f or "temporal" in f or "historical" in f for f in missing_fields)


@pytest.mark.asyncio
async def test_context_detect_complete(context):
    """Context reports low severity when context is complete."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="detect_missing_context",
        parameters={
            "decision": {"action": "deploy"},
            "available_context": {
                "user_info": "admin",
                "domain_knowledge": "beekeeping",
                "temporal_deadline": "2026-03-15",
                "historical_results": [{"id": 1}],
                "constraints_budget": 1000,
            },
        },
    )
    resp = await context.execute(req)
    assert resp.result["impact_severity"] in ("low", "medium")
    assert resp.result["completeness_score"] > 0.5


# ============================================================================
# Direct Execution tests
# ============================================================================


@pytest.mark.asyncio
async def test_execution_run_task(execution):
    """Execution accepts and tracks tasks."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="run_claude_code_task",
        parameters={
            "task_spec": "Refactor the entity extraction module",
            "context_files": ["masterweaver.py"],
        },
    )
    resp = await execution.execute(req)
    assert "task_id" in resp.result
    assert resp.result["status"] in ("completed", "running", "queued")
    assert resp.result["execution_mode"] == "simulation"
    assert resp.confidence.score > 0


@pytest.mark.asyncio
async def test_execution_check_status(execution):
    """Execution returns status of tracked tasks."""
    from hololoom.protocols.department import DepartmentRequest

    # First, submit a task
    run_req = DepartmentRequest(
        task_type="run_claude_code_task",
        parameters={"task_spec": "Test task for status check"},
    )
    run_resp = await execution.execute(run_req)
    task_id = run_resp.result["task_id"]

    # Check its status
    status_req = DepartmentRequest(
        task_type="check_task_status",
        parameters={"task_id": task_id},
    )
    status_resp = await execution.execute(status_req)
    assert status_resp.result["task_id"] == task_id
    assert status_resp.result["status"] == "completed"
    assert status_resp.confidence.score > 0.9


@pytest.mark.asyncio
async def test_execution_unknown_task(execution):
    """Execution handles unknown task IDs gracefully."""
    from hololoom.protocols.department import DepartmentRequest

    req = DepartmentRequest(
        task_type="check_task_status",
        parameters={"task_id": "nonexistent_task"},
    )
    resp = await execution.execute(req)
    assert "error" in resp.result
    assert resp.confidence.score < 0.5


# ============================================================================
# Federation tests (through MCPRouter, no HTTP)
# ============================================================================


@pytest.mark.asyncio
async def test_real_federation_boots():
    """Real federation creates and registers all six departments."""
    router = await create_real_federation()
    assert "Infrastructure" in router.departments
    assert "MasterWeaver" in router.departments
    assert "Verification" in router.departments
    assert "Orchestration" in router.departments
    assert "Context" in router.departments
    assert "Execution" in router.departments


@pytest.mark.asyncio
async def test_real_federation_call_extract():
    """Real federation routes extract_entities through MCPRouter."""
    router = await create_real_federation()
    resp = await router.call_tool(
        target="MasterWeaver",
        tool_name="extract_entities",
        parameters={
            "input_data": "Bees pollinate Flowers in Spring",
            "domain": "beekeeping",
        },
        session_id="test-session",
    )
    from hololoom.infrastructure.mcp.protocol import ResponseStatus
    assert resp.status == ResponseStatus.SUCCESS
    inner = resp.result["result"]
    assert inner["count"] > 0
    assert inner["domain"] == "beekeeping"


@pytest.mark.asyncio
async def test_real_federation_call_validate():
    """Real federation routes validate_confidence_claim."""
    router = await create_real_federation()
    resp = await router.call_tool(
        target="Verification",
        tool_name="validate_confidence_claim",
        parameters={
            "claim": {"confidence": 0.9},
            "evidence": {"data": "some_evidence"},
        },
        session_id="test-session",
    )
    from hololoom.infrastructure.mcp.protocol import ResponseStatus
    assert resp.status == ResponseStatus.SUCCESS
    assert "actual_confidence" in resp.result["result"]


@pytest.mark.asyncio
async def test_real_federation_call_diagnose():
    """Real federation routes diagnose_performance."""
    router = await create_real_federation()
    resp = await router.call_tool(
        target="Infrastructure",
        tool_name="diagnose_performance",
        parameters={},
        session_id="test-session",
    )
    from hololoom.infrastructure.mcp.protocol import ResponseStatus
    assert resp.status == ResponseStatus.SUCCESS


@pytest.mark.asyncio
async def test_real_federation_call_route():
    """Real federation routes route_task through Orchestration."""
    router = await create_real_federation()
    resp = await router.call_tool(
        target="Orchestration",
        tool_name="route_task",
        parameters={
            "task_spec": {"description": "Extract entities from text"},
            "candidate_departments": ["MasterWeaver", "Infrastructure"],
        },
        session_id="test-session",
    )
    from hololoom.infrastructure.mcp.protocol import ResponseStatus
    assert resp.status == ResponseStatus.SUCCESS
    assert "assigned_department" in resp.result["result"]


@pytest.mark.asyncio
async def test_real_federation_call_enrich():
    """Real federation routes enrich_context through Context."""
    router = await create_real_federation()
    resp = await router.call_tool(
        target="Context",
        tool_name="enrich_context",
        parameters={
            "raw_data": "Bees pollinate Flowers in the Spring meadow",
            "context_request": {},
            "passes": 2,
        },
        session_id="test-session",
    )
    from hololoom.infrastructure.mcp.protocol import ResponseStatus
    assert resp.status == ResponseStatus.SUCCESS
    assert "enriched_context" in resp.result["result"]


@pytest.mark.asyncio
async def test_real_federation_call_run_task():
    """Real federation routes run_claude_code_task through Execution."""
    router = await create_real_federation()
    resp = await router.call_tool(
        target="Execution",
        tool_name="run_claude_code_task",
        parameters={
            "task_spec": "Run tests on the department module",
        },
        session_id="test-session",
    )
    from hololoom.infrastructure.mcp.protocol import ResponseStatus
    assert resp.status == ResponseStatus.SUCCESS
    assert "task_id" in resp.result["result"]


# ============================================================================
# HTTP API tests (through FastAPI + real departments)
# ============================================================================


@pytest_asyncio.fixture
async def real_app():
    """FastAPI app with real departments instead of lightweight."""
    import hololoom.apps.server.department_api as dept_api

    # Replace the module-level router with a real federation
    old_router = dept_api._router
    dept_api._router = await create_real_federation()

    app = FastAPI()
    app.include_router(router)
    yield app

    # Restore
    dept_api._router = old_router


@pytest_asyncio.fixture
async def real_client(real_app):
    """Async HTTP client for real department testing."""
    transport = ASGITransport(app=real_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_http_list_departments(real_client):
    resp = await real_client.get("/departments")
    assert resp.status_code == 200
    data = resp.json()
    names = {d["name"] for d in data}
    assert names == {"Infrastructure", "MasterWeaver", "Verification", "Orchestration", "Context", "Execution"}


@pytest.mark.asyncio
async def test_http_extract_entities(real_client):
    resp = await real_client.post(
        "/departments/MasterWeaver/extract_entities",
        json={
            "parameters": {
                "input_data": "Queen Bees produce Royal Jelly",
                "domain": "beekeeping",
            }
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    inner = data["result"]["result"]
    assert inner["count"] > 0


@pytest.mark.asyncio
async def test_http_validate_confidence(real_client):
    resp = await real_client.post(
        "/departments/Verification/validate_confidence_claim",
        json={
            "parameters": {
                "claim": {"confidence": 0.88},
                "evidence": {"method": "cross_check"},
            }
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    inner = data["result"]["result"]
    assert "actual_confidence" in inner
    assert "penalties" in inner


@pytest.mark.asyncio
async def test_http_diagnose_performance(real_client):
    resp = await real_client.post(
        "/departments/Infrastructure/diagnose_performance",
        json={"parameters": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
