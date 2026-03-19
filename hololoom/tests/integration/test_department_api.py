"""
Smoke tests for the Department Federation HTTP API.

Tests the FastAPI routes in department_api.py against the lightweight
federation (no external dependencies needed).
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from hololoom.apps.server.department_api import router, _get_router


@pytest.fixture
def app():
    """Create a minimal FastAPI app with just the department router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest_asyncio.fixture
async def client(app):
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_list_departments(client):
    resp = await client.get("/departments")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    names = {d["name"] for d in data}
    assert names == {"Infrastructure", "MasterWeaver", "Verification"}


@pytest.mark.asyncio
async def test_health_all(client):
    resp = await client.get("/departments/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "Infrastructure" in data
    assert "MasterWeaver" in data
    assert "Verification" in data


@pytest.mark.asyncio
async def test_list_tools(client):
    resp = await client.get("/departments/MasterWeaver/tools")
    assert resp.status_code == 200
    tools = resp.json()
    tool_names = [t["name"] for t in tools]
    assert "extract_entities" in tool_names


@pytest.mark.asyncio
async def test_list_tools_404(client):
    resp = await client.get("/departments/Nonexistent/tools")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_health_one(client):
    resp = await client.get("/departments/Infrastructure/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_call_extract_entities(client):
    resp = await client.post(
        "/departments/MasterWeaver/extract_entities",
        json={
            "parameters": {
                "input_data": "Bees pollinate Flowers in Spring",
                "domain": "beekeeping",
            }
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    # Adapter wraps dept response: {result: {...}, confidence: ..., task_id: ...}
    inner = data["result"]["result"]
    assert inner["domain"] == "beekeeping"
    assert inner["count"] > 0


@pytest.mark.asyncio
async def test_call_diagnose_performance(client):
    resp = await client.post(
        "/departments/Infrastructure/diagnose_performance",
        json={"parameters": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["result"]["result"]["status"] == "healthy"


@pytest.mark.asyncio
async def test_call_validate_confidence(client):
    resp = await client.post(
        "/departments/Verification/validate_confidence_claim",
        json={
            "parameters": {
                "claim": {"confidence": 0.99},
                "evidence": {},
            }
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "actual_confidence" in data["result"]["result"]


@pytest.mark.asyncio
async def test_call_unknown_department(client):
    resp = await client.post(
        "/departments/Nonexistent/some_tool",
        json={"parameters": {}},
    )
    assert resp.status_code == 200  # MCP returns error in body, not HTTP error
    data = resp.json()
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_session_id_propagated(client):
    resp = await client.post(
        "/departments/Infrastructure/diagnose_performance",
        json={"parameters": {}, "session_id": "test-session-42"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-session-42"


@pytest.mark.asyncio
async def test_session_id_generated_when_omitted(client):
    resp = await client.post(
        "/departments/Infrastructure/diagnose_performance",
        json={"parameters": {}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]  # should be auto-generated, not empty
