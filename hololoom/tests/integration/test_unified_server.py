"""
Integration tests for Unified API Server
==========================================
Tests the consolidated server foundation (Wave 1).

**Philosophy**: Framework → Elegance → Verify

Tests:
1. Server startup and initialization
2. Health check endpoint
3. Statistics endpoint
4. SSE event stream
5. Query endpoint (basic)
6. Error handling

Usage:
    pytest HoloLoom/tests/integration/test_unified_server.py -v
"""

import pytest
import asyncio
import json
from typing import AsyncGenerator
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Import server app
from hololoom.apps.server.unified_server import app, state


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create test client."""
    # Initialize server state
    await state.initialize()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert "uptime" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_stats_endpoint(client: AsyncClient):
    """Test statistics endpoint."""
    response = await client.get("/stats")

    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    required_fields = [
        "uptime",
        "queries_total",
        "successful_queries",
        "failed_queries",
        "success_rate",
        "latency_avg",
        "confidence_avg",
        "queries_by_mode",
        "memory_entities",
        "learning_active"
    ]

    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Verify types
    assert isinstance(data["uptime"], (int, float))
    assert isinstance(data["queries_total"], int)
    assert isinstance(data["success_rate"], (int, float))
    assert isinstance(data["queries_by_mode"], dict)


@pytest.mark.asyncio
async def test_query_endpoint_direct_mode(client: AsyncClient):
    """Test query endpoint with direct mode."""
    response = await client.post(
        "/query",
        json={
            "text": "What is Thompson Sampling?",
            "mode": "direct",
            "max_steps": 3
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "response" in data
    assert "confidence" in data
    assert "reasoning_mode" in data
    assert "latency_ms" in data
    assert "timestamp" in data

    # Verify types
    assert isinstance(data["response"], str)
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] > 0


@pytest.mark.asyncio
async def test_query_endpoint_invalid_mode(client: AsyncClient):
    """Test query endpoint with invalid mode."""
    response = await client.post(
        "/query",
        json={
            "text": "Test query",
            "mode": "invalid_mode",  # Invalid
            "max_steps": 3
        }
    )

    # Should fail with 422 validation error (Pydantic)
    # OR 500 if it reaches the ReasoningMode enum
    assert response.status_code in [422, 500]


@pytest.mark.asyncio
async def test_query_endpoint_text_too_large(client: AsyncClient):
    """Test query endpoint with text exceeding limit."""
    large_text = "x" * 200_000  # 200KB (exceeds 100KB limit)

    response = await client.post(
        "/query",
        json={
            "text": large_text,
            "mode": "direct"
        }
    )

    # Should fail with 422 validation error
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_recent_queries(client: AsyncClient):
    """Test recent queries endpoint."""
    # First, make a query to populate history
    await client.post(
        "/query",
        json={
            "text": "Test query",
            "mode": "direct"
        }
    )

    # Now fetch recent queries
    response = await client.get("/queries/recent?limit=10")

    assert response.status_code == 200
    data = response.json()

    assert "queries" in data
    assert isinstance(data["queries"], list)

    # Should have at least 1 query (the one we just made)
    assert len(data["queries"]) >= 1

    # Verify query structure
    if len(data["queries"]) > 0:
        query = data["queries"][-1]
        assert "text" in query
        assert "mode" in query
        assert "confidence" in query
        assert "latency_ms" in query
        assert "timestamp" in query


@pytest.mark.asyncio
async def test_learning_status(client: AsyncClient):
    """Test learning status endpoint."""
    response = await client.get("/learning/status")

    # May return 503 if learning engine not initialized
    # Or 200 with stats if initialized
    if response.status_code == 200:
        data = response.json()
        # Should have some stats
        assert isinstance(data, dict)
    elif response.status_code == 503:
        # Learning engine not initialized (acceptable in test)
        pass
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


@pytest.mark.asyncio
async def test_memory_stats(client: AsyncClient):
    """Test memory statistics endpoint."""
    response = await client.get("/memory/stats")

    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    assert "total_entities" in data
    assert "total_relationships" in data
    assert "total_memories" in data
    assert "backend" in data

    # Verify types
    assert isinstance(data["total_entities"], int)
    assert isinstance(data["backend"], str)


@pytest.mark.asyncio
async def test_safety_status(client: AsyncClient):
    """Test safety status endpoint."""
    response = await client.get("/safety/status")

    assert response.status_code == 200
    data = response.json()

    # Verify safety systems
    assert "guardrails_active" in data
    assert "deception_detector_active" in data
    assert "audit_trail_active" in data

    # Verify types
    assert isinstance(data["guardrails_active"], bool)


@pytest.mark.asyncio
async def test_safety_gate(client: AsyncClient):
    """Test safety gate endpoint."""
    response = await client.post(
        "/safety/gate",
        json={
            "action": "test_action",
            "context": {"test": "data"}
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify gate response
    assert "allowed" in data
    assert "safety_score" in data
    assert "risk_level" in data
    assert "reason" in data

    # Verify types
    assert isinstance(data["allowed"], bool)
    assert isinstance(data["safety_score"], (int, float))
    assert 0.0 <= data["safety_score"] <= 1.0


@pytest.mark.asyncio
async def test_audit_trail(client: AsyncClient):
    """Test audit trail endpoint."""
    response = await client.get("/safety/audit-trail?limit=10")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "entries" in data
    assert "total" in data

    assert isinstance(data["entries"], list)
    assert isinstance(data["total"], int)


@pytest.mark.asyncio
async def test_ingestion_status(client: AsyncClient):
    """Test ingestion status endpoint."""
    response = await client.get("/ingestion/status")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "queue" in data
    assert "total" in data

    assert isinstance(data["queue"], list)
    assert isinstance(data["total"], int)


@pytest.mark.asyncio
async def test_visualization_confidence(client: AsyncClient):
    """Test confidence visualization endpoint."""
    response = await client.get("/viz/confidence")

    assert response.status_code == 200
    data = response.json()

    # Verify structure
    assert "confidences" in data
    assert "timestamps" in data

    assert isinstance(data["confidences"], list)
    assert isinstance(data["timestamps"], list)


@pytest.mark.asyncio
async def test_orchestrator_monitor(client: AsyncClient):
    """Test orchestrator monitor endpoint."""
    response = await client.get("/monitor/orchestrator")

    assert response.status_code == 200
    data = response.json()

    # Verify status field exists
    assert "status" in data


def test_sse_event_stream():
    """Test SSE event stream (synchronous test)."""
    # Use TestClient for SSE (simpler than AsyncClient for streaming)
    with TestClient(app) as client:
        # Note: This is a simplified test
        # Full SSE testing requires more complex setup
        response = client.get("/events", stream=True)

        # Should get 200 with text/event-stream
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


# Performance Tests

@pytest.mark.asyncio
async def test_query_latency(client: AsyncClient):
    """Test query latency is reasonable."""
    import time

    start = time.time()
    response = await client.post(
        "/query",
        json={
            "text": "Quick test",
            "mode": "direct",
            "max_steps": 1
        }
    )
    end = time.time()

    latency_ms = (end - start) * 1000

    assert response.status_code == 200

    # Latency should be < 5 seconds (generous for test environment)
    assert latency_ms < 5000, f"Query took too long: {latency_ms}ms"


@pytest.mark.asyncio
async def test_concurrent_queries(client: AsyncClient):
    """Test handling concurrent queries."""
    # Create 5 concurrent queries
    tasks = [
        client.post(
            "/query",
            json={"text": f"Query {i}", "mode": "direct", "max_steps": 1}
        )
        for i in range(5)
    ]

    responses = await asyncio.gather(*tasks)

    # All should succeed
    for response in responses:
        assert response.status_code == 200


# Integration Tests

@pytest.mark.asyncio
async def test_full_query_flow(client: AsyncClient):
    """Test complete query flow: query → stats update → recent queries."""
    # Get initial stats
    stats_before = await client.get("/stats")
    initial_total = stats_before.json()["queries_total"]

    # Make a query
    query_response = await client.post(
        "/query",
        json={
            "text": "Integration test query",
            "mode": "direct"
        }
    )
    assert query_response.status_code == 200
    query_data = query_response.json()

    # Get updated stats
    stats_after = await client.get("/stats")
    updated_total = stats_after.json()["queries_total"]

    # Stats should be incremented
    assert updated_total == initial_total + 1

    # Check recent queries
    recent_response = await client.get("/queries/recent?limit=1")
    recent_data = recent_response.json()

    # Most recent query should match our query
    assert len(recent_data["queries"]) > 0
    last_query = recent_data["queries"][-1]
    assert last_query["text"] == "Integration test query"
    assert last_query["mode"] == "direct"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
