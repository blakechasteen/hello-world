"""
FastAPI Server Integration Tests
=================================
Comprehensive integration tests for HoloLoom Agentic API server.

Tests the complete FastAPI → AgenticOrchestrator → Response flow
to ensure VS Code extension receives correct responses.

Created: 2025-11-22
Status: Critical for investor demo
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import the FastAPI app
from HoloLoom.apps.server.agentic_api import app, ServerState
from HoloLoom.agentic.core import AgenticResult, ReasoningMode
from HoloLoom.config import Config
# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)
@pytest.fixture
def mock_orchestrator():
    """Mock AgenticOrchestrator for fast, deterministic tests."""
    mock = AsyncMock()

    # Default successful response
    mock.reason.return_value = AgenticResult(
        response="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
        confidence=0.92,
        reasoning_mode="direct",
        steps_taken=[
            {
                "type": "direct_answer",
                "query": "What is Thompson Sampling?",
                "confidence": 0.92,
                "completed": True
            }
        ],
        total_queries=1,
        total_duration_ms=145.3,
        verification=None,
        aggregated_epistemic_confidence=0.88
    )

    return mock
@pytest.fixture(autouse=True)
def setup_server_state():
    """Reset server state before each test."""
    from HoloLoom.apps.server.agentic_api import state

    # Reset stats if exists
    if state.stats:
        state.stats.total_queries = 0
        state.stats.successful_queries = 0
        state.stats.failed_queries = 0

    # Reset rate limiter
    if state.rate_limiter:
        state.rate_limiter.requests.clear()

    yield

    # Cleanup - reset state for next test
    if state.stats:
        state.stats.total_queries = 0
# ============================================================================
# Health Endpoint Tests
# ============================================================================

def test_health_endpoint_success(client):
    """Test /health endpoint returns 200 OK."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "uptime_seconds" in data
def test_health_endpoint_performance(client):
    """Test /health endpoint responds quickly (< 5ms target)."""
    import time

    start = time.time()
    response = client.get("/health")
    duration_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    assert duration_ms < 10  # 10ms generous limit (target is 5ms)
# ============================================================================
# Query Endpoint Tests - All Reasoning Modes
# ============================================================================

@patch('HoloLoom.apps.server.agentic_api.state')
def test_query_endpoint_direct_mode(mock_state, client, mock_orchestrator):
    """Test /query endpoint with DIRECT reasoning mode."""
    # Setup mock
    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    mock_state.rate_limiter = None

    # Make request
    response = client.post("/query", json={
        "text": "What is Thompson Sampling?",
        "mode": "direct",
        "max_steps": 1
    })

    # Verify response structure matches TypeScript AgenticResult interface
    assert response.status_code == 200
    data = response.json()

    # Required fields
    assert "response" in data
    assert "confidence" in data
    assert "reasoning_mode" in data
    assert "steps_taken" in data
    assert "total_queries" in data
    assert "total_duration_ms" in data
    assert "timestamp" in data
    assert "query_id" in data

    # Type validation
    assert isinstance(data["response"], str)
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["steps_taken"], list)
    assert data["reasoning_mode"] == "direct"

    # Orchestrator called correctly
    mock_orchestrator.reason.assert_called_once()
@patch('HoloLoom.apps.server.agentic_api.state')
def test_query_endpoint_verify_mode(mock_state, client, mock_orchestrator):
    """Test /query endpoint with VERIFY reasoning mode (multi-query)."""
    # Setup mock with verification result
    mock_orchestrator.reason.return_value = AgenticResult(
        response="Thompson Sampling balances exploration and exploitation.",
        confidence=0.87,
        reasoning_mode="verify",
        steps_taken=[
            {"type": "initial_answer", "confidence": 0.85},
            {"type": "verification", "confidence": 0.89}
        ],
        total_queries=2,
        total_duration_ms=612.7,
        verification={
            "verified": True,
            "confidence": 0.89,
            "contradictions": [],
            "supporting_evidence": ["Paper by Thompson (1933)", "UCB algorithm comparison"],
            "suggested_refinements": []
        },
        aggregated_epistemic_confidence=0.82
    )

    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    mock_state.rate_limiter = None

    # Make request
    response = client.post("/query", json={
        "text": "Explain Thompson Sampling",
        "mode": "verify",
        "max_steps": 3
    })

    assert response.status_code == 200
    data = response.json()

    # Verification-specific fields
    assert "verification" in data
    verification = data["verification"]
    assert verification["verified"] is True
    assert isinstance(verification["supporting_evidence"], list)
    assert len(verification["supporting_evidence"]) > 0
@patch('HoloLoom.apps.server.agentic_api.state')
def test_query_endpoint_research_mode(mock_state, client, mock_orchestrator):
    """Test /query endpoint with RESEARCH mode (deep exploration)."""
    # Setup mock with multiple reasoning steps
    mock_orchestrator.reason.return_value = AgenticResult(
        response="Comprehensive analysis of Thompson Sampling...",
        confidence=0.91,
        reasoning_mode="research",
        steps_taken=[
            {"type": "initial_query", "query": "What is Thompson Sampling?"},
            {"type": "follow_up", "query": "How does it compare to UCB?"},
            {"type": "follow_up", "query": "What are the tradeoffs?"},
            {"type": "synthesis", "completed": True}
        ],
        total_queries=5,
        total_duration_ms=923.4,
        verification=None,
        aggregated_epistemic_confidence=0.85
    )

    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    
    
    
    
    mock_state.rate_limiter = None

    response = client.post("/query", json={
        "text": "Analyze Thompson Sampling comprehensively",
        "mode": "research",
        "max_steps": 5
    })

    assert response.status_code == 200
    data = response.json()
    assert data["reasoning_mode"] == "research"
    assert len(data["steps_taken"]) >= 3  # Multiple steps
# ============================================================================
# Request Validation Tests
# ============================================================================

def test_query_validation_text_too_large(client):
    """Test request validation rejects queries > 100KB."""
    # Create a 101KB query
    large_query = "x" * 101_000

    response = client.post("/query", json={
        "text": large_query,
        "mode": "direct"
    })

    # Should return 422 Unprocessable Entity
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
def test_query_validation_invalid_max_steps(client):
    """Test request validation rejects invalid max_steps."""
    # max_steps must be 1-20
    response = client.post("/query", json={
        "text": "Valid query",
        "mode": "direct",
        "max_steps": 50  # Invalid: > 20
    })

    assert response.status_code == 422
def test_query_validation_missing_required_fields(client):
    """Test request validation requires 'text' field."""
    response = client.post("/query", json={
        "mode": "direct"
        # Missing 'text' field
    })

    assert response.status_code == 422
# ============================================================================
# Rate Limiting Tests
# ============================================================================

def test_rate_limiting_within_limit(client):
    """Test requests within rate limit are accepted."""
    # Make 5 requests (well within 60 req/min limit)
    for i in range(5):
        response = client.get("/stats")
        assert response.status_code == 200
def test_rate_limiting_exceeds_limit(client):
    """Test rate limiting blocks excessive requests."""
    # Note: This is tricky because TestClient doesn't preserve client.host
    # We'd need to set up the rate limiter differently for testing
    # For now, document that this needs manual testing or integration test
    pytest.skip("Rate limiting requires integration test with real client IPs")
# ============================================================================
# Stats Endpoint Tests
# ============================================================================

def test_stats_endpoint(client):
    """Test /stats endpoint returns server statistics."""
    # Just verify endpoint exists and returns valid structure
    response = client.get("/stats")

    assert response.status_code == 200
    data = response.json()

    # Check expected fields exist (actual values depend on server state)
    assert "service" in data or "total_queries" in data or "status" in data
# ============================================================================
# Error Handling Tests
# ============================================================================

@patch('HoloLoom.apps.server.agentic_api.state')
def test_error_handling_orchestrator_failure(mock_state, client, mock_orchestrator):
    """Test graceful error handling when orchestrator fails."""
    # Setup mock to raise exception
    mock_orchestrator.reason.side_effect = Exception("Orchestrator internal error")
    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    
    
    
    
    mock_state.rate_limiter = None

    response = client.post("/query", json={
        "text": "This will fail",
        "mode": "direct"
    })

    # Should return 500 Internal Server Error
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
# ============================================================================
# CORS Tests
# ============================================================================

def test_cors_headers_present(client):
    """Test CORS headers are present for VS Code extension."""
    response = client.options("/query")

    # Should have CORS headers
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
# ============================================================================
# Code Context Integration Tests
# ============================================================================

@patch('HoloLoom.apps.server.agentic_api.state')
def test_query_with_code_context(mock_state, client, mock_orchestrator):
    """Test /query endpoint accepts and uses code context from VS Code."""
    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    
    
    
    
    mock_state.rate_limiter = None

    response = client.post("/query", json={
        "text": "Explain this code",
        "context": {
            "fileName": "example.ts",
            "languageId": "typescript",
            "selection": "function foo() { return 42; }",
            "workspace": "/path/to/project"
        },
        "mode": "direct"
    })

    assert response.status_code == 200
    data = response.json()
    assert "response" in data
# ============================================================================
# Performance Tests
# ============================================================================

@patch('HoloLoom.apps.server.agentic_api.state')
def test_query_endpoint_performance_direct(mock_state, client, mock_orchestrator):
    """Test /query endpoint performance in DIRECT mode (< 200ms target)."""
    import time

    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    
    
    
    
    mock_state.rate_limiter = None

    start = time.time()
    response = client.post("/query", json={
        "text": "Simple question",
        "mode": "direct"
    })
    duration_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    # Note: With mocked orchestrator, overhead should be minimal
    # In real integration test, verify < 200ms with actual orchestrator
    assert duration_ms < 100  # Mock should be very fast
# ============================================================================
# Concurrent Request Tests
# ============================================================================

@patch('HoloLoom.apps.server.agentic_api.state')
def test_concurrent_requests(mock_state, client, mock_orchestrator):
    """Test server handles concurrent requests correctly."""
    mock_state.orchestrator = mock_orchestrator
    mock_state.stats = None
    
    
    
    
    mock_state.rate_limiter = None

    # Make 5 concurrent requests
    import concurrent.futures

    def make_request():
        return client.post("/query", json={
            "text": "Concurrent test",
            "mode": "direct"
        })

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        responses = [f.result() for f in futures]

    # All should succeed
    assert all(r.status_code == 200 for r in responses)
# ============================================================================
# TypeScript Interface Compatibility Tests
# ============================================================================

def test_response_matches_typescript_interface(client, mock_orchestrator):
    """
    Test that Python AgenticResponse exactly matches TypeScript AgenticResult.

    This is critical for VS Code extension integration.
    """
    with patch('HoloLoom.apps.server.agentic_api.state') as mock_state:
        mock_state.orchestrator = mock_orchestrator
        mock_state.stats = None
        
        
        
        
        mock_state.rate_limiter = None

        response = client.post("/query", json={
            "text": "Test",
            "mode": "verify"
        })

        data = response.json()

        # Exact field names from TypeScript interface
        typescript_fields = {
            "response", "confidence", "reasoning_mode",
            "steps_taken", "total_queries", "total_duration_ms",
            "verification", "timestamp", "query_id"
        }

        # Verify all TypeScript fields present
        for field in typescript_fields:
            assert field in data, f"Missing TypeScript field: {field}"

        # Verify types match
        assert isinstance(data["response"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["reasoning_mode"], str)
        assert isinstance(data["steps_taken"], list)
        assert isinstance(data["total_queries"], int)
        assert isinstance(data["total_duration_ms"], (int, float))
        # verification can be None or dict
        assert data["verification"] is None or isinstance(data["verification"], dict)
# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
======================

✅ Health endpoint (2 tests)
✅ Query endpoint - all reasoning modes (3 tests)
✅ Request validation (3 tests)
✅ Rate limiting (2 tests - 1 skipped)
✅ Stats endpoint (1 test)
✅ Error handling (1 test)
✅ CORS (1 test)
✅ Code context (1 test)
✅ Performance (1 test)
✅ Concurrency (1 test)
✅ TypeScript compatibility (1 test)

Total: 17 tests (16 active, 1 skipped)

Integration Points Covered:
- FastAPI ↔ AgenticOrchestrator
- Request validation (Pydantic)
- Error handling and HTTP status codes
- TypeScript interface compatibility
- Code context from VS Code

Next Steps:
1. Run: pytest HoloLoom/server/tests/test_agentic_api_integration.py -v
2. Create TypeScript tests: squad/tests/test_hololoom_bridge.test.ts
3. Create full stack E2E: HoloLoom/tests/e2e/test_full_stack_vscode.py
"""
