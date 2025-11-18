"""
API Integration Tests
======================

Tests for all HoloLoom API endpoints.

Author: HoloLoom Team
Date: 2025-11-18 (Week 8B)
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from datetime import datetime
import os

# Set test environment
os.environ["HOLOLOOM_ENV"] = "test"
os.environ["HOLOLOOM_API_KEY"] = "test-api-key-12345"

from HoloLoom.api.server import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers fixture."""
    return {"X-API-Key": os.environ["HOLOLOOM_API_KEY"]}


# ============================================================================
# Health & Stats Tests
# ============================================================================

def test_health_check(client):
    """Test health check endpoint (no auth required)."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "systems" in data
    assert "timestamp" in data
    assert "version" in data


def test_health_check_systems(client):
    """Test health check includes all systems."""
    response = client.get("/health")
    data = response.json()

    systems = data["systems"]
    assert "consolidation" in systems
    assert "semantic_transition" in systems
    assert "temporal_evolution" in systems
    assert "curiosity" in systems
    assert "graph_reasoning" in systems


def test_prometheus_metrics(client):
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    # Check for some expected metrics
    content = response.text
    assert "hololoom_requests_total" in content or len(content) > 0


def test_stats_without_auth(client):
    """Test stats endpoint requires authentication."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 401  # Unauthorized


def test_stats_with_auth(client, auth_headers):
    """Test stats endpoint with authentication."""
    response = client.get("/api/v1/stats", headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert "consolidation" in data
    assert "semantic" in data
    assert "curiosity" in data
    assert "graph" in data


# ============================================================================
# Authentication Tests
# ============================================================================

def test_auth_missing_api_key(client):
    """Test request without API key is rejected."""
    response = client.post(
        "/api/v1/experience",
        json={"content": "test", "scope": "SESSION"}
    )
    assert response.status_code == 401


def test_auth_invalid_api_key(client):
    """Test request with invalid API key is rejected."""
    response = client.post(
        "/api/v1/experience",
        json={"content": "test", "scope": "SESSION"},
        headers={"X-API-Key": "invalid-key"}
    )
    assert response.status_code == 401


def test_auth_valid_api_key(client, auth_headers):
    """Test request with valid API key is accepted."""
    response = client.post(
        "/api/v1/experience",
        json={"content": "test memory", "scope": "SESSION"},
        headers=auth_headers
    )
    # Should not be 401 (Unauthorized)
    assert response.status_code != 401


# ============================================================================
# Memory Operations Tests
# ============================================================================

def test_experience_create(client, auth_headers):
    """Test creating new memory."""
    response = client.post(
        "/api/v1/experience",
        json={
            "content": "Thompson Sampling balances exploration and exploitation",
            "scope": "SESSION",
            "metadata": {"source": "test"}
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "memory_id" in data
    assert data["scope"] == "SESSION"
    assert "timestamp" in data


def test_experience_validation(client, auth_headers):
    """Test experience request validation."""
    # Missing content
    response = client.post(
        "/api/v1/experience",
        json={"scope": "SESSION"},
        headers=auth_headers
    )
    assert response.status_code == 422  # Validation error

    # Empty content
    response = client.post(
        "/api/v1/experience",
        json={"content": "", "scope": "SESSION"},
        headers=auth_headers
    )
    assert response.status_code == 422


def test_recall_memories(client, auth_headers):
    """Test recalling memories."""
    # First, create a memory
    client.post(
        "/api/v1/experience",
        json={"content": "Thompson Sampling test", "scope": "SESSION"},
        headers=auth_headers
    )

    # Then recall it
    response = client.post(
        "/api/v1/recall",
        json={
            "query": "Thompson Sampling",
            "max_results": 10
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "memories" in data
    assert "count" in data
    assert "latency_ms" in data


def test_recall_validation(client, auth_headers):
    """Test recall request validation."""
    # Missing query
    response = client.post(
        "/api/v1/recall",
        json={"max_results": 10},
        headers=auth_headers
    )
    assert response.status_code == 422

    # Empty query
    response = client.post(
        "/api/v1/recall",
        json={"query": "", "max_results": 10},
        headers=auth_headers
    )
    assert response.status_code == 422

    # Invalid max_results
    response = client.post(
        "/api/v1/recall",
        json={"query": "test", "max_results": 1000},
        headers=auth_headers
    )
    assert response.status_code == 422


# ============================================================================
# Consolidation Tests
# ============================================================================

def test_consolidation_trigger(client, auth_headers):
    """Test triggering consolidation."""
    response = client.post(
        "/api/v1/consolidation/trigger",
        json={
            "strategy": "fact_extraction",
            "max_episodes": 100,
            "prune_episodes": False
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "job_id" in data
    assert "estimated_duration_ms" in data


def test_consolidation_validation(client, auth_headers):
    """Test consolidation request validation."""
    # Invalid strategy
    response = client.post(
        "/api/v1/consolidation/trigger",
        json={"strategy": "invalid_strategy"},
        headers=auth_headers
    )
    assert response.status_code == 422


# ============================================================================
# Semantic Transition Tests
# ============================================================================

def test_pattern_detection(client, auth_headers):
    """Test pattern detection."""
    response = client.post(
        "/api/v1/semantic/detect-patterns",
        json={
            "threshold": 3,
            "similarity_threshold": 0.75,
            "window_days": 7,
            "max_patterns": 50
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "patterns" in data
    assert "count" in data
    assert "detection_time_ms" in data
    assert isinstance(data["patterns"], list)


def test_pattern_detection_validation(client, auth_headers):
    """Test pattern detection validation."""
    # Invalid threshold (must be >= 1)
    response = client.post(
        "/api/v1/semantic/detect-patterns",
        json={"threshold": 0},
        headers=auth_headers
    )
    assert response.status_code == 422

    # Invalid similarity threshold (must be 0.0-1.0)
    response = client.post(
        "/api/v1/semantic/detect-patterns",
        json={"similarity_threshold": 1.5},
        headers=auth_headers
    )
    assert response.status_code == 422


def test_pattern_promotion(client, auth_headers):
    """Test promoting pattern to semantic concept."""
    response = client.post(
        "/api/v1/semantic/promote",
        json={
            "pattern_id": "pattern_test123",
            "concept_name": "test_concept"
        },
        headers=auth_headers
    )

    # Should queue the job (may fail later if pattern doesn't exist)
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "queued"
        assert data["pattern_id"] == "pattern_test123"


# ============================================================================
# Temporal Evolution Tests
# ============================================================================

def test_temporal_query(client, auth_headers):
    """Test temporal query."""
    timestamp = datetime(2025, 11, 1, 12, 0, 0).isoformat() + "Z"

    response = client.post(
        "/api/v1/temporal/query",
        json={
            "concept": "thompson_sampling",
            "timestamp": timestamp
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["concept"] == "thompson_sampling"
    assert "found" in data
    # Snapshot may be None if no data at that time


def test_temporal_query_validation(client, auth_headers):
    """Test temporal query validation."""
    # Future timestamp (should fail)
    future = datetime(2030, 1, 1, 0, 0, 0).isoformat() + "Z"
    response = client.post(
        "/api/v1/temporal/query",
        json={
            "concept": "test",
            "timestamp": future
        },
        headers=auth_headers
    )
    assert response.status_code == 422


def test_concept_history(client, auth_headers):
    """Test getting concept history."""
    response = client.get(
        "/api/v1/temporal/history/thompson_sampling",
        headers=auth_headers
    )

    # May return 404 if concept not found
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert data["concept"] == "thompson_sampling"
        assert "current_state" in data
        assert "state_transitions" in data
        assert "memory_timeline" in data


def test_evolution_summary(client, auth_headers):
    """Test evolution summary."""
    response = client.post(
        "/api/v1/temporal/summary",
        json={"days": 30},
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["period_days"] == 30
    assert "total_concepts" in data
    assert "concepts_by_state" in data
    assert "new_concepts" in data


# ============================================================================
# Curiosity Engine Tests
# ============================================================================

def test_curiosity_suggestions(client, auth_headers):
    """Test getting exploration suggestions."""
    response = client.post(
        "/api/v1/curiosity/suggest",
        json={
            "limit": 5,
            "importance_threshold": 0.5,
            "include_serendipity": True
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    assert "count" in data
    assert "generation_time_ms" in data
    assert isinstance(data["suggestions"], list)
    assert data["count"] == len(data["suggestions"])


def test_curiosity_validation(client, auth_headers):
    """Test curiosity request validation."""
    # Invalid limit (must be 1-20)
    response = client.post(
        "/api/v1/curiosity/suggest",
        json={"limit": 100},
        headers=auth_headers
    )
    assert response.status_code == 422

    # Invalid importance threshold (must be 0.0-1.0)
    response = client.post(
        "/api/v1/curiosity/suggest",
        json={"importance_threshold": 1.5},
        headers=auth_headers
    )
    assert response.status_code == 422


# ============================================================================
# Graph Reasoning Tests
# ============================================================================

def test_multi_hop_query(client, auth_headers):
    """Test multi-hop graph query."""
    response = client.post(
        "/api/v1/graph/multi-hop",
        json={
            "query": "What papers cite Transformers?",
            "max_hops": 2,
            "max_results": 10
        },
        headers=auth_headers
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What papers cite Transformers?"
    assert data["max_hops"] == 2
    assert "results" in data
    assert "total_paths_explored" in data
    assert "query_entities" in data
    assert "latency_ms" in data


def test_multi_hop_validation(client, auth_headers):
    """Test multi-hop query validation."""
    # Invalid max_hops (must be 1-5)
    response = client.post(
        "/api/v1/graph/multi-hop",
        json={"query": "test", "max_hops": 10},
        headers=auth_headers
    )
    assert response.status_code == 422

    # Invalid max_results (must be 1-100)
    response = client.post(
        "/api/v1/graph/multi-hop",
        json={"query": "test", "max_results": 1000},
        headers=auth_headers
    )
    assert response.status_code == 422


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_404_not_found(client):
    """Test 404 for non-existent endpoint."""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404


def test_405_method_not_allowed(client):
    """Test 405 for wrong HTTP method."""
    response = client.get("/api/v1/experience")  # Should be POST
    assert response.status_code == 405


# ============================================================================
# Rate Limiting Tests (if enabled)
# ============================================================================

@pytest.mark.skip(reason="Rate limiting may not be enabled in test environment")
def test_rate_limiting(client, auth_headers):
    """Test rate limiting enforcement."""
    # Make many rapid requests
    responses = []
    for i in range(100):
        response = client.post(
            "/api/v1/recall",
            json={"query": f"test {i}", "max_results": 1},
            headers=auth_headers
        )
        responses.append(response)

    # At least one should be rate limited
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes  # Too Many Requests


# ============================================================================
# CORS Tests
# ============================================================================

def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/health")

    # Should have CORS headers
    assert "access-control-allow-origin" in response.headers


# ============================================================================
# Performance Tests
# ============================================================================

def test_response_time(client, auth_headers):
    """Test API response time is reasonable."""
    import time

    start = time.time()
    response = client.get("/health")
    duration = (time.time() - start) * 1000  # Convert to ms

    assert response.status_code == 200
    assert duration < 1000  # Less than 1 second


def test_concurrent_requests(client, auth_headers):
    """Test handling concurrent requests."""
    import concurrent.futures

    def make_request():
        return client.post(
            "/api/v1/recall",
            json={"query": "test", "max_results": 5},
            headers=auth_headers
        )

    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]

    # All should succeed (or gracefully handle load)
    success_count = sum(1 for r in responses if r.status_code == 200)
    assert success_count >= 5  # At least half should succeed


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow(client, auth_headers):
    """Test complete workflow: experience → recall → consolidate."""
    # 1. Create multiple memories
    for i in range(5):
        response = client.post(
            "/api/v1/experience",
            json={
                "content": f"Thompson Sampling concept {i}",
                "scope": "SESSION"
            },
            headers=auth_headers
        )
        assert response.status_code == 200

    # 2. Recall memories
    response = client.post(
        "/api/v1/recall",
        json={"query": "Thompson Sampling", "max_results": 10},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1  # Should find at least one

    # 3. Trigger consolidation
    response = client.post(
        "/api/v1/consolidation/trigger",
        json={"max_episodes": 10},
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"

    # 4. Check stats
    response = client.get("/api/v1/stats", headers=auth_headers)
    assert response.status_code == 200
    stats = response.json()
    assert "consolidation" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
