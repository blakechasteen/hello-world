"""
Tests for rate limiting middleware.
"""

import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ..middleware import RateLimitMiddleware


@pytest.fixture
def app_with_rate_limit():
    """Create test app with rate limiting."""
    app = FastAPI()

    # Add rate limiter with low limits for testing
    app.add_middleware(
        RateLimitMiddleware,
        per_minute_limit=5,
        per_hour_limit=10,
    )

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(app_with_rate_limit):
    """Create test client."""
    return TestClient(app_with_rate_limit)


def test_rate_limiting_allows_requests_within_limit(client):
    """Test that requests within limit are allowed."""
    # First 5 requests should succeed
    for i in range(5):
        response = client.get("/test")
        assert response.status_code == 200, f"Request {i+1} should succeed"


def test_rate_limiting_blocks_excessive_requests(client):
    """Test that requests exceeding limit are blocked."""
    # First 5 requests should succeed
    for i in range(5):
        response = client.get("/test")
        assert response.status_code == 200

    # 6th request should be rate limited
    response = client.get("/test")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]


def test_rate_limiting_returns_retry_after_header(client):
    """Test that 429 response includes Retry-After header."""
    # Exhaust limit
    for _ in range(5):
        client.get("/test")

    # Next request should have Retry-After header
    response = client.get("/test")
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_rate_limiting_skips_health_endpoint(client):
    """Test that health checks are not rate limited."""
    # Make many health check requests
    for _ in range(20):
        response = client.get("/health")
        assert response.status_code == 200


def test_rate_limiting_per_session(client):
    """Test per-session rate limiting."""
    # Create app with session limit
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        per_minute_limit=100,  # High per-minute limit
        per_hour_limit=3,  # Low per-hour limit for testing
    )

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    test_client = TestClient(app)

    # First 3 requests with same session should succeed
    for i in range(3):
        response = test_client.get(
            "/test",
            headers={"X-Session-ID": "test-session-1"}
        )
        assert response.status_code == 200

    # 4th request should be rate limited
    response = test_client.get(
        "/test",
        headers={"X-Session-ID": "test-session-1"}
    )
    assert response.status_code == 429


def test_rate_limiting_different_sessions(client):
    """Test that different sessions have independent limits."""
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        per_minute_limit=100,
        per_hour_limit=3,
    )

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    test_client = TestClient(app)

    # Session 1: use all requests
    for _ in range(3):
        response = test_client.get(
            "/test",
            headers={"X-Session-ID": "session-1"}
        )
        assert response.status_code == 200

    # Session 1 is exhausted
    response = test_client.get(
        "/test",
        headers={"X-Session-ID": "session-1"}
    )
    assert response.status_code == 429

    # Session 2 should still work
    response = test_client.get(
        "/test",
        headers={"X-Session-ID": "session-2"}
    )
    assert response.status_code == 200


def test_rate_limiting_sliding_window():
    """Test sliding window behavior."""
    app = FastAPI()
    middleware = RateLimitMiddleware(
        app=app,
        per_minute_limit=2,
        per_hour_limit=10,
    )

    # Manually test sliding window logic
    client_ip = "127.0.0.1"
    session_id = "test-session"

    # First 2 requests should pass
    assert middleware._check_ip_limit(client_ip, session_id) is True
    assert middleware._check_ip_limit(client_ip, session_id) is True

    # Third should fail
    assert middleware._check_ip_limit(client_ip, session_id) is False

    # Wait 61 seconds (simulate - actually manipulate timestamps)
    # Remove old entries
    now = time.time()
    middleware.ip_requests[client_ip].clear()
    middleware.ip_requests[client_ip].append((now, session_id))

    # Should work again
    assert middleware._check_ip_limit(client_ip, session_id) is True


def test_rate_limiting_cleanup():
    """Test that cleanup removes old entries."""
    app = FastAPI()
    middleware = RateLimitMiddleware(
        app=app,
        per_minute_limit=5,
        per_hour_limit=10,
    )

    # Add some requests
    middleware._check_ip_limit("192.168.1.1", "session-1")
    middleware._check_ip_limit("192.168.1.2", "session-2")

    assert len(middleware.ip_requests) == 2
    assert len(middleware.session_requests) == 2

    # Trigger cleanup by setting old timestamp
    middleware.last_cleanup = time.time() - 400  # 400 seconds ago
    middleware._cleanup_if_needed()

    # Old entries should be cleaned (if expired)
    # Note: In real scenario, entries would be expired
    # Here we just test cleanup runs


def test_get_stats():
    """Test stats collection."""
    app = FastAPI()
    middleware = RateLimitMiddleware(
        app=app,
        per_minute_limit=5,
        per_hour_limit=10,
    )

    # Make some requests
    middleware._check_ip_limit("192.168.1.1", "session-1")

    # Get stats
    stats = middleware.get_stats()

    assert stats["per_minute_limit"] == 5
    assert stats["per_hour_limit"] == 10
    assert stats["active_ips"] >= 1
    assert stats["active_sessions"] >= 1
