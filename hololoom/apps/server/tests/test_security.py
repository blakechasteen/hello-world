"""
Comprehensive security tests for HoloLoom API

Tests all security components:
- WAF rules
- Request signing
- Security monitoring
- Rate limiting

Created: 2025-11-26
"""

import pytest
import asyncio
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient

from hololoom.apps.server.waf_middleware import (
    WAFMiddleware,
    SQLInjectionRule,
    XSSRule,
    PathTraversalRule,
    CommandInjectionRule,
    HeaderValidationRule,
    RequestSizeLimitRule
)
from hololoom.apps.server.request_signing import (
    SignatureValidator,
    RequestSigningClient,
    APIKey,
    APIKeyStatus
)
from hololoom.apps.server.security_monitor import (
    SecurityMonitor,
    SecurityEventType,
    AlertSeverity,
    SecurityEvent
)
from hololoom.apps.server.ar_api_secured import (
    EnhancedRateLimiter,
    create_secured_app,
    SecurityConfig
)


# ============================================================================
# WAF Tests
# ============================================================================

class TestWAFRules:
    """Test WAF rule detection"""

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        rule = SQLInjectionRule()

        # Create mock request with SQL injection
        request = Mock(spec=Request)
        request.url = "http://api.example.com/query?id=1' OR '1'='1"
        request.headers = {"content-type": "application/json"}

        # Should detect SQL injection
        violation = rule.check(request, b"")
        assert violation is not None
        assert "SQL injection" in violation

        # Test body detection
        body = b'{"query": "SELECT * FROM users WHERE id=1 OR 1=1"}'
        violation = rule.check(request, body)
        assert violation is not None

    def test_xss_detection(self):
        """Test XSS pattern detection"""
        rule = XSSRule()

        request = Mock(spec=Request)
        request.url = "http://api.example.com/query"
        request.headers = {}

        # Test script tag detection
        body = b'{"html": "<script>alert(1)</script>"}'
        violation = rule.check(request, body)
        assert violation is not None
        assert "XSS pattern" in violation

        # Test event handler detection
        body = b'{"html": "<img src=x onerror=alert(1)>"}'
        violation = rule.check(request, body)
        assert violation is not None

    def test_path_traversal_detection(self):
        """Test path traversal detection"""
        rule = PathTraversalRule()

        request = Mock(spec=Request)
        request.url = "http://api.example.com/file?path=../../etc/passwd"
        request.headers = {}

        violation = rule.check(request, b"")
        assert violation is not None
        assert "Path traversal" in violation

    def test_command_injection_detection(self):
        """Test command injection detection"""
        rule = CommandInjectionRule()

        request = Mock(spec=Request)
        request.url = "http://api.example.com/exec"
        request.headers = {}

        # Test command chaining
        body = b'{"cmd": "ls && cat /etc/passwd"}'
        violation = rule.check(request, body)
        assert violation is not None
        assert "Command injection" in violation

        # Test backticks
        body = b'{"cmd": "`whoami`"}'
        violation = rule.check(request, body)
        assert violation is not None

    def test_header_validation(self):
        """Test header validation"""
        rule = HeaderValidationRule()

        request = Mock(spec=Request)
        request.url = "http://api.example.com/"

        # Test oversized header
        request.headers = {
            "x-large-header": "A" * 10000  # Exceeds 8KB limit
        }
        violation = rule.check(request, b"")
        assert violation is not None
        assert "exceeds maximum size" in violation

        # Test header injection
        request.headers = {
            "x-bad-header": "value\r\nX-Injected: true"
        }
        violation = rule.check(request, b"")
        assert violation is not None
        assert "Line break detected" in violation

    def test_request_size_limit(self):
        """Test request size limits"""
        rule = RequestSizeLimitRule(max_body_size=1024)  # 1KB limit

        request = Mock(spec=Request)
        request.url = "http://api.example.com/"
        request.headers = {}

        # Test oversized body
        body = b"A" * 2048  # 2KB
        violation = rule.check(request, body)
        assert violation is not None
        assert "exceeds maximum size" in violation

    @pytest.mark.asyncio
    async def test_waf_middleware_integration(self):
        """Test WAF middleware integration"""
        app = Mock()
        middleware = WAFMiddleware(
            app=app,
            enabled=True,
            block_on_violation=True
        )

        # Create request with SQL injection
        request = Mock(spec=Request)
        request.url = "http://api.example.com/query?id=1' OR '1'='1"
        request.headers = {}
        request.method = "GET"
        request.client = Mock(host="192.168.1.100")

        # Mock call_next
        async def call_next(req):
            return Mock(headers={})

        # Should block request
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 400

        # Check response content
        content = json.loads(response.body.decode())
        assert content["error"] == "Bad Request"
        assert "WAF" in content["rule_id"]


# ============================================================================
# Request Signing Tests
# ============================================================================

class TestRequestSigning:
    """Test request signing and validation"""

    def test_api_key_generation(self):
        """Test API key generation"""
        validator = SignatureValidator()

        key_id, secret = validator.generate_api_key(
            client_name="Test Client",
            expires_in_days=30,
            allowed_ips=["127.0.0.1"],
            rate_limit_qps=10.0
        )

        assert key_id.startswith("holo_")
        assert len(secret) > 30
        assert key_id in validator.api_keys

        api_key = validator.api_keys[key_id]
        assert api_key.client_name == "Test Client"
        assert api_key.status == APIKeyStatus.ACTIVE
        assert api_key.rate_limit_qps == 10.0

    def test_signature_calculation(self):
        """Test HMAC signature calculation"""
        validator = SignatureValidator()

        secret = "test_secret"
        method = "POST"
        url = "https://api.example.com/endpoint"
        timestamp = str(int(time.time()))
        body = b'{"test": "data"}'
        nonce = "test_nonce"

        signature = validator.calculate_signature(
            secret=secret,
            method=method,
            url=url,
            timestamp=timestamp,
            body=body,
            nonce=nonce
        )

        # Signature should be base64 encoded
        assert signature
        assert len(signature) > 20

        # Verify signature format
        try:
            base64.b64decode(signature)
        except Exception:
            pytest.fail("Signature is not valid base64")

    @pytest.mark.asyncio
    async def test_request_validation(self):
        """Test request validation"""
        validator = SignatureValidator()

        # Generate API key
        key_id, secret = validator.generate_api_key("Test Client")

        # Create signed request
        client = RequestSigningClient(key_id, secret)
        headers = client.sign_request(
            method="POST",
            url="https://api.example.com/endpoint",
            body=b'{"test": "data"}'
        )

        # Create mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = "https://api.example.com/endpoint"
        request.headers = headers
        request.client = Mock(host="127.0.0.1")

        # Mock body reading
        async def body():
            return b'{"test": "data"}'
        request.body = body

        # Should validate successfully
        api_key = await validator.validate_request(request)
        assert api_key.key_id == key_id
        assert api_key.client_name == "Test Client"

    @pytest.mark.asyncio
    async def test_expired_key_rejection(self):
        """Test expired key rejection"""
        validator = SignatureValidator()

        # Generate key with immediate expiration
        key_id, secret = validator.generate_api_key(
            "Test Client",
            expires_in_days=0
        )

        # Manually expire the key
        validator.api_keys[key_id].expires_at = datetime.utcnow() - timedelta(days=1)

        # Create signed request
        client = RequestSigningClient(key_id, secret)
        headers = client.sign_request("GET", "https://api.example.com/")

        # Create mock request
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = "https://api.example.com/"
        request.headers = headers
        request.client = Mock(host="127.0.0.1")
        request.body = AsyncMock(return_value=b"")

        # Should reject expired key
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_request(request)

        assert exc_info.value.status_code == 401
        assert "expired" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_replay_protection(self):
        """Test nonce-based replay protection"""
        validator = SignatureValidator(
            require_nonce=True,
            enforce_replay_protection=True
        )

        # Generate API key
        key_id, secret = validator.generate_api_key("Test Client")

        # Create signed request with nonce
        client = RequestSigningClient(key_id, secret)
        headers = client.sign_request(
            method="POST",
            url="https://api.example.com/",
            include_nonce=True
        )

        # Create mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = "https://api.example.com/"
        request.headers = headers
        request.client = Mock(host="127.0.0.1")
        request.body = AsyncMock(return_value=b"")

        # First request should succeed
        await validator.validate_request(request)

        # Replay same request should fail
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_request(request)

        assert exc_info.value.status_code == 401
        assert "Nonce already used" in str(exc_info.value.detail)


# ============================================================================
# Security Monitoring Tests
# ============================================================================

class TestSecurityMonitoring:
    """Test security monitoring system"""

    @pytest.mark.asyncio
    async def test_event_logging(self):
        """Test security event logging"""
        monitor = SecurityMonitor()

        await monitor.log_event(
            event_type=SecurityEventType.AUTH_FAILURE,
            client_ip="192.168.1.100",
            severity=AlertSeverity.WARNING,
            details={"username": "admin", "reason": "Invalid password"}
        )

        assert len(monitor.events) == 1
        event = monitor.events[0]
        assert event.event_type == SecurityEventType.AUTH_FAILURE
        assert event.client_ip == "192.168.1.100"
        assert event.details["username"] == "admin"

    @pytest.mark.asyncio
    async def test_threshold_alerting(self):
        """Test threshold-based alerting"""
        monitor = SecurityMonitor(
            alert_threshold_auth_failures=3,
            time_window_seconds=60
        )

        # Log multiple auth failures
        for i in range(3):
            await monitor.log_auth_failure(
                client_ip="192.168.1.100",
                username=f"user{i}"
            )

        # Should trigger alert
        assert len(monitor.active_alerts) == 1
        alert = list(monitor.active_alerts.values())[0]
        assert alert.severity == AlertSeverity.HIGH
        assert "Brute Force" in alert.title

    @pytest.mark.asyncio
    async def test_attacker_profiling(self):
        """Test attacker profile creation"""
        monitor = SecurityMonitor()

        # Log events from same IP
        for i in range(5):
            await monitor.log_event(
                event_type=SecurityEventType.WAF_VIOLATION,
                client_ip="192.168.1.100",
                severity=AlertSeverity.HIGH
            )

        # Check attacker profile
        assert "192.168.1.100" in monitor.attacker_profiles
        profile = monitor.attacker_profiles["192.168.1.100"]
        assert profile.total_events == 5
        assert profile.risk_score > 0

    @pytest.mark.asyncio
    async def test_auto_blocking(self):
        """Test automatic IP blocking"""
        monitor = SecurityMonitor()

        # Create high-risk profile
        for i in range(10):
            await monitor.log_event(
                event_type=SecurityEventType.BRUTE_FORCE,
                client_ip="192.168.1.100",
                severity=AlertSeverity.CRITICAL
            )

        # Should auto-block high-risk IP
        assert monitor.is_ip_blocked("192.168.1.100")

    def test_prometheus_metrics(self):
        """Test Prometheus metrics export"""
        monitor = SecurityMonitor()

        # Update some metrics
        monitor.metrics["security_events_total"]["auth_failure"] = 10
        monitor.metrics["active_alerts_count"] = 2
        monitor.metrics["blocked_ips_count"] = 1

        metrics = monitor.get_prometheus_metrics()

        assert "security_events_total" in metrics
        assert 'auth_failure"} 10' in metrics
        assert "active_alerts_count 2" in metrics
        assert "blocked_ips_count 1" in metrics


# ============================================================================
# Rate Limiting Tests
# ============================================================================

class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting"""
        limiter = EnhancedRateLimiter(requests=5, window=10)

        client_ip = "192.168.1.100"
        endpoint = "/test"

        # First 5 requests should pass
        for i in range(5):
            allowed = await limiter.check_rate_limit(client_ip, endpoint)
            assert allowed

        # 6th request should be blocked
        allowed = await limiter.check_rate_limit(client_ip, endpoint)
        assert not allowed

    @pytest.mark.asyncio
    async def test_sliding_window(self):
        """Test sliding window behavior"""
        limiter = EnhancedRateLimiter(requests=2, window=2)

        client_ip = "192.168.1.100"
        endpoint = "/test"

        # Two requests
        assert await limiter.check_rate_limit(client_ip, endpoint)
        assert await limiter.check_rate_limit(client_ip, endpoint)

        # Third should be blocked
        assert not await limiter.check_rate_limit(client_ip, endpoint)

        # Wait for window to slide
        await asyncio.sleep(2.1)

        # Should allow again
        assert await limiter.check_rate_limit(client_ip, endpoint)

    @pytest.mark.asyncio
    async def test_auto_blocking(self):
        """Test auto-blocking after excessive violations"""
        limiter = EnhancedRateLimiter(requests=2, window=60)

        client_ip = "192.168.1.100"
        endpoint = "/test"

        # Exceed limit multiple times
        for i in range(10):
            await limiter.check_rate_limit(client_ip, endpoint)

        # Should be auto-blocked
        assert client_ip in limiter.blocked_ips

        # Even after unblocking and waiting, should remain blocked
        await asyncio.sleep(1)
        assert not await limiter.check_rate_limit(client_ip, endpoint)


# ============================================================================
# Integration Tests
# ============================================================================

class TestSecurityIntegration:
    """Test integrated security features"""

    def test_secured_app_creation(self):
        """Test creation of secured FastAPI app"""
        config = SecurityConfig(
            enable_waf=True,
            enable_signing=True,
            enable_monitoring=True,
            enable_rate_limiting=True
        )

        app = create_secured_app(config)

        # Check middleware added
        assert any("WAFMiddleware" in str(m) for m in app.middleware)

        # Check state initialized
        assert hasattr(app.state, 'security_monitor')
        assert hasattr(app.state, 'signature_validator')
        assert hasattr(app.state, 'rate_limiter')

    def test_health_endpoint(self):
        """Test health check endpoint"""
        app = create_secured_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "security" in data

    @pytest.mark.asyncio
    async def test_waf_blocking(self):
        """Test WAF blocks malicious requests"""
        app = create_secured_app(
            SecurityConfig(enable_waf=True, waf_block_on_violation=True)
        )
        client = TestClient(app)

        # Send request with SQL injection
        response = client.post(
            "/ar/query",
            json={"text": "test' OR '1'='1"}
        )

        # Should be blocked by WAF
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test rate limiting enforcement"""
        app = create_secured_app(
            SecurityConfig(enable_rate_limiting=True)
        )

        # Override rate limit for testing
        app.state.rate_limiter = EnhancedRateLimiter(requests=2, window=60)

        client = TestClient(app)

        # Send multiple requests
        for i in range(3):
            response = client.post(
                "/ar/query",
                json={"text": f"test {i}"}
            )

            if i < 2:
                # First 2 should succeed (or fail auth)
                assert response.status_code in [401, 200]
            else:
                # Third should be rate limited
                assert response.status_code == 429


# ============================================================================
# Performance Tests
# ============================================================================

class TestSecurityPerformance:
    """Test performance impact of security features"""

    def test_waf_performance(self):
        """Test WAF rule checking performance"""
        rule = SQLInjectionRule()

        request = Mock(spec=Request)
        request.url = "http://api.example.com/query?id=123"
        request.headers = {}
        body = b'{"query": "normal query without injection"}'

        # Measure time for 1000 checks
        start = time.time()
        for _ in range(1000):
            rule.check(request, body)
        duration = time.time() - start

        # Should be fast (< 1 second for 1000 checks)
        assert duration < 1.0

        # Average per check should be < 1ms
        avg_ms = (duration / 1000) * 1000
        assert avg_ms < 1.0

    def test_signature_performance(self):
        """Test signature calculation performance"""
        validator = SignatureValidator()

        secret = "test_secret"
        method = "POST"
        url = "https://api.example.com/endpoint"
        body = b'{"test": "data"}' * 100  # Larger body

        # Measure time for 1000 signatures
        start = time.time()
        for _ in range(1000):
            timestamp = str(int(time.time()))
            validator.calculate_signature(
                secret=secret,
                method=method,
                url=url,
                timestamp=timestamp,
                body=body
            )
        duration = time.time() - start

        # Should be fast (< 1 second for 1000 signatures)
        assert duration < 1.0

    @pytest.mark.asyncio
    async def test_monitoring_performance(self):
        """Test security monitoring performance"""
        monitor = SecurityMonitor()

        # Measure time for 1000 event logs
        start = time.time()
        for i in range(1000):
            await monitor.log_event(
                event_type=SecurityEventType.AUTH_FAILURE,
                client_ip=f"192.168.1.{i % 256}",
                severity=AlertSeverity.WARNING
            )
        duration = time.time() - start

        # Should be fast (< 1 second for 1000 events)
        assert duration < 1.0

        # Check events were logged
        assert len(monitor.events) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])