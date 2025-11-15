"""
Tests for SIEM Integration

Created: 2025-11-15
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from HoloLoom.security.siem import (
    SIEMIntegration,
    SIEMConfig,
    LogBuffer,
    SecurityEventCategory,
    SecurityEventSubcategory,
    SeverityLevel,
    create_security_event,
    PIIRedactor,
    get_taxonomy_stats,
)
from HoloLoom.security.siem.core import CircuitBreaker, CircuitState


class MockSIEMBackend:
    """Mock SIEM backend for testing"""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.events_received = []
        self.call_count = 0

    async def send_events(self, events):
        self.call_count += 1
        if self.should_fail:
            return False
        self.events_received.extend(events)
        return True

    async def query_events(self, start_time, end_time, filters=None):
        return self.events_received

    async def health_check(self):
        return not self.should_fail


@pytest.mark.asyncio
class TestSecurityEvent:
    """Test SecurityEvent creation and serialization"""

    async def test_create_security_event(self):
        """Test creating a security event"""
        event = create_security_event(
            category=SecurityEventCategory.ATTACK,
            subcategory=SecurityEventSubcategory.SQL_INJECTION,
            action="query",
            blocked=True,
            risk_score=8.5,
            source_ip="192.168.1.100",
            user_id="user123",
            payload="SELECT * FROM users WHERE id='1' OR '1'='1'",
        )

        assert event.category == SecurityEventCategory.ATTACK
        assert event.subcategory == SecurityEventSubcategory.SQL_INJECTION
        assert event.blocked is True
        assert event.risk_score == 8.5
        assert event.level == SeverityLevel.CRITICAL  # Auto-inferred
        assert event.event_id  # Auto-generated

    async def test_event_to_dict(self):
        """Test event serialization"""
        event = create_security_event(
            category=SecurityEventCategory.AUTH,
            subcategory=SecurityEventSubcategory.LOGIN,
            action="login",
            blocked=False,
            risk_score=1.0,
            user_id="user123",
        )

        event_dict = event.to_dict()
        assert event_dict["category"] == "AUTH"
        assert event_dict["subcategory"] == "login"
        assert "timestamp" in event_dict
        assert "event_id" in event_dict

    async def test_event_roundtrip(self):
        """Test event serialization and deserialization"""
        from HoloLoom.security.siem.taxonomy import SecurityEvent

        event = create_security_event(
            category=SecurityEventCategory.DATA,
            subcategory=SecurityEventSubcategory.EXPORT,
            action="export",
            blocked=False,
            risk_score=3.0,
        )

        event_dict = event.to_dict()
        restored_event = SecurityEvent.from_dict(event_dict)

        assert restored_event.category == event.category
        assert restored_event.subcategory == event.subcategory
        assert restored_event.action == event.action
        assert restored_event.blocked == event.blocked


@pytest.mark.asyncio
class TestPIIRedaction:
    """Test PII redaction"""

    async def test_redact_email(self):
        """Test email redaction"""
        text = "Contact user@example.com for help"
        redacted = PIIRedactor.redact(text)
        assert "[EMAIL_REDACTED]" in redacted
        assert "user@example.com" not in redacted

    async def test_redact_ssn(self):
        """Test SSN redaction"""
        text = "SSN: 123-45-6789"
        redacted = PIIRedactor.redact(text)
        assert "[SSN_REDACTED]" in redacted
        assert "123-45-6789" not in redacted

    async def test_redact_credit_card(self):
        """Test credit card redaction"""
        text = "Card: 1234-5678-9012-3456"
        redacted = PIIRedactor.redact(text)
        assert "[CC_REDACTED]" in redacted
        assert "1234-5678-9012-3456" not in redacted

    async def test_preserve_ips(self):
        """Test preserving IP addresses"""
        text = "Attack from 192.168.1.100"
        redacted = PIIRedactor.redact(text, preserve_ips=True)
        assert "192.168.1.100" in redacted

    async def test_redact_ips(self):
        """Test redacting IP addresses"""
        text = "Attack from 192.168.1.100"
        redacted = PIIRedactor.redact(text, preserve_ips=False)
        assert "[IP_REDACTED]" in redacted
        assert "192.168.1.100" not in redacted

    async def test_hash_user_id(self):
        """Test user ID hashing"""
        user_id = "user123"
        hashed = PIIRedactor.hash_user_id(user_id)
        assert hashed != user_id
        assert len(hashed) == 16  # Truncated hash

        # Same input produces same hash
        hashed2 = PIIRedactor.hash_user_id(user_id)
        assert hashed == hashed2


@pytest.mark.asyncio
class TestLogBuffer:
    """Test log buffer"""

    async def test_add_and_get(self):
        """Test adding and getting events"""
        buffer = LogBuffer(max_size=10)

        events = [
            create_security_event(
                category=SecurityEventCategory.AUTH,
                subcategory=SecurityEventSubcategory.LOGIN,
                action=f"login_{i}",
                blocked=False,
                risk_score=1.0,
            )
            for i in range(5)
        ]

        for event in events:
            await buffer.add(event)

        assert await buffer.size() == 5

        batch = await buffer.get_batch(3)
        assert len(batch) == 3
        assert await buffer.size() == 2

    async def test_buffer_overflow(self):
        """Test buffer overflow behavior"""
        buffer = LogBuffer(max_size=5)

        # Add more than max_size
        for i in range(10):
            event = create_security_event(
                category=SecurityEventCategory.SYSTEM,
                subcategory=SecurityEventSubcategory.STARTUP,
                action=f"event_{i}",
                blocked=False,
                risk_score=1.0,
            )
            await buffer.add(event)

        # Buffer should contain only last 5
        assert await buffer.size() == 5

    async def test_clear(self):
        """Test buffer clearing"""
        buffer = LogBuffer(max_size=10)

        for i in range(5):
            event = create_security_event(
                category=SecurityEventCategory.AUTH,
                subcategory=SecurityEventSubcategory.LOGOUT,
                action=f"logout_{i}",
                blocked=False,
                risk_score=1.0,
            )
            await buffer.add(event)

        await buffer.clear()
        assert await buffer.size() == 0


@pytest.mark.asyncio
class TestCircuitBreaker:
    """Test circuit breaker"""

    async def test_closed_state(self):
        """Test circuit breaker in closed state"""
        cb = CircuitBreaker(failure_threshold=3)

        assert cb.state == CircuitState.CLOSED
        assert cb.can_attempt() is True

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    async def test_open_state(self):
        """Test circuit breaker opening"""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_attempt() is False

    async def test_half_open_state(self):
        """Test circuit breaker half-open state"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        # Open circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should try half-open
        assert cb.can_attempt() is True
        assert cb.state == CircuitState.HALF_OPEN

    async def test_recovery(self):
        """Test circuit breaker recovery"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1, success_threshold=2)

        # Open circuit
        for _ in range(3):
            cb.record_failure()

        # Wait and go half-open
        await asyncio.sleep(0.15)
        cb.can_attempt()

        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
class TestSIEMIntegration:
    """Test SIEM integration"""

    async def test_basic_logging(self):
        """Test basic event logging"""
        backend = MockSIEMBackend()
        config = SIEMConfig(
            backend="mock",
            buffer_size=10,
            flush_interval=0.1,
            batch_size=5,
        )

        async with SIEMIntegration(config, backend) as siem:
            # Log events
            for i in range(3):
                event = create_security_event(
                    category=SecurityEventCategory.AUTH,
                    subcategory=SecurityEventSubcategory.LOGIN,
                    action=f"login_{i}",
                    blocked=False,
                    risk_score=1.0,
                )
                await siem.log_event(event)

            # Wait for flush
            await asyncio.sleep(0.2)

        # Check events were sent
        assert len(backend.events_received) == 3
        assert siem.stats["events_logged"] == 3
        assert siem.stats["events_sent"] == 3

    async def test_batch_logging(self):
        """Test batch event logging"""
        backend = MockSIEMBackend()
        config = SIEMConfig(
            backend="mock",
            buffer_size=100,
            flush_interval=0.1,
            batch_size=10,
        )

        async with SIEMIntegration(config, backend) as siem:
            events = [
                create_security_event(
                    category=SecurityEventCategory.DATA,
                    subcategory=SecurityEventSubcategory.QUERY,
                    action=f"query_{i}",
                    blocked=False,
                    risk_score=2.0,
                )
                for i in range(15)
            ]
            await siem.log_events(events)

            # Wait for flush
            await asyncio.sleep(0.2)

        # Check batching
        assert len(backend.events_received) == 15
        assert backend.call_count >= 2  # At least 2 batches (10 + 5)

    async def test_backend_failure_fallback(self):
        """Test fallback on backend failure"""
        backend = MockSIEMBackend(should_fail=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SIEMConfig(
                backend="mock",
                buffer_size=10,
                flush_interval=0.1,
                batch_size=5,
                fallback_dir=Path(tmpdir),
                enable_fallback=True,
            )

            async with SIEMIntegration(config, backend) as siem:
                event = create_security_event(
                    category=SecurityEventCategory.ATTACK,
                    subcategory=SecurityEventSubcategory.XSS,
                    action="xss_attempt",
                    blocked=True,
                    risk_score=7.0,
                )
                await siem.log_event(event)

                # Wait for flush
                await asyncio.sleep(0.2)

            # Check fallback was used
            assert siem.stats["fallback_writes"] > 0

            # Check fallback file exists
            fallback_files = list(Path(tmpdir).glob("security_events_*.json"))
            assert len(fallback_files) > 0

    async def test_health_check(self):
        """Test health check"""
        backend = MockSIEMBackend()
        config = SIEMConfig(backend="mock")

        async with SIEMIntegration(config, backend) as siem:
            health = await siem.health_check()

            assert health["backend_healthy"] is True
            assert health["circuit_state"] == "closed"
            assert "buffer_size" in health
            assert "stats" in health

    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration"""
        backend = MockSIEMBackend(should_fail=True)
        config = SIEMConfig(
            backend="mock",
            buffer_size=10,
            flush_interval=0.05,
            batch_size=1,
            failure_threshold=2,
            max_retries=1,
        )

        async with SIEMIntegration(config, backend) as siem:
            # Log events that will fail
            for i in range(5):
                event = create_security_event(
                    category=SecurityEventCategory.SYSTEM,
                    subcategory=SecurityEventSubcategory.SERVICE_ERROR,
                    action=f"error_{i}",
                    blocked=False,
                    risk_score=5.0,
                )
                await siem.log_event(event)

            # Wait for flushes
            await asyncio.sleep(0.3)

        # Circuit should be open
        assert siem.circuit_breaker.state == CircuitState.OPEN
        assert siem.stats["backend_failures"] > 0


@pytest.mark.asyncio
class TestTaxonomy:
    """Test taxonomy utilities"""

    async def test_taxonomy_stats(self):
        """Test taxonomy statistics"""
        stats = get_taxonomy_stats()

        assert stats["categories"] == 6  # AUTH, AUTHZ, ATTACK, DATA, SYSTEM, INCIDENT
        assert stats["subcategories"] > 20  # Many subcategories
        assert stats["severity_levels"] == 5  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    async def test_severity_inference(self):
        """Test automatic severity inference"""
        # High-risk attack should be CRITICAL
        event1 = create_security_event(
            category=SecurityEventCategory.ATTACK,
            subcategory=SecurityEventSubcategory.SQL_INJECTION,
            action="attack",
            blocked=True,
            risk_score=9.0,
        )
        assert event1.level == SeverityLevel.CRITICAL

        # Failed auth should be WARNING
        event2 = create_security_event(
            category=SecurityEventCategory.AUTH,
            subcategory=SecurityEventSubcategory.FAILED_AUTH,
            action="login",
            blocked=True,
            risk_score=3.0,
        )
        assert event2.level == SeverityLevel.WARNING

        # Low-risk event should be INFO
        event3 = create_security_event(
            category=SecurityEventCategory.DATA,
            subcategory=SecurityEventSubcategory.QUERY,
            action="query",
            blocked=False,
            risk_score=1.0,
        )
        assert event3.level == SeverityLevel.INFO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
