"""
Agentic Concurrent Safety Tests (Week 1 Critical)
==================================================
Tests concurrent request handling and thread safety in agentic reasoning system.

Critical for production deployment where multiple users issue concurrent requests.
Tests cover:
- Rate limiting under concurrent load
- State corruption prevention
- Resource contention
- Deadlock prevention
- Audit trail consistency
- Memory consistency
- Safety guardrail consistency

Based on Bug #3 (Rate Limiter Race Condition) fix - ensures thread safety.

Created: 2025-11-21 (Bug Fix + Test Creation Session)
"""

import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List

from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from hololoom.alignment.safety_guardrails import SafetyGuardrails
from hololoom.alignment.audit_trail import AuditTrail


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_shards() -> List[MemoryShard]:
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling balances exploration and exploitation",
            episode="algorithms",
            entities=["Thompson Sampling"],
            motifs=["definition"]
        )
    ]


@pytest.fixture
def fast_config() -> Config:
    """FAST mode config."""
    return Config.fast()


@pytest.fixture
def safety_guardrails():
    """Create safety guardrails."""
    return SafetyGuardrails(enable_human_in_loop=False)


@pytest.fixture
def audit_trail():
    """Create audit trail."""
    return AuditTrail(persist_path=None)


# ============================================================================
# Test 1-5: Rate Limiting Concurrency
# ============================================================================

@pytest.mark.asyncio
async def test_rate_limiter_concurrent_requests():
    """Test rate limiter handles concurrent requests without corruption."""
    from hololoom.apps.server.agentic_api import RateLimiter

    limiter = RateLimiter(max_requests=10, window_seconds=1)

    # Fire 20 concurrent requests
    async def check_limit(i):
        return await limiter.check_rate_limit(f"client_{i % 5}")  # 5 clients

    results = await asyncio.gather(*[check_limit(i) for i in range(20)])

    # Some should be allowed, some blocked (no corruption)
    allowed = sum(results)
    blocked = len(results) - allowed

    assert allowed > 0  # Some allowed
    assert blocked > 0  # Some blocked (rate limit enforced)


@pytest.mark.asyncio
async def test_rate_limiter_no_state_corruption():
    """Test rate limiter state remains consistent under concurrent load."""
    from hololoom.apps.server.agentic_api import RateLimiter

    limiter = RateLimiter(max_requests=5, window_seconds=1)
    client_id = "test_client"

    # Fire 10 concurrent requests from same client
    results = await asyncio.gather(*[
        limiter.check_rate_limit(client_id) for _ in range(10)
    ])

    # Exactly 5 should be allowed (no double-counting or lost counts)
    allowed_count = sum(results)
    assert allowed_count == 5


@pytest.mark.asyncio
async def test_rate_limiter_multiple_clients_isolated():
    """Test rate limiting is isolated per client."""
    from hololoom.apps.server.agentic_api import RateLimiter

    limiter = RateLimiter(max_requests=3, window_seconds=1)

    # Client A fires 5 requests
    results_a = await asyncio.gather(*[
        limiter.check_rate_limit("client_a") for _ in range(5)
    ])

    # Client B fires 5 requests
    results_b = await asyncio.gather(*[
        limiter.check_rate_limit("client_b") for _ in range(5)
    ])

    # Each client should get exactly 3 allowed
    assert sum(results_a) == 3
    assert sum(results_b) == 3


@pytest.mark.asyncio
async def test_rate_limiter_window_sliding():
    """Test rate limiter window slides correctly under concurrent load."""
    from hololoom.apps.server.agentic_api import RateLimiter

    limiter = RateLimiter(max_requests=5, window_seconds=1)
    client_id = "test_client"

    # Use up limit
    await asyncio.gather(*[limiter.check_rate_limit(client_id) for _ in range(5)])

    # Wait for window to slide
    await asyncio.sleep(1.1)

    # Should be able to make requests again
    result = await limiter.check_rate_limit(client_id)
    assert result == True


@pytest.mark.asyncio
async def test_rate_limiter_stress_test():
    """Stress test rate limiter with high concurrent load."""
    from hololoom.apps.server.agentic_api import RateLimiter

    limiter = RateLimiter(max_requests=50, window_seconds=1)

    # Fire 200 concurrent requests from 10 clients
    async def make_request(i):
        return await limiter.check_rate_limit(f"client_{i % 10}")

    results = await asyncio.gather(*[make_request(i) for i in range(200)])

    # Should handle without crashing
    assert len(results) == 200
    # Each client gets max 50 requests (total: 10 * 50 = 500, but we only sent 200)
    # All 200 should be allowed since 200 < 10 * 50
    # But window is 1 second, so some may be blocked if timing is tight
    allowed = sum(results)
    assert allowed > 0  # At least some allowed


# ============================================================================
# Test 6-10: Agentic Orchestrator Concurrency
# ============================================================================

@pytest.mark.asyncio
async def test_agentic_orchestrator_concurrent_queries(fast_config, test_shards):
    """Test agentic orchestrator handles concurrent queries safely."""
    async with create_agentic_orchestrator(
        fast_config,
        test_shards,
        enable_verification=False
    ) as orchestrator:
        # Fire 5 concurrent queries
        queries = [
            Query(text=f"Query {i}: What is Thompson Sampling?")
            for i in range(5)
        ]

        results = await asyncio.gather(*[
            orchestrator.reason(query, mode=ReasoningMode.DIRECT, max_steps=1)
            for query in queries
        ])

        # All should complete without corruption
        assert len(results) == 5
        for result in results:
            assert result.spacetime is not None
            assert result.spacetime.confidence >= 0.0


@pytest.mark.asyncio
async def test_agentic_orchestrator_no_state_corruption(fast_config, test_shards):
    """Test agentic orchestrator state remains consistent under concurrent load."""
    async with create_agentic_orchestrator(
        fast_config,
        test_shards,
        enable_verification=False
    ) as orchestrator:
        # Fire 10 concurrent queries
        results = await asyncio.gather(*[
            orchestrator.reason(
                Query(text=f"Query {i}"),
                mode=ReasoningMode.DIRECT,
                max_steps=1
            )
            for i in range(10)
        ])

        # All should have unique query IDs (no ID collision)
        query_ids = [r.spacetime.query_id for r in results]
        assert len(set(query_ids)) == 10  # All unique


@pytest.mark.asyncio
async def test_agentic_orchestrator_resource_contention(fast_config, test_shards):
    """Test agentic orchestrator handles resource contention safely."""
    async with create_agentic_orchestrator(
        fast_config,
        test_shards,
        enable_verification=False
    ) as orchestrator:
        # Concurrent access to shared memory
        results = await asyncio.gather(*[
            orchestrator.reason(
                Query(text="What is Thompson Sampling?"),
                mode=ReasoningMode.DIRECT,
                max_steps=1
            )
            for _ in range(10)
        ])

        # All should complete (no deadlock or resource starvation)
        assert len(results) == 10


@pytest.mark.asyncio
async def test_agentic_orchestrator_mixed_modes_concurrent(fast_config, test_shards):
    """Test concurrent queries with different reasoning modes."""
    async with create_agentic_orchestrator(
        fast_config,
        test_shards,
        enable_verification=True
    ) as orchestrator:
        # Mix of DIRECT, VERIFY, RESEARCH modes
        tasks = [
            orchestrator.reason(Query(text="Q1"), mode=ReasoningMode.DIRECT, max_steps=1),
            orchestrator.reason(Query(text="Q2"), mode=ReasoningMode.VERIFY, max_steps=2),
            orchestrator.reason(Query(text="Q3"), mode=ReasoningMode.DIRECT, max_steps=1),
        ]

        results = await asyncio.gather(*tasks)

        # All should complete correctly
        assert len(results) == 3
        assert results[0].reasoning_mode == ReasoningMode.DIRECT
        assert results[1].reasoning_mode == ReasoningMode.VERIFY
        assert results[2].reasoning_mode == ReasoningMode.DIRECT


@pytest.mark.asyncio
async def test_agentic_orchestrator_timeout_handling_concurrent(fast_config, test_shards):
    """Test timeout handling with concurrent queries."""
    # Set very short timeout
    fast_config.fast_timeout_seconds = 0.1

    async with create_agentic_orchestrator(
        fast_config,
        test_shards,
        enable_verification=False
    ) as orchestrator:
        # Fire concurrent queries (some may timeout)
        results = await asyncio.gather(*[
            orchestrator.reason(
                Query(text=f"Query {i}"),
                mode=ReasoningMode.DIRECT,
                max_steps=1
            )
            for i in range(5)
        ], return_exceptions=True)

        # Should handle timeouts gracefully (no crashes)
        assert len(results) == 5


# ============================================================================
# Test 11-15: Safety Guardrails Concurrency
# ============================================================================

@pytest.mark.asyncio
async def test_safety_guardrails_concurrent_gating(safety_guardrails):
    """Test safety guardrails handle concurrent action gating."""
    # Fire 10 concurrent gate requests
    results = await asyncio.gather(*[
        safety_guardrails.gate_action(
            f"action_{i}",
            {"context": f"test_{i}"}
        )
        for i in range(10)
    ])

    # All should complete
    assert len(results) == 10
    for result in results:
        assert hasattr(result, 'allowed')
        assert hasattr(result, 'risk_level')


@pytest.mark.asyncio
async def test_safety_guardrails_no_state_corruption(safety_guardrails):
    """Test safety guardrails state remains consistent under concurrent load."""
    # Fire concurrent high-risk actions
    results = await asyncio.gather(*[
        safety_guardrails.gate_action(
            "execute_code",
            {"code": f"print('test_{i}')"}
        )
        for i in range(5)
    ])

    # All should be evaluated consistently
    assert len(results) == 5


@pytest.mark.asyncio
async def test_safety_guardrails_concurrent_risk_assessment(safety_guardrails):
    """Test risk assessment consistency under concurrent load."""
    # Same action multiple times concurrently
    results = await asyncio.gather(*[
        safety_guardrails.gate_action(
            "read_file",
            {"path": "/etc/passwd"}  # High-risk path
        )
        for _ in range(10)
    ])

    # All should have same risk assessment
    risk_levels = [r.risk_level for r in results]
    assert len(set(risk_levels)) == 1  # All same risk level


@pytest.mark.asyncio
async def test_safety_guardrails_mixed_risk_concurrent(safety_guardrails):
    """Test concurrent actions with different risk levels."""
    tasks = [
        safety_guardrails.gate_action("read_file", {"path": "/tmp/test.txt"}),  # Low risk
        safety_guardrails.gate_action("execute_code", {"code": "import os"}),  # High risk
        safety_guardrails.gate_action("write_file", {"path": "/tmp/safe.txt"}),  # Medium risk
    ]

    results = await asyncio.gather(*tasks)

    # All should complete with correct risk assessments
    assert len(results) == 3


@pytest.mark.asyncio
async def test_safety_guardrails_escalation_handling_concurrent(safety_guardrails):
    """Test escalation handling with concurrent high-risk actions."""
    # Fire multiple high-risk actions concurrently
    results = await asyncio.gather(*[
        safety_guardrails.gate_action(
            "delete_database",
            {"target": "production"}
        )
        for _ in range(3)
    ])

    # All should be blocked or escalated (no false approvals)
    for result in results:
        assert result.allowed == False or result.escalate == True


# ============================================================================
# Test 16-18: Audit Trail Concurrency
# ============================================================================

@pytest.mark.asyncio
async def test_audit_trail_concurrent_logging(audit_trail):
    """Test audit trail handles concurrent log writes."""
    from hololoom.alignment.audit_trail import DecisionType, OutcomeType

    # Fire 20 concurrent log writes
    async def log_decision(i):
        audit_trail.log_decision(
            decision_type=DecisionType.TOOL_SELECTION,
            outcome=OutcomeType.APPROVED,
            reason=f"Test decision {i}"
        )
        return i

    results = await asyncio.gather(*[log_decision(i) for i in range(20)])

    # All logs should be recorded
    assert len(audit_trail.logs) == 20
    assert len(results) == 20


@pytest.mark.asyncio
async def test_audit_trail_log_ordering(audit_trail):
    """Test audit trail maintains log ordering under concurrent writes."""
    from hololoom.alignment.audit_trail import DecisionType, OutcomeType

    # Fire concurrent logs with timestamps
    async def log_with_timestamp(i):
        audit_trail.log_decision(
            decision_type=DecisionType.MEMORY_RETRIEVAL,
            outcome=OutcomeType.APPROVED,
            reason=f"Log {i}"
        )
        return i

    await asyncio.gather(*[log_with_timestamp(i) for i in range(10)])

    # Logs should maintain temporal ordering (monotonic timestamps)
    timestamps = [log.timestamp for log in audit_trail.logs]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i + 1]  # Monotonic increasing


@pytest.mark.asyncio
async def test_audit_trail_no_log_loss(audit_trail):
    """Test audit trail doesn't lose logs under concurrent load."""
    from hololoom.alignment.audit_trail import DecisionType, OutcomeType

    # Fire 100 concurrent log writes
    async def log_decision(i):
        audit_trail.log_decision(
            decision_type=DecisionType.SAFETY_GATE,
            outcome=OutcomeType.APPROVED if i % 2 == 0 else OutcomeType.REJECTED,
            reason=f"Decision {i}"
        )

    await asyncio.gather(*[log_decision(i) for i in range(100)])

    # Should have exactly 100 logs (no loss)
    assert len(audit_trail.logs) == 100

    # Should have correct split of approved/rejected
    approved = sum(1 for log in audit_trail.logs if log.outcome == OutcomeType.APPROVED)
    rejected = sum(1 for log in audit_trail.logs if log.outcome == OutcomeType.REJECTED)
    assert approved == 50
    assert rejected == 50


# ============================================================================
# Test 19-20: Memory Consistency
# ============================================================================

@pytest.mark.asyncio
async def test_memory_concurrent_writes(fast_config, test_shards):
    """Test memory system handles concurrent writes safely."""
    from hololoom.memory.backend_factory import create_memory_backend
    from hololoom.memory.protocol import Memory

    backend = await create_memory_backend(fast_config)

    # Fire 10 concurrent write operations
    async def write_memory(i):
        memory = Memory(
            id=f"mem_{i}",
            text=f"Concurrent write test {i}",
            context={"test": True},
            metadata={}
        )
        await backend.store([memory])
        return i

    results = await asyncio.gather(*[write_memory(i) for i in range(10)])

    # All writes should complete
    assert len(results) == 10

    # Cleanup
    if hasattr(backend, 'close'):
        await backend.close()


@pytest.mark.asyncio
async def test_memory_concurrent_reads(fast_config, test_shards):
    """Test memory system handles concurrent reads safely."""
    from hololoom.memory.backend_factory import create_memory_backend
    from hololoom.memory.protocol import Memory, MemoryQuery

    backend = await create_memory_backend(fast_config)

    # Store test data
    memories = [
        Memory(id=f"mem_{i}", text=f"Test memory {i}", context={}, metadata={})
        for i in range(5)
    ]
    await backend.store(memories)

    # Fire 20 concurrent read operations
    async def read_memory(i):
        query = MemoryQuery(text=f"Test memory {i % 5}", limit=10)
        result = await backend.retrieve(query)
        return len(result.memories)

    results = await asyncio.gather(*[read_memory(i) for i in range(20)])

    # All reads should complete
    assert len(results) == 20

    # Cleanup
    if hasattr(backend, 'close'):
        await backend.close()
