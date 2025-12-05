"""
Tests for LLM connection pooling.

Tests connection pool functionality:
- Pool initialization and cleanup
- Checkout/checkin lifecycle
- Concurrent access
- Health checks
- Statistics tracking
- Failover behavior
"""

import pytest
import asyncio
import time

from ..pool import (
    LLMConnectionPool,
    PoolStats,
    PooledClient,
    create_pool,
)


# ============================================================================
# Pool Initialization Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pool_initialization():
    """Test pool initializes with correct size."""
    pool = LLMConnectionPool(provider="dummy", pool_size=5)
    await pool.initialize()

    try:
        stats = pool.get_stats()
        assert stats.pool_size == 5
        assert stats.available_connections == 5
        assert stats.active_connections == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_creates_clients():
    """Test that pool creates all clients on initialization."""
    pool = LLMConnectionPool(provider="dummy", pool_size=3)
    await pool.initialize()

    try:
        # All clients should be available
        assert len(pool.available) == 3

        # Each should have both blocking and streaming clients
        for pooled in pool.available:
            assert pooled.client is not None
            assert pooled.streaming_client is not None
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_factory():
    """Test create_pool factory function."""
    pool = await create_pool("dummy", pool_size=3)

    try:
        stats = pool.get_stats()
        assert stats.pool_size == 3
        assert stats.available_connections == 3
    finally:
        await pool.close()


# ============================================================================
# Checkout/Checkin Tests
# ============================================================================

@pytest.mark.asyncio
async def test_checkout_client():
    """Test basic client checkout."""
    pool = await create_pool("dummy", pool_size=5)

    try:
        async with pool.checkout() as client:
            # Should have a client
            assert client is not None

            # Stats should reflect active connection
            stats = pool.get_stats()
            assert stats.active_connections == 1
            assert stats.available_connections == 4

        # After checkout, should be returned
        stats = pool.get_stats()
        assert stats.active_connections == 0
        assert stats.available_connections == 5
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_checkout_streaming_client():
    """Test checkout with streaming=True."""
    pool = await create_pool("dummy", pool_size=5)

    try:
        async with pool.checkout(streaming=True) as client:
            assert client is not None

            # Should be streaming client
            from ..streaming import StreamingDummyClient
            assert isinstance(client, StreamingDummyClient)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_multiple_checkouts():
    """Test multiple concurrent checkouts."""
    pool = await create_pool("dummy", pool_size=5)

    try:
        # Checkout 3 clients concurrently
        clients = []

        for i in range(3):
            ctx = pool.checkout()
            client = await ctx.__aenter__()
            clients.append((ctx, client))

        # Should have 3 active, 2 available
        stats = pool.get_stats()
        assert stats.active_connections == 3
        assert stats.available_connections == 2

        # Return all clients
        for ctx, client in clients:
            await ctx.__aexit__(None, None, None)

        # Should be back to full availability
        stats = pool.get_stats()
        assert stats.active_connections == 0
        assert stats.available_connections == 5
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_checkout_beyond_pool_size():
    """Test that checkout waits when pool exhausted."""
    pool = await create_pool("dummy", pool_size=2)

    try:
        # Checkout all clients
        ctx1 = pool.checkout()
        client1 = await ctx1.__aenter__()

        ctx2 = pool.checkout()
        client2 = await ctx2.__aenter__()

        # Pool should be exhausted
        stats = pool.get_stats()
        assert stats.available_connections == 0

        # Try to checkout third (should wait)
        checkout_started = time.time()
        checkout_completed = False

        async def try_checkout():
            nonlocal checkout_completed
            async with pool.checkout() as client:
                checkout_completed = True

        # Start checkout task (will wait)
        task = asyncio.create_task(try_checkout())

        # Wait briefly
        await asyncio.sleep(0.1)

        # Should still be waiting
        assert not checkout_completed

        # Return one client
        await ctx1.__aexit__(None, None, None)

        # Now checkout should complete
        await asyncio.wait_for(task, timeout=1.0)
        assert checkout_completed
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_checkout_timeout():
    """Test that checkout times out if pool exhausted too long."""
    pool = LLMConnectionPool(
        provider="dummy",
        pool_size=1,
        max_wait_seconds=0.5,  # Short timeout for testing
    )
    await pool.initialize()

    try:
        # Checkout the only client
        async with pool.checkout() as client1:
            # Try to checkout another (should timeout)
            with pytest.raises(TimeoutError, match="Timeout waiting for available client"):
                async with pool.checkout() as client2:
                    pass
    finally:
        await pool.close()


# ============================================================================
# Statistics Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pool_stats_tracking():
    """Test that pool tracks statistics correctly."""
    pool = await create_pool("dummy", pool_size=5)

    try:
        # Initial stats
        stats = pool.get_stats()
        assert stats.total_checkouts == 0
        assert stats.total_checkins == 0

        # Checkout and checkin
        async with pool.checkout() as client:
            pass

        stats = pool.get_stats()
        assert stats.total_checkouts == 1
        assert stats.total_checkins == 1

        # Multiple checkouts
        for _ in range(5):
            async with pool.checkout() as client:
                pass

        stats = pool.get_stats()
        assert stats.total_checkouts == 6
        assert stats.total_checkins == 6
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_utilization():
    """Test pool utilization calculation."""
    pool = await create_pool("dummy", pool_size=10)

    try:
        # 0% utilization
        stats = pool.get_stats()
        assert stats.pool_utilization == 0.0

        # Checkout 5 clients (50% utilization)
        clients = []
        for _ in range(5):
            ctx = pool.checkout()
            client = await ctx.__aenter__()
            clients.append((ctx, client))

        stats = pool.get_stats()
        assert stats.pool_utilization == 0.5

        # Return all
        for ctx, client in clients:
            await ctx.__aexit__(None, None, None)

        stats = pool.get_stats()
        assert stats.pool_utilization == 0.0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_wait_time_tracking():
    """Test that wait times are tracked."""
    pool = await create_pool("dummy", pool_size=5)

    try:
        # Checkout clients (should be instant, minimal wait)
        for _ in range(10):
            async with pool.checkout() as client:
                pass

        stats = pool.get_stats()
        # Wait time should be very low (< 10ms for instant checkout)
        assert stats.average_wait_time_ms < 10
    finally:
        await pool.close()


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.asyncio
async def test_client_health_tracking():
    """Test that client health is tracked."""
    pool = await create_pool("dummy", pool_size=3)

    try:
        # All clients should start healthy
        for pooled in pool.available:
            assert pooled.healthy is True
            assert pooled.failure_count == 0

        # Simulate failure
        async with pool.checkout() as client:
            pass

        # Get the pooled client and mark failure
        pooled = pool.available[0]
        pooled.mark_failure()
        assert pooled.failure_count == 1
        assert pooled.healthy is True  # Still healthy (threshold is 3)

        # Multiple failures should mark unhealthy
        pooled.mark_failure()
        pooled.mark_failure()
        assert pooled.failure_count == 3
        assert pooled.healthy is False
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_client_success_resets_failures():
    """Test that success resets failure count."""
    pool = await create_pool("dummy", pool_size=3)

    try:
        pooled = pool.available[0]

        # Mark failures
        pooled.mark_failure()
        pooled.mark_failure()
        assert pooled.failure_count == 2

        # Mark success
        pooled.mark_success()
        assert pooled.failure_count == 0
        assert pooled.healthy is True
    finally:
        await pool.close()


# ============================================================================
# Concurrent Access Tests
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_access():
    """Test pool handles concurrent access correctly."""
    pool = await create_pool("dummy", pool_size=10)

    try:
        async def use_client(client_id: int):
            """Simulate using a client."""
            async with pool.checkout() as client:
                # Use client briefly
                await asyncio.sleep(0.01)
            return client_id

        # Run 50 concurrent tasks with pool of 10
        tasks = [use_client(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 50

        # Pool should be back to full availability
        stats = pool.get_stats()
        assert stats.active_connections == 0
        assert stats.available_connections == 10
        assert stats.total_checkouts == 50
        assert stats.total_checkins == 50
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_pool_performance_under_load():
    """Test pool performance with heavy concurrent load."""
    pool = await create_pool("dummy", pool_size=5)

    try:
        start_time = time.time()

        async def request():
            async with pool.checkout() as client:
                # Simulate work
                await asyncio.sleep(0.001)

        # 100 concurrent requests with pool of 5
        tasks = [request() for _ in range(100)]
        await asyncio.gather(*tasks)

        duration = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        # With pool of 5, 100 requests = 20 batches
        # At ~1ms each, should be ~20ms + overhead
        assert duration < 5.0

        stats = pool.get_stats()
        assert stats.total_checkouts == 100
        assert stats.total_checkins == 100
    finally:
        await pool.close()


# ============================================================================
# Cleanup Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pool_close():
    """Test pool closes cleanly."""
    pool = await create_pool("dummy", pool_size=5)

    # Close pool
    await pool.close()

    # Should be empty
    assert len(pool.available) == 0
    assert len(pool.active) == 0


@pytest.mark.asyncio
async def test_pool_close_with_active_clients():
    """Test pool closes even with active clients."""
    pool = await create_pool("dummy", pool_size=5)

    # Checkout a client
    ctx = pool.checkout()
    client = await ctx.__aenter__()

    # Close pool (should still work)
    await pool.close()

    # Return client (should not crash)
    await ctx.__aexit__(None, None, None)
