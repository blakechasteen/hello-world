"""
EdWIN Database Tests

Tests for database connections and operations.

Implementation Date: November 15, 2025

Usage:
    pytest EduVerse/edwin/tests/test_database.py -v
"""

import pytest
from EduVerse.edwin.database import DatabaseManager, RedisCache


# ==============================================================================
# Database Manager Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_database_manager_init():
    """Test DatabaseManager initialization"""
    db = DatabaseManager(use_persistent_storage=False, enable_redis=False)

    assert db.use_persistent_storage is False
    assert db.enable_redis is False
    assert db._initialized is False


@pytest.mark.asyncio
async def test_database_manager_initialize():
    """Test DatabaseManager initialization"""
    db = DatabaseManager(use_persistent_storage=False, enable_redis=False)

    await db.initialize()

    assert db._initialized is True
    assert db.knowledge_graph is not None


@pytest.mark.asyncio
async def test_database_manager_close():
    """Test DatabaseManager cleanup"""
    db = DatabaseManager(use_persistent_storage=False, enable_redis=False)

    await db.initialize()
    assert db._initialized is True

    await db.close()
    assert db._initialized is False


@pytest.mark.asyncio
async def test_database_manager_health_check():
    """Test database health check"""
    db = DatabaseManager(use_persistent_storage=False, enable_redis=False)

    await db.initialize()

    health = await db.health_check()

    assert health is not None
    assert hasattr(health, 'status')
    assert hasattr(health, 'neo4j')
    assert hasattr(health, 'qdrant')
    assert hasattr(health, 'redis')

    await db.close()


# ==============================================================================
# Redis Cache Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_redis_cache_init():
    """Test RedisCache initialization"""
    cache = RedisCache()

    assert cache.host is not None
    assert cache.port is not None
    assert cache._connected is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_redis_cache_operations():
    """Test basic Redis operations (requires Redis running)"""
    cache = RedisCache()

    try:
        await cache.connect()

        if not cache._connected:
            pytest.skip("Redis not available")

        # Set value
        await cache.set("test_key", "test_value", expire=60)

        # Get value
        value = await cache.get("test_key")
        assert value == "test_value"

        # Check exists
        exists = await cache.exists("test_key")
        assert exists is True

        # Delete
        await cache.delete("test_key")

        # Verify deleted
        value = await cache.get("test_key")
        assert value is None

    finally:
        await cache.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_redis_cache_json_operations():
    """Test JSON operations (requires Redis running)"""
    cache = RedisCache()

    try:
        await cache.connect()

        if not cache._connected:
            pytest.skip("Redis not available")

        # Set JSON
        test_data = {"name": "Test", "value": 123}
        await cache.set_json("test_json", test_data, expire=60)

        # Get JSON
        retrieved = await cache.get_json("test_json")
        assert retrieved == test_data
        assert retrieved["name"] == "Test"
        assert retrieved["value"] == 123

        # Clean up
        await cache.delete("test_json")

    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_redis_cache_graceful_degradation():
    """Test that Redis failures don't crash the system"""
    # Connect to non-existent Redis
    cache = RedisCache(host="nonexistent.host", port=9999)

    await cache.connect()

    # Should not raise exception, just not connect
    assert cache._connected is False

    # Operations should not crash
    await cache.set("key", "value")  # Should do nothing
    value = await cache.get("key")   # Should return None
    assert value is None

    await cache.close()


# ==============================================================================
# Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_full_database_lifecycle():
    """Test complete database lifecycle"""
    db = DatabaseManager(use_persistent_storage=False, enable_redis=False)

    # Initialize
    await db.initialize()
    assert db._initialized is True

    # Health check
    health = await db.health_check()
    assert health.status in ["healthy", "degraded", "down"]

    # Close
    await db.close()
    assert db._initialized is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
