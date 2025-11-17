"""
HoloLoom Cache Manager

Redis-based caching layer for distributed deployments.
Provides key-value caching, TTL management, and cache invalidation.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
import json
import pickle
from dataclasses import dataclass
from datetime import timedelta
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration"""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50
    decode_responses: bool = False


class CacheManager:
    """
    Redis cache manager for distributed caching.

    Provides:
    - Key-value storage with TTL
    - JSON and pickle serialization
    - Batch operations
    - Cache statistics
    - Automatic reconnection
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        if config is None:
            # Load from environment
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.client = redis.from_url(
                redis_url,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                decode_responses=False,
            )
        else:
            # Create connection pool
            pool = redis.ConnectionPool(
                host=config.host,
                port=config.port,
                db=config.db,
                password=config.password,
                socket_timeout=config.socket_timeout,
                socket_connect_timeout=config.socket_connect_timeout,
                max_connections=config.max_connections,
                decode_responses=config.decode_responses,
            )
            self.client = redis.Redis(connection_pool=pool)

        # Test connection
        try:
            self.client.ping()
            logger.info("Cache manager connected to Redis")
        except RedisError as exc:
            logger.error(f"Failed to connect to Redis: {exc}")
            raise

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: str = 'json',
    ) -> bool:
        """
        Set a cache value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            serialize: Serialization method ('json' or 'pickle')

        Returns:
            True if successful
        """
        try:
            # Serialize value
            if serialize == 'json':
                serialized = json.dumps(value)
            elif serialize == 'pickle':
                serialized = pickle.dumps(value)
            else:
                raise ValueError(f"Unknown serialization method: {serialize}")

            # Set with optional TTL
            if ttl:
                result = self.client.setex(key, ttl, serialized)
            else:
                result = self.client.set(key, serialized)

            return bool(result)

        except Exception as exc:
            logger.error(f"Error setting cache key {key}: {exc}")
            return False

    def get(
        self,
        key: str,
        default: Any = None,
        serialize: str = 'json',
    ) -> Any:
        """
        Get a cache value.

        Args:
            key: Cache key
            default: Default value if key not found
            serialize: Serialization method used

        Returns:
            Cached value or default
        """
        try:
            value = self.client.get(key)

            if value is None:
                return default

            # Deserialize value
            if serialize == 'json':
                return json.loads(value)
            elif serialize == 'pickle':
                return pickle.loads(value)
            else:
                raise ValueError(f"Unknown serialization method: {serialize}")

        except Exception as exc:
            logger.error(f"Error getting cache key {key}: {exc}")
            return default

    def delete(self, *keys: str) -> int:
        """
        Delete one or more cache keys.

        Args:
            *keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        try:
            return self.client.delete(*keys)
        except Exception as exc:
            logger.error(f"Error deleting cache keys: {exc}")
            return 0

    def exists(self, *keys: str) -> int:
        """
        Check if keys exist.

        Args:
            *keys: Keys to check

        Returns:
            Number of keys that exist
        """
        try:
            return self.client.exists(*keys)
        except Exception as exc:
            logger.error(f"Error checking cache keys: {exc}")
            return 0

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for a key.

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        try:
            return bool(self.client.expire(key, ttl))
        except Exception as exc:
            logger.error(f"Error setting TTL for key {key}: {exc}")
            return False

    def ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds (-1 if no expiry, -2 if key doesn't exist)
        """
        try:
            return self.client.ttl(key)
        except Exception as exc:
            logger.error(f"Error getting TTL for key {key}: {exc}")
            return -2

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value.

        Args:
            key: Cache key
            amount: Increment amount

        Returns:
            New value after increment
        """
        try:
            return self.client.incrby(key, amount)
        except Exception as exc:
            logger.error(f"Error incrementing key {key}: {exc}")
            return 0

    def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a numeric value.

        Args:
            key: Cache key
            amount: Decrement amount

        Returns:
            New value after decrement
        """
        try:
            return self.client.decrby(key, amount)
        except Exception as exc:
            logger.error(f"Error decrementing key {key}: {exc}")
            return 0

    def mget(
        self,
        keys: List[str],
        serialize: str = 'json',
    ) -> List[Any]:
        """
        Get multiple values at once.

        Args:
            keys: List of cache keys
            serialize: Serialization method

        Returns:
            List of values (None for missing keys)
        """
        try:
            values = self.client.mget(keys)

            result = []
            for value in values:
                if value is None:
                    result.append(None)
                elif serialize == 'json':
                    result.append(json.loads(value))
                elif serialize == 'pickle':
                    result.append(pickle.loads(value))
                else:
                    result.append(value)

            return result

        except Exception as exc:
            logger.error(f"Error getting multiple cache keys: {exc}")
            return [None] * len(keys)

    def mset(
        self,
        mapping: Dict[str, Any],
        serialize: str = 'json',
    ) -> bool:
        """
        Set multiple values at once.

        Args:
            mapping: Dictionary of key-value pairs
            serialize: Serialization method

        Returns:
            True if successful
        """
        try:
            # Serialize all values
            if serialize == 'json':
                serialized = {k: json.dumps(v) for k, v in mapping.items()}
            elif serialize == 'pickle':
                serialized = {k: pickle.dumps(v) for k, v in mapping.items()}
            else:
                serialized = mapping

            return bool(self.client.mset(serialized))

        except Exception as exc:
            logger.error(f"Error setting multiple cache keys: {exc}")
            return False

    def flush(self, async_mode: bool = False) -> bool:
        """
        Flush all keys in the current database.

        Args:
            async_mode: Use async flush (non-blocking)

        Returns:
            True if successful
        """
        try:
            if async_mode:
                result = self.client.flushdb(asynchronous=True)
            else:
                result = self.client.flushdb()

            logger.warning("Cache flushed")
            return bool(result)

        except Exception as exc:
            logger.error(f"Error flushing cache: {exc}")
            return False

    def keys(self, pattern: str = '*') -> List[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            List of matching keys
        """
        try:
            return [k.decode() if isinstance(k, bytes) else k
                   for k in self.client.keys(pattern)]
        except Exception as exc:
            logger.error(f"Error getting keys with pattern {pattern}: {exc}")
            return []

    def scan(
        self,
        match: Optional[str] = None,
        count: int = 100,
    ):
        """
        Scan keys iteratively (use instead of keys() for large datasets).

        Args:
            match: Key pattern to match
            count: Approximate number of keys per iteration

        Yields:
            Matching keys
        """
        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(
                    cursor=cursor,
                    match=match,
                    count=count,
                )

                for key in keys:
                    yield key.decode() if isinstance(key, bytes) else key

                if cursor == 0:
                    break

        except Exception as exc:
            logger.error(f"Error scanning keys: {exc}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        try:
            info = self.client.info('stats')

            return {
                'total_connections_received': info.get('total_connections_received', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'total_net_input_bytes': info.get('total_net_input_bytes', 0),
                'total_net_output_bytes': info.get('total_net_output_bytes', 0),
                'rejected_connections': info.get('rejected_connections', 0),
                'expired_keys': info.get('expired_keys', 0),
                'evicted_keys': info.get('evicted_keys', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
            }

        except Exception as exc:
            logger.error(f"Error getting cache stats: {exc}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """
        Check cache health.

        Returns:
            Dict with health status
        """
        try:
            # Ping Redis
            ping_result = self.client.ping()

            # Get memory info
            memory_info = self.client.info('memory')

            return {
                'status': 'healthy' if ping_result else 'unhealthy',
                'ping': ping_result,
                'used_memory': memory_info.get('used_memory', 0),
                'used_memory_human': memory_info.get('used_memory_human', 'unknown'),
                'used_memory_peak': memory_info.get('used_memory_peak', 0),
                'connected_clients': self.client.info('clients').get('connected_clients', 0),
            }

        except Exception as exc:
            logger.error(f"Cache health check failed: {exc}")
            return {
                'status': 'unhealthy',
                'error': str(exc),
            }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.client.close()
