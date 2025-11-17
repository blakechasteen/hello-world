#!/usr/bin/env python3
"""
Cache Processor for Promptly Chains

Multi-level caching with memory, Redis, and disk backends.
Includes cache key generation, TTL, invalidation, warming, and metrics.
"""

import time
import json
import hashlib
import pickle
import os
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from promptly.plugins.base import BaseChainStepProcessor


class CacheStrategy(Enum):
    """Cache storage strategies"""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    MULTI_LEVEL = "multi_level"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class CacheMetrics:
    """Track cache performance metrics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.errors = 0
        self.total_latency = 0.0
        self.hit_latencies = []
        self.miss_latencies = []

    def record_hit(self, latency: float) -> None:
        """Record cache hit"""
        self.hits += 1
        self.total_latency += latency
        self.hit_latencies.append(latency)

    def record_miss(self, latency: float) -> None:
        """Record cache miss"""
        self.misses += 1
        self.total_latency += latency
        self.miss_latencies.append(latency)

    def record_set(self) -> None:
        """Record cache set"""
        self.sets += 1

    def record_eviction(self) -> None:
        """Record cache eviction"""
        self.evictions += 1

    def record_error(self) -> None:
        """Record cache error"""
        self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses

        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "errors": self.errors,
            "total_requests": total_requests,
            "hit_rate": self.hits / total_requests if total_requests > 0 else 0.0,
            "miss_rate": self.misses / total_requests if total_requests > 0 else 0.0,
            "avg_latency": self.total_latency / total_requests if total_requests > 0 else 0.0,
            "avg_hit_latency": sum(self.hit_latencies) / len(self.hit_latencies) if self.hit_latencies else 0.0,
            "avg_miss_latency": sum(self.miss_latencies) / len(self.miss_latencies) if self.miss_latencies else 0.0
        }

    def get_cost_savings(self, cost_per_request: float = 0.01) -> Dict[str, Any]:
        """Calculate cost savings from cache hits"""
        saved_requests = self.hits
        cost_saved = saved_requests * cost_per_request

        return {
            "saved_requests": saved_requests,
            "cost_per_request": cost_per_request,
            "total_cost_saved": cost_saved,
            "currency": "USD"
        }


class MemoryCache:
    """In-memory cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.access_counts = defaultdict(int)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check TTL
        if entry["expires_at"] and datetime.now() > entry["expires_at"]:
            del self.cache[key]
            return None

        # Update access
        self.access_counts[key] += 1
        self.cache.move_to_end(key)

        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        # Evict if full
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)

        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None

        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        self.cache.move_to_end(key)

    def delete(self, key: str) -> None:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_counts[key]

    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.access_counts.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class DiskCache:
    """Disk-based cache"""

    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load cache index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_index(self) -> None:
        """Save cache index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path"""
        # Hash key to filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        if key not in self.index:
            return None

        entry = self.index[key]

        # Check TTL
        if entry.get("expires_at"):
            expires_at = datetime.fromisoformat(entry["expires_at"])
            if datetime.now() > expires_at:
                self.delete(key)
                return None

        # Load from disk
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in disk cache"""
        # Check size limit
        self._check_size_limit()

        # Save to disk
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            # Update index
            expires_at = None
            if ttl and ttl > 0:
                expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()

            self.index[key] = {
                "path": str(cache_path),
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at,
                "size": cache_path.stat().st_size
            }

            self._save_index()

        except Exception as e:
            print(f"Error saving to disk cache: {e}")

    def delete(self, key: str) -> None:
        """Delete from disk cache"""
        if key in self.index:
            cache_path = Path(self.index[key]["path"])
            if cache_path.exists():
                cache_path.unlink()
            del self.index[key]
            self._save_index()

    def clear(self) -> None:
        """Clear all disk cache"""
        for key in list(self.index.keys()):
            self.delete(key)

    def size(self) -> int:
        """Get cache size in bytes"""
        return sum(entry.get("size", 0) for entry in self.index.values())

    def _check_size_limit(self) -> None:
        """Check and enforce size limit"""
        current_size_mb = self.size() / (1024 * 1024)

        if current_size_mb > self.max_size_mb:
            # Evict oldest entries
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1].get("created_at", "")
            )

            for key, _ in sorted_entries[:len(sorted_entries) // 4]:
                self.delete(key)


class CacheProcessor(BaseChainStepProcessor):
    """
    Multi-level caching processor.

    Configuration format:
    {
        "type": "cache",
        "operation": "get",  # get, set, delete, clear, warm
        "strategy": "multi_level",  # memory, disk, redis, multi_level
        "key": {
            "fields": ["input", "model"],
            "prefix": "prompt_cache",
            "version": "v1"
        },
        "levels": [
            {
                "type": "memory",
                "max_size": 1000,
                "ttl": 300
            },
            {
                "type": "disk",
                "cache_dir": ".cache",
                "max_size_mb": 100,
                "ttl": 3600
            }
        ],
        "ttl": 300,  # seconds
        "eviction_policy": "lru",  # lru, lfu, fifo, ttl
        "metrics": {
            "enabled": true,
            "track_cost": true,
            "cost_per_request": 0.01
        },
        "warming": {
            "enabled": false,
            "keys": []
        },
        "on_miss": "continue",  # continue, execute, throw
        "on_hit": "return"  # return, continue
    }

    Examples:
    - Cache LLM responses
    - Cache API calls
    - Cache expensive computations
    - Reduce costs
    """

    def __init__(self):
        super().__init__(
            name="cache",
            description="Multi-level caching with metrics and cost tracking"
        )
        self._memory_caches: Dict[str, MemoryCache] = {}
        self._disk_caches: Dict[str, DiskCache] = {}
        self._metrics: Dict[str, CacheMetrics] = defaultdict(CacheMetrics)
        self._key_generators: Dict[str, Callable] = {}

    def register_key_generator(self, name: str, generator: Callable) -> None:
        """Register custom cache key generator"""
        self._key_generators[name] = generator

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute cache operation.

        Args:
            step_input: Input data from previous step
            step_config: Cache configuration

        Returns:
            Step input with cache result
        """
        operation = step_config.get("operation", "get")

        if operation == "get":
            return self._cache_get(step_input, step_config)
        elif operation == "set":
            return self._cache_set(step_input, step_config)
        elif operation == "delete":
            return self._cache_delete(step_input, step_config)
        elif operation == "clear":
            return self._cache_clear(step_input, step_config)
        elif operation == "warm":
            return self._cache_warm(step_input, step_config)
        else:
            return {
                **step_input,
                "cache_error": f"Unknown operation: {operation}"
            }

    def _cache_get(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get value from cache"""
        cache_key = self._generate_cache_key(step_input, step_config)
        strategy = step_config.get("strategy", "memory")
        metrics_config = step_config.get("metrics", {})

        start_time = time.time()

        # Try to get from cache
        cached_value = None

        if strategy == CacheStrategy.MULTI_LEVEL.value:
            cached_value = self._get_multi_level(cache_key, step_config)
        elif strategy == CacheStrategy.MEMORY.value:
            cached_value = self._get_memory(cache_key, step_config)
        elif strategy == CacheStrategy.DISK.value:
            cached_value = self._get_disk(cache_key, step_config)
        elif strategy == CacheStrategy.REDIS.value:
            cached_value = self._get_redis(cache_key, step_config)

        latency = time.time() - start_time

        # Record metrics
        if metrics_config.get("enabled", True):
            cache_name = step_config.get("cache_name", "default")
            if cached_value is not None:
                self._metrics[cache_name].record_hit(latency)
            else:
                self._metrics[cache_name].record_miss(latency)

        # Handle result
        if cached_value is not None:
            on_hit = step_config.get("on_hit", "return")

            result = {
                **step_input,
                "cache_hit": True,
                "cache_key": cache_key,
                "cached_value": cached_value,
                "cache_latency": latency
            }

            if on_hit == "return":
                result["output"] = cached_value

            return result

        else:
            on_miss = step_config.get("on_miss", "continue")

            if on_miss == "throw":
                raise KeyError(f"Cache miss for key: {cache_key}")

            return {
                **step_input,
                "cache_hit": False,
                "cache_key": cache_key,
                "cache_latency": latency
            }

    def _cache_set(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Set value in cache"""
        cache_key = self._generate_cache_key(step_input, step_config)
        value = step_input.get("output")
        strategy = step_config.get("strategy", "memory")
        ttl = step_config.get("ttl", 300)

        # Set in cache
        if strategy == CacheStrategy.MULTI_LEVEL.value:
            self._set_multi_level(cache_key, value, ttl, step_config)
        elif strategy == CacheStrategy.MEMORY.value:
            self._set_memory(cache_key, value, ttl, step_config)
        elif strategy == CacheStrategy.DISK.value:
            self._set_disk(cache_key, value, ttl, step_config)
        elif strategy == CacheStrategy.REDIS.value:
            self._set_redis(cache_key, value, ttl, step_config)

        # Record metrics
        metrics_config = step_config.get("metrics", {})
        if metrics_config.get("enabled", True):
            cache_name = step_config.get("cache_name", "default")
            self._metrics[cache_name].record_set()

        return {
            **step_input,
            "cache_set": True,
            "cache_key": cache_key
        }

    def _cache_delete(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Delete from cache"""
        cache_key = self._generate_cache_key(step_input, step_config)
        strategy = step_config.get("strategy", "memory")

        # Delete from cache
        if strategy == CacheStrategy.MULTI_LEVEL.value:
            levels = step_config.get("levels", [])
            for level in levels:
                if level["type"] == "memory":
                    self._get_memory_cache(step_config).delete(cache_key)
                elif level["type"] == "disk":
                    self._get_disk_cache(step_config).delete(cache_key)
        elif strategy == CacheStrategy.MEMORY.value:
            self._get_memory_cache(step_config).delete(cache_key)
        elif strategy == CacheStrategy.DISK.value:
            self._get_disk_cache(step_config).delete(cache_key)

        return {
            **step_input,
            "cache_deleted": True,
            "cache_key": cache_key
        }

    def _cache_clear(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Clear cache"""
        strategy = step_config.get("strategy", "memory")

        if strategy == CacheStrategy.MULTI_LEVEL.value:
            levels = step_config.get("levels", [])
            for level in levels:
                if level["type"] == "memory":
                    self._get_memory_cache(step_config).clear()
                elif level["type"] == "disk":
                    self._get_disk_cache(step_config).clear()
        elif strategy == CacheStrategy.MEMORY.value:
            self._get_memory_cache(step_config).clear()
        elif strategy == CacheStrategy.DISK.value:
            self._get_disk_cache(step_config).clear()

        return {
            **step_input,
            "cache_cleared": True
        }

    def _cache_warm(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Warm cache with predefined keys"""
        warming_config = step_config.get("warming", {})
        keys = warming_config.get("keys", [])

        warmed_count = 0

        for key_data in keys:
            # Generate key and value
            cache_key = self._generate_cache_key(key_data, step_config)
            value = key_data.get("value")

            if value:
                ttl = step_config.get("ttl", 300)
                self._set_memory(cache_key, value, ttl, step_config)
                warmed_count += 1

        return {
            **step_input,
            "cache_warmed": True,
            "warmed_keys": warmed_count
        }

    def _get_multi_level(self, key: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get from multi-level cache (L1 -> L2 -> L3)"""
        levels = config.get("levels", [])

        for level in levels:
            level_type = level.get("type")

            if level_type == "memory":
                value = self._get_memory_cache(config).get(key)
                if value is not None:
                    # Promote to higher levels
                    self._promote_to_higher_levels(key, value, levels, level)
                    return value

            elif level_type == "disk":
                value = self._get_disk_cache(config).get(key)
                if value is not None:
                    # Promote to higher levels
                    self._promote_to_higher_levels(key, value, levels, level)
                    return value

        return None

    def _set_multi_level(self, key: str, value: Any, ttl: int, config: Dict[str, Any]) -> None:
        """Set in all cache levels"""
        levels = config.get("levels", [])

        for level in levels:
            level_type = level.get("type")
            level_ttl = level.get("ttl", ttl)

            if level_type == "memory":
                self._get_memory_cache(config).set(key, value, level_ttl)
            elif level_type == "disk":
                self._get_disk_cache(config).set(key, value, level_ttl)

    def _promote_to_higher_levels(
        self,
        key: str,
        value: Any,
        levels: List[Dict[str, Any]],
        current_level: Dict[str, Any]
    ) -> None:
        """Promote cache entry to higher (faster) levels"""
        current_index = levels.index(current_level)

        for i in range(current_index):
            level = levels[i]
            level_type = level.get("type")

            if level_type == "memory":
                cache = self._get_memory_cache({"levels": levels})
                cache.set(key, value, level.get("ttl", 300))

    def _get_memory(self, key: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get from memory cache"""
        return self._get_memory_cache(config).get(key)

    def _set_memory(self, key: str, value: Any, ttl: int, config: Dict[str, Any]) -> None:
        """Set in memory cache"""
        self._get_memory_cache(config).set(key, value, ttl)

    def _get_disk(self, key: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get from disk cache"""
        return self._get_disk_cache(config).get(key)

    def _set_disk(self, key: str, value: Any, ttl: int, config: Dict[str, Any]) -> None:
        """Set in disk cache"""
        self._get_disk_cache(config).set(key, value, ttl)

    def _get_redis(self, key: str, config: Dict[str, Any]) -> Optional[Any]:
        """Get from Redis (not implemented)"""
        return None

    def _set_redis(self, key: str, value: Any, ttl: int, config: Dict[str, Any]) -> None:
        """Set in Redis (not implemented)"""
        pass

    def _get_memory_cache(self, config: Dict[str, Any]) -> MemoryCache:
        """Get or create memory cache"""
        cache_name = config.get("cache_name", "default")

        if cache_name not in self._memory_caches:
            max_size = config.get("max_size", 1000)
            default_ttl = config.get("ttl", 300)
            self._memory_caches[cache_name] = MemoryCache(max_size, default_ttl)

        return self._memory_caches[cache_name]

    def _get_disk_cache(self, config: Dict[str, Any]) -> DiskCache:
        """Get or create disk cache"""
        cache_name = config.get("cache_name", "default")

        if cache_name not in self._disk_caches:
            cache_dir = config.get("cache_dir", ".cache")
            max_size_mb = config.get("max_size_mb", 100)
            self._disk_caches[cache_name] = DiskCache(cache_dir, max_size_mb)

        return self._disk_caches[cache_name]

    def _generate_cache_key(self, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate cache key from data"""
        key_config = config.get("key", {})

        # Custom generator
        generator_name = key_config.get("generator")
        if generator_name and generator_name in self._key_generators:
            return self._key_generators[generator_name](data, key_config)

        # Default: hash of field values
        fields = key_config.get("fields", ["input"])
        prefix = key_config.get("prefix", "cache")
        version = key_config.get("version", "v1")

        # Extract field values
        values = []
        for field in fields:
            value = self._get_field(data, field)
            values.append(str(value))

        # Create hash
        key_string = "|".join(values)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"{prefix}:{version}:{key_hash}"

    def _get_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def get_metrics(self, cache_name: str = "default") -> Dict[str, Any]:
        """Get cache metrics"""
        return self._metrics[cache_name].get_stats()

    def get_cost_savings(self, cache_name: str = "default", cost_per_request: float = 0.01) -> Dict[str, Any]:
        """Get cost savings from cache"""
        return self._metrics[cache_name].get_cost_savings(cost_per_request)


__all__ = [
    "CacheProcessor",
    "CacheStrategy",
    "EvictionPolicy",
    "CacheMetrics",
    "MemoryCache",
    "DiskCache"
]
