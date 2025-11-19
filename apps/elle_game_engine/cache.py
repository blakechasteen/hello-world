"""
Response caching for Elle Game Engine.

Implements LRU cache for identical game states to reduce LLM calls and improve latency.
"""

import hashlib
import json
import time
import os
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
from dataclasses import asdict


class ResponseCache:
    """
    LRU cache for game action responses.

    Caches responses based on hash of (game_state + player_intent).
    Uses LRU eviction when cache is full.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
    ):
        """
        Initialize response cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cached entries (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # LRU cache: {cache_key: (response, timestamp)}
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.ttl_expirations = 0

    def get(self, game_state: Any, player_intent: Any) -> Optional[Any]:
        """
        Get cached response if available.

        Args:
            game_state: Game state snapshot
            player_intent: Player intent

        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._compute_key(game_state, player_intent)

        # Check if in cache
        if cache_key not in self.cache:
            self.misses += 1
            return None

        # Get entry
        response, timestamp = self.cache[cache_key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            # Expired - remove and return None
            del self.cache[cache_key]
            self.ttl_expirations += 1
            self.misses += 1
            return None

        # Cache hit - move to end (most recently used)
        self.cache.move_to_end(cache_key)
        self.hits += 1
        return response

    def set(
        self,
        game_state: Any,
        player_intent: Any,
        response: Any,
        skip_cache: bool = False,
    ):
        """
        Cache a response.

        Args:
            game_state: Game state snapshot
            player_intent: Player intent
            response: Response to cache
            skip_cache: If True, don't cache (for debug intents)
        """
        if skip_cache:
            return

        cache_key = self._compute_key(game_state, player_intent)

        # Add to cache
        self.cache[cache_key] = (response, time.time())
        self.cache.move_to_end(cache_key)

        # Evict if over max size (LRU)
        if len(self.cache) > self.max_size:
            # Remove oldest (first item)
            self.cache.popitem(last=False)
            self.evictions += 1

    def _compute_key(self, game_state: Any, player_intent: Any) -> str:
        """
        Compute cache key from game state and intent.

        Uses SHA256 hash of JSON representation for stable keys.

        Args:
            game_state: Game state snapshot
            player_intent: Player intent

        Returns:
            Cache key (hex string)
        """
        # Convert to dictionaries if dataclasses
        if hasattr(game_state, "__dataclass_fields__"):
            state_dict = asdict(game_state)
        else:
            state_dict = game_state

        if hasattr(player_intent, "__dataclass_fields__"):
            intent_dict = asdict(player_intent)
        else:
            intent_dict = player_intent

        # Create deterministic JSON (sorted keys)
        combined = {
            "game_state": state_dict,
            "player_intent": intent_dict,
        }

        json_str = json.dumps(combined, sort_keys=True)

        # Hash for stable key
        return hashlib.sha256(json_str.encode()).hexdigest()

    def clear(self):
        """Clear entire cache."""
        self.cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "ttl_expirations": self.ttl_expirations,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


def create_cache() -> ResponseCache:
    """
    Create response cache from environment variables.

    Environment Variables:
        ELLE_CACHE_SIZE: Maximum cache entries (default: 1000)
        ELLE_CACHE_TTL_SECONDS: Time-to-live in seconds (default: 300)

    Returns:
        Configured response cache
    """
    max_size = int(os.getenv("ELLE_CACHE_SIZE", "1000"))
    ttl_seconds = int(os.getenv("ELLE_CACHE_TTL_SECONDS", "300"))

    return ResponseCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
    )
