"""
Tests for response caching.
"""

import pytest
import time
from ..cache import ResponseCache
from ..models import (
    GameStateSnapshot,
    PlayerState,
    WorldState,
    NPCState,
    PlayerIntent,
    PlayerIntentType,
    ElleGameAction,
    ActionMode,
    DialogueLine,
)


@pytest.fixture
def cache():
    """Create test cache."""
    return ResponseCache(max_size=3, ttl_seconds=2)


@pytest.fixture
def game_state():
    """Create test game state."""
    return GameStateSnapshot(
        scene_id="test_scene",
        npcs=[
            NPCState(
                id="npc1",
                name="TestNPC",
                role="merchant",
                location="village",
            )
        ],
        player=PlayerState(
            name="Player",
            location="village",
        ),
        world=WorldState(
            time_of_day="day",
        ),
    )


@pytest.fixture
def player_intent():
    """Create test player intent."""
    return PlayerIntent(
        type=PlayerIntentType.TALK_TO_NPC,
        target_npc_id="npc1",
    )


@pytest.fixture
def action_response():
    """Create test action response."""
    return ElleGameAction(
        mode=ActionMode.NPC_DIALOGUE,
        priority="medium",
        dialogue=[
            DialogueLine(
                npc_id="npc1",
                text="Hello!",
                tone="friendly",
            )
        ],
    )


def test_cache_miss(cache, game_state, player_intent):
    """Test cache miss returns None."""
    result = cache.get(game_state, player_intent)
    assert result is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_hit(cache, game_state, player_intent, action_response):
    """Test cache hit returns cached response."""
    # Set cache
    cache.set(game_state, player_intent, action_response)

    # Get should return same response
    result = cache.get(game_state, player_intent)
    assert result is not None
    assert result.mode == action_response.mode
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_different_state_miss(cache, game_state, player_intent, action_response):
    """Test that different game state causes cache miss."""
    # Set cache
    cache.set(game_state, player_intent, action_response)

    # Create different game state
    different_state = GameStateSnapshot(
        scene_id="different_scene",  # Different scene
        player=PlayerState(name="Player", location="village"),
        world=WorldState(time_of_day="day"),
    )

    # Should be cache miss
    result = cache.get(different_state, player_intent)
    assert result is None
    assert cache.misses == 1


def test_cache_different_intent_miss(cache, game_state, player_intent, action_response):
    """Test that different intent causes cache miss."""
    # Set cache
    cache.set(game_state, player_intent, action_response)

    # Create different intent
    different_intent = PlayerIntent(
        type=PlayerIntentType.REQUEST_HINT,
    )

    # Should be cache miss
    result = cache.get(game_state, different_intent)
    assert result is None
    assert cache.misses == 1


def test_cache_ttl_expiration(cache, game_state, player_intent, action_response):
    """Test that cache entries expire after TTL."""
    # Set cache
    cache.set(game_state, player_intent, action_response)

    # Verify it's cached
    result = cache.get(game_state, player_intent)
    assert result is not None
    assert cache.hits == 1

    # Wait for TTL expiration (cache has 2 second TTL)
    time.sleep(2.1)

    # Should now be expired
    result = cache.get(game_state, player_intent)
    assert result is None
    assert cache.ttl_expirations == 1


def test_cache_lru_eviction(cache, game_state, player_intent, action_response):
    """Test LRU eviction when cache is full."""
    # Cache has max_size=3

    # Add 3 entries
    for i in range(3):
        state = GameStateSnapshot(
            scene_id=f"scene_{i}",
            player=PlayerState(name="Player", location="village"),
            world=WorldState(time_of_day="day"),
        )
        cache.set(state, player_intent, action_response)

    assert len(cache.cache) == 3
    assert cache.evictions == 0

    # Add 4th entry - should evict oldest
    state_4 = GameStateSnapshot(
        scene_id="scene_4",
        player=PlayerState(name="Player", location="village"),
        world=WorldState(time_of_day="day"),
    )
    cache.set(state_4, player_intent, action_response)

    assert len(cache.cache) == 3
    assert cache.evictions == 1

    # Oldest (scene_0) should be evicted
    state_0 = GameStateSnapshot(
        scene_id="scene_0",
        player=PlayerState(name="Player", location="village"),
        world=WorldState(time_of_day="day"),
    )
    result = cache.get(state_0, player_intent)
    assert result is None  # Evicted


def test_cache_skip_cache_flag(cache, game_state, player_intent, action_response):
    """Test skip_cache flag prevents caching."""
    # Set with skip_cache=True
    cache.set(game_state, player_intent, action_response, skip_cache=True)

    # Should not be cached
    result = cache.get(game_state, player_intent)
    assert result is None


def test_cache_get_stats(cache, game_state, player_intent, action_response):
    """Test cache statistics."""
    # Add some entries
    cache.set(game_state, player_intent, action_response)
    cache.get(game_state, player_intent)  # Hit
    cache.get(game_state, player_intent)  # Hit

    # Different state - miss
    different_state = GameStateSnapshot(
        scene_id="different",
        player=PlayerState(name="Player", location="village"),
        world=WorldState(time_of_day="day"),
    )
    cache.get(different_state, player_intent)  # Miss

    stats = cache.get_stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 2.0 / 3.0
    assert stats["current_size"] == 1
    assert stats["max_size"] == 3
    assert stats["ttl_seconds"] == 2


def test_cache_clear(cache, game_state, player_intent, action_response):
    """Test cache clear."""
    # Add entry
    cache.set(game_state, player_intent, action_response)
    assert len(cache.cache) == 1

    # Clear
    cache.clear()
    assert len(cache.cache) == 0

    # Should be cache miss
    result = cache.get(game_state, player_intent)
    assert result is None


def test_cache_deterministic_keys(cache, game_state, player_intent):
    """Test that cache keys are deterministic."""
    # Compute key twice - should be same
    key1 = cache._compute_key(game_state, player_intent)
    key2 = cache._compute_key(game_state, player_intent)

    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 64  # SHA256 hex


def test_cache_lru_access_order(cache, game_state, player_intent, action_response):
    """Test that accessing entries moves them to end (LRU)."""
    # Add 3 entries
    states = []
    for i in range(3):
        state = GameStateSnapshot(
            scene_id=f"scene_{i}",
            player=PlayerState(name="Player", location="village"),
            world=WorldState(time_of_day="day"),
        )
        states.append(state)
        cache.set(state, player_intent, action_response)

    # Access scene_0 (oldest) - moves to end
    cache.get(states[0], player_intent)

    # Add new entry - should evict scene_1 (now oldest)
    state_new = GameStateSnapshot(
        scene_id="scene_new",
        player=PlayerState(name="Player", location="village"),
        world=WorldState(time_of_day="day"),
    )
    cache.set(state_new, player_intent, action_response)

    # scene_0 should still be cached (was accessed)
    assert cache.get(states[0], player_intent) is not None

    # scene_1 should be evicted
    assert cache.get(states[1], player_intent) is None
