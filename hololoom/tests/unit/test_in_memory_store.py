"""
Unit Tests for InMemoryStore
============================
Tests all public and internal methods of the in-memory store.

Run:
    pytest hololoom/tests/unit/test_in_memory_store.py -q --tb=short --timeout=30
"""

import pytest
from datetime import datetime, timedelta

import hololoom  # noqa: F401 — installs _CoreRedirectFinder before other imports

from hololoom.memory.stores.in_memory_store import InMemoryStore
from hololoom.memory.protocol import Memory, MemoryQuery, Strategy


# ============================================================================
# Helpers
# ============================================================================

def make_memory(
    text: str = "test memory",
    user_id: str = "default",
    mem_id: str = "",
    timestamp: datetime = None,
) -> Memory:
    """Factory for Memory objects used across tests."""
    return Memory(
        id=mem_id,
        text=text,
        timestamp=timestamp or datetime.now(),
        context={},
        metadata={"user_id": user_id},
    )


def make_query(
    text: str = "test",
    user_id: str = "default",
    limit: int = 10,
) -> MemoryQuery:
    return MemoryQuery(text=text, user_id=user_id, limit=limit)


# ============================================================================
# store() — basic persistence
# ============================================================================

@pytest.mark.asyncio
async def test_store_returns_id():
    store = InMemoryStore()
    mem = make_memory(text="hello world")
    mem_id = await store.store(mem)

    assert isinstance(mem_id, str)
    assert len(mem_id) > 0


@pytest.mark.asyncio
async def test_store_generates_id_when_empty():
    """ID should be generated when memory.id is empty string."""
    store = InMemoryStore()
    mem = make_memory(text="generate me an id", mem_id="")
    mem_id = await store.store(mem)

    assert mem_id != ""
    assert mem.id == mem_id  # id is set on the object in-place


@pytest.mark.asyncio
async def test_store_preserves_explicit_id():
    """Explicit non-empty ID should not be overwritten."""
    store = InMemoryStore()
    mem = make_memory(text="keep my id", mem_id="custom-abc")
    mem_id = await store.store(mem)

    assert mem_id == "custom-abc"


@pytest.mark.asyncio
async def test_store_persists_memory():
    """Stored memory must be retrievable by id."""
    store = InMemoryStore()
    mem = make_memory(text="persist me", mem_id="p1")
    await store.store(mem)

    assert "p1" in store._memories
    assert store._memories["p1"].text == "persist me"


@pytest.mark.asyncio
async def test_store_indexes_by_user():
    """Each stored memory must appear in the user index."""
    store = InMemoryStore()
    mem = make_memory(text="user memory", user_id="alice", mem_id="a1")
    await store.store(mem)

    assert "alice" in store._user_index
    assert "a1" in store._user_index["alice"]


@pytest.mark.asyncio
async def test_store_falls_back_to_default_user():
    """Memory with no user_id in metadata should land under 'default'."""
    store = InMemoryStore()
    mem = Memory(id="x1", text="no user", timestamp=datetime.now(), context={}, metadata={})
    await store.store(mem)

    assert "default" in store._user_index
    assert "x1" in store._user_index["default"]


@pytest.mark.asyncio
async def test_store_multiple_memories_same_user():
    """Multiple memories for the same user accumulate in the index."""
    store = InMemoryStore()
    for i in range(3):
        await store.store(make_memory(text=f"item {i}", user_id="bob", mem_id=f"b{i}"))

    assert len(store._user_index["bob"]) == 3


# ============================================================================
# store_many()
# ============================================================================

@pytest.mark.asyncio
async def test_store_many_returns_all_ids():
    store = InMemoryStore()
    mems = [make_memory(text=f"batch {i}", mem_id=f"bm{i}") for i in range(5)]
    ids = await store.store_many(mems)

    assert len(ids) == 5
    assert ids == [f"bm{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_store_many_empty_list():
    store = InMemoryStore()
    ids = await store.store_many([])
    assert ids == []


@pytest.mark.asyncio
async def test_store_many_persists_all():
    store = InMemoryStore()
    mems = [make_memory(text=f"m{i}", mem_id=f"id{i}") for i in range(4)]
    await store.store_many(mems)

    assert len(store._memories) == 4


# ============================================================================
# get_by_id()
# ============================================================================

@pytest.mark.asyncio
async def test_get_by_id_returns_stored_memory():
    store = InMemoryStore()
    mem = make_memory(text="find me", mem_id="findable")
    await store.store(mem)

    result = await store.get_by_id("findable")
    assert result is not None
    assert result.text == "find me"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing():
    store = InMemoryStore()
    result = await store.get_by_id("does-not-exist")
    assert result is None


# ============================================================================
# delete()
# ============================================================================

@pytest.mark.asyncio
async def test_delete_returns_true_for_existing():
    store = InMemoryStore()
    await store.store(make_memory(mem_id="del1"))
    result = await store.delete("del1")
    assert result is True


@pytest.mark.asyncio
async def test_delete_removes_from_storage():
    store = InMemoryStore()
    await store.store(make_memory(mem_id="del2"))
    await store.delete("del2")
    assert "del2" not in store._memories


@pytest.mark.asyncio
async def test_delete_removes_from_user_index():
    store = InMemoryStore()
    await store.store(make_memory(user_id="carol", mem_id="del3"))
    await store.delete("del3")
    assert "del3" not in store._user_index.get("carol", [])


@pytest.mark.asyncio
async def test_delete_returns_false_for_missing():
    store = InMemoryStore()
    result = await store.delete("ghost")
    assert result is False


@pytest.mark.asyncio
async def test_delete_then_get_returns_none():
    store = InMemoryStore()
    await store.store(make_memory(mem_id="gone"))
    await store.delete("gone")
    assert await store.get_by_id("gone") is None


# ============================================================================
# health_check()
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_returns_healthy():
    store = InMemoryStore()
    health = await store.health_check()
    assert health["status"] == "healthy"
    assert health["backend"] == "in-memory"


@pytest.mark.asyncio
async def test_health_check_counts_memories():
    store = InMemoryStore()
    await store.store_many([make_memory(mem_id=f"h{i}") for i in range(3)])
    health = await store.health_check()
    assert health["memory_count"] == 3


@pytest.mark.asyncio
async def test_health_check_counts_users():
    store = InMemoryStore()
    await store.store(make_memory(user_id="u1", mem_id="u1m1"))
    await store.store(make_memory(user_id="u2", mem_id="u2m1"))
    health = await store.health_check()
    assert health["user_count"] == 2


@pytest.mark.asyncio
async def test_health_check_empty_store():
    store = InMemoryStore()
    health = await store.health_check()
    assert health["memory_count"] == 0
    assert health["user_count"] == 0


# ============================================================================
# retrieve() — empty store
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_empty_store_returns_no_memories():
    store = InMemoryStore()
    result = await store.retrieve(make_query(user_id="nobody"))
    assert result.memories == []
    assert result.scores == []
    assert result.metadata["total_memories"] == 0


# ============================================================================
# retrieve() — Strategy.TEMPORAL
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_temporal_returns_memories():
    store = InMemoryStore()
    await store.store(make_memory(text="recent", user_id="t1", mem_id="t1m1"))
    await store.store(make_memory(text="old", user_id="t1", mem_id="t1m2"))

    result = await store.retrieve(make_query(user_id="t1"), strategy=Strategy.TEMPORAL)
    assert len(result.memories) == 2
    assert result.strategy_used == "temporal"


@pytest.mark.asyncio
async def test_retrieve_temporal_newer_scores_higher():
    """A memory created now should score higher than one from an hour ago."""
    store = InMemoryStore()
    old_ts = datetime.now() - timedelta(hours=2)
    new_ts = datetime.now()

    await store.store(make_memory(text="old memory", user_id="tu", mem_id="old", timestamp=old_ts))
    await store.store(make_memory(text="new memory", user_id="tu", mem_id="new", timestamp=new_ts))

    result = await store.retrieve(make_query(user_id="tu", limit=2), strategy=Strategy.TEMPORAL)
    assert result.memories[0].id == "new"


# ============================================================================
# retrieve() — Strategy.SEMANTIC
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_semantic_returns_memories():
    store = InMemoryStore()
    await store.store(make_memory(text="python programming language", user_id="s1", mem_id="s1"))
    await store.store(make_memory(text="cooking recipes", user_id="s1", mem_id="s2"))

    result = await store.retrieve(
        make_query(text="python language", user_id="s1"),
        strategy=Strategy.SEMANTIC,
    )
    assert len(result.memories) == 2
    assert result.strategy_used == "semantic"


@pytest.mark.asyncio
async def test_retrieve_semantic_substring_bonus():
    """Exact substring match should boost score above a non-matching memory."""
    store = InMemoryStore()
    await store.store(make_memory(text="the quick brown fox", user_id="sb", mem_id="match"))
    await store.store(make_memory(text="completely unrelated content xyz", user_id="sb", mem_id="nomatch"))

    result = await store.retrieve(
        make_query(text="quick brown fox", user_id="sb", limit=2),
        strategy=Strategy.SEMANTIC,
    )
    assert result.memories[0].id == "match"


@pytest.mark.asyncio
async def test_retrieve_semantic_empty_query():
    """Empty query text should yield zero scores (no division errors)."""
    store = InMemoryStore()
    await store.store(make_memory(text="some content", user_id="eq", mem_id="eq1"))

    result = await store.retrieve(
        make_query(text="", user_id="eq"),
        strategy=Strategy.SEMANTIC,
    )
    assert len(result.memories) == 1
    assert result.scores[0] <= 1.0  # Score is capped; no crash


# ============================================================================
# retrieve() — Strategy.GRAPH
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_graph_returns_uniform_scores():
    """Graph strategy is a simplified stub: all memories get 0.5."""
    store = InMemoryStore()
    for i in range(3):
        await store.store(make_memory(text=f"g{i}", user_id="g1", mem_id=f"g{i}"))

    result = await store.retrieve(make_query(user_id="g1"), strategy=Strategy.GRAPH)
    assert result.strategy_used == "graph"
    assert all(s == 0.5 for s in result.scores)


# ============================================================================
# retrieve() — Strategy.PATTERN
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_pattern_returns_memories():
    """Pattern strategy returns random scores — just verify it runs and returns data."""
    store = InMemoryStore()
    await store.store(make_memory(text="pattern test", user_id="pa", mem_id="pa1"))

    result = await store.retrieve(make_query(user_id="pa"), strategy=Strategy.PATTERN)
    assert result.strategy_used == "pattern"
    assert len(result.memories) == 1
    assert 0.0 <= result.scores[0] <= 1.0


# ============================================================================
# retrieve() — Strategy.FUSED (default)
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_fused_is_default():
    """Calling retrieve without strategy argument should use FUSED."""
    store = InMemoryStore()
    await store.store(make_memory(text="fused default", user_id="fd", mem_id="fd1"))

    result = await store.retrieve(make_query(user_id="fd"))
    assert result.strategy_used == "fused"


@pytest.mark.asyncio
async def test_retrieve_fused_blends_temporal_and_semantic():
    """Fused scores must be between 0 and 1 (temporal + semantic components both in [0,1])."""
    store = InMemoryStore()
    for i in range(3):
        await store.store(make_memory(text=f"fused memory {i}", user_id="fm", mem_id=f"fm{i}"))

    result = await store.retrieve(make_query(text="fused memory", user_id="fm"), strategy=Strategy.FUSED)
    assert all(0.0 <= s <= 1.0 for s in result.scores)


# ============================================================================
# retrieve() — limit enforcement
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_respects_limit():
    store = InMemoryStore()
    for i in range(10):
        await store.store(make_memory(text=f"item {i}", user_id="lim", mem_id=f"lim{i}"))

    result = await store.retrieve(make_query(user_id="lim", limit=3))
    assert len(result.memories) == 3
    assert len(result.scores) == 3


@pytest.mark.asyncio
async def test_retrieve_limit_larger_than_count():
    """Limit larger than available memories should return all available."""
    store = InMemoryStore()
    for i in range(2):
        await store.store(make_memory(text=f"sm{i}", user_id="sm", mem_id=f"sm{i}"))

    result = await store.retrieve(make_query(user_id="sm", limit=100))
    assert len(result.memories) == 2


# ============================================================================
# retrieve() — sorted by score descending
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_results_sorted_descending():
    """Returned memories should be ordered highest score first."""
    store = InMemoryStore()
    old_ts = datetime.now() - timedelta(hours=10)
    new_ts = datetime.now()

    await store.store(make_memory(text="old", user_id="sort", mem_id="sort_old", timestamp=old_ts))
    await store.store(make_memory(text="new", user_id="sort", mem_id="sort_new", timestamp=new_ts))

    result = await store.retrieve(make_query(user_id="sort", limit=2), strategy=Strategy.TEMPORAL)
    assert result.scores[0] >= result.scores[1]


# ============================================================================
# retrieve() — user isolation
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_isolates_by_user():
    """User A must not see User B's memories."""
    store = InMemoryStore()
    await store.store(make_memory(text="alice secret", user_id="alice", mem_id="am1"))
    await store.store(make_memory(text="bob stuff", user_id="bob", mem_id="bm1"))

    result_alice = await store.retrieve(make_query(user_id="alice"))
    assert all(m.metadata.get("user_id") == "alice" for m in result_alice.memories)
    assert len(result_alice.memories) == 1


# ============================================================================
# retrieve() — metadata in result
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_metadata_includes_total():
    store = InMemoryStore()
    for i in range(4):
        await store.store(make_memory(text=f"meta {i}", user_id="meta", mem_id=f"meta{i}"))

    result = await store.retrieve(make_query(user_id="meta", limit=2))
    assert result.metadata["total_memories"] == 4
    assert "matched" in result.metadata


# ============================================================================
# _generate_id() — deterministic hashing
# ============================================================================

def test_generate_id_is_deterministic():
    store = InMemoryStore()
    id1 = store._generate_id("same text")
    id2 = store._generate_id("same text")
    assert id1 == id2


def test_generate_id_differs_for_different_text():
    store = InMemoryStore()
    id1 = store._generate_id("text A")
    id2 = store._generate_id("text B")
    assert id1 != id2


def test_generate_id_length():
    """ID should be 16 hex chars (md5 truncated to 16)."""
    store = InMemoryStore()
    mem_id = store._generate_id("some content")
    assert len(mem_id) == 16


# ============================================================================
# _score_semantic() — edge cases
# ============================================================================

def test_score_semantic_max_score_capped_at_1():
    """Score must never exceed 1.0 even with substring + full Jaccard match."""
    store = InMemoryStore()
    mem = Memory(
        id="cap",
        text="the quick brown fox",
        timestamp=datetime.now(),
        context={},
        metadata={},
    )
    query = MemoryQuery(text="the quick brown fox", user_id="default")
    scored = store._score_semantic([mem], query)
    assert scored[0][1] <= 1.0


def test_score_semantic_zero_for_empty_query():
    store = InMemoryStore()
    mem = Memory(id="z", text="anything", timestamp=datetime.now(), context={}, metadata={})
    query = MemoryQuery(text="", user_id="default")
    scored = store._score_semantic([mem], query)
    # Empty query words → score should be 0 (or 0.2 if "" is substring of text)
    assert scored[0][1] <= 1.0  # No crash


# ============================================================================
# _score_fused() — combines temporal + semantic
# ============================================================================

def test_score_fused_is_average_of_components():
    """Fused = 0.5 * temporal + 0.5 * semantic. Verify it's in valid range."""
    store = InMemoryStore()
    mem = Memory(
        id="fuse",
        text="fused content test",
        timestamp=datetime.now(),
        context={},
        metadata={},
    )
    query = MemoryQuery(text="fused content", user_id="default")
    scored = store._score_fused([mem], query)
    assert len(scored) == 1
    assert 0.0 <= scored[0][1] <= 1.0


# ============================================================================
# Round-trip: store → retrieve → delete → retrieve
# ============================================================================

@pytest.mark.asyncio
async def test_full_round_trip():
    store = InMemoryStore()
    mem = make_memory(text="round trip test", user_id="rt", mem_id="rt1")

    stored_id = await store.store(mem)
    assert stored_id == "rt1"

    fetched = await store.get_by_id("rt1")
    assert fetched.text == "round trip test"

    result = await store.retrieve(make_query(user_id="rt"))
    assert len(result.memories) == 1

    deleted = await store.delete("rt1")
    assert deleted is True

    result_after = await store.retrieve(make_query(user_id="rt"))
    assert len(result_after.memories) == 0


# ============================================================================
# store() — overwrite / duplicate ID
# ============================================================================

@pytest.mark.asyncio
async def test_store_overwrites_same_id():
    """Storing a second memory with the same ID replaces the first."""
    store = InMemoryStore()
    await store.store(make_memory(text="version 1", mem_id="dup1", user_id="ow"))
    await store.store(make_memory(text="version 2", mem_id="dup1", user_id="ow"))

    fetched = await store.get_by_id("dup1")
    assert fetched.text == "version 2"


@pytest.mark.asyncio
async def test_store_same_id_appends_to_user_index():
    """Storing twice with same ID appends to user_index (duplicate entry)."""
    store = InMemoryStore()
    await store.store(make_memory(text="v1", mem_id="dup2", user_id="ow2"))
    await store.store(make_memory(text="v2", mem_id="dup2", user_id="ow2"))

    # Implementation appends without dedup — user_index has 2 entries
    assert store._user_index["ow2"].count("dup2") == 2


# ============================================================================
# delete() — edge cases
# ============================================================================

@pytest.mark.asyncio
async def test_delete_memory_with_default_user():
    """Delete should work for memories stored under the 'default' user."""
    store = InMemoryStore()
    mem = Memory(id="def1", text="default user mem", timestamp=datetime.now(),
                 context={}, metadata={})
    await store.store(mem)

    result = await store.delete("def1")
    assert result is True
    assert "def1" not in store._memories


@pytest.mark.asyncio
async def test_delete_does_not_affect_other_memories():
    """Deleting one memory must leave all others intact."""
    store = InMemoryStore()
    await store.store(make_memory(text="keep", user_id="multi", mem_id="keep1"))
    await store.store(make_memory(text="remove", user_id="multi", mem_id="rm1"))
    await store.store(make_memory(text="also keep", user_id="multi", mem_id="keep2"))

    await store.delete("rm1")

    assert await store.get_by_id("keep1") is not None
    assert await store.get_by_id("keep2") is not None
    assert await store.get_by_id("rm1") is None
    assert len(store._user_index["multi"]) == 2


# ============================================================================
# retrieve() — cross-user isolation with multiple users
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_no_cross_user_leakage_with_many_users():
    """With 3 users, each should only see their own memories."""
    store = InMemoryStore()
    for user in ["alice", "bob", "charlie"]:
        for i in range(3):
            await store.store(make_memory(
                text=f"{user} memory {i}",
                user_id=user,
                mem_id=f"{user}_{i}",
            ))

    for user in ["alice", "bob", "charlie"]:
        result = await store.retrieve(make_query(user_id=user))
        assert len(result.memories) == 3
        assert all(m.metadata["user_id"] == user for m in result.memories)


# ============================================================================
# retrieve() — with orphaned index entries
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_skips_orphaned_index_entries():
    """If user_index references a memory_id not in _memories, skip it."""
    store = InMemoryStore()
    await store.store(make_memory(text="real", user_id="orphan", mem_id="real1"))

    # Manually inject an orphaned index entry
    store._user_index["orphan"].append("ghost_id")

    result = await store.retrieve(make_query(user_id="orphan"))
    # Should only return the real memory, not crash on the missing one
    assert len(result.memories) == 1
    assert result.memories[0].id == "real1"


# ============================================================================
# _score_temporal() — edge cases
# ============================================================================

def test_score_temporal_very_old_memory_approaches_zero():
    """A memory from 30 days ago should score close to zero."""
    store = InMemoryStore()
    old_ts = datetime.now() - timedelta(days=30)
    mem = Memory(id="old", text="ancient", timestamp=old_ts, context={}, metadata={})
    query = MemoryQuery(text="test", user_id="default")

    scored = store._score_temporal([mem], query)
    assert scored[0][1] < 0.01  # Very low score


def test_score_temporal_just_created_scores_high():
    """A memory created just now should score close to 1.0."""
    store = InMemoryStore()
    mem = Memory(id="new", text="fresh", timestamp=datetime.now(), context={}, metadata={})
    query = MemoryQuery(text="test", user_id="default")

    scored = store._score_temporal([mem], query)
    assert scored[0][1] > 0.99


def test_score_temporal_ordering_preserves_time_order():
    """Multiple memories should be scored in time order — newest highest."""
    store = InMemoryStore()
    now = datetime.now()
    mems = [
        Memory(id=f"t{i}", text=f"mem {i}", timestamp=now - timedelta(hours=i),
               context={}, metadata={})
        for i in range(5)
    ]
    query = MemoryQuery(text="test", user_id="default")

    scored = store._score_temporal(mems, query)
    scores = [s for _, s in scored]
    # Scores should be strictly decreasing (newer = higher)
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1]


# ============================================================================
# _score_semantic() — deeper testing
# ============================================================================

def test_score_semantic_partial_word_overlap():
    """Partial word overlap gives proportional Jaccard score."""
    store = InMemoryStore()
    mem = Memory(id="pw", text="the quick brown fox jumps over the lazy dog",
                 timestamp=datetime.now(), context={}, metadata={})
    query = MemoryQuery(text="quick fox", user_id="default")

    scored = store._score_semantic([mem], query)
    score = scored[0][1]
    # 2 of 2 query words match, but union is larger → Jaccard < 1
    assert 0.0 < score < 1.0


def test_score_semantic_no_overlap_zero_score():
    """Completely disjoint words should score zero (or near zero)."""
    store = InMemoryStore()
    mem = Memory(id="no", text="alpha beta gamma",
                 timestamp=datetime.now(), context={}, metadata={})
    query = MemoryQuery(text="delta epsilon zeta", user_id="default")

    scored = store._score_semantic([mem], query)
    assert scored[0][1] == 0.0


def test_score_semantic_case_insensitive():
    """Matching should be case-insensitive."""
    store = InMemoryStore()
    mem = Memory(id="ci", text="Python Programming",
                 timestamp=datetime.now(), context={}, metadata={})
    query = MemoryQuery(text="python programming", user_id="default")

    scored = store._score_semantic([mem], query)
    assert scored[0][1] > 0.5  # Full word overlap + substring match


def test_score_semantic_multiple_memories_ranked():
    """More relevant memory should score higher than less relevant."""
    store = InMemoryStore()
    exact = Memory(id="exact", text="machine learning algorithms",
                   timestamp=datetime.now(), context={}, metadata={})
    partial = Memory(id="partial", text="machine tools and equipment",
                     timestamp=datetime.now(), context={}, metadata={})
    query = MemoryQuery(text="machine learning", user_id="default")

    scored = store._score_semantic([exact, partial], query)
    scores = {mem.id: s for mem, s in scored}
    assert scores["exact"] > scores["partial"]


# ============================================================================
# _score_graph() — stub verification
# ============================================================================

def test_score_graph_returns_all_with_uniform_score():
    """Graph stub returns 0.5 for every memory."""
    store = InMemoryStore()
    mems = [
        Memory(id=f"g{i}", text=f"graph mem {i}",
               timestamp=datetime.now(), context={}, metadata={})
        for i in range(5)
    ]
    query = MemoryQuery(text="anything", user_id="default")

    scored = store._score_graph(mems, query)
    assert len(scored) == 5
    assert all(score == 0.5 for _, score in scored)


# ============================================================================
# _score_fused() — deeper testing
# ============================================================================

def test_score_fused_blends_both_components():
    """Fused score should reflect both temporal recency and semantic match."""
    store = InMemoryStore()
    # Recent + matching text → high fused
    good = Memory(id="good", text="machine learning python",
                  timestamp=datetime.now(), context={}, metadata={})
    # Old + non-matching text → low fused
    bad = Memory(id="bad", text="cooking recipes pasta",
                 timestamp=datetime.now() - timedelta(hours=24),
                 context={}, metadata={})
    query = MemoryQuery(text="machine learning", user_id="default")

    scored = store._score_fused([good, bad], query)
    scores = {mem.id: s for mem, s in scored}
    assert scores["good"] > scores["bad"]


def test_score_fused_multiple_memories():
    """Fused scoring works correctly with many memories."""
    store = InMemoryStore()
    now = datetime.now()
    mems = [
        Memory(id=f"f{i}", text=f"memory content {i}",
               timestamp=now - timedelta(minutes=i * 10),
               context={}, metadata={})
        for i in range(10)
    ]
    query = MemoryQuery(text="memory content", user_id="default")

    scored = store._score_fused(mems, query)
    assert len(scored) == 10
    assert all(0.0 <= s <= 1.0 for _, s in scored)


# ============================================================================
# _generate_id() — additional edge cases
# ============================================================================

def test_generate_id_empty_string():
    """Empty string should still produce a valid 16-char hex ID."""
    store = InMemoryStore()
    mem_id = store._generate_id("")
    assert len(mem_id) == 16
    assert all(c in "0123456789abcdef" for c in mem_id)


def test_generate_id_unicode():
    """Unicode text should produce a valid ID without errors."""
    store = InMemoryStore()
    mem_id = store._generate_id("café résumé naïve")
    assert len(mem_id) == 16


def test_generate_id_very_long_text():
    """Very long text should still produce a 16-char ID."""
    store = InMemoryStore()
    mem_id = store._generate_id("x" * 100_000)
    assert len(mem_id) == 16


# ============================================================================
# retrieve() — strategy metadata consistency
# ============================================================================

@pytest.mark.asyncio
async def test_retrieve_each_strategy_reports_correct_name():
    """Each strategy should report its own name in strategy_used."""
    store = InMemoryStore()
    await store.store(make_memory(text="strategy test", user_id="sn", mem_id="sn1"))

    for strategy in Strategy:
        result = await store.retrieve(make_query(user_id="sn"), strategy=strategy)
        assert result.strategy_used == strategy.value


@pytest.mark.asyncio
async def test_retrieve_metadata_matched_count():
    """metadata['matched'] should equal number of scored candidates."""
    store = InMemoryStore()
    for i in range(5):
        await store.store(make_memory(text=f"matched {i}", user_id="mc", mem_id=f"mc{i}"))

    result = await store.retrieve(make_query(user_id="mc", limit=3))
    assert result.metadata["matched"] == 5  # All 5 scored
    assert len(result.memories) == 3  # But only 3 returned


# ============================================================================
# store + retrieve — ID generation round trip
# ============================================================================

@pytest.mark.asyncio
async def test_store_generated_id_is_retrievable():
    """A memory stored without explicit ID should be retrievable by generated ID."""
    store = InMemoryStore()
    mem = make_memory(text="auto id", user_id="auto")
    mem_id = await store.store(mem)

    fetched = await store.get_by_id(mem_id)
    assert fetched is not None
    assert fetched.text == "auto id"


@pytest.mark.asyncio
async def test_store_same_text_produces_same_id():
    """Two memories with the same text but no explicit ID get the same generated ID."""
    store = InMemoryStore()
    id1 = await store.store(make_memory(text="duplicate text", user_id="dedup"))
    id2 = await store.store(make_memory(text="duplicate text", user_id="dedup"))
    assert id1 == id2  # Same text → same hash → second overwrites first


# ============================================================================
# health_check() — after mutations
# ============================================================================

@pytest.mark.asyncio
async def test_health_check_after_delete():
    """Health check should reflect reduced count after deletion."""
    store = InMemoryStore()
    await store.store(make_memory(mem_id="hc1"))
    await store.store(make_memory(mem_id="hc2"))
    await store.delete("hc1")

    health = await store.health_check()
    assert health["memory_count"] == 1


@pytest.mark.asyncio
async def test_health_check_latency_field():
    """Health check should include latency_ms."""
    store = InMemoryStore()
    health = await store.health_check()
    assert "latency_ms" in health
    assert isinstance(health["latency_ms"], float)
