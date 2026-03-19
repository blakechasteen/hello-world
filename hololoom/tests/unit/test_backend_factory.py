"""
Tests for hololoom.core.memory.backend_factory

Covers: sanitize_uri, sanitize_error_message, HybridMemoryStore,
        create_memory_backend, _try_init_neo4j, _try_init_qdrant,
        _create_fallback_backend, GuardrailedMemoryStore, check_backend_health.
"""

import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from hololoom.core.memory.backend_factory import (
    sanitize_uri,
    sanitize_error_message,
    HybridMemoryStore,
    GuardrailedMemoryStore,
    create_memory_backend,
    check_backend_health,
    _try_init_neo4j,
    _try_init_qdrant,
    _create_fallback_backend,
)
from hololoom.config import Config, MemoryBackend
from hololoom.core.memory.protocol import Memory, MemoryQuery, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(mid: str = "m1", text: str = "hello") -> Memory:
    return Memory(
        id=mid,
        text=text,
        timestamp=datetime.now(),
        context={},
        metadata={},
    )


def _make_result(memories: Optional[List[Memory]] = None) -> RetrievalResult:
    mems = memories or [_make_memory()]
    return RetrievalResult(
        memories=mems,
        scores=[1.0] * len(mems),
        strategy_used="test",
        metadata={},
    )


_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__


def _selective_import_error(blocked_module: str):
    """Return an __import__ replacement that blocks one specific module."""
    import builtins
    original = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Mocked: {name} unavailable")
        return original(name, *args, **kwargs)

    return _import


class _FakeBackend:
    """Minimal mock backend that satisfies the MemoryStore-like interface."""

    def __init__(self, *, healthy: bool = True):
        self._healthy = healthy
        self.stored: List[Memory] = []

    async def store(self, memory: Memory, user_id: str = "default") -> str:
        self.stored.append(memory)
        return memory.id

    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        return _make_result(self.stored[:limit] or [_make_memory("fallback")])

    async def health_check(self) -> Dict[str, Any]:
        if self._healthy:
            return {"status": "healthy"}
        return {"status": "unhealthy", "error": "down"}


# =========================================================================
# sanitize_uri
# =========================================================================

class TestSanitizeUri:
    def test_empty_string(self):
        assert sanitize_uri("") == ""

    def test_none(self):
        assert sanitize_uri(None) is None

    def test_no_password(self):
        assert sanitize_uri("bolt://localhost:7687") == "bolt://localhost:7687"

    def test_masks_password_with_username(self):
        result = sanitize_uri("bolt://neo4j:secret@localhost:7687")
        assert "secret" not in result
        assert "****" in result
        assert "neo4j" in result
        assert "localhost" in result
        assert "7687" in result

    def test_masks_password_without_username(self):
        result = sanitize_uri("redis://:mysecret@localhost:6379")
        assert "mysecret" not in result
        assert "****" in result

    def test_preserves_path_and_query(self):
        result = sanitize_uri("bolt://user:pass@host:7687/db?timeout=30")
        assert "pass" not in result
        assert "/db" in result
        assert "timeout=30" in result


# =========================================================================
# sanitize_error_message
# =========================================================================

class TestSanitizeErrorMessage:
    def test_masks_password_field(self):
        result = sanitize_error_message("Connection failed password=supersecret foo")
        assert "supersecret" not in result
        assert "password=****" in result

    def test_masks_pwd_field(self):
        result = sanitize_error_message("Error pwd=abc123")
        assert "abc123" not in result

    def test_masks_uri_credentials(self):
        result = sanitize_error_message("Cannot connect to bolt://user:secret@host")
        assert "secret" not in result
        assert "****" in result

    def test_no_credentials_unchanged(self):
        msg = "Connection refused on port 7687"
        assert sanitize_error_message(msg) == msg


# =========================================================================
# HybridMemoryStore.__init__
# =========================================================================

class TestHybridMemoryStoreInit:
    def test_fallback_mode_when_no_production_backends(self):
        store = HybridMemoryStore(fallback=_FakeBackend())
        assert store.fallback_mode is True
        assert store.backends == []

    def test_production_mode_with_neo4j(self):
        neo = _FakeBackend()
        store = HybridMemoryStore(neo4j=neo)
        assert store.fallback_mode is False
        assert len(store.backends) == 1
        assert store.backends[0][0] == "neo4j"

    def test_production_mode_with_both(self):
        store = HybridMemoryStore(neo4j=_FakeBackend(), qdrant=_FakeBackend())
        assert store.fallback_mode is False
        assert len(store.backends) == 2


# =========================================================================
# HybridMemoryStore.store
# =========================================================================

class TestHybridMemoryStoreStore:
    @pytest.mark.asyncio
    async def test_store_returns_memory_id(self):
        neo = _FakeBackend()
        store = HybridMemoryStore(neo4j=neo)
        mem = _make_memory("abc")
        result = await store.store(mem)
        assert result == "abc"

    @pytest.mark.asyncio
    async def test_store_validates_none_memory(self):
        store = HybridMemoryStore(neo4j=_FakeBackend())
        with pytest.raises(ValueError, match="Cannot store"):
            await store.store(None)

    @pytest.mark.asyncio
    async def test_store_validates_none_id(self):
        mem = _make_memory()
        mem.id = None
        store = HybridMemoryStore(neo4j=_FakeBackend())
        with pytest.raises(ValueError, match="Cannot store"):
            await store.store(mem)

    @pytest.mark.asyncio
    async def test_store_partial_backend_failure(self):
        """One backend fails, the other succeeds — store still returns id."""
        good = _FakeBackend()
        bad = AsyncMock()
        bad.store = AsyncMock(side_effect=RuntimeError("down"))
        store = HybridMemoryStore(neo4j=good, qdrant=bad)
        mem = _make_memory("x")
        result = await store.store(mem)
        assert result == "x"
        assert len(good.stored) == 1


# =========================================================================
# HybridMemoryStore.store_many
# =========================================================================

class TestHybridMemoryStoreStoreMany:
    @pytest.mark.asyncio
    async def test_store_many_empty(self):
        store = HybridMemoryStore(neo4j=_FakeBackend())
        assert await store.store_many([]) == []

    @pytest.mark.asyncio
    async def test_store_many_returns_ids(self):
        store = HybridMemoryStore(neo4j=_FakeBackend())
        ids = await store.store_many([_make_memory("a"), _make_memory("b")])
        assert ids == ["a", "b"]


# =========================================================================
# HybridMemoryStore.recall
# =========================================================================

class TestHybridMemoryStoreRecall:
    @pytest.mark.asyncio
    async def test_recall_fallback_mode(self):
        fb = _FakeBackend()
        store = HybridMemoryStore(fallback=fb)
        result = await store.recall(MemoryQuery(text="test"))
        assert result.strategy_used == "fallback"
        assert result.metadata["backend"] == "networkx"

    @pytest.mark.asyncio
    async def test_recall_production_mode(self):
        neo = _FakeBackend()
        await neo.store(_make_memory("prod"))
        store = HybridMemoryStore(neo4j=neo)
        result = await store.recall(MemoryQuery(text="test"))
        assert result.strategy_used == "hybrid_balanced"
        assert "neo4j" in result.metadata["backends"]

    @pytest.mark.asyncio
    async def test_recall_emergency_fallback(self):
        """All production backends fail => emergency fallback to NetworkX."""
        bad = AsyncMock()
        bad.recall = AsyncMock(side_effect=RuntimeError("timeout"))
        fb = _FakeBackend()
        store = HybridMemoryStore(neo4j=bad, fallback=fb)
        result = await store.recall(MemoryQuery(text="test"))
        assert result.strategy_used == "emergency_fallback"

    @pytest.mark.asyncio
    async def test_recall_all_fail_no_fallback(self):
        """All backends fail and no fallback => empty fused results."""
        bad = AsyncMock()
        bad.recall = AsyncMock(side_effect=RuntimeError("down"))
        store = HybridMemoryStore(neo4j=bad)
        result = await store.recall(MemoryQuery(text="test"))
        assert result.strategy_used == "hybrid_balanced"
        assert result.memories == []


# =========================================================================
# HybridMemoryStore._fuse
# =========================================================================

class TestHybridFuse:
    def test_fuse_empty(self):
        store = HybridMemoryStore()
        assert store._fuse({}, 5) == []

    def test_fuse_single_backend(self):
        store = HybridMemoryStore()
        m1 = _make_memory("a")
        result = _make_result([m1])
        fused = store._fuse({"neo4j": result}, 10)
        assert len(fused) == 1
        assert fused[0].id == "a"

    def test_fuse_respects_limit(self):
        store = HybridMemoryStore()
        mems = [_make_memory(f"m{i}") for i in range(5)]
        result = RetrievalResult(
            memories=mems,
            scores=[1.0, 0.9, 0.8, 0.7, 0.6],
            strategy_used="test",
            metadata={},
        )
        fused = store._fuse({"neo4j": result}, 2)
        assert len(fused) == 2

    def test_fuse_boosted_by_overlap(self):
        """Memory appearing in both backends gets a higher fused score."""
        store = HybridMemoryStore()
        shared = _make_memory("shared")
        only_neo = _make_memory("only_neo")

        r1 = RetrievalResult(
            memories=[shared, only_neo],
            scores=[0.5, 0.5],
            strategy_used="t",
            metadata={},
        )
        r2 = RetrievalResult(
            memories=[shared],
            scores=[0.5],
            strategy_used="t",
            metadata={},
        )
        fused = store._fuse({"neo4j": r1, "qdrant": r2}, 10)
        # shared should rank first due to cross-backend boost
        assert fused[0].id == "shared"


# =========================================================================
# HybridMemoryStore.health_check
# =========================================================================

class TestHybridHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy_production(self):
        store = HybridMemoryStore(neo4j=_FakeBackend(healthy=True))
        health = await store.health_check()
        assert health["status"] == "healthy"
        assert health["mode"] == "production"

    @pytest.mark.asyncio
    async def test_degraded_in_fallback_mode(self):
        store = HybridMemoryStore(fallback=_FakeBackend())
        health = await store.health_check()
        assert health["status"] == "degraded"
        assert health["mode"] == "fallback"

    @pytest.mark.asyncio
    async def test_degraded_when_backend_unhealthy(self):
        store = HybridMemoryStore(neo4j=_FakeBackend(healthy=False))
        health = await store.health_check()
        assert health["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_backend_raises(self):
        bad = AsyncMock()
        bad.health_check = AsyncMock(side_effect=RuntimeError("boom"))
        store = HybridMemoryStore(neo4j=bad)
        health = await store.health_check()
        assert health["status"] == "degraded"
        assert "unhealthy" in health["backends"]["neo4j"]["status"]


# =========================================================================
# create_memory_backend — INMEMORY
# =========================================================================

class TestCreateMemoryBackendInmemory:
    @pytest.mark.asyncio
    async def test_inmemory_returns_guardrailed_store(self):
        config = Config(memory_backend=MemoryBackend.INMEMORY)
        backend = await create_memory_backend(config)
        assert isinstance(backend, GuardrailedMemoryStore)

    @pytest.mark.asyncio
    async def test_none_config_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            await create_memory_backend(None)

    @pytest.mark.asyncio
    async def test_missing_memory_backend_attr_raises(self):
        obj = MagicMock(spec=[])  # no attributes at all
        with pytest.raises(ValueError, match="missing memory_backend"):
            await create_memory_backend(obj)


# =========================================================================
# create_memory_backend — HYBRID (mocked backends)
# =========================================================================

class TestCreateMemoryBackendHybrid:
    @pytest.mark.asyncio
    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", False)
    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", False)
    @patch("hololoom.core.memory.backend_factory.NETWORKX_AVAILABLE", True)
    async def test_hybrid_falls_back_to_networkx(self):
        """When neither Neo4j nor Qdrant available, falls back to NetworkX."""
        config = Config(memory_backend=MemoryBackend.HYBRID)
        backend = await create_memory_backend(config)
        assert isinstance(backend, HybridMemoryStore)
        assert backend.fallback_mode is True

    @pytest.mark.asyncio
    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", False)
    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", False)
    @patch("hololoom.core.memory.backend_factory.NETWORKX_AVAILABLE", False)
    async def test_hybrid_raises_when_nothing_available(self):
        config = Config(memory_backend=MemoryBackend.HYBRID)
        with pytest.raises(ValueError, match="No backends available"):
            await create_memory_backend(config)


# =========================================================================
# create_memory_backend — HYPERSPACE
# =========================================================================

class TestCreateMemoryBackendHyperspace:
    @pytest.mark.asyncio
    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", False)
    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", False)
    @patch("hololoom.core.memory.backend_factory.NETWORKX_AVAILABLE", True)
    async def test_hyperspace_falls_back_to_hybrid_when_import_fails(self):
        """When hyperspace import fails, degrades through HYBRID to NetworkX."""
        with patch.dict("sys.modules", {"hololoom.memory.hyperspace_backend": None}):
            with patch(
                "hololoom.core.memory.backend_factory.create_memory_backend",
            ) as _:
                # Force the ImportError path by patching the dynamic import
                pass

        # Simpler: just patch the import to fail inside create_memory_backend
        config = Config(memory_backend=MemoryBackend.HYPERSPACE)

        # Patch away the dynamic import so it raises ImportError
        with patch(
            "builtins.__import__",
            side_effect=_selective_import_error("hololoom.memory.hyperspace_backend"),
        ):
            backend = await create_memory_backend(config)
            # Falls back to HYBRID, then to NetworkX fallback
            assert isinstance(backend, HybridMemoryStore)
            assert backend.fallback_mode is True


# =========================================================================
# _try_init_neo4j
# =========================================================================

class TestTryInitNeo4j:
    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", False)
    def test_returns_none_when_library_missing(self):
        config = Config()
        result, error = _try_init_neo4j(config)
        assert result is None
        assert "not installed" in error

    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", True)
    def test_returns_none_when_uri_empty(self):
        config = Config()
        config.neo4j_uri = ""
        result, error = _try_init_neo4j(config)
        assert result is None
        assert "not configured" in error


# =========================================================================
# _try_init_qdrant
# =========================================================================

class TestTryInitQdrant:
    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", False)
    def test_returns_none_when_library_missing(self):
        config = Config()
        result, error = _try_init_qdrant(config)
        assert result is None
        assert "not installed" in error

    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", True)
    def test_returns_none_when_host_empty(self):
        config = Config()
        config.qdrant_host = ""
        result, error = _try_init_qdrant(config)
        assert result is None
        assert "not configured" in error

    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", True)
    def test_returns_none_when_port_zero(self):
        config = Config()
        config.qdrant_port = 0
        result, error = _try_init_qdrant(config)
        assert result is None
        assert "not configured" in error


# =========================================================================
# _create_fallback_backend
# =========================================================================

class TestCreateFallbackBackend:
    @patch("hololoom.core.memory.backend_factory.NETWORKX_AVAILABLE", True)
    def test_creates_backend_when_available(self):
        backend = _create_fallback_backend()
        assert backend is not None

    @patch("hololoom.core.memory.backend_factory.NETWORKX_AVAILABLE", False)
    def test_raises_when_unavailable(self):
        with pytest.raises(ValueError, match="No backends available"):
            _create_fallback_backend()


# =========================================================================
# GuardrailedMemoryStore
# =========================================================================

class TestGuardrailedMemoryStore:
    @pytest.mark.asyncio
    async def test_delegates_store(self):
        inner = _FakeBackend()
        gms = GuardrailedMemoryStore(backend=inner)
        result = await gms.store(_make_memory("g1"))
        assert result == "g1"
        assert len(inner.stored) == 1

    @pytest.mark.asyncio
    async def test_delegates_recall(self):
        inner = _FakeBackend()
        gms = GuardrailedMemoryStore(backend=inner)
        result = await gms.recall(MemoryQuery(text="test"))
        assert len(result.memories) > 0

    @pytest.mark.asyncio
    async def test_delegates_health_check(self):
        inner = _FakeBackend()
        gms = GuardrailedMemoryStore(backend=inner)
        health = await gms.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_no_method(self):
        """Backend without health_check returns unknown status."""
        inner = MagicMock(spec=[])
        gms = GuardrailedMemoryStore(backend=inner)
        health = await gms.health_check()
        assert health["status"] == "unknown"

    def test_no_double_guarding(self):
        """If inner backend already has guardrails, outer disables its own."""
        inner = _FakeBackend()
        inner.guardrails_enabled = True
        gms = GuardrailedMemoryStore(backend=inner)
        assert gms.guardrails_enabled is False

    @pytest.mark.asyncio
    async def test_store_many(self):
        inner = _FakeBackend()
        gms = GuardrailedMemoryStore(backend=inner)
        ids = await gms.store_many([_make_memory("a"), _make_memory("b")])
        assert ids == ["a", "b"]

    @pytest.mark.asyncio
    async def test_delete_raises_if_unsupported(self):
        """Backend without delete() raises AttributeError (guardrails disabled)."""
        inner = MagicMock(spec=[])
        gms = GuardrailedMemoryStore(backend=inner)
        # Disable guardrails so we reach the AttributeError check
        gms.guardrails_enabled = False
        with pytest.raises(AttributeError, match="does not support delete"):
            await gms.delete("some_id")

    @pytest.mark.asyncio
    async def test_delete_blocked_by_guardrails(self):
        """Deletion is blocked by safety guardrails (high-risk action)."""
        inner = _FakeBackend()
        gms = GuardrailedMemoryStore(backend=inner)
        with pytest.raises(PermissionError):
            await gms.delete("some_id")

    @pytest.mark.asyncio
    async def test_retrieve_raises_if_unsupported(self):
        inner = MagicMock(spec=[])
        gms = GuardrailedMemoryStore(backend=inner)
        with pytest.raises(AttributeError, match="does not support retrieve"):
            await gms.retrieve(MemoryQuery(text="test"))

    def test_getattr_delegates(self):
        """Attributes not on GuardrailedMemoryStore pass through to inner backend."""
        inner = _FakeBackend()
        inner.custom_attr = 42
        gms = GuardrailedMemoryStore(backend=inner)
        assert gms.custom_attr == 42


# =========================================================================
# check_backend_health
# =========================================================================

class TestCheckBackendHealth:
    @pytest.mark.asyncio
    async def test_none_backend(self):
        result = await check_backend_health(None)
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_healthy_backend(self):
        result = await check_backend_health(_FakeBackend(healthy=True))
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_backend_without_health_check(self):
        backend = MagicMock(spec=[])
        result = await check_backend_health(backend)
        assert result["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_health_check_raises(self):
        bad = AsyncMock()
        bad.health_check = AsyncMock(side_effect=RuntimeError("boom"))
        result = await check_backend_health(bad)
        assert result["status"] == "unhealthy"
        assert "boom" in result["message"]
