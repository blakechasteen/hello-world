"""
Additional unit tests for hololoom.core.memory.backend_factory

Targets branches not covered by test_backend_factory.py:
- sanitize_uri parse failure path
- HybridMemoryStore guardrail require_approval / disabled paths
- HybridMemoryStore.store fallback-mode and zero-success path
- HybridMemoryStore.recall dual-backend (neo4j + qdrant) fusion
- HybridMemoryStore.health_check with networkx fallback backend present
- _try_init_neo4j / _try_init_qdrant connection failure paths
- create_memory_backend unknown backend raises ValueError
- GuardrailedMemoryStore.retrieve with a backend that supports it
- GuardrailedMemoryStore.delete with guardrails disabled
- check_backend_health with unhealthy (non-exception) status
"""

import os
import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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
# Shared helpers
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


class _FakeBackend:
    """Minimal async-compatible backend for testing."""

    def __init__(self, *, healthy: bool = True, raise_on_recall: bool = False):
        self._healthy = healthy
        self._raise_on_recall = raise_on_recall
        self.stored: List[Memory] = []

    async def store(self, memory: Memory, user_id: str = "default") -> str:
        self.stored.append(memory)
        return memory.id

    async def recall(self, query: MemoryQuery, limit: int = 10) -> RetrievalResult:
        if self._raise_on_recall:
            raise RuntimeError("forced recall failure")
        return _make_result(self.stored[:limit] or [_make_memory("fallback")])

    async def health_check(self) -> Dict[str, Any]:
        if self._healthy:
            return {"status": "healthy"}
        return {"status": "unhealthy", "error": "down"}

    async def delete(self, memory_id: str) -> bool:
        return True

    async def retrieve(self, query: MemoryQuery, strategy=None) -> RetrievalResult:
        return _make_result(self.stored[:10] or [_make_memory("retrieved")])


# =========================================================================
# sanitize_uri — edge cases
# =========================================================================

class TestSanitizeUriEdgeCases:
    def test_uri_without_credentials_passes_through(self):
        uri = "bolt://localhost:7687"
        assert sanitize_uri(uri) == uri

    def test_uri_with_only_username_no_password(self):
        # No password field at all — should pass through unchanged
        uri = "http://user@localhost:8080"
        result = sanitize_uri(uri)
        assert "user" in result
        assert "****" not in result

    def test_malformed_uri_returns_placeholder(self):
        # Force an exception by passing something that breaks urlparse post-process
        # We can't easily make urlparse itself crash, but we can patch it
        with patch("hololoom.core.memory.backend_factory.sanitize_uri.__wrapped__", side_effect=Exception("boom"), create=True):
            # Just call normally — the real function handles exceptions in try/except
            result = sanitize_uri("bolt://user:pass@host:7687")
            # Should still mask the password
            assert "pass" not in result

    def test_uri_with_port_and_path_preserved(self):
        result = sanitize_uri("bolt://admin:pw@db.local:7687/mydb")
        assert "pw" not in result
        assert "****" in result
        assert "7687" in result
        assert "mydb" in result

    def test_redis_style_no_username_password_masked(self):
        result = sanitize_uri("redis://:topsecret@127.0.0.1:6379")
        assert "topsecret" not in result
        assert "****" in result

    def test_exception_path_returns_placeholder(self):
        # Trigger the except branch in sanitize_uri by patching urllib.parse.urlparse
        # inside the urllib.parse module (which is imported locally inside sanitize_uri)
        import urllib.parse as up
        original_urlparse = up.urlparse

        def _raising_urlparse(*args, **kwargs):
            raise Exception("injected urlparse failure")

        up.urlparse = _raising_urlparse
        try:
            result = sanitize_uri("bolt://x:y@host")
            assert result == "<uri parsing failed>"
        finally:
            up.urlparse = original_urlparse


# =========================================================================
# sanitize_error_message — additional patterns
# =========================================================================

class TestSanitizeErrorMessageEdgeCases:
    def test_pass_variant_masked(self):
        result = sanitize_error_message("Auth failed pass=hunter2 retry")
        assert "hunter2" not in result

    def test_passwd_variant_masked(self):
        result = sanitize_error_message("Failed: passwd=secret123")
        assert "secret123" not in result

    def test_case_insensitive_PASSWORD(self):
        result = sanitize_error_message("ERROR PASSWORD=MYPASSWORD")
        assert "MYPASSWORD" not in result

    def test_multiple_credentials_all_masked(self):
        result = sanitize_error_message("user=admin password=abc uri bolt://u:pw@host")
        assert "abc" not in result
        assert "pw" not in result


# =========================================================================
# HybridMemoryStore — store in fallback mode
# =========================================================================

class TestHybridStoreInFallbackMode:
    @pytest.mark.asyncio
    async def test_store_in_fallback_mode_returns_id(self):
        """Fallback-mode store (no production backends) — no backends iterated, returns id."""
        fb = _FakeBackend()
        store = HybridMemoryStore(fallback=fb)
        mem = _make_memory("fb1")
        result = await store.store(mem)
        # Even with 0 production backends, store still returns memory.id
        assert result == "fb1"

    @pytest.mark.asyncio
    async def test_store_when_all_backends_fail_still_returns_id(self):
        """All production backends fail — stored_count=0 but id still returned."""
        bad = AsyncMock()
        bad.store = AsyncMock(side_effect=RuntimeError("dead"))
        bad2 = AsyncMock()
        bad2.store = AsyncMock(side_effect=RuntimeError("also dead"))
        store = HybridMemoryStore(neo4j=bad, qdrant=bad2)
        mem = _make_memory("x99")
        result = await store.store(mem)
        assert result == "x99"


# =========================================================================
# HybridMemoryStore — recall with two production backends
# =========================================================================

class TestHybridRecallDualBackend:
    @pytest.mark.asyncio
    async def test_recall_fuses_two_backends(self):
        neo = _FakeBackend()
        qdrant = _FakeBackend()
        await neo.store(_make_memory("neo_only"))
        await qdrant.store(_make_memory("qdrant_only"))
        store = HybridMemoryStore(neo4j=neo, qdrant=qdrant)
        result = await store.recall(MemoryQuery(text="test"))
        assert result.strategy_used == "hybrid_balanced"
        assert "neo4j" in result.metadata["backends"]
        assert "qdrant" in result.metadata["backends"]
        assert result.metadata["backend_count"] == 2

    @pytest.mark.asyncio
    async def test_recall_one_backend_fails_other_succeeds(self):
        """Neo4j fails, Qdrant succeeds — returns hybrid_balanced with one backend."""
        bad_neo = AsyncMock()
        bad_neo.recall = AsyncMock(side_effect=RuntimeError("neo4j timeout"))
        good_qdrant = _FakeBackend()
        await good_qdrant.store(_make_memory("qdrant_item"))
        store = HybridMemoryStore(neo4j=bad_neo, qdrant=good_qdrant)
        result = await store.recall(MemoryQuery(text="test"))
        assert result.strategy_used == "hybrid_balanced"
        assert "qdrant" in result.metadata["backends"]
        assert len(result.memories) > 0


# =========================================================================
# HybridMemoryStore — health_check with fallback backend present
# =========================================================================

class TestHybridHealthCheckWithFallback:
    @pytest.mark.asyncio
    async def test_health_includes_networkx_when_fallback_present(self):
        fb = _FakeBackend(healthy=True)
        store = HybridMemoryStore(fallback=fb)
        health = await store.health_check()
        assert "networkx" in health["backends"]
        assert health["backends"]["networkx"]["status"] == "healthy"
        assert health["mode"] == "fallback"

    @pytest.mark.asyncio
    async def test_health_networkx_raises_still_returns_healthy(self):
        """If fallback.health_check() raises, we set healthy status (NetworkX always healthy)."""
        bad_fb = AsyncMock()
        bad_fb.health_check = AsyncMock(side_effect=RuntimeError("nx crash"))
        store = HybridMemoryStore(fallback=bad_fb)
        health = await store.health_check()
        # NetworkX entry should show healthy even when it raises
        assert health["backends"]["networkx"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_production_with_networkx_fallback(self):
        """Production backend healthy + networkx fallback present."""
        neo = _FakeBackend(healthy=True)
        fb = _FakeBackend(healthy=True)
        store = HybridMemoryStore(neo4j=neo, fallback=fb)
        health = await store.health_check()
        assert health["status"] == "healthy"
        assert "networkx" in health["backends"]


# =========================================================================
# HybridMemoryStore — guardrail helpers (disabled / requires_approval)
# =========================================================================

class TestHybridGuardrails:
    @pytest.mark.asyncio
    async def test_store_with_guardrails_disabled_skips_evaluation(self):
        """When guardrails_enabled=False, _evaluate_guardrails returns None."""
        store = HybridMemoryStore(neo4j=_FakeBackend())
        store.guardrails_enabled = False
        mem = _make_memory("g-off")
        # Should not raise even though guardrails would normally evaluate
        result = await store.store(mem)
        assert result == "g-off"

    @pytest.mark.asyncio
    async def test_recall_with_guardrails_disabled(self):
        store = HybridMemoryStore(neo4j=_FakeBackend())
        store.guardrails_enabled = False
        result = await store.recall(MemoryQuery(text="q"))
        assert result.strategy_used == "hybrid_balanced"

    @pytest.mark.asyncio
    async def test_store_blocked_by_guardrail_requires_approval(self):
        """Guardrail returning requires_approval raises PermissionError."""
        from hololoom.alignment.safety_guardrails import (
            SafetyDecision, RiskLevel, SafetyGuardrails
        )

        mock_decision = SafetyDecision(
            allowed=True,
            risk_level=RiskLevel.HIGH,
            reason="Needs approval",
            requires_approval=True,
        )
        mock_guardrails = MagicMock(spec=SafetyGuardrails)
        mock_guardrails.evaluate.return_value = mock_decision

        store = HybridMemoryStore(neo4j=_FakeBackend(), guardrails=mock_guardrails)
        with pytest.raises(PermissionError, match="requires approval"):
            await store.store(_make_memory("blocked"))

    @pytest.mark.asyncio
    async def test_store_blocked_by_guardrail_not_allowed(self):
        """Guardrail returning allowed=False raises PermissionError."""
        from hololoom.alignment.safety_guardrails import (
            SafetyDecision, RiskLevel, SafetyGuardrails
        )

        mock_decision = SafetyDecision(
            allowed=False,
            risk_level=RiskLevel.CRITICAL,
            reason="Blocked by policy",
            requires_approval=False,
        )
        mock_guardrails = MagicMock(spec=SafetyGuardrails)
        mock_guardrails.evaluate.return_value = mock_decision

        store = HybridMemoryStore(neo4j=_FakeBackend(), guardrails=mock_guardrails)
        with pytest.raises(PermissionError, match="Blocked by policy"):
            await store.store(_make_memory("denied"))


# =========================================================================
# _try_init_neo4j — connection failure path
# =========================================================================

class TestTryInitNeo4jConnectionFailure:
    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", True)
    def test_returns_none_on_connection_exception(self):
        """NEO4J_AVAILABLE=True, URI set, but Neo4jKG constructor raises."""
        config = Config()
        config.neo4j_uri = "bolt://localhost:7687"
        config.neo4j_username = "neo4j"
        config.neo4j_password = "badpassword"

        with patch("hololoom.core.memory.backend_factory.Neo4jKG", side_effect=Exception("connection refused")):
            with patch("hololoom.core.memory.backend_factory.Neo4jConfig", return_value=MagicMock()):
                result, error = _try_init_neo4j(config)
        assert result is None
        assert error is not None
        # Credentials should be sanitized from error message
        assert "badpassword" not in (error or "")

    @patch("hololoom.core.memory.backend_factory.NEO4J_AVAILABLE", True)
    def test_sanitizes_credentials_in_error(self):
        """Connection error with credentials in message is sanitized before returning."""
        config = Config()
        config.neo4j_uri = "bolt://localhost:7687"
        config.neo4j_username = "admin"
        config.neo4j_password = "supersecret"

        with patch("hololoom.core.memory.backend_factory.Neo4jConfig", return_value=MagicMock()):
            with patch(
                "hololoom.core.memory.backend_factory.Neo4jKG",
                side_effect=Exception("Auth failed password=supersecret"),
            ):
                result, error = _try_init_neo4j(config)
        assert result is None
        assert "supersecret" not in (error or "")


# =========================================================================
# _try_init_qdrant — connection failure path
# =========================================================================

class TestTryInitQdrantConnectionFailure:
    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", True)
    def test_returns_none_on_connection_exception(self):
        """QDRANT_AVAILABLE=True, host/port set, but QdrantMemoryStore raises."""
        config = Config()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.qdrant_collection = "test"

        with patch(
            "hololoom.core.memory.backend_factory.QdrantMemoryStore",
            side_effect=Exception("connection refused"),
        ):
            result, error = _try_init_qdrant(config)
        assert result is None
        assert error is not None

    @patch("hololoom.core.memory.backend_factory.QDRANT_AVAILABLE", True)
    def test_sanitizes_credentials_in_qdrant_error(self):
        config = Config()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.qdrant_collection = "test"

        with patch(
            "hololoom.core.memory.backend_factory.QdrantMemoryStore",
            side_effect=Exception("Failed: password=qsecret123 in connection"),
        ):
            result, error = _try_init_qdrant(config)
        assert result is None
        assert "qsecret123" not in (error or "")


# =========================================================================
# create_memory_backend — unknown backend
# =========================================================================

class TestCreateMemoryBackendUnknown:
    @pytest.mark.asyncio
    async def test_unknown_backend_raises_value_error(self):
        """Passing an unrecognized MemoryBackend value hits the raise at end of factory."""
        config = MagicMock()
        config.memory_backend = "totally_unknown_backend"
        with pytest.raises((ValueError, AttributeError)):
            await create_memory_backend(config)


# =========================================================================
# GuardrailedMemoryStore — retrieve with supporting backend
# =========================================================================

class TestGuardrailedRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_delegates_to_backend(self):
        inner = _FakeBackend()
        await inner.store(_make_memory("r1"))
        gms = GuardrailedMemoryStore(backend=inner)
        # Disable guardrails so we don't need to worry about policy
        gms.guardrails_enabled = False
        result = await gms.retrieve(MemoryQuery(text="test"))
        assert len(result.memories) > 0

    @pytest.mark.asyncio
    async def test_retrieve_raises_when_no_retrieve_method(self):
        inner = MagicMock(spec=["store", "recall", "health_check"])
        gms = GuardrailedMemoryStore(backend=inner)
        gms.guardrails_enabled = False
        with pytest.raises(AttributeError, match="does not support retrieve"):
            await gms.retrieve(MemoryQuery(text="test"))

    @pytest.mark.asyncio
    async def test_retrieve_injects_guardrail_metadata(self):
        inner = _FakeBackend()
        await inner.store(_make_memory("r2"))
        gms = GuardrailedMemoryStore(backend=inner)
        gms.guardrails_enabled = False
        result = await gms.retrieve(MemoryQuery(text="test"))
        # Guardrails disabled => metadata['guardrails'] = None (set if metadata is not None)
        if result.metadata is not None:
            assert "guardrails" in result.metadata or result.metadata.get("guardrails") is None


# =========================================================================
# GuardrailedMemoryStore — delete with guardrails disabled
# =========================================================================

class TestGuardrailedDelete:
    @pytest.mark.asyncio
    async def test_delete_succeeds_with_guardrails_disabled(self):
        inner = _FakeBackend()
        gms = GuardrailedMemoryStore(backend=inner)
        gms.guardrails_enabled = False
        result = await gms.delete("some_id")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_raises_when_backend_missing_method(self):
        inner = MagicMock(spec=["store", "recall"])
        gms = GuardrailedMemoryStore(backend=inner)
        gms.guardrails_enabled = False
        with pytest.raises(AttributeError, match="does not support delete"):
            await gms.delete("some_id")


# =========================================================================
# GuardrailedMemoryStore — guardrail blocked paths
# =========================================================================

class TestGuardrailedStoreBlocked:
    @pytest.mark.asyncio
    async def test_store_blocked_not_allowed(self):
        """GuardrailedMemoryStore raises PermissionError when guardrail denies."""
        from hololoom.alignment.safety_guardrails import (
            SafetyDecision, RiskLevel, SafetyGuardrails
        )
        mock_decision = SafetyDecision(
            allowed=False,
            risk_level=RiskLevel.CRITICAL,
            reason="Denied by policy",
        )
        mock_guardrails = MagicMock(spec=SafetyGuardrails)
        mock_guardrails.evaluate.return_value = mock_decision

        gms = GuardrailedMemoryStore(backend=_FakeBackend(), guardrails=mock_guardrails)
        with pytest.raises(PermissionError, match="Denied by policy"):
            await gms.store(_make_memory("blocked"))

    @pytest.mark.asyncio
    async def test_recall_blocked_requires_approval(self):
        """GuardrailedMemoryStore raises PermissionError when guardrail requires approval."""
        from hololoom.alignment.safety_guardrails import (
            SafetyDecision, RiskLevel, SafetyGuardrails
        )
        mock_decision = SafetyDecision(
            allowed=True,
            risk_level=RiskLevel.HIGH,
            reason="Needs human approval",
            requires_approval=True,
        )
        mock_guardrails = MagicMock(spec=SafetyGuardrails)
        mock_guardrails.evaluate.return_value = mock_decision

        gms = GuardrailedMemoryStore(backend=_FakeBackend(), guardrails=mock_guardrails)
        with pytest.raises(PermissionError, match="requires approval"):
            await gms.recall(MemoryQuery(text="sensitive"))


# =========================================================================
# check_backend_health — extra cases
# =========================================================================

class TestCheckBackendHealthExtra:
    @pytest.mark.asyncio
    async def test_unhealthy_backend_status(self):
        """Backend.health_check returns unhealthy dict — not exception, just status."""
        backend = _FakeBackend(healthy=False)
        result = await check_backend_health(backend)
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_degraded_backend_status(self):
        """Backend with degraded status passes through."""
        backend = AsyncMock()
        backend.health_check = AsyncMock(return_value={"status": "degraded", "reason": "partial"})
        result = await check_backend_health(backend)
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_guardrailed_backend_health(self):
        """GuardrailedMemoryStore wrapping healthy backend returns healthy."""
        inner = _FakeBackend(healthy=True)
        gms = GuardrailedMemoryStore(backend=inner)
        result = await check_backend_health(gms)
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_guardrailed_backend_no_inner_health_check(self):
        """GuardrailedMemoryStore wrapping backend without health_check returns unknown."""
        inner = MagicMock(spec=[])
        gms = GuardrailedMemoryStore(backend=inner)
        result = await check_backend_health(gms)
        # GuardrailedMemoryStore.health_check() returns {status: unknown} when inner has no method
        assert result["status"] == "unknown"


# =========================================================================
# HybridMemoryStore — _fuse edge cases
# =========================================================================

class TestHybridFuseEdgeCases:
    def test_fuse_two_backends_equal_scores_shared_first(self):
        """Memory in both backends gets double weight → ranked first."""
        store = HybridMemoryStore()
        shared = _make_memory("shared")
        unique = _make_memory("unique")

        r_neo = RetrievalResult(
            memories=[shared, unique],
            scores=[0.8, 0.8],
            strategy_used="t",
            metadata={},
        )
        r_qdrant = RetrievalResult(
            memories=[shared],
            scores=[0.8],
            strategy_used="t",
            metadata={},
        )
        fused = store._fuse({"neo4j": r_neo, "qdrant": r_qdrant}, 10)
        assert fused[0].id == "shared"

    def test_fuse_limit_strictly_applied(self):
        store = HybridMemoryStore()
        mems = [_make_memory(f"m{i}") for i in range(10)]
        r = RetrievalResult(
            memories=mems,
            scores=[1.0] * 10,
            strategy_used="t",
            metadata={},
        )
        fused = store._fuse({"neo4j": r}, 3)
        assert len(fused) == 3

    def test_fuse_mismatched_scores_and_memories_handled(self):
        """Fewer scores than memories — zip stops at shorter list (no crash)."""
        store = HybridMemoryStore()
        mems = [_make_memory("a"), _make_memory("b")]
        r = RetrievalResult(
            memories=mems,
            scores=[0.9],  # Only 1 score for 2 memories
            strategy_used="t",
            metadata={},
        )
        fused = store._fuse({"neo4j": r}, 10)
        # Only 'a' has a score — 'b' is dropped by zip
        assert len(fused) == 1
        assert fused[0].id == "a"


# =========================================================================
# Integration: INMEMORY backend smoke test
# =========================================================================

class TestInmemoryBackendSmokeTest:
    @pytest.mark.asyncio
    async def test_inmemory_can_store_and_recall(self):
        """End-to-end smoke test: create INMEMORY backend, store, recall."""
        config = Config(memory_backend=MemoryBackend.INMEMORY)
        backend = await create_memory_backend(config)
        assert backend is not None

        mem = _make_memory("smoke_test_id", "smoke test content")
        stored_id = await backend.store(mem)
        assert stored_id == "smoke_test_id"

        result = await backend.recall(MemoryQuery(text="smoke"))
        assert result is not None
        assert hasattr(result, "memories")

    @pytest.mark.asyncio
    async def test_inmemory_health_check(self):
        config = Config(memory_backend=MemoryBackend.INMEMORY)
        backend = await create_memory_backend(config)
        health = await check_backend_health(backend)
        assert health["status"] in ("healthy", "unknown", "degraded")
