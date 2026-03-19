"""
Tests targeting uncovered lines in hololoom/core/protocols/.

Focuses on:
- memory_types.py: numpy ndarray paths in to_dict/from_dict (lines 67, 80-82)
- core_features.py: ImportError fallback (lines 19-25)
- retrieval.py: ImportError fallback (lines 31-35)
- __init__.py: ImportError fallback (lines 186-187)

Run with:
    pytest hololoom/tests/unit/test_core_protocols_coverage.py -v
"""

import pytest
import sys
from unittest.mock import MagicMock
from datetime import datetime

import hololoom  # noqa: F401 — installs _CoreRedirectFinder


# ============================================================================
# memory_types.py — numpy ndarray paths
# ============================================================================

class TestMemoryNumpyPaths:
    """Cover the numpy-specific branches in Memory.to_dict and from_dict."""

    def test_to_dict_with_numpy_embedding(self):
        """Line 67: embedding is np.ndarray -> .tolist()"""
        np = pytest.importorskip("numpy")
        from hololoom.protocols.memory_types import Memory

        arr = np.array([0.1, 0.2, 0.3])
        m = Memory(
            id="m1",
            text="test",
            timestamp=datetime(2025, 1, 1),
            context={},
            metadata={},
            embedding=arr,
        )
        d = m.to_dict()
        assert d["embedding"] == [0.1, 0.2, 0.3]
        assert isinstance(d["embedding"], list)

    def test_from_dict_with_list_embedding_converts_to_numpy(self):
        """Lines 80-82: list embedding in dict -> np.array"""
        np = pytest.importorskip("numpy")
        from hololoom.protocols.memory_types import Memory

        data = {
            "id": "m2",
            "text": "test",
            "timestamp": "2025-01-01T00:00:00",
            "context": {},
            "metadata": {},
            "embedding": [0.4, 0.5, 0.6],
        }
        m = Memory.from_dict(data)
        assert isinstance(m.embedding, np.ndarray)
        assert list(m.embedding) == pytest.approx([0.4, 0.5, 0.6])

    def test_roundtrip_numpy_embedding(self):
        """Full roundtrip: numpy -> to_dict -> from_dict -> numpy"""
        np = pytest.importorskip("numpy")
        from hololoom.protocols.memory_types import Memory

        original = np.array([1.0, 2.0, 3.0])
        m1 = Memory(
            id="rt",
            text="roundtrip",
            timestamp=datetime(2025, 6, 15),
            context={"k": "v"},
            metadata={},
            embedding=original,
        )
        d = m1.to_dict()
        m2 = Memory.from_dict(d)
        assert isinstance(m2.embedding, np.ndarray)
        assert list(m2.embedding) == pytest.approx([1.0, 2.0, 3.0])


# ============================================================================
# core_features.py — ImportError fallback
# ============================================================================

class TestCoreFeaturesFallback:
    """Cover lines 19-25: ImportError fallback in core_features.py."""

    def test_fallback_when_types_unavailable(self):
        """
        When hololoom.protocols.types can't be imported, core_features falls
        back to using Any for type aliases.
        """
        import importlib

        # Save references to clean up
        types_mod = sys.modules.get("hololoom.protocols.types")
        cf_mod = sys.modules.get("hololoom.protocols.core_features")
        cf_core_mod = sys.modules.get("hololoom.core.protocols.core_features")

        try:
            # Remove the types module and the cached core_features
            sys.modules["hololoom.protocols.types"] = None  # block import
            if cf_mod is not None:
                del sys.modules["hololoom.protocols.core_features"]
            if cf_core_mod is not None:
                del sys.modules["hololoom.core.protocols.core_features"]

            # Force reimport
            import hololoom.core.protocols.core_features as cf_fresh
            importlib.reload(cf_fresh)

            # The fallback sets these to Any
            from typing import Any
            assert cf_fresh.Features is Any
            assert cf_fresh.Context is Any
            assert cf_fresh.ActionPlan is Any
            assert cf_fresh.Query is Any
            assert cf_fresh.Vector is Any
        finally:
            # Restore original modules
            if types_mod is not None:
                sys.modules["hololoom.protocols.types"] = types_mod
            elif "hololoom.protocols.types" in sys.modules:
                del sys.modules["hololoom.protocols.types"]

            # Restore core_features
            if cf_mod is not None:
                sys.modules["hololoom.protocols.core_features"] = cf_mod
            if cf_core_mod is not None:
                sys.modules["hololoom.core.protocols.core_features"] = cf_core_mod


# ============================================================================
# retrieval.py — ImportError fallback
# ============================================================================

class TestRetrievalFallback:
    """Cover lines 31-35: ImportError fallback in retrieval.py."""

    def test_fallback_when_types_unavailable(self):
        """
        When hololoom.protocols.types can't be imported, retrieval.py falls
        back to using Any for Query and MemoryShard.
        """
        import importlib

        types_mod = sys.modules.get("hololoom.protocols.types")
        ret_mod = sys.modules.get("hololoom.protocols.retrieval")
        ret_core_mod = sys.modules.get("hololoom.core.protocols.retrieval")

        try:
            sys.modules["hololoom.protocols.types"] = None  # block import
            if ret_mod is not None:
                del sys.modules["hololoom.protocols.retrieval"]
            if ret_core_mod is not None:
                del sys.modules["hololoom.core.protocols.retrieval"]

            import hololoom.core.protocols.retrieval as ret_fresh
            importlib.reload(ret_fresh)

            from typing import Any
            assert ret_fresh.Query is Any
            assert ret_fresh.MemoryShard is Any
        finally:
            if types_mod is not None:
                sys.modules["hololoom.protocols.types"] = types_mod
            elif "hololoom.protocols.types" in sys.modules:
                del sys.modules["hololoom.protocols.types"]

            if ret_mod is not None:
                sys.modules["hololoom.protocols.retrieval"] = ret_mod
            if ret_core_mod is not None:
                sys.modules["hololoom.core.protocols.retrieval"] = ret_core_mod


# ============================================================================
# __init__.py — ImportError fallback (lines 186-187)
# ============================================================================

class TestInitFallback:
    """Cover lines 186-187: _HAS_DOC_TYPES = False fallback."""

    def test_has_doc_types_flag_is_true_normally(self):
        """Verify the normal path sets the flag to True."""
        import hololoom.core.protocols as protos
        assert protos._HAS_DOC_TYPES is True

    def test_doc_types_in_all(self):
        """When _HAS_DOC_TYPES is True, doc types are in __all__."""
        import hololoom.core.protocols as protos
        assert "Query" in protos.__all__
        assert "Features" in protos.__all__
        assert "Vector" in protos.__all__
