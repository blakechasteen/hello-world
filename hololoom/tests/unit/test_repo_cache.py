"""Unit tests for RepoCache — persistent hash-indexed code chunk cache."""

import json
import tempfile
from pathlib import Path

import pytest

from hololoom.core.memory.repo_cache import RepoCache


def _sample_chunks():
    return [
        {"id": "chunk1", "content": "def foo(): pass", "token_estimate": 5},
        {"id": "chunk2", "content": "def bar(): pass", "token_estimate": 5},
    ]


class TestLookup:
    def test_miss_on_empty(self):
        cache = RepoCache()
        assert cache.lookup("file.py", "abc123") is None

    def test_hit_after_store(self):
        cache = RepoCache()
        chunks = _sample_chunks()
        cache.store("file.py", "abc123", chunks)
        assert cache.lookup("file.py", "abc123") == chunks

    def test_miss_on_hash_change(self):
        cache = RepoCache()
        cache.store("file.py", "abc123", _sample_chunks())
        assert cache.lookup("file.py", "def456") is None

    def test_miss_on_different_path(self):
        cache = RepoCache()
        cache.store("file.py", "abc123", _sample_chunks())
        assert cache.lookup("other.py", "abc123") is None


class TestInvalidation:
    def test_invalidate_single(self):
        cache = RepoCache()
        cache.store("file.py", "abc123", _sample_chunks())
        assert cache.invalidate("file.py") is True
        assert cache.lookup("file.py", "abc123") is None

    def test_invalidate_nonexistent(self):
        cache = RepoCache()
        assert cache.invalidate("missing.py") is False

    def test_invalidate_stale(self):
        cache = RepoCache()
        cache.store("a.py", "hash_a", _sample_chunks())
        cache.store("b.py", "hash_b", _sample_chunks())
        cache.store("c.py", "hash_c", _sample_chunks())

        # b.py changed, d.py is new, c.py deleted
        current = {"a.py": "hash_a", "b.py": "new_hash_b", "d.py": "hash_d"}
        removed = cache.invalidate_stale(current)
        assert removed == 2  # b.py (hash changed) + c.py (file gone)
        assert cache.lookup("a.py", "hash_a") is not None
        assert cache.lookup("b.py", "hash_b") is None
        assert cache.lookup("c.py", "hash_c") is None


class TestPersistence:
    def test_save_load_round_trip(self):
        cache = RepoCache()
        chunks = _sample_chunks()
        cache.store("file.py", "abc123", chunks, language="python",
                     entity_names=["foo", "bar"], line_count=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.repo_cache.json")
            cache.save(path)

            cache2 = RepoCache()
            assert cache2.load(path) is True
            assert cache2.lookup("file.py", "abc123") == chunks

    def test_load_nonexistent(self):
        cache = RepoCache()
        assert cache.load("/nonexistent/path.json") is False

    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "cache.json")
            cache = RepoCache()
            cache.store("f.py", "h", [{"id": "1"}])
            cache.save(path)
            assert Path(path).exists()

    def test_save_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            cache = RepoCache()
            cache.store("f.py", "h", _sample_chunks())
            cache.save(path)
            with open(path) as f:
                data = json.load(f)
            assert data["_version"] == 1
            assert data["_file_count"] == 1
            assert "f.py" in data["entries"]


class TestStats:
    def test_stats_track_hits_misses(self):
        cache = RepoCache()
        cache.store("f.py", "h", _sample_chunks())
        cache.lookup("f.py", "h")  # hit
        cache.lookup("f.py", "x")  # miss
        cache.lookup("g.py", "h")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["file_count"] == 1
        assert stats["total_chunks"] == 2

    def test_hit_rate(self):
        cache = RepoCache()
        cache.store("f.py", "h", _sample_chunks())
        cache.lookup("f.py", "h")
        cache.lookup("f.py", "h")
        cache.lookup("f.py", "bad")
        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(0.667, abs=0.01)


class TestUtilities:
    def test_hash_content(self):
        h1 = RepoCache.hash_content("hello")
        h2 = RepoCache.hash_content("hello")
        h3 = RepoCache.hash_content("world")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # SHA256

    def test_cached_files(self):
        cache = RepoCache()
        cache.store("a.py", "h1", [])
        cache.store("b.py", "h2", [])
        assert cache.cached_files() == {"a.py", "b.py"}

    def test_metadata_preserved(self):
        cache = RepoCache()
        cache.store("f.py", "h", _sample_chunks(), language="python",
                     entity_names=["foo"], line_count=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            cache.save(path)
            cache2 = RepoCache()
            cache2.load(path)
            entry = cache2._entries["f.py"]
            assert entry.language == "python"
            assert entry.entity_names == ["foo"]
            assert entry.line_count == 42
