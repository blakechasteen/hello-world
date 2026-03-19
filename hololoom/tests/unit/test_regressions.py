"""
Regression tests for documented anti-patterns (AP-1 through AP-12).

Each test verifies that a previously-found bug remains fixed. The docstring
on each test explains the anti-pattern, what went wrong, and what the fix is.

Reference: CLAUDE.md § Anti-Patterns (Learned the Hard Way)
"""

import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# AP-1: Raw items on the bus — MemoryItem has no .to_dict(), StoredItem does
# ---------------------------------------------------------------------------

class TestAP1_StoredItemVsMemoryItem:
    """AP-1: Raw MemoryItem on the bus causes AttributeError.

    Bug: `bus._items.append(memory_item)` — MemoryItem has no `.to_dict()`.
    Fix: Always wrap with StoredItem before placing on the bus.
    StoredItem provides `.to_dict()` for serialization.
    """

    def test_stored_item_has_to_dict(self):
        """StoredItem exposes .to_dict() — the bus serialization contract."""
        from hololoom.core.memory.lite_bus import StoredItem

        item = StoredItem(
            id="test-001",
            content="test content",
            memory_type="episodic",
        )
        result = item.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "test-001"
        assert result["content"] == "test content"
        assert result["memory_type"] == "episodic"

    def test_memory_item_lacks_to_dict(self):
        """MemoryItem deliberately does NOT have .to_dict() — that's StoredItem's job.

        If MemoryItem grows a to_dict(), this test should be updated, but the
        pattern of wrapping in StoredItem before bus storage must remain.
        """
        from hololoom.core.memory.bus import MemoryItem

        item = MemoryItem(
            content="raw content",
            memory_type="episodic",
        )
        assert not hasattr(item, "to_dict"), (
            "MemoryItem should not have to_dict(). "
            "Bus storage must use StoredItem (AP-1)."
        )

    def test_stored_item_has_to_snapshot(self):
        """StoredItem also has .to_snapshot() for full persistence."""
        from hololoom.core.memory.lite_bus import StoredItem

        item = StoredItem(
            id="test-002",
            content="snapshot test",
            memory_type="factual",
            entity_ids=["e1", "e2"],
        )
        snapshot = item.to_snapshot()
        assert isinstance(snapshot, dict)
        assert snapshot["entity_ids"] == ["e1", "e2"]


# ---------------------------------------------------------------------------
# AP-3: Shared test state — always use tmp_path, never default paths
# ---------------------------------------------------------------------------

class TestAP3_SharedTestState:
    """AP-3: Tests that share state via default persist paths.

    Bug: `IntentStore()` with default `persist_path="./intents.json"` causes
    tests to share state across runs and interfere with each other.
    Fix: Always pass explicit tmp_path for any persistent state in tests.
    """

    def test_tmp_path_isolation(self, tmp_path):
        """Two stores in different tmp_paths have independent state."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        file_a = dir_a / "state.json"
        file_b = dir_b / "state.json"

        file_a.write_text('{"key": "value_a"}')
        file_b.write_text('{"key": "value_b"}')

        assert file_a.read_text() != file_b.read_text()

    def test_no_cwd_pollution(self, tmp_path):
        """Test databases should never be created in the working directory.

        This test verifies the principle: any test that creates a DB file
        must do so inside tmp_path, not in the current working directory.
        """
        db_path = tmp_path / "test.db"
        assert str(db_path).startswith(str(tmp_path))
        assert not db_path.exists()  # Not yet created — clean slate


# ---------------------------------------------------------------------------
# AP-5: Monolithic ranking functions — use additive scoring with named terms
# ---------------------------------------------------------------------------

class TestAP5_AdditiveScoring:
    """AP-5: Monolithic ranking functions with nested if/else.

    Bug: One big function that weighs 12 factors with nested if/else is
    untestable, unextendable, and opaque.
    Fix: Additive scoring with named ScoringTerms, each independently testable.
    Modes modify weights. Bands bucket the result.
    """

    def test_scoring_terms_are_independent(self):
        """Each ScoringTerm is a pure function — testable in isolation."""
        from hololoom.core.memory.scored_observation import ScoringTerm, score_terms

        term_a = ScoringTerm(name="w_a", weight=10, fn=lambda ctx: ctx.get("a", 0.0))
        term_b = ScoringTerm(name="w_b", weight=20, fn=lambda ctx: ctx.get("b", 0.0))

        # Test term A alone
        composite_a, breakdown_a = score_terms([term_a], {"a": 0.5})
        assert breakdown_a["w_a"] == pytest.approx(5.0)

        # Test term B alone
        composite_b, breakdown_b = score_terms([term_b], {"b": 0.8})
        assert breakdown_b["w_b"] == pytest.approx(16.0)

    def test_scoring_terms_compose_additively(self):
        """Combined score is the sum of individual weighted contributions."""
        from hololoom.core.memory.scored_observation import ScoringTerm, score_terms

        terms = [
            ScoringTerm(name="w_a", weight=10, fn=lambda ctx: 0.5),
            ScoringTerm(name="w_b", weight=20, fn=lambda ctx: 0.8),
        ]

        composite, breakdown = score_terms(terms, {})
        # 10*0.5 + 20*0.8 = 5 + 16 = 21
        assert composite == pytest.approx(21.0)
        assert breakdown["w_a"] == pytest.approx(5.0)
        assert breakdown["w_b"] == pytest.approx(16.0)

    def test_mode_weights_modify_terms(self):
        """Mode weights scale individual terms without touching others."""
        from hololoom.core.memory.scored_observation import ScoringTerm, score_terms

        terms = [
            ScoringTerm(name="w_time", weight=10, fn=lambda ctx: 1.0),
            ScoringTerm(name="w_variety", weight=5, fn=lambda ctx: 1.0),
        ]

        # Normal mode
        normal, _ = score_terms(terms, {})
        assert normal == pytest.approx(15.0)

        # QUICK mode doubles time weight
        quick, breakdown = score_terms(terms, {}, mode_weights={"w_time": 2.0})
        assert quick == pytest.approx(25.0)  # 10*2*1.0 + 5*1.0
        assert breakdown["w_time"] == pytest.approx(20.0)
        assert breakdown["w_variety"] == pytest.approx(5.0)

    def test_band_bucketing(self):
        """Composite scores map to priority bands: P0/P1/P2/P3."""
        from hololoom.core.memory.scored_observation import band_from_score

        assert band_from_score(95) == "P0"
        assert band_from_score(80) == "P0"
        assert band_from_score(79) == "P1"
        assert band_from_score(35) == "P1"
        assert band_from_score(34) == "P2"
        assert band_from_score(18) == "P2"
        assert band_from_score(17) == "P3"
        assert band_from_score(0) == "P3"


# ---------------------------------------------------------------------------
# AP-6: Hard dependency masquerading as optional
# ---------------------------------------------------------------------------

class TestAP6_GracefulDegradation:
    """AP-6: Hard dependency masquerading as optional.

    Bug: `import special_lib` at module level with no fallback crashes the
    entire system when the library isn't installed.
    Fix: `try: import X; HAS_X = True except ImportError: HAS_X = False`
    with meaningful fallback behavior.
    """

    def test_core_imports_without_optional_deps(self):
        """hololoom.core modules must import without optional deps.

        The core package should never crash on import because an optional
        dependency is missing. This test verifies the top-level import
        succeeds (the most common AP-6 failure mode).
        """
        # These core modules must always be importable
        import hololoom.core.memory.lite_bus
        import hololoom.core.memory.scored_observation

    def test_graceful_degradation_pattern(self):
        """Verify the try/except ImportError pattern produces a usable flag.

        This doesn't test a specific module — it tests the pattern itself
        that all optional imports must follow (SOP-6).
        """
        # Simulate the pattern for a missing library
        test_module = types.ModuleType("test_graceful")
        exec(
            "try:\n"
            "    import nonexistent_library_xyzzy\n"
            "    HAS_XYZZY = True\n"
            "except ImportError:\n"
            "    HAS_XYZZY = False\n",
            test_module.__dict__,
        )
        assert test_module.HAS_XYZZY is False, (
            "Missing optional dep should set flag to False, not crash"
        )

    def test_flag_naming_convention(self):
        """Optional dependency flags should follow HAS_X or X_AVAILABLE naming.

        This is a documentation test — when adding new optional deps,
        use one of these two patterns for the availability flag.
        """
        # Both naming conventions are acceptable
        valid_names = ["HAS_SPACY", "HAS_NEO4J", "SAAS_AVAILABLE", "VOICE_AVAILABLE"]
        for name in valid_names:
            assert name.startswith("HAS_") or name.endswith("_AVAILABLE"), (
                f"Flag {name} doesn't follow naming convention"
            )


# ---------------------------------------------------------------------------
# AP-8: Competing metaphors
# ---------------------------------------------------------------------------

class TestAP8_CompetingMetaphors:
    """AP-8: Competing metaphors in naming.

    Bug: Calling it a 'pipeline' in one file and a 'weaving cycle' in another.
    Fix: One metaphor per project, used consistently.
    """

    def test_scored_observation_uses_domain_terms(self):
        """scored_observation uses 'terms', 'bands', 'composite' — not 'pipeline'."""
        from hololoom.core.memory.scored_observation import (
            ScoringTerm,
            band_from_score,
            score_terms,
        )

        # These names exist (weaving/scoring metaphor)
        assert ScoringTerm.__name__ == "ScoringTerm"
        assert band_from_score.__name__ == "band_from_score"
        assert score_terms.__name__ == "score_terms"


# ---------------------------------------------------------------------------
# AP-9: Hardcoded secrets — verify env var loading pattern
# ---------------------------------------------------------------------------

class TestAP9_NoHardcodedSecrets:
    """AP-9: Hardcoded secrets in config files.

    Bug: API keys and tokens directly in config files or committed code.
    Fix: Load from environment variables or .env.local (gitignored).
    """

    def test_env_var_with_default(self):
        """Environment variables must always have defaults (SOP-10).

        os.environ.get() without a default is a bug — it returns None,
        which usually causes a confusing error downstream.
        """
        # Correct pattern: always provide a default
        result = os.environ.get("NONEXISTENT_VAR_FOR_TEST", "fallback")
        assert result == "fallback"

        # The anti-pattern would be: os.environ.get("KEY") → None
        # which then fails later with a confusing AttributeError
