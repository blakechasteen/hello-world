"""
Tests for ChunkCompiler auto-tuning.

Validates that compiled chunks self-correct their trigger ranges
based on accumulated success/failure data.
"""
import sys
import pytest

sys.path.insert(0, ".")

from hololoom.memory.v2.chunk_compiler import (
    ChunkCompiler,
    ChunkConfig,
    ChunkAction,
    CompiledChunk,
    ExperienceRecord,
)


def make_chunk(
    chunk_id="test_chunk",
    preset_id="focused",
    trigger=None,
    successes=0,
    failures=0,
    last_tuned_at=0,
):
    if trigger is None:
        trigger = {
            "query_length": (0.2, 0.6),
            "seed_count": (0.3, 0.7),
        }
    return CompiledChunk(
        chunk_id=chunk_id,
        trigger=trigger,
        action=ChunkAction(preset_id=preset_id),
        successes=successes,
        failures=failures,
        last_tuned_at=last_tuned_at,
    )


def make_compiler(interval=10, shrink=0.9, expand=1.15):
    return ChunkCompiler(config=ChunkConfig(
        auto_tune_interval=interval,
        auto_tune_shrink=shrink,
        auto_tune_expand=expand,
        store_in_graph=False,
        success_threshold=0.6,
    ))


# -----------------------------------------------------------------------
# 1. Auto-tune triggers after interval reached
# -----------------------------------------------------------------------

def test_auto_tune_triggers_after_interval():
    compiler = make_compiler(interval=5)
    chunk = make_chunk(successes=8, failures=2, last_tuned_at=5)
    compiler.add_chunk(chunk)
    original_trigger = dict(chunk.trigger)

    # total=10, last_tuned_at=5, diff=5 >= interval=5 => should tune
    compiler._auto_tune_chunk(chunk)

    assert chunk.last_tuned_at == 10
    assert chunk.trigger != original_trigger


# -----------------------------------------------------------------------
# 2. Auto-tune does NOT trigger before interval
# -----------------------------------------------------------------------

def test_auto_tune_does_not_trigger_before_interval():
    compiler = make_compiler(interval=10)
    chunk = make_chunk(successes=5, failures=2, last_tuned_at=0)
    compiler.add_chunk(chunk)
    original_trigger = dict(chunk.trigger)

    # total=7, last_tuned_at=0, diff=7 < interval=10 => no tune
    compiler._auto_tune_chunk(chunk)

    assert chunk.last_tuned_at == 0
    assert chunk.trigger == original_trigger


# -----------------------------------------------------------------------
# 3. Tighten on high hit_rate (ranges get narrower)
# -----------------------------------------------------------------------

def test_tighten_on_high_hit_rate():
    compiler = make_compiler(interval=5, shrink=0.8)
    chunk = make_chunk(
        trigger={"query_length": (0.2, 0.8), "seed_count": (0.1, 0.9)},
        successes=9,
        failures=1,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    # hit_rate = 9/10 = 0.9 >= 0.8 => tighten
    compiler._auto_tune_chunk(chunk)

    ql_lo, ql_hi = chunk.trigger["query_length"]
    sc_lo, sc_hi = chunk.trigger["seed_count"]

    # Original width was 0.6, shrunk by 0.8 => new width 0.48
    # Centroid for query_length was 0.5, so new range: (0.5-0.24, 0.5+0.24) = (0.26, 0.74)
    assert abs((ql_hi - ql_lo) - 0.6 * 0.8) < 1e-9
    assert abs((ql_lo + ql_hi) / 2.0 - 0.5) < 1e-9  # centroid preserved

    # Original width was 0.8, shrunk by 0.8 => new width 0.64
    assert abs((sc_hi - sc_lo) - 0.8 * 0.8) < 1e-9
    assert abs((sc_lo + sc_hi) / 2.0 - 0.5) < 1e-9  # centroid preserved


# -----------------------------------------------------------------------
# 4. Widen on low hit_rate (ranges get wider)
# -----------------------------------------------------------------------

def test_widen_on_low_hit_rate():
    compiler = make_compiler(interval=5, expand=1.2)
    chunk = make_chunk(
        trigger={"query_length": (0.3, 0.5), "seed_count": (0.4, 0.6)},
        successes=3,
        failures=7,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    # hit_rate = 3/10 = 0.3 < 0.6 => widen
    compiler._auto_tune_chunk(chunk)

    ql_lo, ql_hi = chunk.trigger["query_length"]
    sc_lo, sc_hi = chunk.trigger["seed_count"]

    # query_length: width 0.2, expanded by 1.2 => 0.24, centroid 0.4
    assert abs((ql_hi - ql_lo) - 0.2 * 1.2) < 1e-9
    assert abs((ql_lo + ql_hi) / 2.0 - 0.4) < 1e-9

    # seed_count: width 0.2, expanded by 1.2 => 0.24, centroid 0.5
    assert abs((sc_hi - sc_lo) - 0.2 * 1.2) < 1e-9
    assert abs((sc_lo + sc_hi) / 2.0 - 0.5) < 1e-9


# -----------------------------------------------------------------------
# 5. Recompute on mid hit_rate
# -----------------------------------------------------------------------

def test_recompute_on_mid_hit_rate():
    compiler = make_compiler(interval=5)
    chunk = make_chunk(
        trigger={
            "query_length": (0.1, 0.9),
            "seed_count": (0.1, 0.9),
        },
        preset_id="focused",
        successes=7,
        failures=3,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    # Add matching experiences with tight features so recompute produces new ranges
    for i in range(6):
        compiler._experience.append(ExperienceRecord(
            features={"query_length": 0.45 + i * 0.01, "seed_count": 0.50,
                       "ppr_entropy": 0.3, "entity_ratio": 0.2},
            preset_id="focused",
            reward=0.8,
            shell_count=1,
        ))

    # hit_rate = 7/10 = 0.7, in [0.6, 0.8) => recompute
    compiler._auto_tune_chunk(chunk)

    assert chunk.last_tuned_at == 10
    # Ranges should have been recomputed from the tight cluster
    ql_lo, ql_hi = chunk.trigger["query_length"]
    # The recomputed ranges should be tighter than the original (0.1, 0.9)
    assert ql_hi - ql_lo < 0.8


# -----------------------------------------------------------------------
# 6. Ranges stay clamped to [0, 1]
# -----------------------------------------------------------------------

def test_ranges_clamped_to_zero_one():
    compiler = make_compiler(interval=5, expand=2.0)
    chunk = make_chunk(
        trigger={"query_length": (0.0, 0.1), "seed_count": (0.9, 1.0)},
        successes=2,
        failures=8,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    # hit_rate = 2/10 = 0.2 < 0.6 => widen with factor=2.0
    # query_length centroid=0.05, half_width=0.05, new_half=0.10
    #   new range: (0.05-0.10, 0.05+0.10) = (-0.05, 0.15) => clamped to (0.0, 0.15)
    # seed_count centroid=0.95, half_width=0.05, new_half=0.10
    #   new range: (0.95-0.10, 0.95+0.10) = (0.85, 1.05) => clamped to (0.85, 1.0)
    compiler._auto_tune_chunk(chunk)

    ql_lo, ql_hi = chunk.trigger["query_length"]
    sc_lo, sc_hi = chunk.trigger["seed_count"]

    assert ql_lo >= 0.0
    assert ql_hi <= 1.0
    assert sc_lo >= 0.0
    assert sc_hi <= 1.0

    assert abs(ql_lo - 0.0) < 1e-9
    assert abs(sc_hi - 1.0) < 1e-9


# -----------------------------------------------------------------------
# 7. last_tuned_at persists in snapshot round-trip
# -----------------------------------------------------------------------

def test_last_tuned_at_snapshot_round_trip():
    chunk = make_chunk(successes=15, failures=5, last_tuned_at=12)

    snapshot = chunk.to_snapshot()
    assert snapshot["last_tuned_at"] == 12

    restored = CompiledChunk.from_snapshot(snapshot)
    assert restored.last_tuned_at == 12
    assert restored.successes == 15
    assert restored.failures == 5
    assert restored.chunk_id == chunk.chunk_id


def test_last_tuned_at_in_to_dict():
    chunk = make_chunk(successes=10, failures=3, last_tuned_at=8)
    d = chunk.to_dict()
    assert "last_tuned_at" in d
    assert d["last_tuned_at"] == 8


# -----------------------------------------------------------------------
# 8. Auto-tune only considers experiences matching chunk's preset
# -----------------------------------------------------------------------

def test_auto_tune_only_matches_preset():
    compiler = make_compiler(interval=5)
    chunk = make_chunk(
        trigger={
            "query_length": (0.1, 0.9),
            "seed_count": (0.1, 0.9),
        },
        preset_id="focused",
        successes=7,
        failures=3,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    # Add experiences with WRONG preset (should be ignored in recompute)
    for i in range(10):
        compiler._experience.append(ExperienceRecord(
            features={"query_length": 0.01, "seed_count": 0.01,
                       "ppr_entropy": 0.01, "entity_ratio": 0.01},
            preset_id="wide",  # Wrong preset
            reward=0.9,
            shell_count=1,
        ))

    # Add experiences with correct preset but NOT matching trigger ranges
    for i in range(3):
        compiler._experience.append(ExperienceRecord(
            features={"query_length": 0.95, "seed_count": 0.95,
                       "ppr_entropy": 0.95, "entity_ratio": 0.95},
            preset_id="focused",
            reward=0.9,
            shell_count=1,
        ))

    # Add a few matching experiences with correct preset
    for i in range(4):
        compiler._experience.append(ExperienceRecord(
            features={"query_length": 0.5, "seed_count": 0.5,
                       "ppr_entropy": 0.5, "entity_ratio": 0.5},
            preset_id="focused",
            reward=0.8,
            shell_count=1,
        ))

    # hit_rate = 0.7 => recompute path
    compiler._auto_tune_chunk(chunk)

    # The recomputed ranges should be based only on the 4 matching "focused"
    # experiences at 0.5, not the "wide" ones at 0.01
    ql_lo, ql_hi = chunk.trigger["query_length"]
    centroid = (ql_lo + ql_hi) / 2.0
    # Centroid should be near 0.5, not near 0.01
    assert abs(centroid - 0.5) < 0.3


# -----------------------------------------------------------------------
# 9. Auto-tune fires through _record_experience path
# -----------------------------------------------------------------------

def test_auto_tune_fires_through_record_experience():
    compiler = make_compiler(interval=3)
    chunk = make_chunk(
        trigger={"query_length": (0.2, 0.8), "seed_count": (0.2, 0.8)},
        successes=9,
        failures=0,
        last_tuned_at=6,
    )
    compiler.add_chunk(chunk)
    original_width = 0.6  # 0.8 - 0.2

    # We need to drive _record_experience which needs a blackboard.
    # Import the Blackboard to create a mock context.
    from hololoom.memory.v2.blackboard import Blackboard, WorkingMemory

    for i in range(3):
        bb = Blackboard(query=f"q{i}", working_memory=WorkingMemory())
        bb.post_navigation(seed_count=3, seed_sources={"alias": 2},
                           ppr_entropy=0.3, ppr_converged=True, entity_ids=set())
        bb.post_confidence(combined=0.8, structural=0.7,
                           narrative=0.9, decision="sufficient")
        bb.post_flag("chunk_action", {"chunk_id": "test_chunk", "preset_id": "focused"})

        class MockLoop:
            shell_count = 1
            class final_confidence:
                combined = 0.85

        compiler._record_experience(bb, {"loop_result": MockLoop()})

    # After 3 successes: total=12, last_tuned_at=6, diff=6 >= interval=3
    # hit_rate = 12/12 = 1.0 >= 0.8 => tighten
    ql_lo, ql_hi = chunk.trigger["query_length"]
    new_width = ql_hi - ql_lo
    assert new_width < original_width


# -----------------------------------------------------------------------
# 10. Tighten preserves centroid exactly
# -----------------------------------------------------------------------

def test_tighten_preserves_centroid():
    compiler = make_compiler(interval=5, shrink=0.5)
    chunk = make_chunk(
        trigger={
            "query_length": (0.1, 0.7),
            "seed_count": (0.25, 0.75),
        },
        successes=9,
        failures=1,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    original_centroids = {}
    for fname, (lo, hi) in chunk.trigger.items():
        original_centroids[fname] = (lo + hi) / 2.0

    compiler._auto_tune_chunk(chunk)

    for fname, (lo, hi) in chunk.trigger.items():
        new_centroid = (lo + hi) / 2.0
        assert abs(new_centroid - original_centroids[fname]) < 1e-9, \
            f"Centroid shifted for {fname}: {original_centroids[fname]} -> {new_centroid}"


# -----------------------------------------------------------------------
# 11. Widen preserves centroid exactly
# -----------------------------------------------------------------------

def test_widen_preserves_centroid():
    compiler = make_compiler(interval=5, expand=1.5)
    chunk = make_chunk(
        trigger={
            "query_length": (0.3, 0.5),
            "seed_count": (0.4, 0.6),
        },
        successes=2,
        failures=8,
        last_tuned_at=5,
    )
    compiler.add_chunk(chunk)

    original_centroids = {}
    for fname, (lo, hi) in chunk.trigger.items():
        original_centroids[fname] = (lo + hi) / 2.0

    compiler._auto_tune_chunk(chunk)

    for fname, (lo, hi) in chunk.trigger.items():
        new_centroid = (lo + hi) / 2.0
        # Centroid may shift slightly due to clamping, but should be close
        assert abs(new_centroid - original_centroids[fname]) < 0.1, \
            f"Centroid shifted too much for {fname}"


# -----------------------------------------------------------------------
# 12. from_snapshot defaults last_tuned_at to 0 when missing
# -----------------------------------------------------------------------

def test_from_snapshot_defaults_last_tuned_at():
    snapshot = {
        "chunk_id": "legacy",
        "trigger": {"query_length": [0.2, 0.6]},
        "action": {"preset_id": "balanced"},
        "successes": 5,
        "failures": 2,
        "consecutive_failures": 0,
        # no last_tuned_at key
    }
    chunk = CompiledChunk.from_snapshot(snapshot)
    assert chunk.last_tuned_at == 0
