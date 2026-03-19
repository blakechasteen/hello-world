"""Tests for WorkingMemory capacity limits and pressure-based eviction."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from hololoom.memory.v2.blackboard import WorkingMemory, Blackboard


class TestLoad:
    def test_load_tracks_all_categories(self):
        wm = WorkingMemory()
        wm.active_goals = ["g1", "g2"]
        wm.entity_focus = {"e1": 0.9, "e2": 0.5, "e3": 0.3}
        wm.recent_corrections = ["c1"]
        wm.session_facts = {"k1": "v1", "k2": "v2"}
        assert wm.load == 2 + 3 + 1 + 2  # 8


class TestPressure:
    def test_pressure_equals_load_over_capacity(self):
        wm = WorkingMemory(capacity=10)
        wm.entity_focus = {f"e{i}": 0.5 for i in range(5)}
        assert wm.pressure == pytest.approx(0.5)

    def test_empty_wm_has_zero_pressure(self):
        wm = WorkingMemory()
        assert wm.pressure == 0.0


class TestEvict:
    def test_evict_removes_lowest_focus_entity_first(self):
        wm = WorkingMemory()
        wm.entity_focus = {"strong": 0.9, "weak": 0.1, "mid": 0.5}
        evicted = wm.evict(1)
        assert len(evicted) == 1
        assert "weak" in evicted[0]
        assert "weak" not in wm.entity_focus
        assert "strong" in wm.entity_focus
        assert "mid" in wm.entity_focus

    def test_evict_removes_oldest_correction_when_no_entities(self):
        wm = WorkingMemory()
        wm.recent_corrections = ["old_fix", "new_fix"]
        evicted = wm.evict(1)
        assert len(evicted) == 1
        assert "old_fix" in evicted[0]
        assert wm.recent_corrections == ["new_fix"]

    def test_evict_respects_priority_order(self):
        wm = WorkingMemory()
        wm.entity_focus = {"e1": 0.2}
        wm.recent_corrections = ["c1"]
        wm.session_facts = {"f1": "v1"}
        wm.active_goals = ["g1"]

        evicted = wm.evict(4)
        assert len(evicted) == 4
        assert evicted[0].startswith("entity:")
        assert evicted[1].startswith("correction:")
        assert evicted[2].startswith("fact:")
        assert evicted[3].startswith("goal:")

    def test_eviction_count_increments(self):
        wm = WorkingMemory()
        wm.entity_focus = {"e1": 0.1, "e2": 0.2, "e3": 0.3}
        assert wm.eviction_count == 0
        wm.evict(2)
        assert wm.eviction_count == 2
        wm.evict(1)
        assert wm.eviction_count == 3

    def test_eviction_returns_descriptive_items(self):
        wm = WorkingMemory()
        wm.entity_focus = {"alpha": 0.42}
        wm.recent_corrections = ["spelling fix"]
        wm.session_facts = {"pref": "dark_mode"}
        wm.active_goals = ["cook dinner"]

        evicted = wm.evict(4)
        assert any("alpha" in e and "0.42" in e for e in evicted)
        assert any("spelling fix" in e for e in evicted)
        assert any("pref=dark_mode" in e for e in evicted)
        assert any("cook dinner" in e for e in evicted)


class TestUpdateTurnAutoEvict:
    def test_update_turn_auto_evicts_when_over_capacity(self):
        wm = WorkingMemory(capacity=5)
        # Load 7 items (over capacity of 5)
        wm.entity_focus = {"e1": 0.9, "e2": 0.8, "e3": 0.7}
        wm.active_goals = ["g1", "g2"]
        wm.recent_corrections = ["c1", "c2"]
        assert wm.load == 7

        wm.update_turn()
        # After decay + auto-evict, load should be <= capacity
        assert wm.load <= wm.capacity
        assert wm.eviction_count > 0

    def test_capacity_limit_respected_after_heavy_loading(self):
        wm = WorkingMemory(capacity=10)
        # Massively overload
        for i in range(15):
            wm.entity_focus[f"e{i}"] = 0.9
        for i in range(10):
            wm.recent_corrections.append(f"c{i}")
        wm.session_facts = {f"f{i}": f"v{i}" for i in range(5)}
        wm.active_goals = ["g1", "g2", "g3"]
        assert wm.load == 33

        wm.update_turn()
        assert wm.load <= wm.capacity


class TestSnapshot:
    def test_round_trip_preserves_all_fields(self):
        wm = WorkingMemory()
        wm.active_goals = ["goal_a", "goal_b"]
        wm.entity_focus = {"ent1": 0.75, "ent2": 0.3}
        wm.recent_corrections = ["fix1", "fix2", "fix3"]
        wm.session_facts = {"theme": "dark", "lang": "en"}
        wm.turn_count = 42
        wm.eviction_count = 7

        snapshot = wm.to_snapshot()

        wm2 = WorkingMemory()
        wm2.from_snapshot(snapshot)

        assert wm2.active_goals == ["goal_a", "goal_b"]
        assert wm2.entity_focus == {"ent1": 0.75, "ent2": 0.3}
        assert wm2.recent_corrections == ["fix1", "fix2", "fix3"]
        assert wm2.session_facts == {"theme": "dark", "lang": "en"}
        assert wm2.turn_count == 42
        assert wm2.eviction_count == 7


class TestBlackboardProperties:
    def test_wm_pressure_and_wm_evictions(self):
        bb = Blackboard()
        wm = bb.working_memory
        wm.entity_focus = {f"e{i}": 0.5 for i in range(25)}
        wm.active_goals = ["g1"]
        # 26 items / 50 capacity = 0.52
        assert bb.wm_pressure == pytest.approx(26 / 50)
        assert bb.wm_evictions == 0

        wm.evict(3)
        assert bb.wm_evictions == 3
