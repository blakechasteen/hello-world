"""
Unit tests for data models — 2026-02-09
"""

from datetime import datetime

from memory_bus.models import (
    AuditRow,
    AuditAction,
    DetailLevel,
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    MemoryQuery,
    MemoryResult,
    MemoryStoreRequest,
    PlanRow,
    PlanState,
    PressureSignals,
    PressureTier,
    ResolveCandidate,
    ResolveResult,
)


class TestPressureSignals:
    def test_tier0_low_utilization(self):
        s = PressureSignals(context_utilization_pct=10.0)
        assert s.tier == PressureTier.TIER0

    def test_tier1_moderate(self):
        s = PressureSignals(context_utilization_pct=55.0)
        assert s.tier == PressureTier.TIER1

    def test_tier2_high(self):
        s = PressureSignals(context_utilization_pct=80.0)
        assert s.tier == PressureTier.TIER2

    def test_tier3_critical(self):
        s = PressureSignals(context_utilization_pct=95.0)
        assert s.tier == PressureTier.TIER3

    def test_tier_boundary_50(self):
        s = PressureSignals(context_utilization_pct=50.0)
        assert s.tier == PressureTier.TIER1

    def test_tier_boundary_75(self):
        s = PressureSignals(context_utilization_pct=75.0)
        assert s.tier == PressureTier.TIER2

    def test_tier_boundary_90(self):
        s = PressureSignals(context_utilization_pct=90.0)
        assert s.tier == PressureTier.TIER3


class TestEntity:
    def test_default_id_generated(self):
        e = Entity(name="Test")
        assert e.id.startswith("ent_")
        assert len(e.id) > 4

    def test_explicit_fields(self):
        e = Entity(
            id="ent_42",
            type=EntityType.PERSON,
            name="Alice",
            importance=0.9,
        )
        assert e.id == "ent_42"
        assert e.type == EntityType.PERSON
        assert e.importance == 0.9


class TestEpisode:
    def test_default_id_generated(self):
        ep = Episode(summary="test")
        assert ep.id.startswith("ep_")

    def test_entity_ids(self):
        ep = Episode(
            summary="Observed behavior",
            entity_ids=["ent_1", "ent_2"],
        )
        assert len(ep.entity_ids) == 2


class TestMemoryQuery:
    def test_defaults(self):
        q = MemoryQuery(text="hello")
        assert q.max_results == 20
        assert q.detail_level == DetailLevel.COMPACT
        assert q.entity_ids == []


class TestMemoryResult:
    def test_empty_result(self):
        r = MemoryResult()
        assert r.items == []
        assert r.total_tokens_used == 0
        assert r.formatted_context == ""


class TestResolveResult:
    def test_best_match(self):
        c1 = ResolveCandidate(entity_id="e1", name="Aurora", confidence=0.9)
        c2 = ResolveCandidate(entity_id="e2", name="Aura", confidence=0.5)
        r = ResolveResult(
            query_name="Aurora",
            candidates=[c1, c2],
            best_match=c1,
        )
        assert r.best_match.entity_id == "e1"
        assert len(r.candidates) == 2


class TestAuditRow:
    def test_defaults(self):
        row = AuditRow(action=AuditAction.QUERY)
        assert row.tokens_used == 0
        assert row.pressure_tier == PressureTier.TIER0

    def test_serialization(self):
        row = AuditRow(
            action=AuditAction.STORE,
            details={"entity_id": "ent_1"},
        )
        d = row.model_dump(mode="json")
        assert d["action"] == "store"
        assert d["details"]["entity_id"] == "ent_1"


class TestPlanRow:
    def test_defaults(self):
        p = PlanRow(name="Test plan")
        assert p.state == PlanState.DRAFT
        assert p.progress == 0.0
