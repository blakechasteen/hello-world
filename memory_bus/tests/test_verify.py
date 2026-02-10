"""
Unit tests for entity verification — 2026-02-10

Tests the post-response entity verification module:
- Entity mention extraction from text
- Grounding cascade (primed names → aliases → neo4j)
- Confabulation rate calculation
- Audit logging of verification results
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from memory_bus.models import (
    AuditAction,
    EntityType,
    Entity,
    MemoryItem,
    ResolutionPath,
)
from memory_bus.verify import (
    EntityMention,
    EntityVerifier,
    VerificationResult,
    extract_entity_mentions,
)


# ---------------------------------------------------------------------------
# Entity extraction tests
# ---------------------------------------------------------------------------


class TestEntityExtraction:
    def test_extracts_capitalized_names(self):
        text = "Aurora is a healthy hive. Bombus is nearby."
        mentions = extract_entity_mentions(text)
        names = [m for m in mentions]
        assert "Aurora" in names
        assert "Bombus" in names

    def test_extracts_multi_word_names(self):
        text = "Queen Elizabeth was spotted near North Garden."
        mentions = extract_entity_mentions(text)
        assert "Queen Elizabeth" in mentions
        assert "North Garden" in mentions

    def test_filters_stopwords(self):
        text = "The quick fox. This is important. However it failed."
        mentions = extract_entity_mentions(text)
        assert "The" not in mentions
        assert "This" not in mentions
        assert "However" not in mentions

    def test_filters_months_days(self):
        text = "On Monday in January, we inspected."
        mentions = extract_entity_mentions(text)
        assert "Monday" not in mentions
        assert "January" not in mentions

    def test_deduplicates(self):
        text = "Aurora is great. Aurora is thriving."
        mentions = extract_entity_mentions(text)
        assert mentions.count("Aurora") == 1

    def test_empty_text(self):
        assert extract_entity_mentions("") == []

    def test_no_entities(self):
        text = "all lowercase text with no entities at all"
        assert extract_entity_mentions(text) == []


# ---------------------------------------------------------------------------
# VerificationResult tests
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_confabulation_rate_all_grounded(self):
        mentions = [
            EntityMention("Aurora", grounded=True, grounded_via="primed_name"),
            EntityMention("Bombus", grounded=True, grounded_via="alias"),
        ]
        result = VerificationResult(mentions=mentions, response_text="test")
        assert result.confabulation_rate == 0.0
        assert result.grounding_rate == 1.0

    def test_confabulation_rate_none_grounded(self):
        mentions = [
            EntityMention("Phantom", grounded=False),
            EntityMention("Ghost", grounded=False),
        ]
        result = VerificationResult(mentions=mentions, response_text="test")
        assert result.confabulation_rate == 1.0
        assert result.grounding_rate == 0.0

    def test_confabulation_rate_mixed(self):
        mentions = [
            EntityMention("Aurora", grounded=True, grounded_via="primed_name"),
            EntityMention("Phantom", grounded=False),
            EntityMention("Bombus", grounded=True, grounded_via="alias"),
            EntityMention("Ghost", grounded=False),
        ]
        result = VerificationResult(mentions=mentions, response_text="test")
        assert result.confabulation_rate == 0.5
        assert result.grounding_rate == 0.5

    def test_confabulation_rate_empty(self):
        result = VerificationResult(mentions=[], response_text="test")
        assert result.confabulation_rate == 0.0
        assert result.grounding_rate == 1.0

    def test_to_dict(self):
        mentions = [
            EntityMention("Aurora", grounded=True, grounded_via="primed_name"),
            EntityMention("Phantom", grounded=False),
        ]
        result = VerificationResult(mentions=mentions, response_text="test")
        d = result.to_dict()
        assert d["total_mentions"] == 2
        assert d["grounded_count"] == 1
        assert d["ungrounded_count"] == 1
        assert d["confabulation_rate"] == 0.5
        assert len(d["grounded"]) == 1
        assert len(d["ungrounded"]) == 1
        assert d["grounded"][0]["text"] == "Aurora"
        assert d["ungrounded"][0]["text"] == "Phantom"


# ---------------------------------------------------------------------------
# EntityVerifier integration tests
# ---------------------------------------------------------------------------


def _mock_neo4j_for_verify():
    neo4j = AsyncMock()
    neo4j.search_entities_by_name = AsyncMock(return_value=[])
    return neo4j


def _mock_pg_for_verify():
    pg = AsyncMock()
    pg.find_aliases = AsyncMock(return_value=[])
    pg.write_audit = AsyncMock()
    return pg


@pytest.mark.asyncio
class TestEntityVerifier:
    async def test_grounding_via_primed_names(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()
        verifier = EntityVerifier(neo4j=neo4j, pg=pg)

        result = await verifier.verify(
            response_text="Aurora is healthy. Bombus is nearby.",
            primed_entity_names=["Aurora"],
        )

        # Aurora should be grounded, Bombus should not
        grounded = {m.text for m in result.grounded_mentions}
        ungrounded = {m.text for m in result.ungrounded_mentions}
        assert "Aurora" in grounded
        assert "Bombus" in ungrounded

    async def test_grounding_via_primed_items(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()
        verifier = EntityVerifier(neo4j=neo4j, pg=pg)

        primed_items = [
            MemoryItem(
                id="ent_1", kind="entity", title="Bombus",
                resolution_path=ResolutionPath.EXACT,
            ),
        ]

        result = await verifier.verify(
            response_text="Bombus is thriving.",
            primed_items=primed_items,
        )

        assert result.grounded_count == 1
        assert result.grounded_mentions[0].text == "Bombus"

    async def test_grounding_via_alias(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()

        from memory_bus.models import EntityAlias
        pg.find_aliases = AsyncMock(side_effect=lambda name, **kw: (
            [EntityAlias(alias=name.lower(), entity_id="ent_x", confidence=0.9)]
            if name.lower() == "bombus" else []
        ))

        verifier = EntityVerifier(neo4j=neo4j, pg=pg)
        result = await verifier.verify(
            response_text="Bombus is a genus of bees.",
        )

        grounded = {m.text for m in result.grounded_mentions}
        assert "Bombus" in grounded
        assert result.grounded_mentions[0].grounded_via == "alias"
        assert result.grounded_mentions[0].entity_id == "ent_x"

    async def test_grounding_via_neo4j_search(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()

        neo4j.search_entities_by_name = AsyncMock(side_effect=lambda name, **kw: (
            [Entity(id="ent_y", name="Aurora", type=EntityType.OBJECT)]
            if "aurora" in name.lower() else []
        ))

        verifier = EntityVerifier(neo4j=neo4j, pg=pg)
        result = await verifier.verify(
            response_text="Aurora is a great hive.",
        )

        grounded = {m.text for m in result.grounded_mentions}
        assert "Aurora" in grounded
        assert result.grounded_mentions[0].grounded_via == "neo4j_search"
        assert result.grounded_mentions[0].entity_id == "ent_y"

    async def test_ungrounded_entity(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()
        verifier = EntityVerifier(neo4j=neo4j, pg=pg)

        result = await verifier.verify(
            response_text="Phantom Hive produced lots of honey.",
        )

        # "Phantom Hive" should be ungrounded (not in any memory)
        assert result.ungrounded_count > 0
        ungrounded_names = {m.text for m in result.ungrounded_mentions}
        assert "Phantom Hive" in ungrounded_names

    async def test_audit_written(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()
        verifier = EntityVerifier(neo4j=neo4j, pg=pg)

        await verifier.verify(
            response_text="Aurora is healthy.",
            primed_entity_names=["Aurora"],
            loom_id="loom_test",
            loop_id="loop_42",
        )

        pg.write_audit.assert_called_once()
        audit_row = pg.write_audit.call_args[0][0]
        assert audit_row.action == AuditAction.VERIFY
        assert audit_row.loom_id == "loom_test"
        assert audit_row.loop_id == "loop_42"
        assert "confabulation_rate" in audit_row.details

    async def test_case_insensitive_grounding(self):
        neo4j = _mock_neo4j_for_verify()
        pg = _mock_pg_for_verify()
        verifier = EntityVerifier(neo4j=neo4j, pg=pg)

        result = await verifier.verify(
            response_text="Aurora is healthy.",
            primed_entity_names=["aurora"],  # lowercase
        )

        assert result.grounded_count == 1
        assert result.confabulation_rate == 0.0


# ---------------------------------------------------------------------------
# Aggressive priming router tests
# ---------------------------------------------------------------------------

from memory_bus.models import (
    MemoryQuery,
    Episode,
    EpisodeType,
)
from memory_bus.pressure import PressureEngine
from memory_bus.router import MemoryRouter


def _make_entity(id: str = "ent_1", name: str = "Aurora") -> Entity:
    return Entity(id=id, name=name, type=EntityType.OBJECT, importance=0.8)


def _make_episode(id: str = "ep_1", entity_id: str = "ent_1") -> Episode:
    return Episode(
        id=id, type=EpisodeType.OBSERVATION,
        summary="Test episode", significance=0.7,
        entity_ids=[entity_id],
    )


def _mock_neo4j(entity=None, episodes=None, recent=None):
    neo4j = AsyncMock()
    neo4j.get_entity = AsyncMock(return_value=entity)
    neo4j.get_episode = AsyncMock(return_value=None)
    neo4j.query_episodes_for_entity = AsyncMock(return_value=episodes or [])
    neo4j.query_recent_episodes = AsyncMock(return_value=recent or [])
    neo4j.search_entities_by_name = AsyncMock(return_value=[entity] if entity else [])

    from memory_bus.neo4j_adapter import Neo4jAdapter
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    neo4j.entity_to_item = adapter.entity_to_item
    neo4j.episode_to_item = adapter.episode_to_item
    return neo4j


def _mock_pg():
    pg = AsyncMock()
    pg.write_audit = AsyncMock()
    pg.find_aliases = AsyncMock(return_value=[])
    return pg


@pytest.mark.asyncio
class TestAggressivePriming:
    async def test_aggressive_prime_adds_recent_episodes(self):
        entity = _make_entity()
        episode = _make_episode(id="ep_1")
        recent_ep = _make_episode(id="ep_recent")

        neo4j = _mock_neo4j(
            entity=entity,
            episodes=[episode],
            recent=[recent_ep],
        )

        router = MemoryRouter(
            neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine(),
        )

        query = MemoryQuery(
            text="", entity_ids=["ent_1"],
            aggressive_prime=True,
        )
        result = await router.route(query)

        # Should have entity + episode + recent episode
        ids = {i.id for i in result.items}
        assert "ent_1" in ids
        assert "ep_1" in ids
        assert "ep_recent" in ids

    async def test_aggressive_prime_deduplicates(self):
        entity = _make_entity()
        episode = _make_episode(id="ep_1")

        # Recent episodes include the same one already fetched
        neo4j = _mock_neo4j(
            entity=entity,
            episodes=[episode],
            recent=[episode],  # Same episode
        )

        router = MemoryRouter(
            neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine(),
        )

        query = MemoryQuery(
            text="", entity_ids=["ent_1"],
            aggressive_prime=True,
        )
        result = await router.route(query)

        # No duplicates
        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids))

    async def test_aggressive_prime_disabled_at_tier2(self):
        entity = _make_entity()
        episode = _make_episode(id="ep_1")
        recent_ep = _make_episode(id="ep_extra")

        neo4j = _mock_neo4j(
            entity=entity,
            episodes=[episode],
            recent=[recent_ep],
        )

        pressure = PressureEngine(context_window=100_000)
        pressure.update(context_tokens_used=80_000)  # Tier2

        router = MemoryRouter(
            neo4j=neo4j, pg=_mock_pg(), pressure=pressure,
        )

        query = MemoryQuery(
            text="", entity_ids=["ent_1"],
            aggressive_prime=True,
        )
        result = await router.route(query)

        # At Tier2, aggressive priming should NOT add recent episodes
        ids = {i.id for i in result.items}
        assert "ep_extra" not in ids

    async def test_no_aggressive_prime_by_default(self):
        entity = _make_entity()
        episode = _make_episode(id="ep_1")
        recent_ep = _make_episode(id="ep_extra")

        neo4j = _mock_neo4j(
            entity=entity,
            episodes=[episode],
            recent=[recent_ep],
        )

        router = MemoryRouter(
            neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine(),
        )

        # Default: aggressive_prime=False
        query = MemoryQuery(text="", entity_ids=["ent_1"])
        result = await router.route(query)

        ids = {i.id for i in result.items}
        assert "ep_extra" not in ids
