"""
Post-response entity verification — 2026-02-10

Compensating design pattern #2: After the model generates a response,
check whether entities it mentioned were backed by memory queries
(grounded) or generated from training data (ungrounded).

Ungrounded entities are confabulation signals. The orchestrator can
then re-query the Memory Bus for those entities and ask the model
to revise with real data injected.

Usage:
    verifier = EntityVerifier(neo4j=neo4j, pg=pg)
    result = await verifier.verify(
        response_text="Aurora's queen is healthy...",
        primed_entity_ids=["ent_aurora"],
        primed_entity_names=["Aurora"],
        loom_id="loom_bees",
        loop_id="loop_42",
    )
    # result.ungrounded_mentions → entities mentioned but not in memory
    # result.confabulation_rate → ratio of ungrounded / total
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .models import (
    AuditAction,
    AuditRow,
    MemoryItem,
)
from .neo4j_adapter import Neo4jAdapter
from .postgres_adapter import PostgresAdapter

logger = logging.getLogger(__name__)


class EntityMention:
    """A detected entity mention in model output."""

    __slots__ = ("text", "grounded", "grounded_via", "entity_id")

    def __init__(
        self,
        text: str,
        grounded: bool = False,
        grounded_via: str = "",
        entity_id: str | None = None,
    ):
        self.text = text
        self.grounded = grounded
        self.grounded_via = grounded_via  # "primed_id" | "primed_name" | "alias" | "neo4j_search"
        self.entity_id = entity_id

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "grounded": self.grounded,
            "grounded_via": self.grounded_via,
            "entity_id": self.entity_id,
        }


class VerificationResult:
    """Result of post-response entity verification."""

    def __init__(
        self,
        mentions: list[EntityMention],
        response_text: str,
    ):
        self.mentions = mentions
        self.response_text = response_text

    @property
    def grounded_mentions(self) -> list[EntityMention]:
        return [m for m in self.mentions if m.grounded]

    @property
    def ungrounded_mentions(self) -> list[EntityMention]:
        return [m for m in self.mentions if not m.grounded]

    @property
    def total_mentions(self) -> int:
        return len(self.mentions)

    @property
    def grounded_count(self) -> int:
        return len(self.grounded_mentions)

    @property
    def ungrounded_count(self) -> int:
        return len(self.ungrounded_mentions)

    @property
    def confabulation_rate(self) -> float:
        """Ratio of ungrounded mentions to total. 0.0 = fully grounded."""
        if self.total_mentions == 0:
            return 0.0
        return self.ungrounded_count / self.total_mentions

    @property
    def grounding_rate(self) -> float:
        """Ratio of grounded mentions to total. 1.0 = fully grounded."""
        if self.total_mentions == 0:
            return 1.0
        return self.grounded_count / self.total_mentions

    def to_dict(self) -> dict:
        return {
            "total_mentions": self.total_mentions,
            "grounded_count": self.grounded_count,
            "ungrounded_count": self.ungrounded_count,
            "confabulation_rate": round(self.confabulation_rate, 4),
            "grounding_rate": round(self.grounding_rate, 4),
            "grounded": [m.to_dict() for m in self.grounded_mentions],
            "ungrounded": [m.to_dict() for m in self.ungrounded_mentions],
        }


# ---------------------------------------------------------------------------
# Entity extraction heuristics
# ---------------------------------------------------------------------------

# Capitalized multi-word phrases likely to be entity names
_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

# Common false positives (sentence starters, generic terms)
_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "There",
    "Here", "When", "Where", "What", "Which", "Who",
    "How", "Why", "Some", "Any", "All", "Each",
    "Every", "Many", "Much", "More", "Most", "Other",
    "Another", "Both", "Few", "Several", "Such",
    "However", "Therefore", "Furthermore", "Moreover",
    "Also", "Then", "Now", "Today", "Yesterday",
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
    "November", "December", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Yes", "No", "True", "False", "None",
})


def extract_entity_mentions(text: str) -> list[str]:
    """
    Extract likely entity names from model output text.

    Uses capitalized phrase heuristic. MVP approach — future versions
    can use NER models or the model's own entity annotations.
    """
    matches = _ENTITY_PATTERN.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for match in matches:
        normalized = match.strip()
        if normalized in _STOPWORDS:
            continue
        if len(normalized) < 2:
            continue
        lower = normalized.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class EntityVerifier:
    """
    Post-response entity verification engine.

    Checks whether entities mentioned in model output are grounded
    in the primed context or discoverable via Memory Bus queries.
    """

    def __init__(
        self,
        *,
        neo4j: Neo4jAdapter,
        pg: PostgresAdapter,
    ):
        self._neo4j = neo4j
        self._pg = pg

    async def verify(
        self,
        response_text: str,
        primed_entity_ids: list[str] | None = None,
        primed_entity_names: list[str] | None = None,
        primed_items: list[MemoryItem] | None = None,
        loom_id: str | None = None,
        loop_id: str | None = None,
    ) -> VerificationResult:
        """
        Verify entity mentions in model response against memory.

        Grounding cascade:
        1. Match against primed entity IDs (from MemoryResult.items)
        2. Match against primed entity names (case-insensitive)
        3. Check Postgres aliases
        4. Check Neo4j name search

        Anything not grounded by steps 1-4 is flagged as ungrounded.
        """
        primed_ids = set(primed_entity_ids or [])
        primed_names_lower = {
            n.lower() for n in (primed_entity_names or [])
        }

        # Also extract names/IDs from primed items
        if primed_items:
            for item in primed_items:
                primed_ids.add(item.id)
                if item.title:
                    primed_names_lower.add(item.title.lower())
                for rel_id in item.related_entity_ids:
                    primed_ids.add(rel_id)

        # Extract entity mentions from response
        raw_mentions = extract_entity_mentions(response_text)

        mentions: list[EntityMention] = []

        for mention_text in raw_mentions:
            mention = EntityMention(text=mention_text)
            mention_lower = mention_text.lower()

            # Step 1: Check primed IDs (exact match on text is unlikely,
            # but entity names in primed items are)
            if mention_lower in primed_names_lower:
                mention.grounded = True
                mention.grounded_via = "primed_name"
                mentions.append(mention)
                continue

            # Step 2: Check Postgres aliases
            try:
                aliases = await self._pg.find_aliases(
                    mention_text, loom_id=loom_id
                )
                if aliases:
                    mention.grounded = True
                    mention.grounded_via = "alias"
                    mention.entity_id = aliases[0].entity_id
                    mentions.append(mention)
                    continue
            except Exception:
                pass

            # Step 3: Check Neo4j name search
            try:
                entities = await self._neo4j.search_entities_by_name(
                    mention_text, loom_origin=loom_id, limit=1
                )
                if entities:
                    entity = entities[0]
                    # Require reasonable match
                    if (
                        mention_lower == entity.name.lower()
                        or mention_lower in entity.name.lower()
                        or entity.name.lower() in mention_lower
                    ):
                        mention.grounded = True
                        mention.grounded_via = "neo4j_search"
                        mention.entity_id = entity.id
                        mentions.append(mention)
                        continue
            except Exception:
                pass

            # Not grounded
            mention.grounded = False
            mentions.append(mention)

        result = VerificationResult(
            mentions=mentions,
            response_text=response_text,
        )

        # Write audit row for verification
        await self._audit_verification(result, loom_id=loom_id, loop_id=loop_id)

        return result

    async def _audit_verification(
        self,
        result: VerificationResult,
        *,
        loom_id: str | None = None,
        loop_id: str | None = None,
    ) -> None:
        """Write verification result to audit log."""
        try:
            await self._pg.write_audit(
                AuditRow(
                    loop_id=loop_id,
                    loom_id=loom_id,
                    action=AuditAction.VERIFY,
                    details={
                        "total_mentions": result.total_mentions,
                        "grounded_count": result.grounded_count,
                        "ungrounded_count": result.ungrounded_count,
                        "confabulation_rate": round(
                            result.confabulation_rate, 4
                        ),
                        "ungrounded_names": [
                            m.text for m in result.ungrounded_mentions[:10]
                        ],
                        "grounded_names": [
                            m.text for m in result.grounded_mentions[:10]
                        ],
                    },
                )
            )
        except Exception as exc:
            logger.error("Failed to write verify audit: %s", exc)
