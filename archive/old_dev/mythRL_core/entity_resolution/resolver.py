"""
Entity Resolver - Domain-Agnostic Canonical Entity Mapping
===========================================================

Maps aliases and mentions to canonical entity IDs across any domain.

Examples:
    Beekeeping:
        "jodi" → "hive-003"
        "dennis's hive" → "hive-002"
        "thymol" → "treatment-thymol"

    Gardening:
        "tomatoes" → "plant-tomato-001"
        "raised bed 1" → "bed-001"

    Finance:
        "emergency fund" → "account-savings-emergency"
"""

from typing import Dict, List, Optional, Set, Any
import re
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Entity:
    """Represents a canonical entity with aliases and metadata."""

    canonical_id: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    entity_type: str = "unknown"
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if text matches this entity (canonical name or any alias)."""
        if not case_sensitive:
            text = text.lower()

        # Check canonical name
        check_name = self.canonical_name if case_sensitive else self.canonical_name.lower()
        if check_name in text:
            return True

        # Check aliases
        for alias in self.aliases:
            check_alias = alias if case_sensitive else alias.lower()
            if check_alias in text:
                return True

        return False

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        """Deserialize from dictionary."""
        return cls(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            aliases=data.get("aliases", []),
            entity_type=data.get("entity_type", "unknown"),
            attributes=data.get("attributes", {}),
            metadata=data.get("metadata", {})
        )


class EntityRegistry:
    """
    Registry of canonical entities for a domain.

    Manages entity definitions, provides resolution, and handles persistence.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.entities: Dict[str, Entity] = {}
        self._alias_index: Dict[str, str] = {}  # alias → canonical_id
        self.measurement_patterns: Dict[str, Any] = {}  # Store custom measurement patterns

    def register(self, entity: Entity):
        """Register a new entity in the registry."""
        # Store entity
        self.entities[entity.canonical_id] = entity

        # Index canonical name
        self._alias_index[entity.canonical_name.lower()] = entity.canonical_id

        # Index all aliases
        for alias in entity.aliases:
            self._alias_index[alias.lower()] = entity.canonical_id

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by canonical ID."""
        return self.entities.get(entity_id)

    def resolve_alias(self, text: str) -> Optional[str]:
        """Resolve alias/mention to canonical ID (fast lookup)."""
        return self._alias_index.get(text.lower())

    def find_entities(self, text: str, case_sensitive: bool = False) -> List[Entity]:
        """Find all entities mentioned in text."""
        found = []

        for entity in self.entities.values():
            if entity.matches(text, case_sensitive):
                found.append(entity)

        return found

    def get_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def save(self, filepath: Path):
        """Save registry to JSON file."""
        data = {
            "domain": self.domain,
            "entities": [e.to_dict() for e in self.entities.values()]
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'EntityRegistry':
        """Load registry from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        registry = cls(domain=data["domain"])

        for entity_data in data["entities"]:
            entity = Entity.from_dict(entity_data)
            registry.register(entity)

        # Load measurement patterns if present
        if "measurement_patterns" in data:
            registry.measurement_patterns = data["measurement_patterns"]

        return registry

    def stats(self) -> Dict:
        """Get registry statistics."""
        return {
            "domain": self.domain,
            "total_entities": len(self.entities),
            "total_aliases": len(self._alias_index),
            "entities_by_type": {
                entity_type: len(entities)
                for entity_type, entities in self._group_by_type().items()
            }
        }

    def _group_by_type(self) -> Dict[str, List[Entity]]:
        """Group entities by type."""
        grouped = {}
        for entity in self.entities.values():
            if entity.entity_type not in grouped:
                grouped[entity.entity_type] = []
            grouped[entity.entity_type].append(entity)
        return grouped


class EntityResolver:
    """
    Domain-agnostic entity resolver.

    Resolves mentions, aliases, and references to canonical entity IDs.
    Supports multiple resolution strategies and confidence scoring.
    """

    def __init__(self, registry: EntityRegistry):
        self.registry = registry

    def resolve(
        self,
        text: str,
        entity_type: Optional[str] = None,
        strategy: str = "greedy"
    ) -> List[Dict[str, Any]]:
        """
        Resolve entities in text to canonical IDs.

        Args:
            text: Text to search for entity mentions
            entity_type: Optional filter by entity type
            strategy: Resolution strategy ("greedy", "longest", "all")

        Returns:
            List of resolution results with confidence scores
        """
        if strategy == "greedy":
            return self._resolve_greedy(text, entity_type)
        elif strategy == "longest":
            return self._resolve_longest(text, entity_type)
        elif strategy == "all":
            return self._resolve_all(text, entity_type)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _resolve_greedy(
        self,
        text: str,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Greedy resolution: First match wins.
        Fast but may miss overlapping entities.
        """
        results = []
        text_lower = text.lower()

        # Try exact alias lookup first (fastest)
        for word in text_lower.split():
            canonical_id = self.registry.resolve_alias(word)
            if canonical_id:
                entity = self.registry.get(canonical_id)
                if entity_type is None or entity.entity_type == entity_type:
                    results.append({
                        "canonical_id": canonical_id,
                        "entity": entity,
                        "matched_text": word,
                        "confidence": 1.0,
                        "method": "exact_alias"
                    })

        # Then try substring matching
        entities = self.registry.entities.values()
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]

        for entity in entities:
            # Skip if already found via alias
            if any(r["canonical_id"] == entity.canonical_id for r in results):
                continue

            if entity.matches(text, case_sensitive=False):
                # Find what matched
                matched = None
                if entity.canonical_name.lower() in text_lower:
                    matched = entity.canonical_name
                else:
                    for alias in entity.aliases:
                        if alias.lower() in text_lower:
                            matched = alias
                            break

                results.append({
                    "canonical_id": entity.canonical_id,
                    "entity": entity,
                    "matched_text": matched,
                    "confidence": 0.8,  # Lower confidence for substring match
                    "method": "substring"
                })

        return results

    def _resolve_longest(
        self,
        text: str,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Longest match resolution: Prefer longer entity names.
        Better for nested entities (e.g., "Jodi's Hive" vs "Jodi").
        """
        candidates = self._resolve_all(text, entity_type)

        # Sort by matched text length (descending)
        candidates.sort(key=lambda x: len(x["matched_text"] or ""), reverse=True)

        # Remove overlapping matches (keep longest)
        results = []
        used_positions = set()

        for candidate in candidates:
            matched_text = candidate["matched_text"]
            if not matched_text:
                continue

            # Find position in text
            start = text.lower().find(matched_text.lower())
            if start == -1:
                continue

            end = start + len(matched_text)
            positions = range(start, end)

            # Check if overlaps with already used positions
            if not any(pos in used_positions for pos in positions):
                results.append(candidate)
                used_positions.update(positions)

        return results

    def _resolve_all(
        self,
        text: str,
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Resolve all possible entities (may include overlaps).
        Most comprehensive but may have duplicates.
        """
        results = []

        entities = self.registry.entities.values()
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]

        for entity in entities:
            if entity.matches(text, case_sensitive=False):
                # Find what matched
                matched = None
                confidence = 0.8

                if entity.canonical_name.lower() in text.lower():
                    matched = entity.canonical_name
                    confidence = 1.0
                else:
                    for alias in entity.aliases:
                        if alias.lower() in text.lower():
                            matched = alias
                            confidence = 0.9
                            break

                results.append({
                    "canonical_id": entity.canonical_id,
                    "entity": entity,
                    "matched_text": matched,
                    "confidence": confidence,
                    "method": "all"
                })

        return results

    def resolve_to_ids(
        self,
        text: str,
        entity_type: Optional[str] = None,
        strategy: str = "longest"
    ) -> List[str]:
        """
        Convenience method: Resolve to just canonical IDs.

        Returns:
            List of canonical entity IDs found in text
        """
        results = self.resolve(text, entity_type, strategy)
        return [r["canonical_id"] for r in results]

    def tag_text(
        self,
        text: str,
        entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tag text with all resolved entities and return structured output.

        Returns:
            Dictionary with original text, entities, and metadata
        """
        resolutions = self.resolve(text, entity_type, strategy="longest")

        return {
            "text": text,
            "entities": [
                {
                    "canonical_id": r["canonical_id"],
                    "canonical_name": r["entity"].canonical_name,
                    "entity_type": r["entity"].entity_type,
                    "matched_text": r["matched_text"],
                    "confidence": r["confidence"]
                }
                for r in resolutions
            ],
            "entity_ids": [r["canonical_id"] for r in resolutions],
            "entity_count": len(resolutions)
        }


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Create a registry for beekeeping domain
    registry = EntityRegistry(domain="beekeeping")

    # Register some hives
    registry.register(Entity(
        canonical_id="hive-001",
        canonical_name="Half Door Hive",
        aliases=["half door", "hive 1", "first hive"],
        entity_type="hive",
        attributes={"strength": 5.5, "temperament": "calm"}
    ))

    registry.register(Entity(
        canonical_id="hive-003",
        canonical_name="Jodi's Hive",
        aliases=["jodi", "jodis", "jodi hive"],
        entity_type="hive",
        attributes={"strength": 8.0, "temperament": "calm"}
    ))

    # Create resolver
    resolver = EntityResolver(registry)

    # Test resolution
    text = "Checked jodi today, she's doing great! The half door hive also looks good."

    results = resolver.tag_text(text)

    print("="*80)
    print("Entity Resolution Demo")
    print("="*80)
    print(f"\nOriginal text: {text}")
    print(f"\nFound {results['entity_count']} entities:")

    for entity in results['entities']:
        print(f"\n  {entity['matched_text']} -> {entity['canonical_name']}")
        print(f"    ID: {entity['canonical_id']}")
        print(f"    Type: {entity['entity_type']}")
        print(f"    Confidence: {entity['confidence']}")

    print(f"\nCanonical IDs: {results['entity_ids']}")

    # Show registry stats
    print(f"\n{'='*80}")
    print("Registry Stats:")
    stats = registry.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
