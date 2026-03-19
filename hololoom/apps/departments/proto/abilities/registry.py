"""Proto Ability Registry.

Central registry for discovering, managing, and executing Proto abilities.

Features:
- Register/unregister abilities at runtime
- Query abilities by name, tier, tag, or trust level
- Trust level validation
- Manifest introspection
- Lifecycle management

Usage:
    registry = AbilityRegistry()
    registry.register(my_ability)
    ability = registry.get("my_ability")
    result = await ability.execute(params, context)

Status: Production ready (2025-12-02)
"""

import logging

from .protocol import (
    Ability,
    AbilityManifest,
    AbilityTier,
    AbilityTrustLevel,
)


class AbilityRegistry:
    """Central registry for Proto abilities.

    Manages ability registration, discovery, and trust validation.

    Thread-safe for concurrent access (uses dict which is thread-safe for
    basic operations in CPython).

    Attributes:
        max_trust_level: Maximum trust level allowed for registration
        logger: Logger instance for registry events
    """

    def __init__(
        self,
        max_trust_level: AbilityTrustLevel = AbilityTrustLevel.LOCAL
    ):
        """Initialize ability registry.

        Args:
            max_trust_level: Maximum trust level for abilities to register
                (default: LOCAL - user-defined abilities)
        """
        self._abilities: dict[str, Ability] = {}
        self._max_trust_level = max_trust_level
        self.logger = logging.getLogger("proto.abilities.registry")

    def register(self, ability: Ability) -> bool:
        """Register an ability in the registry.

        Performs trust level validation before registration.

        Args:
            ability: Ability instance to register

        Returns:
            True if registered successfully, False if rejected
        """
        name = ability.manifest.name

        # Check if already registered
        if name in self._abilities:
            self.logger.warning(
                f"Ability '{name}' already registered, overwriting"
            )

        # Check trust level
        if not self._check_trust(ability):
            self.logger.warning(
                f"Ability '{name}' rejected: trust level {ability.manifest.trust_level.value} "
                f"exceeds maximum {self._max_trust_level.value}"
            )
            return False

        self._abilities[name] = ability
        self.logger.info(
            f"Registered ability: {name} v{ability.manifest.version} "
            f"(tier={ability.manifest.tier.name}, trust={ability.manifest.trust_level.value})"
        )
        return True

    def unregister(self, name: str) -> bool:
        """Unregister an ability by name.

        Args:
            name: Ability name

        Returns:
            True if unregistered, False if not found
        """
        if name in self._abilities:
            del self._abilities[name]
            self.logger.info(f"Unregistered ability: {name}")
            return True
        return False

    def get(self, name: str) -> Ability | None:
        """Get ability instance by name.

        Args:
            name: Ability name

        Returns:
            Ability instance or None if not found
        """
        return self._abilities.get(name)

    def has(self, name: str) -> bool:
        """Check if ability is registered.

        Args:
            name: Ability name

        Returns:
            True if ability exists, False otherwise
        """
        return name in self._abilities

    def list_all(self) -> list[str]:
        """List all registered ability names.

        Returns:
            List of ability names in registration order
        """
        return list(self._abilities.keys())

    def list_by_tier(self, tier: AbilityTier) -> list[str]:
        """List abilities filtered by tier.

        Args:
            tier: AbilityTier to filter by

        Returns:
            List of ability names matching tier
        """
        return [
            name for name, ability in self._abilities.items()
            if ability.manifest.tier == tier
        ]

    def list_by_trust(self, trust_level: AbilityTrustLevel) -> list[str]:
        """List abilities filtered by trust level.

        Args:
            trust_level: AbilityTrustLevel to filter by

        Returns:
            List of ability names matching trust level
        """
        return [
            name for name, ability in self._abilities.items()
            if ability.manifest.trust_level == trust_level
        ]

    def list_by_tag(self, tag: str) -> list[str]:
        """List abilities filtered by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of ability names with matching tag
        """
        return [
            name for name, ability in self._abilities.items()
            if ability.manifest.has_tag(tag)
        ]

    def list_by_permission(self, permission: str) -> list[str]:
        """List abilities that require a specific permission.

        Args:
            permission: Permission name (e.g., "read_file")

        Returns:
            List of ability names requiring permission
        """
        return [
            name for name, ability in self._abilities.items()
            if ability.manifest.requires_permission(permission)
        ]

    def get_manifest(self, name: str) -> AbilityManifest | None:
        """Get manifest for an ability.

        Args:
            name: Ability name

        Returns:
            AbilityManifest or None if not found
        """
        ability = self._abilities.get(name)
        return ability.manifest if ability else None

    def get_all_manifests(self) -> dict[str, AbilityManifest]:
        """Get manifests for all registered abilities.

        Returns:
            Dict mapping ability names to manifests
        """
        return {
            name: ability.manifest
            for name, ability in self._abilities.items()
        }

    def find_by_tags(self, tags: list[str], match_all: bool = False) -> list[str]:
        """Find abilities matching tags.

        Args:
            tags: List of tags to search for
            match_all: If True, ability must have all tags.
                      If False, ability must have any tag.

        Returns:
            List of matching ability names
        """
        results = []
        for name, ability in self._abilities.items():
            ability_tags = set(ability.manifest.tags)
            tag_set = set(tags)

            if match_all:
                if tag_set.issubset(ability_tags):
                    results.append(name)
            else:
                if tag_set & ability_tags:  # Intersection
                    results.append(name)

        return results

    def find_by_requirement(self, requirement: str) -> list[str]:
        """Find abilities that require a specific package/tool.

        Args:
            requirement: Requirement string (e.g., "numpy>=1.20")

        Returns:
            List of ability names with matching requirement
        """
        return [
            name for name, ability in self._abilities.items()
            if requirement in ability.manifest.requires
        ]

    def count(self) -> int:
        """Get number of registered abilities.

        Returns:
            Total count of abilities
        """
        return len(self._abilities)

    def count_by_tier(self, tier: AbilityTier) -> int:
        """Get count of abilities for a tier.

        Args:
            tier: AbilityTier to count

        Returns:
            Number of abilities in tier
        """
        return len(self.list_by_tier(tier))

    def count_by_trust(self, trust_level: AbilityTrustLevel) -> int:
        """Get count of abilities for a trust level.

        Args:
            trust_level: AbilityTrustLevel to count

        Returns:
            Number of abilities at trust level
        """
        return len(self.list_by_trust(trust_level))

    def clear(self) -> None:
        """Clear all registered abilities.

        Warning: This unregisters all abilities. Use with caution.
        """
        count = len(self._abilities)
        self._abilities.clear()
        self.logger.warning(f"Cleared {count} abilities from registry")

    def get_summary(self) -> dict[str, any]:
        """Get registry summary statistics.

        Returns:
            Dict with counts by tier, trust level, etc.
        """
        return {
            "total_count": self.count(),
            "by_tier": {
                tier.name: self.count_by_tier(tier)
                for tier in AbilityTier
            },
            "by_trust": {
                trust.name: self.count_by_trust(trust)
                for trust in AbilityTrustLevel
            },
        }

    def _check_trust(self, ability: Ability) -> bool:
        """Check if ability meets trust requirements.

        Uses a hierarchy: UNTRUSTED < COMMUNITY < LOCAL < VERIFIED < CORE

        Args:
            ability: Ability to check

        Returns:
            True if ability's trust level is acceptable, False otherwise
        """
        trust_hierarchy = [
            AbilityTrustLevel.UNTRUSTED,
            AbilityTrustLevel.COMMUNITY,
            AbilityTrustLevel.LOCAL,
            AbilityTrustLevel.VERIFIED,
            AbilityTrustLevel.CORE,
        ]

        ability_level = ability.manifest.trust_level
        max_level = self._max_trust_level

        # Find indices in hierarchy
        try:
            ability_idx = trust_hierarchy.index(ability_level)
            max_idx = trust_hierarchy.index(max_level)
        except ValueError:
            return False

        # Ability is acceptable if its level is >= max allowed level
        return ability_idx >= max_idx
