"""Proto Ability Protocol and Registry.

Three-tier extensibility for Proto code agent abilities.

Tier 1: Skill Mapping
    Thin wrappers around HoloLoom's built-in skills.
    Direct skill invocation with unified interface.

Tier 2: Plugin Protocol
    Structured interface with manifest and permissions.
    User-defined or verified marketplace abilities.

Tier 3: Full Sandbox (Future - Phase 4)
    Container/process isolation for untrusted code.

Core Classes:
    - Ability: Protocol that all abilities implement
    - BaseAbility: Base class for Tier 2 plugin abilities
    - AbilityRegistry: Central ability registry and discovery

Enums:
    - AbilityTier: Implementation tier (SKILL_MAPPING, PLUGIN, SANDBOX)
    - AbilityTrustLevel: Trust level (CORE, VERIFIED, COMMUNITY, LOCAL, UNTRUSTED)

Data Classes:
    - AbilityManifest: Ability metadata and capabilities
    - AbilityContext: Execution context with permissions
    - PreflightResult: Preflight check result
    - AbilityResult: Ability execution result
    - VerificationResult: Output verification result

Example:
    from hololoom.apps.departments.proto.abilities import (
        BaseAbility,
        AbilityManifest,
        AbilityRegistry,
        AbilityTier,
        AbilityTrustLevel,
        AbilityContext,
        AbilityResult,
    )

    # Create an ability
    class MyAbility(BaseAbility):
        def __init__(self):
            super().__init__(AbilityManifest(
                name="my_ability",
                version="1.0.0",
                description="Does something useful",
                author="me",
                tier=AbilityTier.PLUGIN,
                trust_level=AbilityTrustLevel.LOCAL,
                permissions=["read_file"],
                tags=["utility", "analysis"]
            ))

        async def execute(self, params, context):
            # Implementation
            return AbilityResult(
                success=True,
                output={"result": "data"},
                confidence=0.95
            )

    # Register and use
    registry = AbilityRegistry()
    ability = MyAbility()
    registry.register(ability)

    # Execute
    context = AbilityContext(session_id="123")
    preflight = await ability.preflight(context)
    if preflight.can_execute:
        result = await ability.execute({"param": "value"}, context)
        print(result.output)

Status: Production ready (2025-12-02)
"""

from .protocol import (
    # Protocol and Base Class
    Ability,
    AbilityContext,
    # Data Classes
    AbilityManifest,
    AbilityResult,
    # Enums
    AbilityTier,
    AbilityTrustLevel,
    BaseAbility,
    PreflightResult,
    VerificationResult,
)
from .registry import AbilityRegistry

__all__ = [
    # Enums
    "AbilityTier",
    "AbilityTrustLevel",

    # Data Classes
    "AbilityManifest",
    "AbilityContext",
    "PreflightResult",
    "AbilityResult",
    "VerificationResult",

    # Classes
    "Ability",
    "BaseAbility",
    "AbilityRegistry",
]
