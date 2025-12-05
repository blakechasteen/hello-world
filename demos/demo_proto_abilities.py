"""Proto Ability Protocol & Registry Demonstration.

Showcases the three-tier ability extensibility architecture:
  - Tier 1: Skill Mapping (HoloLoom skill wrappers)
  - Tier 2: Plugin Protocol (user-defined abilities)
  - Tier 3: Sandbox (untrusted code execution - Phase 4)

Run with: PYTHONPATH=. python demos/demo_proto_abilities.py
"""

import asyncio
from HoloLoom.departments.proto.abilities import (
    AbilityTier,
    AbilityTrustLevel,
    AbilityManifest,
    AbilityContext,
    AbilityResult,
    PreflightResult,
    VerificationResult,
    BaseAbility,
    AbilityRegistry,
)


# Tier 1: Skill Mapping (thin wrapper around HoloLoom skill)
class SkillMappingAbility(BaseAbility):
    """Tier 1: Skill Mapping - Wrapper around HoloLoom's built-in skills."""

    def __init__(self):
        super().__init__(AbilityManifest(
            name='memory_skill_wrapper',
            version='1.0.0',
            description='Memory skill wrapper from HoloLoom',
            author='HoloLoom',
            tier=AbilityTier.SKILL_MAPPING,
            trust_level=AbilityTrustLevel.CORE,
            tags=['core', 'memory']
        ))

    async def execute(self, params, context):
        # Direct wrapper around HoloLoom skill
        return AbilityResult(
            success=True,
            output={'skill': 'memory', 'action': 'recall', 'items': 5},
            confidence=1.0
        )


# Tier 2: Plugin Protocol (structured interface with manifest)
class CodeAnalyzerAbility(BaseAbility):
    """Tier 2: Plugin Protocol - User-defined ability with manifest."""

    def __init__(self):
        super().__init__(AbilityManifest(
            name='code_analyzer',
            version='2.1.0',
            description='Analyzes Python code for issues',
            author='user_team',
            tier=AbilityTier.PLUGIN,
            trust_level=AbilityTrustLevel.VERIFIED,
            permissions=['read_file', 'write_file'],
            requires=['ast', 'flake8'],
            tags=['analysis', 'code', 'python'],
            requires_confirmation=True
        ))

    async def preflight(self, context):
        if context.trust_level == AbilityTrustLevel.UNTRUSTED:
            return PreflightResult(
                can_execute=False,
                reason='Untrusted context cannot execute this ability'
            )
        if not context.user_confirmed:
            return PreflightResult(
                can_execute=False,
                reason='User confirmation required',
                warnings=['This ability requires read/write file access']
            )
        return PreflightResult(can_execute=True)

    async def execute(self, params, context):
        code = params.get('code', '')
        if not code:
            return AbilityResult(success=False, error='No code provided')

        issues = []
        if 'import *' in code:
            issues.append('Found wildcard import')
        if 'eval(' in code:
            issues.append('Found eval() usage')

        return AbilityResult(
            success=True,
            output={'code_length': len(code), 'issues_found': len(issues), 'issues': issues},
            confidence=0.88
        )

    async def verify(self, result):
        if not result.success:
            return VerificationResult(verified=False, issues=['Execution failed'])

        issues = []
        if not isinstance(result.output, dict):
            issues.append('Output is not a dictionary')
        if 'issues' not in result.output:
            issues.append('Missing "issues" field in output')

        if issues:
            return VerificationResult(
                verified=False,
                issues=issues,
                suggestions=['Ensure output includes all required fields']
            )
        return VerificationResult(verified=True)


# Tier 3: Sandbox (placeholder - Phase 4 implementation)
class SandboxAbility(BaseAbility):
    """Tier 3: Sandbox - For untrusted code execution."""

    def __init__(self):
        super().__init__(AbilityManifest(
            name='untrusted_script_runner',
            version='1.0.0',
            description='Runs untrusted scripts in sandboxed environment',
            author='security_team',
            tier=AbilityTier.SANDBOX,
            trust_level=AbilityTrustLevel.UNTRUSTED,
            permissions=['execute', 'network'],
            tags=['sandbox', 'security', 'experimental']
        ))

    async def execute(self, params, context):
        # Phase 4: Would run in Docker container with resource limits
        return AbilityResult(
            success=True,
            output={'status': 'Would run in sandboxed environment'},
            confidence=0.5  # Lower confidence for untested feature
        )


async def main():
    print('=== Proto Ability Protocol: Three-Tier Demonstration ===\n')

    # Create registry
    registry = AbilityRegistry(max_trust_level=AbilityTrustLevel.VERIFIED)

    # Register abilities of different tiers
    abilities = [
        SkillMappingAbility(),
        CodeAnalyzerAbility(),
        SandboxAbility(),
    ]

    print('[Tier 1] SKILL_MAPPING: Thin wrappers around HoloLoom skills')
    for ability in abilities:
        if ability.manifest.tier == AbilityTier.SKILL_MAPPING:
            registry.register(ability)
            print('  - {} (trust: {}, core built-in)'.format(
                ability.name, ability.manifest.trust_level.value))

    print()
    print('[Tier 2] PLUGIN: User-defined abilities with manifests')
    for ability in abilities:
        if ability.manifest.tier == AbilityTier.PLUGIN:
            registered = registry.register(ability)
            status = 'registered' if registered else 'rejected'
            print('  - {} (trust: {}, {})'.format(
                ability.name, ability.manifest.trust_level.value, status))

    print()
    print('[Tier 3] SANDBOX: Untrusted code execution (Phase 4)')
    for ability in abilities:
        if ability.manifest.tier == AbilityTier.SANDBOX:
            registered = registry.register(ability)
            status = 'registered' if registered else 'rejected (trust too low)'
            print('  - {} (trust: {}, {})'.format(
                ability.name, ability.manifest.trust_level.value, status))

    print()
    print('=== Registry Summary ===')
    summary = registry.get_summary()
    print('Total registered: {}'.format(summary['total_count']))
    print('By tier:')
    for tier, count in summary['by_tier'].items():
        print('  - {}: {}'.format(tier, count))

    print()
    print('=== Ability Execution Example ===')

    # Execute Tier 2 ability
    analyzer = registry.get('code_analyzer')
    if analyzer:
        context = AbilityContext(
            session_id='demo_123',
            user_confirmed=True,
            trust_level=AbilityTrustLevel.VERIFIED
        )

        test_code = '''
def process_data(data):
    from math import *
    result = eval(input())
    return result
'''

        # Preflight check
        preflight = await analyzer.preflight(context)
        print('Preflight: {}'.format('OK' if preflight.can_execute else 'FAILED'))

        # Execute
        result = await analyzer.execute({'code': test_code}, context)
        print('Execution: {}'.format('SUCCESS' if result.success else 'FAILED'))
        print('Output: {} issues found'.format(result.output['issues_found']))
        for issue in result.output['issues']:
            print('  - {}'.format(issue))
        print('Confidence: {:.0f}%'.format(result.confidence * 100))

        # Verify
        verification = await analyzer.verify(result)
        print('Verification: {}'.format('PASSED' if verification.verified else 'FAILED'))

    print()
    print('=== Discovery & Query Examples ===')
    print('All abilities: {}'.format(registry.list_all()))
    print('Code analysis abilities: {}'.format(registry.list_by_tag('code')))
    print('File-access abilities: {}'.format(registry.list_by_permission('read_file')))

    print()
    print('[SUCCESS] All demonstrations completed')


if __name__ == '__main__':
    asyncio.run(main())
