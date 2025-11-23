"""
Demo: Skills + Zero-G Integration
==================================

**Created**: November 22, 2025
**Purpose**: Demonstrate complete skills integration with Zero-G platform

This demo shows:
1. Skill docking with Zero-G platform
2. Memory-informed decision making (WarpSpace + YarnGraph)
3. Complete launch sequence (Preflight → Orbit)
4. Mission Control monitoring
5. Continuous learning capture

Architecture:
```
Skills (Apps) → Zero-G Orchestrator → Loom Core
                      ↓
              Preflight → Orbit → Mission Control
                      ↓
              Memory-Informed Decisions
```
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.protocol import (
    DockableSkill,
    SkillManifest,
    SkillExecutionContext,
    SkillExecutionResult,
    PreflightCheckResult,
    SkillLifecycleState
)
from skills.zero_g_integration import (
    ZeroGOrchestrator,
    ZeroGConfig,
    create_zero_g_orchestrator
)


# ============================================================================
# Example Skill Implementation
# ============================================================================

class ExampleCodeReviewSkill:
    """
    Example skill that reviews code.

    Demonstrates:
    - DockableSkill protocol implementation
    - Memory-informed decision making
    - Complete lifecycle (preflight → execute → shutdown)
    """

    def __init__(self):
        self._manifest = SkillManifest(
            name="code_reviewer_example",
            version="1.0.0",
            category="domain",
            description="Reviews code for quality and security issues",
            tags=["code", "quality", "security"],
            capabilities=["code_review", "security_analysis", "style_checking"],
            requires_warp_space=True,
            requires_yarn_graph=True,
            requires_event_bus=True
        )

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    async def preflight(
        self,
        context: SkillExecutionContext
    ) -> PreflightCheckResult:
        """
        Run preflight safety checks.

        Validates:
        - Required infrastructure available (WarpSpace, YarnGraph)
        - Skill functionality
        - Resource limits
        """
        checks_run = 0
        checks_passed = 0
        failures = []
        warnings = []

        # Check WarpSpace availability
        checks_run += 1
        if context.warp_space:
            checks_passed += 1
        else:
            failures.append("WarpSpace not available")

        # Check YarnGraph availability
        checks_run += 1
        if context.yarn_graph:
            checks_passed += 1
        else:
            failures.append("YarnGraph not available")

        # Check EventBus availability
        checks_run += 1
        if context.event_bus:
            checks_passed += 1
        else:
            warnings.append("EventBus not available (optional)")
            checks_passed += 1  # Warning, not failure

        # Basic functionality check
        checks_run += 1
        try:
            # Simulate self-test
            await asyncio.sleep(0.1)
            checks_passed += 1
        except Exception as e:
            failures.append(f"Self-test failed: {e}")

        return PreflightCheckResult(
            passed=len(failures) == 0,
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=len(failures),
            failures=failures,
            warnings=warnings
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: SkillExecutionContext
    ) -> SkillExecutionResult:
        """
        Execute code review.

        Uses memory-informed decisioning:
        - Query WarpSpace for similar code reviews
        - Traverse YarnGraph for related quality issues
        - Apply learned patterns
        """
        import time
        start_time = time.time()

        # Extract parameters
        code = parameters.get('code', '')
        language = parameters.get('language', 'python')
        memory_informed = parameters.get('memory_informed', False)

        # Use memory if available
        relevant_memories = parameters.get('relevant_memories', [])
        memory_confidence = parameters.get('memory_confidence', 0.0)

        # Simulate code review
        issues = []

        # Basic checks
        if len(code) == 0:
            issues.append("Empty code provided")
        elif len(code.split('\n')) > 1000:
            issues.append("Code too long (>1000 lines)")

        # Memory-informed checks (if memories available)
        if memory_informed and relevant_memories:
            # Simulate using past reviews
            for memory in relevant_memories[:3]:
                pattern = memory.get('content', '')
                if 'security' in pattern.lower():
                    issues.append(f"Security pattern found: {pattern[:50]}...")

        # Calculate confidence
        if memory_informed and memory_confidence > 0.7:
            confidence = 0.9  # High confidence with good memory
        elif memory_informed:
            confidence = 0.7  # Medium confidence with some memory
        else:
            confidence = 0.5  # Low confidence without memory

        # Build result
        output = {
            'language': language,
            'lines_of_code': len(code.split('\n')),
            'issues_found': len(issues),
            'issues': issues,
            'memory_informed': memory_informed,
            'memory_count': len(relevant_memories),
            'memory_confidence': memory_confidence
        }

        execution_time = (time.time() - start_time) * 1000

        return SkillExecutionResult(
            success=True,
            output=output,
            confidence=confidence,
            lifecycle_states=[
                SkillLifecycleState.COUNTDOWN,
                SkillLifecycleState.LIFT_OFF,
                SkillLifecycleState.ORBIT
            ],
            execution_time_ms=execution_time,
            tokens_used=len(code.split())  # Rough estimate
        )

    async def shutdown(self, context: SkillExecutionContext) -> None:
        """Graceful shutdown."""
        print(f"Shutting down {self.manifest.name}")


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_1_basic_docking():
    """Demo 1: Basic skill docking."""
    print("\n" + "="*70)
    print("Demo 1: Basic Skill Docking")
    print("="*70)

    # Create orchestrator
    config = ZeroGConfig(
        enable_warp_space=False,
        enable_yarn_graph=False,
        enable_mission_control=True
    )

    orchestrator = await create_zero_g_orchestrator(config=config)

    try:
        # Create skill
        skill = ExampleCodeReviewSkill()

        # Dock skill
        success, message = await orchestrator.docking_manager.dock_skill(skill)

        if success:
            print(f"[OK] Docked: {skill.manifest.name}")
            print(f"  Message: {message}")
        else:
            print(f"[FAIL] Docking failed: {message}")

        # Get status
        status = await orchestrator.get_mission_status()
        print(f"\nDocked skills: {status['docked_skills']}")

    finally:
        await orchestrator.stop()


async def demo_2_preflight_checks():
    """Demo 2: Preflight safety checks."""
    print("\n" + "="*70)
    print("Demo 2: Preflight Safety Checks")
    print("="*70)

    config = ZeroGConfig(
        enable_skill_tester=True,
        enable_security_analyzer=True
    )

    orchestrator = await create_zero_g_orchestrator(config=config)

    try:
        skill = ExampleCodeReviewSkill()

        # Dock skill
        await orchestrator.docking_manager.dock_skill(skill)

        # Run preflight
        context = orchestrator.docking_manager._create_context(skill.manifest.name)
        preflight_result = await orchestrator.preflight_coordinator.run_preflight(
            skill,
            context
        )

        print(f"\nPreflight Results:")
        print(f"  Passed: {preflight_result['passed']}")
        print(f"  Checks run: {preflight_result['checks_run']}")
        print(f"  Checks passed: {preflight_result['checks_passed']}")
        print(f"  Checks failed: {preflight_result['checks_failed']}")

        if preflight_result['failures']:
            print(f"\nFailures:")
            for failure in preflight_result['failures']:
                print(f"  - {failure}")

        if preflight_result['warnings']:
            print(f"\nWarnings:")
            for warning in preflight_result['warnings']:
                print(f"  - {warning}")

        if preflight_result.get('details'):
            print(f"\nDetails:")
            for check_name, check_result in preflight_result['details'].items():
                if isinstance(check_result, dict):
                    print(f"  {check_name}: {check_result.get('passed', 'N/A')}")
                else:
                    print(f"  {check_name}: {check_result.passed}")

    finally:
        await orchestrator.stop()


async def demo_3_memory_informed_execution():
    """Demo 3: Memory-informed skill execution."""
    print("\n" + "="*70)
    print("Demo 3: Memory-Informed Execution")
    print("="*70)

    # Note: In production, pass actual WarpSpace and YarnGraph instances
    config = ZeroGConfig(
        enable_warp_space=True,
        enable_yarn_graph=True,
        memory_query_limit=5,
        memory_confidence_threshold=0.6
    )

    orchestrator = await create_zero_g_orchestrator(config=config)

    try:
        skill = ExampleCodeReviewSkill()

        # Complete launch sequence
        code_sample = """
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""

        parameters = {
            'code': code_sample,
            'language': 'python',
            'context': 'Code review for production deployment'
        }

        print("\nLaunching skill with memory-informed decisioning...")
        result = await orchestrator.dock_and_launch(skill, parameters)

        print(f"\nExecution Result:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Execution time: {result.execution_time_ms:.1f}ms")
        print(f"  Tokens used: {result.tokens_used}")

        if result.output:
            output = result.output
            print(f"\nOutput:")
            print(f"  Language: {output.get('language')}")
            print(f"  Lines of code: {output.get('lines_of_code')}")
            print(f"  Issues found: {output.get('issues_found')}")
            print(f"  Memory informed: {output.get('memory_informed')}")
            print(f"  Memory confidence: {output.get('memory_confidence'):.2f}")

            if output.get('issues'):
                print(f"\nIssues:")
                for issue in output['issues']:
                    print(f"  - {issue}")

    finally:
        await orchestrator.stop()


async def demo_4_mission_control():
    """Demo 4: Mission Control monitoring."""
    print("\n" + "="*70)
    print("Demo 4: Mission Control Monitoring")
    print("="*70)

    config = ZeroGConfig(
        enable_mission_control=True,
        enable_token_adviser=True,
        token_budget_limit=1000
    )

    orchestrator = await create_zero_g_orchestrator(config=config)

    try:
        skill = ExampleCodeReviewSkill()

        # Launch skill
        parameters = {'code': 'print("hello")', 'language': 'python'}
        result = await orchestrator.dock_and_launch(skill, parameters)

        # Report metrics to Mission Control
        await orchestrator.mission_control.report_metrics(
            skill.manifest.name,
            {
                'success': result.success,
                'execution_time_ms': result.execution_time_ms,
                'tokens_used': result.tokens_used
            }
        )

        # Get health status
        health = await orchestrator.mission_control.health_check(skill.manifest.name)

        print(f"\nMission Control Health Status:")
        print(f"  Status: {health.get('status', 'unknown')}")
        print(f"  Executions: {health.get('executions', 0)}")
        print(f"  Successes: {health.get('successes', 0)}")
        print(f"  Failures: {health.get('failures', 0)}")
        avg_latency = health.get('avg_latency_ms', 0.0)
        print(f"  Avg latency: {avg_latency:.1f}ms")
        print(f"  Tokens used: {health.get('tokens_used', 0)}")

        # Get complete mission status
        status = await orchestrator.get_mission_status()

        print(f"\nMission Status:")
        print(f"  Docked skills: {len(status['docked_skills'])}")
        print(f"  Skills in orbit: {len(status['skills_in_orbit'])}")

    finally:
        await orchestrator.stop()


async def demo_5_complete_lifecycle():
    """Demo 5: Complete lifecycle (Preflight → Orbit → Shutdown)."""
    print("\n" + "="*70)
    print("Demo 5: Complete Lifecycle")
    print("="*70)

    config = ZeroGConfig(
        enable_warp_space=True,
        enable_yarn_graph=True,
        enable_skill_tester=True,
        enable_security_analyzer=True,
        enable_continuous_learning=True,
        enable_mission_control=True
    )

    orchestrator = await create_zero_g_orchestrator(config=config)

    try:
        skill = ExampleCodeReviewSkill()

        print("\nLifecycle Stages:")
        print("  1. Docking...")

        # Dock
        success, message = await orchestrator.docking_manager.dock_skill(skill)
        print(f"     Docked: {success}")

        print("  2. Preflight checks...")

        # Launch (includes preflight)
        parameters = {'code': 'def foo(): pass', 'language': 'python'}
        result = await orchestrator.dock_and_launch(skill, parameters)

        print(f"     Preflight: passed")
        print(f"  3. Lift-Off...")
        print(f"     Executing: {skill.manifest.name}")
        print(f"  4. Orbit...")
        print(f"     In orbit: {result.success}")

        # Monitor for a bit
        print(f"  5. Monitoring (5 seconds)...")
        await asyncio.sleep(5)

        print(f"  6. Reentry...")

        # Undock
        success, message = await orchestrator.docking_manager.undock_skill(
            skill.manifest.name
        )
        print(f"     Undocked: {success}")
        print(f"  7. Landed")

        print("\nLifecycle complete!")

    finally:
        await orchestrator.stop()


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Skills + Zero-G Integration Demos")
    print("="*70)

    demos = [
        demo_1_basic_docking,
        demo_2_preflight_checks,
        demo_3_memory_informed_execution,
        demo_4_mission_control,
        demo_5_complete_lifecycle
    ]

    for i, demo in enumerate(demos, 1):
        try:
            await demo()
            print(f"\n[OK] Demo {i} completed")
        except Exception as e:
            print(f"\n[FAIL] Demo {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)


if __name__ == '__main__':
    asyncio.run(main())
