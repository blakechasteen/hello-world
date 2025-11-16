"""
SOAR Playbook Demo

Demonstrates automated incident response with pre-built playbooks.

**Implemented: 2025-11-16**

Usage:
    PYTHONPATH=. python demos/demo_soar_playbooks.py
"""

import asyncio
import logging
from pathlib import Path

from HoloLoom.security.soar import (
    create_soar_engine,
    SecurityEvent,
    ExecutionMode,
    PlaybookSeverity,
    register_all_playbooks,
    get_playbook_catalog,
)

from HoloLoom.security.soar.testing import (
    run_all_playbook_tests,
    create_test_event,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_basic_playbook_execution():
    """Demo 1: Basic playbook execution"""
    print("=" * 70)
    print("Demo 1: Basic Playbook Execution")
    print("=" * 70)

    # Create SOAR engine in dry-run mode
    async with create_soar_engine(mode=ExecutionMode.DRY_RUN) as soar:
        # Register all playbooks
        playbook_names = register_all_playbooks(soar)
        print(f"\n✅ Registered {len(playbook_names)} playbooks:")
        for name in playbook_names:
            print(f"   - {name}")

        # Create SQL injection event
        print("\n🔍 Simulating SQL injection attack...")
        event = SecurityEvent(
            event_type="security.attack.sql_injection",
            severity=PlaybookSeverity.CRITICAL,
            source_ip="192.168.1.100",
            user_id="attacker",
            payload="' OR '1'='1",
        )

        # Process event
        results = await soar.process_event(event)

        # Display results
        print(f"\n📊 Playbook Results:")
        for result in results:
            print(f"\n   Playbook: {result.playbook_name}")
            print(f"   Status: {result.status.value}")
            print(f"   Duration: {result.duration_ms:.0f}ms")
            print(f"   Actions Taken ({len(result.actions_taken)}):")
            for action in result.actions_taken:
                print(f"      ✓ {action}")

            if result.incident_id:
                print(f"   Incident ID: {result.incident_id}")

    print("\n" + "=" * 70 + "\n")


async def demo_multiple_event_types():
    """Demo 2: Multiple event types"""
    print("=" * 70)
    print("Demo 2: Multiple Event Types")
    print("=" * 70)

    async with create_soar_engine(mode=ExecutionMode.DRY_RUN) as soar:
        register_all_playbooks(soar)

        # Test different attack types
        test_events = [
            {
                "name": "SQL Injection",
                "event": SecurityEvent(
                    event_type="security.attack.sql_injection",
                    source_ip="1.2.3.4",
                    payload="DROP TABLE users",
                ),
            },
            {
                "name": "Brute Force",
                "event": SecurityEvent(
                    event_type="security.attack.brute_force",
                    source_ip="5.6.7.8",
                    user_id="target_user",
                    metadata={"attempt_count": 50},
                ),
            },
            {
                "name": "DDoS Attack",
                "event": SecurityEvent(
                    event_type="security.attack.ddos",
                    metadata={
                        "is_distributed": True,
                        "source_count": 500,
                        "requests_per_second": 20000,
                    },
                ),
            },
        ]

        for test in test_events:
            print(f"\n🚨 {test['name']} Detected!")
            results = await soar.process_event(test["event"])

            for result in results:
                print(f"   Playbook: {result.playbook_name}")
                print(f"   Actions: {', '.join(result.actions_taken[:3])}...")

        # Display statistics
        print(f"\n📈 SOAR Statistics:")
        stats = soar.get_statistics()
        print(f"   Total Executions: {stats['total_executions']}")
        print(f"   Successful: {stats['successful_executions']}")
        print(f"   Dry-Run: {stats['dry_run_executions']}")

    print("\n" + "=" * 70 + "\n")


async def demo_production_vs_dry_run():
    """Demo 3: Production vs Dry-Run Mode"""
    print("=" * 70)
    print("Demo 3: Production vs Dry-Run Mode")
    print("=" * 70)

    event = SecurityEvent(
        event_type="security.attack.sql_injection",
        source_ip="10.0.0.1",
    )

    # Dry-run mode
    print("\n🧪 DRY-RUN MODE (safe testing)")
    async with create_soar_engine(mode=ExecutionMode.DRY_RUN) as soar:
        register_all_playbooks(soar)
        results = await soar.process_event(event)
        print(f"   Mode: DRY_RUN")
        print(f"   Actions: Simulated (no real changes)")
        print(f"   Playbook: {results[0].playbook_name}")

    # Production mode (still safe in this demo, no actual backends)
    print("\n🚀 PRODUCTION MODE (real execution)")
    async with create_soar_engine(mode=ExecutionMode.PRODUCTION) as soar:
        register_all_playbooks(soar)
        results = await soar.process_event(event)
        print(f"   Mode: PRODUCTION")
        print(f"   Actions: Would execute for real")
        print(f"   Playbook: {results[0].playbook_name}")

    print("\n" + "=" * 70 + "\n")


async def demo_playbook_catalog():
    """Demo 4: Playbook catalog"""
    print("=" * 70)
    print("Demo 4: Playbook Catalog")
    print("=" * 70)

    catalog = get_playbook_catalog()

    print(f"\n📚 Available Playbooks ({len(catalog)}):\n")

    for playbook in catalog:
        print(f"   {playbook['name']}")
        print(f"   Description: {playbook['description']}")
        print(f"   Severity: {playbook['severity'].upper()}")
        print(f"   Triggers: {', '.join(playbook['triggers'])}")
        print(f"   Requires Approval: {'Yes' if playbook['requires_approval'] else 'No'}")
        print(f"   Timeout: {playbook['timeout_seconds']}s")
        print(f"   Tags: {', '.join(playbook['tags'])}")
        print()

    print("=" * 70 + "\n")


async def demo_data_breach_response():
    """Demo 5: Complete data breach response"""
    print("=" * 70)
    print("Demo 5: Data Breach Response (High Severity)")
    print("=" * 70)

    async with create_soar_engine(mode=ExecutionMode.DRY_RUN) as soar:
        register_all_playbooks(soar)

        # Simulate data breach
        print("\n🚨 DATA BREACH DETECTED")
        print("\n   Breach Details:")
        print("   - Data Type: PII (Personal Identifiable Information)")
        print("   - Affected Users: 150 users")
        print("   - Affected Resources: 3 databases")
        print("   - Source IP: 203.0.113.42 (attacker)")

        event = SecurityEvent(
            event_type="security.incident.breach_detected",
            severity=PlaybookSeverity.CRITICAL,
            source_ip="203.0.113.42",
            metadata={
                "affected_users": [f"user{i}" for i in range(150)],
                "affected_resources": [
                    {"id": "db_users", "type": "database"},
                    {"id": "db_transactions", "type": "database"},
                    {"id": "logs_2024", "type": "file"},
                ],
                "data_type": "PII",
            },
        )

        # Process breach
        results = await soar.process_event(event)

        print(f"\n📋 Automated Response:")
        for result in results:
            print(f"\n   Playbook: {result.playbook_name}")
            print(f"   Status: {result.status.value}")
            print(f"   Incident ID: {result.incident_id}")

            print(f"\n   ✅ Actions Executed ({len(result.actions_taken)}):")
            for i, action in enumerate(result.actions_taken, 1):
                print(f"      {i}. {action}")

            print(f"\n   ⚠️  Manual Steps Required:")
            print("      1. Legal team: Review breach notification requirements")
            print("      2. Security team: Full forensics investigation")
            print("      3. Executive team: Customer communication planning")
            print("      4. PR team: Media response preparation")

    print("\n" + "=" * 70 + "\n")


async def demo_testing_framework():
    """Demo 6: Playbook testing framework"""
    print("=" * 70)
    print("Demo 6: Playbook Testing Framework")
    print("=" * 70)

    print("\n🧪 Running comprehensive playbook tests...\n")

    # Run all built-in tests
    results = await run_all_playbook_tests()

    print(f"\n📊 Test Suite Results:")
    print(f"   Total Tests: {results['total_tests']}")
    print(f"   Passed: {results['passed']} ✅")
    print(f"   Failed: {results['failed']} ❌")
    print(f"   Pass Rate: {results['pass_rate']:.1%}")
    print(f"   Total Duration: {results['total_duration_ms']:.0f}ms")
    print(f"   Avg Duration: {results['avg_duration_ms']:.0f}ms per test")

    print("\n" + "=" * 70 + "\n")


async def main():
    """Run all demos"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SOAR Playbook Demo - Automated Incident Response".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Run demos
    await demo_basic_playbook_execution()
    await demo_multiple_event_types()
    await demo_production_vs_dry_run()
    await demo_playbook_catalog()
    await demo_data_breach_response()
    await demo_testing_framework()

    print("🎉 All demos complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
