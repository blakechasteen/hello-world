"""
Standalone SIEM Integration Demo

Demonstrates SIEM integration without heavy dependencies.

Created: 2025-11-15
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Direct imports from SIEM modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.security.siem.taxonomy import (
    SecurityEventCategory,
    SecurityEventSubcategory,
    SeverityLevel,
    SecurityEvent,
    create_security_event,
    PIIRedactor,
    get_taxonomy_stats,
    CATEGORY_SUBCATEGORIES,
)

from HoloLoom.security.siem.core import (
    SIEMIntegration,
    SIEMConfig,
    LogBuffer,
    CircuitBreaker,
    CircuitState,
)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_event(event):
    """Print event details"""
    print(f"\n  Event ID: {event.event_id}")
    print(f"  Category: {event.category.value} / {event.subcategory.value}")
    print(f"  Action: {event.action}")
    print(f"  Severity: {event.level.value}")
    print(f"  Risk Score: {event.risk_score}/10.0")
    print(f"  Blocked: {event.blocked}")
    if event.source_ip:
        print(f"  Source IP: {event.source_ip}")
    if event.user_id:
        print(f"  User ID: {event.user_id} (hashed)")


async def demo_taxonomy():
    """Demo: Security event taxonomy"""
    print_section("1. Security Event Taxonomy")

    stats = get_taxonomy_stats()
    print(f"\n  Categories: {stats['categories']}")
    print(f"  Subcategories: {stats['subcategories']}")
    print(f"  Severity Levels: {stats['severity_levels']}")

    print("\n  Category → Subcategory Mapping:")
    for category, subcategories in CATEGORY_SUBCATEGORIES.items():
        print(f"\n    {category.value}: {len(subcategories)} subcategories")
        for subcat in subcategories[:3]:  # Show first 3
            print(f"      - {subcat.value}")
        if len(subcategories) > 3:
            print(f"      ... and {len(subcategories) - 3} more")

    print("\n  Sample Events:")

    # AUTH event
    auth_event = create_security_event(
        category=SecurityEventCategory.AUTH,
        subcategory=SecurityEventSubcategory.LOGIN,
        action="user_login",
        blocked=False,
        risk_score=1.0,
        source_ip="192.168.1.100",
        user_id="alice@example.com",
    )
    print("\n  → Authentication Event:")
    print_event(auth_event)

    # ATTACK event
    attack_event = create_security_event(
        category=SecurityEventCategory.ATTACK,
        subcategory=SecurityEventSubcategory.SQL_INJECTION,
        action="query_database",
        blocked=True,
        risk_score=9.0,
        source_ip="10.0.0.5",
        user_id="attacker@evil.com",
        payload="SELECT * FROM users WHERE id='1' OR '1'='1'",
    )
    print("\n  → Attack Event:")
    print_event(attack_event)


async def demo_pii_redaction():
    """Demo: Automatic PII redaction"""
    print_section("2. Automatic PII Redaction")

    # Sample text with PII
    pii_text = """Contact: alice@example.com, Phone: 555-123-4567, SSN: 123-45-6789, Card: 1234-5678-9012-3456, IP: 192.168.1.100"""

    print("\n  Original Text:")
    print(f"  {pii_text}")

    print("\n  Redacted (preserve IPs):")
    redacted = PIIRedactor.redact(pii_text, preserve_ips=True)
    print(f"  {redacted}")

    print("\n  Redacted (full):")
    redacted_full = PIIRedactor.redact(pii_text, preserve_ips=False)
    print(f"  {redacted_full}")

    print("\n  User ID Hashing:")
    users = ["alice@example.com", "bob@example.com", "alice@example.com"]
    for user in users:
        hashed = PIIRedactor.hash_user_id(user)
        print(f"    {user:25s} → {hashed}")


async def demo_log_buffer():
    """Demo: Log buffer"""
    print_section("3. Log Buffer")

    buffer = LogBuffer(max_size=10)

    print("\n  Adding events to buffer...")
    for i in range(5):
        event = create_security_event(
            category=SecurityEventCategory.AUTH,
            subcategory=SecurityEventSubcategory.LOGIN,
            action=f"login_{i}",
            blocked=False,
            risk_score=1.0,
        )
        await buffer.add(event)

    print(f"  Buffer size: {await buffer.size()}")

    print("\n  Getting batch (3 events)...")
    batch = await buffer.get_batch(3)
    print(f"  Retrieved: {len(batch)} events")
    print(f"  Buffer size: {await buffer.size()}")


async def demo_circuit_breaker():
    """Demo: Circuit breaker pattern"""
    print_section("4. Circuit Breaker Pattern")

    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        success_threshold=2,
    )

    print(f"\n  Initial state: {cb.state.value}")

    # Simulate failures
    print("\n  Simulating backend failures...")
    for i in range(3):
        cb.record_failure()
        print(f"    Failure {i+1}: state = {cb.state.value}, can_attempt = {cb.can_attempt()}")

    print(f"\n  Circuit opened: {cb.state == CircuitState.OPEN}")

    # Wait for recovery
    print("\n  Waiting for recovery timeout (1s)...")
    await asyncio.sleep(1.1)

    print(f"  Can attempt: {cb.can_attempt()}")
    print(f"  State: {cb.state.value}")

    # Simulate recovery
    print("\n  Simulating successful operations...")
    cb.record_success()
    print(f"    Success 1: state = {cb.state.value}")
    cb.record_success()
    print(f"    Success 2: state = {cb.state.value}")

    print(f"\n  Circuit closed: {cb.state == CircuitState.CLOSED}")


async def demo_event_serialization():
    """Demo: Event serialization"""
    print_section("5. Event Serialization")

    # Create event
    event = create_security_event(
        category=SecurityEventCategory.DATA,
        subcategory=SecurityEventSubcategory.EXPORT,
        action="export_user_data",
        blocked=False,
        risk_score=4.5,
        source_ip="192.168.1.50",
        user_id="bob@example.com",
        target="users_table",
        metadata={"record_count": 1500},
    )

    print("\n  Original Event:")
    print_event(event)

    print("\n  Serialized to JSON:")
    event_dict = event.to_dict()
    print(json.dumps(event_dict, indent=2))

    print("\n  Deserialized from JSON:")
    restored = SecurityEvent.from_dict(event_dict)
    print(f"    Category: {restored.category.value}")
    print(f"    Subcategory: {restored.subcategory.value}")
    print(f"    Event ID match: {restored.event_id == event.event_id}")


async def demo_severity_inference():
    """Demo: Automatic severity inference"""
    print_section("6. Automatic Severity Inference")

    test_cases = [
        ("High-risk attack", SecurityEventCategory.ATTACK, SecurityEventSubcategory.SQL_INJECTION, 9.0, True),
        ("Medium-risk attack", SecurityEventCategory.ATTACK, SecurityEventSubcategory.XSS, 6.5, True),
        ("Failed auth", SecurityEventCategory.AUTH, SecurityEventSubcategory.FAILED_AUTH, 3.0, True),
        ("Normal login", SecurityEventCategory.AUTH, SecurityEventSubcategory.LOGIN, 1.0, False),
        ("Data query", SecurityEventCategory.DATA, SecurityEventSubcategory.QUERY, 2.0, False),
        ("Incident", SecurityEventCategory.INCIDENT, SecurityEventSubcategory.BREACH_DETECTED, 10.0, True),
    ]

    print("\n  Severity inference for different event types:")
    for desc, category, subcategory, risk_score, blocked in test_cases:
        event = create_security_event(
            category=category,
            subcategory=subcategory,
            action="test_action",
            blocked=blocked,
            risk_score=risk_score,
        )
        print(f"\n    {desc}")
        print(f"      Category: {category.value}/{subcategory.value}")
        print(f"      Risk: {risk_score}/10, Blocked: {blocked}")
        print(f"      → Severity: {event.level.value}")


async def demo_query_examples():
    """Demo: Query examples for different SIEM backends"""
    print_section("7. SIEM Backend Query Examples")

    print("\n  Splunk SPL Query:")
    print("""
    index=hololoom_security
    | search category=ATTACK risk_score>=7.0
    | stats count by subcategory, blocked
    | timechart span=1h count by subcategory
    """)

    print("\n  Elasticsearch DSL Query:")
    print("""
    GET hololoom-security-*/_search
    {
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": "ATTACK"}},
                    {"range": {"risk_score": {"gte": 7.0}}}
                ]
            }
        },
        "aggs": {
            "attacks_over_time": {
                "date_histogram": {
                    "field": "timestamp",
                    "interval": "1h"
                },
                "aggs": {
                    "by_type": {
                        "terms": {"field": "subcategory"}
                    }
                }
            }
        }
    }
    """)

    print("\n  Datadog Log Search Query:")
    print("""
    service:hololoom source:security
    @category:ATTACK @risk_score:>=7.0
    | group by @subcategory, @blocked
    | timeseries count by @subcategory
    """)


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  HoloLoom SIEM Integration Demo (Standalone)")
    print("  Created: 2025-11-15")
    print("=" * 80)

    await demo_taxonomy()
    await demo_pii_redaction()
    await demo_log_buffer()
    await demo_circuit_breaker()
    await demo_event_serialization()
    await demo_severity_inference()
    await demo_query_examples()

    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)
    print("\n  Key Features Demonstrated:")
    print("    ✓ Security event taxonomy (6 categories, 28+ subcategories)")
    print("    ✓ Automatic PII redaction (emails, SSNs, credit cards, phones)")
    print("    ✓ Log buffering with overflow handling")
    print("    ✓ Circuit breaker for SIEM backend failures")
    print("    ✓ Event serialization (JSON round-trip)")
    print("    ✓ Automatic severity inference")
    print("    ✓ Multi-backend query examples (Splunk, ELK, Datadog)")
    print("\n  Next Steps:")
    print("    1. Configure your SIEM backend")
    print("    2. Integrate with HoloLoom security components")
    print("    3. Set up monitoring dashboards")
    print("    4. Configure alerting rules")
    print()


if __name__ == "__main__":
    asyncio.run(main())
