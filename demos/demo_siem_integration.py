"""
SIEM Integration Demo

Demonstrates HoloLoom's Security Information and Event Management (SIEM) integration.

Shows:
- Multi-backend support (Splunk, ELK, Datadog)
- Structured security logging with taxonomy
- Automatic PII redaction
- Circuit breaker and retry logic
- Fallback to file logging
- Query and search capabilities

Created: 2025-11-15
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.security.siem import (
    SIEMIntegration,
    SIEMConfig,
    SecurityEventCategory,
    SecurityEventSubcategory,
    SeverityLevel,
    create_security_event,
    PIIRedactor,
    get_taxonomy_stats,
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
    print(f"  Total Event Types: {stats['total_event_types']}")

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

    # DATA event
    data_event = create_security_event(
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
    print("\n  → Data Access Event:")
    print_event(data_event)


async def demo_pii_redaction():
    """Demo: Automatic PII redaction"""
    print_section("2. Automatic PII Redaction")

    # Sample text with PII
    pii_text = """
    Contact: alice@example.com
    Phone: 555-123-4567
    SSN: 123-45-6789
    Card: 1234-5678-9012-3456
    IP: 192.168.1.100
    """

    print("\n  Original Text:")
    print(pii_text)

    print("\n  Redacted (preserve IPs):")
    redacted = PIIRedactor.redact(pii_text, preserve_ips=True)
    print(redacted)

    print("\n  Redacted (full):")
    redacted_full = PIIRedactor.redact(pii_text, preserve_ips=False)
    print(redacted_full)

    print("\n  User ID Hashing:")
    user_id = "alice@example.com"
    hashed = PIIRedactor.hash_user_id(user_id)
    print(f"  Original: {user_id}")
    print(f"  Hashed: {hashed}")


async def demo_file_logging():
    """Demo: File-based logging (fallback)"""
    print_section("3. File-Based Logging (Fallback)")

    # Create temporary directory for logs
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())

    config = SIEMConfig(
        backend="file",
        buffer_size=100,
        flush_interval=1.0,
        batch_size=10,
        fallback_dir=tmpdir,
        enable_fallback=True,
    )

    print(f"\n  Log directory: {tmpdir}")

    async with SIEMIntegration(config) as siem:
        print("\n  Logging events...")

        # Log various events
        events = [
            create_security_event(
                category=SecurityEventCategory.AUTH,
                subcategory=SecurityEventSubcategory.LOGIN,
                action="login",
                blocked=False,
                risk_score=1.0,
                user_id=f"user{i}@example.com",
            )
            for i in range(5)
        ]

        events.append(
            create_security_event(
                category=SecurityEventCategory.ATTACK,
                subcategory=SecurityEventSubcategory.BRUTE_FORCE,
                action="brute_force_login",
                blocked=True,
                risk_score=8.5,
                source_ip="10.0.0.99",
                metadata={"attempts": 50},
            )
        )

        events.append(
            create_security_event(
                category=SecurityEventCategory.INCIDENT,
                subcategory=SecurityEventSubcategory.ANOMALY_DETECTED,
                action="unusual_activity",
                blocked=False,
                risk_score=7.0,
                metadata={"anomaly_score": 0.95},
            )
        )

        await siem.log_events(events)

        # Wait for flush
        await asyncio.sleep(1.5)

        stats = siem.get_stats()
        print(f"\n  Events logged: {stats['events_logged']}")
        print(f"  Events sent: {stats['events_sent']}")
        print(f"  Fallback writes: {stats['fallback_writes']}")

    # Show fallback file
    fallback_files = list(tmpdir.glob("security_events_*.json"))
    if fallback_files:
        print(f"\n  Fallback file created: {fallback_files[0].name}")

        # Show sample events
        import json
        with open(fallback_files[0]) as f:
            logged_events = json.load(f)
            print(f"\n  Sample logged event:")
            sample = logged_events[0]
            print(f"    Category: {sample['category']}")
            print(f"    Subcategory: {sample['subcategory']}")
            print(f"    Action: {sample['action']}")
            print(f"    Risk Score: {sample['risk_score']}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


async def demo_circuit_breaker():
    """Demo: Circuit breaker pattern"""
    print_section("4. Circuit Breaker Pattern")

    from HoloLoom.security.siem.core import CircuitBreaker, CircuitState

    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2.0,
        success_threshold=2,
    )

    print(f"\n  Initial state: {cb.state.value}")

    # Simulate failures
    print("\n  Simulating backend failures...")
    for i in range(3):
        cb.record_failure()
        print(f"    Failure {i+1}: state = {cb.state.value}")

    print(f"\n  Circuit opened: {cb.state == CircuitState.OPEN}")
    print(f"  Can attempt: {cb.can_attempt()}")

    # Wait for recovery
    print("\n  Waiting for recovery timeout (2s)...")
    await asyncio.sleep(2.1)

    print(f"  Can attempt: {cb.can_attempt()}")
    print(f"  State: {cb.state.value}")

    # Simulate recovery
    print("\n  Simulating successful operations...")
    cb.record_success()
    print(f"    Success 1: state = {cb.state.value}")
    cb.record_success()
    print(f"    Success 2: state = {cb.state.value}")

    print(f"\n  Circuit closed: {cb.state == CircuitState.CLOSED}")


async def demo_backend_configs():
    """Demo: Backend configurations"""
    print_section("5. Backend Configurations")

    print("\n  Splunk Configuration:")
    print("""
    {
        "backend": "splunk",
        "splunk_config": {
            "hec_url": "https://splunk.example.com:8088",
            "hec_token": "your-hec-token-here",
            "index": "hololoom_security",
            "source": "hololoom",
            "verify_ssl": True
        }
    }
    """)

    print("\n  Query Example (Splunk SPL):")
    print("""
    index=hololoom_security
    | search category=ATTACK risk_score>=7.0
    | stats count by subcategory, blocked
    | sort -count
    """)

    print("\n  ELK Configuration:")
    print("""
    {
        "backend": "elk",
        "elk_config": {
            "es_url": "https://elasticsearch.example.com:9200",
            "username": "elastic",
            "password": "your-password",
            "index_pattern": "hololoom-security-{date}",
            "verify_ssl": True
        }
    }
    """)

    print("\n  Query Example (Elasticsearch DSL):")
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
            "by_subcategory": {
                "terms": {"field": "subcategory"}
            }
        }
    }
    """)

    print("\n  Datadog Configuration:")
    print("""
    {
        "backend": "datadog",
        "datadog_config": {
            "api_key": "your-datadog-api-key",
            "site": "datadoghq.com",
            "service": "hololoom",
            "source": "security",
            "tags": ["env:production", "team:security"]
        }
    }
    """)

    print("\n  Query Example (Datadog Log Search):")
    print("""
    service:hololoom source:security
    @category:ATTACK @risk_score:>=7.0
    | group by @subcategory, @blocked
    """)


async def demo_integration_points():
    """Demo: Integration with HoloLoom security components"""
    print_section("6. Integration Points")

    print("\n  SIEM integrates with:")
    print("    • SafetyGuardrails - Log blocked actions")
    print("    • DeceptionDetection - Log deception attempts")
    print("    • InstrumentalConvergence - Log power-seeking behavior")
    print("    • AuditTrail - Forward audit entries")
    print("    • RateLimiter - Log rate limit violations")
    print("    • WAF - Log attack attempts")

    print("\n  Example: Log safety guardrail action")
    event = create_security_event(
        category=SecurityEventCategory.AUTHZ,
        subcategory=SecurityEventSubcategory.PERMISSION_DENIED,
        action="execute_code",
        blocked=True,
        risk_score=8.0,
        source_ip="192.168.1.100",
        user_id="user@example.com",
        target="os.system",
        metadata={
            "guardrail": "code_execution",
            "reason": "high_risk_action",
        },
    )
    print_event(event)


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  HoloLoom SIEM Integration Demo")
    print("  Created: 2025-11-15")
    print("=" * 80)

    await demo_taxonomy()
    await demo_pii_redaction()
    await demo_file_logging()
    await demo_circuit_breaker()
    await demo_backend_configs()
    await demo_integration_points()

    print("\n" + "=" * 80)
    print("  Demo Complete!")
    print("=" * 80)
    print("\n  Next Steps:")
    print("    1. Configure your SIEM backend (Splunk/ELK/Datadog)")
    print("    2. Integrate with HoloLoom security components")
    print("    3. Set up monitoring dashboards")
    print("    4. Configure alerting rules")
    print("    5. Run pytest HoloLoom/security/tests/test_siem.py")
    print()


if __name__ == "__main__":
    asyncio.run(main())
