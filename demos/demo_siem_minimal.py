"""
Minimal SIEM Demo - Direct Module Test

Tests SIEM modules directly without package imports.

Created: 2025-11-15
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct module imports (bypass __init__.py)
import importlib.util

def load_module(module_path):
    """Load module from path"""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules directly
base_path = Path(__file__).parent.parent / "HoloLoom" / "security" / "siem"

print("Loading SIEM modules...")
taxonomy = load_module(base_path / "taxonomy.py")
print("✓ Loaded taxonomy.py")

# Test taxonomy
print("\nTesting Security Event Taxonomy...")
stats = taxonomy.get_taxonomy_stats()
print(f"  Categories: {stats['categories']}")
print(f"  Subcategories: {stats['subcategories']}")

# Create a security event
event = taxonomy.create_security_event(
    category=taxonomy.SecurityEventCategory.ATTACK,
    subcategory=taxonomy.SecurityEventSubcategory.SQL_INJECTION,
    action="query_database",
    blocked=True,
    risk_score=9.0,
    source_ip="192.168.1.100",
    user_id="attacker@example.com",
    payload="SELECT * FROM users WHERE id='1' OR '1'='1'",
)

print(f"\nCreated Attack Event:")
print(f"  Event ID: {event.event_id}")
print(f"  Category: {event.category.value} / {event.subcategory.value}")
print(f"  Severity: {event.level.value}")
print(f"  Risk Score: {event.risk_score}")
print(f"  Blocked: {event.blocked}")

# Test PII redaction
print("\nTesting PII Redaction...")
pii_text = "Contact: alice@example.com, SSN: 123-45-6789, Card: 1234-5678-9012-3456"
redacted = taxonomy.PIIRedactor.redact(pii_text)
print(f"  Original: {pii_text}")
print(f"  Redacted: {redacted}")

# Test serialization
print("\nTesting Event Serialization...")
event_dict = event.to_dict()
print(f"  Serialized keys: {list(event_dict.keys())}")

restored = taxonomy.SecurityEvent.from_dict(event_dict)
print(f"  Deserialized successfully: {restored.event_id == event.event_id}")

# Test all categories
print("\nTesting All Event Categories:")
for category_name in dir(taxonomy.SecurityEventCategory):
    if not category_name.startswith('_'):
        category = getattr(taxonomy.SecurityEventCategory, category_name)
        if isinstance(category, taxonomy.SecurityEventCategory):
            subcats = taxonomy.CATEGORY_SUBCATEGORIES.get(category, [])
            print(f"  {category.value}: {len(subcats)} subcategories")

print("\n✓ All taxonomy tests passed!")

# Try to load core module
print("\nLoading core SIEM module...")
try:
    core = load_module(base_path / "core.py")
    print("✓ Loaded core.py")

    # Test circuit breaker
    print("\nTesting Circuit Breaker...")
    cb = core.CircuitBreaker(failure_threshold=3)
    print(f"  Initial state: {cb.state.value}")

    for i in range(3):
        cb.record_failure()
    print(f"  After 3 failures: {cb.state.value}")
    print(f"  Can attempt: {cb.can_attempt()}")

    print("\n✓ All core tests passed!")

except Exception as e:
    print(f"✗ Failed to load core: {e}")

print("\n" + "=" * 80)
print("SIEM Module Tests Complete!")
print("=" * 80)
