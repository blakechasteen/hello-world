#!/usr/bin/env python3
"""
Quick manual validation of Privacy & Compliance module.
Run with: python validate_privacy.py
"""

import asyncio
from datetime import datetime

# Test imports
try:
    from HoloLoom.privacy import (
        create_default_detector,
        PIIType,
        TenantRegistry,
        TenantIsolationLayer,
        TenantContext,
        TenantTier,
        PIIFlowTracker,
        create_encryption_manager,
        ComplianceManager,
    )
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)


async def main():
    print("\n🔍 Testing Privacy & Compliance Module\n")
    print("="* 60)

    # 1. PII Detection
    print("\n1️⃣  PII Detection")
    print("-" * 60)
    detector = create_default_detector()

    test_text = "Contact john.doe@example.com, SSN: 123-45-6789, Phone: 555-1234"
    result = detector.analyze(test_text)

    print(f"Original: {test_text}")
    print(f"Risk Level: {result.risk_level}")
    print(f"PII Types Found: {[p.value for p in result.pii_types]}")
    print(f"Redacted: {result.redacted_text}")

    assert result.has_pii, "Should detect PII"
    assert result.risk_level == "CRITICAL", "Should be CRITICAL (SSN present)"
    print("✅ PII Detection working correctly")

    # 2. Tenant Isolation
    print("\n2️⃣  Tenant Isolation")
    print("-" * 60)
    registry = TenantRegistry()

    tenant = await registry.create_tenant(
        tenant_id="test_corp",
        name="Test Corporation",
        tier=TenantTier.PROFESSIONAL
    )

    print(f"Created tenant: {tenant.tenant_id}")
    print(f"Tier: {tenant.tier.value}")
    print(f"Max memories: {tenant.max_memories}")

    isolation = TenantIsolationLayer(registry)

    # Test key scoping
    scoped = isolation.scope_key("memory_123", "test_corp")
    print(f"Scoped key: {scoped}")
    assert scoped == "tenant:test_corp:memory_123", "Key scoping failed"

    # Test unscoping
    tenant_id, key = isolation.unscope_key(scoped)
    assert tenant_id == "test_corp" and key == "memory_123", "Unscoping failed"
    print("✅ Tenant Isolation working correctly")

    # 3. PII Flow Tracking
    print("\n3️⃣  PII Flow Tracking")
    print("-" * 60)
    tracker = PIIFlowTracker(detector)

    context = TenantContext(
        tenant_id="test_corp",
        user_id="john@test.com",
        permissions={"read", "write"}
    )

    event = await tracker.track_ingestion(
        text="User email: john@test.com",
        context=context,
        purpose="Test ingestion"
    )

    print(f"Tracked event: {event.event_id}")
    print(f"Stage: {event.stage.value}")
    print(f"Action: {event.action.value}")
    print(f"PII detected: {len(event.pii_detections)}")

    assert len(event.pii_detections) > 0, "Should detect PII in event"
    print("✅ PII Flow Tracking working correctly")

    # 4. Encryption
    print("\n4️⃣  Data Encryption")
    print("-" * 60)
    encryption = create_encryption_manager()

    key = encryption.generate_key_for_tenant("test_corp")
    print(f"Generated key: {key.key_id}")

    plaintext = "sensitive@data.com"
    encrypted = encryption.encrypt_string(plaintext, "test_corp")
    print(f"Encrypted: {encrypted.ciphertext[:20]}... ({len(encrypted.ciphertext)} bytes)")

    decrypted = encryption.decrypt_string(encrypted, "test_corp")
    print(f"Decrypted: {decrypted}")

    assert decrypted == plaintext, "Encryption roundtrip failed"
    print("✅ Encryption working correctly")

    # 5. Compliance
    print("\n5️⃣  Compliance Reporting")
    print("-" * 60)
    compliance = ComplianceManager(registry, tracker, detector)

    report = await compliance.generate_gdpr_article30_report(
        tenant_id="test_corp"
    )

    print(f"Report for: {report.tenant_id}")
    print(f"Processing events: {report.total_processing_events}")
    print(f"Security measures: {len(report.security_measures)}")
    print(f"Compliant: {report.compliant}")

    assert report.tenant_id == "test_corp", "Report generation failed"
    print("✅ Compliance Reporting working correctly")

    # Summary
    print("\n" + "=" * 60)
    print("🎉 All validation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
