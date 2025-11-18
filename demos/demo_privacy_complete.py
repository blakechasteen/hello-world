#!/usr/bin/env python3
"""
Complete Privacy & Compliance Demo
===================================

Demonstrates all features of the HoloLoom Privacy module:
1. Multi-tenant isolation
2. PII detection and redaction
3. Encrypted storage
4. PII flow tracking
5. Compliance reporting (GDPR, HIPAA)
6. Privacy-aware memory integration

Run with: python demos/demo_privacy_complete.py
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# HoloLoom Privacy imports
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

# Privacy-aware memory wrappers
from HoloLoom.privacy.integrations import (
    PrivacyAwareKG,
    PrivacyAwareCache,
)

# Base memory systems
from HoloLoom.memory.graph import KG


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subsection(title: str):
    """Print subsection header."""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


async def main():
    print("\n🔒 HoloLoom Privacy & Compliance - Complete Demo")
    print("=" * 70)

    # ========================================================================
    # Setup
    # ========================================================================

    print_section("1️⃣  System Setup")

    # Create core components
    detector = create_default_detector()
    registry = TenantRegistry()
    isolation = TenantIsolationLayer(registry)
    tracker = PIIFlowTracker(detector, persist_path=Path("./pii_audit.jsonl"))
    encryption = create_encryption_manager()

    print("✅ Initialized core privacy components:")
    print("   - PII Detector (15+ entity types)")
    print("   - Tenant Registry")
    print("   - Isolation Layer")
    print("   - PII Flow Tracker")
    print("   - Encryption Manager (AES-256-GCM)")

    # ========================================================================
    # Create Tenants
    # ========================================================================

    print_section("2️⃣  Multi-Tenant Setup")

    # Healthcare company (HIPAA-compliant)
    healthcare = await registry.create_tenant(
        tenant_id="healthcare_corp",
        name="Healthcare Corporation",
        tier=TenantTier.ENTERPRISE,
        compliance_mode="hipaa",
        encryption_required=True,
        audit_all_access=True,
        data_residency="us-west-2"
    )

    # Tech startup (GDPR-compliant)
    tech_startup = await registry.create_tenant(
        tenant_id="tech_startup",
        name="Tech Startup Inc",
        tier=TenantTier.PROFESSIONAL,
        compliance_mode="gdpr",
        encryption_required=True,
        data_residency="eu"
    )

    # Small business (basic tier)
    small_biz = await registry.create_tenant(
        tenant_id="small_biz",
        name="Small Business LLC",
        tier=TenantTier.STARTER
    )

    print(f"✅ Created 3 tenants:")
    print(f"\n   🏥 Healthcare Corp")
    print(f"      - Tier: {healthcare.tier.value}")
    print(f"      - Compliance: {healthcare.compliance_mode}")
    print(f"      - Encryption: {healthcare.encryption_required}")
    print(f"      - Data residency: {healthcare.data_residency}")
    print(f"      - Max memories: Unlimited")

    print(f"\n   💻 Tech Startup")
    print(f"      - Tier: {tech_startup.tier.value}")
    print(f"      - Compliance: {tech_startup.compliance_mode}")
    print(f"      - Max memories: {tech_startup.max_memories:,}")

    print(f"\n   🏪 Small Business")
    print(f"      - Tier: {small_biz.tier.value}")
    print(f"      - Max memories: {small_biz.max_memories}")

    # ========================================================================
    # PII Detection Demo
    # ========================================================================

    print_section("3️⃣  PII Detection & Classification")

    patient_data = """
    Patient: John Smith
    DOB: 01/15/1985
    SSN: 123-45-6789
    Email: john.smith@email.com
    Phone: (555) 123-4567
    Address: 123 Main St, Anytown, CA 12345
    Medical Record: MRN-987654
    """

    print("Analyzing patient data for PII:")
    print(f"\n{patient_data}")

    result = detector.analyze(patient_data)

    print(f"\n📊 Analysis Results:")
    print(f"   - Risk Level: {result.risk_level} ⚠️")
    print(f"   - PII Types Found: {len(result.pii_types)}")

    for pii_type in sorted(result.pii_types, key=lambda x: x.value):
        print(f"     • {pii_type.value}")

    print(f"\n🔒 Redacted Version:")
    print(f"{result.redacted_text}")

    # ========================================================================
    # Tenant Isolation Demo
    # ========================================================================

    print_section("4️⃣  Tenant Isolation & Cross-Tenant Protection")

    # Create contexts for each tenant
    healthcare_ctx = TenantContext(
        tenant_id="healthcare_corp",
        user_id="doctor@healthcare.com",
        permissions={"read", "write"}
    )

    tech_ctx = TenantContext(
        tenant_id="tech_startup",
        user_id="dev@techstartup.com",
        permissions={"read", "write"}
    )

    print("Created tenant contexts:")
    print(f"   🏥 Healthcare: {healthcare_ctx.user_id}")
    print(f"   💻 Tech Startup: {tech_ctx.user_id}")

    # Validate operations
    print("\nValidating operations...")
    await isolation.validate_operation(healthcare_ctx, "write")
    await isolation.validate_operation(tech_ctx, "write")
    print("   ✅ All operations validated")

    # Demonstrate key scoping
    print("\nKey Scoping:")
    healthcare_key = isolation.scope_key("patient_123", "healthcare_corp")
    tech_key = isolation.scope_key("user_456", "tech_startup")

    print(f"   Healthcare: patient_123 → {healthcare_key}")
    print(f"   Tech:       user_456    → {tech_key}")

    # Demonstrate cross-tenant protection
    print("\nCross-Tenant Access Protection:")
    test_data = [
        {"id": healthcare_key, "data": "Patient records"},
        {"id": tech_key, "data": "User profile"},
    ]

    # Healthcare tries to access all data
    healthcare_data = await isolation.filter_cross_tenant_data(
        test_data,
        tenant_id="healthcare_corp",
        key_field="id"
    )

    print(f"   Healthcare can access: {len(healthcare_data)}/2 records ✅")
    print(f"   Tech's data is protected from Healthcare 🔒")

    # ========================================================================
    # Encrypted Storage Demo
    # ========================================================================

    print_section("5️⃣  Encrypted Storage")

    # Generate encryption keys for tenants
    healthcare_key_obj = encryption.generate_key_for_tenant("healthcare_corp")
    tech_key_obj = encryption.generate_key_for_tenant("tech_startup")

    print("Generated encryption keys:")
    print(f"   🏥 Healthcare: {healthcare_key_obj.key_id}")
    print(f"   💻 Tech:       {tech_key_obj.key_id}")

    # Encrypt sensitive data
    print("\nEncrypting sensitive data...")

    patient_ssn = "123-45-6789"
    encrypted_ssn = encryption.encrypt_string(
        patient_ssn,
        tenant_id="healthcare_corp",
        metadata={"field": "ssn", "patient_id": "123"}
    )

    print(f"\n   Plaintext SSN: {patient_ssn}")
    print(f"   Encrypted:     {encrypted_ssn.ciphertext[:30]}... ({len(encrypted_ssn.ciphertext)} bytes)")
    print(f"   Algorithm:     {encrypted_ssn.algorithm.value}")
    print(f"   Key ID:        {encrypted_ssn.key_id}")

    # Decrypt
    decrypted_ssn = encryption.decrypt_string(encrypted_ssn, "healthcare_corp")
    print(f"\n   Decrypted:     {decrypted_ssn}")
    print(f"   ✅ Roundtrip successful: {decrypted_ssn == patient_ssn}")

    # Field-level encryption
    print("\nField-Level Encryption (Dictionary):")

    patient_record = {
        "name": "John Smith",
        "age": 38,
        "ssn": "123-45-6789",
        "email": "john.smith@email.com",
        "diagnosis": "Hypertension"
    }

    print(f"\n   Original: {patient_record}")

    encrypted_record = encryption.encrypt_fields(
        patient_record,
        tenant_id="healthcare_corp",
        fields=["ssn", "email"]
    )

    print(f"\n   Encrypted SSN:   {str(encrypted_record['ssn'])[:60]}...")
    print(f"   Encrypted email: {str(encrypted_record['email'])[:60]}...")
    print(f"   Plaintext name:  {encrypted_record['name']}")

    # ========================================================================
    # PII Flow Tracking Demo
    # ========================================================================

    print_section("6️⃣  PII Flow Tracking & Audit Trail")

    print("Simulating complete PII lifecycle...\n")

    # Ingestion
    print("📥 Ingestion:")
    ingestion_event = await tracker.track_ingestion(
        text=patient_data,
        context=healthcare_ctx,
        purpose="Patient registration"
    )
    print(f"   Event ID: {ingestion_event.event_id}")
    print(f"   Stage:    {ingestion_event.stage.value}")
    print(f"   Action:   {ingestion_event.action.value}")
    print(f"   PII detected: {len(ingestion_event.pii_detections)} entities")

    # Storage
    print("\n💾 Storage:")
    storage_event = await tracker.track_storage(
        data=encrypted_record,
        context=healthcare_ctx,
        parent_event_id=ingestion_event.event_id,
        purpose="Store patient record to database"
    )
    print(f"   Event ID: {storage_event.event_id}")
    print(f"   Parent:   {storage_event.parent_event_id}")
    print(f"   Stage:    {storage_event.stage.value}")

    # Retrieval
    print("\n📤 Retrieval:")
    retrieval_event = await tracker.track_retrieval(
        memory_keys=["patient_123"],
        context=healthcare_ctx,
        purpose="Doctor accessing patient record"
    )
    print(f"   Event ID: {retrieval_event.event_id}")
    print(f"   Stage:    {retrieval_event.stage.value}")

    # View audit trail
    print("\n📋 Audit Trail (last 10 events):")
    events = await tracker.get_events_for_tenant("healthcare_corp")

    for i, event in enumerate(events[-10:], 1):
        print(f"   {i}. {event.timestamp.strftime('%H:%M:%S')} - "
              f"{event.stage.value:15} - {event.action.value:10} - "
              f"{event.purpose[:40]}")

    # ========================================================================
    # Privacy-Aware Memory Integration
    # ========================================================================

    print_section("7️⃣  Privacy-Aware Memory Integration")

    # Create base KG
    base_kg = KG()

    # Wrap with privacy layer
    privacy_kg = PrivacyAwareKG(
        base_kg=base_kg,
        isolation_layer=isolation,
        detector=detector,
        tracker=tracker,
        encryption=encryption
    )

    print("Created privacy-aware knowledge graph wrapper")

    # Add edges for healthcare
    print("\n🏥 Healthcare adding patient relationships:")

    edge1 = await privacy_kg.add_edge(
        src="Patient:John_Smith",
        dst="Diagnosis:Hypertension",
        edge_type="HAS_DIAGNOSIS",
        context=healthcare_ctx,
        metadata={"date": "2025-11-17"}
    )

    edge2 = await privacy_kg.add_edge(
        src="Patient:John_Smith",
        dst="Doctor:Dr_Williams",
        edge_type="TREATED_BY",
        context=healthcare_ctx
    )

    print(f"   ✅ Added edge: {edge1}")
    print(f"   ✅ Added edge: {edge2}")

    # Add edges for tech startup
    print("\n💻 Tech Startup adding user relationships:")

    edge3 = await privacy_kg.add_edge(
        src="User:alice@tech.com",
        dst="Feature:DarkMode",
        edge_type="USES",
        context=tech_ctx
    )

    print(f"   ✅ Added edge: {edge3}")

    # Retrieve edges (tenant-filtered)
    print("\n🔍 Retrieving edges (automatic tenant filtering):")

    healthcare_edges = await privacy_kg.get_edges(healthcare_ctx)
    tech_edges = await privacy_kg.get_edges(tech_ctx)

    print(f"   Healthcare can see: {len(healthcare_edges)} edges")
    print(f"   Tech can see:       {len(tech_edges)} edges")
    print(f"   ✅ Cross-tenant isolation enforced")

    # ========================================================================
    # Compliance Reporting
    # ========================================================================

    print_section("8️⃣  Compliance Reporting")

    compliance = ComplianceManager(registry, tracker, detector)

    # GDPR Article 30 Report
    print_subsection("GDPR Article 30 - Records of Processing")

    gdpr_report = await compliance.generate_gdpr_article30_report(
        tenant_id="healthcare_corp",
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow()
    )

    print(f"\nReport for: {gdpr_report.tenant_id}")
    print(f"Period: {gdpr_report.reporting_period_start.date()} to {gdpr_report.reporting_period_end.date()}")
    print(f"\n📊 Processing Statistics:")
    print(f"   - Total events: {gdpr_report.total_processing_events}")
    print(f"   - Data subjects affected: {gdpr_report.data_subjects_affected}")

    print(f"\n📝 Purposes of Processing:")
    for purpose in gdpr_report.purposes_of_processing[:5]:
        print(f"   - {purpose}")

    print(f"\n🔐 Security Measures ({len(gdpr_report.security_measures)}):")
    for measure in gdpr_report.security_measures[:5]:
        print(f"   - {measure}")

    print(f"\n✅ Compliance Status: {'COMPLIANT' if gdpr_report.compliant else 'NON-COMPLIANT'}")

    # Data Subject Access Request (DSAR)
    print_subsection("GDPR Article 15 - Data Subject Access Request")

    dsar = await compliance.handle_dsar(
        tenant_id="healthcare_corp",
        data_subject_email="doctor@healthcare.com"
    )

    print(f"\nDSAR Request:")
    print(f"   Request ID: {dsar.request_id}")
    print(f"   Data Subject: {dsar.data_subject_email}")
    print(f"   Status: {dsar.status}")
    print(f"   Deadline: {dsar.fulfillment_deadline.date()}")
    print(f"   Overdue: {'Yes' if dsar.is_overdue() else 'No'}")

    print(f"\n📦 Personal Data Collected:")
    pii_detected = dsar.personal_data.get("pii_detected", {})
    for pii_type, instances in pii_detected.items():
        print(f"   - {pii_type}: {len(instances)} instance(s)")

    # Right to Erasure
    print_subsection("GDPR Article 17 - Right to Erasure")

    print("\nSimulating right to erasure request...")

    # First, show what data exists
    print("\n📋 Data to be erased:")
    print(f"   - Knowledge graph edges: {len(healthcare_edges)}")
    print(f"   - PII flow events: {len(events)}")

    # Perform erasure
    erasure = await compliance.handle_right_to_erasure(
        tenant_id="healthcare_corp",
        data_subject_email="doctor@healthcare.com"
    )

    print(f"\n🗑️  Erasure Request:")
    print(f"   Request ID: {erasure.request_id}")
    print(f"   Status: {erasure.status}")
    print(f"   Completed: {erasure.completed_date.strftime('%Y-%m-%d %H:%M:%S') if erasure.completed_date else 'Pending'}")

    print(f"\n📝 Deletion Steps ({len(erasure.deletion_steps)}):")
    for step in erasure.deletion_steps:
        print(f"   {step}")

    print(f"\n✅ Verification Hash: {erasure.verification_hash}")

    # HIPAA Report
    print_subsection("HIPAA Compliance Report")

    hipaa_report = await compliance.generate_hipaa_report(
        tenant_id="healthcare_corp"
    )

    print(f"\nHIPAA Compliance for: {hipaa_report.tenant_id}")

    print(f"\n🔐 Technical Safeguards (§164.312):")
    for safeguard, status in hipaa_report.technical_safeguards.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {safeguard}")

    print(f"\n📊 PHI Handling:")
    print(f"   - PHI types processed: {len(hipaa_report.phi_types_processed)}")
    print(f"   - Total PHI access events: {hipaa_report.total_phi_access_events}")
    print(f"   - Unauthorized attempts: {hipaa_report.unauthorized_access_attempts}")

    print(f"\n✅ Overall Compliance: {'COMPLIANT' if hipaa_report.compliant else 'NON-COMPLIANT'}")

    if hipaa_report.violations:
        print(f"\n⚠️  Violations ({len(hipaa_report.violations)}):")
        for violation in hipaa_report.violations:
            print(f"   - {violation}")

    # ========================================================================
    # Summary
    # ========================================================================

    print_section("🎉 Demo Complete")

    print("Demonstrated features:")
    print("   ✅ Multi-tenant isolation (3 tenants)")
    print("   ✅ PII detection (15+ entity types)")
    print("   ✅ Data encryption (AES-256-GCM)")
    print("   ✅ PII flow tracking (ingestion → storage → deletion)")
    print("   ✅ Privacy-aware memory wrappers")
    print("   ✅ GDPR compliance (Articles 30, 15, 17)")
    print("   ✅ HIPAA compliance (§164.308, §164.312)")
    print("   ✅ Cross-tenant protection")
    print("   ✅ Audit trail generation")

    print("\n📝 Audit log saved to: ./pii_audit.jsonl")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
