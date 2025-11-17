"""
Comprehensive Integration Tests for Privacy and Compliance Module
=================================================================

Tests tenant isolation, PII detection, flow tracking, encryption, and compliance.

Run with:
    pytest HoloLoom/privacy/tests/test_privacy_integration.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from HoloLoom.privacy import (
    # PII Detection
    create_default_detector,
    PIIType,

    # Tenant Isolation
    TenantRegistry,
    TenantIsolationLayer,
    TenantContext,
    TenantTier,
    CrossTenantAccessError,
    TenantQuotaExceededError,

    # PII Flow Tracking
    PIIFlowTracker,
    PIIFlowStage,
    PIIAction,

    # Encryption
    create_encryption_manager,

    # Compliance
    ComplianceManager,
)


# ============================================================================
# PII Detection Tests
# ============================================================================

class TestPIIDetection:
    """Test PII detection and classification."""

    def test_email_detection(self):
        """Test email address detection."""
        detector = create_default_detector()

        text = "Contact me at john.doe@example.com for details"
        result = detector.analyze(text)

        assert result.has_pii
        assert PIIType.EMAIL in result.pii_types
        assert result.risk_level == "MEDIUM"
        assert "[EMAIL_REDACTED]" in result.redacted_text or "@example.com" in result.redacted_text

    def test_ssn_detection(self):
        """Test SSN detection."""
        detector = create_default_detector()

        text = "My SSN is 123-45-6789"
        result = detector.analyze(text)

        assert result.has_pii
        assert PIIType.SSN in result.pii_types
        assert result.risk_level == "CRITICAL"
        assert "[SSN_REDACTED]" in result.redacted_text

    def test_credit_card_detection(self):
        """Test credit card detection with Luhn validation."""
        detector = create_default_detector()

        # Valid Visa card (test number)
        text = "Card: 4532-1488-0343-6467"
        result = detector.analyze(text, enable_luhn_check=True)

        assert result.has_pii
        assert PIIType.CREDIT_CARD in result.pii_types
        assert result.risk_level == "CRITICAL"

    def test_multiple_pii_types(self):
        """Test detection of multiple PII types."""
        detector = create_default_detector()

        text = "Contact john@example.com, phone 555-123-4567, SSN 123-45-6789"
        result = detector.analyze(text)

        assert len(result.detections) >= 3
        assert PIIType.EMAIL in result.pii_types
        assert PIIType.PHONE in result.pii_types
        assert PIIType.SSN in result.pii_types
        assert result.risk_level == "CRITICAL"  # SSN present

    def test_no_pii(self):
        """Test text without PII."""
        detector = create_default_detector()

        text = "This is a normal sentence with no personal information."
        result = detector.analyze(text)

        assert not result.has_pii
        assert len(result.detections) == 0
        assert result.risk_level == "LOW"


# ============================================================================
# Tenant Isolation Tests
# ============================================================================

class TestTenantIsolation:
    """Test tenant isolation layer."""

    @pytest.mark.asyncio
    async def test_create_tenant(self):
        """Test tenant creation."""
        registry = TenantRegistry()

        tenant = await registry.create_tenant(
            tenant_id="acme_corp",
            name="Acme Corporation",
            tier=TenantTier.ENTERPRISE
        )

        assert tenant.tenant_id == "acme_corp"
        assert tenant.name == "Acme Corporation"
        assert tenant.tier == TenantTier.ENTERPRISE
        assert tenant.max_memories == -1  # Unlimited for enterprise

    @pytest.mark.asyncio
    async def test_tenant_key_scoping(self):
        """Test tenant key scoping."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        # Scope key to tenant
        scoped = isolation.scope_key("memory_123", "acme_corp")
        assert scoped == "tenant:acme_corp:memory_123"

        # Unscope key
        tenant_id, key = isolation.unscope_key(scoped)
        assert tenant_id == "acme_corp"
        assert key == "memory_123"

    @pytest.mark.asyncio
    async def test_cross_tenant_access_prevention(self):
        """Test prevention of cross-tenant access."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry, strict_mode=True)

        await registry.create_tenant("tenant_a", "Tenant A", TenantTier.FREE)
        await registry.create_tenant("tenant_b", "Tenant B", TenantTier.FREE)

        # Tenant A tries to access Tenant B's data
        items = [
            {"id": "tenant:tenant_a:mem_1", "data": "A's data"},
            {"id": "tenant:tenant_b:mem_2", "data": "B's data"},  # Should be filtered
        ]

        filtered = await isolation.filter_cross_tenant_data(
            items,
            tenant_id="tenant_a",
            key_field="id"
        )

        assert len(filtered) == 1
        assert filtered[0]["id"] == "tenant:tenant_a:mem_1"

    @pytest.mark.asyncio
    async def test_quota_enforcement(self):
        """Test quota enforcement."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        # Create free tier tenant (quota: 100 memories)
        tenant = await registry.create_tenant(
            "small_corp",
            "Small Corp",
            TenantTier.FREE
        )

        # Simulate hitting quota
        tenant.current_memories = 100

        context = TenantContext(
            tenant_id="small_corp",
            user_id="user@small.com",
            permissions={"write"}
        )

        # Should raise quota exceeded error
        with pytest.raises(TenantQuotaExceededError):
            await isolation.validate_operation(context, operation="write")

    @pytest.mark.asyncio
    async def test_permission_checks(self):
        """Test permission checking."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        await registry.create_tenant("test_corp", "Test Corp", TenantTier.FREE)

        # User with only read permission
        context = TenantContext(
            tenant_id="test_corp",
            user_id="user@test.com",
            permissions={"read"}
        )

        # Should pass for read
        await isolation.validate_operation(context, operation="read")

        # Should fail for write
        from HoloLoom.privacy.tenant_isolation import InsufficientPermissionsError
        with pytest.raises(InsufficientPermissionsError):
            await isolation.validate_operation(context, operation="write")


# ============================================================================
# PII Flow Tracking Tests
# ============================================================================

class TestPIIFlowTracking:
    """Test PII flow tracking and audit trail."""

    @pytest.mark.asyncio
    async def test_track_ingestion(self):
        """Test PII ingestion tracking."""
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)

        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john@acme.com",
            permissions={"write"}
        )

        event = await tracker.track_ingestion(
            text="My email is john@acme.com",
            context=context,
            purpose="User query"
        )

        assert event.stage == PIIFlowStage.INGESTION
        assert event.action == PIIAction.DETECTED
        assert len(event.pii_detections) > 0
        assert PIIType.EMAIL in {d.pii_type for d in event.pii_detections}

    @pytest.mark.asyncio
    async def test_pii_flow_chain(self):
        """Test complete PII flow chain."""
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)

        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john@acme.com",
            permissions={"write", "delete"}
        )

        # Ingestion
        ingestion = await tracker.track_ingestion(
            text="Email: john@acme.com",
            context=context,
            purpose="User query"
        )

        # Storage
        storage = await tracker.track_storage(
            data={"email": "john@acme.com"},
            context=context,
            parent_event_id=ingestion.event_id,
            purpose="Store to graph"
        )

        # Deletion
        deletion = await tracker.track_deletion(
            pii_references=["mem_123"],
            context=context,
            purpose="Right to erasure"
        )

        # Verify chain
        events = await tracker.get_events_for_tenant("acme_corp")
        assert len(events) >= 3

        # Verify chain completion
        chain = tracker._chains.get(ingestion.event_id)
        assert chain is not None
        assert chain.is_complete()

    @pytest.mark.asyncio
    async def test_compliance_report_generation(self):
        """Test compliance report generation."""
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)

        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john@acme.com",
            permissions={"write"}
        )

        # Generate some events
        await tracker.track_ingestion(
            "Email: john@acme.com",
            context=context,
            purpose="Query"
        )

        # Generate compliance report
        report = await tracker.generate_compliance_report(
            tenant_id="acme_corp",
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow()
        )

        assert report["tenant_id"] == "acme_corp"
        assert report["summary"]["total_events"] > 0
        assert "gdpr_compliance" in report


# ============================================================================
# Encryption Tests
# ============================================================================

class TestEncryption:
    """Test data encryption."""

    def test_encryption_roundtrip(self):
        """Test encrypt/decrypt roundtrip."""
        manager = create_encryption_manager()

        # Generate key for tenant
        key = manager.generate_key_for_tenant("acme_corp")
        assert key is not None

        # Encrypt data
        plaintext = "user@example.com"
        encrypted = manager.encrypt_string(plaintext, "acme_corp")

        assert encrypted.ciphertext != plaintext.encode()
        assert encrypted.key_id == key.key_id

        # Decrypt data
        decrypted = manager.decrypt_string(encrypted, "acme_corp")
        assert decrypted == plaintext

    def test_field_level_encryption(self):
        """Test field-level encryption for dictionaries."""
        manager = create_encryption_manager()
        manager.generate_key_for_tenant("acme_corp")

        data = {
            "email": "user@example.com",
            "name": "John Doe",
            "age": 30
        }

        # Encrypt only email field
        encrypted_data = manager.encrypt_fields(
            data,
            tenant_id="acme_corp",
            fields=["email"]
        )

        # Email should be encrypted
        assert isinstance(encrypted_data["email"], dict)
        assert "ciphertext" in encrypted_data["email"]

        # Other fields should be plaintext
        assert encrypted_data["name"] == "John Doe"
        assert encrypted_data["age"] == 30

        # Decrypt
        decrypted_data = manager.decrypt_fields(
            encrypted_data,
            tenant_id="acme_corp",
            fields=["email"]
        )

        assert decrypted_data["email"] == "user@example.com"


# ============================================================================
# Compliance Tests
# ============================================================================

class TestCompliance:
    """Test compliance reporting."""

    @pytest.mark.asyncio
    async def test_gdpr_article30_report(self):
        """Test GDPR Article 30 report generation."""
        registry = TenantRegistry()
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)
        compliance = ComplianceManager(registry, tracker, detector)

        # Create tenant
        await registry.create_tenant(
            "acme_corp",
            "Acme Corp",
            TenantTier.ENTERPRISE,
            compliance_mode="gdpr"
        )

        # Generate some PII events
        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john@acme.com",
            permissions={"write"}
        )

        await tracker.track_ingestion(
            "Email: john@acme.com, Phone: 555-1234",
            context=context,
            purpose="User registration"
        )

        # Generate report
        report = await compliance.generate_gdpr_article30_report(
            tenant_id="acme_corp",
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )

        assert report.tenant_id == "acme_corp"
        assert report.total_processing_events > 0
        assert len(report.categories_of_personal_data) > 0
        assert "User registration" in report.purposes_of_processing

    @pytest.mark.asyncio
    async def test_dsar_handling(self):
        """Test Data Subject Access Request handling."""
        registry = TenantRegistry()
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)
        compliance = ComplianceManager(registry, tracker, detector)

        await registry.create_tenant("acme_corp", "Acme Corp", TenantTier.FREE)

        # Generate user data
        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john@acme.com",
            permissions={"write"}
        )

        await tracker.track_ingestion(
            "My email is john@acme.com",
            context=context,
            purpose="User query"
        )

        # Handle DSAR
        dsar = await compliance.handle_dsar(
            tenant_id="acme_corp",
            data_subject_email="john@acme.com"
        )

        assert dsar.status == "completed"
        assert dsar.personal_data is not None
        assert "pii_detected" in dsar.personal_data
        assert not dsar.is_overdue()

    @pytest.mark.asyncio
    async def test_right_to_erasure(self):
        """Test right to erasure handling."""
        registry = TenantRegistry()
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)
        compliance = ComplianceManager(registry, tracker, detector)

        await registry.create_tenant("acme_corp", "Acme Corp", TenantTier.FREE)

        # Generate user data
        context = TenantContext(
            tenant_id="acme_corp",
            user_id="john@acme.com",
            permissions={"write", "delete"}
        )

        await tracker.track_ingestion(
            "Email: john@acme.com",
            context=context,
            purpose="User data"
        )

        # Handle erasure request
        erasure = await compliance.handle_right_to_erasure(
            tenant_id="acme_corp",
            data_subject_email="john@acme.com"
        )

        assert erasure.status == "completed"
        assert len(erasure.deletion_steps) > 0
        assert erasure.verification_hash is not None

    @pytest.mark.asyncio
    async def test_hipaa_report(self):
        """Test HIPAA compliance report."""
        registry = TenantRegistry()
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)
        compliance = ComplianceManager(registry, tracker, detector)

        # Create HIPAA-compliant tenant
        await registry.create_tenant(
            "healthcare_corp",
            "Healthcare Corp",
            TenantTier.ENTERPRISE,
            compliance_mode="hipaa",
            encryption_required=True,
            audit_all_access=True
        )

        # Generate report
        report = await compliance.generate_hipaa_report(
            tenant_id="healthcare_corp"
        )

        assert report.tenant_id == "healthcare_corp"
        assert report.technical_safeguards["encryption"]
        assert report.technical_safeguards["audit_controls"]


# ============================================================================
# Integration Test
# ============================================================================

class TestEndToEndIntegration:
    """Test complete end-to-end privacy workflow."""

    @pytest.mark.asyncio
    async def test_complete_privacy_workflow(self):
        """Test complete privacy workflow from ingestion to compliance reporting."""
        # 1. Set up components
        registry = TenantRegistry()
        detector = create_default_detector()
        tracker = PIIFlowTracker(detector)
        encryption = create_encryption_manager()
        isolation = TenantIsolationLayer(registry)
        compliance = ComplianceManager(registry, tracker, detector)

        # 2. Create tenant
        tenant = await registry.create_tenant(
            tenant_id="demo_corp",
            name="Demo Corporation",
            tier=TenantTier.PROFESSIONAL,
            compliance_mode="gdpr",
            encryption_required=True
        )

        # 3. Generate encryption key
        enc_key = encryption.generate_key_for_tenant("demo_corp")

        # 4. Create user context
        context = TenantContext(
            tenant_id="demo_corp",
            user_id="alice@demo.com",
            permissions={"read", "write", "delete"}
        )

        # 5. Validate operation
        await isolation.validate_operation(context, "write")

        # 6. Detect and track PII ingestion
        user_input = "My email is alice@demo.com and phone is 555-123-4567"

        pii_analysis = detector.analyze(user_input)
        assert pii_analysis.has_pii

        ingestion_event = await tracker.track_ingestion(
            text=user_input,
            context=context,
            purpose="User registration"
        )

        # 7. Encrypt sensitive data
        encrypted_email = encryption.encrypt_string(
            "alice@demo.com",
            tenant_id="demo_corp",
            metadata={"field": "email"}
        )

        # 8. Track storage with scoped key
        scoped_key = isolation.scope_key("user_alice", "demo_corp")

        storage_event = await tracker.track_storage(
            data={"email": encrypted_email.to_dict()},
            context=context,
            parent_event_id=ingestion_event.event_id,
            purpose="Store user profile"
        )

        # 9. Generate compliance reports
        gdpr_report = await compliance.generate_gdpr_article30_report(
            tenant_id="demo_corp"
        )

        assert gdpr_report.compliant
        assert len(gdpr_report.security_measures) > 0

        # 10. Handle DSAR
        dsar = await compliance.handle_dsar(
            tenant_id="demo_corp",
            data_subject_email="alice@demo.com"
        )

        assert dsar.status == "completed"

        # 11. Handle right to erasure
        erasure = await compliance.handle_right_to_erasure(
            tenant_id="demo_corp",
            data_subject_email="alice@demo.com"
        )

        assert erasure.status == "completed"

        # 12. Verify deletion was tracked
        events = await tracker.get_events_for_tenant("demo_corp")
        deletion_events = [e for e in events if e.action == PIIAction.DELETED]
        assert len(deletion_events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
