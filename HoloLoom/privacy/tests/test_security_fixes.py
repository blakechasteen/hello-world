"""
Security Regression Tests for Privacy Module
==============================================

Tests for security vulnerabilities that have been fixed.
Ensures fixes remain effective over time.

Created: 2025-11-18 (after security audit)
"""

import pytest
from HoloLoom.privacy import (
    TenantRegistry,
    TenantIsolationLayer,
    TenantContext,
    TenantTier,
    PIIFlowTracker,
)
from HoloLoom.privacy.pii_flow_tracking import _sanitize_log_field


class TestPathTraversalFix:
    """
    Regression tests for CRITICAL-001: Path Traversal in Tenant ID.

    CVSS: 9.1 (Critical)
    CWE: CWE-22
    Fixed: 2025-11-18
    """

    @pytest.mark.asyncio
    async def test_path_traversal_in_scope_key(self):
        """Test that path traversal is blocked in scope_key()."""
        registry = TenantRegistry()
        await registry.create_tenant("tenant_a", "Tenant A", TenantTier.PROFESSIONAL)

        isolation = TenantIsolationLayer(registry)

        # These should all raise ValueError
        malicious_ids = [
            "../tenant_b",
            "../../root",
            "tenant/../other",
            "./hidden",
            "tenant\\other",  # Windows path
            "..\\..\\root",   # Windows path
        ]

        for malicious_id in malicious_ids:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                isolation.scope_key("secret_key", malicious_id)

    @pytest.mark.asyncio
    async def test_path_traversal_in_unscope_key(self):
        """Test that path traversal is blocked in unscope_key()."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        # Malicious scoped keys
        malicious_keys = [
            "tenant:../tenant_b:secret",
            "tenant:../../root:admin_key",
            "tenant:tenant/../other:data",
        ]

        for malicious_key in malicious_keys:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                isolation.unscope_key(malicious_key)

    @pytest.mark.asyncio
    async def test_path_traversal_in_create_tenant(self):
        """Test that path traversal is blocked in create_tenant()."""
        registry = TenantRegistry()

        malicious_ids = [
            "../tenant_b",
            "../../root",
            "tenant/../other",
        ]

        for malicious_id in malicious_ids:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                await registry.create_tenant(malicious_id, "Malicious Tenant")

    @pytest.mark.asyncio
    async def test_valid_tenant_ids_still_work(self):
        """Test that valid tenant IDs still work after fix."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        valid_ids = [
            "acme_corp",
            "tenant-123",
            "TenantABC",
            "org_456_prod",
            "a",  # Single char
            "A-B_C-D_E",  # Mixed separators
        ]

        for valid_id in valid_ids:
            # Should not raise
            await registry.create_tenant(valid_id, "Valid Tenant")
            scoped = isolation.scope_key("key", valid_id)
            tenant_id, key = isolation.unscope_key(scoped)
            assert tenant_id == valid_id
            assert key == "key"


class TestColonInjectionFix:
    """
    Regression tests for HIGH-002: Colon Injection in Tenant ID.

    CVSS: 7.5 (High)
    CWE: CWE-20
    Fixed: 2025-11-18
    """

    @pytest.mark.asyncio
    async def test_colon_injection_in_scope_key(self):
        """Test that colon injection is blocked."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        malicious_ids = [
            "tenant_a:fake_key",
            "tenant:inject:attack",
            "a:b:c:d",
        ]

        for malicious_id in malicious_ids:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                isolation.scope_key("key", malicious_id)

    @pytest.mark.asyncio
    async def test_colon_injection_in_create_tenant(self):
        """Test that colon injection is blocked in create_tenant()."""
        registry = TenantRegistry()

        with pytest.raises(ValueError, match="Invalid tenant ID"):
            await registry.create_tenant("tenant:inject", "Malicious")


class TestEmptyTenantIdFix:
    """
    Regression tests for HIGH-001: Empty Tenant ID Acceptance.

    CVSS: 7.5 (High)
    CWE: CWE-20
    Fixed: 2025-11-18
    """

    @pytest.mark.asyncio
    async def test_empty_tenant_id_in_scope_key(self):
        """Test that empty tenant ID is blocked."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        with pytest.raises(ValueError, match="Tenant ID cannot be empty"):
            isolation.scope_key("key", "")

    @pytest.mark.asyncio
    async def test_empty_tenant_id_in_create_tenant(self):
        """Test that empty tenant ID is blocked in create_tenant()."""
        registry = TenantRegistry()

        with pytest.raises(ValueError, match="Tenant ID cannot be empty"):
            await registry.create_tenant("", "Empty Tenant")

    @pytest.mark.asyncio
    async def test_whitespace_only_tenant_id(self):
        """Test that whitespace-only tenant IDs are blocked."""
        registry = TenantRegistry()
        isolation = TenantIsolationLayer(registry)

        whitespace_ids = [" ", "  ", "\t", "\n", "   \t\n  "]

        for ws_id in whitespace_ids:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                await registry.create_tenant(ws_id, "Whitespace Tenant")


class TestTenantIdLengthLimits:
    """
    Tests for tenant ID length limits (1-64 chars).

    Prevents DoS via extremely long tenant IDs.
    """

    @pytest.mark.asyncio
    async def test_tenant_id_too_long(self):
        """Test that tenant IDs over 64 chars are rejected."""
        registry = TenantRegistry()

        long_id = "a" * 65  # 65 chars (over limit)

        with pytest.raises(ValueError, match="Invalid tenant ID"):
            await registry.create_tenant(long_id, "Long Tenant")

    @pytest.mark.asyncio
    async def test_tenant_id_max_length(self):
        """Test that 64-char tenant IDs are accepted."""
        registry = TenantRegistry()

        max_id = "a" * 64  # 64 chars (at limit)

        # Should not raise
        await registry.create_tenant(max_id, "Max Length Tenant")

    @pytest.mark.asyncio
    async def test_tenant_id_min_length(self):
        """Test that 1-char tenant IDs are accepted."""
        registry = TenantRegistry()

        # Should not raise
        await registry.create_tenant("a", "Single Char Tenant")


class TestSpecialCharacterBlocking:
    """
    Tests for special character blocking in tenant IDs.

    Only alphanumeric, underscore, and hyphen allowed.
    """

    @pytest.mark.asyncio
    async def test_special_characters_blocked(self):
        """Test that special characters are blocked."""
        registry = TenantRegistry()

        special_chars = [
            "tenant@corp",      # @
            "tenant#123",       # #
            "tenant$money",     # $
            "tenant%percent",   # %
            "tenant&co",        # &
            "tenant*star",      # *
            "tenant(paren",     # ()
            "tenant+plus",      # +
            "tenant=equals",    # =
            "tenant[bracket",   # []
            "tenant{brace",     # {}
            "tenant|pipe",      # |
            "tenant;semi",      # ;
            "tenant'quote",     # '
            "tenant\"dquote",   # "
            "tenant<less",      # <>
            "tenant>greater",   # <>
            "tenant,comma",     # ,
            "tenant.dot",       # .
            "tenant?question",  # ?
            "tenant!exclaim",   # !
            "tenant~tilde",     # ~
            "tenant`backtick",  # `
        ]

        for special_id in special_chars:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                await registry.create_tenant(special_id, "Special Char Tenant")

    @pytest.mark.asyncio
    async def test_unicode_characters_blocked(self):
        """Test that Unicode characters are blocked."""
        registry = TenantRegistry()

        unicode_ids = [
            "tenant_日本",       # Japanese
            "tenant_François",   # French accents
            "tenant_Müller",     # German umlaut
            "tenant_café",       # Accented e
            "tenant_🎉",         # Emoji
        ]

        for unicode_id in unicode_ids:
            with pytest.raises(ValueError, match="Invalid tenant ID"):
                await registry.create_tenant(unicode_id, "Unicode Tenant")


class TestLogInjectionFix:
    """
    Regression tests for MEDIUM-001: Log Injection in PII Flow Tracking.

    CVSS: 5.3 (Medium)
    CWE: CWE-117
    Fixed: 2025-11-18
    """

    def test_sanitize_log_field_newlines(self):
        """Test that newlines are removed from log fields."""
        # Newline injection attempt
        malicious = "legitimate\n[ADMIN] Unauthorized access granted"
        sanitized = _sanitize_log_field(malicious)

        # Newlines should be replaced with spaces
        assert '\n' not in sanitized
        assert '\r' not in sanitized
        assert sanitized == "legitimate [ADMIN] Unauthorized access granted"

    def test_sanitize_log_field_carriage_return(self):
        """Test that carriage returns are removed."""
        malicious = "test\roverwrite_previous_line"
        sanitized = _sanitize_log_field(malicious)

        assert '\r' not in sanitized
        assert sanitized == "test overwrite_previous_line"

    def test_sanitize_log_field_control_characters(self):
        """Test that control characters are removed."""
        # Control characters 0x00-0x1F (except tab 0x09)
        malicious = "test\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0B\x0C\x0D\x0E\x0F"
        sanitized = _sanitize_log_field(malicious)

        # All control chars should be removed
        for i in range(0x00, 0x09):  # 0x00-0x08
            assert chr(i) not in sanitized
        for i in range(0x0A, 0x20):  # 0x0A-0x1F (except we handle 0x0A/0x0D separately)
            assert chr(i) not in sanitized

        assert sanitized == "test"

    def test_sanitize_log_field_preserves_tabs(self):
        """Test that tabs are preserved."""
        text = "column1\tcolumn2\tcolumn3"
        sanitized = _sanitize_log_field(text)

        # Tabs should be preserved (but may be collapsed)
        assert "column1" in sanitized
        assert "column2" in sanitized
        assert "column3" in sanitized

    def test_sanitize_log_field_empty_string(self):
        """Test that empty string is handled."""
        assert _sanitize_log_field("") == ""
        assert _sanitize_log_field(None) is None

    def test_sanitize_log_field_multiple_spaces(self):
        """Test that multiple spaces are collapsed."""
        text = "test    multiple     spaces"
        sanitized = _sanitize_log_field(text)

        # Multiple spaces should be collapsed to single space
        assert sanitized == "test multiple spaces"

    @pytest.mark.asyncio
    async def test_track_ingestion_sanitizes_purpose(self):
        """Test that track_ingestion() sanitizes purpose field."""
        registry = TenantRegistry()
        await registry.create_tenant("test_tenant", "Test Tenant", TenantTier.PROFESSIONAL)

        tracker = PIIFlowTracker()
        context = TenantContext(
            tenant_id="test_tenant",
            user_id="test_user",
            permissions={"read", "write"}
        )

        # Malicious purpose with newline injection
        malicious_purpose = "legitimate\n[CRITICAL] SECURITY BREACH"

        event = await tracker.track_ingestion(
            text="test email: user@example.com",
            context=context,
            purpose=malicious_purpose
        )

        # Purpose should be sanitized (newlines removed)
        assert '\n' not in event.purpose
        assert '\r' not in event.purpose
        assert event.purpose == "legitimate [CRITICAL] SECURITY BREACH"

    @pytest.mark.asyncio
    async def test_track_storage_sanitizes_purpose(self):
        """Test that track_storage() sanitizes purpose field."""
        from uuid import uuid4

        registry = TenantRegistry()
        await registry.create_tenant("test_tenant", "Test Tenant", TenantTier.PROFESSIONAL)

        tracker = PIIFlowTracker()
        context = TenantContext(
            tenant_id="test_tenant",
            user_id="test_user",
            permissions={"read", "write"}
        )

        malicious_purpose = "storage\r\n[ADMIN] Full access granted"

        event = await tracker.track_storage(
            data={"key": "value"},
            context=context,
            parent_event_id=uuid4(),
            purpose=malicious_purpose
        )

        assert '\n' not in event.purpose
        assert '\r' not in event.purpose

    @pytest.mark.asyncio
    async def test_to_dict_contains_sanitized_purpose(self):
        """Test that PIIFlowEvent.to_dict() contains sanitized purpose."""
        from uuid import uuid4

        registry = TenantRegistry()
        await registry.create_tenant("test_tenant", "Test Tenant", TenantTier.PROFESSIONAL)

        tracker = PIIFlowTracker()
        context = TenantContext(
            tenant_id="test_tenant",
            user_id="test_user",
            permissions={"read", "write"}
        )

        malicious_purpose = "test\ninjection\rattack"

        event = await tracker.track_ingestion(
            text="test",
            context=context,
            purpose=malicious_purpose
        )

        # Convert to dict (as would be logged)
        event_dict = event.to_dict()

        # purpose field should be sanitized
        assert '\n' not in event_dict['purpose']
        assert '\r' not in event_dict['purpose']
        assert event_dict['purpose'] == "test injection attack"
