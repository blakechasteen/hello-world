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
)


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
