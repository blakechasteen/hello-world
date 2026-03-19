"""
Tests for Phase 6: Security Screen.

Covers:
- default_security_screen pattern matching
- Custom patterns via create_security_screen
- Disabled patterns
- Integration with LiteMemoryBus.store() blocking
"""

import pytest

from hololoom.core.memory.bus import MemoryItem
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.security_screen import (
    SecurityScreenError,
    default_security_screen,
    create_security_screen,
)


# =============================================================================
# Pattern Detection
# =============================================================================

class TestDefaultScreen:

    def test_normal_content_passes(self):
        assert default_security_screen("Garlic is a wonderful ingredient") is True

    def test_aws_key_blocked(self):
        assert default_security_screen("my key is AKIAIOSFODNN7EXAMPLE") is False

    def test_openai_key_blocked(self):
        assert default_security_screen("sk-abcdefghijklmnopqrstuvwxyz1234") is False

    def test_github_token_blocked(self):
        token = "ghp_" + "A" * 36
        assert default_security_screen(f"token: {token}") is False

    def test_generic_secret_blocked(self):
        assert default_security_screen('password = "mysecretpassword123"') is False
        assert default_security_screen("secret: 'verylongsecretvalue'") is False

    def test_ssn_blocked(self):
        assert default_security_screen("SSN: 123-45-6789") is False

    def test_private_key_blocked(self):
        assert default_security_screen("-----BEGIN RSA PRIVATE KEY-----") is False
        assert default_security_screen("-----BEGIN PRIVATE KEY-----") is False

    def test_short_password_passes(self):
        """Generic secret pattern requires 8+ chars in value."""
        assert default_security_screen('password = "short"') is True

    def test_embedded_in_content(self):
        """Secrets embedded in larger content are still caught."""
        content = (
            "During the meeting, someone accidentally shared "
            "AKIAIOSFODNN7EXAMPLE in the chat."
        )
        assert default_security_screen(content) is False


# =============================================================================
# Custom Screens
# =============================================================================

class TestCustomScreen:

    def test_extra_patterns(self):
        screen = create_security_screen(
            extra_patterns=[("credit_card", r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "Credit Card")],
        )
        assert screen("card: 4111-1111-1111-1111") is False
        assert screen("normal text") is True

    def test_disabled_patterns(self):
        screen = create_security_screen(disabled_patterns=["ssn"])
        # SSN should now pass
        assert screen("SSN: 123-45-6789") is True
        # But AWS key should still be blocked
        assert screen("AKIAIOSFODNN7EXAMPLE") is False

    def test_disabled_and_extra(self):
        screen = create_security_screen(
            extra_patterns=[("custom", r"CUSTOM_SECRET_\w+", "Custom Secret")],
            disabled_patterns=["aws_key"],
        )
        assert screen("AKIAIOSFODNN7EXAMPLE") is True  # Disabled
        assert screen("CUSTOM_SECRET_abc123") is False  # Added


# =============================================================================
# SecurityScreenError
# =============================================================================

class TestSecurityScreenError:

    def test_is_value_error(self):
        err = SecurityScreenError("aws_key", "AWS Access Key ID")
        assert isinstance(err, ValueError)
        assert "aws_key" in str(err)
        assert err.pattern_name == "aws_key"


# =============================================================================
# Bus Integration
# =============================================================================

class TestBusIntegration:

    @pytest.mark.asyncio
    async def test_blocks_sensitive_content(self):
        bus = LiteMemoryBus(security_screen=default_security_screen)
        await bus.initialize()

        with pytest.raises(SecurityScreenError):
            await bus.store(MemoryItem(
                content="my aws key is AKIAIOSFODNN7EXAMPLE",
                memory_type="factual",
            ))
        # Nothing stored
        assert len(bus._items) == 0

    @pytest.mark.asyncio
    async def test_allows_safe_content(self):
        bus = LiteMemoryBus(security_screen=default_security_screen)
        await bus.initialize()

        item_id = await bus.store(MemoryItem(
            content="Garlic is a key ingredient in many dishes",
            memory_type="factual",
        ))
        assert item_id in bus._items

    @pytest.mark.asyncio
    async def test_no_screen_allows_everything(self):
        """Without security_screen, anything goes (backward compat)."""
        bus = LiteMemoryBus()
        await bus.initialize()

        item_id = await bus.store(MemoryItem(
            content="AKIAIOSFODNN7EXAMPLE",
            memory_type="factual",
        ))
        assert item_id in bus._items

    @pytest.mark.asyncio
    async def test_custom_screen(self):
        screen = create_security_screen(
            extra_patterns=[("internal_id", r"INT-\d{8}", "Internal ID")],
        )
        bus = LiteMemoryBus(security_screen=screen)
        await bus.initialize()

        with pytest.raises(SecurityScreenError):
            await bus.store(MemoryItem(
                content="Reference: INT-12345678",
                memory_type="factual",
            ))
