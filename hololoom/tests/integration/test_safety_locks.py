"""
Comprehensive Safety Lock Tests

Tests for Layer 6 code-level safety locks to ensure:
1. Layer 6 is blocked by default
2. Unlock requires BOTH environment variable AND config flag
3. All Layer 6 capabilities are properly guarded
4. Clear error messages guide users
5. Audit trail logs all attempts
"""

import os
import pytest
from hololoom.config import Config
from hololoom.safety import (
    SafetyLock,
    Layer6Capability,
    Layer6BlockedException,
    SafetyMonitor,
)


class TestSafetyLockDefaults:
    """Test that Layer 6 is blocked by default."""

    def test_layer6_disabled_by_default(self):
        """Layer 6 should be disabled with default config."""
        config = Config.fast()
        assert config.layer6_enabled is False
        assert not SafetyLock.is_layer6_enabled(config)

    def test_layer6_disabled_without_env_var(self):
        """Layer 6 should be disabled even if config flag is True."""
        # Ensure environment variable is not set
        if "HOLOLOOM_LAYER6_UNLOCK" in os.environ:
            del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

        config = Config.fast()
        config.layer6_enabled = True  # Try to enable
        assert not SafetyLock.is_layer6_enabled(config)  # Still blocked!

    def test_layer6_disabled_without_config_flag(self):
        """Layer 6 should be disabled even if env var is set."""
        os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

        config = Config.fast()
        config.layer6_enabled = False  # Keep disabled
        assert not SafetyLock.is_layer6_enabled(config)  # Still blocked!

        # Cleanup
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]


class TestSafetyLockUnlock:
    """Test Layer 6 unlock mechanism."""

    def test_unlock_with_both_requirements(self):
        """Layer 6 should unlock with both env var and config flag."""
        os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

        config = Config.fast()
        config.layer6_enabled = True
        assert SafetyLock.is_layer6_enabled(config)  # Unlocked!

        # Cleanup
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    def test_wrong_env_var_value(self):
        """Wrong env var value should not unlock."""
        os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "wrong_value"

        config = Config.fast()
        config.layer6_enabled = True
        assert not SafetyLock.is_layer6_enabled(config)  # Still blocked!

        # Cleanup
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]


class TestLayer6BlockedException:
    """Test Layer 6 exception behavior."""

    def test_exception_message_quality(self):
        """Exception should have helpful error message."""
        exc = Layer6BlockedException(Layer6Capability.CODE_MODIFICATION)
        message = str(exc)

        # Check for key information in message
        assert "Layer 6" in message
        assert "code_modification" in message
        assert "README_SAFETY.md" in message
        assert "HOLOLOOM_LAYER6_UNLOCK" in message
        assert "research" in message
        assert "Air-gapped" in message or "environment" in message

    def test_exception_includes_capability(self):
        """Exception should include capability that was blocked."""
        exc = Layer6BlockedException(Layer6Capability.UNRESTRICTED_LEARNING)
        assert exc.capability == Layer6Capability.UNRESTRICTED_LEARNING
        assert "unrestricted_learning" in str(exc)

    def test_custom_exception_message(self):
        """Should support custom exception messages."""
        custom_msg = "Custom error message"
        exc = Layer6BlockedException(
            Layer6Capability.CODE_MODIFICATION,
            message=custom_msg
        )
        assert custom_msg in str(exc)


class TestSafetyLockRequire:
    """Test SafetyLock.require_layer6() method."""

    def test_require_raises_when_locked(self):
        """Should raise exception when Layer 6 is locked."""
        config = Config.fast()

        with pytest.raises(Layer6BlockedException) as exc_info:
            SafetyLock.require_layer6(
                Layer6Capability.CODE_MODIFICATION,
                config=config
            )

        assert exc_info.value.capability == Layer6Capability.CODE_MODIFICATION

    def test_require_succeeds_when_unlocked(self):
        """Should not raise when Layer 6 is unlocked."""
        os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"
        config = Config.fast()
        config.layer6_enabled = True

        # Should not raise
        SafetyLock.require_layer6(
            Layer6Capability.CODE_MODIFICATION,
            config=config
        )

        # Cleanup
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    def test_require_with_context(self):
        """Should accept context for logging."""
        config = Config.fast()
        context = {"caller": "test", "reason": "testing"}

        with pytest.raises(Layer6BlockedException):
            SafetyLock.require_layer6(
                Layer6Capability.CODE_MODIFICATION,
                config=config,
                context=context
            )


class TestSafetyLockCheck:
    """Test SafetyLock.check_layer6() method (non-throwing)."""

    def test_check_returns_false_when_locked(self):
        """Should return False when Layer 6 is locked."""
        config = Config.fast()

        result = SafetyLock.check_layer6(
            Layer6Capability.CODE_MODIFICATION,
            config=config
        )

        assert result is False

    def test_check_returns_true_when_unlocked(self):
        """Should return True when Layer 6 is unlocked."""
        os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"
        config = Config.fast()
        config.layer6_enabled = True

        result = SafetyLock.check_layer6(
            Layer6Capability.CODE_MODIFICATION,
            config=config
        )

        assert result is True

        # Cleanup
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]


class TestSafetyMonitor:
    """Test SafetyMonitor audit trail."""

    def test_monitor_records_attempts(self):
        """Monitor should record all Layer 6 attempts."""
        monitor = SafetyMonitor()

        monitor.record_attempt(
            Layer6Capability.CODE_MODIFICATION,
            allowed=False,
            context={"test": "value"}
        )

        attempts = monitor.get_attempts()
        assert len(attempts) == 1
        assert attempts[0].capability == Layer6Capability.CODE_MODIFICATION
        assert attempts[0].allowed is False
        assert attempts[0].context == {"test": "value"}

    def test_monitor_filters_by_capability(self):
        """Monitor should filter attempts by capability."""
        monitor = SafetyMonitor()

        monitor.record_attempt(Layer6Capability.CODE_MODIFICATION, False)
        monitor.record_attempt(Layer6Capability.UNRESTRICTED_LEARNING, False)
        monitor.record_attempt(Layer6Capability.CODE_MODIFICATION, True)

        code_mod_attempts = monitor.get_attempts(
            capability=Layer6Capability.CODE_MODIFICATION
        )
        assert len(code_mod_attempts) == 2

    def test_monitor_filters_by_allowed(self):
        """Monitor should filter attempts by allowed status."""
        monitor = SafetyMonitor()

        monitor.record_attempt(Layer6Capability.CODE_MODIFICATION, False)
        monitor.record_attempt(Layer6Capability.UNRESTRICTED_LEARNING, True)
        monitor.record_attempt(Layer6Capability.CODE_MODIFICATION, True)

        blocked_attempts = monitor.get_attempts(allowed=False)
        assert len(blocked_attempts) == 1

        allowed_attempts = monitor.get_attempts(allowed=True)
        assert len(allowed_attempts) == 2

    def test_monitor_statistics(self):
        """Monitor should provide summary statistics."""
        monitor = SafetyMonitor()

        monitor.record_attempt(Layer6Capability.CODE_MODIFICATION, False)
        monitor.record_attempt(Layer6Capability.CODE_MODIFICATION, False)
        monitor.record_attempt(Layer6Capability.UNRESTRICTED_LEARNING, True)

        stats = monitor.get_statistics()
        assert stats["total_attempts"] == 3
        assert stats["allowed"] == 1
        assert stats["blocked"] == 2

        # Check by-capability stats
        code_mod_stats = stats["by_capability"]["code_modification"]
        assert code_mod_stats["total"] == 2
        assert code_mod_stats["blocked"] == 2
        assert code_mod_stats["allowed"] == 0


class TestLayer6Capabilities:
    """Test all Layer 6 capability definitions."""

    def test_all_capabilities_defined(self):
        """All Layer 6 capabilities should be defined."""
        expected_capabilities = [
            "CODE_MODIFICATION",
            "MODULE_INJECTION",
            "ARCHITECTURE_SEARCH",
            "UNRESTRICTED_LEARNING",
            "OBJECTIVE_MODIFICATION",
            "CONSTRAINT_REMOVAL",
            "AUTONOMOUS_MODIFICATION",
            "RECURSIVE_IMPROVEMENT",
            "META_OPTIMIZATION",
        ]

        for cap_name in expected_capabilities:
            assert hasattr(Layer6Capability, cap_name)

    def test_capability_values(self):
        """Capabilities should have meaningful string values."""
        assert Layer6Capability.CODE_MODIFICATION.value == "code_modification"
        assert Layer6Capability.UNRESTRICTED_LEARNING.value == "unrestricted_learning"


class TestUnlockInstructions:
    """Test unlock instruction generation."""

    def test_get_unlock_instructions(self):
        """Should provide clear unlock instructions."""
        instructions = SafetyLock.get_unlock_instructions()

        assert "Layer 6" in instructions
        assert "HOLOLOOM_LAYER6_UNLOCK" in instructions
        assert "research" in instructions
        assert "README_SAFETY.md" in instructions
        assert "SANDBOX_CHECKLIST.md" in instructions
        assert "Air-gapped" in instructions or "environment" in instructions


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_production_scenario(self):
        """Production config should have Layer 6 locked."""
        config = Config.fused()  # Production config

        # Verify Layer 6 is locked
        assert not SafetyLock.is_layer6_enabled(config)

        # Attempt Layer 6 operation should fail
        with pytest.raises(Layer6BlockedException):
            SafetyLock.require_layer6(
                Layer6Capability.CODE_MODIFICATION,
                config=config,
                context={"environment": "production"}
            )

    def test_research_scenario(self):
        """Research config with unlock should work."""
        os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

        config = Config.fast()
        config.layer6_enabled = True

        # Verify Layer 6 is unlocked
        assert SafetyLock.is_layer6_enabled(config)

        # Layer 6 operations should succeed
        SafetyLock.require_layer6(
            Layer6Capability.META_OPTIMIZATION,
            config=config,
            context={"environment": "research_sandbox"}
        )

        # Cleanup
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    def test_developer_scenario(self):
        """Developer without unlock should get helpful errors."""
        config = Config.fast()

        try:
            SafetyLock.require_layer6(
                Layer6Capability.CODE_MODIFICATION,
                config=config
            )
        except Layer6BlockedException as e:
            # Error should mention:
            # 1. What was blocked
            # 2. How to unlock
            # 3. Safety requirements
            # 4. Where to get more info
            message = str(e)
            assert "code_modification" in message
            assert "HOLOLOOM_LAYER6_UNLOCK" in message
            assert "README_SAFETY.md" in message
            assert "Air-gapped" in message or "environment" in message


class TestConvenienceFunctions:
    """Test convenience function imports."""

    def test_is_layer6_enabled_function(self):
        """is_layer6_enabled() function should work."""
        from hololoom.safety.locks import is_layer6_enabled

        config = Config.fast()
        assert not is_layer6_enabled(config)

    def test_require_layer6_function(self):
        """require_layer6() function should work."""
        from hololoom.safety.locks import require_layer6

        config = Config.fast()
        with pytest.raises(Layer6BlockedException):
            require_layer6(Layer6Capability.CODE_MODIFICATION, config)

    def test_check_layer6_function(self):
        """check_layer6() function should work."""
        from hololoom.safety.locks import check_layer6

        config = Config.fast()
        assert not check_layer6(Layer6Capability.CODE_MODIFICATION, config)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
