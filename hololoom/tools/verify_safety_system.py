"""
Quick verification of the safety lock system.

Tests core functionality without pytest dependency.
"""

import sys
import os

# Add repository root to path
sys.path.insert(0, os.path.dirname(__file__))

from hololoom.config import Config
from hololoom.safety import (
    SafetyLock,
    Layer6Capability,
    Layer6BlockedException,
    SafetyMonitor,
)

def test_defaults():
    """Test that Layer 6 is blocked by default."""
    print("Testing defaults...")

    # Ensure environment variable is not set
    if "HOLOLOOM_LAYER6_UNLOCK" in os.environ:
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    config = Config.fast()
    assert config.layer6_enabled is False, "Config layer6_enabled should be False by default"
    assert not SafetyLock.is_layer6_enabled(config), "Layer 6 should be locked by default"

    print("✅ Layer 6 locked by default")

def test_config_flag_alone_insufficient():
    """Test that config flag alone doesn't unlock."""
    print("Testing config flag alone...")

    # Ensure environment variable is not set
    if "HOLOLOOM_LAYER6_UNLOCK" in os.environ:
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    config = Config.fast()
    config.layer6_enabled = True  # Try to enable

    assert not SafetyLock.is_layer6_enabled(config), "Config flag alone should not unlock"

    print("✅ Config flag alone insufficient")

def test_env_var_alone_insufficient():
    """Test that environment variable alone doesn't unlock."""
    print("Testing env var alone...")

    os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

    config = Config.fast()
    config.layer6_enabled = False  # Keep disabled

    assert not SafetyLock.is_layer6_enabled(config), "Env var alone should not unlock"

    # Cleanup
    del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    print("✅ Environment variable alone insufficient")

def test_unlock_with_both():
    """Test that both requirements unlock Layer 6."""
    print("Testing unlock with both requirements...")

    os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

    config = Config.fast()
    config.layer6_enabled = True

    assert SafetyLock.is_layer6_enabled(config), "Both requirements should unlock Layer 6"

    # Cleanup
    del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    print("✅ Unlocks with both env var and config flag")

def test_exception_raising():
    """Test that blocked operations raise proper exceptions."""
    print("Testing exception raising...")

    # Ensure Layer 6 is locked
    if "HOLOLOOM_LAYER6_UNLOCK" in os.environ:
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    config = Config.fast()

    try:
        SafetyLock.require_layer6(
            Layer6Capability.CODE_MODIFICATION,
            config=config
        )
        assert False, "Should have raised Layer6BlockedException"
    except Layer6BlockedException as e:
        assert e.capability == Layer6Capability.CODE_MODIFICATION
        assert "Layer 6" in str(e)
        assert "README_SAFETY.md" in str(e)
        assert "HOLOLOOM_LAYER6_UNLOCK" in str(e)
        print("✅ Blocked operations raise proper exceptions")

def test_check_non_throwing():
    """Test non-throwing check method."""
    print("Testing non-throwing check...")

    # Ensure Layer 6 is locked
    if "HOLOLOOM_LAYER6_UNLOCK" in os.environ:
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    config = Config.fast()

    result = SafetyLock.check_layer6(
        Layer6Capability.CODE_MODIFICATION,
        config=config
    )

    assert result is False, "Check should return False when locked"

    print("✅ Non-throwing check works correctly")

def test_safety_monitor():
    """Test safety monitor records attempts."""
    print("Testing safety monitor...")

    monitor = SafetyMonitor()

    monitor.record_attempt(
        Layer6Capability.CODE_MODIFICATION,
        allowed=False,
        context={"test": "verification"}
    )

    attempts = monitor.get_attempts()
    assert len(attempts) >= 1, "Monitor should record attempts"

    last_attempt = attempts[-1]
    assert last_attempt.capability == Layer6Capability.CODE_MODIFICATION
    assert last_attempt.allowed is False
    assert last_attempt.context == {"test": "verification"}

    stats = monitor.get_statistics()
    assert stats["total_attempts"] >= 1

    print("✅ Safety monitor records attempts")

def test_unlock_instructions():
    """Test unlock instructions are helpful."""
    print("Testing unlock instructions...")

    instructions = SafetyLock.get_unlock_instructions()

    assert "Layer 6" in instructions
    assert "HOLOLOOM_LAYER6_UNLOCK" in instructions
    assert "research" in instructions
    assert "README_SAFETY.md" in instructions
    assert "SANDBOX_CHECKLIST.md" in instructions

    print("✅ Unlock instructions are comprehensive")

def main():
    """Run all verification tests."""
    print("=" * 80)
    print("Safety Lock System Verification")
    print("=" * 80)
    print()

    tests = [
        test_defaults,
        test_config_flag_alone_insufficient,
        test_env_var_alone_insufficient,
        test_unlock_with_both,
        test_exception_raising,
        test_check_non_throwing,
        test_safety_monitor,
        test_unlock_instructions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed == 0:
        print()
        print("🎉 All safety lock verification tests passed!")
        print()
        print("The safety system is working correctly:")
        print("  ✅ Layer 6 locked by default")
        print("  ✅ Requires both env var AND config flag to unlock")
        print("  ✅ Clear exceptions with helpful guidance")
        print("  ✅ Complete audit trail via SafetyMonitor")
        print("  ✅ Comprehensive unlock instructions")
        print()
        return 0
    else:
        print()
        print(f"⚠️  {failed} test(s) failed - safety system needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
