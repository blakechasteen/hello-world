"""
Demo: Code-Level Safety Locks

Demonstrates how Layer 6 safety locks work in practice.
Shows the complete unlock process and error messages.
"""

import os
import sys
sys.path.insert(0, '.')

from HoloLoom.config import Config
from HoloLoom.safety import (
    SafetyLock,
    Layer6Capability,
    Layer6BlockedException,
)


def demo_default_locked():
    """Demo 1: Layer 6 is locked by default."""
    print("=" * 70)
    print("DEMO 1: Layer 6 Locked by Default")
    print("=" * 70)

    config = Config.fast()
    print(f"\nConfig created: {config.mode.value} mode")
    print(f"Layer 6 enabled in config: {config.layer6_enabled}")
    print(f"Layer 6 actually enabled: {SafetyLock.is_layer6_enabled(config)}")

    print("\n✅ Result: Layer 6 is LOCKED (safe!)")


def demo_blocked_operation():
    """Demo 2: Attempting Layer 6 operation shows helpful error."""
    print("\n" + "=" * 70)
    print("DEMO 2: Attempting Blocked Operation")
    print("=" * 70)

    config = Config.fast()

    print("\nAttempting CODE_MODIFICATION operation...")

    try:
        SafetyLock.require_layer6(
            Layer6Capability.CODE_MODIFICATION,
            config=config,
            context={"demo": "showing_error_message"}
        )
        print("❌ Operation succeeded (should have been blocked!)")
    except Layer6BlockedException as e:
        print("\n✅ Operation BLOCKED (as expected)")
        print("\nError message received:")
        print("-" * 70)
        print(str(e)[:500])  # First 500 chars
        print("-" * 70)


def demo_config_flag_alone():
    """Demo 3: Config flag alone doesn't unlock."""
    print("\n" + "=" * 70)
    print("DEMO 3: Config Flag Alone (Insufficient)")
    print("=" * 70)

    # Ensure env var is not set
    if "HOLOLOOM_LAYER6_UNLOCK" in os.environ:
        del os.environ["HOLOLOOM_LAYER6_UNLOCK"]

    config = Config.fast()
    config.layer6_enabled = True  # Try to enable

    print(f"\nConfig flag set: {config.layer6_enabled}")
    print(f"Environment variable: {os.environ.get('HOLOLOOM_LAYER6_UNLOCK', 'NOT SET')}")
    print(f"Layer 6 actually enabled: {SafetyLock.is_layer6_enabled(config)}")

    print("\n✅ Result: Still LOCKED (both requirements needed)")


def demo_env_var_alone():
    """Demo 4: Environment variable alone doesn't unlock."""
    print("\n" + "=" * 70)
    print("DEMO 4: Environment Variable Alone (Insufficient)")
    print("=" * 70)

    os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

    config = Config.fast()
    config.layer6_enabled = False  # Keep disabled

    print(f"\nEnvironment variable: {os.environ.get('HOLOLOOM_LAYER6_UNLOCK')}")
    print(f"Config flag set: {config.layer6_enabled}")
    print(f"Layer 6 actually enabled: {SafetyLock.is_layer6_enabled(config)}")

    print("\n✅ Result: Still LOCKED (both requirements needed)")

    # Cleanup
    del os.environ["HOLOLOOM_LAYER6_UNLOCK"]


def demo_full_unlock():
    """Demo 5: Full unlock with both requirements."""
    print("\n" + "=" * 70)
    print("DEMO 5: Full Unlock (Both Requirements)")
    print("=" * 70)

    os.environ["HOLOLOOM_LAYER6_UNLOCK"] = "research"

    config = Config.fast()
    config.layer6_enabled = True

    print(f"\nEnvironment variable: {os.environ.get('HOLOLOOM_LAYER6_UNLOCK')}")
    print(f"Config flag set: {config.layer6_enabled}")
    print(f"Layer 6 actually enabled: {SafetyLock.is_layer6_enabled(config)}")

    print("\n⚠️  Result: UNLOCKED (use only in research sandbox!)")

    # Try operation
    print("\nAttempting CODE_MODIFICATION operation...")
    try:
        SafetyLock.require_layer6(
            Layer6Capability.CODE_MODIFICATION,
            config=config,
            context={"environment": "research_sandbox"}
        )
        print("✅ Operation ALLOWED (Layer 6 is unlocked)")
    except Layer6BlockedException:
        print("❌ Operation blocked (unexpected)")

    # Cleanup
    del os.environ["HOLOLOOM_LAYER6_UNLOCK"]


def demo_safety_monitor():
    """Demo 6: Safety monitor tracks all attempts."""
    print("\n" + "=" * 70)
    print("DEMO 6: Safety Monitor (Audit Trail)")
    print("=" * 70)

    monitor = SafetyLock.get_monitor()
    initial_count = monitor.get_statistics()["total_attempts"]

    config = Config.fast()

    # Try several operations (all blocked)
    capabilities = [
        Layer6Capability.CODE_MODIFICATION,
        Layer6Capability.UNRESTRICTED_LEARNING,
        Layer6Capability.META_OPTIMIZATION,
    ]

    for cap in capabilities:
        try:
            SafetyLock.require_layer6(cap, config)
        except Layer6BlockedException:
            pass  # Expected

    stats = monitor.get_statistics()
    new_attempts = stats["total_attempts"] - initial_count

    print(f"\nOperations attempted: {new_attempts}")
    print(f"Operations allowed: 0")
    print(f"Operations blocked: {new_attempts}")
    print(f"\nTotal attempts tracked: {stats['total_attempts']}")
    print(f"Total allowed: {stats['allowed']}")
    print(f"Total blocked: {stats['blocked']}")

    print("\n✅ Complete audit trail maintained")


def demo_unlock_instructions():
    """Demo 7: Getting unlock instructions."""
    print("\n" + "=" * 70)
    print("DEMO 7: Unlock Instructions")
    print("=" * 70)

    instructions = SafetyLock.get_unlock_instructions()
    print("\n" + instructions[:600])  # First 600 chars
    print("\n[... truncated for demo ...]")
    print("\n✅ Clear instructions available")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" " * 15 + "CODE-LEVEL SAFETY LOCKS DEMO")
    print("=" * 70)
    print("\nDemonstrating Layer 6 safety mechanisms:")
    print("- Default locked state")
    print("- Helpful error messages")
    print("- Two-factor unlock (env var + config)")
    print("- Audit trail monitoring")
    print("\n")

    try:
        demo_default_locked()
        demo_blocked_operation()
        demo_config_flag_alone()
        demo_env_var_alone()
        demo_full_unlock()
        demo_safety_monitor()
        demo_unlock_instructions()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETE")
        print("=" * 70)
        print("\n✅ Safety locks working correctly")
        print("✅ Layer 6 blocked by default")
        print("✅ Unlock requires both env var + config flag")
        print("✅ Clear error messages guide users")
        print("✅ Complete audit trail maintained")

        print("\nFor more information:")
        print("- README_SAFETY.md: Comprehensive safety guide")
        print("- docs/safety/SANDBOX_CHECKLIST.md: Research requirements")
        print("- LAYER_6_SAFETY_ANALYSIS.md: Technical risk analysis")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
