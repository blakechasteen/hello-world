"""Code Execution Ability - Usage Examples.

This file demonstrates various ways to use the Code Execution ability.

Run any example with:
    python -m HoloLoom.departments.proto.abilities.core.examples <example_name>

Status: Production ready (2025-12-03)
"""

import asyncio
import sys
from code_execution import CodeExecutionAbility, CodeExecutionConfig
from ..protocol import AbilityContext, AbilityTrustLevel


async def example_simple_print():
    """Example 1: Simple print statement."""
    print("\n=== Example 1: Simple Print ===\n")

    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="example_1",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    result = await ability.execute(
        {"code": "print('Hello, World!')"},
        context
    )

    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Duration: {result.duration_ms:.1f}ms")


async def example_arithmetic():
    """Example 2: Arithmetic operations."""
    print("\n=== Example 2: Arithmetic ===\n")

    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="example_2",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    code = """
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)
print(f"Total: {total}, Average: {average}")
"""

    result = await ability.execute({"code": code}, context)

    print(f"Success: {result.success}")
    print(f"Output: {result.output}")


async def example_error_handling():
    """Example 3: Error handling."""
    print("\n=== Example 3: Error Handling ===\n")

    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="example_3",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    code = """
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")
    result = 0
print(f"Result: {result}")
"""

    result = await ability.execute({"code": code}, context)

    print(f"Success: {result.success}")
    print(f"Output: {result.output}")


async def example_timeout():
    """Example 4: Timeout protection."""
    print("\n=== Example 4: Timeout Protection ===\n")

    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="example_4",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    code = """
import time
print("Starting long operation...")
time.sleep(10)
print("This will not print due to timeout")
"""

    result = await ability.execute(
        {"code": code, "timeout": 1.0},
        context
    )

    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print(f"Timeout: {result.metadata.get('timeout')}")


async def example_custom_config():
    """Example 5: Custom configuration."""
    print("\n=== Example 5: Custom Configuration ===\n")

    config = CodeExecutionConfig(
        max_timeout=60.0,
        max_output_length=5_000,
        enable_sandbox=True
    )

    ability = CodeExecutionAbility(config)
    context = AbilityContext(
        session_id="example_5",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    code = """
# Generate some output
for i in range(100):
    print(f"Line {i}")
"""

    result = await ability.execute({"code": code}, context)

    print(f"Success: {result.success}")
    print(f"Output length: {len(result.output)}")
    print(f"Truncated: {result.metadata.get('truncated')}")
    print(f"First 200 chars: {result.output[:200]}")


async def example_full_workflow():
    """Example 6: Complete workflow with all steps."""
    print("\n=== Example 6: Full Workflow ===\n")

    # Setup
    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="example_6",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    # Preflight check
    print("1. Preflight check...")
    preflight = await ability.preflight(context)
    print(f"   Can execute: {preflight.can_execute}")
    if preflight.warnings:
        for warning in preflight.warnings:
            print(f"   Warning: {warning}")

    if not preflight.can_execute:
        print("   Execution not allowed")
        return

    # Execute
    print("2. Executing code...")
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"fibonacci(10) = {result}")
"""

    result = await ability.execute({"code": code}, context)
    print(f"   Success: {result.success}")
    print(f"   Duration: {result.duration_ms:.1f}ms")

    # Verify
    print("3. Verifying output...")
    verification = await ability.verify(result)
    print(f"   Verified: {verification.verified}")
    if not verification.verified:
        for issue in verification.issues:
            print(f"   Issue: {issue}")

    # Display result
    print("4. Result:")
    print(f"   Output: {result.output}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Return code: {result.metadata.get('return_code')}")


async def example_statistics():
    """Example 7: Statistics tracking."""
    print("\n=== Example 7: Statistics Tracking ===\n")

    ability = CodeExecutionAbility()
    context = AbilityContext(
        session_id="example_7",
        user_confirmed=True,
        trust_level=AbilityTrustLevel.VERIFIED
    )

    # Execute multiple times
    for i in range(3):
        code = f"print(f'Execution {i+1}')"
        result = await ability.execute({"code": code}, context)
        print(f"Execution {i+1}: {result.output.strip()}")

    # Show statistics
    print(f"\nStatistics:")
    print(f"  Total executions: {ability.execution_count}")
    print(f"  Total duration: {ability.total_duration_ms:.1f}ms")
    print(f"  Last execution: {ability.last_execution}")
    if ability.execution_count > 0:
        avg_duration = ability.total_duration_ms / ability.execution_count
        print(f"  Avg duration: {avg_duration:.1f}ms")


async def example_preflight_denied():
    """Example 8: Preflight rejection."""
    print("\n=== Example 8: Preflight Rejection ===\n")

    ability = CodeExecutionAbility()

    # Context without user confirmation
    context = AbilityContext(
        session_id="example_8",
        user_confirmed=False,  # Not confirmed
        trust_level=AbilityTrustLevel.VERIFIED
    )

    print("Checking preflight WITHOUT user confirmation...")
    preflight = await ability.preflight(context)

    print(f"Can execute: {preflight.can_execute}")
    print(f"Reason: {preflight.reason}")


async def main():
    """Run all examples or specific example."""
    examples = {
        "1": ("Simple Print", example_simple_print),
        "2": ("Arithmetic", example_arithmetic),
        "3": ("Error Handling", example_error_handling),
        "4": ("Timeout", example_timeout),
        "5": ("Custom Config", example_custom_config),
        "6": ("Full Workflow", example_full_workflow),
        "7": ("Statistics", example_statistics),
        "8": ("Preflight Denial", example_preflight_denied),
    }

    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key in examples:
            title, func = examples[key]
            print(f"\nRunning: Example {key}: {title}")
            print("=" * 50)
            await func()
        else:
            print(f"Unknown example: {key}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        # Run all examples
        for key in sorted(examples.keys()):
            title, func = examples[key]
            try:
                await func()
            except Exception as e:
                print(f"Example {key} failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
