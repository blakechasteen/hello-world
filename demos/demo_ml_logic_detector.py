#!/usr/bin/env python3
"""
Demo: ML Logic Error Detector
==============================

Demonstrates Trough's ML-based logic error detection.

Shows detection of:
- Infinite loops
- Unreachable code
- Division by zero
- Null dereference
- Logic contradictions
- Missing returns
- Constant conditions
- Array out of bounds

Usage:
    python demos/demo_ml_logic_detector.py
"""

import asyncio
from HoloLoom.agentic.ml_logic_detector import MLLogicDetector
from HoloLoom.agentic.codebase_ingestion import Language


# Test cases with known logic errors
TEST_CASES = {
    "division_by_zero": """
def divide(a, b):
    return a / b  # b might be zero!
""",

    "infinite_loop": """
def process_forever():
    while True:
        work()
        # No break or return - infinite loop!
""",

    "unreachable_code": """
def validate():
    return True
    print("This will never run")  # Unreachable!
""",

    "null_dereference": """
def get_name(user_id):
    user = find_user(user_id)
    if not user:
        user = None
    return user.name  # user might be None!
""",

    "logic_contradiction": """
def check_access(user):
    if user.is_admin and not user.is_admin:
        grant_access()  # Never executes!
""",

    "missing_return": """
def calculate_total(items):
    total = sum(item.price for item in items)
    # Missing: return total
""",

    "constant_condition": """
def debug_mode():
    if True:
        log_debug()  # Why the if?
""",

    "array_out_of_bounds": """
items = [1, 2, 3]
last = items[len(items)]  # IndexError!
""",

    "multiple_errors": """
def process_data(values):
    # Error 1: No return statement
    total = 0

    # Error 2: Infinite loop
    while True:
        for value in values:
            # Error 3: Division by zero
            avg = total / len(values)
            total += value

    # Error 4: Unreachable code
    print("Done")
"""
}


async def demo_logic_detection():
    """Demonstrate ML logic error detection."""
    print("=" * 80)
    print("Trough ML Logic Error Detector - Demo")
    print("=" * 80)
    print()

    detector = MLLogicDetector()

    for name, code in TEST_CASES.items():
        print(f"\n{'─' * 80}")
        print(f"Test Case: {name.upper()}")
        print(f"{'─' * 80}")
        print("\nCode:")
        print(code)
        print()

        # Detect errors
        errors = await detector.detect(code, Language.PYTHON)

        if errors:
            print(f"✗ Found {len(errors)} error(s):\n")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error.error_type.value.upper()}")
                print(f"   Line {error.line_number}, Column {error.column}")
                print(f"   Description: {error.description}")
                print(f"   Confidence: {error.confidence:.2f}")
                if error.proof:
                    print(f"   Proof: {error.proof}")
                if error.fix_suggestion:
                    print(f"   Suggested Fix: {error.fix_suggestion}")
                print()
        else:
            print("✓ No errors detected\n")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nDetection algorithms demonstrated:")
    print("  1. ✅ Infinite loop detection (Tarjan's SCC)")
    print("  2. ✅ Unreachable code detection (BFS reachability)")
    print("  3. ✅ Division by zero detection (abstract interpretation)")
    print("  4. ✅ Null dereference detection (data flow analysis)")
    print("  5. ✅ Logic contradiction detection (boolean analysis)")
    print("  6. ✅ Missing return detection (AST analysis)")
    print("  7. ✅ Constant condition detection (constant folding)")
    print("  8. ✅ Array out of bounds detection (bounds checking)")
    print("\nPhase 1: 9/15 algorithms complete")
    print("Phase 2: 6/15 algorithms planned (memory leaks, race conditions, etc.)")
    print()


async def demo_complex_case():
    """Demonstrate detection of multiple errors in complex code."""
    print("\n" + "=" * 80)
    print("Complex Case: Multiple Errors")
    print("=" * 80)
    print()

    complex_code = """
def process_users(user_ids):
    # Error 1: Missing return statement
    results = []

    # Error 2: Infinite loop (no break)
    while True:
        for user_id in user_ids:
            # Error 3: Division by zero potential
            avg_age = total_age / len(user_ids)

            # Error 4: Null dereference
            user = find_user(user_id)
            if not user:
                user = None
            name = user.name  # Crash if user is None

            # Error 5: Array out of bounds
            first = user_ids[len(user_ids)]

            results.append(name)

    # Error 6: Unreachable code
    print("Done processing")
    return results
"""

    print("Code:")
    print(complex_code)
    print()

    detector = MLLogicDetector()
    errors = await detector.detect(complex_code, Language.PYTHON)

    print(f"✗ Found {len(errors)} error(s):\n")
    for i, error in enumerate(errors, 1):
        print(f"{i}. [{error.error_type.value.upper()}] Line {error.line_number}")
        print(f"   {error.description}")
        print(f"   Confidence: {error.confidence:.2f}")
        if error.proof:
            print(f"   Proof: {error.proof}")
        print()

    # Get summary
    summary = await detector.get_summary(errors)
    print("\nSummary:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  By type: {summary['by_type']}")
    print(f"  High confidence (>0.8): {len(summary['high_confidence'])}")
    print(f"  Proven errors: {len(summary['proven'])}")
    print()


async def demo_javascript_detection():
    """Demonstrate JavaScript/TypeScript error detection."""
    print("\n" + "=" * 80)
    print("JavaScript/TypeScript Detection")
    print("=" * 80)
    print()

    js_code = """
function processData() {
    let status = "active";

    // Error 1: Assignment instead of comparison
    if (status = "inactive") {
        stop();
    }

    // Error 2: Infinite loop
    while (true) {
        fetchData();
        // No break or return
    }
}
"""

    print("Code:")
    print(js_code)
    print()

    detector = MLLogicDetector()
    errors = await detector.detect(js_code, Language.JAVASCRIPT)

    print(f"✗ Found {len(errors)} error(s):\n")
    for i, error in enumerate(errors, 1):
        print(f"{i}. [{error.error_type.value.upper()}] Line {error.line_number}")
        print(f"   {error.description}")
        print(f"   Confidence: {error.confidence:.2f}")
        print(f"   Fix: {error.fix_suggestion}")
        print()


async def main():
    """Run all demos."""
    await demo_logic_detection()
    await demo_complex_case()
    await demo_javascript_detection()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Start the server: uvicorn HoloLoom.apps.server.agentic_api:app --reload")
    print("  2. Test endpoint: curl -X POST http://localhost:8000/detect/logic \\")
    print("                    -H 'Content-Type: application/json' \\")
    print("                    -d '{\"code\": \"def divide(a, b): return a / b\", \"language\": \"python\"}'")
    print("  3. Use in VS Code via Trough extension (Pig Out!)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
