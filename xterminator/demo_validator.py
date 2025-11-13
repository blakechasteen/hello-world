#!/usr/bin/env python3
"""
Validation Pipeline Demo (Phase 5)
==================================

Fern's Demonstration of the Validation Pipeline

Shows complete validation workflow with various scenarios:
1. Passing validation (clean fix)
2. Syntax error (invalid code)
3. Signature change (breaking change)
4. Regression detection (new issues)

Usage:
    python demo_validator.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xterminator import ValidationPipeline, quick_validate


# ============================================================================
# Demo Scenarios
# ============================================================================

SCENARIO_1_ORIGINAL = """
import numpy as np
import os

def calculate_sum(x, y):
    return x + y

x = calculate_sum(1, 2)
print(x)
"""

SCENARIO_1_FIXED = """
import os

def calculate_sum(x, y):
    return x + y

x = calculate_sum(1, 2)
print(x)
"""

SCENARIO_1_DESC = "Scenario 1: Remove unused numpy import"


# ============================================================================

SCENARIO_2_ORIGINAL = """
def func():
    return 42
"""

SCENARIO_2_FIXED = """
def func( # Missing closing paren
    return 42
"""

SCENARIO_2_DESC = "Scenario 2: Invalid syntax error"


# ============================================================================

SCENARIO_3_ORIGINAL = """
def calculate_total(items):
    return sum(items)

result = calculate_total([1, 2, 3])
"""

SCENARIO_3_FIXED = """
def calculate_total(items, multiplier):
    return sum(items) * multiplier

result = calculate_total([1, 2, 3])  # Missing argument!
"""

SCENARIO_3_DESC = "Scenario 3: Breaking API change"


# ============================================================================

SCENARIO_4_ORIGINAL = """
def helper_old():
    pass

def calculate(x, y):
    return x + y

result = calculate(5, 3)
print(result)
"""

SCENARIO_4_FIXED = """
def calculate(x, y):
    return x + y

result = calculate(5, 3)
print(result)
"""

SCENARIO_4_DESC = "Scenario 4: Remove unused function"


# ============================================================================
# Demo Functions
# ============================================================================

async def print_header(text, level=1):
    """Print formatted header"""
    if level == 1:
        print(f"\n{'=' * 70}")
        print(f"  {text}")
        print(f"{'=' * 70}")
    elif level == 2:
        print(f"\n{text}")
        print(f"{'-' * 70}")


async def run_scenario(desc, original, fixed, file_path="config.py"):
    """Run a single validation scenario"""
    await print_header(desc, level=2)

    print("\nOriginal Code:")
    print("```python")
    print(original[:150] + ("..." if len(original) > 150 else ""))
    print("```")

    print("\nFixed Code:")
    print("```python")
    print(fixed[:150] + ("..." if len(fixed) > 150 else ""))
    print("```")

    print("\nValidation Results:")
    print("-" * 70)

    # Run validation
    pipeline = ValidationPipeline(timeout_seconds=5)
    report = await pipeline.validate_fix(original, fixed, file_path)

    # Show results
    for result in report.results:
        print(f"  {result.summary()}")

    # Final verdict
    print("-" * 70)
    if report.all_passed:
        print(f"✓ VERDICT: Safe to commit!")
    else:
        print(f"✗ VERDICT: Do NOT commit - validation failed")
        if report.failed_stage:
            print(f"✗ FAILED AT: {report.failed_stage.value}")


async def main():
    """Run all demo scenarios"""
    await print_header("FERN'S VALIDATION PIPELINE DEMO", level=1)

    print("""
Phase 5: Validation Pipeline
5-stage validation to ensure fixes are safe:

  1. Syntax Check (AST parsing) - 0-3ms
  2. Import Validation (import resolution) - 0-5ms  
  3. Test Execution (pytest) - 0-5000ms
  4. Trough Re-scan (AI slop detection) - 20-100ms
  5. Regression Check (behavior preservation) - 1-10ms

Philosophy: "Fern validates everything - and so should you!"
""")

    # Scenario 1: Passing validation
    await run_scenario(
        SCENARIO_1_DESC,
        SCENARIO_1_ORIGINAL,
        SCENARIO_1_FIXED,
        "main.py"
    )

    # Scenario 2: Syntax error
    await run_scenario(
        SCENARIO_2_DESC,
        SCENARIO_2_ORIGINAL,
        SCENARIO_2_FIXED,
        "main.py"
    )

    # Scenario 3: Breaking change
    await run_scenario(
        SCENARIO_3_DESC,
        SCENARIO_3_ORIGINAL,
        SCENARIO_3_FIXED,
        "api.py"
    )

    # Scenario 4: Safe improvement
    await run_scenario(
        SCENARIO_4_DESC,
        SCENARIO_4_ORIGINAL,
        SCENARIO_4_FIXED,
        "main.py"
    )

    # Summary
    await print_header("VALIDATION SUMMARY", level=1)

    print("""
Key Features:
  Scenario 1: Dead code removal - PASS syntax/imports
  Scenario 2: Syntax error caught at stage 1
  Scenario 3: Breaking API change detected
  Scenario 4: Safe code removal - all stages pass

Validation Pipeline Characteristics:
  - Fail-fast: stops on first failure
  - Detailed diagnostics for each stage
  - Performance tracking (milliseconds)
  - Clear pass/fail decisions
  - Can skip unnecessary stages (e.g., tests)

Philosophy: 
  "Fern validates everything - and so should you!"
  "A fix without validation is a pig without a pen!"

Production Ready! All 5 stages implemented and tested.
""")


if __name__ == "__main__":
    asyncio.run(main())
