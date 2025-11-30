#!/usr/bin/env python3
"""Quick test of ML logic detector basic functionality."""

import asyncio
from HoloLoom.agentic.ml_logic_detector import MLLogicDetector
from HoloLoom.agentic.codebase_ingestion import Language


async def test_basic():
    """Test basic detection without CFG."""
    detector = MLLogicDetector()

    # Test 1: Division by zero (no CFG needed)
    code1 = """
def divide(a, b):
    return a / b
"""
    print("Test 1: Division by zero")
    errors = await detector.detect(code1, Language.PYTHON)
    print(f"  Errors found: {len(errors)}")
    for error in errors:
        print(f"  - {error.error_type.value}: {error.description}")

    # Test 2: Null dereference (no CFG needed)
    code2 = """
user = None
name = user.name
"""
    print("\nTest 2: Null dereference")
    errors = await detector.detect(code2, Language.PYTHON)
    print(f"  Errors found: {len(errors)}")
    for error in errors:
        print(f"  - {error.error_type.value}: {error.description}")

    # Test 3: Constant condition (no CFG needed)
    code3 = """
if True:
    process()
"""
    print("\nTest 3: Constant condition")
    errors = await detector.detect(code3, Language.PYTHON)
    print(f"  Errors found: {len(errors)}")
    for error in errors:
        print(f"  - {error.error_type.value}: {error.description}")

    print("\n✓ Basic tests completed")


if __name__ == "__main__":
    asyncio.run(test_basic())
