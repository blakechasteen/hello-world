"""
Debug Fix Validation Failures
==============================

Tests a single fix operation to understand why validation is failing.
"""

import asyncio
from xterminator.simple_llm_fixer import SimpleLLMFixer


async def test_single_fix():
    """Test a simple unused import fix."""
    print("=" * 70)
    print("TESTING: Simple Unused Import Fix")
    print("=" * 70)
    print()

    # Sample code with unused import
    code = """from typing import Dict, List, Optional
import json

def process():
    data = json.loads("{}")
    return data
"""

    # Create issue
    issue = {
        'category': 'dead_code',
        'message': "Unused import 'Dict'",
        'line_number': 1,
        'code_snippet': 'from typing import Dict, List, Optional',
        'file_path': 'test.py'
    }

    print("Original code:")
    print(code)
    print()

    # Try to fix
    fixer = SimpleLLMFixer(use_llm=False)  # Disable LLM for simple test
    result = await fixer.fix_issue(issue, code, 'test.py')

    if result:
        fixed_code, diff = result
        print("✓ Fix successful!")
        print()
        print("Fixed code:")
        print(fixed_code)
        print()
        print("Diff:")
        print(diff)
    else:
        print("✗ Fix failed - fixer returned None")

    print()


async def test_magic_number_fix():
    """Test a magic number extraction fix."""
    print("=" * 70)
    print("TESTING: Magic Number Fix")
    print("=" * 70)
    print()

    code = """def calculate_timeout():
    return 30  # seconds
"""

    issue = {
        'category': 'hardcoded_values',
        'message': 'Magic number 30 should be a named constant',
        'line_number': 2,
        'code_snippet': '    return 30  # seconds',
        'file_path': 'test.py'
    }

    print("Original code:")
    print(code)
    print()

    fixer = SimpleLLMFixer(use_llm=False)
    result = await fixer.fix_issue(issue, code, 'test.py')

    if result:
        fixed_code, diff = result
        print("✓ Fix successful!")
        print()
        print("Fixed code:")
        print(fixed_code)
        print()
        print("Diff:")
        print(diff)
    else:
        print("✗ Fix failed - fixer returned None")

    print()


if __name__ == '__main__':
    asyncio.run(test_single_fix())
    asyncio.run(test_magic_number_fix())
