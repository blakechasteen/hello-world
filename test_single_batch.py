"""
Test Single Batch Fix
=====================

Tests a single file with detailed error logging.
"""

import asyncio
from xterminator.mcp_server import XTerminatorMCPServer


async def test_batch():
    """Test batch fix on a single file with few issues."""
    print("=" * 70)
    print("TESTING: Batch Fix with Detailed Errors")
    print("=" * 70)
    print()

    # Sample file with simple issues
    code = """from typing import Dict, List, Optional
from collections import defaultdict
import json
import warnings

def process():
    data = json.loads("{}")
    return data
"""

    issues = [
        {
            'category': 'dead_code',
            'message': "Unused import 'Dict'",
            'line_number': 1,
            'code_snippet': 'from typing import Dict, List, Optional',
            'confidence': 1.0,
            'severity': 'low'
        },
        {
            'category': 'dead_code',
            'message': "Unused import 'defaultdict'",
            'line_number': 2,
            'code_snippet': 'from collections import defaultdict',
            'confidence': 1.0,
            'severity': 'low'
        },
        {
            'category': 'dead_code',
            'message': "Unused import 'warnings'",
            'line_number': 4,
            'code_snippet': 'import warnings',
            'confidence': 1.0,
            'severity': 'low'
        }
    ]

    # Create server
    server = XTerminatorMCPServer(use_simple_fixer=True)

    # Run batch fix
    result = await server.xterminator_batch_fix(
        issues=issues,
        code=code,
        file_path='test_file.py',
        max_fixes=3
    )

    print(f"Total issues: {result['total_issues']}")
    print(f"Fixes proposed: {result['fixes_proposed']}")
    print(f"Fixes validated: {result['fixes_validated']}")
    print(f"Fixes applied: {result['fixes_applied']}")
    print()

    # Show detailed results
    for i, fix_result in enumerate(result['results'], 1):
        print(f"{i}. {fix_result['issue']['message']}")
        print(f"   Status: {fix_result['status']}")
        if 'error' in fix_result:
            print(f"   Error: {fix_result['error']}")
        if 'failed_steps' in fix_result:
            print(f"   Failed steps: {len(fix_result['failed_steps'])}")
            for step in fix_result['failed_steps']:
                print(f"     - {step['step']}: {step['message']}")
        print()


if __name__ == '__main__':
    asyncio.run(test_batch())
