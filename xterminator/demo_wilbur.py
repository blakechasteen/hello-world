#!/usr/bin/env python3
"""
🐷 Wilbur Demo - Show Wilbur Fixing Real Issues
================================================

"Some Pig fixed this code!" - Charlotte
"""

import asyncio
import sys
sys.path.insert(0, 'C:\\Users\\blake\\OneDrive\\Documents\\mythRL')

from xterminator.classification_engine import ClassificationEngine
from xterminator.ast_fixer import ASTFixer


# Example 1: Unused Import
EXAMPLE_1 = """
import os
import sys
import json  # This is unused!

def main():
    print(os.getcwd())
    sys.exit(0)
"""


# Example 2: Magic Number
EXAMPLE_2 = """
def calculate_tax(amount):
    return amount * 1.08  # Magic number!
"""


# Example 3: Dead Code
EXAMPLE_3 = """
def process(x):
    if x > 0:
        return x * 2
        print("Never executed!")  # Dead code
    return 0
"""


async def demo_fix(name: str, code: str, issue_category: str):
    """Demo a single fix"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}\n")

    print("Original Code:")
    print(code)
    print()

    # Create fake issue for demo
    class FakeIssue:
        def __init__(self):
            self.category = issue_category
            self.severity = "low"
            self.confidence = 0.95
            self.file_path = "demo.py"
            self.line_number = 3
            self.code_snippet = code.split('\n')[2] if len(code.split('\n')) > 2 else ""

    issue = FakeIssue()

    # Classify
    engine = ClassificationEngine()
    proposal = await engine.classify_and_propose(issue, code, "demo.py")

    print(f"Classification: {proposal.fix_strategy.value}")
    print(f"Risk: {proposal.risk_level.value}")
    print(f"Confidence: {proposal.confidence:.2f}")
    print(f"Safe to Autofix: {proposal.safe_to_autofix}")
    print()

    if not proposal.safe_to_autofix:
        print("[WARN] Not safe to autofix - skipping")
        return

    # Apply fix
    fixer = ASTFixer()
    result = await fixer.fix_issue(proposal, code)

    if result:
        fixed_code, diff = result
        print("[OK] FIX APPLIED!\n")
        print("Diff:")
        print(diff)
        print("\nFixed Code:")
        print(fixed_code)
    else:
        error = proposal.metadata.get('error', 'Unknown error')
        print(f"[FAIL] FIX FAILED: {error}")


async def main():
    print("="*60)
    print("WILBUR THE AST AUTO-FIXER - DEMONSTRATION")
    print("Some Pig is making this code prize-winning!")
    print("="*60)

    await demo_fix("Unused Import Fix", EXAMPLE_1, "unused_import")
    await demo_fix("Magic Number Fix", EXAMPLE_2, "magic_number")
    await demo_fix("Dead Code Fix", EXAMPLE_3, "dead_code")

    print(f"\n{'='*60}")
    print("Charlotte says: 'Some Pig did TERRIFIC work!'")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    asyncio.run(main())
