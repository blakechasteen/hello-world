#!/usr/bin/env python3
"""Debug AST fixer issues"""

import asyncio
from xterminator.ast_fixer import ASTFixer
from xterminator.xterminator_types import (
    FixProposal,
    FixStrategy,
    RiskLevel,
    CodeContext,
    ContextType
)


async def test_dead_code():
    """Debug dead code removal"""
    print("=== Testing Dead Code Removal ===\n")

    code = """
def calculate(x):
    if x > 0:
        return x * 2
        print("This is dead code!")  # After return
        y = x + 1  # Also dead

    return 0
"""

    proposal = FixProposal(
        fix_id="dead_code",
        issue_category="dead_code",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.95,
        original_code='print("This is dead code!")',
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.EXECUTABLE,
            file_path="test.py",
            line_number=4
        )
    )

    fixer = ASTFixer()
    result = await fixer.fix_issue(proposal, code)

    if result:
        fixed_code, diff = result
        print("SUCCESS!")
        print("\nFixed code:")
        print(fixed_code)
        print("\nDiff:")
        print(diff)
    else:
        print("FAILED!")
        print(f"Error: {proposal.metadata.get('error', 'Unknown')}")


async def test_extract_constant():
    """Debug constant extraction"""
    print("\n=== Testing Constant Extraction ===\n")

    code = """
def calculate_area(radius):
    return 3.14159 * radius ** 2
"""

    proposal = FixProposal(
        fix_id="magic_number",
        issue_category="magic_number",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.92,
        original_code="return 3.14159 * radius ** 2",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.EXECUTABLE,
            file_path="test.py",
            line_number=2
        )
    )

    fixer = ASTFixer()
    result = await fixer.fix_issue(proposal, code)

    if result:
        fixed_code, diff = result
        print("SUCCESS!")
        print("\nFixed code:")
        print(fixed_code)
        print("\nDiff:")
        print(diff)
    else:
        print("FAILED!")
        print(f"Error: {proposal.metadata.get('error', 'Unknown')}")


async def test_add_type_hint():
    """Debug type hint addition"""
    print("\n=== Testing Type Hint Addition ===\n")

    code = """
def process_data(x, y):
    return x + y
"""

    proposal = FixProposal(
        fix_id="type_hint",
        issue_category="missing_type_hint",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.88,
        original_code="def process_data(x, y):",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.DEFINITION,
            file_path="test.py",
            line_number=1
        )
    )

    fixer = ASTFixer()
    result = await fixer.fix_issue(proposal, code)

    if result:
        fixed_code, diff = result
        print("SUCCESS!")
        print("\nFixed code:")
        print(fixed_code)
        print("\nDiff:")
        print(diff)
    else:
        print("FAILED!")
        print(f"Error: {proposal.metadata.get('error', 'Unknown')}")


if __name__ == '__main__':
    asyncio.run(test_dead_code())
    asyncio.run(test_extract_constant())
    asyncio.run(test_add_type_hint())
