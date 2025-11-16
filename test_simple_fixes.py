"""
Test Simple Template-Based Fixes
==================================

Tests the new remove_unused_import and extract_magic_number templates.
"""

import asyncio
from xterminator.templates import get_template
from xterminator.template_fixer import TemplateFixer
from xterminator.xterminator_types import FixProposal, FixStrategy, RiskLevel


async def test_remove_unused_import():
    """Test unused import removal."""
    print("=" * 70)
    print("TEST: Remove Unused Import")
    print("=" * 70)
    print()

    code = """from typing import Dict, List, Optional
import unused_module
import json

def process():
    data = json.loads("{}")
    return data
"""

    # Create fix proposal
    proposal = FixProposal(
        issue_id="test_001",
        issue_category="dead_code",
        severity="low",
        message="Unused import 'unused_module'",
        line_number=2,
        original_code="import unused_module",
        file_path="test.py",
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.95,
        risk_level=RiskLevel.LOW,
        context={},
        metadata={}
    )

    # Try to fix
    fixer = TemplateFixer()
    result = await fixer.fix_issue(proposal, code)

    if result:
        fixed_code, diff = result
        print("✅ Fix successful!")
        print()
        print("Fixed code:")
        print(fixed_code)
        print()
        print("Diff:")
        print(diff)
    else:
        print("❌ Fix failed - template fixer returned None")

    print()


async def test_extract_magic_number():
    """Test magic number extraction."""
    print("=" * 70)
    print("TEST: Extract Magic Number")
    print("=" * 70)
    print()

    code = """def calculate_timeout():
    return 30  # seconds
"""

    # Create fix proposal
    proposal = FixProposal(
        issue_id="test_002",
        issue_category="hardcoded_values",
        severity="low",
        message="Magic number 30 should be a named constant",
        line_number=2,
        original_code="    return 30  # seconds",
        file_path="test.py",
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.95,
        risk_level=RiskLevel.LOW,
        context={},
        metadata={}
    )

    # Try to fix
    fixer = TemplateFixer()
    result = await fixer.fix_issue(proposal, code)

    if result:
        fixed_code, diff = result
        print("✅ Fix successful!")
        print()
        print("Fixed code:")
        print(fixed_code)
        print()
        print("Diff:")
        print(diff)
    else:
        print("❌ Fix failed - template fixer returned None")

    print()


async def test_template_catalog():
    """Check if new templates are registered."""
    print("=" * 70)
    print("TEST: Template Catalog")
    print("=" * 70)
    print()

    remove_import = get_template("remove_unused_import")
    extract_number = get_template("extract_magic_number")

    if remove_import:
        print("✅ remove_unused_import template found")
        print(f"   Category: {remove_import.category}")
        print(f"   Description: {remove_import.description}")
    else:
        print("❌ remove_unused_import template NOT found")

    print()

    if extract_number:
        print("✅ extract_magic_number template found")
        print(f"   Category: {extract_number.category}")
        print(f"   Description: {extract_number.description}")
    else:
        print("❌ extract_magic_number template NOT found")

    print()


async def main():
    """Run all tests."""
    await test_template_catalog()
    await test_remove_unused_import()
    await test_extract_magic_number()


if __name__ == '__main__':
    asyncio.run(main())
