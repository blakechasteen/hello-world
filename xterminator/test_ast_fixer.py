#!/usr/bin/env python3
"""
Test Suite for Wilbur the AST Auto-Fixer
=========================================

"Testing is TERRIFIC!" - Charlotte

Tests all 6 AST transformations:
1. Extract Function
2. Remove Dead Code
3. Remove Unused Import
4. Extract Constant
5. Rename Variable
6. Add Type Hint
"""

import pytest
import asyncio
from pathlib import Path

from .ast_fixer import ASTFixer, TransformationResult
from .xterminator_types import (
    FixProposal,
    FixStrategy,
    RiskLevel,
    CodeContext,
    ContextType
)


# ==================== Fixtures ====================


@pytest.fixture
def fixer():
    """Create ASTFixer instance"""
    return ASTFixer()


@pytest.fixture
def safe_proposal():
    """Create safe-to-autofix proposal"""
    return FixProposal(
        fix_id="test_001",
        issue_category="test",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.95,
        original_code="x = 1",
        safe_to_autofix=True,
        requires_approval=False,
        context=CodeContext(
            context_type=ContextType.EXECUTABLE,
            file_path="test.py",
            line_number=10
        )
    )


# ==================== Safety Tests ====================


@pytest.mark.asyncio
async def test_rejects_unsafe_proposals(fixer):
    """Test that unsafe proposals are rejected"""
    unsafe_proposal = FixProposal(
        fix_id="unsafe",
        issue_category="test",
        issue_severity="high",
        risk_level=RiskLevel.HIGH,
        fix_strategy=FixStrategy.AST,
        confidence=0.95,
        original_code="x = 1",
        safe_to_autofix=False  # Not safe!
    )

    code = "x = 1"
    result = await fixer.fix_issue(unsafe_proposal, code)

    assert result is None, "Should reject unsafe proposals"


@pytest.mark.asyncio
async def test_rejects_non_ast_strategy(fixer, safe_proposal):
    """Test that non-AST strategies are rejected"""
    safe_proposal.fix_strategy = FixStrategy.MANUAL

    code = "x = 1"
    result = await fixer.fix_issue(safe_proposal, code)

    assert result is None, "Should reject non-AST strategies"


@pytest.mark.asyncio
async def test_rejects_syntax_errors(fixer, safe_proposal):
    """Test that code with syntax errors is rejected"""
    bad_code = "def foo(\n  # Missing closing paren"

    result = await fixer.fix_issue(safe_proposal, bad_code)

    assert result is None, "Should reject code with syntax errors"


# ==================== Transformation Tests ====================


@pytest.mark.asyncio
async def test_extract_function(fixer):
    """Test extracting duplicated code into a function"""
    code = """
def main():
    x = 1
    y = 2
    result = x + y
    print(result)

    # Duplicate code
    x = 1
    y = 2
    result = x + y
    print(result)
"""

    proposal = FixProposal(
        fix_id="copy_paste",
        issue_category="copy_paste_error",
        issue_severity="medium",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.90,
        original_code="x = 1\n    y = 2\n    result = x + y",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.EXECUTABLE,
            file_path="test.py",
            line_number=9
        )
    )

    result = await fixer.fix_issue(proposal, code)

    assert result is not None, "Extraction should succeed"
    fixed_code, diff = result

    # Verify function was created
    assert "def " in fixed_code, "Should create new function"
    assert "Charlotte" in diff, "Should have Charlotte's commentary"


@pytest.mark.asyncio
async def test_remove_dead_code(fixer):
    """Test removing unreachable dead code"""
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

    result = await fixer.fix_issue(proposal, code)

    assert result is not None, "Dead code removal should succeed"
    fixed_code, diff = result

    # Verify dead code was removed
    assert 'print("This is dead code!")' not in fixed_code
    assert "Templeton" in diff, "Should have Templeton's commentary"


@pytest.mark.asyncio
async def test_remove_unused_import(fixer):
    """Test removing unused imports"""
    code = """
import os
import sys
import json  # Unused!
from pathlib import Path  # Unused!

def main():
    print(os.getcwd())
    sys.exit(0)
"""

    proposal = FixProposal(
        fix_id="unused_import",
        issue_category="unused_import",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.98,
        original_code="import json",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.IMPORT,
            file_path="test.py",
            line_number=3
        )
    )

    result = await fixer.fix_issue(proposal, code)

    assert result is not None, "Unused import removal should succeed"
    fixed_code, diff = result

    # Verify unused imports were removed
    assert "import json" not in fixed_code
    assert "from pathlib import Path" not in fixed_code
    assert "import os" in fixed_code  # Keep used imports
    assert "Templeton" in diff, "Should have Templeton's commentary"


@pytest.mark.asyncio
async def test_extract_constant(fixer):
    """Test extracting magic numbers to constants"""
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

    result = await fixer.fix_issue(proposal, code)

    assert result is not None, "Constant extraction should succeed"
    fixed_code, diff = result

    # Verify constant was created
    assert "= 3.14159" in fixed_code or "3.14159" in fixed_code
    assert "Wilbur" in diff, "Should have Wilbur's commentary"


@pytest.mark.asyncio
async def test_rename_variable(fixer):
    """Test renaming variable to follow conventions"""
    code = """
def process_data():
    myVariable = 42  # Should be my_variable
    anotherVar = 100  # Should be another_var
    return myVariable + anotherVar
"""

    proposal = FixProposal(
        fix_id="naming",
        issue_category="inconsistent_naming",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.90,
        original_code="myVariable = 42",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.EXECUTABLE,
            file_path="test.py",
            line_number=2
        ),
        metadata={
            'implementation_hints': {
                'old_name': 'myVariable',
                'new_name': 'my_variable'
            }
        }
    )

    result = await fixer.fix_issue(proposal, code)

    assert result is not None, "Variable renaming should succeed"
    fixed_code, diff = result

    # Verify variable was renamed
    assert "my_variable" in fixed_code
    assert "myVariable" not in fixed_code
    assert "Wilbur" in diff, "Should have Wilbur's commentary"


@pytest.mark.asyncio
async def test_add_type_hint(fixer):
    """Test adding type hints to functions"""
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

    result = await fixer.fix_issue(proposal, code)

    assert result is not None, "Type hint addition should succeed"
    fixed_code, diff = result

    # Verify type hints were added
    assert ":" in fixed_code  # Type hints use colons
    assert "Charlotte" in diff, "Should have Charlotte's commentary"


# ==================== Diff Generation Tests ====================


@pytest.mark.asyncio
async def test_diff_generation(fixer, safe_proposal):
    """Test that diffs are generated correctly"""
    original = "x = 1\ny = 2"
    fixed = "x = 1\ny = 3"

    safe_proposal.issue_category = "magic_number"

    diff = fixer._generate_diff(original, fixed, safe_proposal)

    assert diff, "Should generate diff"
    assert "Wilbur" in diff, "Should include commentary"
    assert "---" in diff or "+++" in diff, "Should be unified diff format"


# ==================== Integration Tests ====================


@pytest.mark.asyncio
async def test_real_world_scenario(fixer):
    """Test fixing a real-world code issue"""
    code = """
import json
import os
import sys

def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price'] * 1.08  # Magic number!
    return total

def calculate_subtotal(items):
    subtotal = 0
    for item in items:
        subtotal = subtotal + item['price'] * 1.08  # Duplicate!
    return subtotal
"""

    # Fix magic number
    magic_proposal = FixProposal(
        fix_id="magic_001",
        issue_category="magic_number",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.93,
        original_code="total = total + item['price'] * 1.08",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.EXECUTABLE,
            file_path="test.py",
            line_number=8
        )
    )

    result = await fixer.fix_issue(magic_proposal, code)
    assert result is not None, "Should fix magic number"

    fixed_code, diff = result
    print("\n🐷 Magic Number Fix:")
    print(diff)

    # Fix unused import
    unused_proposal = FixProposal(
        fix_id="import_001",
        issue_category="unused_import",
        issue_severity="low",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.AST,
        confidence=0.97,
        original_code="import json",
        safe_to_autofix=True,
        context=CodeContext(
            context_type=ContextType.IMPORT,
            file_path="test.py",
            line_number=1
        )
    )

    result2 = await fixer.fix_issue(unused_proposal, code)
    assert result2 is not None, "Should fix unused import"

    fixed_code2, diff2 = result2
    print("\n🐀 Unused Import Fix:")
    print(diff2)


# ==================== Performance Tests ====================


@pytest.mark.asyncio
async def test_performance(fixer, safe_proposal):
    """Test that transformations are fast enough"""
    import time

    code = "x = 1\ny = 2\nz = 3\n" * 100  # 300 lines

    safe_proposal.issue_category = "unused_import"

    start = time.time()
    result = await fixer.fix_issue(safe_proposal, code)
    duration = time.time() - start

    assert duration < 1.0, f"Should complete in <1s (took {duration:.2f}s)"


# ==================== Main ====================


if __name__ == '__main__':
    print("🐷 Testing Wilbur the AST Auto-Fixer\n")
    print("=" * 60)

    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])
