#!/usr/bin/env python3
"""
xTerminator Phase 4 Demo: Git Integration
==========================================

Demonstrates the complete workflow:
1. Classify an issue (Phase 1)
2. Apply fix with git commit (Phase 4)
3. Manage rollback (Phase 4)

Run with:
    cd xterminator
    python3 demo_git_integration.py
"""

import asyncio
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Tuple

from xterminator import (
    RiskLevel,
    FixStrategy,
    CodeContext,
    ContextType,
    FixProposal,
    GitApplicator,
    RollbackManager,
    BranchManager,
)


# ============================================================================
# Demo Setup
# ============================================================================

def setup_demo_repo() -> Tuple[Path, GitApplicator]:
    """Create a demo git repository"""
    print("\n" + "=" * 70)
    print("SETTING UP DEMO REPOSITORY")
    print("=" * 70)

    temp_dir = Path(tempfile.mkdtemp(prefix="xterminator_demo_"))
    print(f"\nCreating demo repository at: {temp_dir}")

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    # Configure git
    subprocess.run(
        ["git", "config", "user.email", "templeton@example.com"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    subprocess.run(
        ["git", "config", "user.name", "Templeton the Rat"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    # Create initial files
    readme = temp_dir / "README.md"
    readme.write_text("# Demo Project\n\nxTerminator Phase 4 Git Integration Demo\n")

    config_file = temp_dir / "config.py"
    config_file.write_text("""# Configuration
API_KEY = 'hardcoded_secret_key_12345'
DEBUG = True
DATABASE_URL = 'postgresql://root:password@localhost/db'
""")

    app_file = temp_dir / "app.py"
    app_file.write_text("""def authenticate(username, password):
    # CRITICAL: No input validation!
    user = find_user(username)
    if user.password == password:
        return True
    return False

def process_data(raw_data):
    result = raw_data.split(',')
    return result
""")

    # Create initial commit
    subprocess.run(
        ["git", "add", "."],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    subprocess.run(
        ["git", "commit", "-m", "Initial commit: Project setup"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    applicator = GitApplicator(str(temp_dir))
    print(f"\nRepository initialized with initial files:")
    print(f"  - README.md (project documentation)")
    print(f"  - config.py (hardcoded secrets)")
    print(f"  - app.py (security issues)")

    return temp_dir, applicator


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_low_risk_fix(repo_dir: Path, applicator: GitApplicator):
    """Demo: Apply a low-risk fix (hardcoded values)"""
    print("\n" + "=" * 70)
    print("DEMO 1: LOW-RISK FIX - Move Hardcoded Values")
    print("=" * 70)

    print("\nIssue: Hardcoded API key in config.py")
    print("Risk: LOW")
    print("Confidence: 0.92")

    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="config.py",
        line_number=1,
        column=10,
        code_snippet="API_KEY = 'hardcoded_secret_key_12345'",
        parent_node_type="Module",
        scope_depth=0,
        has_tests=False
    )

    proposal = FixProposal(
        fix_id="FIX_HARDCODED_001",
        issue_category="hardcoded_values",
        issue_severity="medium",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.92,
        original_code="API_KEY = 'hardcoded_secret_key_12345'",
        proposed_code="API_KEY = os.getenv('API_KEY', 'default')",
        explanation="Move hardcoded API key to environment variable for security",
        context=context,
        safe_to_autofix=True,
        requires_approval=False
    )

    # Read current config
    config_path = repo_dir / "config.py"
    current_config = config_path.read_text()

    # Apply fix
    fixed_config = """# Configuration
import os
API_KEY = os.getenv('API_KEY', 'default')
DEBUG = True
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///:memory:')
"""

    print(f"\nApplying fix...")
    result = await applicator.apply_fix(
        file_path="config.py",
        fixed_code=fixed_config,
        proposal=proposal,
        auto_branch=False  # Low-risk, no branch needed
    )

    if result.success:
        print(f"\nSUCCESS!")
        print(f"  Commit: {result.commit_hash[:8]}...")
        print(f"  File: {result.details['file']}")
        print(f"  Risk Level: {result.details['risk']}")
        print(f"  Confidence: {result.details['confidence']:.2f}")

        # Show changes
        print(f"\nChanges applied to config.py:")
        print(f"  - Before: API_KEY = 'hardcoded_secret_key_12345'")
        print(f"  - After:  API_KEY = os.getenv('API_KEY', 'default')")
    else:
        print(f"\nFAILED: {result.message}")

    return result.commit_hash if result.success else None


async def demo_high_risk_fix(repo_dir: Path, applicator: GitApplicator):
    """Demo: Apply a high-risk fix (security issue)"""
    print("\n" + "=" * 70)
    print("DEMO 2: HIGH-RISK FIX - Security Issue (Creates Feature Branch)")
    print("=" * 70)

    print("\nIssue: No input validation in authenticate() function")
    print("Risk: HIGH")
    print("Confidence: 0.88")

    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="app.py",
        line_number=1,
        column=0,
        code_snippet="def authenticate(username, password):",
        parent_node_type="FunctionDef",
        parent_node_name="authenticate",
        scope_depth=0,
        has_tests=False
    )

    proposal = FixProposal(
        fix_id="FIX_SECURITY_001",
        issue_category="error_handling",
        issue_severity="high",
        risk_level=RiskLevel.HIGH,
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.88,
        original_code="def authenticate(username, password):\n    user = find_user(username)",
        proposed_code="""def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user with input validation.\"\"\"
    if not username or not password:
        raise ValueError("Username and password required")
    if len(password) < 8:
        raise ValueError("Password too short")
    user = find_user(username)""",
        explanation="Add input validation and type hints to authenticate function",
        context=context,
        safe_to_autofix=False,
        requires_approval=True
    )

    fixed_app = """def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate user with input validation.\"\"\"
    if not username or not password:
        raise ValueError("Username and password required")
    if len(password) < 8:
        raise ValueError("Password too short")
    user = find_user(username)
    if user.password == password:
        return True
    return False

def process_data(raw_data):
    if not raw_data:
        return []
    result = raw_data.split(',')
    return result
"""

    print(f"\nApplying fix to app.py...")
    print(f"Note: HIGH-RISK fix will create a feature branch")

    result = await applicator.apply_fix(
        file_path="app.py",
        fixed_code=fixed_app,
        proposal=proposal,
        auto_branch=True  # High-risk: create branch
    )

    if result.success:
        print(f"\nSUCCESS!")
        print(f"  Commit: {result.commit_hash[:8]}...")
        print(f"  Risk Level: {result.details['risk']}")
        print(f"  Confidence: {result.details['confidence']:.2f}")

        # Show current branch
        branch = applicator._get_current_branch()
        print(f"  Current Branch: {branch} (returned to original)")
    else:
        print(f"\nFAILED: {result.message}")

    return result.commit_hash if result.success else None


async def demo_rollback(repo_dir: Path, applicator: GitApplicator):
    """Demo: Rollback fixes"""
    print("\n" + "=" * 70)
    print("DEMO 3: Rollback Management")
    print("=" * 70)

    rollback_mgr = RollbackManager(str(repo_dir), applicator)

    # Show history
    history = await rollback_mgr.get_rollback_history()
    print(f"\nCommit History ({len(history)} xTerminator commits):")
    for i, commit in enumerate(history, 1):
        print(f"\n  {i}. {commit['fix_id']} - {commit['category']}")
        print(f"     Risk: {commit['risk']} | Confidence: {commit['confidence']:.2f}")
        print(f"     File: {commit['file']}")
        print(f"     Commit: {commit['commit'][:8]}...")

    # Show dry-run rollback
    print(f"\n\nDemonstrating rollback (DRY-RUN):")
    print(f"  Would rollback: {len(history)} commit(s)")

    if history:
        first_commit = history[0]
        print(f"\n  Most recent fix: {first_commit['fix_id']}")
        print(f"  Would revert to: {history[1]['commit'][:8]}..." if len(history) > 1 else "  Would revert to: Initial commit")


async def demo_branch_management(repo_dir: Path):
    """Demo: Branch management"""
    print("\n" + "=" * 70)
    print("DEMO 4: Branch Management")
    print("=" * 70)

    branch_mgr = BranchManager(str(repo_dir))

    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="test.py",
        line_number=1,
        column=0,
        code_snippet="",
        scope_depth=0
    )

    proposals = [
        FixProposal(
            fix_id="FIX_001",
            issue_category="formatting",
            issue_severity="low",
            risk_level=RiskLevel.LOW,
            fix_strategy=FixStrategy.TEMPLATE,
            confidence=0.95,
            original_code="",
            explanation="Format code",
            context=context,
            safe_to_autofix=True,
            requires_approval=False
        ),
        FixProposal(
            fix_id="FIX_002",
            issue_category="security",
            issue_severity="critical",
            risk_level=RiskLevel.CRITICAL,
            fix_strategy=FixStrategy.MANUAL,
            confidence=0.85,
            original_code="",
            explanation="Critical security issue",
            context=context,
            safe_to_autofix=False,
            requires_approval=True
        ),
    ]

    print("\nFeature branch naming strategy:")
    for proposal in proposals:
        branch_name = branch_mgr.get_feature_branch_name(proposal)
        risk_indicator = "CREATES BRANCH" if proposal.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} else "direct commit"

        print(f"\n  {proposal.issue_category.upper()}")
        print(f"    Risk: {proposal.risk_level.value}")
        print(f"    Branch: {branch_name} ({risk_indicator})")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run the complete demo"""
    print("\n")
    print("=" * 70)
    print("  xTerminator Phase 4: Git Integration Demo")
    print("  Templeton commits with style!")
    print("=" * 70)

    repo_dir, applicator = setup_demo_repo()

    try:
        # Run demos
        commit1 = await demo_low_risk_fix(repo_dir, applicator)
        commit2 = await demo_high_risk_fix(repo_dir, applicator)
        await demo_rollback(repo_dir, applicator)
        await demo_branch_management(repo_dir)

        # Summary
        print("\n" + "=" * 70)
        print("DEMO SUMMARY")
        print("=" * 70)
        print(f"""
xTerminator Phase 4 demonstrates:

1. LOW-RISK FIX
   - Automatic commit to main branch
   - Minimal oversight needed
   - Direct application

2. HIGH-RISK FIX
   - Creates feature branch for safety
   - Requires manual review
   - Easy to rollback

3. ROLLBACK MANAGEMENT
   - Complete audit trail
   - Rollback by last N commits
   - Rollback by category or file

4. BRANCH MANAGEMENT
   - Automatic branch naming
   - Risk-based strategy
   - Safe merging workflow

Commit Metadata:
  - Stored in .xterminator/commits.json
  - Includes risk, confidence, strategy, fix_id
  - Enables intelligent rollback decisions

Philosophy:
  "Templeton commits carefully, rolls back fearlessly!"

For Production Use:
  - Review all HIGH/CRITICAL fixes before merging
  - Use branch protection rules on main
  - Enable CI/CD testing on feature branches
  - Archive rollback history for compliance
""")

    finally:
        # Cleanup
        print(f"\nCleaning up demo repository...")
        shutil.rmtree(repo_dir, ignore_errors=True)
        print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
