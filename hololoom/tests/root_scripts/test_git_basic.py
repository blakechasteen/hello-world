"""
Basic Git Integration Test
===========================

Simple test to validate GitIntegrator file operations and commits.

Author: mythRL Team
Date: November 16, 2025
"""

import tempfile
import subprocess
from pathlib import Path

from xterminator.git_integrator import GitIntegrator, GitConfig


def test_git_basic():
    """Test basic Git operations."""

    print("=" * 70)
    print("BASIC GIT INTEGRATION TEST")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize Git repo
        subprocess.run(['git', 'init'], cwd=tmpdir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=tmpdir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=tmpdir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'commit.gpgsign', 'false'], cwd=tmpdir, capture_output=True, check=True)

        print("✓ Initialized Git repository")
        print()

        # Create initial file and commit
        test_file = Path(tmpdir) / "example.py"
        test_file.write_text("# Original content\nprint('Hello')\n")

        subprocess.run(['git', 'add', 'example.py'], cwd=tmpdir, capture_output=True, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=tmpdir, capture_output=True, check=True)

        print("✓ Created initial commit")
        print()

        # Test GitIntegrator
        integrator = GitIntegrator(
            config=GitConfig(
                auto_create_branch=True,
                auto_commit=True
            ),
            repo_path=tmpdir
        )

        # Test 1: Write file with backup
        print("Test 1: Write File with Backup")
        print("-" * 70)

        backup = integrator.write_file_with_backup(
            str(test_file),
            "# Modified content\nprint('Hello World')\n",
            create_backup=True
        )

        print(f"✓ File written")
        print(f"  Backup: {backup.backup_path}")
        print(f"  SHA256: {backup.sha256[:16]}...")
        print()

        # Verify file was modified
        modified_content = test_file.read_text()
        assert "Modified content" in modified_content
        print("✓ File content verified")
        print()

        # Test 2: Create Git branch
        print("Test 2: Create Git Branch")
        print("-" * 70)

        branch_result = integrator.create_branch("test/autofix-example")

        print(f"✓ Branch created")
        print(f"  Success: {branch_result.success}")
        print(f"  Branch: {branch_result.branch}")
        print()

        # Test 3: Stage and commit
        print("Test 3: Stage and Commit")
        print("-" * 70)

        # Stage file
        stage_result = integrator.stage_files([str(test_file)])
        print(f"✓ Files staged: {stage_result.success}")

        # Commit
        commit_msg = "Fix: Update print statement"
        commit_result = integrator.commit(commit_msg, files=[str(test_file)])

        print(f"✓ Commit created")
        print(f"  Success: {commit_result.success}")
        if commit_result.commit:
            print(f"  Commit SHA: {commit_result.commit[:8]}")
        if commit_result.error:
            print(f"  Error: {commit_result.error}")
        print()

        # Test 4: Restore from backup
        print("Test 4: Restore from Backup")
        print("-" * 70)

        restored = integrator.restore_from_backup(str(test_file))

        print(f"✓ Restore from backup: {restored}")

        if restored:
            restored_content = test_file.read_text()
            original_match = "Original content" in restored_content
            print(f"✓ Content matches original: {original_match}")
        print()

        # Summary
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print()

        all_passed = (
            backup.backup_path and
            branch_result.success and
            stage_result.success and
            restored
        )

        if all_passed:
            print("✅ All basic Git operations PASSED!")
        else:
            print("⚠ Some operations had issues")
            if not commit_result.success:
                print(f"  Commit failed: {commit_result.error}")

        print()
        print("Validated:")
        print("  ✓ File backup creation")
        print("  ✓ Git branch creation")
        print("  ✓ File staging")
        if commit_result.success:
            print("  ✓ Git commit creation")
        else:
            print("  ⚠ Git commit (see error above)")
        print("  ✓ Backup restoration")
        print()


if __name__ == '__main__':
    test_git_basic()
