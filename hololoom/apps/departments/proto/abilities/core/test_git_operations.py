"""Tests for Git Operations ability.

Run with: pytest HoloLoom/departments/proto/abilities/core/test_git_operations.py -v

Tests cover:
- Manifest validation
- Preflight checks
- All operation types
- Error handling
- Path validation/security
- Verification
"""

import os
import subprocess

import pytest
from git_operations import GitOperationsAbility

from ..protocol import AbilityContext


class TestGitOperationsManifest:
    """Test ability manifest."""

    def test_manifest_properties(self):
        """Verify manifest is correctly configured."""
        ability = GitOperationsAbility()
        manifest = ability.manifest

        assert manifest.name == "git_operations"
        assert manifest.version == "1.0.0"
        assert manifest.tier.value == 2
        assert manifest.trust_level.value == "verified"
        assert "git" in manifest.tags
        assert "vcs" in manifest.tags
        assert "git" in manifest.requires


class TestGitOperationsPreflight:
    """Test preflight checks."""

    @pytest.mark.asyncio
    async def test_preflight_git_available(self):
        """Preflight should pass when git is available."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.preflight(context)
        assert result.can_execute is True

    @pytest.mark.asyncio
    async def test_preflight_working_dir_exists(self):
        """Preflight should pass when working directory exists."""
        ability = GitOperationsAbility()
        context = AbilityContext(
            session_id="test",
            working_directory=os.getcwd()
        )

        result = await ability.preflight(context)
        assert result.can_execute is True

    @pytest.mark.asyncio
    async def test_preflight_working_dir_not_exists(self):
        """Preflight should fail when working directory doesn't exist."""
        ability = GitOperationsAbility()
        context = AbilityContext(
            session_id="test",
            working_directory="/nonexistent/path/to/nowhere"
        )

        result = await ability.preflight(context)
        assert result.can_execute is False
        assert "not found" in result.reason


class TestGitOperationsStatus:
    """Test git status operation."""

    @pytest.mark.asyncio
    async def test_status_in_git_repo(self, git_repo):
        """Status should work in a git repository."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "status",
            "repo_path": git_repo
        }, context)

        assert result.success is True
        assert "status_output" in result.output
        assert "staged_files" in result.output
        assert "unstaged_files" in result.output
        assert "untracked_files" in result.output
        assert "has_changes" in result.output

    @pytest.mark.asyncio
    async def test_status_not_git_repo(self, tmp_path):
        """Status should fail in non-git directory."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "status",
            "repo_path": str(tmp_path)
        }, context)

        assert result.success is False
        assert "not a git repository" in result.error.lower()


class TestGitOperationsBranch:
    """Test git branch operation."""

    @pytest.mark.asyncio
    async def test_branch_listing(self, git_repo):
        """Branch operation should list branches."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "branch",
            "repo_path": git_repo
        }, context)

        assert result.success is True
        assert "branch_output" in result.output
        assert "branches" in result.output
        assert "current_branch" in result.output
        assert isinstance(result.output["branches"], list)


class TestGitOperationsLog:
    """Test git log operation."""

    @pytest.mark.asyncio
    async def test_log_with_limit(self, git_repo):
        """Log operation should respect limit parameter."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "log",
            "repo_path": git_repo,
            "limit": 5
        }, context)

        assert result.success is True
        assert "log_output" in result.output
        assert "commits" in result.output
        assert isinstance(result.output["commits"], list)
        assert result.output["limit"] == 5

    @pytest.mark.asyncio
    async def test_log_limit_clamped(self, git_repo):
        """Log limit should be clamped to 1-100."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        # Test upper limit
        result = await ability.execute({
            "operation": "log",
            "repo_path": git_repo,
            "limit": 200
        }, context)

        assert result.success is True
        assert result.output["limit"] == 100  # Clamped

        # Test lower limit
        result = await ability.execute({
            "operation": "log",
            "repo_path": git_repo,
            "limit": 0
        }, context)

        assert result.success is True
        assert result.output["limit"] == 1  # Clamped


class TestGitOperationsValidation:
    """Test parameter validation and security."""

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, git_repo):
        """Should prevent path traversal attacks."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "diff",
            "repo_path": git_repo,
            "file_path": "../../../etc/passwd"
        }, context)

        assert result.success is False
        assert "path traversal" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_operation(self, git_repo):
        """Should fail with missing operation."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "repo_path": git_repo
        }, context)

        assert result.success is False
        assert "operation" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, git_repo):
        """Should fail with unsupported operation."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "push",  # Not allowed
            "repo_path": git_repo
        }, context)

        assert result.success is False
        assert "unsupported" in result.error.lower()

    @pytest.mark.asyncio
    async def test_blame_requires_file(self, git_repo):
        """Blame operation should require file_path."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        result = await ability.execute({
            "operation": "blame",
            "repo_path": git_repo
        }, context)

        assert result.success is False
        assert "file_path" in result.error.lower()


class TestGitOperationsVerification:
    """Test result verification."""

    @pytest.mark.asyncio
    async def test_verify_successful_result(self, git_repo):
        """Verification should pass for successful operations."""
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="test")

        exec_result = await ability.execute({
            "operation": "status",
            "repo_path": git_repo
        }, context)

        verify_result = await ability.verify(exec_result)
        assert verify_result.verified is True

    @pytest.mark.asyncio
    async def test_verify_failed_result(self):
        """Verification should fail for failed operations."""
        ability = GitOperationsAbility()

        from ..protocol import AbilityResult
        failed_result = AbilityResult(
            success=False,
            error="Test error"
        )

        verify_result = await ability.verify(failed_result)
        assert verify_result.verified is False
        assert "failed" in verify_result.issues[0].lower()


# ========== Fixtures ==========

@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        check=True,
        capture_output=True
    )

    # Configure git user
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True
    )

    # Create initial commit
    test_file = repo_path / "test.txt"
    test_file.write_text("Test content\n")

    subprocess.run(
        ["git", "add", "test.txt"],
        cwd=repo_path,
        check=True,
        capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True
    )

    return str(repo_path)


# ========== Main Entry Point ==========

if __name__ == "__main__":
    # Run with: python -m pytest test_git_operations.py -v
    pytest.main([__file__, "-v"])
