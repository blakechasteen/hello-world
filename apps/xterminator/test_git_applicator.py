"""
Tests for Git Applicator
========================

Unit and integration tests for GitApplicator, RollbackManager, and BranchManager.

Test Organization:
- Fixtures: Temporary git repos and test data
- Unit Tests: Individual component functionality
- Integration Tests: Complete workflows
- Safety Tests: Error handling and edge cases
"""

import pytest
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
)
from xterminator.git_applicator import (
    GitApplicator,
    RollbackManager,
    BranchManager,
    CommitMetadata,
    GitOperationResult,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_git_repo() -> Tuple[Path, GitApplicator]:
    """Create a temporary git repository"""
    temp_dir = Path(tempfile.mkdtemp(prefix="xterminator_test_"))

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    # Configure git (required for commits)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    # Create initial commit
    init_file = temp_dir / "README.md"
    init_file.write_text("# Test Repository\n")

    subprocess.run(
        ["git", "add", "README.md"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=temp_dir,
        capture_output=True,
        check=True
    )

    # Create applicator
    applicator = GitApplicator(str(temp_dir))

    yield temp_dir, applicator

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_fix_proposal() -> FixProposal:
    """Create a sample fix proposal"""
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="test.py",
        line_number=10,
        column=0,
        code_snippet="x = 'hardcoded_value'",
        parent_node_type="FunctionDef",
        parent_node_name="test_function",
        scope_depth=1,
        has_tests=True
    )

    return FixProposal(
        fix_id="FIX_001",
        issue_category="hardcoded_values",
        issue_severity="medium",
        risk_level=RiskLevel.LOW,
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.92,
        original_code="x = 'hardcoded_value'",
        proposed_code="x = os.getenv('VALUE', 'default')",
        explanation="Move hardcoded value to environment variable",
        context=context,
        safe_to_autofix=True,
        requires_approval=False
    )


@pytest.fixture
def high_risk_proposal() -> FixProposal:
    """Create a high-risk fix proposal"""
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="auth/security.py",
        line_number=50,
        column=0,
        code_snippet="password = request.args.get('password')",
        parent_node_type="FunctionDef",
        parent_node_name="authenticate",
        scope_depth=1,
        has_tests=False
    )

    return FixProposal(
        fix_id="FIX_HIGH_001",
        issue_category="error_handling",
        issue_severity="high",
        risk_level=RiskLevel.HIGH,
        fix_strategy=FixStrategy.TEMPLATE,
        confidence=0.88,
        original_code="password = request.args.get('password')",
        proposed_code="password = request.form.get('password', '')\nif not password:\n    raise ValueError('Password required')",
        explanation="Add proper error handling for authentication",
        context=context,
        safe_to_autofix=False,
        requires_approval=True
    )


# ============================================================================
# Unit Tests: GitApplicator
# ============================================================================

class TestGitApplicator:
    """Tests for GitApplicator"""

    def test_initialization(self, temp_git_repo):
        """Test GitApplicator initialization"""
        temp_dir, applicator = temp_git_repo
        assert applicator.repo == temp_dir
        assert applicator.metadata_file.exists() or applicator.metadata_file.parent.exists()

    def test_verify_git_repo(self, temp_git_repo):
        """Test git repository verification"""
        temp_dir, applicator = temp_git_repo
        # Should not raise - repo exists
        applicator._verify_git_repo()

    def test_verify_git_repo_failure(self):
        """Test git repo verification fails for non-git dir"""
        with pytest.raises(RuntimeError):
            GitApplicator("/tmp/nonexistent/path/xyz")

    def test_commit_message_generation(self, temp_git_repo, sample_fix_proposal):
        """Test commit message generation"""
        temp_dir, applicator = temp_git_repo

        msg = applicator._generate_commit_message("test.py", sample_fix_proposal)

        assert "fix(hardcoded_values):" in msg
        assert "File: test.py" in msg
        assert "Line: 10" in msg
        assert "Strategy: template" in msg
        assert "Confidence: 0.92" in msg
        assert "Risk: low" in msg
        assert "xTerminator v0.1.0" in msg

    def test_feature_branch_creation(self, temp_git_repo, sample_fix_proposal):
        """Test feature branch creation"""
        temp_dir, applicator = temp_git_repo

        branch_name = applicator._create_feature_branch(sample_fix_proposal)

        assert branch_name is not None
        assert "xterminator" in branch_name
        assert "fix_001" in branch_name.lower()

        # Verify branch exists
        result = applicator._run_git(["branch", "--list", branch_name])
        assert result.returncode == 0
        assert branch_name in result.stdout

    def test_get_current_branch(self, temp_git_repo):
        """Test getting current branch"""
        temp_dir, applicator = temp_git_repo

        branch = applicator._get_current_branch()
        assert branch in {"main", "master"}

    def test_has_uncommitted_changes_clean(self, temp_git_repo):
        """Test uncommitted changes detection on clean repo"""
        temp_dir, applicator = temp_git_repo

        changes = applicator._has_uncommitted_changes()
        assert changes == []

    def test_has_uncommitted_changes_dirty(self, temp_git_repo):
        """Test uncommitted changes detection on dirty repo"""
        temp_dir, applicator = temp_git_repo

        # Create a new file
        test_file = temp_dir / "test_file.py"
        test_file.write_text("print('test')")

        changes = applicator._has_uncommitted_changes()
        assert "test_file.py" in changes

    def test_metadata_persistence(self, temp_git_repo, sample_fix_proposal):
        """Test commit metadata persistence"""
        temp_dir, applicator = temp_git_repo

        # Create metadata
        metadata = CommitMetadata(
            commit_hash="abc123",
            timestamp=1234567890.0,
            file_path="test.py",
            issue_category="hardcoded_values",
            risk_level="low",
            confidence=0.92,
            fix_strategy="template",
            fix_id="FIX_001"
        )

        applicator.commits_metadata["abc123"] = metadata
        applicator._save_commits_metadata()

        # Load in new applicator instance
        applicator2 = GitApplicator(str(temp_dir))
        assert "abc123" in applicator2.commits_metadata
        assert applicator2.commits_metadata["abc123"].fix_id == "FIX_001"


# ============================================================================
# Integration Tests: Apply Fix Workflow
# ============================================================================

class TestApplyFixWorkflow:
    """Tests for complete apply fix workflow"""

    @pytest.mark.asyncio
    async def test_apply_low_risk_fix(self, temp_git_repo, sample_fix_proposal):
        """Test applying low-risk fix"""
        temp_dir, applicator = temp_git_repo

        # Create test file
        test_file = temp_dir / "test.py"
        test_file.write_text("x = 'hardcoded_value'\n")

        # Apply fix
        result = await applicator.apply_fix(
            file_path="test.py",
            fixed_code="x = os.getenv('VALUE', 'default')\n",
            proposal=sample_fix_proposal
        )

        assert result.success
        assert result.commit_hash is not None
        assert "hardcoded_values" in result.details["category"]

        # Verify file was changed
        assert test_file.read_text() == "x = os.getenv('VALUE', 'default')\n"

        # Verify metadata was saved
        assert result.commit_hash in applicator.commits_metadata

    @pytest.mark.asyncio
    async def test_apply_high_risk_fix_creates_branch(self, temp_git_repo, high_risk_proposal):
        """Test that high-risk fixes create feature branches"""
        temp_dir, applicator = temp_git_repo

        # Create test file
        test_file = temp_dir / "auth" / "security.py"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("password = request.args.get('password')\n")

        # Apply fix
        result = await applicator.apply_fix(
            file_path="auth/security.py",
            fixed_code="password = request.form.get('password', '')\nif not password:\n    raise ValueError('Password required')\n",
            proposal=high_risk_proposal,
            auto_branch=True
        )

        assert result.success

        # Verify we're back on original branch
        branch = applicator._get_current_branch()
        assert branch in {"main", "master"}

    @pytest.mark.asyncio
    async def test_apply_fix_dry_run(self, temp_git_repo, sample_fix_proposal):
        """Test dry-run mode doesn't actually commit"""
        temp_dir, applicator = temp_git_repo

        # Create test file
        test_file = temp_dir / "test.py"
        original_content = "x = 'hardcoded_value'\n"
        test_file.write_text(original_content)

        # Apply fix in dry-run mode
        result = await applicator.apply_fix(
            file_path="test.py",
            fixed_code="x = os.getenv('VALUE', 'default')\n",
            proposal=sample_fix_proposal,
            dry_run=True
        )

        assert result.success
        assert result.commit_hash is None  # No commit in dry-run

        # Verify file was NOT changed
        assert test_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_apply_fix_with_uncommitted_changes(self, temp_git_repo, sample_fix_proposal):
        """Test apply_fix fails with uncommitted changes"""
        temp_dir, applicator = temp_git_repo

        # Create uncommitted file
        test_file = temp_dir / "uncommitted.py"
        test_file.write_text("print('uncommitted')")

        # Try to apply fix
        result = await applicator.apply_fix(
            file_path="test.py",
            fixed_code="new code\n",
            proposal=sample_fix_proposal
        )

        assert not result.success
        assert "uncommitted" in result.message.lower()


# ============================================================================
# Unit Tests: RollbackManager
# ============================================================================

class TestRollbackManager:
    """Tests for RollbackManager"""

    def test_initialization(self, temp_git_repo):
        """Test RollbackManager initialization"""
        temp_dir, applicator = temp_git_repo
        rollback_mgr = RollbackManager(str(temp_dir), applicator)

        assert rollback_mgr.repo == temp_dir
        assert rollback_mgr.applicator is applicator

    @pytest.mark.asyncio
    async def test_rollback_history_empty(self, temp_git_repo):
        """Test rollback history on new repo"""
        temp_dir, applicator = temp_git_repo
        rollback_mgr = RollbackManager(str(temp_dir), applicator)

        history = await rollback_mgr.get_rollback_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_find_commits_by_category(self, temp_git_repo, sample_fix_proposal):
        """Test finding commits by category"""
        temp_dir, applicator = temp_git_repo
        rollback_mgr = RollbackManager(str(temp_dir), applicator)

        # Add some metadata
        applicator.commits_metadata["commit1"] = CommitMetadata(
            commit_hash="commit1",
            timestamp=100.0,
            file_path="test.py",
            issue_category="hardcoded_values",
            risk_level="low",
            confidence=0.9,
            fix_strategy="template",
            fix_id="FIX_001"
        )

        applicator.commits_metadata["commit2"] = CommitMetadata(
            commit_hash="commit2",
            timestamp=200.0,
            file_path="test2.py",
            issue_category="dead_code",
            risk_level="low",
            confidence=0.95,
            fix_strategy="ast",
            fix_id="FIX_002"
        )

        commits = rollback_mgr._find_commits_by_category("hardcoded_values")
        assert len(commits) == 1
        assert "commit1" in commits

    @pytest.mark.asyncio
    async def test_find_commits_by_file(self, temp_git_repo):
        """Test finding commits by file"""
        temp_dir, applicator = temp_git_repo
        rollback_mgr = RollbackManager(str(temp_dir), applicator)

        # Add metadata
        applicator.commits_metadata["commit1"] = CommitMetadata(
            commit_hash="commit1",
            timestamp=100.0,
            file_path="module/test.py",
            issue_category="hardcoded_values",
            risk_level="low",
            confidence=0.9,
            fix_strategy="template",
            fix_id="FIX_001"
        )

        commits = rollback_mgr._find_commits_by_file("module/test.py")
        assert len(commits) == 1
        assert "commit1" in commits


# ============================================================================
# Unit Tests: BranchManager
# ============================================================================

class TestBranchManager:
    """Tests for BranchManager"""

    def test_initialization(self, temp_git_repo):
        """Test BranchManager initialization"""
        temp_dir, applicator = temp_git_repo
        branch_mgr = BranchManager(str(temp_dir))

        assert branch_mgr.repo == temp_dir

    def test_feature_branch_name_generation(self, sample_fix_proposal):
        """Test feature branch name generation"""
        branch_mgr = BranchManager()

        branch_name = branch_mgr.get_feature_branch_name(sample_fix_proposal)

        assert "xterminator" in branch_name
        assert "fix_001" in branch_name.lower()
        assert "hardcoded" in branch_name.lower()
        assert len(branch_name) <= 50


# ============================================================================
# Safety and Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    @pytest.mark.asyncio
    async def test_apply_fix_to_missing_directory(self, temp_git_repo, sample_fix_proposal):
        """Test apply_fix with missing parent directory"""
        temp_dir, applicator = temp_git_repo

        # Try to apply fix to nested path
        result = await applicator.apply_fix(
            file_path="deep/nested/path/test.py",
            fixed_code="new code\n",
            proposal=sample_fix_proposal
        )

        assert result.success  # Should create directories
        assert (temp_dir / "deep" / "nested" / "path" / "test.py").exists()

    @pytest.mark.asyncio
    async def test_apply_fix_invalid_proposal(self, temp_git_repo):
        """Test apply_fix with malformed proposal"""
        temp_dir, applicator = temp_git_repo

        # Create invalid proposal (missing required fields)
        invalid_proposal = FixProposal(
            fix_id="",
            issue_category="",
            issue_severity="",
            risk_level=RiskLevel.LOW,
            fix_strategy=FixStrategy.SKIP,
            confidence=0.0,
            original_code="",
            explanation=""
        )

        # Should handle gracefully
        result = await applicator.apply_fix(
            file_path="test.py",
            fixed_code="code",
            proposal=invalid_proposal,
            dry_run=True
        )

        assert result.success


# ============================================================================
# Performance and Edge Cases
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Tests for performance and edge cases"""

    @pytest.mark.asyncio
    async def test_apply_multiple_fixes_sequentially(self, temp_git_repo, sample_fix_proposal):
        """Test applying multiple fixes in sequence"""
        temp_dir, applicator = temp_git_repo

        for i in range(3):
            test_file = temp_dir / f"test{i}.py"
            test_file.write_text(f"# File {i}\n")

            proposal = FixProposal(
                fix_id=f"FIX_{i:03d}",
                issue_category="formatting",
                issue_severity="low",
                risk_level=RiskLevel.LOW,
                fix_strategy=FixStrategy.TEMPLATE,
                confidence=0.95,
                original_code=f"# File {i}\n",
                proposed_code=f"# File {i} - Fixed\n",
                explanation=f"Fix {i}",
                context=sample_fix_proposal.context,
                safe_to_autofix=True,
                requires_approval=False
            )

            result = await applicator.apply_fix(
                file_path=f"test{i}.py",
                fixed_code=f"# File {i} - Fixed\n",
                proposal=proposal
            )

            assert result.success

        # Verify all files were created and changed
        for i in range(3):
            assert (temp_dir / f"test{i}.py").read_text() == f"# File {i} - Fixed\n"

        # Verify all commits were recorded
        assert len(applicator.commits_metadata) == 3

    @pytest.mark.asyncio
    async def test_large_file_handling(self, temp_git_repo, sample_fix_proposal):
        """Test handling of large files"""
        temp_dir, applicator = temp_git_repo

        # Create a large file
        large_code = "# Large file\n" + "x = 1\n" * 10000
        test_file = temp_dir / "large.py"
        test_file.write_text(large_code)

        result = await applicator.apply_fix(
            file_path="large.py",
            fixed_code=large_code + "# Fixed\n",
            proposal=sample_fix_proposal
        )

        assert result.success

    def test_special_characters_in_path(self, temp_git_repo, sample_fix_proposal):
        """Test handling of special characters in file paths"""
        # Note: Some special chars may not work on all platforms
        branch_mgr = BranchManager()

        proposal = FixProposal(
            fix_id="FIX-001",
            issue_category="test-category",
            issue_severity="low",
            risk_level=RiskLevel.LOW,
            fix_strategy=FixStrategy.TEMPLATE,
            confidence=0.9,
            original_code="",
            explanation="Test with special chars",
            context=sample_fix_proposal.context,
            safe_to_autofix=True,
            requires_approval=False
        )

        branch_name = branch_mgr.get_feature_branch_name(proposal)

        # Should be sanitized
        assert " " not in branch_name
        assert len(branch_name) <= 50


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
