"""
Git Applicator
==============

Phase 4 of xTerminator: Git Integration

Safely applies fixes as atomic git commits with full rollback support.

Architecture:
1. GitApplicator - Applies fixes as git commits
2. RollbackManager - Manages undo operations
3. BranchManager - Handles feature branches for high-risk fixes

Philosophy:
"Templeton commits carefully, rolls back fearlessly!" - The Rat's Wisdom
"""

import subprocess
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

from .xterminator_types import FixProposal, RiskLevel, FixStrategy


@dataclass
class CommitMetadata:
    """Metadata for tracking xTerminator commits"""
    commit_hash: str
    timestamp: float
    file_path: str
    issue_category: str
    risk_level: str
    confidence: float
    fix_strategy: str
    fix_id: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class GitOperationResult:
    """Result of a git operation"""
    success: bool
    message: str
    commit_hash: Optional[str] = None
    details: Dict = field(default_factory=dict)


class GitApplicator:
    """
    Applies fixes as atomic git commits.

    Philosophy:
    - One commit per fix (easy cherry-pick/revert)
    - Descriptive messages with metadata
    - Dry-run support for testing
    - Auto-branch for high-risk fixes
    """

    def __init__(self, repo_path: str = "."):
        """
        Initialize GitApplicator.

        Args:
            repo_path: Path to git repository (default: current directory)

        Raises:
            RuntimeError: If not a git repository
        """
        self.repo = Path(repo_path)
        self.metadata_file = self.repo / ".xterminator" / "commits.json"
        self._verify_git_repo()
        self._ensure_metadata_dir()
        self._load_commits_metadata()

    def _verify_git_repo(self) -> None:
        """Verify this is a git repository"""
        try:
            result = self._run_git(["rev-parse", "--git-dir"])
            if result.returncode != 0:
                raise RuntimeError(f"Not a git repository: {self.repo}")
        except Exception as e:
            raise RuntimeError(f"Git verification failed: {e}")

    def _ensure_metadata_dir(self) -> None:
        """Ensure metadata directory exists"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_commits_metadata(self) -> None:
        """Load existing commits metadata"""
        self.commits_metadata: Dict[str, CommitMetadata] = {}

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for commit_hash, meta in data.items():
                        self.commits_metadata[commit_hash] = CommitMetadata(**meta)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")

    def _save_commits_metadata(self) -> None:
        """Save commits metadata to file"""
        try:
            data = {
                commit_hash: meta.to_dict()
                for commit_hash, meta in self.commits_metadata.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    async def apply_fix(
        self,
        file_path: str,
        fixed_code: str,
        proposal: FixProposal,
        dry_run: bool = False,
        auto_branch: bool = True
    ) -> GitOperationResult:
        """
        Apply fix and create git commit.

        Args:
            file_path: Path to file being fixed
            fixed_code: Fixed code content
            proposal: FixProposal with fix details
            dry_run: If True, don't actually commit
            auto_branch: If True, create branch for HIGH/CRITICAL fixes

        Returns:
            GitOperationResult with success status and commit hash
        """
        # For dry-run, skip uncommitted changes check
        if not dry_run:
            # Check for uncommitted changes (exclude the file we're about to fix)
            uncommitted = self._has_uncommitted_changes()
            # Filter out:
            # 1. The file we're fixing
            # 2. xTerminator metadata directory (with/without trailing slash)
            # 3. Parent directories of file we're fixing
            file_parts = Path(file_path).parts
            uncommitted = [
                f for f in uncommitted
                if f != file_path and
                   f != ".xterminator" and
                   not f.startswith(".xterminator/") and
                   not f.startswith(".xterminator\\") and
                   not any(f.startswith(part + "/") or f.startswith(part + "\\") for part in file_parts[:-1])
            ]
            if uncommitted:
                return GitOperationResult(
                    success=False,
                    message="Repository has uncommitted changes. Commit or stash them first.",
                    details={"uncommitted_files": uncommitted}
                )

        # Handle branching for high-risk fixes
        original_branch = None
        if auto_branch and proposal.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            original_branch = self._get_current_branch()
            branch_name = self._create_feature_branch(proposal)
            if not branch_name:
                return GitOperationResult(
                    success=False,
                    message=f"Failed to create feature branch"
                )

        try:
            if dry_run:
                return self._dry_run_apply_fix(file_path, fixed_code, proposal)

            # Write fixed code
            file_full_path = self.repo / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(file_full_path, 'w') as f:
                    f.write(fixed_code)
            except Exception as e:
                return GitOperationResult(
                    success=False,
                    message=f"Failed to write file: {e}"
                )

            # Stage file
            result = self._git_add(file_path)
            if result.returncode != 0:
                return GitOperationResult(
                    success=False,
                    message=f"Failed to stage file: {result.stderr}"
                )

            # Create commit
            commit_msg = self._generate_commit_message(file_path, proposal)
            result = self._git_commit(commit_msg)

            if result.returncode != 0:
                return GitOperationResult(
                    success=False,
                    message=f"Failed to create commit: {result.stderr}"
                )

            # Get commit hash
            commit_hash = self._get_latest_commit_hash()

            # Save metadata
            if commit_hash:
                metadata = CommitMetadata(
                    commit_hash=commit_hash,
                    timestamp=datetime.now().timestamp(),
                    file_path=file_path,
                    issue_category=proposal.issue_category,
                    risk_level=proposal.risk_level.value,
                    confidence=proposal.confidence,
                    fix_strategy=proposal.fix_strategy.value,
                    fix_id=proposal.fix_id
                )
                self.commits_metadata[commit_hash] = metadata
                self._save_commits_metadata()

            return GitOperationResult(
                success=True,
                message=f"Fix applied and committed successfully",
                commit_hash=commit_hash,
                details={
                    "file": file_path,
                    "category": proposal.issue_category,
                    "risk": proposal.risk_level.value,
                    "confidence": proposal.confidence,
                    "strategy": proposal.fix_strategy.value
                }
            )

        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Error applying fix: {e}"
            )
        finally:
            # Switch back to original branch if needed
            if original_branch:
                self._checkout_branch(original_branch)

    def _dry_run_apply_fix(
        self,
        file_path: str,
        fixed_code: str,
        proposal: FixProposal
    ) -> GitOperationResult:
        """Simulate applying fix without committing"""
        commit_msg = self._generate_commit_message(file_path, proposal)

        return GitOperationResult(
            success=True,
            message="[DRY RUN] Fix would be applied and committed",
            details={
                "file": file_path,
                "file_size": len(fixed_code),
                "commit_message_preview": commit_msg[:100] + "...",
                "category": proposal.issue_category,
                "risk": proposal.risk_level.value,
                "confidence": proposal.confidence
            }
        )

    def _generate_commit_message(
        self,
        file_path: str,
        proposal: FixProposal
    ) -> str:
        """Generate descriptive commit message with metadata"""
        category = proposal.issue_category
        line = proposal.context.line_number if proposal.context else "unknown"
        confidence = proposal.confidence
        strategy = proposal.fix_strategy.value

        # First line (conventional commit format)
        first_line = f"fix({category}): {proposal.explanation.split(chr(10))[0][:50]}"

        # Detailed body
        body = f"""
File: {file_path}
Line: {line}
Strategy: {strategy}
Confidence: {confidence:.2f}
Risk: {proposal.risk_level.value}
Fix ID: {proposal.fix_id}

{proposal.explanation}

--- xTerminator v0.1.0 Phase 4: Git Integration ---"""

        return first_line + body

    def _create_feature_branch(self, proposal: FixProposal) -> Optional[str]:
        """Create feature branch for high-risk fix"""
        branch_name = f"xterminator/{proposal.fix_id}/{proposal.issue_category}"
        branch_name = branch_name.replace(' ', '_').lower()[:50]  # Sanitize

        result = self._run_git(["checkout", "-b", branch_name])

        if result.returncode != 0:
            print(f"Failed to create branch {branch_name}: {result.stderr}")
            return None

        return branch_name

    def _get_current_branch(self) -> str:
        """Get current branch name"""
        result = self._run_git(["branch", "--show-current"])
        return result.stdout.strip() if result.returncode == 0 else "main"

    def _checkout_branch(self, branch_name: str) -> bool:
        """Switch to specified branch"""
        result = self._run_git(["checkout", branch_name])
        return result.returncode == 0

    def _has_uncommitted_changes(self) -> List[str]:
        """Check for uncommitted changes"""
        result = self._run_git(["status", "--porcelain"])
        if result.returncode != 0:
            return []

        return [line.split()[-1] for line in result.stdout.strip().split('\n') if line.strip()]

    def _get_latest_commit_hash(self) -> Optional[str]:
        """Get hash of latest commit"""
        result = self._run_git(["rev-parse", "HEAD"])
        return result.stdout.strip() if result.returncode == 0 else None

    def _git_add(self, file_path: str) -> subprocess.CompletedProcess:
        """Stage file for commit"""
        return self._run_git(["add", file_path])

    def _git_commit(self, message: str) -> subprocess.CompletedProcess:
        """Create commit"""
        return self._run_git(["commit", "-m", message])

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run git command"""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo)] + args,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=args,
                returncode=124,
                stdout="",
                stderr="Command timed out"
            )


class RollbackManager:
    """
    Manages rollback of xTerminator fixes.

    Provides safe undo operations:
    - Rollback last N commits
    - Rollback by category
    - Rollback by file
    - Safety checks (no rollback of pushed commits)

    Philosophy:
    "Every good rat knows when to retreat!" - Templeton
    """

    def __init__(self, repo_path: str = ".", applicator: Optional[GitApplicator] = None):
        """
        Initialize RollbackManager.

        Args:
            repo_path: Path to git repository
            applicator: GitApplicator instance (creates if None)
        """
        self.repo = Path(repo_path)
        self.applicator = applicator or GitApplicator(repo_path)

    async def rollback_last(self, n: int = 1, force: bool = False) -> GitOperationResult:
        """
        Rollback last N xTerminator commits.

        Args:
            n: Number of commits to rollback
            force: If True, rollback even if pushed

        Returns:
            GitOperationResult indicating success/failure
        """
        commits = self._find_xterminator_commits(n)

        if not commits:
            return GitOperationResult(
                success=False,
                message=f"No xTerminator commits found to rollback"
            )

        reverted = []
        failed = []

        for commit in commits:
            # Check if pushed
            if self._is_pushed(commit) and not force:
                failed.append((commit, "Commit already pushed (use --force to rollback)"))
                continue

            result = self._git_revert(commit)

            if result.returncode != 0:
                failed.append((commit, result.stderr))
            else:
                reverted.append(commit)

        if reverted and not failed:
            # Update metadata
            for commit in reverted:
                if commit in self.applicator.commits_metadata:
                    del self.applicator.commits_metadata[commit]
                self.applicator._save_commits_metadata()

            return GitOperationResult(
                success=True,
                message=f"Successfully reverted {len(reverted)} commit(s)",
                details={
                    "reverted_commits": reverted,
                    "revert_count": len(reverted)
                }
            )
        elif reverted and failed:
            return GitOperationResult(
                success=False,
                message=f"Partially reverted: {len(reverted)} succeeded, {len(failed)} failed",
                details={
                    "reverted": reverted,
                    "failed": failed
                }
            )
        else:
            return GitOperationResult(
                success=False,
                message=f"Failed to revert commits",
                details={
                    "failed": failed
                }
            )

    async def rollback_category(self, category: str) -> GitOperationResult:
        """
        Rollback all fixes in a specific category.

        Args:
            category: Issue category to rollback

        Returns:
            GitOperationResult indicating success/failure
        """
        commits = self._find_commits_by_category(category)

        if not commits:
            return GitOperationResult(
                success=False,
                message=f"No commits found for category: {category}"
            )

        return await self.rollback_last(len(commits), force=False)

    async def rollback_file(self, file_path: str) -> GitOperationResult:
        """
        Rollback all fixes to a specific file.

        Args:
            file_path: Path to file

        Returns:
            GitOperationResult indicating success/failure
        """
        commits = self._find_commits_by_file(file_path)

        if not commits:
            return GitOperationResult(
                success=False,
                message=f"No commits found for file: {file_path}"
            )

        return await self.rollback_last(len(commits), force=False)

    def _find_xterminator_commits(self, limit: int = 10) -> List[str]:
        """Find recent xTerminator commits"""
        metadata = self.applicator.commits_metadata

        # Get commits sorted by timestamp (newest first)
        sorted_commits = sorted(
            metadata.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )

        return [commit_hash for commit_hash, _ in sorted_commits[:limit]]

    def _find_commits_by_category(self, category: str) -> List[str]:
        """Find commits for specific category"""
        return [
            commit_hash
            for commit_hash, meta in self.applicator.commits_metadata.items()
            if meta.issue_category.lower() == category.lower()
        ]

    def _find_commits_by_file(self, file_path: str) -> List[str]:
        """Find commits for specific file"""
        return [
            commit_hash
            for commit_hash, meta in self.applicator.commits_metadata.items()
            if meta.file_path == file_path
        ]

    def _is_pushed(self, commit_hash: str) -> bool:
        """Check if commit has been pushed"""
        # Check if commit exists in remote
        result = subprocess.run(
            ["git", "-C", str(self.repo), "log", "--oneline", "origin..HEAD"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # If commit appears in "ahead of origin", it's not pushed
        return commit_hash not in result.stdout

    def _git_revert(self, commit_hash: str) -> subprocess.CompletedProcess:
        """Revert a commit"""
        return subprocess.run(
            ["git", "-C", str(self.repo), "revert", "--no-edit", commit_hash],
            capture_output=True,
            text=True,
            timeout=30
        )

    async def get_rollback_history(self) -> List[Dict]:
        """Get history of all xTerminator commits"""
        commits = sorted(
            self.applicator.commits_metadata.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )

        return [
            {
                "commit": commit_hash,
                "timestamp": datetime.fromtimestamp(meta.timestamp).isoformat(),
                "file": meta.file_path,
                "category": meta.issue_category,
                "risk": meta.risk_level,
                "confidence": meta.confidence,
                "strategy": meta.fix_strategy,
                "fix_id": meta.fix_id
            }
            for commit_hash, meta in commits
        ]


class BranchManager:
    """
    Manages feature branches for high-risk fixes.

    Provides:
    - Auto-branch creation for HIGH/CRITICAL risk
    - Branch merging/cleanup
    - PR preparation
    """

    def __init__(self, repo_path: str = "."):
        """Initialize BranchManager"""
        self.repo = Path(repo_path)

    def get_feature_branch_name(self, proposal: FixProposal) -> str:
        """Generate feature branch name"""
        branch_name = f"xterminator/{proposal.fix_id}/{proposal.issue_category}"
        # Sanitize: lowercase, replace spaces with underscores, limit length
        branch_name = branch_name.replace(' ', '_').lower()[:50]
        return branch_name

    async def merge_branch(self, branch_name: str, delete_after: bool = True) -> GitOperationResult:
        """
        Merge feature branch to main.

        Args:
            branch_name: Name of branch to merge
            delete_after: If True, delete branch after merging

        Returns:
            GitOperationResult indicating success/failure
        """
        # Switch to main
        result = subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "main"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return GitOperationResult(
                success=False,
                message=f"Failed to checkout main: {result.stderr}"
            )

        # Merge branch
        result = subprocess.run(
            ["git", "-C", str(self.repo), "merge", branch_name],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return GitOperationResult(
                success=False,
                message=f"Failed to merge {branch_name}: {result.stderr}"
            )

        # Delete branch if requested
        if delete_after:
            subprocess.run(
                ["git", "-C", str(self.repo), "branch", "-d", branch_name],
                capture_output=True,
                timeout=10
            )

        return GitOperationResult(
            success=True,
            message=f"Successfully merged {branch_name} to main",
            details={"branch": branch_name}
        )
