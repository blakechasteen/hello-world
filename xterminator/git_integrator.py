"""
Git Integration for xTerminator
================================

Phase 4: Automated Git operations for applying code fixes.

Features:
- File writing with atomic backups
- Git branch creation and management
- Commit message generation
- Pull request creation via gh CLI
- Rollback support

Author: The Barnyard Brigade
Date: November 16, 2025
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GitConfig:
    """Git integration configuration."""
    auto_create_branch: bool = True
    branch_prefix: str = "autofix"
    require_clean_tree: bool = False
    auto_commit: bool = True
    auto_push: bool = False
    auto_pr: bool = False
    pr_reviewers: List[str] = None

    def __post_init__(self):
        if self.pr_reviewers is None:
            self.pr_reviewers = []


@dataclass
class FileBackup:
    """File backup metadata."""
    original_path: str
    backup_path: str
    timestamp: str
    sha256: str


@dataclass
class GitOperation:
    """Git operation result."""
    success: bool
    operation: str
    branch: Optional[str] = None
    commit: Optional[str] = None
    pr_url: Optional[str] = None
    error: Optional[str] = None
    rollback_available: bool = False


class GitIntegrator:
    """
    Manages Git operations for applying code fixes.

    Provides atomic file writes, branch management, commits, and PR creation.
    """

    def __init__(self, config: Optional[GitConfig] = None, repo_path: Optional[str] = None):
        """
        Initialize Git integrator.

        Args:
            config: Git integration configuration
            repo_path: Path to Git repository (default: current directory)
        """
        self.config = config or GitConfig()
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.backups: Dict[str, FileBackup] = {}
        self.current_branch: Optional[str] = None
        self.original_branch: Optional[str] = None

    def _safe_decode_stderr(self, stderr) -> str:
        """Safely decode stderr from subprocess, handling both bytes and str."""
        if stderr is None:
            return ""
        if isinstance(stderr, bytes):
            return stderr.decode('utf-8', errors='replace')
        return str(stderr)

    # ================================================================
    # FILE OPERATIONS
    # ================================================================

    def write_file_with_backup(
        self,
        file_path: str,
        content: str,
        create_backup: bool = True
    ) -> FileBackup:
        """
        Write file atomically with backup.

        Args:
            file_path: Path to file to write
            content: New file content
            create_backup: Whether to create .bak backup

        Returns:
            FileBackup metadata

        Raises:
            IOError: If file write fails
        """
        file_path = Path(file_path)

        # Create backup if file exists
        backup_path = None
        if file_path.exists() and create_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f".bak.{timestamp}")

            # Copy to backup
            shutil.copy2(file_path, backup_path)

            # Calculate SHA256 for verification
            import hashlib
            with open(backup_path, 'rb') as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()

            backup = FileBackup(
                original_path=str(file_path),
                backup_path=str(backup_path),
                timestamp=timestamp,
                sha256=sha256
            )
            self.backups[str(file_path)] = backup
        else:
            sha256 = ""
            backup = FileBackup(
                original_path=str(file_path),
                backup_path="",
                timestamp="",
                sha256=""
            )

        # Write new content atomically
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            dir=file_path.parent,
            delete=False
        )

        try:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            # Atomic rename
            os.replace(temp_file.name, file_path)

        except Exception as e:
            # Clean up temp file on failure
            try:
                os.unlink(temp_file.name)
            except:
                pass
            raise IOError(f"Failed to write {file_path}: {e}")

        return backup

    def restore_from_backup(self, file_path: str) -> bool:
        """
        Restore file from backup.

        Args:
            file_path: Path to file to restore

        Returns:
            True if restored, False if no backup
        """
        if file_path not in self.backups:
            return False

        backup = self.backups[file_path]
        if not backup.backup_path or not Path(backup.backup_path).exists():
            return False

        # Restore from backup
        shutil.copy2(backup.backup_path, file_path)
        return True

    def cleanup_backups(self, keep_latest: int = 5):
        """
        Clean up old backup files.

        Args:
            keep_latest: Number of latest backups to keep per file
        """
        # Group backups by original file
        by_file: Dict[str, List[FileBackup]] = {}
        for backup in self.backups.values():
            if backup.original_path not in by_file:
                by_file[backup.original_path] = []
            by_file[backup.original_path].append(backup)

        # Sort by timestamp and remove old backups
        for file_path, backups in by_file.items():
            backups.sort(key=lambda b: b.timestamp, reverse=True)

            for old_backup in backups[keep_latest:]:
                if old_backup.backup_path and Path(old_backup.backup_path).exists():
                    try:
                        os.unlink(old_backup.backup_path)
                    except:
                        pass

    # ================================================================
    # GIT OPERATIONS
    # ================================================================

    def is_git_repo(self) -> bool:
        """Check if current directory is a Git repository."""
        try:
            subprocess.run(
                ['git', 'rev-parse', '--git-dir'], cwd=self.repo_path,
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_current_branch(self) -> Optional[str]:
        """Get current Git branch name."""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'], cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def is_tree_clean(self) -> bool:
        """Check if Git working tree is clean."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'], cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return len(result.stdout.strip()) == 0
        except subprocess.CalledProcessError:
            return False

    def create_branch(self, branch_name: Optional[str] = None) -> GitOperation:
        """
        Create and checkout new Git branch.

        Args:
            branch_name: Branch name (auto-generated if None)

        Returns:
            GitOperation result
        """
        if not self.is_git_repo():
            return GitOperation(
                success=False,
                operation="create_branch",
                error="Not a Git repository"
            )

        # Save original branch
        self.original_branch = self.get_current_branch()

        # Check for clean tree if required
        if self.config.require_clean_tree and not self.is_tree_clean():
            return GitOperation(
                success=False,
                operation="create_branch",
                error="Working tree not clean"
            )

        # Generate branch name if not provided
        if not branch_name:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            branch_name = f"{self.config.branch_prefix}/{timestamp}"

        # Create and checkout branch
        try:
            subprocess.run(
                ['git', 'checkout', '-b', branch_name], cwd=self.repo_path,
                capture_output=True,
                check=True
            )

            self.current_branch = branch_name

            return GitOperation(
                success=True,
                operation="create_branch",
                branch=branch_name,
                rollback_available=True
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="create_branch",
                error=f"Failed to create branch: {e.stderr.decode() if e.stderr else str(e)}"
            )

    def stage_files(self, files: List[str]) -> GitOperation:
        """
        Stage files for commit.

        Args:
            files: List of file paths to stage

        Returns:
            GitOperation result
        """
        try:
            # Convert absolute paths to relative paths
            relative_files = []
            for f in files:
                f_path = Path(f)
                if f_path.is_absolute():
                    try:
                        relative_files.append(str(f_path.relative_to(self.repo_path)))
                    except ValueError:
                        # File is outside repo, use absolute path
                        relative_files.append(str(f_path))
                else:
                    relative_files.append(str(f_path))

            subprocess.run(
                ['git', 'add'] + relative_files,
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )

            return GitOperation(
                success=True,
                operation="stage_files"
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="stage_files",
                error=f"Failed to stage files: {e.stderr.decode() if e.stderr else str(e)}"
            )

    def generate_commit_message(
        self,
        fix_metadata: Dict[str, Any]
    ) -> str:
        """
        Generate commit message from fix metadata.

        Args:
            fix_metadata: Fix metadata dict

        Returns:
            Formatted commit message
        """
        category = fix_metadata.get('category', 'code_quality')
        file_path = fix_metadata.get('file_path', 'unknown')
        message = fix_metadata.get('message', 'Apply auto-fix')
        confidence = fix_metadata.get('confidence', 0.0)

        # Format: Fix [category]: [message] in [file]
        commit_msg = f"Fix {category}: {message}\n\n"
        commit_msg += f"File: {file_path}\n"
        commit_msg += f"Confidence: {confidence:.1%}\n"
        commit_msg += f"Auto-generated by xTerminator\n"

        return commit_msg

    def commit(
        self,
        message: str,
        files: Optional[List[str]] = None
    ) -> GitOperation:
        """
        Create Git commit.

        Args:
            message: Commit message
            files: Files to stage (all staged if None)

        Returns:
            GitOperation result
        """
        # Stage files if provided
        if files:
            stage_result = self.stage_files(files)
            if not stage_result.success:
                return stage_result

        # Create commit
        try:
            result = subprocess.run(
                ['git', 'commit', '-m', message], cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            # Get commit SHA
            commit_sha = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()

            return GitOperation(
                success=True,
                operation="commit",
                commit=commit_sha,
                rollback_available=True
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="commit",
                error=f"Failed to commit: {e.stderr.decode() if e.stderr else str(e)}"
            )

    def push(self, set_upstream: bool = True) -> GitOperation:
        """
        Push current branch to remote.

        Args:
            set_upstream: Set upstream tracking branch

        Returns:
            GitOperation result
        """
        if not self.current_branch:
            return GitOperation(
                success=False,
                operation="push",
                error="No current branch"
            )

        try:
            cmd = ['git', 'push']
            if set_upstream:
                cmd.extend(['-u', 'origin', self.current_branch])

            subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                check=True
            )

            return GitOperation(
                success=True,
                operation="push",
                branch=self.current_branch
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="push",
                error=f"Failed to push: {e.stderr.decode() if e.stderr else str(e)}"
            )

    def create_pull_request(
        self,
        title: str,
        body: str,
        base: str = "main"
    ) -> GitOperation:
        """
        Create pull request using gh CLI.

        Args:
            title: PR title
            body: PR description
            base: Base branch (default: main)

        Returns:
            GitOperation result with PR URL
        """
        if not self.current_branch:
            return GitOperation(
                success=False,
                operation="create_pr",
                error="No current branch"
            )

        try:
            # Build gh pr create command
            cmd = [
                'gh', 'pr', 'create',
                '--title', title,
                '--body', body,
                '--base', base
            ]

            # Add reviewers if configured
            if self.config.pr_reviewers:
                cmd.extend(['--reviewer', ','.join(self.config.pr_reviewers)])

            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            # Extract PR URL from output
            pr_url = result.stdout.strip()

            return GitOperation(
                success=True,
                operation="create_pr",
                branch=self.current_branch,
                pr_url=pr_url
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="create_pr",
                error=f"Failed to create PR: {e.stderr.decode() if e.stderr else str(e)}"
            )

    # ================================================================
    # ROLLBACK OPERATIONS
    # ================================================================

    def rollback_commit(self) -> GitOperation:
        """
        Rollback last commit (soft reset).

        Returns:
            GitOperation result
        """
        try:
            subprocess.run(
                ['git', 'reset', '--soft', 'HEAD~1'], cwd=self.repo_path,
                capture_output=True,
                check=True
            )

            return GitOperation(
                success=True,
                operation="rollback_commit"
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="rollback_commit",
                error=f"Failed to rollback: {e.stderr.decode() if e.stderr else str(e)}"
            )

    def rollback_branch(self) -> GitOperation:
        """
        Rollback to original branch and delete autofix branch.

        Returns:
            GitOperation result
        """
        if not self.original_branch:
            return GitOperation(
                success=False,
                operation="rollback_branch",
                error="No original branch saved"
            )

        autofix_branch = self.current_branch

        try:
            # Checkout original branch
            subprocess.run(
                ['git', 'checkout', self.original_branch], cwd=self.repo_path,
                capture_output=True,
                check=True
            )

            # Delete autofix branch
            if autofix_branch and autofix_branch != self.original_branch:
                subprocess.run(
                    ['git', 'branch', '-D', autofix_branch], cwd=self.repo_path,
                    capture_output=True,
                    check=True
                )

            self.current_branch = self.original_branch

            return GitOperation(
                success=True,
                operation="rollback_branch"
            )

        except subprocess.CalledProcessError as e:
            return GitOperation(
                success=False,
                operation="rollback_branch",
                error=f"Failed to rollback branch: {e.stderr.decode() if e.stderr else str(e)}"
            )

    # ================================================================
    # HIGH-LEVEL WORKFLOWS
    # ================================================================

    def apply_fix_with_git(
        self,
        file_path: str,
        fixed_content: str,
        fix_metadata: Dict[str, Any],
        create_pr: bool = False
    ) -> GitOperation:
        """
        Complete workflow: write file → commit → push → PR.

        Args:
            file_path: File to fix
            fixed_content: Fixed file content
            fix_metadata: Fix metadata for commit message
            create_pr: Whether to create PR

        Returns:
            GitOperation result with all operations
        """
        operations = []

        # 1. Create branch if configured
        if self.config.auto_create_branch:
            branch_result = self.create_branch()
            operations.append(branch_result)

            if not branch_result.success:
                return branch_result

        # 2. Write file with backup
        try:
            self.write_file_with_backup(file_path, fixed_content)
        except Exception as e:
            # Rollback branch if created
            if self.config.auto_create_branch:
                self.rollback_branch()

            return GitOperation(
                success=False,
                operation="write_file",
                error=str(e),
                rollback_available=False
            )

        # 3. Commit if configured
        if self.config.auto_commit:
            commit_msg = self.generate_commit_message(fix_metadata)
            commit_result = self.commit(commit_msg, files=[file_path])
            operations.append(commit_result)

            if not commit_result.success:
                # Restore file from backup
                self.restore_from_backup(file_path)
                # Rollback branch if created
                if self.config.auto_create_branch:
                    self.rollback_branch()

                return commit_result

        # 4. Push if configured
        pr_url = None
        if self.config.auto_push:
            push_result = self.push()
            operations.append(push_result)

            if not push_result.success:
                # Rollback commit
                self.rollback_commit()
                # Restore file
                self.restore_from_backup(file_path)
                # Rollback branch
                if self.config.auto_create_branch:
                    self.rollback_branch()

                return push_result

            # 5. Create PR if requested
            if create_pr or self.config.auto_pr:
                pr_title = f"Auto-fix: {fix_metadata.get('category', 'code_quality')}"
                pr_body = self.generate_pr_description([fix_metadata])

                pr_result = self.create_pull_request(pr_title, pr_body)
                operations.append(pr_result)

                if pr_result.success:
                    pr_url = pr_result.pr_url

        # Success!
        return GitOperation(
            success=True,
            operation="apply_fix_complete",
            branch=self.current_branch,
            commit=operations[-2].commit if len(operations) >= 2 else None,
            pr_url=pr_url,
            rollback_available=True
        )

    def generate_pr_description(self, fixes: List[Dict[str, Any]]) -> str:
        """
        Generate PR description from multiple fixes.

        Args:
            fixes: List of fix metadata dicts

        Returns:
            Formatted PR description
        """
        desc = "## Auto-Generated Fixes\n\n"
        desc += f"This PR contains {len(fixes)} automated code quality fixes.\n\n"

        # Group by category
        by_category = {}
        for fix in fixes:
            category = fix.get('category', 'other')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(fix)

        desc += "### Fixes by Category\n\n"
        for category, category_fixes in by_category.items():
            desc += f"**{category.replace('_', ' ').title()}** ({len(category_fixes)} fixes)\n"
            for fix in category_fixes[:5]:  # Show first 5
                file_path = fix.get('file_path', 'unknown')
                message = fix.get('message', 'Applied fix')
                desc += f"- `{file_path}`: {message}\n"

            if len(category_fixes) > 5:
                desc += f"- ... and {len(category_fixes) - 5} more\n"

            desc += "\n"

        desc += "### Testing\n\n"
        desc += "- [ ] All tests passing\n"
        desc += "- [ ] Code review complete\n\n"

        desc += "---\n"
        desc += "*Generated by xTerminator Auto-Fix System*\n"

        return desc
