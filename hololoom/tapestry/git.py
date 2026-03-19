"""
Git Integration for Tapestry

Git operations for session tracking:
- Commit staged changes
- Ensure clean working directory
- Rollback to specific commits
- Track commit hashes

Created: December 2025
"""

import asyncio
import logging
import subprocess
from pathlib import Path

from hololoom.tapestry.protocol import DirtyWorkingDirectoryError

logger = logging.getLogger(__name__)


class GitIntegration:
    """
    Git operations for tapestry tracking.

    Provides async wrappers around git commands for:
    - Committing woven threads
    - Ensuring clean working directory
    - Rolling back tangled threads
    - Tracking commit history
    """

    def __init__(self, repo_path: str | None = None):
        """
        Initialize Git integration.

        Args:
            repo_path: Path to git repository. None uses current directory.
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()

    async def _run(self, *args: str, check: bool = True) -> str:
        """
        Run git command asynchronously.

        Args:
            *args: Command arguments (e.g., "status", "--porcelain")
            check: If True, raise on non-zero exit

        Returns:
            Command stdout as string
        """
        cmd = ["git"] + list(args)
        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.repo_path
            )
            stdout, stderr = await proc.communicate()

            if check and proc.returncode != 0:
                error_msg = stderr.decode().strip() or f"Command failed with code {proc.returncode}"
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stdout, stderr
                )

            return stdout.decode().strip()

        except FileNotFoundError:
            logger.warning("Git not found in PATH")
            raise RuntimeError("Git is not installed or not in PATH")

    async def is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        try:
            await self._run("rev-parse", "--git-dir")
            return True
        except (subprocess.CalledProcessError, RuntimeError):
            return False

    async def ensure_clean(self) -> bool:
        """
        Ensure working directory is clean.

        Raises:
            DirtyWorkingDirectoryError: If uncommitted changes exist

        Returns:
            True if clean
        """
        if not await self.is_git_repo():
            logger.debug("Not a git repository, skipping clean check")
            return True

        status = await self._run("status", "--porcelain")
        if status.strip():
            # Has uncommitted changes
            raise DirtyWorkingDirectoryError(
                f"Working directory has uncommitted changes:\n{status[:500]}"
            )
        return True

    async def get_current_commit(self) -> str:
        """
        Get current HEAD commit hash.

        Returns:
            Short commit hash (7 chars)
        """
        if not await self.is_git_repo():
            return ""
        try:
            return await self._run("rev-parse", "--short", "HEAD")
        except subprocess.CalledProcessError:
            return ""

    async def get_current_branch(self) -> str:
        """Get current branch name."""
        if not await self.is_git_repo():
            return ""
        try:
            return await self._run("rev-parse", "--abbrev-ref", "HEAD")
        except subprocess.CalledProcessError:
            return ""

    async def commit(self, message: str, add_all: bool = True) -> str:
        """
        Commit staged changes.

        Args:
            message: Commit message
            add_all: If True, stage all changes first

        Returns:
            Commit hash
        """
        if not await self.is_git_repo():
            logger.warning("Not a git repository, skipping commit")
            return ""

        # Stage all changes
        if add_all:
            await self._run("add", "-A")

        # Check if there are staged changes
        status = await self._run("diff", "--cached", "--stat")
        if not status.strip():
            logger.debug("No staged changes to commit")
            return await self.get_current_commit()

        # Commit with message
        full_message = f"{message}\n\n🤖 Generated with HoloLoom Tapestry"
        await self._run("commit", "-m", full_message)

        # Return new commit hash
        commit_hash = await self.get_current_commit()
        logger.info(f"Created commit: {commit_hash} - {message[:50]}")
        return commit_hash

    async def rollback(self, commit_hash: str) -> None:
        """
        Rollback to specific commit.

        WARNING: This is a hard reset - uncommitted changes will be lost.

        Args:
            commit_hash: Commit to reset to
        """
        if not await self.is_git_repo():
            logger.warning("Not a git repository, skipping rollback")
            return

        logger.warning(f"Rolling back to {commit_hash}")
        await self._run("reset", "--hard", commit_hash)

    async def get_changed_files(self, since_commit: str | None = None) -> list[str]:
        """
        Get list of changed files.

        Args:
            since_commit: Compare to this commit. None compares to working tree.

        Returns:
            List of changed file paths
        """
        if not await self.is_git_repo():
            return []

        try:
            if since_commit:
                output = await self._run("diff", "--name-only", since_commit)
            else:
                # Unstaged + staged changes
                output = await self._run("diff", "--name-only")
                staged = await self._run("diff", "--cached", "--name-only")
                output = output + "\n" + staged

            files = [f.strip() for f in output.split("\n") if f.strip()]
            return list(set(files))  # Deduplicate

        except subprocess.CalledProcessError:
            return []

    async def get_recent_commits(self, count: int = 5) -> list[tuple[str, str]]:
        """
        Get recent commits.

        Args:
            count: Number of commits to retrieve

        Returns:
            List of (hash, message) tuples
        """
        if not await self.is_git_repo():
            return []

        try:
            output = await self._run(
                "log",
                f"-{count}",
                "--pretty=format:%h|%s"
            )
            commits = []
            for line in output.split("\n"):
                if "|" in line:
                    hash_part, msg = line.split("|", 1)
                    commits.append((hash_part.strip(), msg.strip()))
            return commits

        except subprocess.CalledProcessError:
            return []

    async def create_branch(self, branch_name: str) -> bool:
        """
        Create a new branch.

        Args:
            branch_name: Name for new branch

        Returns:
            True if created successfully
        """
        if not await self.is_git_repo():
            return False

        try:
            await self._run("checkout", "-b", branch_name)
            logger.info(f"Created branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False

    async def stash(self, message: str | None = None) -> bool:
        """
        Stash current changes.

        Args:
            message: Optional stash message

        Returns:
            True if stashed successfully
        """
        if not await self.is_git_repo():
            return False

        try:
            if message:
                await self._run("stash", "push", "-m", message)
            else:
                await self._run("stash")
            logger.info("Stashed changes")
            return True
        except subprocess.CalledProcessError:
            return False

    async def stash_pop(self) -> bool:
        """
        Pop most recent stash.

        Returns:
            True if popped successfully
        """
        if not await self.is_git_repo():
            return False

        try:
            await self._run("stash", "pop")
            logger.info("Popped stash")
            return True
        except subprocess.CalledProcessError:
            return False

    def describe(self) -> str:
        """Get description of git integration state."""
        return f"GitIntegration(repo={self.repo_path})"
