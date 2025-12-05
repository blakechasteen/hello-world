#!/usr/bin/env python3
"""
Git Handler for Promptly Matrix Bot

Handles git operations from Matrix chat commands.
"""

import os
import subprocess
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class GitHandler:
    """Handle git operations with safety checks"""

    # Whitelist of allowed git commands
    ALLOWED_COMMANDS = {
        'status', 'log', 'diff', 'branch',  # Read-only
        'commit', 'push', 'pull',  # Write operations
        'checkout', 'merge', 'rebase'  # Advanced
    }

    # Commands that require confirmation
    REQUIRES_CONFIRMATION = {
        'push', 'merge', 'rebase', 'checkout'
    }

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize Git handler.

        Args:
            repo_path: Path to git repository. If None, uses current directory.
        """
        self.repo_path = repo_path or os.getcwd()
        self._validate_repo()

    def _validate_repo(self) -> None:
        """Validate that repo_path is a valid git repository"""
        git_dir = Path(self.repo_path) / '.git'
        if not git_dir.exists():
            raise ValueError(f"{self.repo_path} is not a git repository")

    def _run_git_command(
        self,
        args: List[str],
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a git command safely.

        Args:
            args: Git command arguments (e.g., ['status', '-s'])
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess with result

        Raises:
            ValueError: If command not allowed
            subprocess.CalledProcessError: If git command fails
        """
        # Validate command is allowed
        if args and args[0] not in self.ALLOWED_COMMANDS:
            raise ValueError(f"Git command '{args[0]}' not allowed")

        full_cmd = ['git'] + args

        logger.info(f"Running git command: {' '.join(full_cmd)}")

        return subprocess.run(
            full_cmd,
            cwd=self.repo_path,
            capture_output=capture_output,
            text=True,
            check=True
        )

    def status(self, short: bool = False) -> str:
        """
        Get git status.

        Args:
            short: Use short format

        Returns:
            Status output as string
        """
        args = ['status']
        if short:
            args.append('-s')

        result = self._run_git_command(args)
        return result.stdout

    def log(self, max_count: int = 10, oneline: bool = True) -> str:
        """
        Get git log.

        Args:
            max_count: Maximum number of commits to show
            oneline: Use oneline format

        Returns:
            Log output as string
        """
        args = ['log', f'-{max_count}']
        if oneline:
            args.append('--oneline')

        result = self._run_git_command(args)
        return result.stdout

    def diff(self, cached: bool = False, file: Optional[str] = None) -> str:
        """
        Get git diff.

        Args:
            cached: Show staged changes
            file: Specific file to diff

        Returns:
            Diff output as string
        """
        args = ['diff']
        if cached:
            args.append('--cached')
        if file:
            args.append(file)

        result = self._run_git_command(args)
        return result.stdout

    def branch(self, list_all: bool = False) -> str:
        """
        Get git branches.

        Args:
            list_all: Show all branches (including remote)

        Returns:
            Branch list as string
        """
        args = ['branch']
        if list_all:
            args.append('-a')

        result = self._run_git_command(args)
        return result.stdout

    def commit(self, message: str, add_all: bool = False) -> str:
        """
        Create a git commit.

        Args:
            message: Commit message
            add_all: Add all changes before committing

        Returns:
            Commit output as string
        """
        if add_all:
            self._run_git_command(['add', '.'])

        result = self._run_git_command(['commit', '-m', message])
        return result.stdout

    def push(self, remote: str = 'origin', branch: Optional[str] = None) -> str:
        """
        Push to remote repository.

        Args:
            remote: Remote name (default: origin)
            branch: Branch to push (default: current branch)

        Returns:
            Push output as string
        """
        args = ['push', remote]
        if branch:
            args.append(branch)

        result = self._run_git_command(args)
        return result.stdout if result.stdout else result.stderr

    def pull(self, remote: str = 'origin', branch: Optional[str] = None) -> str:
        """
        Pull from remote repository.

        Args:
            remote: Remote name (default: origin)
            branch: Branch to pull (default: current branch)

        Returns:
            Pull output as string
        """
        args = ['pull', remote]
        if branch:
            args.append(branch)

        result = self._run_git_command(args)
        return result.stdout if result.stdout else result.stderr

    def checkout(self, branch: str, create: bool = False) -> str:
        """
        Checkout a branch.

        Args:
            branch: Branch name
            create: Create new branch

        Returns:
            Checkout output as string
        """
        args = ['checkout']
        if create:
            args.append('-b')
        args.append(branch)

        result = self._run_git_command(args)
        return result.stdout if result.stdout else result.stderr

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        result = self._run_git_command(['branch', '--show-current'])
        return result.stdout.strip()

    def get_remote_url(self, remote: str = 'origin') -> str:
        """Get the URL of a remote"""
        result = self._run_git_command(['remote', 'get-url', remote])
        return result.stdout.strip()

    def requires_confirmation(self, command: str) -> bool:
        """Check if command requires user confirmation"""
        return command in self.REQUIRES_CONFIRMATION


# Example usage
if __name__ == "__main__":
    import sys

    # Test git handler
    try:
        handler = GitHandler()

        print("=== Git Status ===")
        print(handler.status(short=True))

        print("\n=== Current Branch ===")
        print(handler.get_current_branch())

        print("\n=== Recent Log ===")
        print(handler.log(max_count=5))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
