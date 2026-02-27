"""Proto Git Operations Ability - Tier 2 Plugin.

Safe, read-only Git operations for repository analysis and exploration.
Enables Proto to understand code repository state, history, and structure.

Supported Operations:
    - status: Get repository status (staged, unstaged, untracked files)
    - diff: Show changes (staged or unstaged, whole repo or specific file)
    - log: Get commit history with customizable depth
    - branch: List branches or get current branch
    - show: Display specific commit details
    - blame: Show line-by-line commit attribution for a file
    - stash_list: List all stashed changes

Safety Features:
    - Read-only operations only (no commit, push, pull, checkout)
    - Repository validation (must be valid git repo)
    - Path traversal prevention
    - Command timeouts (10s default)
    - Comprehensive error handling

Implementation: December 2025
Status: Production ready
"""

import asyncio
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..protocol import (
    AbilityContext,
    AbilityManifest,
    AbilityResult,
    AbilityTier,
    AbilityTrustLevel,
    BaseAbility,
    PreflightResult,
    VerificationResult,
)


logger = logging.getLogger("proto.git_operations")


# Timeout for git commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 10.0

# Maximum output size to prevent memory issues
MAX_OUTPUT_SIZE = 1_000_000  # 1MB

# Safe operations that are read-only
SAFE_OPERATIONS = {
    "status",
    "diff",
    "log",
    "branch",
    "show",
    "blame",
    "stash_list",
}


@dataclass
class GitOperationResult:
    """Result from a git operation.

    Attributes:
        operation: Operation name (status, diff, log, etc.)
        success: Whether operation succeeded
        output: Command output (string or structured dict)
        error: Error message if failed
        repo_path: Path to git repository
    """
    operation: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    repo_path: Optional[str] = None


class GitOperationsAbility(BaseAbility):
    """Tier 2 Ability: Safe read-only Git operations.

    Provides structured access to git repository information:
    - Repository status and changes
    - Commit history and details
    - Branch information
    - File-level attribution (blame)
    - Stashed changes

    Example:
        ability = GitOperationsAbility()
        context = AbilityContext(session_id="123", working_directory="/path/to/repo")

        result = await ability.execute({
            "operation": "status",
            "repo_path": "/path/to/repo"
        }, context)

        if result.success:
            print(result.output["status_output"])
    """

    def __init__(self):
        """Initialize Git Operations ability."""
        manifest = AbilityManifest(
            name="git_operations",
            version="1.0.0",
            description="Safe read-only Git repository operations for analysis and exploration",
            author="Proto Team",
            tier=AbilityTier.PLUGIN,
            trust_level=AbilityTrustLevel.VERIFIED,
            permissions=["read_file", "execute_command"],
            requires_confirmation=False,
            requires=["git"],
            tags=["git", "vcs", "repository", "version-control", "analysis"],
            homepage="https://github.com/hololoom/proto",
        )
        super().__init__(manifest)

    async def preflight(self, context: AbilityContext) -> PreflightResult:
        """Check if git operations can execute.

        Verifies:
        - git command is available
        - Working directory exists (if specified)

        Args:
            context: Execution context

        Returns:
            PreflightResult indicating execution feasibility
        """
        # Check if git is installed
        if not shutil.which("git"):
            return PreflightResult(
                can_execute=False,
                reason="git command not found. Please install Git.",
                estimated_duration_ms=0.0,
            )

        # Check if working directory exists (if specified)
        if context.working_directory and not os.path.isdir(context.working_directory):
            return PreflightResult(
                can_execute=False,
                reason=f"Working directory not found: {context.working_directory}",
                estimated_duration_ms=0.0,
            )

        return PreflightResult(
            can_execute=True,
            estimated_duration_ms=500.0,
        )

    async def execute(
        self,
        params: Dict[str, Any],
        context: AbilityContext,
    ) -> AbilityResult:
        """Execute git operation with given parameters.

        Parameters:
            operation: str - Operation name (status, diff, log, branch, show, blame, stash_list)
            repo_path: Optional[str] - Repository path (defaults to working_directory)
            file_path: Optional[str] - For file-specific operations (diff, blame)
            commit: Optional[str] - For commit-specific operations (show)
            limit: int - For log operations (default: 10)
            staged: bool - For diff (staged=True) vs unstaged (default: False)

        Args:
            params: Operation parameters
            context: Execution context

        Returns:
            AbilityResult with git operation output
        """
        start_time = time.time()

        try:
            # Extract and validate parameters
            operation = params.get("operation", "").lower().strip()
            repo_path = params.get("repo_path") or context.working_directory
            file_path = params.get("file_path")
            commit = params.get("commit")
            limit = params.get("limit", 10)
            staged = params.get("staged", False)

            # Validate operation
            if not operation:
                return AbilityResult(
                    success=False,
                    error="Missing required parameter: operation",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            if operation not in SAFE_OPERATIONS:
                return AbilityResult(
                    success=False,
                    error=f"Unsupported operation: {operation}. Safe operations: {SAFE_OPERATIONS}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Validate repo path
            repo_path = self._validate_repo_path(repo_path, context.working_directory)
            if isinstance(repo_path, str) and repo_path.startswith("ERROR:"):
                return AbilityResult(
                    success=False,
                    error=repo_path[7:],  # Remove "ERROR:" prefix
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Validate file path if provided
            if file_path:
                file_path = self._validate_file_path(file_path, repo_path)
                if isinstance(file_path, str) and file_path.startswith("ERROR:"):
                    return AbilityResult(
                        success=False,
                        error=file_path[7:],
                        duration_ms=(time.time() - start_time) * 1000,
                    )

            # Execute appropriate operation
            git_result: GitOperationResult
            if operation == "status":
                git_result = await self._git_status(repo_path)
            elif operation == "diff":
                git_result = await self._git_diff(repo_path, file_path, staged)
            elif operation == "log":
                git_result = await self._git_log(repo_path, limit)
            elif operation == "branch":
                git_result = await self._git_branch(repo_path)
            elif operation == "show":
                git_result = await self._git_show(repo_path, commit)
            elif operation == "blame":
                git_result = await self._git_blame(repo_path, file_path)
            elif operation == "stash_list":
                git_result = await self._git_stash_list(repo_path)
            else:
                git_result = GitOperationResult(
                    operation=operation,
                    success=False,
                    error=f"Unknown operation: {operation}",
                    repo_path=repo_path,
                )

            duration_ms = (time.time() - start_time) * 1000

            if git_result.success:
                return AbilityResult(
                    success=True,
                    output=self._format_result(git_result),
                    confidence=0.95,
                    duration_ms=duration_ms,
                    metadata={
                        "operation": operation,
                        "repo_path": repo_path,
                    },
                )
            else:
                return AbilityResult(
                    success=False,
                    error=git_result.error,
                    duration_ms=duration_ms,
                    metadata={
                        "operation": operation,
                        "repo_path": repo_path,
                    },
                )

        except Exception as e:
            logger.error(f"Git operation execution failed: {e}", exc_info=True)
            return AbilityResult(
                success=False,
                error=f"Execution error: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def verify(self, result: AbilityResult) -> VerificationResult:
        """Verify git operation result.

        Checks:
        - Operation succeeded
        - Output is non-empty or valid for operation
        - No error messages

        Args:
            result: Result from execute()

        Returns:
            VerificationResult with verification status
        """
        if not result.success:
            return VerificationResult(
                verified=False,
                issues=[result.error or "Operation failed"],
            )

        # Output can be a dict or string depending on operation
        output = result.output

        if isinstance(output, dict):
            # Check if result has expected structure
            if not output:
                return VerificationResult(
                    verified=False,
                    issues=["Empty result output"],
                )
        elif isinstance(output, str):
            if not output.strip():
                # Some operations (like diff on clean repo) may have empty output
                # This is valid, so we verify as True
                pass

        return VerificationResult(verified=True)

    # ========== Git Command Implementations ==========

    async def _git_status(self, repo_path: str) -> GitOperationResult:
        """Get git status (staged, unstaged, untracked files).

        Args:
            repo_path: Repository path

        Returns:
            GitOperationResult with status information
        """
        try:
            # Use porcelain format for easy parsing
            output = await self._run_git_command(
                ["git", "status", "--porcelain"],
                repo_path,
            )

            # Parse status output
            staged = []
            unstaged = []
            untracked = []

            for line in output.strip().split("\n"):
                if not line:
                    continue

                status = line[:2]
                filename = line[3:]

                if status[0] in "MRA":
                    staged.append(f"{status[0]} {filename}")
                if status[1] in "M D":
                    unstaged.append(f"{status[1]} {filename}")
                if status == "??":
                    untracked.append(filename)

            return GitOperationResult(
                operation="status",
                success=True,
                output={
                    "status_output": output,
                    "staged_files": staged,
                    "unstaged_files": unstaged,
                    "untracked_files": untracked,
                    "has_changes": bool(staged or unstaged or untracked),
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="status",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    async def _git_diff(
        self, repo_path: str, file_path: Optional[str], staged: bool
    ) -> GitOperationResult:
        """Get git diff (staged or unstaged changes).

        Args:
            repo_path: Repository path
            file_path: Optional specific file path
            staged: If True, show staged changes; if False, show unstaged

        Returns:
            GitOperationResult with diff output
        """
        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")
            if file_path:
                cmd.append("--")
                cmd.append(file_path)

            output = await self._run_git_command(cmd, repo_path)

            return GitOperationResult(
                operation="diff",
                success=True,
                output={
                    "diff_output": output,
                    "has_changes": bool(output.strip()),
                    "staged": staged,
                    "file_path": file_path,
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="diff",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    async def _git_log(self, repo_path: str, limit: int) -> GitOperationResult:
        """Get commit history.

        Args:
            repo_path: Repository path
            limit: Maximum number of commits to retrieve

        Returns:
            GitOperationResult with commit history
        """
        try:
            # Limit range validation
            limit = max(1, min(limit, 100))  # Clamp 1-100

            # Get log with detailed format
            cmd = [
                "git",
                "log",
                f"-n {limit}",
                "--oneline",
                "--decorate",
                "--graph",
            ]

            output = await self._run_git_command(cmd, repo_path)

            # Also get raw commits for structured output
            cmd_raw = [
                "git",
                "log",
                f"-n {limit}",
                "--format=%H|%s|%an|%ai",
            ]

            raw_output = await self._run_git_command(cmd_raw, repo_path)

            commits = []
            for line in raw_output.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "subject": parts[1],
                        "author": parts[2],
                        "date": parts[3],
                    })

            return GitOperationResult(
                operation="log",
                success=True,
                output={
                    "log_output": output,
                    "commits": commits,
                    "limit": limit,
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="log",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    async def _git_branch(self, repo_path: str) -> GitOperationResult:
        """List branches and get current branch.

        Args:
            repo_path: Repository path

        Returns:
            GitOperationResult with branch information
        """
        try:
            # Get all branches
            output = await self._run_git_command(["git", "branch", "-a"], repo_path)

            # Get current branch
            current = await self._run_git_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                repo_path,
            )

            branches = []
            for line in output.strip().split("\n"):
                if line:
                    is_current = line.startswith("*")
                    branch_name = line.lstrip("* ").strip()
                    branches.append({
                        "name": branch_name,
                        "is_current": is_current,
                    })

            return GitOperationResult(
                operation="branch",
                success=True,
                output={
                    "branch_output": output,
                    "branches": branches,
                    "current_branch": current.strip(),
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="branch",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    async def _git_show(
        self, repo_path: str, commit: Optional[str]
    ) -> GitOperationResult:
        """Show specific commit details.

        Args:
            repo_path: Repository path
            commit: Commit hash/ref (defaults to HEAD)

        Returns:
            GitOperationResult with commit details
        """
        try:
            if not commit:
                commit = "HEAD"

            # Validate commit reference (basic check)
            if not re.match(r"^[a-f0-9]{7,40}$|^HEAD|^[a-zA-Z0-9_/-]+$", commit):
                return GitOperationResult(
                    operation="show",
                    success=False,
                    error=f"Invalid commit reference: {commit}",
                    repo_path=repo_path,
                )

            output = await self._run_git_command(
                ["git", "show", commit],
                repo_path,
            )

            return GitOperationResult(
                operation="show",
                success=True,
                output={
                    "show_output": output,
                    "commit": commit,
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="show",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    async def _git_blame(
        self, repo_path: str, file_path: Optional[str]
    ) -> GitOperationResult:
        """Show line-by-line commit attribution for a file.

        Args:
            repo_path: Repository path
            file_path: File path (required)

        Returns:
            GitOperationResult with blame information
        """
        try:
            if not file_path:
                return GitOperationResult(
                    operation="blame",
                    success=False,
                    error="file_path parameter is required for blame operation",
                    repo_path=repo_path,
                )

            output = await self._run_git_command(
                ["git", "blame", "-l", "--", file_path],
                repo_path,
            )

            return GitOperationResult(
                operation="blame",
                success=True,
                output={
                    "blame_output": output,
                    "file_path": file_path,
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="blame",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    async def _git_stash_list(self, repo_path: str) -> GitOperationResult:
        """List all stashed changes.

        Args:
            repo_path: Repository path

        Returns:
            GitOperationResult with stash list
        """
        try:
            output = await self._run_git_command(
                ["git", "stash", "list", "--decorate"],
                repo_path,
            )

            stashes = []
            for line in output.strip().split("\n"):
                if line:
                    stashes.append(line)

            return GitOperationResult(
                operation="stash_list",
                success=True,
                output={
                    "stash_output": output,
                    "stashes": stashes,
                    "count": len(stashes),
                },
                repo_path=repo_path,
            )

        except Exception as e:
            return GitOperationResult(
                operation="stash_list",
                success=False,
                error=str(e),
                repo_path=repo_path,
            )

    # ========== Helper Methods ==========

    async def _run_git_command(
        self, cmd: List[str], cwd: str, timeout: float = DEFAULT_COMMAND_TIMEOUT
    ) -> str:
        """Run git command and return output.

        Args:
            cmd: Git command as list of strings
            cwd: Working directory for command
            timeout: Command timeout in seconds

        Returns:
            Command output (stdout)

        Raises:
            subprocess.CalledProcessError: If command fails
            subprocess.TimeoutExpired: If command times out
            OSError: If directory doesn't exist
        """
        try:
            # Verify directory exists
            if not os.path.isdir(cwd):
                raise OSError(f"Directory not found: {cwd}")

            # Run command with timeout
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    text=True,
                ),
                timeout=timeout,
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=timeout,
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    cmd,
                    output=stdout,
                    stderr=stderr,
                )

            # Check output size
            if len(stdout) > MAX_OUTPUT_SIZE:
                logger.warning(
                    f"Git output truncated. Size: {len(stdout)} > {MAX_OUTPUT_SIZE}"
                )
                return stdout[:MAX_OUTPUT_SIZE]

            return stdout

        except asyncio.TimeoutError:
            raise subprocess.TimeoutExpired(
                " ".join(cmd),
                timeout,
            )

    def _validate_repo_path(
        self, repo_path: Optional[str], working_dir: str
    ) -> str:
        """Validate and resolve repository path.

        Args:
            repo_path: User-provided repo path (optional)
            working_dir: Context working directory

        Returns:
            Validated absolute path, or "ERROR: message" string

        Safety:
            - Prevents path traversal (no ../)
            - Checks path exists
            - Verifies it's a git repository
        """
        # Use provided path or working directory
        if not repo_path:
            repo_path = working_dir or os.getcwd()

        # Prevent path traversal
        if ".." in repo_path:
            return "ERROR: Path traversal not allowed"

        # Resolve to absolute path
        try:
            repo_path = os.path.abspath(repo_path)
        except Exception as e:
            return f"ERROR: Invalid path: {str(e)}"

        # Check path exists
        if not os.path.exists(repo_path):
            return f"ERROR: Path not found: {repo_path}"

        # Check it's a directory
        if not os.path.isdir(repo_path):
            return f"ERROR: Not a directory: {repo_path}"

        # Check it's a git repository (has .git)
        git_dir = os.path.join(repo_path, ".git")
        if not os.path.exists(git_dir):
            return f"ERROR: Not a git repository: {repo_path}"

        return repo_path

    def _validate_file_path(self, file_path: str, repo_path: str) -> str:
        """Validate file path within repository.

        Args:
            file_path: User-provided file path
            repo_path: Repository path

        Returns:
            Validated relative path, or "ERROR: message" string

        Safety:
            - Prevents path traversal
            - Ensures file is within repository
        """
        # Prevent path traversal
        if ".." in file_path:
            return "ERROR: Path traversal not allowed"

        # Resolve relative to repo
        full_path = os.path.normpath(os.path.join(repo_path, file_path))

        # Ensure it's within repo (prevent escape)
        try:
            rel = os.path.relpath(full_path, repo_path)
            if rel.startswith(".."):
                return "ERROR: File path escapes repository"
        except ValueError:
            return "ERROR: File path is on different drive"

        # File doesn't need to exist (could be untracked)
        return file_path

    def _format_result(self, git_result: GitOperationResult) -> Dict[str, Any]:
        """Format git operation result for output.

        Args:
            git_result: GitOperationResult from operation

        Returns:
            Dict with formatted result
        """
        return {
            "operation": git_result.operation,
            "repo_path": git_result.repo_path,
            "result": git_result.output,
        }
