"""
Git Auto-Commit Module
======================

Automated git commits after worker runs.

Safety features:
- Only commits to domain_memory/ and src/ by default
- Includes feature ID and status in commit message
- Optional push to remote

Usage:
    from hololoom.domain_harness.strands.git_autocommit import GitAutoCommit
    
    committer = GitAutoCommit()
    committer.commit_worker_run("F001", passed=True, summary="Implemented auth")
"""

import subprocess
from datetime import datetime
from pathlib import Path


class GitAutoCommit:
    """
    Handles automated git commits for worker runs.
    
    Safe by default:
    - Commits only allowed paths
    - Never force pushes
    - Validates git state before committing
    """

    def __init__(
        self,
        repo_path: Path = Path("."),
        allowed_paths: list[str] | None = None,
        auto_push: bool = False,
        remote: str = "origin",
        branch: str = "main"
    ):
        self.repo_path = Path(repo_path)
        self.allowed_paths = allowed_paths or [
            "domain_memory/",
            "src/",
            "tests/"
        ]
        self.auto_push = auto_push
        self.remote = remote
        self.branch = branch

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        """Run a git command."""
        return subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

    def is_git_repo(self) -> bool:
        """Check if we're in a git repository."""
        result = self._run_git("rev-parse", "--git-dir")
        return result.returncode == 0

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        result = self._run_git("status", "--porcelain")
        return bool(result.stdout.strip())

    def get_changed_files(self) -> list[str]:
        """Get list of changed files."""
        result = self._run_git("status", "--porcelain")
        if result.returncode != 0:
            return []

        files = []
        for line in result.stdout.strip().split("\n"):
            if line:
                # Format is "XY filename" where XY is status
                parts = line.split(maxsplit=1)
                if len(parts) >= 2:
                    files.append(parts[1])

        return files

    def filter_allowed_files(self, files: list[str]) -> list[str]:
        """Filter to only allowed paths."""
        allowed = []
        for f in files:
            for path in self.allowed_paths:
                if f.startswith(path):
                    allowed.append(f)
                    break
        return allowed

    def stage_files(self, files: list[str]) -> bool:
        """Stage specific files for commit."""
        if not files:
            return False

        for f in files:
            result = self._run_git("add", f)
            if result.returncode != 0:
                print(f"Warning: Failed to stage {f}")

        return True

    def commit(self, message: str) -> bool:
        """Create a commit with the given message."""
        result = self._run_git("commit", "-m", message)
        return result.returncode == 0

    def push(self) -> bool:
        """Push to remote."""
        result = self._run_git("push", self.remote, self.branch)
        return result.returncode == 0

    def get_last_commit_hash(self) -> str | None:
        """Get the hash of the last commit."""
        result = self._run_git("rev-parse", "HEAD")
        if result.returncode == 0:
            return result.stdout.strip()[:8]
        return None

    # ========================================================================
    # High-Level Operations
    # ========================================================================

    def commit_worker_run(
        self,
        feature_id: str,
        passed: bool,
        summary: str,
        worker_id: str | None = None
    ) -> str | None:
        """
        Commit after a worker run.
        
        Args:
            feature_id: Feature that was worked on
            passed: Whether tests passed
            summary: Brief description
            worker_id: Optional worker identifier
            
        Returns:
            Commit hash if successful, None otherwise
        """
        if not self.is_git_repo():
            print("Not a git repository, skipping commit")
            return None

        if not self.has_changes():
            print("No changes to commit")
            return None

        # Get and filter changed files
        changed = self.get_changed_files()
        allowed = self.filter_allowed_files(changed)

        if not allowed:
            print(f"No changes in allowed paths: {self.allowed_paths}")
            return None

        # Stage files
        self.stage_files(allowed)

        # Build commit message
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        status = "PASS" if passed else "FAIL"
        worker_tag = f"[{worker_id}] " if worker_id else ""

        message = f"{worker_tag}{status} {feature_id}: {summary}\n\n"
        message += "Automated commit by domain harness worker\n"
        message += f"Timestamp: {timestamp}\n"
        message += f"Files changed: {len(allowed)}"

        # Commit
        if not self.commit(message):
            print("Commit failed")
            return None

        commit_hash = self.get_last_commit_hash()
        print(f"Committed: {commit_hash}")

        # Push if enabled
        if self.auto_push:
            if self.push():
                print(f"Pushed to {self.remote}/{self.branch}")
            else:
                print("Push failed")

        return commit_hash

    def commit_initialization(self, project_name: str) -> str | None:
        """
        Commit after domain initialization.
        
        Creates initial commit with all domain memory files.
        """
        if not self.is_git_repo():
            return None

        # Stage domain memory
        self._run_git("add", "domain_memory/")

        message = f"Initialize domain memory: {project_name}\n\n"
        message += "Created by domain harness initializer\n"
        message += f"Timestamp: {datetime.utcnow().isoformat()}"

        if self.commit(message):
            return self.get_last_commit_hash()
        return None

    def commit_all(self, message: str) -> str | None:
        """
        Commit all changes (use with caution).
        """
        if not self.is_git_repo():
            return None

        self._run_git("add", "-A")

        if self.commit(message):
            return self.get_last_commit_hash()
        return None


# ============================================================================
# Standalone Functions
# ============================================================================

def git_commit_all(msg_prefix: str = "worker") -> str | None:
    """
    Quick commit all changes.
    
    Matches the GPT handoff interface.
    """
    timestamp = datetime.utcnow().isoformat()
    msg = f"{msg_prefix}: automated commit at {timestamp}"

    subprocess.run(["git", "add", "-A"])
    result = subprocess.run(["git", "commit", "-m", msg], capture_output=True)

    if result.returncode == 0:
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True
        )
        return hash_result.stdout.strip()[:8]
    return None


def git_push(remote: str = "origin", branch: str = "main") -> bool:
    """Push to remote."""
    result = subprocess.run(["git", "push", remote, branch])
    return result.returncode == 0


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Git Auto-Commit")
    parser.add_argument("--feature", "-f", help="Feature ID")
    parser.add_argument("--status", "-s", choices=["pass", "fail"], default="pass")
    parser.add_argument("--message", "-m", default="Worker run")
    parser.add_argument("--push", "-p", action="store_true", help="Push after commit")

    args = parser.parse_args()

    committer = GitAutoCommit(auto_push=args.push)

    if args.feature:
        result = committer.commit_worker_run(
            feature_id=args.feature,
            passed=(args.status == "pass"),
            summary=args.message
        )
    else:
        result = git_commit_all(args.message)

    if result:
        print(f"Success: {result}")
    else:
        print("No commit created")
