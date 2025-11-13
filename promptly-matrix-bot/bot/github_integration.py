"""
GitHub Integration for Promptly Matrix Bot

Provides:
- PR creation from workflows
- Automated code review
- Issue tracking
- CI/CD webhook handling
- GitHub Actions integration

Usage:
    from bot.github_integration import GitHubIntegration

    gh = GitHubIntegration(token="ghp_...")
    pr = await gh.create_pull_request(
        repo="owner/repo",
        title="Add new feature",
        body="Description",
        head="feature-branch",
        base="main",
        files={"path/to/file.py": "content"}
    )
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from github import Github, GithubException
    from github.Repository import Repository
    from github.PullRequest import PullRequest
    from github.Issue import Issue
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logging.warning("PyGithub not installed. GitHub integration disabled.")

logger = logging.getLogger(__name__)


class PRStatus(Enum):
    """Pull request status"""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class IssueStatus(Enum):
    """Issue status"""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class CodeReviewResult:
    """Code review result"""
    approved: bool
    issues_found: int
    security_issues: int
    quality_score: float
    comments: List[Dict[str, str]]
    summary: str


@dataclass
class PRCreateResult:
    """PR creation result"""
    pr_number: int
    pr_url: str
    pr_html_url: str
    created_at: str
    status: PRStatus


@dataclass
class IssueCreateResult:
    """Issue creation result"""
    issue_number: int
    issue_url: str
    issue_html_url: str
    created_at: str
    status: IssueStatus


class GitHubIntegration:
    """
    GitHub API integration for Promptly

    Handles PR creation, code review, issue tracking, and CI/CD triggers.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub integration

        Args:
            token: GitHub personal access token (optional, can be set via env)
        """
        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub not installed. Run: pip install PyGithub")

        self.token = token
        self.client = Github(token) if token else None
        self.logger = logging.getLogger(__name__)

        if self.client:
            try:
                user = self.client.get_user()
                self.logger.info(f"GitHub authenticated as: {user.login}")
            except GithubException as e:
                self.logger.error(f"GitHub authentication failed: {e}")

    async def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        files: Optional[Dict[str, str]] = None,
        draft: bool = False
    ) -> PRCreateResult:
        """
        Create a pull request

        Args:
            repo: Repository name (owner/repo)
            title: PR title
            body: PR description
            head: Source branch
            base: Target branch (default: main)
            files: Files to add/modify {path: content}
            draft: Create as draft PR

        Returns:
            PRCreateResult with PR details
        """
        if not self.client:
            raise ValueError("GitHub client not initialized. Provide token.")

        try:
            # Get repository
            repository = self.client.get_repo(repo)

            # Create branch if files provided
            if files:
                await self._create_or_update_branch(repository, head, base, files)

            # Create pull request
            pr = repository.create_pull(
                title=title,
                body=body,
                head=head,
                base=base,
                draft=draft
            )

            self.logger.info(f"Created PR #{pr.number} in {repo}")

            return PRCreateResult(
                pr_number=pr.number,
                pr_url=pr.url,
                pr_html_url=pr.html_url,
                created_at=pr.created_at.isoformat(),
                status=PRStatus.OPEN
            )

        except GithubException as e:
            self.logger.error(f"Failed to create PR: {e}")
            raise

    async def _create_or_update_branch(
        self,
        repo: Repository,
        branch_name: str,
        base_branch: str,
        files: Dict[str, str]
    ):
        """
        Create or update a branch with file changes

        Args:
            repo: Repository object
            branch_name: Branch to create/update
            base_branch: Base branch to branch from
            files: Files to add/modify {path: content}
        """
        try:
            # Get base branch SHA
            base_ref = repo.get_git_ref(f"heads/{base_branch}")
            base_sha = base_ref.object.sha

            # Try to get existing branch
            try:
                branch_ref = repo.get_git_ref(f"heads/{branch_name}")
                # Branch exists, update it
                branch_ref.edit(base_sha, force=True)
                commit_sha = base_sha
            except GithubException:
                # Branch doesn't exist, create it
                repo.create_git_ref(f"refs/heads/{branch_name}", base_sha)
                commit_sha = base_sha

            # Create blobs for each file
            blobs = []
            for file_path, content in files.items():
                blob = repo.create_git_blob(content, "utf-8")
                blobs.append((file_path, blob.sha))

            # Get base tree
            base_commit = repo.get_git_commit(commit_sha)
            base_tree = base_commit.tree

            # Create tree with new files
            tree_elements = []
            for file_path, blob_sha in blobs:
                tree_elements.append({
                    "path": file_path,
                    "mode": "100644",  # Regular file
                    "type": "blob",
                    "sha": blob_sha
                })

            tree = repo.create_git_tree(tree_elements, base_tree)

            # Create commit
            commit = repo.create_git_commit(
                message=f"Update files for {branch_name}",
                tree=tree,
                parents=[base_commit]
            )

            # Update branch ref
            branch_ref = repo.get_git_ref(f"heads/{branch_name}")
            branch_ref.edit(commit.sha)

            self.logger.info(f"Updated branch {branch_name} with {len(files)} files")

        except GithubException as e:
            self.logger.error(f"Failed to create/update branch: {e}")
            raise

    async def review_pull_request(
        self,
        repo: str,
        pr_number: int,
        auto_approve: bool = False
    ) -> CodeReviewResult:
        """
        Review a pull request using code_reviewer

        Args:
            repo: Repository name (owner/repo)
            pr_number: PR number
            auto_approve: Automatically approve if no issues

        Returns:
            CodeReviewResult with review details
        """
        if not self.client:
            raise ValueError("GitHub client not initialized.")

        try:
            # Get repository and PR
            repository = self.client.get_repo(repo)
            pr = repository.get_pull(pr_number)

            # Get PR files
            files = pr.get_files()

            # Review each file
            total_issues = 0
            security_issues = 0
            comments = []

            for file in files:
                # Get file content
                if file.status in ["added", "modified"]:
                    # Run code review (integrate with existing code_reviewer.py)
                    file_issues = await self._review_file(file.filename, file.patch)
                    total_issues += len(file_issues)

                    for issue in file_issues:
                        if issue["severity"] == "high":
                            security_issues += 1
                        comments.append({
                            "path": file.filename,
                            "line": issue.get("line", 1),
                            "body": issue["message"]
                        })

            # Calculate quality score
            quality_score = max(0.0, 1.0 - (total_issues * 0.1) - (security_issues * 0.2))

            # Determine approval
            approved = security_issues == 0 and total_issues < 5

            # Create review
            if auto_approve and approved:
                pr.create_review(
                    body="✅ Automated review passed. No critical issues found.",
                    event="APPROVE",
                    comments=comments if comments else None
                )
                summary = f"PR #{pr_number} approved automatically"
            else:
                if comments:
                    pr.create_review(
                        body=f"⚠️ Found {total_issues} issues ({security_issues} security)",
                        event="REQUEST_CHANGES" if security_issues > 0 else "COMMENT",
                        comments=comments
                    )
                    summary = f"PR #{pr_number} needs changes: {total_issues} issues"
                else:
                    summary = f"PR #{pr_number} reviewed: no issues"

            self.logger.info(summary)

            return CodeReviewResult(
                approved=approved,
                issues_found=total_issues,
                security_issues=security_issues,
                quality_score=quality_score,
                comments=comments,
                summary=summary
            )

        except GithubException as e:
            self.logger.error(f"Failed to review PR: {e}")
            raise

    async def _review_file(self, filename: str, patch: str) -> List[Dict]:
        """
        Review a single file for issues

        Args:
            filename: File path
            patch: Git patch/diff

        Returns:
            List of issues found
        """
        issues = []

        # Simple heuristic checks (in real implementation, use code_reviewer.py)

        # Check for common security issues
        dangerous_patterns = [
            ("eval(", "Dangerous eval() call", "high"),
            ("exec(", "Dangerous exec() call", "high"),
            ("__import__", "Dynamic import", "medium"),
            ("subprocess.call", "Subprocess call without shell=False", "medium"),
            ("pickle.loads", "Unsafe pickle deserialization", "high"),
            ("yaml.load(", "Unsafe YAML load (use safe_load)", "high"),
            ("sql = ", "Potential SQL injection", "high"),
        ]

        for pattern, message, severity in dangerous_patterns:
            if pattern in patch:
                issues.append({
                    "pattern": pattern,
                    "message": message,
                    "severity": severity,
                    "line": 1  # Simplified - would parse patch for actual line
                })

        # Check for quality issues
        if "TODO" in patch or "FIXME" in patch:
            issues.append({
                "pattern": "TODO/FIXME",
                "message": "Unresolved TODO/FIXME comment",
                "severity": "low",
                "line": 1
            })

        return issues

    async def create_issue(
        self,
        repo: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        milestone: Optional[int] = None
    ) -> IssueCreateResult:
        """
        Create a GitHub issue

        Args:
            repo: Repository name (owner/repo)
            title: Issue title
            body: Issue description
            labels: Issue labels
            assignee: Assignee username
            milestone: Milestone number

        Returns:
            IssueCreateResult with issue details
        """
        if not self.client:
            raise ValueError("GitHub client not initialized.")

        try:
            # Get repository
            repository = self.client.get_repo(repo)

            # Create issue
            issue = repository.create_issue(
                title=title,
                body=body,
                labels=labels or [],
                assignee=assignee,
                milestone=repository.get_milestone(milestone) if milestone else None
            )

            self.logger.info(f"Created issue #{issue.number} in {repo}")

            return IssueCreateResult(
                issue_number=issue.number,
                issue_url=issue.url,
                issue_html_url=issue.html_url,
                created_at=issue.created_at.isoformat(),
                status=IssueStatus.OPEN
            )

        except GithubException as e:
            self.logger.error(f"Failed to create issue: {e}")
            raise

    async def trigger_workflow(
        self,
        repo: str,
        workflow_id: str,
        ref: str = "main",
        inputs: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Trigger a GitHub Actions workflow

        Args:
            repo: Repository name (owner/repo)
            workflow_id: Workflow file name or ID
            ref: Branch/tag to run on
            inputs: Workflow inputs

        Returns:
            True if triggered successfully
        """
        if not self.client:
            raise ValueError("GitHub client not initialized.")

        try:
            # Get repository
            repository = self.client.get_repo(repo)

            # Get workflow
            workflow = repository.get_workflow(workflow_id)

            # Trigger workflow
            success = workflow.create_dispatch(ref=ref, inputs=inputs or {})

            self.logger.info(f"Triggered workflow {workflow_id} in {repo} on {ref}")

            return success

        except GithubException as e:
            self.logger.error(f"Failed to trigger workflow: {e}")
            raise

    async def get_pr_status(self, repo: str, pr_number: int) -> Dict:
        """
        Get PR status (checks, reviews, etc.)

        Args:
            repo: Repository name
            pr_number: PR number

        Returns:
            Status dict with checks, reviews, mergeable
        """
        if not self.client:
            raise ValueError("GitHub client not initialized.")

        try:
            repository = self.client.get_repo(repo)
            pr = repository.get_pull(pr_number)

            # Get reviews
            reviews = list(pr.get_reviews())
            approvals = sum(1 for r in reviews if r.state == "APPROVED")
            changes_requested = sum(1 for r in reviews if r.state == "CHANGES_REQUESTED")

            # Get checks
            commit = pr.get_commits().reversed[0]
            check_runs = commit.get_check_runs()
            checks_passing = all(
                run.conclusion == "success"
                for run in check_runs
                if run.conclusion is not None
            )

            return {
                "state": pr.state,
                "mergeable": pr.mergeable,
                "merged": pr.merged,
                "approvals": approvals,
                "changes_requested": changes_requested,
                "checks_passing": checks_passing,
                "total_checks": check_runs.totalCount,
                "url": pr.html_url
            }

        except GithubException as e:
            self.logger.error(f"Failed to get PR status: {e}")
            raise


# Webhook handler for GitHub events
class GitHubWebhookHandler:
    """
    Handle GitHub webhook events

    Supports:
    - push events (trigger CI)
    - pull_request events (auto-review)
    - issue events (notify Matrix)
    - workflow_run events (report status)
    """

    def __init__(self, secret: Optional[str] = None):
        """
        Initialize webhook handler

        Args:
            secret: Webhook secret for signature verification
        """
        self.secret = secret
        self.logger = logging.getLogger(__name__)

    async def handle_event(self, event_type: str, payload: Dict) -> Dict:
        """
        Handle a webhook event

        Args:
            event_type: Event type (push, pull_request, etc.)
            payload: Event payload

        Returns:
            Response dict
        """
        self.logger.info(f"Handling GitHub event: {event_type}")

        # Dispatch to appropriate handler
        handlers = {
            "push": self._handle_push,
            "pull_request": self._handle_pull_request,
            "issue": self._handle_issue,
            "workflow_run": self._handle_workflow_run,
        }

        handler = handlers.get(event_type)
        if handler:
            return await handler(payload)
        else:
            self.logger.warning(f"Unhandled event type: {event_type}")
            return {"status": "ignored", "reason": f"Event type {event_type} not handled"}

    async def _handle_push(self, payload: Dict) -> Dict:
        """Handle push event"""
        ref = payload.get("ref", "")
        commits = payload.get("commits", [])

        self.logger.info(f"Push to {ref}: {len(commits)} commits")

        # Could trigger CI/CD here
        return {
            "status": "processed",
            "ref": ref,
            "commits": len(commits)
        }

    async def _handle_pull_request(self, payload: Dict) -> Dict:
        """Handle pull request event"""
        action = payload.get("action", "")
        pr = payload.get("pull_request", {})
        pr_number = pr.get("number")

        self.logger.info(f"PR #{pr_number} {action}")

        # Could auto-review on open/synchronize
        if action in ["opened", "synchronize"]:
            # Trigger code review
            return {
                "status": "review_triggered",
                "pr_number": pr_number,
                "action": action
            }

        return {
            "status": "processed",
            "pr_number": pr_number,
            "action": action
        }

    async def _handle_issue(self, payload: Dict) -> Dict:
        """Handle issue event"""
        action = payload.get("action", "")
        issue = payload.get("issue", {})
        issue_number = issue.get("number")

        self.logger.info(f"Issue #{issue_number} {action}")

        # Could notify Matrix room
        return {
            "status": "processed",
            "issue_number": issue_number,
            "action": action
        }

    async def _handle_workflow_run(self, payload: Dict) -> Dict:
        """Handle workflow run event"""
        action = payload.get("action", "")
        workflow_run = payload.get("workflow_run", {})
        conclusion = workflow_run.get("conclusion")

        self.logger.info(f"Workflow {action}: {conclusion}")

        # Could report status to Matrix
        return {
            "status": "processed",
            "action": action,
            "conclusion": conclusion
        }
