"""
Phase 5B: PR Management
Pull request creation, review, and merge from Matrix commands.

Features:
- Create PRs from branches with auto-generated descriptions
- Add reviewers based on CODEOWNERS
- Comment on PRs
- Approve/request changes
- Merge with checks
- PR status monitoring

Matrix Commands:
- !pr create <branch> [title] - Create PR from branch
- !pr review <number> - Review PR
- !pr merge <number> - Merge PR
- !pr status <number> - Get PR status
- !pr comment <number> <text> - Add comment
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from github import Github, GithubException
    from github.PullRequest import PullRequest
    from github.Repository import Repository
except ImportError:
    print("⚠️  PyGithub not installed. Run: pip install PyGithub==2.1.1")
    Github = None

from .github_auth import GitHubAuth


class PRManager:
    """Manage GitHub Pull Requests from Matrix."""

    def __init__(self, auth: Optional[GitHubAuth] = None):
        """Initialize PR manager with authentication."""
        self.auth = auth or GitHubAuth()
        self._github_client: Optional[Github] = None

    def _get_client(self, installation_id: int) -> Github:
        """Get authenticated GitHub client for installation."""
        if Github is None:
            raise ImportError("PyGithub not installed")

        token = self.auth.get_installation_token(installation_id)
        return Github(token)

    def _get_repo(self, installation_id: int, repo_full_name: str) -> Repository:
        """Get repository object."""
        client = self._get_client(installation_id)
        return client.get_repo(repo_full_name)

    async def create_pr(
        self,
        installation_id: int,
        repo_full_name: str,
        head_branch: str,
        base_branch: str = "main",
        title: Optional[str] = None,
        body: Optional[str] = None,
        draft: bool = False,
        reviewers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a pull request.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository (e.g., "owner/repo")
            head_branch: Source branch
            base_branch: Target branch (default: main)
            title: PR title (auto-generated if None)
            body: PR description (auto-generated if None)
            draft: Create as draft PR
            reviewers: List of GitHub usernames to request review

        Returns:
            PR details dict
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)

            # Auto-generate title from branch name if not provided
            if title is None:
                title = self._generate_title_from_branch(head_branch)

            # Auto-generate body from commits if not provided
            if body is None:
                body = await self._generate_body_from_commits(repo, head_branch, base_branch)

            # Create PR
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch,
                draft=draft,
            )

            # Add reviewers
            if reviewers:
                pr.create_review_request(reviewers=reviewers)
            else:
                # Try to add reviewers from CODEOWNERS
                codeowners_reviewers = self._get_codeowners_reviewers(repo)
                if codeowners_reviewers:
                    pr.create_review_request(reviewers=codeowners_reviewers[:5])  # Max 5

            return {
                "success": True,
                "pr_number": pr.number,
                "pr_url": pr.html_url,
                "title": pr.title,
                "state": pr.state,
                "draft": pr.draft,
                "reviewers": [r.login for r in pr.requested_reviewers],
            }

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status,
            }

    async def get_pr_status(
        self,
        installation_id: int,
        repo_full_name: str,
        pr_number: int,
    ) -> Dict[str, Any]:
        """
        Get detailed PR status.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            pr_number: PR number

        Returns:
            PR status details
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            pr = repo.get_pull(pr_number)

            # Get reviews
            reviews = list(pr.get_reviews())
            review_states = {}
            for review in reviews:
                # Keep only latest review from each user
                review_states[review.user.login] = review.state

            # Count approvals/changes requested
            approvals = sum(1 for state in review_states.values() if state == "APPROVED")
            changes_requested = sum(
                1 for state in review_states.values() if state == "CHANGES_REQUESTED"
            )

            # Get CI status
            commit = pr.get_commits().reversed[0]
            ci_status = commit.get_combined_status()

            # Check if mergeable
            mergeable = pr.mergeable and pr.mergeable_state == "clean"

            return {
                "success": True,
                "pr_number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "draft": pr.draft,
                "mergeable": mergeable,
                "mergeable_state": pr.mergeable_state,
                "reviews": {
                    "total": len(review_states),
                    "approved": approvals,
                    "changes_requested": changes_requested,
                    "reviewers": review_states,
                },
                "ci": {
                    "state": ci_status.state,
                    "total_checks": ci_status.total_count,
                    "statuses": [
                        {"context": s.context, "state": s.state, "description": s.description}
                        for s in ci_status.statuses
                    ],
                },
                "commits": pr.commits,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changed_files": pr.changed_files,
                "url": pr.html_url,
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def add_comment(
        self,
        installation_id: int,
        repo_full_name: str,
        pr_number: int,
        comment: str,
    ) -> Dict[str, Any]:
        """Add comment to PR."""
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            pr = repo.get_pull(pr_number)

            issue_comment = pr.create_issue_comment(comment)

            return {
                "success": True,
                "comment_id": issue_comment.id,
                "comment_url": issue_comment.html_url,
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def review_pr(
        self,
        installation_id: int,
        repo_full_name: str,
        pr_number: int,
        event: str = "COMMENT",  # APPROVE, REQUEST_CHANGES, COMMENT
        body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a review on a PR.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            pr_number: PR number
            event: Review event (APPROVE, REQUEST_CHANGES, COMMENT)
            body: Review comment

        Returns:
            Review details
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            pr = repo.get_pull(pr_number)

            review = pr.create_review(body=body, event=event)

            return {
                "success": True,
                "review_id": review.id,
                "state": review.state,
                "body": review.body,
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def merge_pr(
        self,
        installation_id: int,
        repo_full_name: str,
        pr_number: int,
        merge_method: str = "squash",  # merge, squash, rebase
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Merge a pull request.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            pr_number: PR number
            merge_method: Merge method (merge, squash, rebase)
            commit_title: Merge commit title
            commit_message: Merge commit message

        Returns:
            Merge result
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            pr = repo.get_pull(pr_number)

            # Check if mergeable
            if not pr.mergeable:
                return {
                    "success": False,
                    "error": f"PR is not mergeable. State: {pr.mergeable_state}",
                }

            # Merge
            merge_result = pr.merge(
                commit_title=commit_title,
                commit_message=commit_message,
                merge_method=merge_method,
            )

            return {
                "success": merge_result.merged,
                "sha": merge_result.sha,
                "message": merge_result.message,
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    # Helper methods

    def _generate_title_from_branch(self, branch_name: str) -> str:
        """Generate PR title from branch name."""
        # Remove common prefixes
        for prefix in ["feature/", "bugfix/", "hotfix/", "fix/", "feat/"]:
            if branch_name.startswith(prefix):
                branch_name = branch_name[len(prefix) :]
                break

        # Convert kebab-case or snake_case to Title Case
        title = branch_name.replace("-", " ").replace("_", " ").title()

        return title

    async def _generate_body_from_commits(
        self, repo: Repository, head_branch: str, base_branch: str
    ) -> str:
        """Generate PR body from commit messages."""
        try:
            # Compare branches
            comparison = repo.compare(base_branch, head_branch)

            # Extract commit messages
            commits = comparison.commits
            if not commits:
                return "No commits found."

            # Build body
            body_lines = ["## Changes\n"]

            for commit in commits:
                message = commit.commit.message.split("\n")[0]  # First line only
                body_lines.append(f"- {message}")

            body_lines.append(
                f"\n## Summary\n\n{comparison.total_commits} commit(s), "
                f"+{comparison.ahead_by} ahead of {base_branch}"
            )

            return "\n".join(body_lines)

        except Exception:
            return "Auto-generated pull request."

    def _get_codeowners_reviewers(self, repo: Repository) -> List[str]:
        """Get reviewers from CODEOWNERS file."""
        try:
            # Try to fetch CODEOWNERS
            for path in [".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"]:
                try:
                    content = repo.get_contents(path)
                    if content:
                        codeowners_text = content.decoded_content.decode("utf-8")
                        return self._parse_codeowners(codeowners_text)
                except GithubException:
                    continue

            return []

        except Exception:
            return []

    def _parse_codeowners(self, codeowners_text: str) -> List[str]:
        """Parse CODEOWNERS file to extract usernames."""
        reviewers = set()

        for line in codeowners_text.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Extract usernames (format: * @username1 @username2)
            parts = line.split()
            for part in parts:
                if part.startswith("@") and "/" not in part:  # User, not team
                    reviewers.add(part[1:])  # Remove @

        return list(reviewers)


# Matrix command handlers (to be integrated with Matrix bot)


async def handle_pr_create_command(
    manager: PRManager,
    installation_id: int,
    repo: str,
    branch: str,
    title: Optional[str] = None,
) -> str:
    """Handle !pr create command."""
    result = await manager.create_pr(
        installation_id=installation_id,
        repo_full_name=repo,
        head_branch=branch,
        title=title,
    )

    if result["success"]:
        return (
            f"✅ **PR Created!**\n\n"
            f"**#{result['pr_number']}**: {result['title']}\n"
            f"🔗 {result['pr_url']}\n"
            f"👥 Reviewers: {', '.join(result['reviewers']) if result['reviewers'] else 'None'}"
        )
    else:
        return f"❌ **Failed to create PR**: {result['error']}"


async def handle_pr_status_command(
    manager: PRManager, installation_id: int, repo: str, pr_number: int
) -> str:
    """Handle !pr status command."""
    result = await manager.get_pr_status(
        installation_id=installation_id, repo_full_name=repo, pr_number=pr_number
    )

    if not result["success"]:
        return f"❌ **Error**: {result['error']}"

    # Build status message
    msg_lines = [
        f"📊 **PR #{result['pr_number']} Status**\n",
        f"**Title**: {result['title']}",
        f"**State**: {result['state'].upper()}",
        f"**Draft**: {'Yes' if result['draft'] else 'No'}",
        f"**Mergeable**: {'✅ Yes' if result['mergeable'] else '❌ No'} ({result['mergeable_state']})",
        "",
        "**Reviews**:",
        f"  ✅ Approved: {result['reviews']['approved']}",
        f"  ⚠️  Changes Requested: {result['reviews']['changes_requested']}",
        "",
        "**CI Checks**:",
        f"  State: {result['ci']['state'].upper()}",
        f"  Total: {result['ci']['total_checks']}",
    ]

    # Add failing checks
    failing = [s for s in result["ci"]["statuses"] if s["state"] == "failure"]
    if failing:
        msg_lines.append("\n  ❌ Failed:")
        for check in failing:
            msg_lines.append(f"    - {check['context']}: {check['description']}")

    msg_lines.append(f"\n🔗 {result['url']}")

    return "\n".join(msg_lines)


async def handle_pr_merge_command(
    manager: PRManager, installation_id: int, repo: str, pr_number: int
) -> str:
    """Handle !pr merge command."""
    # First check status
    status = await manager.get_pr_status(installation_id, repo, pr_number)

    if not status["success"]:
        return f"❌ **Error**: {status['error']}"

    # Check if mergeable
    if not status["mergeable"]:
        return (
            f"❌ **Cannot merge PR #{pr_number}**\n\n"
            f"State: {status['mergeable_state']}\n"
            f"Approvals: {status['reviews']['approved']}\n"
            f"CI: {status['ci']['state']}"
        )

    # Merge
    result = await manager.merge_pr(installation_id, repo, pr_number)

    if result["success"]:
        return f"✅ **PR #{pr_number} merged!**\n\nCommit: {result['sha'][:7]}"
    else:
        return f"❌ **Merge failed**: {result['error']}"


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo PR management."""
        manager = PRManager()

        # Example: Create PR
        print("\n=== Creating PR ===")
        result = await manager.create_pr(
            installation_id=12345678,  # Replace with real installation ID
            repo_full_name="your-username/your-repo",
            head_branch="feature-dashboard",
            base_branch="main",
            title="Add Phase 4 Dashboard",
        )
        print(result)

        if result["success"]:
            pr_number = result["pr_number"]

            # Example: Get status
            print(f"\n=== PR #{pr_number} Status ===")
            status = await manager.get_pr_status(
                installation_id=12345678, repo_full_name="your-username/your-repo", pr_number=pr_number
            )
            print(status)

            # Example: Add comment
            print(f"\n=== Adding Comment ===")
            comment_result = await manager.add_comment(
                installation_id=12345678,
                repo_full_name="your-username/your-repo",
                pr_number=pr_number,
                comment="🤖 Automated review: LGTM!",
            )
            print(comment_result)

    asyncio.run(demo())
