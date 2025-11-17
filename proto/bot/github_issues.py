"""
Phase 5D: Issue Tracking
Create, manage, and track GitHub issues from Matrix.

Features:
- Create issues with title, body, labels, assignees
- Add comments to existing issues
- Close/reopen issues
- Link issues to PRs
- List and search issues
- Add/remove labels
- Assign/unassign users

Matrix Commands:
- !issue create <title> [--body <text>] [--label <label>] [--assign <user>]
- !issue comment <number> <text>
- !issue close <number>
- !issue reopen <number>
- !issue label <number> <label>
- !issue assign <number> <user>
- !issue list [--state open|closed|all] [--label <label>]
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from github import Github, GithubException
    from github.Issue import Issue
    from github.Repository import Repository
except ImportError:
    print("⚠️  PyGithub not installed. Run: pip install PyGithub==2.1.1")
    Github = None

from .github_auth import GitHubAuth


@dataclass
class IssueInfo:
    """Issue information."""

    number: int
    title: str
    state: str  # "open" or "closed"
    labels: List[str]
    assignees: List[str]
    author: str
    created_at: str
    updated_at: str
    comments_count: int
    url: str
    body: Optional[str] = None


class IssueManager:
    """Manage GitHub issues from Matrix."""

    def __init__(self, auth: Optional[GitHubAuth] = None):
        """Initialize issue manager."""
        self.auth = auth or GitHubAuth()

    def _get_client(self, installation_id: int) -> Github:
        """Get authenticated GitHub client."""
        if Github is None:
            raise ImportError("PyGithub not installed")

        token = self.auth.get_installation_token(installation_id)
        return Github(token)

    def _get_repo(self, installation_id: int, repo_full_name: str) -> Repository:
        """Get repository object."""
        client = self._get_client(installation_id)
        return client.get_repo(repo_full_name)

    async def create_issue(
        self,
        installation_id: int,
        repo_full_name: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository (e.g., "owner/repo")
            title: Issue title
            body: Issue description (optional)
            labels: List of label names (optional)
            assignees: List of GitHub usernames to assign (optional)

        Returns:
            Issue details dict
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)

            # Create issue
            issue = repo.create_issue(
                title=title,
                body=body or "",
                labels=labels or [],
                assignees=assignees or [],
            )

            return {
                "success": True,
                "issue_number": issue.number,
                "issue_url": issue.html_url,
                "title": issue.title,
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "assignees": [assignee.login for assignee in issue.assignees],
            }

        except GithubException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status,
            }

    async def get_issue(
        self, installation_id: int, repo_full_name: str, issue_number: int
    ) -> Dict[str, Any]:
        """
        Get issue details.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number

        Returns:
            Issue details dict
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue_info = IssueInfo(
                number=issue.number,
                title=issue.title,
                state=issue.state,
                labels=[label.name for label in issue.labels],
                assignees=[assignee.login for assignee in issue.assignees],
                author=issue.user.login,
                created_at=issue.created_at.isoformat(),
                updated_at=issue.updated_at.isoformat(),
                comments_count=issue.comments,
                url=issue.html_url,
                body=issue.body,
            )

            return {
                "success": True,
                "issue": issue_info.__dict__,
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def add_comment(
        self, installation_id: int, repo_full_name: str, issue_number: int, comment: str
    ) -> Dict[str, Any]:
        """
        Add comment to issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number
            comment: Comment text

        Returns:
            Comment details dict
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue_comment = issue.create_comment(comment)

            return {
                "success": True,
                "comment_id": issue_comment.id,
                "comment_url": issue_comment.html_url,
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def close_issue(
        self, installation_id: int, repo_full_name: str, issue_number: int
    ) -> Dict[str, Any]:
        """
        Close an issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number

        Returns:
            Success status
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue.edit(state="closed")

            return {
                "success": True,
                "issue_number": issue.number,
                "state": "closed",
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def reopen_issue(
        self, installation_id: int, repo_full_name: str, issue_number: int
    ) -> Dict[str, Any]:
        """
        Reopen a closed issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number

        Returns:
            Success status
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue.edit(state="open")

            return {
                "success": True,
                "issue_number": issue.number,
                "state": "open",
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def add_labels(
        self, installation_id: int, repo_full_name: str, issue_number: int, labels: List[str]
    ) -> Dict[str, Any]:
        """
        Add labels to issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number
            labels: List of label names

        Returns:
            Updated labels
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue.add_to_labels(*labels)

            return {
                "success": True,
                "labels": [label.name for label in issue.labels],
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def remove_label(
        self, installation_id: int, repo_full_name: str, issue_number: int, label: str
    ) -> Dict[str, Any]:
        """
        Remove label from issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number
            label: Label name to remove

        Returns:
            Updated labels
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue.remove_from_labels(label)

            return {
                "success": True,
                "labels": [label.name for label in issue.labels],
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def assign_user(
        self, installation_id: int, repo_full_name: str, issue_number: int, username: str
    ) -> Dict[str, Any]:
        """
        Assign user to issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number
            username: GitHub username

        Returns:
            Updated assignees
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue.add_to_assignees(username)

            return {
                "success": True,
                "assignees": [assignee.login for assignee in issue.assignees],
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def unassign_user(
        self, installation_id: int, repo_full_name: str, issue_number: int, username: str
    ) -> Dict[str, Any]:
        """
        Unassign user from issue.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            issue_number: Issue number
            username: GitHub username

        Returns:
            Updated assignees
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            issue = repo.get_issue(issue_number)

            issue.remove_from_assignees(username)

            return {
                "success": True,
                "assignees": [assignee.login for assignee in issue.assignees],
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def list_issues(
        self,
        installation_id: int,
        repo_full_name: str,
        state: str = "open",  # "open", "closed", "all"
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        List issues with filters.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            state: Issue state filter
            labels: Filter by labels
            assignee: Filter by assignee
            limit: Maximum issues to return

        Returns:
            List of issues
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)

            # Build query parameters
            params = {"state": state}
            if labels:
                params["labels"] = labels
            if assignee:
                params["assignee"] = assignee

            # Get issues
            issues = repo.get_issues(**params)

            # Convert to issue info
            issue_list = []
            for i, issue in enumerate(issues):
                if i >= limit:
                    break

                issue_list.append(
                    IssueInfo(
                        number=issue.number,
                        title=issue.title,
                        state=issue.state,
                        labels=[label.name for label in issue.labels],
                        assignees=[assignee.login for assignee in issue.assignees],
                        author=issue.user.login,
                        created_at=issue.created_at.isoformat(),
                        updated_at=issue.updated_at.isoformat(),
                        comments_count=issue.comments,
                        url=issue.html_url,
                    ).__dict__
                )

            return {
                "success": True,
                "issues": issue_list,
                "count": len(issue_list),
            }

        except GithubException as e:
            return {"success": False, "error": str(e)}


# Matrix command handlers


async def handle_issue_create_command(
    manager: IssueManager,
    installation_id: int,
    repo: str,
    title: str,
    body: Optional[str] = None,
    labels: Optional[List[str]] = None,
    assignees: Optional[List[str]] = None,
) -> str:
    """Handle !issue create command."""
    result = await manager.create_issue(
        installation_id=installation_id,
        repo_full_name=repo,
        title=title,
        body=body,
        labels=labels,
        assignees=assignees,
    )

    if result["success"]:
        msg_lines = [
            f"✅ **Issue Created!**\n",
            f"**#{result['issue_number']}**: {result['title']}",
            f"🔗 {result['issue_url']}",
        ]

        if result["labels"]:
            msg_lines.append(f"🏷️  Labels: {', '.join(result['labels'])}")

        if result["assignees"]:
            msg_lines.append(f"👤 Assigned: {', '.join(result['assignees'])}")

        return "\n".join(msg_lines)
    else:
        return f"❌ **Failed to create issue**: {result['error']}"


async def handle_issue_comment_command(
    manager: IssueManager, installation_id: int, repo: str, issue_number: int, comment: str
) -> str:
    """Handle !issue comment command."""
    result = await manager.add_comment(
        installation_id=installation_id, repo_full_name=repo, issue_number=issue_number, comment=comment
    )

    if result["success"]:
        return f"✅ **Comment added to issue #{issue_number}**\n\n🔗 {result['comment_url']}"
    else:
        return f"❌ **Failed to add comment**: {result['error']}"


async def handle_issue_close_command(
    manager: IssueManager, installation_id: int, repo: str, issue_number: int
) -> str:
    """Handle !issue close command."""
    result = await manager.close_issue(
        installation_id=installation_id, repo_full_name=repo, issue_number=issue_number
    )

    if result["success"]:
        return f"✅ **Issue #{issue_number} closed**"
    else:
        return f"❌ **Failed to close issue**: {result['error']}"


async def handle_issue_list_command(
    manager: IssueManager,
    installation_id: int,
    repo: str,
    state: str = "open",
    labels: Optional[List[str]] = None,
    limit: int = 10,
) -> str:
    """Handle !issue list command."""
    result = await manager.list_issues(
        installation_id=installation_id,
        repo_full_name=repo,
        state=state,
        labels=labels,
        limit=limit,
    )

    if not result["success"]:
        return f"❌ **Error**: {result['error']}"

    if result["count"] == 0:
        return f"📋 No {state} issues found."

    # Build issue list
    msg_lines = [f"📋 **{state.title()} Issues** ({result['count']})\n"]

    for issue in result["issues"]:
        labels_str = f"[{', '.join(issue['labels'])}]" if issue["labels"] else ""
        assignees_str = f"👤 {', '.join(issue['assignees'])}" if issue["assignees"] else ""

        msg_lines.append(f"**#{issue['number']}** {issue['title']} {labels_str}")

        if assignees_str:
            msg_lines.append(f"  {assignees_str}")

        msg_lines.append(f"  🔗 {issue['url']}")
        msg_lines.append("")  # Blank line

    return "\n".join(msg_lines)


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo issue management."""
        manager = IssueManager()

        print("\n=== Creating Issue ===")
        result = await manager.create_issue(
            installation_id=12345678,  # Replace with real installation ID
            repo_full_name="your-username/your-repo",
            title="Dashboard fails to load on Firefox",
            body="**Steps to Reproduce:**\n1. Open dashboard in Firefox\n2. Click on 'Workflows' tab\n3. Page hangs\n\n**Expected:** Tab loads\n**Actual:** Page hangs",
            labels=["bug", "dashboard"],
            assignees=["alice"],
        )
        print(result)

        if result["success"]:
            issue_number = result["issue_number"]

            # Add comment
            print(f"\n=== Adding Comment to Issue #{issue_number} ===")
            comment_result = await manager.add_comment(
                installation_id=12345678,
                repo_full_name="your-username/your-repo",
                issue_number=issue_number,
                comment="Investigating this issue. Appears to be a React Flow compatibility problem.",
            )
            print(comment_result)

            # List issues
            print("\n=== Listing Open Issues ===")
            list_result = await manager.list_issues(
                installation_id=12345678, repo_full_name="your-username/your-repo", state="open", limit=5
            )
            print(list_result)

    asyncio.run(demo())
