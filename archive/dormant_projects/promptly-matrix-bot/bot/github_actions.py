"""
Phase 5E: CI/CD Triggers
Trigger and monitor GitHub Actions workflows from Matrix.

Features:
- List available workflows
- Trigger workflow runs
- Get workflow run status
- Cancel workflow runs
- Monitor real-time build progress
- Send notifications to Matrix on workflow events

Matrix Commands:
- !build trigger <workflow> [branch] - Trigger workflow
- !build status <run_id> - Get run status
- !build cancel <run_id> - Cancel run
- !build list - List recent runs
- !build workflows - List available workflows

Webhooks:
- workflow_run events → Real-time notifications
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from github import Github, GithubException
    from github.Workflow import Workflow
    from github.WorkflowRun import WorkflowRun
    from github.Repository import Repository
except ImportError:
    print("⚠️  PyGithub not installed. Run: pip install PyGithub==2.1.1")
    Github = None

from .github_auth import GitHubAuth


class WorkflowStatus(Enum):
    """Workflow run status."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class WorkflowConclusion(Enum):
    """Workflow run conclusion."""

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


@dataclass
class WorkflowRunInfo:
    """Workflow run information."""

    id: int
    workflow_name: str
    status: str
    conclusion: Optional[str]
    branch: str
    commit_sha: str
    commit_message: str
    author: str
    created_at: str
    updated_at: str
    url: str
    duration_seconds: Optional[int] = None
    jobs: Optional[List[Dict[str, Any]]] = None


class CICDManager:
    """Manage GitHub Actions CI/CD from Matrix."""

    def __init__(self, auth: Optional[GitHubAuth] = None):
        """Initialize CI/CD manager."""
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

    async def list_workflows(self, installation_id: int, repo_full_name: str) -> Dict[str, Any]:
        """
        List available workflows.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository (e.g., "owner/repo")

        Returns:
            List of workflows
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            workflows = repo.get_workflows()

            workflow_list = []
            for workflow in workflows:
                workflow_list.append(
                    {
                        "id": workflow.id,
                        "name": workflow.name,
                        "path": workflow.path,
                        "state": workflow.state,
                    }
                )

            return {"success": True, "workflows": workflow_list, "count": len(workflow_list)}

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def trigger_workflow(
        self,
        installation_id: int,
        repo_full_name: str,
        workflow_id: str,  # Workflow file name or ID
        ref: str = "main",  # Branch/tag/commit to run on
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger a workflow run.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            workflow_id: Workflow file name (e.g., "ci.yml") or workflow ID
            ref: Branch, tag, or commit SHA to run on
            inputs: Workflow inputs (for workflow_dispatch)

        Returns:
            Workflow run details
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            workflow = repo.get_workflow(workflow_id)

            # Trigger workflow
            workflow.create_dispatch(ref=ref, inputs=inputs or {})

            # Get the latest run (just triggered)
            runs = workflow.get_runs(branch=ref)
            if runs.totalCount > 0:
                latest_run = runs[0]

                return {
                    "success": True,
                    "run_id": latest_run.id,
                    "run_url": latest_run.html_url,
                    "workflow_name": workflow.name,
                    "status": latest_run.status,
                }
            else:
                return {
                    "success": True,
                    "message": "Workflow triggered, but run not yet visible",
                }

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def get_run_status(
        self, installation_id: int, repo_full_name: str, run_id: int
    ) -> Dict[str, Any]:
        """
        Get workflow run status.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            run_id: Workflow run ID

        Returns:
            Detailed run status
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            run = repo.get_workflow_run(run_id)

            # Calculate duration
            duration_seconds = None
            if run.updated_at and run.created_at:
                duration = run.updated_at - run.created_at
                duration_seconds = int(duration.total_seconds())

            # Get jobs
            jobs = []
            try:
                for job in run.jobs():
                    job_duration = None
                    if job.completed_at and job.started_at:
                        job_duration = int((job.completed_at - job.started_at).total_seconds())

                    jobs.append(
                        {
                            "name": job.name,
                            "status": job.status,
                            "conclusion": job.conclusion,
                            "duration_seconds": job_duration,
                            "steps": [
                                {
                                    "name": step.name,
                                    "status": step.status,
                                    "conclusion": step.conclusion,
                                    "number": step.number,
                                }
                                for step in job.steps
                            ],
                        }
                    )
            except Exception:
                # Jobs may not be available yet
                pass

            run_info = WorkflowRunInfo(
                id=run.id,
                workflow_name=run.name,
                status=run.status,
                conclusion=run.conclusion,
                branch=run.head_branch,
                commit_sha=run.head_sha[:7],
                commit_message=run.head_commit.message.split("\n")[0] if run.head_commit else "",
                author=run.head_commit.author.name if run.head_commit else "Unknown",
                created_at=run.created_at.isoformat(),
                updated_at=run.updated_at.isoformat(),
                url=run.html_url,
                duration_seconds=duration_seconds,
                jobs=jobs if jobs else None,
            )

            return {"success": True, "run": run_info.__dict__}

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def cancel_run(
        self, installation_id: int, repo_full_name: str, run_id: int
    ) -> Dict[str, Any]:
        """
        Cancel a workflow run.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            run_id: Workflow run ID

        Returns:
            Cancellation status
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)
            run = repo.get_workflow_run(run_id)

            run.cancel()

            return {"success": True, "run_id": run_id, "status": "cancelled"}

        except GithubException as e:
            return {"success": False, "error": str(e)}

    async def list_runs(
        self,
        installation_id: int,
        repo_full_name: str,
        branch: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        List recent workflow runs.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository
            branch: Filter by branch
            status: Filter by status (queued, in_progress, completed)
            limit: Maximum runs to return

        Returns:
            List of workflow runs
        """
        try:
            repo = self._get_repo(installation_id, repo_full_name)

            # Build query parameters
            params = {}
            if branch:
                params["branch"] = branch
            if status:
                params["status"] = status

            # Get runs
            runs = repo.get_workflow_runs(**params)

            # Convert to run info
            run_list = []
            for i, run in enumerate(runs):
                if i >= limit:
                    break

                # Calculate duration
                duration_seconds = None
                if run.updated_at and run.created_at:
                    duration = run.updated_at - run.created_at
                    duration_seconds = int(duration.total_seconds())

                run_list.append(
                    WorkflowRunInfo(
                        id=run.id,
                        workflow_name=run.name,
                        status=run.status,
                        conclusion=run.conclusion,
                        branch=run.head_branch,
                        commit_sha=run.head_sha[:7],
                        commit_message=run.head_commit.message.split("\n")[0]
                        if run.head_commit
                        else "",
                        author=run.head_commit.author.name if run.head_commit else "Unknown",
                        created_at=run.created_at.isoformat(),
                        updated_at=run.updated_at.isoformat(),
                        url=run.html_url,
                        duration_seconds=duration_seconds,
                    ).__dict__
                )

            return {"success": True, "runs": run_list, "count": len(run_list)}

        except GithubException as e:
            return {"success": False, "error": str(e)}


# Matrix command handlers


async def handle_build_trigger_command(
    manager: CICDManager,
    installation_id: int,
    repo: str,
    workflow: str,
    branch: str = "main",
) -> str:
    """Handle !build trigger command."""
    result = await manager.trigger_workflow(
        installation_id=installation_id, repo_full_name=repo, workflow_id=workflow, ref=branch
    )

    if result["success"]:
        if "run_id" in result:
            return (
                f"🏗️  **Build Triggered!**\n\n"
                f"**Workflow**: {result['workflow_name']}\n"
                f"**Run ID**: {result['run_id']}\n"
                f"**Status**: {result['status']}\n"
                f"🔗 {result['run_url']}"
            )
        else:
            return f"🏗️  **Build Triggered!**\n\n{result['message']}"
    else:
        return f"❌ **Failed to trigger build**: {result['error']}"


async def handle_build_status_command(
    manager: CICDManager, installation_id: int, repo: str, run_id: int
) -> str:
    """Handle !build status command."""
    result = await manager.get_run_status(
        installation_id=installation_id, repo_full_name=repo, run_id=run_id
    )

    if not result["success"]:
        return f"❌ **Error**: {result['error']}"

    run = result["run"]

    # Status emoji
    status_emoji = {
        "queued": "⏳",
        "in_progress": "🏗️",
        "completed": "✅" if run["conclusion"] == "success" else "❌",
    }.get(run["status"], "ℹ️")

    # Conclusion emoji
    conclusion_emoji = {
        "success": "✅",
        "failure": "❌",
        "cancelled": "🚫",
        "skipped": "⏭️",
        "timed_out": "⏱️",
    }.get(run["conclusion"] or "", "")

    msg_lines = [
        f"{status_emoji} **Build Status**\n",
        f"**Workflow**: {run['workflow_name']}",
        f"**Run ID**: {run['id']}",
        f"**Status**: {run['status'].replace('_', ' ').title()}",
    ]

    if run["conclusion"]:
        msg_lines.append(f"**Result**: {conclusion_emoji} {run['conclusion'].replace('_', ' ').title()}")

    msg_lines.extend(
        [
            f"**Branch**: {run['branch']}",
            f"**Commit**: {run['commit_sha']} - {run['commit_message']}",
            f"**Author**: {run['author']}",
        ]
    )

    if run["duration_seconds"]:
        minutes = run["duration_seconds"] // 60
        seconds = run["duration_seconds"] % 60
        msg_lines.append(f"**Duration**: {minutes}m {seconds}s")

    # Add job details
    if run["jobs"]:
        msg_lines.append("\n**Jobs**:")
        for job in run["jobs"]:
            job_emoji = {
                "queued": "⏳",
                "in_progress": "🏗️",
                "completed": "✅" if job["conclusion"] == "success" else "❌",
            }.get(job["status"], "")

            duration_str = ""
            if job["duration_seconds"]:
                m = job["duration_seconds"] // 60
                s = job["duration_seconds"] % 60
                duration_str = f" ({m}m {s}s)"

            msg_lines.append(f"  {job_emoji} {job['name']}{duration_str}")

    msg_lines.append(f"\n🔗 {run['url']}")

    return "\n".join(msg_lines)


async def handle_build_list_command(
    manager: CICDManager,
    installation_id: int,
    repo: str,
    branch: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Handle !build list command."""
    result = await manager.list_runs(
        installation_id=installation_id, repo_full_name=repo, branch=branch, limit=limit
    )

    if not result["success"]:
        return f"❌ **Error**: {result['error']}"

    if result["count"] == 0:
        return "📋 No recent builds found."

    msg_lines = [f"📋 **Recent Builds** ({result['count']})\n"]

    for run in result["runs"]:
        # Status/conclusion emoji
        if run["status"] == "completed":
            emoji = {"success": "✅", "failure": "❌", "cancelled": "🚫"}.get(
                run["conclusion"] or "", "ℹ️"
            )
        else:
            emoji = {"queued": "⏳", "in_progress": "🏗️"}.get(run["status"], "ℹ️")

        duration_str = ""
        if run["duration_seconds"]:
            m = run["duration_seconds"] // 60
            s = run["duration_seconds"] % 60
            duration_str = f" ({m}m {s}s)"

        msg_lines.extend(
            [
                f"{emoji} **{run['workflow_name']}** #{run['id']}{duration_str}",
                f"  {run['branch']} - {run['commit_sha']}: {run['commit_message'][:50]}",
                f"  🔗 {run['url']}",
                "",
            ]
        )

    return "\n".join(msg_lines)


async def handle_build_workflows_command(
    manager: CICDManager, installation_id: int, repo: str
) -> str:
    """Handle !build workflows command."""
    result = await manager.list_workflows(installation_id=installation_id, repo_full_name=repo)

    if not result["success"]:
        return f"❌ **Error**: {result['error']}"

    if result["count"] == 0:
        return "📋 No workflows found."

    msg_lines = [f"📋 **Available Workflows** ({result['count']})\n"]

    for workflow in result["workflows"]:
        state_emoji = "✅" if workflow["state"] == "active" else "⚠️"
        msg_lines.append(f"{state_emoji} **{workflow['name']}** (`{workflow['path']}`)")

    msg_lines.append("\n💡 Use `!build trigger <workflow>` to trigger a workflow.")

    return "\n".join(msg_lines)


# Webhook handler for workflow events
def handle_workflow_run_webhook(payload: Dict[str, Any]) -> str:
    """
    Handle workflow_run webhook event.

    Args:
        payload: GitHub webhook payload

    Returns:
        Matrix message to send
    """
    action = payload.get("action")
    workflow_run = payload.get("workflow_run", {})

    workflow_name = workflow_run.get("name", "Workflow")
    run_id = workflow_run.get("id")
    status = workflow_run.get("status")
    conclusion = workflow_run.get("conclusion")
    branch = workflow_run.get("head_branch")
    commit = workflow_run.get("head_commit", {})
    commit_message = commit.get("message", "").split("\n")[0]
    author = commit.get("author", {}).get("name", "Unknown")
    url = workflow_run.get("html_url")

    # Build message based on action
    if action == "requested" or action == "queued":
        return (
            f"⏳ **Build Queued**\n\n"
            f"**Workflow**: {workflow_name}\n"
            f"**Branch**: {branch}\n"
            f"**Commit**: {commit_message}\n"
            f"🔗 {url}"
        )

    elif action == "in_progress":
        return (
            f"🏗️  **Build Started**\n\n"
            f"**Workflow**: {workflow_name}\n"
            f"**Branch**: {branch}\n"
            f"**Commit**: {commit_message}\n"
            f"**Author**: {author}\n"
            f"🔗 {url}"
        )

    elif action == "completed":
        # Conclusion emoji
        conclusion_emoji = {
            "success": "✅",
            "failure": "❌",
            "cancelled": "🚫",
            "skipped": "⏭️",
            "timed_out": "⏱️",
        }.get(conclusion or "", "ℹ️")

        return (
            f"{conclusion_emoji} **Build {conclusion.replace('_', ' ').title() if conclusion else 'Complete'}**\n\n"
            f"**Workflow**: {workflow_name}\n"
            f"**Branch**: {branch}\n"
            f"**Commit**: {commit_message}\n"
            f"**Author**: {author}\n"
            f"🔗 {url}"
        )

    else:
        return ""


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo CI/CD management."""
        manager = CICDManager()

        print("\n=== Listing Workflows ===")
        workflows = await manager.list_workflows(
            installation_id=12345678,  # Replace with real installation ID
            repo_full_name="your-username/your-repo",
        )
        print(workflows)

        if workflows["success"] and workflows["count"] > 0:
            workflow_path = workflows["workflows"][0]["path"]

            print(f"\n=== Triggering Workflow: {workflow_path} ===")
            trigger_result = await manager.trigger_workflow(
                installation_id=12345678,
                repo_full_name="your-username/your-repo",
                workflow_id=workflow_path,
                ref="main",
            )
            print(trigger_result)

        print("\n=== Listing Recent Runs ===")
        runs = await manager.list_runs(
            installation_id=12345678, repo_full_name="your-username/your-repo", limit=5
        )
        print(runs)

    asyncio.run(demo())
