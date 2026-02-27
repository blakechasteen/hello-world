"""
GitHub Actions Skill
===================
Complete CI/CD workflow automation for GitHub Actions.

This skill wraps GitHub CLI (`gh`) to provide:
- Workflow triggering and monitoring
- Status checking and logs retrieval
- Workflow enable/disable management
- Run cancellation
- Artifacts download
"""

import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from hololoom.skills.base import (
    BaseSkill,
    SkillInput,
    SkillOutput,
    SkillMetadata,
    SkillCategory,
    SkillStatus
)


class WorkflowStatus(Enum):
    """GitHub Actions workflow status."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class WorkflowRun:
    """Structured workflow run result."""
    run_id: str
    workflow_name: str
    status: WorkflowStatus
    conclusion: Optional[str]
    created_at: str
    updated_at: str
    html_url: str
    run_number: int
    event: str
    head_branch: str
    head_sha: str


class GitHubActionsSkill(BaseSkill):
    """
    GitHub Actions Skill - CI/CD workflow automation.

    Operations:
    - trigger_workflow: Trigger a workflow run
    - list_workflows: List all workflows
    - get_workflow_status: Get status of a workflow run
    - get_workflow_logs: Download workflow logs
    - cancel_workflow: Cancel a running workflow
    - list_runs: List recent workflow runs
    - download_artifact: Download workflow artifacts
    - enable_workflow: Enable a workflow
    - disable_workflow: Disable a workflow
    """

    def __init__(self, gh_path: str = "gh", repo: Optional[str] = None):
        """
        Initialize GitHub Actions Skill.

        Args:
            gh_path: Path to gh executable (default: "gh" in PATH)
            repo: Repository in format "owner/repo" (default: auto-detect from git remote)
        """
        super().__init__()
        self.gh_path = gh_path
        self.repo = repo

    def get_metadata(self) -> SkillMetadata:
        """Return skill metadata."""
        return SkillMetadata(
            name="github_actions",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.INFRASTRUCTURE,
            tags=["ci", "cd", "github", "automation", "workflows", "devops"],
            description_short="Complete CI/CD workflow automation for GitHub Actions",
            description_long=(
                "GitHub Actions skill provides comprehensive automation for CI/CD "
                "workflows including triggering, monitoring, status checking, log "
                "retrieval, artifact downloading, and workflow management. Wraps "
                "GitHub CLI (gh) for seamless integration with GitHub Actions."
            ),
            requires_code_execution=True,
            requires_filesystem_read=True,
            requires_network=True,
            external_dependencies=["gh (GitHub CLI)"],
            expected_latency_ms="500-5000ms",  # 0.5-5 seconds depending on operation
            token_usage="200-1000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        """
        Validate GitHub Actions input.

        Args:
            skill_input: Input to validate

        Returns:
            True if valid

        Raises:
            ValueError: If input is invalid
        """
        valid_operations = [
            "trigger_workflow", "list_workflows", "get_workflow_status",
            "get_workflow_logs", "cancel_workflow", "list_runs",
            "download_artifact", "enable_workflow", "disable_workflow"
        ]

        if skill_input.operation not in valid_operations:
            raise ValueError(
                f"Invalid operation: {skill_input.operation}. "
                f"Must be one of: {', '.join(valid_operations)}"
            )

        # Validate operation-specific parameters
        if skill_input.operation == "trigger_workflow":
            if "workflow" not in skill_input.parameters:
                raise ValueError("trigger_workflow requires 'workflow' parameter")

        if skill_input.operation == "get_workflow_status":
            if "run_id" not in skill_input.parameters:
                raise ValueError("get_workflow_status requires 'run_id' parameter")

        if skill_input.operation == "get_workflow_logs":
            if "run_id" not in skill_input.parameters:
                raise ValueError("get_workflow_logs requires 'run_id' parameter")

        if skill_input.operation == "cancel_workflow":
            if "run_id" not in skill_input.parameters:
                raise ValueError("cancel_workflow requires 'run_id' parameter")

        if skill_input.operation == "download_artifact":
            if "run_id" not in skill_input.parameters or "artifact_name" not in skill_input.parameters:
                raise ValueError("download_artifact requires 'run_id' and 'artifact_name' parameters")

        if skill_input.operation in ["enable_workflow", "disable_workflow"]:
            if "workflow" not in skill_input.parameters:
                raise ValueError(f"{skill_input.operation} requires 'workflow' parameter")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        """
        Execute GitHub Actions skill.

        Args:
            skill_input: Skill input with operation and parameters

        Returns:
            SkillOutput with workflow results
        """
        start_time = time.time()

        try:
            # Dispatch to operation handler
            if skill_input.operation == "trigger_workflow":
                result = await self._trigger_workflow(skill_input.parameters)
            elif skill_input.operation == "list_workflows":
                result = await self._list_workflows(skill_input.parameters)
            elif skill_input.operation == "get_workflow_status":
                result = await self._get_workflow_status(skill_input.parameters)
            elif skill_input.operation == "get_workflow_logs":
                result = await self._get_workflow_logs(skill_input.parameters)
            elif skill_input.operation == "cancel_workflow":
                result = await self._cancel_workflow(skill_input.parameters)
            elif skill_input.operation == "list_runs":
                result = await self._list_runs(skill_input.parameters)
            elif skill_input.operation == "download_artifact":
                result = await self._download_artifact(skill_input.parameters)
            elif skill_input.operation == "enable_workflow":
                result = await self._enable_workflow(skill_input.parameters)
            elif skill_input.operation == "disable_workflow":
                result = await self._disable_workflow(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            execution_time = (time.time() - start_time) * 1000

            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"GitHub Actions operation '{skill_input.operation}' completed successfully",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result},
                warnings=[],
                errors=[],
                skill_name=self._metadata.name,
                skill_version=self._metadata.version
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"GitHub Actions operation failed: {e}",
                execution_time_ms=execution_time,
                errors=[str(e)],
                skill_name=self._metadata.name,
                skill_version=self._metadata.version
            )

    async def _run_gh_command(
        self,
        args: List[str],
        capture_output: bool = True
    ) -> str:
        """
        Run GitHub CLI command.

        Args:
            args: Command arguments
            capture_output: Whether to capture stdout

        Returns:
            Command output as string
        """
        # Build command
        cmd = [self.gh_path] + args

        # Add repo if specified
        if self.repo and "--repo" not in args:
            cmd.extend(["--repo", self.repo])

        # Run command
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
            raise RuntimeError(f"GitHub CLI command failed: {error_msg}")

        return stdout.decode('utf-8', errors='ignore') if stdout else ""

    async def _trigger_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger a workflow run."""
        workflow = parameters["workflow"]
        ref = parameters.get("ref", "main")
        inputs = parameters.get("inputs", {})

        # Build command
        args = ["workflow", "run", workflow, "--ref", ref]

        # Add inputs if provided
        for key, value in inputs.items():
            args.extend(["-f", f"{key}={value}"])

        # Trigger workflow
        await self._run_gh_command(args)

        return {
            "workflow": workflow,
            "ref": ref,
            "inputs": inputs,
            "message": f"Workflow '{workflow}' triggered on ref '{ref}'"
        }

    async def _list_workflows(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List all workflows."""
        limit = parameters.get("limit", 10)

        # Get workflows as JSON
        output = await self._run_gh_command([
            "workflow", "list",
            "--json", "name,id,state,path",
            "--limit", str(limit)
        ])

        workflows = json.loads(output)
        return workflows

    async def _get_workflow_status(self, parameters: Dict[str, Any]) -> WorkflowRun:
        """Get status of a workflow run."""
        run_id = parameters["run_id"]

        # Get run details as JSON
        output = await self._run_gh_command([
            "run", "view", str(run_id),
            "--json", "databaseId,workflowName,status,conclusion,createdAt,updatedAt,url,number,event,headBranch,headSha"
        ])

        data = json.loads(output)

        # Map status
        status_str = data.get("status", "").lower()
        if status_str == "completed":
            conclusion = data.get("conclusion", "").lower()
            if conclusion == "success":
                status = WorkflowStatus.SUCCESS
            elif conclusion == "failure":
                status = WorkflowStatus.FAILURE
            elif conclusion == "cancelled":
                status = WorkflowStatus.CANCELLED
            else:
                status = WorkflowStatus.COMPLETED
        elif status_str == "in_progress":
            status = WorkflowStatus.IN_PROGRESS
        elif status_str == "queued":
            status = WorkflowStatus.QUEUED
        else:
            status = WorkflowStatus.SKIPPED

        return WorkflowRun(
            run_id=str(data.get("databaseId", run_id)),
            workflow_name=data.get("workflowName", ""),
            status=status,
            conclusion=data.get("conclusion"),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
            html_url=data.get("url", ""),
            run_number=data.get("number", 0),
            event=data.get("event", ""),
            head_branch=data.get("headBranch", ""),
            head_sha=data.get("headSha", "")
        )

    async def _get_workflow_logs(self, parameters: Dict[str, Any]) -> str:
        """Download workflow logs."""
        run_id = parameters["run_id"]

        # Get logs
        output = await self._run_gh_command([
            "run", "view", str(run_id),
            "--log"
        ])

        return output

    async def _cancel_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a running workflow."""
        run_id = parameters["run_id"]

        # Cancel run
        await self._run_gh_command([
            "run", "cancel", str(run_id)
        ])

        return {
            "run_id": run_id,
            "message": f"Workflow run {run_id} cancelled"
        }

    async def _list_runs(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List recent workflow runs."""
        workflow = parameters.get("workflow")
        limit = parameters.get("limit", 10)

        # Build command
        args = ["run", "list", "--json", "databaseId,workflowName,status,conclusion,createdAt,url,number,event,headBranch"]

        if workflow:
            args.extend(["--workflow", workflow])

        args.extend(["--limit", str(limit)])

        # Get runs
        output = await self._run_gh_command(args)
        runs = json.loads(output)

        return runs

    async def _download_artifact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Download workflow artifacts."""
        run_id = parameters["run_id"]
        artifact_name = parameters["artifact_name"]
        output_dir = parameters.get("output_dir", ".")

        # Download artifact
        await self._run_gh_command([
            "run", "download", str(run_id),
            "--name", artifact_name,
            "--dir", output_dir
        ])

        return {
            "run_id": run_id,
            "artifact_name": artifact_name,
            "output_dir": output_dir,
            "message": f"Artifact '{artifact_name}' downloaded to '{output_dir}'"
        }

    async def _enable_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enable a workflow."""
        workflow = parameters["workflow"]

        # Enable workflow
        await self._run_gh_command([
            "workflow", "enable", workflow
        ])

        return {
            "workflow": workflow,
            "message": f"Workflow '{workflow}' enabled"
        }

    async def _disable_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Disable a workflow."""
        workflow = parameters["workflow"]

        # Disable workflow
        await self._run_gh_command([
            "workflow", "disable", workflow
        ])

        return {
            "workflow": workflow,
            "message": f"Workflow '{workflow}' disabled"
        }


# Register skill
from hololoom.skills.base import register_skill
register_skill(GitHubActionsSkill())
