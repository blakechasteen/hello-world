"""
Enhanced Kubernetes Deployer for CARTS Sandbox
===============================================

Kubernetes deployment with Kustomize overlays, RBAC,
NetworkPolicy, and multi-environment support.

Author: CARTS Team
Date: 2025-12-09
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import DeploymentConfig

logger = logging.getLogger("carts-deploy.k8s")


@dataclass
class PodStatus:
    """Kubernetes pod status."""
    name: str
    namespace: str
    phase: str  # Pending, Running, Succeeded, Failed, Unknown
    ready: bool
    restarts: int
    age_seconds: float
    node: str
    ip: str


@dataclass
class JobStatus:
    """Kubernetes job status."""
    name: str
    namespace: str
    active: int
    succeeded: int
    failed: int
    completions: int
    parallelism: int
    duration_seconds: float
    conditions: list[str]


class EnhancedKubernetesDeployer:
    """Enhanced Kubernetes deployer with Kustomize support."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifests_dir = Path(__file__).parent.parent.parent.parent / "infra" / "kubernetes" / "sandbox"

    async def setup_cluster(self) -> dict[str, Any]:
        """Verify cluster access and setup namespace with RBAC."""
        logger.info("Setting up Kubernetes cluster access...")

        results = {
            "cluster_accessible": False,
            "namespace_ready": False,
            "rbac_configured": False,
            "network_policy_ready": False,
        }

        # Check cluster access
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            results["cluster_accessible"] = result.returncode == 0

            if not results["cluster_accessible"]:
                logger.error("Cannot access Kubernetes cluster")
                return results

            logger.info("Cluster accessible")

        except Exception as e:
            logger.error(f"Failed to check cluster: {e}")
            return results

        # Apply base manifests
        try:
            await self._apply_kustomize(self.manifests_dir / "base")
            results["namespace_ready"] = True
            results["rbac_configured"] = True
            results["network_policy_ready"] = True
            logger.info(f"Base manifests applied to namespace: {self.config.namespace}")

        except Exception as e:
            logger.error(f"Failed to apply base manifests: {e}")

        return results

    async def deploy(self, overlay: str = None) -> dict[str, Any]:
        """Deploy sandbox using Kustomize overlay."""
        overlay = overlay or self.config.environment.value
        overlay_path = self.manifests_dir / "overlays" / overlay

        logger.info(f"Deploying with overlay: {overlay}")

        if not overlay_path.exists():
            logger.error(f"Overlay not found: {overlay_path}")
            return {"success": False, "error": f"Overlay not found: {overlay}"}

        try:
            await self._apply_kustomize(overlay_path)

            return {
                "success": True,
                "overlay": overlay,
                "namespace": self.config.namespace,
                "name": self.config.name,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_job(
        self,
        command: list[str],
        name: str = None,
        env: dict[str, str] = None,
    ) -> dict[str, Any]:
        """Create a Kubernetes Job for command execution."""
        name = name or f"{self.config.name}-job-{int(time.time())}"

        job_manifest = self._generate_job(name, command, env)

        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-", "-n", self.config.namespace],
                input=yaml.dump(job_manifest),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return {"success": False, "error": result.stderr}

            logger.info(f"Job created: {name}")
            return {"success": True, "job_name": name, "namespace": self.config.namespace}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_job(
        self,
        job_name: str,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        """Wait for job completion."""
        logger.info(f"Waiting for job: {job_name}")
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            status = await self.get_job_status(job_name)

            if status:
                if status.succeeded >= status.completions:
                    logger.info(f"Job completed successfully in {time.time() - start_time:.1f}s")
                    return {
                        "success": True,
                        "status": "succeeded",
                        "duration_seconds": status.duration_seconds,
                    }
                elif status.failed > 0:
                    logger.error(f"Job failed: {job_name}")
                    return {"success": False, "status": "failed", "failed_count": status.failed}

            await asyncio.sleep(2)

        return {"success": False, "status": "timeout", "elapsed": timeout_seconds}

    async def get_job_status(self, job_name: str) -> JobStatus | None:
        """Get detailed job status."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "job", job_name, "-n", self.config.namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            job = json.loads(result.stdout)
            status = job.get("status", {})

            # Calculate duration
            duration = 0.0
            if status.get("startTime") and status.get("completionTime"):
                from datetime import datetime
                start = datetime.fromisoformat(status["startTime"].replace('Z', '+00:00'))
                end = datetime.fromisoformat(status["completionTime"].replace('Z', '+00:00'))
                duration = (end - start).total_seconds()

            return JobStatus(
                name=job_name,
                namespace=self.config.namespace,
                active=status.get("active", 0),
                succeeded=status.get("succeeded", 0),
                failed=status.get("failed", 0),
                completions=job.get("spec", {}).get("completions", 1),
                parallelism=job.get("spec", {}).get("parallelism", 1),
                duration_seconds=duration,
                conditions=[c.get("type") for c in status.get("conditions", [])],
            )

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None

    async def get_pod_logs(
        self,
        job_name: str = None,
        pod_name: str = None,
        tail_lines: int = 100,
    ) -> str:
        """Get logs from job pods."""
        if job_name:
            # Find pods for job
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", self.config.namespace,
                 "-l", f"job-name={job_name}",
                 "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            pod_name = result.stdout.strip()

        if not pod_name:
            return "No pod found"

        try:
            result = subprocess.run(
                ["kubectl", "logs", pod_name, "-n", self.config.namespace,
                 "--tail", str(tail_lines)],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr

        except Exception as e:
            return f"Failed to get logs: {e}"

    async def delete_job(self, job_name: str, cascade: bool = True) -> bool:
        """Delete a job and optionally its pods."""
        try:
            cmd = ["kubectl", "delete", "job", job_name, "-n", self.config.namespace]
            if cascade:
                cmd.append("--cascade=foreground")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to delete job: {e}")
            return False

    async def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Delete completed jobs older than specified age."""
        try:
            # List completed jobs
            result = subprocess.run(
                ["kubectl", "get", "jobs", "-n", self.config.namespace,
                 "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return 0

            jobs = json.loads(result.stdout).get("items", [])
            deleted = 0

            for job in jobs:
                status = job.get("status", {})
                if status.get("succeeded") or status.get("failed"):
                    # Check age
                    completion_time = status.get("completionTime")
                    if completion_time:
                        from datetime import datetime, timezone
                        completed = datetime.fromisoformat(completion_time.replace('Z', '+00:00'))
                        age_hours = (datetime.now(timezone.utc) - completed).total_seconds() / 3600

                        if age_hours > max_age_hours:
                            name = job["metadata"]["name"]
                            if await self.delete_job(name):
                                deleted += 1
                                logger.info(f"Deleted old job: {name}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup jobs: {e}")
            return 0

    async def scale_deployment(self, replicas: int) -> bool:
        """Scale sandbox deployment."""
        try:
            result = subprocess.run(
                ["kubectl", "scale", "deployment", self.config.name,
                 "-n", self.config.namespace, f"--replicas={replicas}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to scale: {e}")
            return False

    async def _apply_kustomize(self, path: Path) -> None:
        """Apply Kustomize manifests."""
        result = subprocess.run(
            ["kubectl", "apply", "-k", str(path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kustomize apply failed: {result.stderr}")

    def _generate_job(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] = None,
    ) -> dict[str, Any]:
        """Generate Kubernetes Job manifest."""
        limits = self.config.resources.to_k8s_limits()

        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "carts-sandbox",
                    "carts.hololoom.dev/environment": self.config.environment.value,
                    **self.config.cost_labels,
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": 3600,
                "backoffLimit": 0,
                "activeDeadlineSeconds": self.config.resources.timeout_seconds,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "carts-sandbox",
                            "job-name": name,
                        },
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "automountServiceAccountToken": False,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [{
                            "name": "sandbox",
                            "image": self.config.image,
                            "imagePullPolicy": self.config.image_pull_policy,
                            "command": command,
                            "resources": {
                                "limits": limits,
                                "requests": {
                                    "cpu": f"{int(float(limits['cpu'].rstrip('m')) / 2)}m",
                                    "memory": f"{self.config.resources.memory_mb // 2}Mi",
                                },
                            },
                            "securityContext": {
                                "readOnlyRootFilesystem": True,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "output", "mountPath": "/output"},
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                        }],
                        "volumes": [
                            {"name": "workspace", "emptyDir": {}},
                            {"name": "output", "emptyDir": {}},
                            {"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "100Mi"}},
                        ],
                    },
                },
            },
        }

        # Add environment variables
        if env or self.config.env_vars:
            all_env = {**self.config.env_vars, **(env or {})}
            job["spec"]["template"]["spec"]["containers"][0]["env"] = [
                {"name": k, "value": v} for k, v in all_env.items()
            ]

        return job
