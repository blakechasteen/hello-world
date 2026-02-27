#!/usr/bin/env python3
"""
CARTS Sandbox Deployer CLI
==========================

One-command deployment system for CARTS sandbox with Docker, Kubernetes,
and cloud environment support.

Commands:
  carts-deploy setup     - Initialize deployment environment
  carts-deploy start     - Launch sandbox
  carts-deploy execute   - Run attack in sandbox
  carts-deploy status    - Check health and status
  carts-deploy stop      - Graceful shutdown
  carts-deploy destroy   - Cleanup all resources
  carts-deploy logs      - View sandbox logs

Usage:
  python -m HoloLoom.redteam.deploy.cli setup --target docker
  python -m HoloLoom.redteam.deploy.cli start --env dev
  python -m HoloLoom.redteam.deploy.cli execute --campaign campaign.yaml
  python -m HoloLoom.redteam.deploy.cli status
  python -m HoloLoom.redteam.deploy.cli stop
  python -m HoloLoom.redteam.deploy.cli destroy --force

Author: CARTS Team
Date: 2025-12-09
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import (
    DeploymentConfig,
    DeploymentTarget,
    DeploymentEnvironment,
    ResourceLimits,
    NetworkConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("carts-deploy")


# =============================================================================
# Deployment State Management
# =============================================================================

class DeploymentState:
    """Manages deployment state persistence."""

    def __init__(self, state_dir: Path = None):
        self.state_dir = state_dir or Path.home() / ".carts" / "deploy"
        self.state_file = self.state_dir / "state.json"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure state directory exists."""
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        """Load current state."""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"deployments": {}, "active": None}

    def save(self, state: Dict[str, Any]) -> None:
        """Save state to file."""
        self.state_file.write_text(json.dumps(state, indent=2))

    def get_active(self) -> Optional[Dict[str, Any]]:
        """Get active deployment info."""
        state = self.load()
        active_name = state.get("active")
        if active_name:
            return state["deployments"].get(active_name)
        return None

    def set_active(self, name: str, info: Dict[str, Any]) -> None:
        """Set active deployment."""
        state = self.load()
        state["deployments"][name] = info
        state["active"] = name
        self.save(state)

    def clear_active(self) -> None:
        """Clear active deployment."""
        state = self.load()
        if state.get("active"):
            active_name = state["active"]
            if active_name in state["deployments"]:
                del state["deployments"][active_name]
            state["active"] = None
        self.save(state)


# =============================================================================
# Deployer Interface
# =============================================================================

class BaseDeployer:
    """Base class for deployment backends."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.state = DeploymentState()

    async def setup(self) -> bool:
        """Setup deployment environment. Returns True on success."""
        raise NotImplementedError

    async def start(self) -> Dict[str, Any]:
        """Start sandbox deployment. Returns deployment info."""
        raise NotImplementedError

    async def execute(self, command: List[str], env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute command in sandbox. Returns result."""
        raise NotImplementedError

    async def status(self) -> Dict[str, Any]:
        """Get deployment status."""
        raise NotImplementedError

    async def stop(self) -> bool:
        """Stop deployment gracefully. Returns True on success."""
        raise NotImplementedError

    async def destroy(self, force: bool = False) -> bool:
        """Destroy deployment and cleanup. Returns True on success."""
        raise NotImplementedError

    async def logs(self, tail: int = 100, follow: bool = False) -> str:
        """Get deployment logs."""
        raise NotImplementedError


class DockerDeployer(BaseDeployer):
    """Docker/Docker Compose deployment backend."""

    async def setup(self) -> bool:
        """Verify Docker is available and pull required images."""
        import subprocess

        logger.info("Setting up Docker deployment environment...")

        # Check Docker availability
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error("Docker not available or not running")
                return False
            logger.info(f"Docker version: {result.stdout.strip()}")
        except Exception as e:
            logger.error(f"Failed to check Docker: {e}")
            return False

        # Pull sandbox image
        logger.info(f"Pulling image: {self.config.image}")
        try:
            result = subprocess.run(
                ["docker", "pull", self.config.image],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"Failed to pull image (may use local): {result.stderr}")
        except Exception as e:
            logger.warning(f"Image pull failed (may use local): {e}")

        logger.info("Docker setup complete")
        return True

    async def start(self) -> Dict[str, Any]:
        """Start Docker container for sandbox."""
        import subprocess

        logger.info(f"Starting sandbox container: {self.config.name}")

        # Build docker run command
        cmd = ["docker", "run", "-d", "--name", self.config.name]

        # Resource limits
        limits = self.config.resources.to_docker_limits()
        cmd.extend(["--cpu-quota", str(limits["cpu_quota"])])
        cmd.extend(["--cpu-period", str(limits["cpu_period"])])
        cmd.extend(["--memory", limits["mem_limit"]])
        cmd.extend(["--pids-limit", str(limits["pids_limit"])])

        # Network
        if not self.config.network.enabled:
            cmd.extend(["--network", "none"])

        # Environment variables
        for key, value in self.config.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Working directory
        cmd.extend(["-w", self.config.working_dir])

        # Image
        cmd.append(self.config.image)

        # Keep container running
        cmd.extend(["tail", "-f", "/dev/null"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Failed to start container: {result.stderr}")
                return {"success": False, "error": result.stderr}

            container_id = result.stdout.strip()[:12]
            info = {
                "success": True,
                "container_id": container_id,
                "name": self.config.name,
                "target": self.config.target.value,
                "started_at": time.time(),
            }

            self.state.set_active(self.config.name, info)
            logger.info(f"Container started: {container_id}")
            return info

        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return {"success": False, "error": str(e)}

    async def execute(self, command: List[str], env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute command in running container."""
        import subprocess

        active = self.state.get_active()
        if not active:
            return {"success": False, "error": "No active deployment"}

        container_name = active["name"]
        logger.info(f"Executing in {container_name}: {' '.join(command)}")

        cmd = ["docker", "exec"]

        # Add environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(container_name)
        cmd.extend(command)

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.resources.timeout_seconds
            )
            elapsed_ms = (time.time() - start_time) * 1000

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": elapsed_ms,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def status(self) -> Dict[str, Any]:
        """Get container status."""
        import subprocess

        active = self.state.get_active()
        if not active:
            return {"running": False, "message": "No active deployment"}

        container_name = active["name"]

        try:
            result = subprocess.run(
                ["docker", "inspect", "--format",
                 '{"state":"{{.State.Status}}","health":"{{.State.Health.Status}}"}',
                 container_name],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {"running": False, "message": "Container not found"}

            status_str = result.stdout.strip()
            # Parse status (handle empty health)
            if '""' in status_str:
                status_str = status_str.replace('"health":""', '"health":"none"')

            import json as json_lib
            status_info = json_lib.loads(status_str)

            return {
                "running": status_info["state"] == "running",
                "state": status_info["state"],
                "health": status_info["health"],
                "name": container_name,
                "deployment_info": active,
            }
        except Exception as e:
            return {"running": False, "error": str(e)}

    async def stop(self) -> bool:
        """Stop container gracefully."""
        import subprocess

        active = self.state.get_active()
        if not active:
            logger.warning("No active deployment to stop")
            return True

        container_name = active["name"]
        logger.info(f"Stopping container: {container_name}")

        try:
            result = subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info("Container stopped")
                return True
            else:
                logger.error(f"Failed to stop: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    async def destroy(self, force: bool = False) -> bool:
        """Remove container and cleanup."""
        import subprocess

        active = self.state.get_active()
        if not active:
            logger.warning("No active deployment to destroy")
            return True

        container_name = active["name"]
        logger.info(f"Destroying container: {container_name}")

        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(container_name)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.state.clear_active()
                logger.info("Container destroyed")
                return True
            else:
                logger.error(f"Failed to destroy: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to destroy container: {e}")
            return False

    async def logs(self, tail: int = 100, follow: bool = False) -> str:
        """Get container logs."""
        import subprocess

        active = self.state.get_active()
        if not active:
            return "No active deployment"

        container_name = active["name"]
        cmd = ["docker", "logs", "--tail", str(tail)]
        if follow:
            cmd.append("-f")
        cmd.append(container_name)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {e}"


class KubernetesDeployer(BaseDeployer):
    """Kubernetes deployment backend."""

    async def setup(self) -> bool:
        """Verify kubectl is configured and namespace exists."""
        import subprocess

        logger.info("Setting up Kubernetes deployment environment...")

        # Check kubectl
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error("kubectl not available")
                return False
            logger.info("kubectl available")
        except Exception as e:
            logger.error(f"Failed to check kubectl: {e}")
            return False

        # Create namespace if needed
        namespace = self.config.namespace
        logger.info(f"Ensuring namespace: {namespace}")

        try:
            result = subprocess.run(
                ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"],
                capture_output=True,
                text=True
            )
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=result.stdout,
                capture_output=True,
                text=True
            )
            logger.info(f"Namespace {namespace} ready")
        except Exception as e:
            logger.warning(f"Namespace creation: {e}")

        logger.info("Kubernetes setup complete")
        return True

    async def start(self) -> Dict[str, Any]:
        """Deploy sandbox to Kubernetes."""
        import subprocess
        import yaml

        logger.info(f"Deploying to Kubernetes: {self.config.name}")

        # Generate Job manifest
        job_manifest = self._generate_job_manifest()

        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-", "-n", self.config.namespace],
                input=yaml.dump(job_manifest),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"Failed to deploy: {result.stderr}")
                return {"success": False, "error": result.stderr}

            info = {
                "success": True,
                "job_name": self.config.name,
                "namespace": self.config.namespace,
                "target": self.config.target.value,
                "started_at": time.time(),
            }

            self.state.set_active(self.config.name, info)
            logger.info(f"Job deployed: {self.config.name}")
            return info

        except Exception as e:
            logger.error(f"Failed to deploy: {e}")
            return {"success": False, "error": str(e)}

    def _generate_job_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Job manifest."""
        limits = self.config.resources.to_k8s_limits()

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.config.name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "carts-sandbox",
                    "environment": self.config.environment.value,
                    **self.config.cost_labels,
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": 3600,
                "backoffLimit": 0,
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "sandbox",
                            "image": self.config.image,
                            "imagePullPolicy": self.config.image_pull_policy,
                            "resources": {
                                "limits": limits,
                                "requests": {
                                    "cpu": f"{int(float(limits['cpu'].rstrip('m')) / 2)}m",
                                    "memory": f"{self.config.resources.memory_mb // 2}Mi",
                                },
                            },
                            "securityContext": {
                                "runAsNonRoot": True,
                                "readOnlyRootFilesystem": True,
                                "allowPrivilegeEscalation": False,
                            },
                            "env": [
                                {"name": k, "value": v}
                                for k, v in self.config.env_vars.items()
                            ],
                        }],
                    },
                },
            },
        }

    async def execute(self, command: List[str], env: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute command in K8s pod."""
        import subprocess

        active = self.state.get_active()
        if not active:
            return {"success": False, "error": "No active deployment"}

        # Find pod for job
        job_name = active["job_name"]
        namespace = active["namespace"]

        try:
            # Get pod name
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace,
                 "-l", f"job-name={job_name}",
                 "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0 or not result.stdout.strip():
                return {"success": False, "error": "Pod not found"}

            pod_name = result.stdout.strip()

            # Execute command
            cmd = ["kubectl", "exec", pod_name, "-n", namespace, "--"]
            cmd.extend(command)

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.resources.timeout_seconds
            )
            elapsed_ms = (time.time() - start_time) * 1000

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": elapsed_ms,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def status(self) -> Dict[str, Any]:
        """Get K8s job status."""
        import subprocess

        active = self.state.get_active()
        if not active:
            return {"running": False, "message": "No active deployment"}

        job_name = active["job_name"]
        namespace = active["namespace"]

        try:
            result = subprocess.run(
                ["kubectl", "get", "job", job_name, "-n", namespace,
                 "-o", "jsonpath={.status.active},{.status.succeeded},{.status.failed}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {"running": False, "message": "Job not found"}

            parts = result.stdout.strip().split(",")
            active_count = int(parts[0]) if parts[0] else 0
            succeeded = int(parts[1]) if parts[1] else 0
            failed = int(parts[2]) if parts[2] else 0

            return {
                "running": active_count > 0,
                "active": active_count,
                "succeeded": succeeded,
                "failed": failed,
                "name": job_name,
                "namespace": namespace,
            }

        except Exception as e:
            return {"running": False, "error": str(e)}

    async def stop(self) -> bool:
        """Scale job to 0 (stop processing)."""
        # Jobs don't have replicas, so we just wait for completion
        logger.info("Kubernetes jobs complete on their own - use destroy to remove")
        return True

    async def destroy(self, force: bool = False) -> bool:
        """Delete job and pods."""
        import subprocess

        active = self.state.get_active()
        if not active:
            logger.warning("No active deployment to destroy")
            return True

        job_name = active["job_name"]
        namespace = active["namespace"]

        logger.info(f"Deleting job: {job_name}")

        try:
            result = subprocess.run(
                ["kubectl", "delete", "job", job_name, "-n", namespace,
                 "--grace-period=0" if force else "--grace-period=30"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.state.clear_active()
                logger.info("Job deleted")
                return True
            else:
                logger.error(f"Failed to delete: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete job: {e}")
            return False

    async def logs(self, tail: int = 100, follow: bool = False) -> str:
        """Get pod logs."""
        import subprocess

        active = self.state.get_active()
        if not active:
            return "No active deployment"

        job_name = active["job_name"]
        namespace = active["namespace"]

        try:
            # Get pod name
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace,
                 "-l", f"job-name={job_name}",
                 "-o", "jsonpath={.items[0].metadata.name}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if not result.stdout.strip():
                return "Pod not found"

            pod_name = result.stdout.strip()

            cmd = ["kubectl", "logs", pod_name, "-n", namespace, "--tail", str(tail)]
            if follow:
                cmd.append("-f")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout + result.stderr

        except Exception as e:
            return f"Failed to get logs: {e}"


# =============================================================================
# Factory
# =============================================================================

def create_deployer(config: DeploymentConfig) -> BaseDeployer:
    """Create appropriate deployer based on target."""
    if config.target in (DeploymentTarget.LOCAL, DeploymentTarget.DOCKER):
        return DockerDeployer(config)
    elif config.target == DeploymentTarget.KUBERNETES:
        return KubernetesDeployer(config)
    else:
        raise ValueError(f"Unsupported deployment target: {config.target}")


# =============================================================================
# CLI Commands
# =============================================================================

async def cmd_setup(args: argparse.Namespace) -> int:
    """Handle setup command."""
    target = DeploymentTarget(args.target)

    if args.env == "dev":
        config = DeploymentConfig.for_dev()
    elif args.env == "staging":
        config = DeploymentConfig.for_staging()
    elif args.env == "prod":
        config = DeploymentConfig.for_prod()
    else:
        config = DeploymentConfig()

    config.target = target

    deployer = create_deployer(config)
    success = await deployer.setup()
    return 0 if success else 1


async def cmd_start(args: argparse.Namespace) -> int:
    """Handle start command."""
    if args.env == "dev":
        config = DeploymentConfig.for_dev(args.name)
    elif args.env == "staging":
        config = DeploymentConfig.for_staging(args.name)
    elif args.env == "prod":
        config = DeploymentConfig.for_prod(args.name)
    else:
        config = DeploymentConfig(name=args.name)

    if args.target:
        config.target = DeploymentTarget(args.target)

    if args.image:
        config.image = args.image

    deployer = create_deployer(config)
    result = await deployer.start()

    if result.get("success"):
        print(json.dumps(result, indent=2))
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        return 1


async def cmd_execute(args: argparse.Namespace) -> int:
    """Handle execute command."""
    config = DeploymentConfig()
    deployer = create_deployer(config)

    command = args.command if isinstance(args.command, list) else [args.command]
    result = await deployer.execute(command)

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


async def cmd_status(args: argparse.Namespace) -> int:
    """Handle status command."""
    config = DeploymentConfig()
    deployer = create_deployer(config)

    status = await deployer.status()
    print(json.dumps(status, indent=2))
    return 0


async def cmd_stop(args: argparse.Namespace) -> int:
    """Handle stop command."""
    config = DeploymentConfig()
    deployer = create_deployer(config)

    success = await deployer.stop()
    return 0 if success else 1


async def cmd_destroy(args: argparse.Namespace) -> int:
    """Handle destroy command."""
    config = DeploymentConfig()
    deployer = create_deployer(config)

    success = await deployer.destroy(force=args.force)
    return 0 if success else 1


async def cmd_logs(args: argparse.Namespace) -> int:
    """Handle logs command."""
    config = DeploymentConfig()
    deployer = create_deployer(config)

    logs = await deployer.logs(tail=args.tail, follow=args.follow)
    print(logs)
    return 0


# =============================================================================
# Main
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="carts-deploy",
        description="CARTS Sandbox Deployer - One-command deployment system",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Setup
    setup_parser = subparsers.add_parser("setup", help="Initialize deployment environment")
    setup_parser.add_argument(
        "--target", "-t",
        choices=["local", "docker", "kubernetes"],
        default="docker",
        help="Deployment target"
    )
    setup_parser.add_argument(
        "--env", "-e",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment"
    )

    # Start
    start_parser = subparsers.add_parser("start", help="Launch sandbox")
    start_parser.add_argument("--name", "-n", default="carts-sandbox", help="Deployment name")
    start_parser.add_argument(
        "--target", "-t",
        choices=["local", "docker", "kubernetes"],
        help="Deployment target (overrides env default)"
    )
    start_parser.add_argument(
        "--env", "-e",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment"
    )
    start_parser.add_argument("--image", "-i", help="Container image to use")

    # Execute
    exec_parser = subparsers.add_parser("execute", help="Run command in sandbox")
    exec_parser.add_argument("command", nargs="+", help="Command to execute")

    # Status
    subparsers.add_parser("status", help="Check deployment status")

    # Stop
    subparsers.add_parser("stop", help="Graceful shutdown")

    # Destroy
    destroy_parser = subparsers.add_parser("destroy", help="Cleanup resources")
    destroy_parser.add_argument("--force", "-f", action="store_true", help="Force removal")

    # Logs
    logs_parser = subparsers.add_parser("logs", help="View sandbox logs")
    logs_parser.add_argument("--tail", "-n", type=int, default=100, help="Lines to show")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Map commands to handlers
    handlers = {
        "setup": cmd_setup,
        "start": cmd_start,
        "execute": cmd_execute,
        "status": cmd_status,
        "stop": cmd_stop,
        "destroy": cmd_destroy,
        "logs": cmd_logs,
    }

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    # Run async handler
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
