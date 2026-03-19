"""
Enhanced Docker Deployer for CARTS Sandbox
===========================================

Extended Docker deployment capabilities including Docker Compose
integration, image building, and health monitoring.

Author: CARTS Team
Date: 2025-12-09
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DeploymentConfig

logger = logging.getLogger("carts-deploy.docker")


@dataclass
class DockerHealth:
    """Container health status."""
    container_id: str
    name: str
    running: bool
    state: str
    health: str
    cpu_percent: float
    memory_mb: float
    memory_limit_mb: float
    network_rx_bytes: int
    network_tx_bytes: int
    pids: int
    uptime_seconds: float


class EnhancedDockerDeployer:
    """Enhanced Docker deployer with Compose support and monitoring."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.compose_file: Path | None = None

    async def build_image(
        self,
        dockerfile: Path = None,
        context: Path = None,
        tag: str = None,
        build_args: dict[str, str] = None,
    ) -> dict[str, Any]:
        """Build sandbox Docker image."""
        dockerfile = dockerfile or Path(__file__).parent.parent.parent.parent / "infra" / "docker" / "Dockerfile.sandbox"
        context = context or Path(__file__).parent.parent.parent.parent
        tag = tag or self.config.image

        logger.info(f"Building image: {tag}")

        cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile)]

        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

        cmd.append(str(context))

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for builds
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info(f"Image built in {elapsed:.1f}s: {tag}")
            return {
                "success": True,
                "tag": tag,
                "build_time_seconds": elapsed,
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Build timed out after 10 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def compose_up(
        self,
        compose_file: Path = None,
        env_vars: dict[str, str] = None,
        detach: bool = True,
    ) -> dict[str, Any]:
        """Start sandbox using Docker Compose."""
        compose_file = compose_file or self._get_default_compose_file()
        self.compose_file = compose_file

        logger.info(f"Starting with Compose: {compose_file}")

        cmd = ["docker-compose", "-f", str(compose_file)]

        # Add project name based on config
        project_name = f"carts-{self.config.environment.value}"
        cmd.extend(["-p", project_name])

        cmd.append("up")
        if detach:
            cmd.append("-d")

        # Prepare environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        # Add config-based env vars
        env.update({
            "SANDBOX_CPU_LIMIT": str(self.config.resources.cpu_cores),
            "SANDBOX_MEMORY_LIMIT": f"{self.config.resources.memory_mb}m",
            "SANDBOX_TIMEOUT": str(self.config.resources.timeout_seconds),
            "SANDBOX_NETWORK_ENABLED": "true" if self.config.network.enabled else "false",
        })

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=env
            )

            if result.returncode != 0:
                logger.error(f"Compose up failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info(f"Compose started: {project_name}")
            return {
                "success": True,
                "project": project_name,
                "compose_file": str(compose_file),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def compose_down(
        self,
        compose_file: Path = None,
        volumes: bool = False,
    ) -> dict[str, Any]:
        """Stop and remove Compose services."""
        compose_file = compose_file or self.compose_file or self._get_default_compose_file()

        logger.info(f"Stopping Compose: {compose_file}")

        project_name = f"carts-{self.config.environment.value}"
        cmd = ["docker-compose", "-f", str(compose_file), "-p", project_name, "down"]

        if volumes:
            cmd.append("-v")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                logger.error(f"Compose down failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

            logger.info("Compose stopped")
            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_container_health(self, container_name: str = None) -> DockerHealth | None:
        """Get detailed container health information."""
        container_name = container_name or self.config.name

        try:
            # Get container stats
            result = subprocess.run(
                ["docker", "stats", container_name, "--no-stream", "--format",
                 "{{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.PIDs}}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            stats_line = result.stdout.strip()
            if not stats_line:
                return None

            parts = stats_line.split(",")

            # Parse CPU percentage
            cpu_percent = float(parts[0].rstrip('%')) if parts[0] else 0.0

            # Parse memory usage
            mem_parts = parts[1].split('/') if len(parts) > 1 else ['0MiB', '0MiB']
            mem_used = self._parse_mem_value(mem_parts[0].strip())
            mem_limit = self._parse_mem_value(mem_parts[1].strip()) if len(mem_parts) > 1 else 0

            # Parse network I/O
            net_parts = parts[2].split('/') if len(parts) > 2 else ['0B', '0B']
            net_rx = self._parse_size_value(net_parts[0].strip())
            net_tx = self._parse_size_value(net_parts[1].strip()) if len(net_parts) > 1 else 0

            # Parse PIDs
            pids = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0

            # Get container state and health
            result = subprocess.run(
                ["docker", "inspect", "--format",
                 '{{.Id}},{{.State.Status}},{{.State.Health.Status}},{{.State.StartedAt}}',
                 container_name],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return None

            inspect_parts = result.stdout.strip().split(",")
            container_id = inspect_parts[0][:12] if inspect_parts else ""
            state = inspect_parts[1] if len(inspect_parts) > 1 else "unknown"
            health = inspect_parts[2] if len(inspect_parts) > 2 else "none"

            # Calculate uptime
            uptime = 0.0
            if len(inspect_parts) > 3:
                try:
                    from datetime import datetime
                    started = datetime.fromisoformat(inspect_parts[3].replace('Z', '+00:00'))
                    uptime = (datetime.now(started.tzinfo) - started).total_seconds()
                except Exception:
                    pass

            return DockerHealth(
                container_id=container_id,
                name=container_name,
                running=state == "running",
                state=state,
                health=health if health else "none",
                cpu_percent=cpu_percent,
                memory_mb=mem_used,
                memory_limit_mb=mem_limit,
                network_rx_bytes=net_rx,
                network_tx_bytes=net_tx,
                pids=pids,
                uptime_seconds=uptime,
            )

        except Exception as e:
            logger.error(f"Failed to get container health: {e}")
            return None

    async def wait_for_healthy(
        self,
        container_name: str = None,
        timeout_seconds: int = 60,
        poll_interval: float = 2.0,
    ) -> bool:
        """Wait for container to become healthy."""
        container_name = container_name or self.config.name
        start_time = time.time()

        logger.info(f"Waiting for {container_name} to become healthy...")

        while time.time() - start_time < timeout_seconds:
            health = await self.get_container_health(container_name)

            if health and health.running:
                if health.health in ("healthy", "none"):  # 'none' means no healthcheck defined
                    logger.info(f"Container healthy after {time.time() - start_time:.1f}s")
                    return True

            await asyncio.sleep(poll_interval)

        logger.error(f"Container did not become healthy within {timeout_seconds}s")
        return False

    async def run_with_network_isolation(
        self,
        command: list[str],
        allowed_hosts: list[str] = None,
    ) -> dict[str, Any]:
        """Run command with network isolation."""
        # Create isolated network if needed
        network_name = f"carts-isolated-{self.config.name}"

        if not self.config.network.enabled:
            # Use 'none' network for complete isolation
            return await self._run_in_container(command, network="none")

        # Create network with specific egress rules
        try:
            # Create network
            subprocess.run(
                ["docker", "network", "create", "--driver", "bridge", network_name],
                capture_output=True,
                timeout=30
            )

            # Run container with network
            result = await self._run_in_container(command, network=network_name)

            # Cleanup network
            subprocess.run(
                ["docker", "network", "rm", network_name],
                capture_output=True,
                timeout=30
            )

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_in_container(
        self,
        command: list[str],
        network: str = "none",
    ) -> dict[str, Any]:
        """Run a command in a new container."""
        container_name = f"{self.config.name}-exec-{int(time.time())}"

        cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            "--network", network,
        ]

        # Add resource limits
        limits = self.config.resources.to_docker_limits()
        cmd.extend(["--cpu-quota", str(limits["cpu_quota"])])
        cmd.extend(["--cpu-period", str(limits["cpu_period"])])
        cmd.extend(["--memory", limits["mem_limit"]])
        cmd.extend(["--pids-limit", str(limits["pids_limit"])])

        # Security options
        cmd.extend([
            "--read-only",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
        ])

        cmd.append(self.config.image)
        cmd.extend(command)

        try:
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

        except subprocess.TimeoutExpired:
            # Kill container on timeout
            subprocess.run(["docker", "kill", container_name], capture_output=True)
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_default_compose_file(self) -> Path:
        """Get default compose file path."""
        infra_dir = Path(__file__).parent.parent.parent.parent / "infra" / "docker"
        return infra_dir / "docker-compose-sandbox.yml"

    def _parse_mem_value(self, value: str) -> float:
        """Parse memory value string to MB."""
        value = value.strip()
        multipliers = {
            'B': 1 / (1024 * 1024),
            'KiB': 1 / 1024, 'KB': 1 / 1024,
            'MiB': 1, 'MB': 1,
            'GiB': 1024, 'GB': 1024,
        }
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                return float(value[:-len(suffix)].strip()) * mult
        return 0.0

    def _parse_size_value(self, value: str) -> int:
        """Parse size value string to bytes."""
        value = value.strip()
        multipliers = {
            'B': 1,
            'kB': 1000, 'KB': 1000,
            'MB': 1000000, 'GB': 1000000000,
            'KiB': 1024, 'MiB': 1048576, 'GiB': 1073741824,
        }
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                return int(float(value[:-len(suffix)].strip()) * mult)
        return 0


def generate_compose_file(config: DeploymentConfig) -> str:
    """Generate Docker Compose YAML for sandbox deployment."""
    limits = config.resources.to_docker_limits()

    compose = f"""# CARTS Sandbox Docker Compose
# Generated for environment: {config.environment.value}
# Auto-generated - do not edit manually

services:
  sandbox:
    image: {config.image}
    container_name: {config.name}
    deploy:
      resources:
        limits:
          cpus: '{config.resources.cpu_cores}'
          memory: {config.resources.memory_mb}M
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    networks:
      - {'carts-isolated' if not config.network.enabled else 'carts-sandbox'}
    volumes:
      - sandbox_workspace:/workspace
      - sandbox_output:/output:rw
    environment:
      - SANDBOX_MODE=isolated
      - TIMEOUT_SECONDS={config.resources.timeout_seconds}
"""

    # Add environment variables
    for key, value in config.env_vars.items():
        compose += f"      - {key}={value}\n"

    # Add healthcheck
    compose += """    healthcheck:
      test: ["CMD", "true"]
      interval: 10s
      timeout: 5s
      retries: 3
"""

    # Add metrics port if enabled
    if config.enable_metrics:
        compose += f"""    ports:
      - "{config.metrics_port}:{config.metrics_port}"
"""

    # Networks and volumes
    compose += """
networks:
  carts-sandbox:
    driver: bridge
  carts-isolated:
    driver: bridge
    internal: true  # No external access

volumes:
  sandbox_workspace:
    driver: local
  sandbox_output:
    driver: local
"""

    return compose
