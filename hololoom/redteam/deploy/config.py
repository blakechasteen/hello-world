"""
Deployment Configuration for CARTS Sandbox Deployer
====================================================

Configuration classes for deploying CARTS sandboxes to Docker,
Kubernetes, and cloud environments.

Author: CARTS Team
Date: 2025-12-09
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeploymentTarget(Enum):
    """Target deployment environment."""

    LOCAL = "local"          # Local Docker/Docker Compose
    DOCKER = "docker"        # Remote Docker host
    KUBERNETES = "kubernetes"  # Kubernetes cluster
    AWS = "aws"              # AWS (ECS/Fargate)
    GCP = "gcp"              # Google Cloud Run
    AZURE = "azure"          # Azure Container Instances


class DeploymentEnvironment(Enum):
    """Deployment environment/stage."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox deployment."""

    cpu_cores: float = 1.0       # CPU cores (e.g., 0.5, 1.0, 2.0)
    memory_mb: int = 512         # Memory in MB
    disk_mb: int = 1024          # Disk space in MB
    timeout_seconds: int = 300   # Max execution time
    max_processes: int = 10      # Max spawned processes

    def to_docker_limits(self) -> dict[str, Any]:
        """Convert to Docker resource limits."""
        return {
            "cpu_quota": int(self.cpu_cores * 100000),
            "cpu_period": 100000,
            "mem_limit": f"{self.memory_mb}m",
            "pids_limit": self.max_processes,
        }

    def to_k8s_limits(self) -> dict[str, str]:
        """Convert to Kubernetes resource limits."""
        return {
            "cpu": f"{int(self.cpu_cores * 1000)}m",
            "memory": f"{self.memory_mb}Mi",
            "ephemeral-storage": f"{self.disk_mb}Mi",
        }


@dataclass
class NetworkConfig:
    """Network configuration for sandbox."""

    enabled: bool = False                      # Enable network access
    egress_allowed: bool = False               # Allow outbound connections
    allowed_hosts: list[str] = field(default_factory=list)
    allowed_ports: list[int] = field(default_factory=list)
    dns_enabled: bool = True                   # Allow DNS resolution

    def to_network_policy(self) -> dict[str, Any]:
        """Convert to Kubernetes NetworkPolicy spec."""
        if not self.enabled:
            return {"podSelector": {}, "policyTypes": ["Egress", "Ingress"]}

        egress_rules = []
        if self.dns_enabled:
            egress_rules.append({
                "ports": [{"port": 53, "protocol": "UDP"}]
            })

        for host in self.allowed_hosts:
            for port in self.allowed_ports or [443]:
                egress_rules.append({
                    "to": [{"ipBlock": {"cidr": f"{host}/32"}}] if host else [],
                    "ports": [{"port": port}]
                })

        return {
            "podSelector": {},
            "policyTypes": ["Egress"],
            "egress": egress_rules
        }


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""

    # Target and environment
    target: DeploymentTarget = DeploymentTarget.LOCAL
    environment: DeploymentEnvironment = DeploymentEnvironment.DEV

    # Naming
    name: str = "carts-sandbox"
    namespace: str = "carts"

    # Container image
    image: str = "hololoom/carts-sandbox:latest"
    image_pull_policy: str = "IfNotPresent"

    # Resources
    resources: ResourceLimits = field(default_factory=ResourceLimits)

    # Network
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # Replicas (for K8s)
    replicas: int = 1

    # Secrets and environment
    env_vars: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)  # Secret names to mount

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Cost tracking
    enable_cost_tracking: bool = True
    cost_labels: dict[str, str] = field(default_factory=dict)

    # Paths
    working_dir: str = "/workspace"
    output_dir: str = "/output"

    @classmethod
    def for_dev(cls, name: str = "carts-sandbox-dev") -> "DeploymentConfig":
        """Create development configuration."""
        return cls(
            name=name,
            target=DeploymentTarget.LOCAL,
            environment=DeploymentEnvironment.DEV,
            resources=ResourceLimits(cpu_cores=0.5, memory_mb=256),
            enable_metrics=False,
            enable_cost_tracking=False,
        )

    @classmethod
    def for_staging(cls, name: str = "carts-sandbox-staging") -> "DeploymentConfig":
        """Create staging configuration."""
        return cls(
            name=name,
            target=DeploymentTarget.KUBERNETES,
            environment=DeploymentEnvironment.STAGING,
            resources=ResourceLimits(cpu_cores=1.0, memory_mb=512),
            replicas=2,
            enable_metrics=True,
        )

    @classmethod
    def for_prod(cls, name: str = "carts-sandbox-prod") -> "DeploymentConfig":
        """Create production configuration."""
        return cls(
            name=name,
            target=DeploymentTarget.KUBERNETES,
            environment=DeploymentEnvironment.PROD,
            resources=ResourceLimits(cpu_cores=2.0, memory_mb=1024),
            replicas=3,
            enable_metrics=True,
            enable_cost_tracking=True,
            image_pull_policy="Always",
        )

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of warnings."""
        warnings = []

        if self.environment == DeploymentEnvironment.PROD:
            if self.resources.memory_mb < 512:
                warnings.append("Production memory limit < 512MB may cause OOM")
            if self.replicas < 2:
                warnings.append("Production should have replicas >= 2 for HA")
            if not self.enable_metrics:
                warnings.append("Metrics disabled in production")

        if self.network.egress_allowed and not self.network.allowed_hosts:
            warnings.append("Egress allowed but no hosts specified - all traffic allowed!")

        return warnings
