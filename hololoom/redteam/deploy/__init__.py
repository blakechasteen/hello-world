"""
CARTS Sandbox Deployer
======================

One-command deployment system for CARTS sandbox with Docker,
Kubernetes, and cloud environment support.

Usage:
    # CLI
    python -m HoloLoom.redteam.deploy.cli setup --target docker
    python -m HoloLoom.redteam.deploy.cli start --env dev
    python -m HoloLoom.redteam.deploy.cli execute python script.py
    python -m HoloLoom.redteam.deploy.cli status
    python -m HoloLoom.redteam.deploy.cli stop
    python -m HoloLoom.redteam.deploy.cli destroy

    # Programmatic
    from hololoom.redteam.deploy import DeploymentConfig, create_deployer

    config = DeploymentConfig.for_dev()
    deployer = create_deployer(config)
    await deployer.setup()
    await deployer.start()

Author: CARTS Team
Date: 2025-12-09
"""

from .cli import (
    BaseDeployer,
    DeploymentState,
    DockerDeployer,
    KubernetesDeployer,
    create_deployer,
)
from .config import (
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentTarget,
    NetworkConfig,
    ResourceLimits,
)

__all__ = [
    # Config
    "DeploymentConfig",
    "DeploymentTarget",
    "DeploymentEnvironment",
    "ResourceLimits",
    "NetworkConfig",
    # Deployers
    "BaseDeployer",
    "DockerDeployer",
    "KubernetesDeployer",
    "DeploymentState",
    "create_deployer",
]
