"""
Node Daemon configuration.

Loads from YAML file with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


class NodeConfig(BaseModel):
    """Configuration for the Node Daemon."""

    node_id: str = Field(description="Unique node identifier")
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=9090, ge=1, le=65535, description="Bind port")
    portal_url: str = Field(
        default="http://localhost:8080",
        description="Portal Server URL"
    )
    wasm_module_dir: str = Field(
        default="./wasm_modules",
        description="Directory containing WASM modules"
    )
    shared_secret: str = Field(
        default="dev-secret-change-me",
        description="Shared secret for Portal authentication"
    )
    heartbeat_interval_seconds: int = Field(
        default=30,
        ge=10,
        description="Heartbeat interval to Portal"
    )
    max_concurrent_jobs: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent job executions"
    )
    job_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Default job timeout"
    )
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=True, description="Use JSON logging")


def load_config(config_path: Optional[str] = None) -> NodeConfig:
    """
    Load Node configuration from YAML file with env overrides.

    Priority: Environment variables > Config file > Defaults

    Args:
        config_path: Path to YAML config file

    Returns:
        NodeConfig instance
    """
    config_data = {}

    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                config_data = yaml.safe_load(f) or {}

    # Environment variable overrides
    env_mapping = {
        "NODE_ID": "node_id",
        "NODE_HOST": "host",
        "NODE_PORT": "port",
        "NODE_PORTAL_URL": "portal_url",
        "NODE_WASM_DIR": "wasm_module_dir",
        "NODE_SHARED_SECRET": "shared_secret",
        "NODE_HEARTBEAT_INTERVAL": "heartbeat_interval_seconds",
        "NODE_MAX_JOBS": "max_concurrent_jobs",
        "NODE_JOB_TIMEOUT": "job_timeout_seconds",
        "NODE_LOG_LEVEL": "log_level",
        "NODE_LOG_JSON": "log_json",
    }

    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Handle type conversions
            if config_key in ("port", "heartbeat_interval_seconds",
                              "max_concurrent_jobs", "job_timeout_seconds"):
                value = int(value)
            elif config_key == "log_json":
                value = value.lower() in ("true", "1", "yes")
            config_data[config_key] = value

    # Generate default node_id if not provided
    if "node_id" not in config_data:
        import socket
        config_data["node_id"] = f"node-{socket.gethostname()}"

    return NodeConfig(**config_data)