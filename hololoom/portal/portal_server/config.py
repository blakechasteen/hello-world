"""
Portal Server configuration.

Loads from YAML file with environment variable overrides.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PortalConfig(BaseModel):
    """Configuration for the Portal Server."""

    loom_id: str = Field(default="loom-default", description="Loom identifier")
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8080, ge=1, le=65535, description="Bind port")
    shared_secret: str = Field(
        default="dev-secret-change-me",
        description="Shared secret for node authentication"
    )
    node_timeout_seconds: int = Field(
        default=90,
        ge=30,
        description="Mark node offline after N seconds without heartbeat"
    )
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=True, description="Use JSON logging")


def load_config(config_path: str | None = None) -> PortalConfig:
    """
    Load Portal configuration from YAML file with env overrides.

    Priority: Environment variables > Config file > Defaults

    Args:
        config_path: Path to YAML config file

    Returns:
        PortalConfig instance
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
        "PORTAL_LOOM_ID": "loom_id",
        "PORTAL_HOST": "host",
        "PORTAL_PORT": "port",
        "PORTAL_SHARED_SECRET": "shared_secret",
        "PORTAL_NODE_TIMEOUT": "node_timeout_seconds",
        "PORTAL_LOG_LEVEL": "log_level",
        "PORTAL_LOG_JSON": "log_json",
    }

    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Handle type conversions
            if config_key == "port" or config_key == "node_timeout_seconds":
                value = int(value)
            elif config_key == "log_json":
                value = value.lower() in ("true", "1", "yes")
            config_data[config_key] = value

    return PortalConfig(**config_data)
