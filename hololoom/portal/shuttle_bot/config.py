"""
Shuttle Bot configuration.

Loads from YAML file with environment variable overrides.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ShuttleConfig(BaseModel):
    """Configuration for the Shuttle Bot."""

    # Matrix configuration
    homeserver: str = Field(
        default="https://your-homeserver.local",
        description="Matrix homeserver URL"
    )
    user_id: str = Field(
        default="@shuttle:your-homeserver.local",
        description="Bot Matrix user ID"
    )
    password: str = Field(
        default="bot-password-change-me",
        description="Bot Matrix password"
    )
    room_id: str = Field(
        default="!roomid:your-homeserver.local",
        description="Matrix room ID to join"
    )

    # Portal configuration
    portal_url: str = Field(
        default="http://localhost:8080",
        description="Portal Server URL"
    )
    shared_secret: str = Field(
        default="dev-secret-change-me",
        description="Shared secret for Portal authentication"
    )

    # Bot behavior
    command_prefix: str = Field(
        default="!",
        description="Command prefix (e.g., '!' for !loom, !nodes)"
    )
    job_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Default job timeout"
    )
    max_message_length: int = Field(
        default=4000,
        description="Maximum message length to send"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_json: bool = Field(default=True, description="Use JSON logging")


def load_config(config_path: str | None = None) -> ShuttleConfig:
    """
    Load Shuttle configuration from YAML file with env overrides.

    Priority: Environment variables > Config file > Defaults

    Args:
        config_path: Path to YAML config file

    Returns:
        ShuttleConfig instance
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
        "SHUTTLE_HOMESERVER": "homeserver",
        "SHUTTLE_USER_ID": "user_id",
        "SHUTTLE_PASSWORD": "password",
        "SHUTTLE_ROOM_ID": "room_id",
        "SHUTTLE_PORTAL_URL": "portal_url",
        "SHUTTLE_SHARED_SECRET": "shared_secret",
        "SHUTTLE_COMMAND_PREFIX": "command_prefix",
        "SHUTTLE_JOB_TIMEOUT": "job_timeout_seconds",
        "SHUTTLE_LOG_LEVEL": "log_level",
        "SHUTTLE_LOG_JSON": "log_json",
    }

    for env_var, config_key in env_mapping.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            # Handle type conversions
            if config_key == "job_timeout_seconds":
                value = int(value)
            elif config_key == "log_json":
                value = value.lower() in ("true", "1", "yes")
            config_data[config_key] = value

    return ShuttleConfig(**config_data)
