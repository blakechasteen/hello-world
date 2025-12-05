"""HoloLoom Bridge Configuration.

Manages bridge settings for HoloLoom server connection,
timeouts, retries, and local fallback behavior.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class BridgeConfig(BaseModel):
    """Configuration for HoloLoom Bridge.

    Handles connection settings, timeouts, retries, and fallback behavior.
    """

    hololoom_url: str = Field(
        default="http://localhost:8000",
        description="HoloLoom server URL",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts for failed requests",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0,
        description="Delay between retries in seconds",
    )
    enable_local_fallback: bool = Field(
        default=True,
        description="Use local HoloLoom if server unavailable",
    )
    default_recall_k: int = Field(
        default=5,
        ge=1,
        description="Default number of results for recall operations",
    )
    default_weave_mode: str = Field(
        default="fast",
        description="Default reasoning mode (bare/fast/fused)",
    )

    class Config:
        """Pydantic config."""
        validate_assignment = True


def load_bridge_config(path: Optional[str] = None) -> BridgeConfig:
    """Load bridge configuration from YAML or environment variables.

    Args:
        path: Optional path to YAML config file.
              If not provided, uses default location or env var.

    Returns:
        BridgeConfig: Loaded configuration with environment overrides applied.
    """
    config_data = {}

    # Try loading from YAML file
    if path is None:
        # Check environment variable for config path
        path = os.getenv("HOLOLOOM_BRIDGE_CONFIG")

        # Fall back to default location
        if path is None:
            default_path = Path(__file__).parent.parent / "configs" / "bridge.yaml"
            if default_path.exists():
                path = str(default_path)

    if path and Path(path).exists():
        with open(path) as f:
            file_config = yaml.safe_load(f) or {}
            config_data.update(file_config)

    # Apply environment variable overrides
    env_overrides = {
        "hololoom_url": os.getenv("HOLOLOOM_URL"),
        "timeout_seconds": os.getenv("HOLOLOOM_TIMEOUT"),
        "retry_count": os.getenv("HOLOLOOM_RETRY_COUNT"),
        "enable_local_fallback": os.getenv("HOLOLOOM_LOCAL_FALLBACK"),
        "default_recall_k": os.getenv("HOLOLOOM_RECALL_K"),
        "default_weave_mode": os.getenv("HOLOLOOM_WEAVE_MODE"),
    }

    # Filter out None values and convert types
    for key, value in env_overrides.items():
        if value is not None:
            if key == "timeout_seconds":
                config_data[key] = float(value)
            elif key == "retry_count":
                config_data[key] = int(value)
            elif key == "default_recall_k":
                config_data[key] = int(value)
            elif key == "enable_local_fallback":
                config_data[key] = value.lower() in ("true", "1", "yes")
            else:
                config_data[key] = value

    return BridgeConfig(**config_data)
