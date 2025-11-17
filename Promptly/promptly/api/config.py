"""
API Configuration
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """API Settings with environment variable support"""

    # API Info
    app_name: str = "Promptly API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Security
    secret_key: str = "change-this-in-production-use-env-var"
    api_key_header: str = "X-API-Key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 100

    # Storage
    promptly_root_dir: Optional[str] = None
    storage_backend: str = "sqlite"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    # WebSocket
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 100

    class Config:
        env_file = ".env"
        env_prefix = "PROMPTLY_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
