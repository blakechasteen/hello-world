"""
Production Configuration: Environment-Specific Settings

Part 5: Production Hardening - Day 24

Provides configuration profiles for different environments:
- Development: Full logging, no limits, circuit breakers disabled
- Staging: Production-like with relaxed limits
- Production: Strict limits, monitoring, circuit breakers enabled

Features:
- Automatic environment detection
- Configuration validation
- Safe defaults
- Environment variable overrides
"""

import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Environment Types
# ============================================================================

class Environment(Enum):
    """Deployment environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_export: str = "none"  # "none", "prometheus", "json"
    metrics_port: int = 9090
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration"""
    max_retries: int = 3
    retry_backoff: float = 2.0
    retry_jitter: float = 0.1
    fallback_enabled: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    timeout: float = 10.0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = True
    global_qps: float = 100.0
    session_qps: float = 10.0
    max_concurrent: int = 50


@dataclass
class ResourceConfig:
    """Resource limits configuration"""
    max_memory_mb: int = 2048
    max_cache_size: int = 10000
    max_query_budget: int = 100


@dataclass
class LearningConfig:
    """Learning system configuration"""
    enabled: bool = True
    calibration_enabled: bool = True
    strategy_updates_enabled: bool = True


# ============================================================================
# Production Configuration
# ============================================================================

@dataclass
class ProductionConfig:
    """
    Complete production configuration

    Combines all configuration sections with environment-specific defaults
    """
    environment: Environment = Environment.DEVELOPMENT

    # Component configurations
    monitoring: MonitoringConfig = None
    error_handling: ErrorHandlingConfig = None
    circuit_breaker: CircuitBreakerConfig = None
    rate_limit: RateLimitConfig = None
    resource: ResourceConfig = None
    learning: LearningConfig = None

    def __post_init__(self):
        """Initialize sub-configs with defaults if not provided"""
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.error_handling is None:
            self.error_handling = ErrorHandlingConfig()
        if self.circuit_breaker is None:
            self.circuit_breaker = CircuitBreakerConfig()
        if self.rate_limit is None:
            self.rate_limit = RateLimitConfig()
        if self.resource is None:
            self.resource = ResourceConfig()
        if self.learning is None:
            self.learning = LearningConfig()

    @staticmethod
    def development() -> "ProductionConfig":
        """
        Development configuration

        Features:
        - Full logging (DEBUG)
        - No rate limits
        - Circuit breakers disabled
        - All learning enabled
        - Relaxed resource limits
        """
        return ProductionConfig(
            environment=Environment.DEVELOPMENT,
            monitoring=MonitoringConfig(
                enabled=True,
                metrics_export="none",
                log_level="DEBUG"
            ),
            error_handling=ErrorHandlingConfig(
                max_retries=3,
                retry_backoff=1.0,
                retry_jitter=0.0,
                fallback_enabled=True
            ),
            circuit_breaker=CircuitBreakerConfig(
                enabled=False,  # Disabled in dev
                failure_threshold=10,
                recovery_timeout=30.0
            ),
            rate_limit=RateLimitConfig(
                enabled=False,  # No limits in dev
                global_qps=1000.0,
                session_qps=100.0,
                max_concurrent=100
            ),
            resource=ResourceConfig(
                max_memory_mb=4096,  # Generous in dev
                max_cache_size=50000,
                max_query_budget=1000
            ),
            learning=LearningConfig(
                enabled=True,
                calibration_enabled=True,
                strategy_updates_enabled=True
            )
        )

    @staticmethod
    def staging() -> "ProductionConfig":
        """
        Staging configuration

        Features:
        - Production-like settings
        - Moderate logging (INFO)
        - Circuit breakers enabled
        - Relaxed rate limits
        - Prometheus metrics
        """
        return ProductionConfig(
            environment=Environment.STAGING,
            monitoring=MonitoringConfig(
                enabled=True,
                metrics_export="prometheus",
                metrics_port=9090,
                log_level="INFO"
            ),
            error_handling=ErrorHandlingConfig(
                max_retries=3,
                retry_backoff=2.0,
                retry_jitter=0.1,
                fallback_enabled=True
            ),
            circuit_breaker=CircuitBreakerConfig(
                enabled=True,
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=2,
                timeout=10.0
            ),
            rate_limit=RateLimitConfig(
                enabled=True,
                global_qps=100.0,
                session_qps=10.0,
                max_concurrent=100
            ),
            resource=ResourceConfig(
                max_memory_mb=2048,
                max_cache_size=10000,
                max_query_budget=100
            ),
            learning=LearningConfig(
                enabled=True,
                calibration_enabled=True,
                strategy_updates_enabled=True
            )
        )

    @staticmethod
    def production() -> "ProductionConfig":
        """
        Production configuration

        Features:
        - Strict limits
        - Minimal logging (WARNING)
        - Circuit breakers enabled
        - Rate limiting enabled
        - Prometheus metrics
        - Conservative resource limits
        """
        return ProductionConfig(
            environment=Environment.PRODUCTION,
            monitoring=MonitoringConfig(
                enabled=True,
                metrics_export="prometheus",
                metrics_port=9090,
                log_level="WARNING"
            ),
            error_handling=ErrorHandlingConfig(
                max_retries=5,
                retry_backoff=2.0,
                retry_jitter=0.1,
                fallback_enabled=True
            ),
            circuit_breaker=CircuitBreakerConfig(
                enabled=True,
                failure_threshold=3,
                recovery_timeout=120.0,
                success_threshold=2,
                timeout=10.0
            ),
            rate_limit=RateLimitConfig(
                enabled=True,
                global_qps=1000.0,
                session_qps=50.0,
                max_concurrent=100
            ),
            resource=ResourceConfig(
                max_memory_mb=2048,
                max_cache_size=10000,
                max_query_budget=100
            ),
            learning=LearningConfig(
                enabled=True,
                calibration_enabled=True,
                strategy_updates_enabled=True
            )
        )

    @staticmethod
    def from_environment() -> "ProductionConfig":
        """
        Detect environment and load appropriate configuration

        Checks environment variables:
        - CONTEXT_ENV: "development", "staging", or "production"

        Returns:
            ProductionConfig for detected environment
        """
        env_name = os.getenv("CONTEXT_ENV", "development").lower()

        if env_name == "production":
            return ProductionConfig.production()
        elif env_name == "staging":
            return ProductionConfig.staging()
        else:
            return ProductionConfig.development()

    @staticmethod
    def load(env_name: str) -> "ProductionConfig":
        """
        Load configuration by environment name

        Args:
            env_name: "development", "staging", or "production"

        Returns:
            ProductionConfig for specified environment

        Raises:
            ValueError: If env_name is invalid
        """
        env_name = env_name.lower()

        if env_name == "development" or env_name == "dev":
            return ProductionConfig.development()
        elif env_name == "staging":
            return ProductionConfig.staging()
        elif env_name == "production" or env_name == "prod":
            return ProductionConfig.production()
        else:
            raise ValueError(
                f"Invalid environment: {env_name}. "
                f"Must be 'development', 'staging', or 'production'"
            )

    def validate(self) -> list[str]:
        """
        Validate configuration

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Monitoring validation
        if self.monitoring.metrics_export not in ["none", "prometheus", "json"]:
            errors.append(
                f"Invalid metrics_export: {self.monitoring.metrics_export}"
            )

        if self.monitoring.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            errors.append(
                f"Invalid log_level: {self.monitoring.log_level}"
            )

        # Error handling validation
        if self.error_handling.max_retries < 0:
            errors.append(
                f"max_retries must be >= 0: {self.error_handling.max_retries}"
            )

        if self.error_handling.retry_backoff <= 0:
            errors.append(
                f"retry_backoff must be > 0: {self.error_handling.retry_backoff}"
            )

        # Circuit breaker validation
        if self.circuit_breaker.failure_threshold < 1:
            errors.append(
                f"failure_threshold must be >= 1: {self.circuit_breaker.failure_threshold}"
            )

        if self.circuit_breaker.recovery_timeout <= 0:
            errors.append(
                f"recovery_timeout must be > 0: {self.circuit_breaker.recovery_timeout}"
            )

        # Rate limit validation
        if self.rate_limit.global_qps <= 0:
            errors.append(
                f"global_qps must be > 0: {self.rate_limit.global_qps}"
            )

        if self.rate_limit.session_qps > self.rate_limit.global_qps:
            errors.append(
                f"session_qps ({self.rate_limit.session_qps}) cannot exceed "
                f"global_qps ({self.rate_limit.global_qps})"
            )

        # Resource validation
        if self.resource.max_memory_mb < 256:
            errors.append(
                f"max_memory_mb must be >= 256: {self.resource.max_memory_mb}"
            )

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate()) == 0

    def summary(self) -> str:
        """
        Get human-readable configuration summary

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Configuration Summary: {self.environment.value.upper()}")
        lines.append("=" * 80)

        lines.append("\nMonitoring:")
        lines.append(f"  Enabled: {self.monitoring.enabled}")
        lines.append(f"  Metrics Export: {self.monitoring.metrics_export}")
        lines.append(f"  Log Level: {self.monitoring.log_level}")

        lines.append("\nError Handling:")
        lines.append(f"  Max Retries: {self.error_handling.max_retries}")
        lines.append(f"  Retry Backoff: {self.error_handling.retry_backoff}x")
        lines.append(f"  Fallback: {self.error_handling.fallback_enabled}")

        lines.append("\nCircuit Breaker:")
        lines.append(f"  Enabled: {self.circuit_breaker.enabled}")
        lines.append(f"  Failure Threshold: {self.circuit_breaker.failure_threshold}")
        lines.append(f"  Recovery Timeout: {self.circuit_breaker.recovery_timeout}s")

        lines.append("\nRate Limiting:")
        lines.append(f"  Enabled: {self.rate_limit.enabled}")
        lines.append(f"  Global QPS: {self.rate_limit.global_qps}")
        lines.append(f"  Session QPS: {self.rate_limit.session_qps}")
        lines.append(f"  Max Concurrent: {self.rate_limit.max_concurrent}")

        lines.append("\nResources:")
        lines.append(f"  Max Memory: {self.resource.max_memory_mb} MB")
        lines.append(f"  Max Cache Size: {self.resource.max_cache_size}")

        lines.append("\nLearning:")
        lines.append(f"  Enabled: {self.learning.enabled}")
        lines.append(f"  Calibration: {self.learning.calibration_enabled}")
        lines.append(f"  Strategy Updates: {self.learning.strategy_updates_enabled}")

        lines.append("")
        return "\n".join(lines)


# ============================================================================
# Factory Functions
# ============================================================================

def create_config(environment: str = "development") -> ProductionConfig:
    """
    Create production configuration

    Args:
        environment: "development", "staging", or "production"

    Returns:
        ProductionConfig for specified environment
    """
    return ProductionConfig.load(environment)


def detect_environment() -> ProductionConfig:
    """
    Detect environment and create appropriate configuration

    Checks CONTEXT_ENV environment variable

    Returns:
        ProductionConfig for detected environment
    """
    return ProductionConfig.from_environment()
