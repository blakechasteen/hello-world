"""
Elle Production Configuration

Environment-based configuration for Elle AR guide system.

Features:
- Feature flags for production deployment
- Environment variable support
- Graceful defaults for development

Usage:
    from elle.config import ElleConfig

    config = ElleConfig()
    if config.enable_prompt_refinement:
        # Use refined prompts

Created: 2025-11-22
Part of: Option 2 - Automation (Week 1)
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ElleConfig:
    """Elle production configuration"""

    # ============================
    # Prompt Refinement Settings
    # ============================
    enable_prompt_refinement: bool = False
    """Enable HoloLoom prompt refinement (+30% AR response quality)"""

    refinement_provider: str = "anthropic"
    """LLM provider for refinement ("anthropic", "google", "openai")"""

    # ============================
    # Performance Settings
    # ============================
    enable_prompt_caching: bool = True
    """Enable MD5-based prompt caching (eliminates overhead)"""

    # ============================
    # Logging Settings
    # ============================
    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""

    log_refinement_stats: bool = True
    """Log refinement statistics (latency, expansion ratio, cache hits)"""

    @classmethod
    def from_env(cls) -> 'ElleConfig':
        """
        Load configuration from environment variables

        Environment Variables:
            ELLE_ENABLE_REFINEMENT: "true" to enable refinement (default: false)
            ELLE_REFINEMENT_PROVIDER: LLM provider (default: anthropic)
            ELLE_ENABLE_CACHING: "true" to enable caching (default: true)
            ELLE_LOG_LEVEL: Logging level (default: INFO)
            ELLE_LOG_REFINEMENT_STATS: "true" to log stats (default: true)

        Returns:
            ElleConfig instance with values from environment

        Example:
            # Enable refinement in production
            export ELLE_ENABLE_REFINEMENT=true
            export ELLE_REFINEMENT_PROVIDER=anthropic

            # Load config
            from elle.config import ElleConfig
            config = ElleConfig.from_env()
        """

        return cls(
            enable_prompt_refinement=_env_bool('ELLE_ENABLE_REFINEMENT', False),
            refinement_provider=os.getenv('ELLE_REFINEMENT_PROVIDER', 'anthropic'),
            enable_prompt_caching=_env_bool('ELLE_ENABLE_CACHING', True),
            log_level=os.getenv('ELLE_LOG_LEVEL', 'INFO'),
            log_refinement_stats=_env_bool('ELLE_LOG_REFINEMENT_STATS', True),
        )

    @classmethod
    def production(cls) -> 'ElleConfig':
        """
        Production configuration with refinement enabled

        Returns:
            ElleConfig with production settings

        Example:
            from elle.config import ElleConfig
            config = ElleConfig.production()
        """
        return cls(
            enable_prompt_refinement=True,
            refinement_provider='anthropic',
            enable_prompt_caching=True,
            log_level='INFO',
            log_refinement_stats=True,
        )

    @classmethod
    def development(cls) -> 'ElleConfig':
        """
        Development configuration with refinement disabled

        Returns:
            ElleConfig with development settings (faster, no LLM calls)

        Example:
            from elle.config import ElleConfig
            config = ElleConfig.development()
        """
        return cls(
            enable_prompt_refinement=False,
            refinement_provider='anthropic',
            enable_prompt_caching=True,
            log_level='DEBUG',
            log_refinement_stats=True,
        )


def _env_bool(key: str, default: bool) -> bool:
    """
    Parse boolean from environment variable

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default

    return value.lower() in ('true', '1', 'yes', 'on')


# Default configuration instance (development mode)
default_config = ElleConfig.development()


if __name__ == "__main__":
    print("Elle Configuration")
    print("=" * 70)
    print()

    # Show default config
    print("Default (Development) Configuration:")
    config = ElleConfig.development()
    print(f"  Enable Refinement: {config.enable_prompt_refinement}")
    print(f"  Refinement Provider: {config.refinement_provider}")
    print(f"  Enable Caching: {config.enable_prompt_caching}")
    print(f"  Log Level: {config.log_level}")
    print()

    # Show production config
    print("Production Configuration:")
    config = ElleConfig.production()
    print(f"  Enable Refinement: {config.enable_prompt_refinement}")
    print(f"  Refinement Provider: {config.refinement_provider}")
    print(f"  Enable Caching: {config.enable_prompt_caching}")
    print(f"  Log Level: {config.log_level}")
    print()

    # Show environment-based config
    print("Environment-Based Configuration:")
    print("Set environment variables to customize:")
    print("  export ELLE_ENABLE_REFINEMENT=true")
    print("  export ELLE_REFINEMENT_PROVIDER=anthropic")
    print()
    config = ElleConfig.from_env()
    print(f"  Current: Enable Refinement = {config.enable_prompt_refinement}")
    print(f"  Current: Refinement Provider = {config.refinement_provider}")
    print()

    print("Usage:")
    print("  from elle.config import ElleConfig")
    print("  config = ElleConfig.production()  # or .development() or .from_env()")
    print("  builder = PromptBuilder(enable_refinement=config.enable_prompt_refinement)")
