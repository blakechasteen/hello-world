"""
Jenny Configuration Base Class
==============================
Shared configuration patterns for Jenny components.

Date: December 2025
Status: Phase 2 Elegance Enhancement (Task 2.4)

Philosophy:
> "Configuration should be discoverable, serializable, and predictable."

Provides:
- Generic to_dict() / from_dict() serialization
- Standard fast() / reliable() factory methods
- Type-safe dataclass inheritance

Usage:
    from hololoom.visualization.jenny_config_base import BaseConfig

    @dataclass
    class MyServiceConfig(BaseConfig):
        host: str = "localhost"
        port: int = 8080
        timeout_seconds: float = 30.0

        @classmethod
        def fast(cls) -> 'MyServiceConfig':
            return cls(timeout_seconds=5.0)

        @classmethod
        def reliable(cls) -> 'MyServiceConfig':
            return cls(timeout_seconds=60.0)

    # Serialize
    config = MyServiceConfig.fast()
    data = config.to_dict()  # {'host': 'localhost', 'port': 8080, ...}

    # Deserialize
    restored = MyServiceConfig.from_dict(data)
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

# Self type for Python 3.10 compatibility
T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig(ABC):
    """
    Abstract base configuration class.

    Provides:
    - to_dict(): Serialize config to dictionary
    - from_dict(): Deserialize config from dictionary
    - fast(): Factory for fast/low-latency settings (subclass must implement)
    - reliable(): Factory for reliable/robust settings (subclass must implement)

    Subclasses should:
    1. Be decorated with @dataclass
    2. Define their configuration fields with defaults
    3. Implement fast() and reliable() class methods

    Example:
        @dataclass
        class LLMClientConfig(BaseConfig):
            timeout: float = 30.0
            retries: int = 3

            @classmethod
            def fast(cls) -> 'LLMClientConfig':
                return cls(timeout=5.0, retries=1)

            @classmethod
            def reliable(cls) -> 'LLMClientConfig':
                return cls(timeout=60.0, retries=5)
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize configuration to dictionary.

        Returns:
            Dictionary with all config fields and values.
            Nested dataclasses are recursively converted.

        Example:
            config = MyConfig(host="localhost", port=8080)
            data = config.to_dict()
            # {'host': 'localhost', 'port': 8080}
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary with config fields

        Returns:
            New config instance

        Example:
            data = {'host': 'localhost', 'port': 8080}
            config = MyConfig.from_dict(data)

        Note:
            Only fields defined in the dataclass are used.
            Extra fields in data are ignored (allows forward compatibility).
        """
        # Get valid field names for this dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter data to only valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    @classmethod
    @abstractmethod
    def fast(cls: type[T]) -> T:
        """
        Create configuration optimized for speed/low-latency.

        Should prioritize:
        - Lower timeouts
        - Fewer retries
        - Smaller batch sizes
        - Less thorough validation

        Returns:
            Config instance with fast settings
        """
        ...

    @classmethod
    @abstractmethod
    def reliable(cls: type[T]) -> T:
        """
        Create configuration optimized for reliability/robustness.

        Should prioritize:
        - Higher timeouts
        - More retries
        - Thorough validation
        - Safe defaults

        Returns:
            Config instance with reliable settings
        """
        ...


@dataclass
class BaseConfigWithDefaults(BaseConfig):
    """
    Base config with default fast/reliable implementations.

    Use when fast/reliable don't need customization
    (just returns default instance).

    Subclasses can still override fast() and reliable()
    if needed.
    """

    @classmethod
    def fast(cls: type[T]) -> T:
        """Default fast config (returns default instance)."""
        return cls()

    @classmethod
    def reliable(cls: type[T]) -> T:
        """Default reliable config (returns default instance)."""
        return cls()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'BaseConfig',
    'BaseConfigWithDefaults',
]
