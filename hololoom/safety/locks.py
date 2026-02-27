"""
Code-Level Safety Locks for Layer 6

Provides runtime enforcement of Layer 6 restrictions. Layer 6 (self-modification)
requires specialized research infrastructure and is disabled by default.

This module implements:
- Layer 6 capability detection
- Runtime blocking mechanisms
- Unlock requirements (environment variable + config flag)
- Logging of all Layer 6 attempts
- Clear error messages with guidance

Safety Philosophy:
- Fail secure (Layer 6 disabled by default)
- Explicit unlock required (multiple checks)
- Complete audit trail (log all attempts)
- Clear error messages (guide users to documentation)
"""

import os
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logger = logging.getLogger("hololoom.safety")


class Layer6Capability(Enum):
    """
    Layer 6 capabilities that require safety locks.

    These represent operations that could modify system behavior in ways
    requiring specialized research infrastructure.
    """

    # Code modification capabilities
    CODE_MODIFICATION = "code_modification"  # Modifying Python code at runtime
    MODULE_INJECTION = "module_injection"    # Adding new modules dynamically
    ARCHITECTURE_SEARCH = "architecture_search"  # Modifying neural architecture

    # Unrestricted learning capabilities
    UNRESTRICTED_LEARNING = "unrestricted_learning"  # Learning without bounds
    OBJECTIVE_MODIFICATION = "objective_modification"  # Changing goals/objectives
    CONSTRAINT_REMOVAL = "constraint_removal"  # Removing safety constraints

    # Autonomous capabilities
    AUTONOMOUS_MODIFICATION = "autonomous_modification"  # Unsupervised self-mod
    RECURSIVE_IMPROVEMENT = "recursive_improvement"  # Self-improving loops
    META_OPTIMIZATION = "meta_optimization"  # Optimizing optimization process


class Layer6BlockedException(Exception):
    """
    Raised when Layer 6 operation is attempted without proper unlock.

    Includes clear guidance on how to proceed safely.
    """

    def __init__(self, capability: Layer6Capability, message: Optional[str] = None):
        self.capability = capability
        self.timestamp = datetime.now()

        if message is None:
            message = self._default_message(capability)

        super().__init__(message)

    def _default_message(self, capability: Layer6Capability) -> str:
        """Generate helpful error message."""
        return f"""
Layer 6 Capability Blocked: {capability.value}

This operation requires Layer 6 (self-modification) to be unlocked, which
requires specialized research infrastructure.

Current Status:
- Layers 1-5: ✅ Fully available (safe for production use)
- Layer 6: 🔒 Locked (requires research environment)

To proceed:
1. Review safety documentation: README_SAFETY.md
2. Set up research environment: docs/safety/SANDBOX_CHECKLIST.md
3. Unlock Layer 6: Set environment variable HOLOLOOM_LAYER6_UNLOCK=research
4. Enable in config: Config.layer6_enabled = True

⚠️  Layer 6 requires:
- Air-gapped environment (isolated VM/network)
- Real-time monitoring (resource limits)
- Formal verification (safety invariants)
- Multi-party oversight (external review)

These are standard practices in AI safety research (Anthropic, OpenAI, DeepMind).

Questions? See README_SAFETY.md or open a GitHub Discussion.
"""


@dataclass
class Layer6Attempt:
    """Record of Layer 6 capability access attempt."""

    timestamp: datetime
    capability: Layer6Capability
    allowed: bool
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "capability": self.capability.value,
            "allowed": self.allowed,
            "context": self.context,
        }


class SafetyMonitor:
    """
    Monitors and logs Layer 6 access attempts.

    Provides audit trail for security and research purposes.
    """

    def __init__(self):
        self.attempts: List[Layer6Attempt] = []
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for safety events."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def record_attempt(
        self,
        capability: Layer6Capability,
        allowed: bool,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record Layer 6 access attempt.

        Args:
            capability: Which Layer 6 capability was attempted
            allowed: Whether access was granted
            context: Additional context (caller, reason, etc.)
        """
        attempt = Layer6Attempt(
            timestamp=datetime.now(),
            capability=capability,
            allowed=allowed,
            context=context or {},
        )

        self.attempts.append(attempt)

        # Log attempt
        if allowed:
            logger.warning(
                f"Layer 6 capability ALLOWED: {capability.value} | Context: {context}"
            )
        else:
            logger.info(
                f"Layer 6 capability BLOCKED: {capability.value} | Context: {context}"
            )

    def get_attempts(
        self,
        capability: Optional[Layer6Capability] = None,
        allowed: Optional[bool] = None
    ) -> List[Layer6Attempt]:
        """
        Get recorded attempts, optionally filtered.

        Args:
            capability: Filter by capability type
            allowed: Filter by allowed status

        Returns:
            List of matching attempts
        """
        filtered = self.attempts

        if capability is not None:
            filtered = [a for a in filtered if a.capability == capability]

        if allowed is not None:
            filtered = [a for a in filtered if a.allowed == allowed]

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of Layer 6 attempts."""
        total = len(self.attempts)
        allowed = sum(1 for a in self.attempts if a.allowed)
        blocked = total - allowed

        by_capability = {}
        for capability in Layer6Capability:
            attempts = [a for a in self.attempts if a.capability == capability]
            by_capability[capability.value] = {
                "total": len(attempts),
                "allowed": sum(1 for a in attempts if a.allowed),
                "blocked": sum(1 for a in attempts if not a.allowed),
            }

        return {
            "total_attempts": total,
            "allowed": allowed,
            "blocked": blocked,
            "by_capability": by_capability,
        }


# Global safety monitor instance
_monitor = SafetyMonitor()


class SafetyLock:
    """
    Main safety lock interface for Layer 6 capabilities.

    Provides methods to check if Layer 6 is enabled and enforce restrictions.
    """

    # Environment variable for Layer 6 unlock
    UNLOCK_ENV_VAR = "HOLOLOOM_LAYER6_UNLOCK"
    UNLOCK_VALUE = "research"

    @staticmethod
    def is_layer6_enabled(config: Optional[Any] = None) -> bool:
        """
        Check if Layer 6 is enabled.

        Requires BOTH:
        1. Environment variable: HOLOLOOM_LAYER6_UNLOCK=research
        2. Config flag: config.layer6_enabled = True

        Args:
            config: HoloLoom Config object (optional)

        Returns:
            True if Layer 6 is unlocked, False otherwise
        """
        # Check environment variable
        env_unlocked = os.environ.get(SafetyLock.UNLOCK_ENV_VAR) == SafetyLock.UNLOCK_VALUE

        # Check config flag (if provided)
        config_enabled = False
        if config is not None:
            config_enabled = getattr(config, "layer6_enabled", False)

        # Both required
        return env_unlocked and config_enabled

    @staticmethod
    def require_layer6(
        capability: Layer6Capability,
        config: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Require Layer 6 to be enabled, raise exception if not.

        Args:
            capability: Which Layer 6 capability is required
            config: HoloLoom Config object (optional)
            context: Additional context for logging

        Raises:
            Layer6BlockedException: If Layer 6 is not enabled
        """
        enabled = SafetyLock.is_layer6_enabled(config)

        # Record attempt
        _monitor.record_attempt(capability, enabled, context)

        if not enabled:
            raise Layer6BlockedException(capability)

    @staticmethod
    def check_layer6(
        capability: Layer6Capability,
        config: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if Layer 6 capability is available (non-throwing).

        Args:
            capability: Which Layer 6 capability to check
            config: HoloLoom Config object (optional)
            context: Additional context for logging

        Returns:
            True if capability is available, False otherwise
        """
        enabled = SafetyLock.is_layer6_enabled(config)

        # Record attempt
        _monitor.record_attempt(capability, enabled, context)

        return enabled

    @staticmethod
    def blocked_exception(
        capability: Layer6Capability,
        message: Optional[str] = None
    ) -> Layer6BlockedException:
        """
        Create a Layer6BlockedException for the given capability.

        Utility method for raising clear exceptions.

        Args:
            capability: Which capability was blocked
            message: Optional custom message

        Returns:
            Layer6BlockedException with helpful guidance
        """
        return Layer6BlockedException(capability, message)

    @staticmethod
    def get_monitor() -> SafetyMonitor:
        """Get global safety monitor instance."""
        return _monitor

    @staticmethod
    def get_unlock_instructions() -> str:
        """
        Get instructions for unlocking Layer 6.

        Returns:
            String with step-by-step unlock instructions
        """
        return f"""
Layer 6 Unlock Instructions

Layer 6 (self-modification) requires specialized research infrastructure.
Follow these steps to unlock safely:

Step 1: Review Safety Documentation
- README_SAFETY.md: Comprehensive safety guide
- docs/safety/SANDBOX_CHECKLIST.md: Research requirements
- LAYER_6_SAFETY_ANALYSIS.md: Technical risk analysis

Step 2: Set Up Research Environment
- Air-gapped VM (isolated network)
- Resource monitoring (CPU, memory, disk)
- Automated testing framework
- Version control with backups

Step 3: Unlock Layer 6
# Set environment variable
export {SafetyLock.UNLOCK_ENV_VAR}={SafetyLock.UNLOCK_VALUE}

# Enable in config
from hololoom.config import Config
config = Config.fast()
config.layer6_enabled = True

Step 4: Verify Unlock
from hololoom.safety import SafetyLock
if SafetyLock.is_layer6_enabled(config):
    print("Layer 6 unlocked successfully")
else:
    print("Layer 6 still locked")

⚠️  Safety Requirements:
- Use only in isolated research environment
- Monitor all system behavior
- Log all modifications
- Have rollback mechanism ready
- Follow experimental protocols (docs/safety/SANDBOX_CHECKLIST.md)

These are standard practices in AI safety research.

Questions? Open a GitHub Discussion.
"""


# Convenience functions
def is_layer6_enabled(config: Optional[Any] = None) -> bool:
    """Check if Layer 6 is enabled."""
    return SafetyLock.is_layer6_enabled(config)


def require_layer6(
    capability: Layer6Capability,
    config: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Require Layer 6 to be enabled, raise exception if not."""
    SafetyLock.require_layer6(capability, config, context)


def check_layer6(
    capability: Layer6Capability,
    config: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Check if Layer 6 capability is available (non-throwing)."""
    return SafetyLock.check_layer6(capability, config, context)
