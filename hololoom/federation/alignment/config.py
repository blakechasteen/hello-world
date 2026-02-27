"""
Federation Alignment Configuration - Configurable parameters for alignment verification.

Provides sensible defaults with the ability to customize for different
deployment scenarios (development, staging, production).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from .consensus_engine import ConsensusStrategy
from .node_profile import (
    DECEPTION_PENALTY,
    INITIAL_TRUST_SCORE,
    RESOURCE_VIOLATION_PENALTY,
    TRUST_THRESHOLD,
    VERIFIED_RESPONSE_BONUS,
)


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS - Configuration modes
# ═══════════════════════════════════════════════════════════════════════════


class AlignmentMode(Enum):
    """Operating mode for alignment verification."""

    DISABLED = auto()      # No alignment verification (development only)
    PERMISSIVE = auto()    # Log but don't block (testing)
    STANDARD = auto()      # Normal verification (production)
    STRICT = auto()        # Enhanced verification (high-security)


# ═══════════════════════════════════════════════════════════════════════════
#  LAYER CONFIGS - Per-layer verification settings
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class L1Config:
    """Configuration for L1 (Local) verification layer."""

    enabled: bool = True
    target_latency_ms: float = 10.0    # <10ms target
    max_latency_ms: float = 50.0       # Hard limit
    skip_for_trusted: bool = False     # Skip L1 for highly trusted nodes


@dataclass(frozen=True)
class L2Config:
    """Configuration for L2 (Response) verification layer."""

    enabled: bool = True
    target_latency_ms: float = 100.0   # 10-100ms target
    max_latency_ms: float = 500.0      # Hard limit
    alignment_threshold: float = 0.8   # Minimum confidence to pass
    deception_threshold: float = 0.5   # Maximum deception score
    require_convergence_check: bool = True


@dataclass(frozen=True)
class L3Config:
    """Configuration for L3 (Consensus) verification layer."""

    enabled: bool = True
    target_latency_ms: float = 500.0   # 100-500ms target
    max_latency_ms: float = 2000.0     # Hard limit (2 seconds)
    min_verifiers: int = 3             # Minimum nodes required
    max_verifiers: int = 7             # Maximum nodes to query
    consensus_strategy: ConsensusStrategy = ConsensusStrategy.BYZANTINE_TOLERANT
    quorum_ratio: float = 0.67         # 2/3 majority
    timeout_ms: float = 1000.0         # Per-node timeout


# ═══════════════════════════════════════════════════════════════════════════
#  TRUST CONFIG - Trust score evolution parameters
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrustConfig:
    """Configuration for trust score management."""

    initial_score: float = INITIAL_TRUST_SCORE
    min_score: float = 0.0
    max_score: float = 1.0
    trust_threshold: float = TRUST_THRESHOLD

    # Score adjustments
    verified_bonus: float = VERIFIED_RESPONSE_BONUS
    deception_penalty: float = DECEPTION_PENALTY
    resource_violation_penalty: float = RESOURCE_VIOLATION_PENALTY
    consensus_agree_bonus: float = 0.005
    consensus_dissent_penalty: float = -0.02
    escalation_penalty: float = -0.05
    cleared_bonus: float = 0.02

    # Decay settings (trust decays over time without activity)
    enable_decay: bool = False
    decay_rate_per_day: float = 0.001  # 0.1% per day
    decay_floor: float = 0.7           # Don't decay below this


# ═══════════════════════════════════════════════════════════════════════════
#  DECEPTION CONFIG - Deception detection parameters
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DeceptionConfig:
    """Configuration for deception detection."""

    enabled: bool = True
    max_score: float = 0.5             # Threshold for blocking

    # Per-flag thresholds
    consistency_threshold: float = 0.7  # Allow some variance
    goal_drift_threshold: float = 0.8   # Strict on goal alignment
    capability_hiding_threshold: float = 0.6
    reward_hacking_threshold: float = 0.5
    resource_seeking_threshold: float = 0.5

    # Probe settings
    probe_frequency: float = 0.1       # Probe 10% of responses
    probe_timeout_ms: float = 500.0


# ═══════════════════════════════════════════════════════════════════════════
#  RESOURCE CONFIG - Resource monitoring parameters
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ResourceConfig:
    """Configuration for resource usage monitoring."""

    enabled: bool = True

    # Anomaly thresholds
    max_cpu_ms: float = 10000.0        # 10 seconds CPU time
    max_memory_mb: float = 1024.0      # 1GB memory
    max_api_calls: int = 100           # External API calls
    max_tokens_out: int = 10000        # Output tokens
    max_tool_calls: int = 50           # Tool invocations
    max_memory_writes: int = 100       # Knowledge graph writes


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CONFIG - Complete federation alignment configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FederationAlignmentConfig:
    """
    Complete configuration for federation alignment verification.

    Combines all sub-configurations into a single, cohesive settings object.
    Use factory methods for common presets.
    """

    mode: AlignmentMode = AlignmentMode.STANDARD

    # Layer configurations
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)

    # Component configurations
    trust: TrustConfig = field(default_factory=TrustConfig)
    deception: DeceptionConfig = field(default_factory=DeceptionConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Global settings
    enable_audit_trail: bool = True
    audit_retention_days: int = 90
    enable_metrics: bool = True
    metrics_export_interval_seconds: float = 60.0

    # Signature settings
    require_signatures: bool = True
    signature_algorithm: str = "ed25519"

    @property
    def is_enabled(self) -> bool:
        """Check if alignment is enabled."""
        return self.mode != AlignmentMode.DISABLED

    @property
    def is_strict(self) -> bool:
        """Check if strict mode is enabled."""
        return self.mode == AlignmentMode.STRICT

    @property
    def is_permissive(self) -> bool:
        """Check if permissive (logging only) mode."""
        return self.mode == AlignmentMode.PERMISSIVE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mode": self.mode.name,
            "l1": {
                "enabled": self.l1.enabled,
                "target_latency_ms": self.l1.target_latency_ms,
                "max_latency_ms": self.l1.max_latency_ms,
                "skip_for_trusted": self.l1.skip_for_trusted,
            },
            "l2": {
                "enabled": self.l2.enabled,
                "target_latency_ms": self.l2.target_latency_ms,
                "max_latency_ms": self.l2.max_latency_ms,
                "alignment_threshold": self.l2.alignment_threshold,
                "deception_threshold": self.l2.deception_threshold,
            },
            "l3": {
                "enabled": self.l3.enabled,
                "target_latency_ms": self.l3.target_latency_ms,
                "min_verifiers": self.l3.min_verifiers,
                "max_verifiers": self.l3.max_verifiers,
                "consensus_strategy": self.l3.consensus_strategy.name,
            },
            "trust": {
                "initial_score": self.trust.initial_score,
                "trust_threshold": self.trust.trust_threshold,
                "enable_decay": self.trust.enable_decay,
            },
            "deception": {
                "enabled": self.deception.enabled,
                "max_score": self.deception.max_score,
            },
            "resources": {
                "enabled": self.resources.enabled,
                "max_cpu_ms": self.resources.max_cpu_ms,
                "max_memory_mb": self.resources.max_memory_mb,
            },
            "global": {
                "enable_audit_trail": self.enable_audit_trail,
                "enable_metrics": self.enable_metrics,
                "require_signatures": self.require_signatures,
            },
        }

    # ───────────────────────────────────────────────────────────────────────
    #  Factory Methods - Common presets
    # ───────────────────────────────────────────────────────────────────────

    @classmethod
    def disabled(cls) -> "FederationAlignmentConfig":
        """Create disabled configuration (development only)."""
        return cls(
            mode=AlignmentMode.DISABLED,
            l1=L1Config(enabled=False),
            l2=L2Config(enabled=False),
            l3=L3Config(enabled=False),
            deception=DeceptionConfig(enabled=False),
            resources=ResourceConfig(enabled=False),
            enable_audit_trail=False,
            enable_metrics=False,
            require_signatures=False,
        )

    @classmethod
    def permissive(cls) -> "FederationAlignmentConfig":
        """Create permissive configuration (testing/staging)."""
        return cls(
            mode=AlignmentMode.PERMISSIVE,
            l1=L1Config(enabled=True),
            l2=L2Config(
                enabled=True,
                alignment_threshold=0.5,  # Lower threshold
                deception_threshold=0.7,   # Higher tolerance
            ),
            l3=L3Config(enabled=False),    # Skip consensus in permissive
            enable_audit_trail=True,       # Still log everything
            require_signatures=False,       # Don't require signatures
        )

    @classmethod
    def standard(cls) -> "FederationAlignmentConfig":
        """Create standard configuration (production default)."""
        return cls(
            mode=AlignmentMode.STANDARD,
            l1=L1Config(enabled=True),
            l2=L2Config(enabled=True),
            l3=L3Config(enabled=True),
            enable_audit_trail=True,
            enable_metrics=True,
            require_signatures=True,
        )

    @classmethod
    def strict(cls) -> "FederationAlignmentConfig":
        """Create strict configuration (high-security environments)."""
        return cls(
            mode=AlignmentMode.STRICT,
            l1=L1Config(
                enabled=True,
                skip_for_trusted=False,    # Verify everyone
            ),
            l2=L2Config(
                enabled=True,
                alignment_threshold=0.9,   # Higher threshold
                deception_threshold=0.3,   # Lower tolerance
                require_convergence_check=True,
            ),
            l3=L3Config(
                enabled=True,
                min_verifiers=5,           # More verifiers
                consensus_strategy=ConsensusStrategy.SUPERMAJORITY,
                quorum_ratio=0.8,          # 80% agreement
            ),
            trust=TrustConfig(
                trust_threshold=0.7,       # Higher threshold
                deception_penalty=-0.15,   # Harsher penalties
                enable_decay=True,         # Enable trust decay
            ),
            deception=DeceptionConfig(
                enabled=True,
                max_score=0.3,             # Lower tolerance
                probe_frequency=0.25,      # More frequent probing
            ),
            enable_audit_trail=True,
            audit_retention_days=365,      # Longer retention
            enable_metrics=True,
            require_signatures=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def create_alignment_config(
    mode: str = "standard",
    **overrides: Any
) -> FederationAlignmentConfig:
    """
    Create alignment configuration with optional overrides.

    Args:
        mode: Configuration preset ("disabled", "permissive", "standard", "strict")
        **overrides: Individual setting overrides

    Returns:
        FederationAlignmentConfig instance
    """
    presets = {
        "disabled": FederationAlignmentConfig.disabled,
        "permissive": FederationAlignmentConfig.permissive,
        "standard": FederationAlignmentConfig.standard,
        "strict": FederationAlignmentConfig.strict,
    }

    factory = presets.get(mode.lower(), FederationAlignmentConfig.standard)
    config = factory()

    # Apply overrides (only top-level for simplicity)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
