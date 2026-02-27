"""Proto configuration module.

Configuration classes for Proto engine with support for different execution
modes, personalities, and capability settings.

Status: Production ready (2025-12-01)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ProtoMode(Enum):
    """Proto execution modes.

    Modes:
        INTERACTIVE: CLI/TUI REPL mode (default)
        SINGLE: Single query mode
        CHATOPS: Matrix bot integration mode
        SERVER: FastAPI server mode
    """
    INTERACTIVE = "interactive"
    SINGLE = "single"
    CHATOPS = "chatops"
    SERVER = "server"


class PersonalityStyle(Enum):
    """Proto personality styles.

    Styles affect response tone and verbosity:
        RELAXED: Casual, friendly tone
        PROFESSIONAL: Formal, structured tone
        MINIMAL: Terse, direct responses
        TECHNICAL: Detailed, expert tone
    """
    RELAXED = "relaxed"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    TECHNICAL = "technical"


class TrustLevel(Enum):
    """Trust levels for ability execution.

    Trust levels determine which abilities can be executed:
        CORE: Only core built-in abilities
        VERIFIED: Core + verified third-party abilities
        COMMUNITY: Verified + community-contributed abilities
        LOCAL: All local abilities
        UNTRUSTED: Sandbox mode, safe operations only
    """
    CORE = "core"
    VERIFIED = "verified"
    COMMUNITY = "community"
    LOCAL = "local"
    UNTRUSTED = "untrusted"


@dataclass
class ProtoConfig:
    """Configuration for Proto engine.

    Attributes:
        mode: Execution mode (interactive, single, chatops, server)
        personality_style: Response style and tone
        verbosity: Output verbosity (0=minimal, 1=normal, 2=verbose)
        enable_code_execution: Allow code execution (safety-critical)
        enable_git_operations: Allow git operations
        enable_file_operations: Allow file system operations
        enable_agentic_reasoning: Use HoloLoom agentic system
        enable_memory: Use HoloLoom memory system
        reasoning_mode: HoloLoom reasoning mode (direct, verify, research, plan_execute)
        abilities_path: Path to abilities directory (~/.proto/abilities/)
        enable_sideloading: Allow loading custom abilities
        max_response_tokens: Maximum response length
        timeout_seconds: Operation timeout in seconds
        trust_level: Ability trust level (core, verified, community, local, untrusted)
        session_timeout_seconds: Session inactivity timeout
        enable_logging: Enable detailed logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """

    # Mode
    mode: ProtoMode = ProtoMode.INTERACTIVE

    # Personality
    personality_style: PersonalityStyle = PersonalityStyle.RELAXED
    verbosity: int = 1  # 0=minimal, 1=normal, 2=verbose

    # Capabilities
    enable_code_execution: bool = False  # Safety: disabled by default
    enable_git_operations: bool = True
    enable_file_operations: bool = True
    enable_shell_operations: bool = False  # Safety: disabled by default

    # HoloLoom integration
    enable_agentic_reasoning: bool = True
    enable_memory: bool = True
    reasoning_mode: str = "verify"  # direct, verify, research, plan_execute

    # Abilities
    abilities_path: Optional[str] = None  # ~/.proto/abilities/
    enable_sideloading: bool = True

    # Limits
    max_response_tokens: int = 4000
    timeout_seconds: float = 60.0
    session_timeout_seconds: float = 3600.0  # 1 hour

    # Trust
    trust_level: TrustLevel = TrustLevel.LOCAL

    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"

    # Internal tracking
    _version: str = "1.0"
    _created_at_timestamp: float = field(default_factory=lambda: __import__('time').time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "mode": self.mode.value,
            "personality_style": self.personality_style.value,
            "verbosity": self.verbosity,
            "enable_code_execution": self.enable_code_execution,
            "enable_git_operations": self.enable_git_operations,
            "enable_file_operations": self.enable_file_operations,
            "enable_shell_operations": self.enable_shell_operations,
            "enable_agentic_reasoning": self.enable_agentic_reasoning,
            "enable_memory": self.enable_memory,
            "reasoning_mode": self.reasoning_mode,
            "abilities_path": self.abilities_path,
            "enable_sideloading": self.enable_sideloading,
            "max_response_tokens": self.max_response_tokens,
            "timeout_seconds": self.timeout_seconds,
            "session_timeout_seconds": self.session_timeout_seconds,
            "trust_level": self.trust_level.value,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
        }

    @classmethod
    def default(cls) -> 'ProtoConfig':
        """Create default configuration."""
        return cls()

    @classmethod
    def minimal(cls) -> 'ProtoConfig':
        """Create minimal configuration (disabled advanced features)."""
        return cls(
            verbosity=0,
            enable_agentic_reasoning=False,
            enable_memory=False,
            personality_style=PersonalityStyle.MINIMAL,
        )

    @classmethod
    def full(cls) -> 'ProtoConfig':
        """Create full configuration (enable all safe features)."""
        return cls(
            verbosity=2,
            enable_code_execution=False,  # Still disabled for safety
            enable_git_operations=True,
            enable_file_operations=True,
            enable_agentic_reasoning=True,
            enable_memory=True,
            reasoning_mode="research",
            personality_style=PersonalityStyle.TECHNICAL,
            trust_level=TrustLevel.VERIFIED,
        )

    @classmethod
    def sandbox(cls) -> 'ProtoConfig':
        """Create sandbox configuration (most restrictive)."""
        return cls(
            verbosity=1,
            enable_code_execution=False,
            enable_git_operations=False,
            enable_file_operations=False,
            enable_shell_operations=False,
            enable_agentic_reasoning=False,
            enable_memory=False,
            enable_sideloading=False,
            personality_style=PersonalityStyle.MINIMAL,
            trust_level=TrustLevel.UNTRUSTED,
        )

    def is_safe(self) -> bool:
        """Check if configuration is safe (no dangerous operations enabled)."""
        dangerous_enabled = (
            self.enable_code_execution or
            self.enable_shell_operations
        )
        return not dangerous_enabled

    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings."""
        warnings = []

        if self.enable_code_execution and self.trust_level == TrustLevel.UNTRUSTED:
            warnings.append("Code execution enabled with UNTRUSTED trust level")

        if self.enable_shell_operations and not self.enable_code_execution:
            warnings.append("Shell operations enabled but code execution disabled")

        if self.max_response_tokens > 16000:
            warnings.append("Very large response token limit may exceed LLM context")

        if self.timeout_seconds < 5.0:
            warnings.append("Timeout is very short, may cause operation failures")

        return warnings
