"""
Judgment Types
==============
Core data types for the Conscience system.

Voice levels represent graduated response - not binary allow/deny,
but a spectrum of concern that guides action.

Judgment captures the result of consideration - what Voice spoke,
why it spoke, and what wisdom it offers.

Wisdom captures learned patterns - what has worked, what hasn't,
and how to improve.

Philosophy:
    "A conscience that speaks, not shouts. That guides, not gates."

Author: HoloLoom Team
Date: 2025-12-03
"""

from enum import IntEnum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


class Voice(IntEnum):
    """
    Graduated response levels.

    Unlike binary allow/deny, Voice represents a spectrum:
    - QUIET: Safe, proceed silently
    - WHISPER: Minor concern, proceed with logging
    - VOICE: Significant concern, human review recommended
    - SHOUT: Critical concern, block and escalate

    The numeric values enable comparison:
        if judgment.voice <= Voice.WHISPER:
            proceed()

    Design Principle:
        Most interactions should be QUIET or WHISPER.
        VOICE and SHOUT should be rare exceptions.
    """
    QUIET = 0      # Safe - proceed automatically, log only
    WHISPER = 1    # Minor concern - proceed with monitoring
    VOICE = 2      # Significant concern - recommend human review
    SHOUT = 3      # Critical concern - block and escalate

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def should_block(self) -> bool:
        """Whether this voice level should block the action."""
        return self >= Voice.SHOUT

    @property
    def requires_human(self) -> bool:
        """Whether this voice level recommends human review."""
        return self >= Voice.VOICE

    @property
    def emoji(self) -> str:
        """Visual representation of voice level."""
        return {
            Voice.QUIET: "🔇",
            Voice.WHISPER: "🔈",
            Voice.VOICE: "🔊",
            Voice.SHOUT: "📢",
        }[self]


@dataclass
class Concern:
    """
    A specific concern raised during consideration.

    Concerns are the building blocks of Judgment - each lens
    may raise zero or more concerns about an action.

    Attributes:
        lens: Which lens raised this concern
        category: Type of concern (e.g., "harm", "deception", "power")
        description: Human-readable explanation
        confidence: 0.0-1.0 confidence in this concern
        evidence: Supporting evidence for the concern
        suggested_mitigation: How to address this concern
    """
    lens: str
    category: str
    description: str
    confidence: float = 0.5
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_mitigation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "lens": self.lens,
            "category": self.category,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "suggested_mitigation": self.suggested_mitigation,
        }


@dataclass
class Judgment:
    """
    Result of conscience consideration.

    Captures the Voice level, reasons, and guidance for an action.

    Attributes:
        voice: The voice level (QUIET/WHISPER/VOICE/SHOUT)
        concerns: List of specific concerns raised
        summary: Brief human-readable summary
        guidance: Suggested course of action
        metadata: Additional context
        timestamp: When this judgment was made

    Usage:
        judgment = await conscience.consider(action, context)

        if judgment.voice <= Voice.WHISPER:
            # Safe to proceed
            result = await execute(action)
        elif judgment.voice == Voice.VOICE:
            # Recommend human review
            result = await request_approval(action, judgment)
        else:  # SHOUT
            # Block and escalate
            raise SafetyException(judgment.summary)
    """
    voice: Voice
    concerns: List[Concern] = field(default_factory=list)
    summary: str = ""
    guidance: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def allowed(self) -> bool:
        """Whether the action is allowed (voice not SHOUT)."""
        return self.voice < Voice.SHOUT

    @property
    def blocked(self) -> bool:
        """Whether the action is blocked (voice is SHOUT)."""
        return self.voice >= Voice.SHOUT

    @property
    def requires_approval(self) -> bool:
        """Whether human approval is recommended."""
        return self.voice >= Voice.VOICE

    @property
    def confidence(self) -> float:
        """Average confidence across all concerns."""
        if not self.concerns:
            return 1.0  # No concerns = high confidence in safety
        return sum(c.confidence for c in self.concerns) / len(self.concerns)

    @property
    def top_concern(self) -> Optional[Concern]:
        """The concern with highest confidence."""
        if not self.concerns:
            return None
        return max(self.concerns, key=lambda c: c.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "voice": self.voice.name,
            "voice_level": self.voice.value,
            "concerns": [c.to_dict() for c in self.concerns],
            "summary": self.summary,
            "guidance": self.guidance,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "allowed": self.allowed,
            "requires_approval": self.requires_approval,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        parts = [f"{self.voice.emoji} {self.voice.name}"]
        if self.summary:
            parts.append(f": {self.summary}")
        if self.concerns:
            parts.append(f" ({len(self.concerns)} concerns)")
        return "".join(parts)


@dataclass
class Wisdom:
    """
    Learned patterns from past judgments.

    Wisdom captures what has worked and what hasn't, enabling
    the conscience to improve over time.

    Attributes:
        pattern_id: Unique identifier for this pattern
        description: What this wisdom represents
        success_count: Times this pattern led to good outcomes
        failure_count: Times this pattern led to bad outcomes
        alpha: Thompson Sampling alpha (prior successes)
        beta: Thompson Sampling beta (prior failures)
        contexts: Contexts where this wisdom applies
        last_updated: When this wisdom was last updated

    Thompson Sampling:
        Expected success rate = alpha / (alpha + beta)
        Higher confidence with more observations
    """
    pattern_id: str
    description: str
    success_count: int = 0
    failure_count: int = 0
    alpha: float = 1.0  # Thompson Sampling prior (successes)
    beta: float = 1.0   # Thompson Sampling prior (failures)
    contexts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def expected_success_rate(self) -> float:
        """Expected success rate using Thompson Sampling."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def total_observations(self) -> int:
        """Total number of observations."""
        return self.success_count + self.failure_count

    @property
    def confidence(self) -> float:
        """Confidence in this wisdom (based on observations)."""
        # More observations = higher confidence, capped at 0.99
        return min(0.99, 1.0 - 1.0 / (1.0 + self.total_observations * 0.1))

    def update_success(self, confidence: float = 1.0) -> None:
        """Update wisdom with successful outcome."""
        self.success_count += 1
        self.alpha += confidence
        self.last_updated = datetime.now()

    def update_failure(self, confidence: float = 1.0) -> None:
        """Update wisdom with failed outcome."""
        self.failure_count += 1
        self.beta += confidence
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "alpha": self.alpha,
            "beta": self.beta,
            "expected_success_rate": self.expected_success_rate,
            "confidence": self.confidence,
            "contexts": self.contexts,
            "last_updated": self.last_updated.isoformat(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def quiet_judgment(summary: str = "Safe to proceed") -> Judgment:
    """Create a QUIET judgment (safe, no concerns)."""
    return Judgment(
        voice=Voice.QUIET,
        summary=summary,
        guidance="Proceed normally",
    )


def whisper_judgment(
    summary: str,
    concerns: Optional[List[Concern]] = None,
    guidance: str = "Proceed with monitoring"
) -> Judgment:
    """Create a WHISPER judgment (minor concern)."""
    return Judgment(
        voice=Voice.WHISPER,
        concerns=concerns or [],
        summary=summary,
        guidance=guidance,
    )


def voice_judgment(
    summary: str,
    concerns: List[Concern],
    guidance: str = "Request human review before proceeding"
) -> Judgment:
    """Create a VOICE judgment (significant concern)."""
    return Judgment(
        voice=Voice.VOICE,
        concerns=concerns,
        summary=summary,
        guidance=guidance,
    )


def shout_judgment(
    summary: str,
    concerns: List[Concern],
    guidance: str = "Action blocked for safety"
) -> Judgment:
    """Create a SHOUT judgment (critical concern, blocked)."""
    return Judgment(
        voice=Voice.SHOUT,
        concerns=concerns,
        summary=summary,
        guidance=guidance,
    )
