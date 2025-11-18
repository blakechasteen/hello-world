"""
Contract-First Prompting Types

Data structures for contract-first prompting system.

Created: 2025-11-18
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class DiggingStrategy(Enum):
    """Strategy for iterative questioning."""

    BREADTH_FIRST = "breadth_first"  # Cover all dimensions broadly first
    DEPTH_FIRST = "depth_first"  # Deep dive into one dimension at a time
    ESSENTIAL_ONLY = "essential_only"  # Ask only critical questions
    ADAPTIVE = "adaptive"  # Adjust based on user responses


class ConfidenceLevel(Enum):
    """Confidence levels for contract clarity."""

    LOW = 0.5  # <50% confidence
    MEDIUM = 0.7  # 50-70% confidence
    HIGH = 0.85  # 70-85% confidence
    VERY_HIGH = 0.95  # 85-95% confidence
    CERTAIN = 1.0  # >95% confidence


class UserResponse(Enum):
    """User response options at echo check stage."""

    YES = "yes"  # Approve and proceed
    EDIT = "edit"  # Request changes
    BLUEPRINT = "blueprint"  # Request outline
    RISKS = "risks"  # Request risk analysis
    RESET = "reset"  # Start over
    SHOW_GAPS = "showgaps"  # Show gap analysis
    CONFIDENCE = "confidence"  # Show confidence level


class DeliverableType(Enum):
    """Types of deliverables."""

    CODE = "code"  # Software code
    DOCUMENT = "document"  # Text document
    PRD = "prd"  # Product requirements document
    SPEC = "spec"  # Technical specification
    ARCHITECTURE = "architecture"  # System architecture
    ANALYSIS = "analysis"  # Data analysis
    OTHER = "other"  # Other type


@dataclass
class Gap:
    """Represents a gap in understanding."""

    dimension: str  # What dimension (purpose, audience, etc.)
    question: str  # Question to ask
    priority: float  # Priority (0.0-1.0)
    optional: bool = False  # Can be skipped if time-limited
    asked: bool = False  # Has this been asked?
    answer: Optional[str] = None  # User's answer
    confidence_impact: float = 0.1  # How much this affects confidence


@dataclass
class GapAnalysis:
    """Analysis of gaps in understanding."""

    gaps: List[Gap] = field(default_factory=list)
    confidence: float = 0.0  # Current confidence (0.0-1.0)
    essential_gaps: List[Gap] = field(default_factory=list)
    optional_gaps: List[Gap] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_gap(
        self,
        dimension: str,
        question: str,
        priority: float,
        optional: bool = False,
        confidence_impact: float = 0.1,
    ) -> None:
        """Add a gap to the analysis."""
        gap = Gap(
            dimension=dimension,
            question=question,
            priority=priority,
            optional=optional,
            confidence_impact=confidence_impact,
        )
        self.gaps.append(gap)
        if optional:
            self.optional_gaps.append(gap)
        else:
            self.essential_gaps.append(gap)

    def next_question(self) -> Optional[Gap]:
        """Get the next question to ask."""
        # Sort by priority (highest first)
        unanswered = [g for g in self.gaps if not g.asked]
        if not unanswered:
            return None
        return sorted(unanswered, key=lambda g: g.priority, reverse=True)[0]

    def update_confidence(self) -> float:
        """Recalculate confidence based on answered gaps."""
        total_impact = sum(g.confidence_impact for g in self.gaps if not g.optional)
        answered_impact = sum(
            g.confidence_impact for g in self.gaps if g.answer and not g.optional
        )
        self.confidence = min(1.0, answered_impact / total_impact if total_impact > 0 else 0.0)
        return self.confidence


@dataclass
class Contract:
    """Represents an agreed work contract."""

    deliverable: str  # What will be built
    key_includes: List[str]  # Key features/sections
    hard_constraints: List[str]  # Hard limits/requirements
    success_criteria: List[str]  # How to know it's good enough
    deliverable_type: DeliverableType = DeliverableType.OTHER
    confidence: float = 0.0  # Confidence in clarity (0.0-1.0)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    approved: bool = False

    def echo_check(self) -> str:
        """Generate echo check statement."""
        includes_str = ", ".join(self.key_includes[:3])  # Top 3
        constraint = self.hard_constraints[0] if self.hard_constraints else "meet requirements"

        return f"""I will create {self.deliverable} that {includes_str}.
It must {constraint}.

Is this correct? Reply:
- yes (to lock it in)
- edit (to change something)
- blueprint (to see the outline first)
- risks (to call out potential issues)"""


@dataclass
class BlueprintSection:
    """A section in a blueprint."""

    name: str
    description: str
    subsections: List[str] = field(default_factory=list)
    priority: int = 1  # 1=critical, 2=important, 3=nice-to-have


@dataclass
class Blueprint:
    """Structured outline of deliverable."""

    title: str
    sections: List[BlueprintSection] = field(default_factory=list)
    testing_approach: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def render(self) -> str:
        """Render blueprint as markdown."""
        lines = [f"# Blueprint: {self.title}\n"]

        for i, section in enumerate(self.sections, 1):
            priority_marker = {1: "=4", 2: "=á", 3: "=5"}.get(section.priority, "")
            lines.append(f"\n{i}. {priority_marker} **{section.name}**")
            lines.append(f"   {section.description}")

            if section.subsections:
                for subsection in section.subsections:
                    lines.append(f"   - {subsection}")

        if self.testing_approach:
            lines.append(f"\n**Testing Approach**: {self.testing_approach}")

        return "\n".join(lines)


@dataclass
class Risk:
    """Represents a risk and mitigation strategy."""

    description: str
    severity: str  # "low", "medium", "high", "critical"
    probability: str  # "low", "medium", "high"
    mitigation: str
    impact: Optional[str] = None


@dataclass
class RiskAnalysis:
    """Analysis of risks for a deliverable."""

    risks: List[Risk] = field(default_factory=list)
    overall_risk_level: str = "medium"  # "low", "medium", "high", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_risk(
        self,
        description: str,
        severity: str,
        probability: str,
        mitigation: str,
        impact: Optional[str] = None,
    ) -> None:
        """Add a risk."""
        self.risks.append(
            Risk(
                description=description,
                severity=severity,
                probability=probability,
                mitigation=mitigation,
                impact=impact,
            )
        )

    def render(self) -> str:
        """Render risk analysis as markdown."""
        lines = [f"# Risk Analysis ({self.overall_risk_level.upper()} overall)\n"]

        severity_emoji = {
            "low": "=â",
            "medium": "=á",
            "high": "=à",
            "critical": "=4",
        }

        for i, risk in enumerate(self.risks, 1):
            emoji = severity_emoji.get(risk.severity, "ª")
            lines.append(f"\n{i}. {emoji} **{risk.description}**")
            lines.append(f"   - Severity: {risk.severity} | Probability: {risk.probability}")
            lines.append(f"   - **Mitigation**: {risk.mitigation}")
            if risk.impact:
                lines.append(f"   - **Impact if unmitigated**: {risk.impact}")

        return "\n".join(lines)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractSession:
    """A complete contract-first prompting session."""

    session_id: str
    initial_idea: str
    gap_analysis: Optional[GapAnalysis] = None
    contract: Optional[Contract] = None
    blueprint: Optional[Blueprint] = None
    risk_analysis: Optional[RiskAnalysis] = None
    conversation: List[ConversationTurn] = field(default_factory=list)
    final_deliverable: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_turn(self, role: str, content: str, **metadata) -> None:
        """Add a conversation turn."""
        self.conversation.append(
            ConversationTurn(role=role, content=content, metadata=metadata)
        )

    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.final_deliverable is not None and self.completed_at is not None
