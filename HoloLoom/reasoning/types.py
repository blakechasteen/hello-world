#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Engine Core Types
============================
Type definitions for the Layer 6 Reasoning Engine.

Author: Claude Code
Date: 2025-11-15
Phase: 1 (Foundation)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================

class ReasoningMode(Enum):
    """
    Reasoning complexity modes.

    FAST: Single-step reasoning (<50ms overhead)
    STANDARD: Multi-step chain-of-thought (3-5 steps, ~200ms)
    DEEP: Planning + verification + backtracking (~500ms+)
    """
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class StepType(Enum):
    """
    Types of reasoning steps in the chain.
    """
    UNDERSTANDING = "understanding"      # Analyze query intent
    EVIDENCE = "evidence"                # Gather evidence from context
    SYNTHESIS = "synthesis"              # Synthesize reasoning
    VERIFICATION = "verification"        # Self-check consistency
    BACKTRACK = "backtrack"             # Revise earlier steps
    PLANNING = "planning"                # Create sub-question plan
    CORRECTION = "correction"            # Correct detected error


class QueryType(Enum):
    """
    Types of queries for intent classification.
    """
    FACTUAL = "factual"                 # "What is X?"
    PROCEDURAL = "procedural"           # "How to do X?"
    COMPARATIVE = "comparative"         # "Compare X and Y"
    ANALYTICAL = "analytical"           # "Why does X happen?"
    CREATIVE = "creative"               # "Design X for Y"
    VERIFICATION = "verification"       # "Is X true?"


class VerificationSeverity(Enum):
    """
    Severity levels for verification issues.
    """
    INFO = "info"                       # Minor issue, informational
    WARNING = "warning"                 # Potential issue, worth noting
    CRITICAL = "critical"               # Major issue, requires correction


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QueryIntent:
    """
    Analyzed intent of a query.

    Attributes:
        type: Query type classification
        requirements: List of what's needed to answer
        complexity: Estimated complexity [0.0, 1.0]
        confidence: Confidence in intent classification [0.0, 1.0]
        key_concepts: Main concepts identified
    """
    type: QueryType
    requirements: List[str]
    complexity: float
    confidence: float
    key_concepts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.complexity <= 1.0, "Complexity must be [0.0, 1.0]"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be [0.0, 1.0]"


@dataclass
class ReasoningStep:
    """
    Single step in a reasoning chain.

    Attributes:
        thought: The reasoning at this step
        evidence: Supporting evidence from context
        confidence: Confidence in this step [0.0, 1.0]
        step_type: Type of reasoning step
        timestamp: When this step was created
        metadata: Additional step-specific data
    """
    thought: str
    evidence: str
    confidence: float
    step_type: StepType = StepType.SYNTHESIS
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be [0.0, 1.0]"


@dataclass
class VerificationResult:
    """
    Result of verification check.

    Attributes:
        passed: Whether verification passed
        issue: Description of issue (if any)
        correction: Suggested correction (if any)
        severity: Severity of issue
        confidence: Confidence in verification [0.0, 1.0]
    """
    passed: bool
    issue: Optional[str] = None
    correction: Optional[str] = None
    severity: VerificationSeverity = VerificationSeverity.INFO
    confidence: float = 1.0

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be [0.0, 1.0]"


@dataclass
class MultiPassVerification:
    """
    Result of multi-pass verification (DEEP mode).

    Attributes:
        passes: List of verification results (one per pass)
        has_contradictions: Whether contradictions were detected
        contradictions: List of contradiction descriptions
        overall_confidence: Overall confidence after all passes
    """
    passes: List[VerificationResult]
    has_contradictions: bool = False
    contradictions: List[str] = field(default_factory=list)
    overall_confidence: float = 1.0

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.overall_confidence <= 1.0, "Confidence must be [0.0, 1.0]"


@dataclass
class EvidencePiece:
    """
    Single piece of evidence extracted from context.

    Attributes:
        text: Evidence text
        source: Source identifier (shard ID, entity, etc.)
        relevance: Relevance score [0.0, 1.0]
        confidence: Confidence in this evidence [0.0, 1.0]
    """
    text: str
    source: str
    relevance: float
    confidence: float = 1.0

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.relevance <= 1.0, "Relevance must be [0.0, 1.0]"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be [0.0, 1.0]"


@dataclass
class Synthesis:
    """
    Synthesized reasoning from evidence.

    Attributes:
        reasoning: Main reasoning text
        key_points: List of key points
        confidence: Confidence in synthesis [0.0, 1.0]
        evidence_used: Number of evidence pieces used
    """
    reasoning: str
    key_points: List[str]
    confidence: float
    evidence_used: int = 0

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be [0.0, 1.0]"


@dataclass
class QueryPlan:
    """
    Plan for answering complex query (DEEP mode).

    Attributes:
        steps: List of sub-questions/steps
        estimated_complexity: Estimated total complexity [0.0, 1.0]
        dependencies: Dependencies between steps
    """
    steps: List['PlanStep']
    estimated_complexity: float
    dependencies: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.estimated_complexity <= 1.0, "Complexity must be [0.0, 1.0]"


@dataclass
class PlanStep:
    """
    Single step in a query plan.

    Attributes:
        question: Sub-question to answer
        required_for: What this enables
        complexity: Step complexity [0.0, 1.0]
    """
    question: str
    required_for: str
    complexity: float = 0.5

    def __post_init__(self):
        """Validate ranges."""
        assert 0.0 <= self.complexity <= 1.0, "Complexity must be [0.0, 1.0]"


@dataclass
class ReasoningResult:
    """
    Complete result of reasoning process.

    Attributes:
        chain: List of reasoning steps
        mode: Reasoning mode used
        total_confidence: Overall confidence [0.0, 1.0]
        duration_ms: Time taken for reasoning
        metadata: Additional result data
    """
    chain: List[ReasoningStep]
    mode: ReasoningMode
    total_confidence: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and calculate total confidence if not provided."""
        if self.total_confidence == 0.0 and self.chain:
            # Average confidence across all steps
            self.total_confidence = sum(s.confidence for s in self.chain) / len(self.chain)

        assert 0.0 <= self.total_confidence <= 1.0, "Confidence must be [0.0, 1.0]"

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Reasoning Summary:\n"
            f"  Mode: {self.mode.value}\n"
            f"  Steps: {len(self.chain)}\n"
            f"  Confidence: {self.total_confidence:.2f}\n"
            f"  Duration: {self.duration_ms:.1f}ms"
        )


# ============================================================================
# Constants
# ============================================================================

# Default confidence thresholds for mode selection
DEFAULT_CONFIDENCE_THRESHOLDS = {
    ReasoningMode.FAST: 0.85,      # High confidence → FAST
    ReasoningMode.STANDARD: 0.70,  # Medium confidence → STANDARD
    ReasoningMode.DEEP: 0.0,       # Low confidence → DEEP
}

# Step type icons for visualization
STEP_TYPE_ICONS = {
    StepType.UNDERSTANDING: "🧠",
    StepType.EVIDENCE: "🔍",
    StepType.SYNTHESIS: "🔗",
    StepType.VERIFICATION: "✓",
    StepType.BACKTRACK: "↩",
    StepType.PLANNING: "📋",
    StepType.CORRECTION: "🔧",
}

# Query type complexity estimates (baseline)
QUERY_TYPE_COMPLEXITY = {
    QueryType.FACTUAL: 0.3,
    QueryType.PROCEDURAL: 0.5,
    QueryType.COMPARATIVE: 0.6,
    QueryType.ANALYTICAL: 0.7,
    QueryType.CREATIVE: 0.8,
    QueryType.VERIFICATION: 0.6,
}
