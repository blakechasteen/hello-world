"""
xTerminator Type Definitions
============================

Core types for classification and fix proposal system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class RiskLevel(str, Enum):
    """Risk level for proposed fixes"""
    LOW = "low"           # Safe to autofix (copy-paste, formatting)
    MEDIUM = "medium"     # Needs review (error handling, logic changes)
    HIGH = "high"         # Requires careful review (security-adjacent, data handling)
    CRITICAL = "critical" # Manual only (actual security issues, breaking changes)


class FixStrategy(str, Enum):
    """Strategy for applying fixes"""
    AST = "ast"               # AST transformation (extract function, rename variable)
    TEMPLATE = "template"     # Template-based (add try/except, move to env)
    MANUAL = "manual"         # Requires human implementation
    SKIP = "skip"             # Don't fix (false positive or too risky)


class ContextType(str, Enum):
    """Type of code context where issue appears"""
    EXECUTABLE = "executable"     # Normal executable code
    COMMENT = "comment"           # Inside comment
    DOCSTRING = "docstring"       # Inside docstring
    STRING_LITERAL = "string"     # Inside string literal
    ANNOTATION = "annotation"     # Type annotation
    IMPORT = "import"             # Import statement
    DEFINITION = "definition"     # Function/class definition
    UNKNOWN = "unknown"           # Unable to determine


@dataclass
class CodeContext:
    """Context information about code location"""
    context_type: ContextType
    file_path: str
    line_number: int
    column: int = 0

    # Surrounding code
    code_snippet: str = ""
    lines_before: List[str] = field(default_factory=list)
    lines_after: List[str] = field(default_factory=list)

    # AST context (if available)
    parent_node_type: Optional[str] = None  # 'FunctionDef', 'ClassDef', etc.
    parent_node_name: Optional[str] = None
    scope_depth: int = 0

    # Semantic context
    in_try_block: bool = False
    in_loop: bool = False
    in_conditional: bool = False
    has_tests: bool = False  # Does this module have tests?

    def is_safe_location(self) -> bool:
        """Check if location is safe for automated fixing"""
        unsafe_contexts = {
            ContextType.COMMENT,
            ContextType.DOCSTRING,
            ContextType.STRING_LITERAL,
            ContextType.UNKNOWN
        }
        return self.context_type not in unsafe_contexts


@dataclass
class FixProposal:
    """Proposed fix for an issue"""
    fix_id: str
    issue_category: str
    issue_severity: str

    # Classification results
    risk_level: RiskLevel
    fix_strategy: FixStrategy
    confidence: float  # 0.0-1.0

    # Fix details
    original_code: str
    proposed_code: Optional[str] = None
    diff: Optional[str] = None
    explanation: str = ""

    # Context
    context: Optional[CodeContext] = None

    # Decision flags
    safe_to_autofix: bool = False
    requires_approval: bool = True
    requires_tests: bool = True

    # Validation plan
    validation_steps: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_skip(self) -> bool:
        """Should this fix be skipped?"""
        return self.fix_strategy == FixStrategy.SKIP

    def is_automated(self) -> bool:
        """Can this be applied automatically?"""
        return (
            self.safe_to_autofix and
            not self.requires_approval and
            self.confidence >= 0.85 and
            self.risk_level == RiskLevel.LOW
        )

    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Fix {self.fix_id}: {self.issue_category}\n"
            f"  Risk: {self.risk_level.value}, Strategy: {self.fix_strategy.value}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Auto-fix: {'Yes' if self.is_automated() else 'No'}\n"
            f"  {self.explanation}"
        )


@dataclass
class ClassificationResult:
    """Complete classification of an issue"""
    # Required fields (no defaults)
    issue_id: str
    context: CodeContext
    is_false_positive: bool
    risk_level: RiskLevel
    selected_strategy: FixStrategy

    # Optional fields (with defaults)
    false_positive_reason: str = ""
    risk_factors: List[str] = field(default_factory=list)
    strategy_reason: str = ""
    alternative_strategies: List[FixStrategy] = field(default_factory=list)
    confidence: float = 0.0
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""

    def to_fix_proposal(self, issue, original_code: str) -> FixProposal:
        """Convert to FixProposal"""
        return FixProposal(
            fix_id=self.issue_id,
            issue_category=getattr(issue, 'category', 'unknown'),
            issue_severity=getattr(issue, 'severity', 'unknown'),
            risk_level=self.risk_level,
            fix_strategy=self.selected_strategy,
            confidence=self.confidence,
            original_code=original_code,
            explanation=self.recommendation,
            context=self.context,
            safe_to_autofix=(
                self.risk_level == RiskLevel.LOW and
                self.confidence >= 0.85 and
                not self.is_false_positive
            ),
            requires_approval=(
                self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or
                self.confidence < 0.7
            ),
            requires_tests=True,
            validation_steps=[
                "syntax_check",
                "test_execution" if self.context.has_tests else "manual_test",
                "trough_rescan"
            ]
        )
