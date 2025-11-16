"""
Trough Type Definitions
========================

Core types and enums for Trough detectors.
No external dependencies - pure Python stdlib.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional


# ============================================================================
# Language Support
# ============================================================================

class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"  # TypeScript + JSX
    JSX = "jsx"  # JavaScript + JSX
    JAVA = "java"
    RUST = "rust"
    GO = "go"
    CPP = "cpp"


# ============================================================================
# Severity Levels
# ============================================================================

class Severity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"  # Security vulnerabilities, data loss risks
    HIGH = "high"          # Logic errors, resource leaks
    MEDIUM = "medium"      # Performance issues, code smells
    LOW = "low"            # Style issues, minor improvements
    INFO = "info"          # Suggestions, best practices


# ============================================================================
# AI Slop Categories
# ============================================================================

class SlopCategory(str, Enum):
    """Categories of AI-generated code issues."""
    HALLUCINATION = "hallucination"              # Non-existent functions/classes
    ERROR_HANDLING = "error_handling"            # Missing try/except
    HARDCODED_VALUES = "hardcoded_values"        # Secrets, magic numbers
    RACE_CONDITION = "race_condition"            # Threading without locks
    RESOURCE_LEAK = "resource_leak"              # Files/connections not closed
    TYPE_MISMATCH = "type_mismatch"              # Type inconsistencies
    SECURITY = "security"                        # SQL injection, XSS, etc.
    PERFORMANCE = "performance"                  # N+1 queries, inefficient loops
    DEAD_CODE = "dead_code"                      # Unused imports/variables
    NAMING = "naming"                            # Inconsistent naming
    DOCUMENTATION = "documentation"              # Missing docs
    COPY_PASTE = "copy_paste"                    # Duplicated code
    INCOMPLETE = "incomplete"                    # TODO, pass, ...
    OFF_BY_ONE = "off_by_one"                    # Array index errors
    TIMEZONE = "timezone"                        # Timezone handling issues


# ============================================================================
# Logic Error Categories
# ============================================================================

class ErrorCategory(str, Enum):
    """Categories of logic errors detected by ML."""
    DIVISION_BY_ZERO = "division_by_zero"
    NULL_DEREFERENCE = "null_dereference"
    LOGIC_CONTRADICTION = "logic_contradiction"
    MISSING_RETURN = "missing_return"
    CONSTANT_CONDITION = "constant_condition"
    ARRAY_BOUNDS = "array_bounds"
    WRONG_OPERATOR = "wrong_operator"
    INFINITE_LOOP = "infinite_loop"
    UNREACHABLE_CODE = "unreachable_code"
    MEMORY_LEAK = "memory_leak"
    RACE_CONDITION = "race_condition"
    INTEGER_OVERFLOW = "integer_overflow"
    TYPE_CONFUSION = "type_confusion"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEADLOCK = "deadlock"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SlopIssue:
    """An AI slop detection issue."""
    category: SlopCategory
    severity: Severity
    message: str
    file_path: str
    line_number: int
    code_snippet: str
    suggestion: str
    confidence: float = 1.0  # 0.0-1.0


@dataclass
class LogicError:
    """A logic error detected by ML analysis."""
    category: ErrorCategory
    severity: Severity
    message: str
    file_path: str
    line_number: int
    code_snippet: str
    explanation: str
    confidence: float  # 0.0-1.0
    suggested_fix: Optional[str] = None
    proof: Optional[Dict[str, Any]] = None  # Concrete values demonstrating the error


@dataclass
class DetectionResult:
    """Complete detection result for a file."""
    file_path: str
    language: Language
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    issues: List[Any]  # List[SlopIssue] or List[LogicError]
    scan_time_ms: float


@dataclass
class DetectionSummary:
    """Summary of detection results across multiple files."""
    total_files: int
    total_issues: int
    by_severity: Dict[Severity, int]
    by_category: Dict[str, int]
    files_with_issues: List[str]
    scan_time_ms: float
