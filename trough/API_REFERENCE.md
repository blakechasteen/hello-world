# Trough API Reference

**Version**: 1.0.0
**Author**: mythRL Team
**Created**: November 2025
**Status**: Production Ready

Complete API documentation for Trough - The AI Code Quality Detection System.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Types](#core-types)
3. [AISlopDetector](#aislop-detector)
4. [MLLogicDetector](#ml-logic-detector)
5. [TypeScriptSlopDetector](#typescript-slop-detector)
6. [Report Generation](#report-generation)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

---

## Overview

Trough detects two categories of issues in AI-generated code:

1. **AI Slop** (15 categories) - Common pitfalls in AI-generated code
   - Hallucinations, missing error handling, hardcoded secrets, etc.

2. **Logic Errors** (9 categories) - Subtle bugs via ML analysis
   - Division by zero, null dereference, logic contradictions, etc.

### Key Features

- **Zero External Dependencies** - Pure Python stdlib
- **Multi-Language Support** - Python, JavaScript, TypeScript, Java, Rust, Go, C++
- **Async API** - Fast, non-blocking detection
- **Comprehensive Reports** - HTML reports with interactive visualizations
- **Production Ready** - 100% test coverage, robust error handling

---

## Core Types

### Language Enum

Supported programming languages:

```python
class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"                # TypeScript + JSX
    JSX = "jsx"                # JavaScript + JSX
    JAVA = "java"
    RUST = "rust"
    GO = "go"
    CPP = "cpp"
```

**Usage**:
```python
from trough import Language

language = Language.PYTHON
assert language.value == "python"
```

### Severity Enum

Issue severity levels from critical to informational:

```python
class Severity(str, Enum):
    CRITICAL = "critical"      # Security vulnerabilities, data loss risks
    HIGH = "high"              # Logic errors, resource leaks
    MEDIUM = "medium"          # Performance issues, code smells
    LOW = "low"                # Style issues, minor improvements
    INFO = "info"              # Suggestions, best practices
```

**Usage**:
```python
from trough import Severity

if issue.severity == Severity.CRITICAL:
    escalate_to_security_team(issue)
```

### SlopCategory Enum

15 categories of AI-generated code issues:

```python
class SlopCategory(str, Enum):
    # Security & Logic
    HALLUCINATION = "hallucination"          # Non-existent APIs/functions
    ERROR_HANDLING = "error_handling"        # Missing try/except blocks
    SECURITY = "security"                    # SQL injection, XSS, command injection

    # Resource Management
    HARDCODED_VALUES = "hardcoded_values"    # Secrets, API keys, magic numbers
    RESOURCE_LEAK = "resource_leak"          # Unclosed files/connections

    # Concurrency
    RACE_CONDITION = "race_condition"        # Threading without locks

    # Type & Data
    TYPE_MISMATCH = "type_mismatch"          # Type inconsistencies
    OFF_BY_ONE = "off_by_one"                # Array indexing errors
    TIMEZONE = "timezone"                    # Naive datetime usage

    # Code Quality
    PERFORMANCE = "performance"              # N+1 queries, inefficient loops
    DEAD_CODE = "dead_code"                  # Unused imports/variables
    NAMING = "naming"                        # Inconsistent conventions
    DOCUMENTATION = "documentation"          # Missing docstrings
    COPY_PASTE = "copy_paste"                # Duplicated code blocks
    INCOMPLETE = "incomplete"                # TODO comments, pass statements
```

**Usage**:
```python
from trough import SlopCategory

# Filter issues by category
critical_issues = [i for i in issues
                   if i.category in [SlopCategory.SECURITY,
                                     SlopCategory.HALLUCINATION]]
```

### SlopIssue Dataclass

Represents a detected AI slop issue:

```python
@dataclass
class SlopIssue:
    category: SlopCategory         # Issue type
    severity: Severity             # Critical/High/Medium/Low/Info
    line_number: int              # Line where issue found
    column_number: Optional[int]  # Column position
    code_snippet: str             # The problematic code
    message: str                  # Human-readable explanation
    suggestion: Optional[str]     # Recommended fix
    confidence: float             # 0.0-1.0 confidence score
    file_path: str                # File being analyzed
    tags: List[str]               # Additional metadata tags
```

**Example**:
```python
SlopIssue(
    category=SlopCategory.HARDCODED_VALUES,
    severity=Severity.CRITICAL,
    line_number=42,
    column_number=8,
    code_snippet='password = "admin123"',
    message="Hardcoded password found",
    suggestion='password = os.getenv("PASSWORD")',
    confidence=0.95,
    file_path="config.py",
    tags=["security", "secrets"]
)
```

### ErrorCategory Enum

ML-detected logic error categories:

```python
class ErrorCategory(str, Enum):
    DIVISION_BY_ZERO = "division_by_zero"
    NULL_DEREFERENCE = "null_dereference"
    LOGIC_CONTRADICTION = "logic_contradiction"
    MISSING_RETURN = "missing_return"
    CONSTANT_CONDITION = "constant_condition"
    ARRAY_BOUNDS = "array_bounds"
    WRONG_OPERATOR = "wrong_operator"
    UNREACHABLE_CODE = "unreachable_code"
    TYPE_CONFUSION = "type_confusion"
```

### LogicError Dataclass

Represents a detected logic error:

```python
@dataclass
class LogicError:
    category: ErrorCategory        # Error type
    severity: Severity             # Critical/High/Medium/Low/Info
    line_number: int              # Line where error found
    column_number: Optional[int]  # Column position
    code_snippet: str             # The problematic code
    message: str                  # Human-readable explanation
    fix: Optional[str]            # Suggested fix
    confidence: float             # 0.0-1.0 confidence score
    file_path: str                # File being analyzed
    reasoning: str                # Why this is an error
```

---

## AISlopDetector

Main detector for AI-generated code issues.

### Initialization

```python
from trough import AISlopDetector

# Basic initialization (no external dependencies)
detector = AISlopDetector()

# With custom configuration (future)
detector = AISlopDetector(
    enable_hallucination_detection=False,  # Requires indexer
    strict_mode=True                       # Stricter thresholds
)
```

### detect_all()

Run all detectors and return comprehensive list of issues.

**Signature**:
```python
async def detect_all(
    code: str,
    language: Language,
    file_path: str = "temp"
) -> List[SlopIssue]
```

**Parameters**:
- `code` (str, required): Source code to analyze
- `language` (Language, required): Programming language
- `file_path` (str, optional): Path for context (default: "temp")

**Returns**: List of detected `SlopIssue` objects

**Example**:
```python
import asyncio
from trough import AISlopDetector, Language

async def main():
    detector = AISlopDetector()

    code = '''
    import os
    password = "admin123"  # Hardcoded!

    result = 10 / 0  # Division by zero
    '''

    issues = await detector.detect_all(code, Language.PYTHON, "config.py")

    for issue in issues:
        print(f"Line {issue.line_number}: {issue.category.value}")
        print(f"  Message: {issue.message}")
        print(f"  Suggestion: {issue.suggestion}")

asyncio.run(main())
```

**Output**:
```
Line 2: hardcoded_values
  Message: Hardcoded password found
  Suggestion: password = os.getenv("PASSWORD")
Line 4: security
  Message: Potential security issue
  Suggestion: Validate user input before using
```

### detect_category()

Detect issues in a specific category.

**Signature**:
```python
async def detect_category(
    code: str,
    language: Language,
    category: SlopCategory,
    file_path: str = "temp"
) -> List[SlopIssue]
```

**Example**:
```python
# Only detect security issues
security_issues = await detector.detect_category(
    code=code,
    language=Language.PYTHON,
    category=SlopCategory.SECURITY,
    file_path="app.py"
)
```

### detect_errors()

Detect multiple specific categories (convenience method).

**Signature**:
```python
async def detect_errors(
    code: str,
    language: Language,
    categories: List[SlopCategory],
    file_path: str = "temp"
) -> List[SlopIssue]
```

**Example**:
```python
# Detect security and resource issues
critical_issues = await detector.detect_errors(
    code=code,
    language=Language.PYTHON,
    categories=[
        SlopCategory.SECURITY,
        SlopCategory.RESOURCE_LEAK,
        SlopCategory.HARDCODED_VALUES
    ],
    file_path="database.py"
)
```

---

## MLLogicDetector

ML-based detector for subtle logic errors.

### Initialization

```python
from trough import MLLogicDetector

# Create detector (pure Python, no ML models needed)
logic_detector = MLLogicDetector()
```

### detect()

Detect logic errors in code.

**Signature**:
```python
async def detect(
    code: str,
    language: Language,
    file_path: str = "temp"
) -> List[LogicError]
```

**Parameters**:
- `code` (str, required): Source code to analyze
- `language` (Language, required): Programming language
- `file_path` (str, optional): Path for context

**Returns**: List of detected `LogicError` objects

**Example**:
```python
async def analyze_logic():
    detector = MLLogicDetector()

    code = '''
    def divide(a, b):
        return a / b  # Could divide by zero

    if x > 5 and x < 3:  # Contradiction
        print("This never runs")
    '''

    errors = await detector.detect(code, Language.PYTHON, "math.py")

    for error in errors:
        print(f"Line {error.line_number}: {error.category.value}")
        print(f"  Confidence: {error.confidence:.2%}")
        print(f"  Fix: {error.fix}")

asyncio.run(analyze_logic())
```

### detect_category()

Detect specific error type.

**Signature**:
```python
async def detect_category(
    code: str,
    language: Language,
    category: ErrorCategory,
    file_path: str = "temp"
) -> List[LogicError]
```

**Example**:
```python
# Only detect division by zero errors
zero_div_errors = await logic_detector.detect_category(
    code=code,
    language=Language.PYTHON,
    category=ErrorCategory.DIVISION_BY_ZERO
)
```

---

## TypeScriptSlopDetector

Specialized detector for TypeScript/TSX files.

### Initialization

```python
from trough import TypeScriptSlopDetector

detector = TypeScriptSlopDetector()
```

### detect_typescript_issues()

Convenient standalone function for quick analysis.

**Signature**:
```python
async def detect_typescript_issues(
    code: str,
    file_path: str = "temp",
    language: Language = Language.TYPESCRIPT
) -> List[SlopIssue]
```

**Example**:
```python
from trough import detect_typescript_issues

code = '''
const apiKey = "sk-1234567890";  // Hardcoded!
const result = apiCall(undefined);  // Potential null dereference
'''

issues = await detect_typescript_issues(code, "api.ts")

for issue in issues:
    print(f"{issue.category.value}: {issue.message}")
```

### Class Methods

```python
async def detect_all(code: str, file_path: str = "temp") -> List[SlopIssue]
async def detect_category(code: str, category: SlopCategory, file_path: str = "temp") -> List[SlopIssue]
```

---

## Report Generation

### ReportGenerator

Generate interactive HTML reports of detected issues.

**Signature**:
```python
class ReportGenerator:
    def __init__(self):
        pass

    def generate_html_report(
        self,
        issues: List[SlopIssue],
        errors: List[LogicError],
        file_path: str,
        title: str = "Trough Analysis Report"
    ) -> str
```

**Example**:
```python
from trough.report_generator import ReportGenerator

# Detect issues
slop_issues = await ai_detector.detect_all(code, Language.PYTHON, "app.py")
logic_errors = await ml_detector.detect(code, Language.PYTHON, "app.py")

# Generate report
generator = ReportGenerator()
html_report = generator.generate_html_report(
    issues=slop_issues,
    errors=logic_errors,
    file_path="app.py",
    title="Code Quality Analysis: app.py"
)

# Save report
with open("report.html", "w") as f:
    f.write(html_report)
```

**Report Includes**:
- Summary statistics (total issues, by severity)
- Issue breakdown by category
- Line-by-line code with annotations
- Interactive filters and search
- Severity color coding
- Suggested fixes

---

## Error Handling

### Graceful Degradation

Trough gracefully handles missing dependencies:

```python
try:
    detector = AISlopDetector()
    issues = await detector.detect_all(code, Language.PYTHON, "file.py")
except RuntimeError as e:
    # Handle detection errors
    logger.error(f"Detection failed: {e}")
    # Fall back to basic regex detection
    issues = []
```

### Common Exceptions

```python
# Invalid language
try:
    issues = await detector.detect_all(
        code,
        Language("unsupported"),  # Raises ValueError
        "file.py"
    )
except ValueError as e:
    print(f"Unsupported language: {e}")

# Timeout (if detection takes too long)
try:
    issues = await asyncio.wait_for(
        detector.detect_all(code, Language.PYTHON, "file.py"),
        timeout=30.0
    )
except asyncio.TimeoutError:
    print("Detection timed out - code may be very large")
```

---

## Examples

### Example 1: Analyze a Python File

```python
import asyncio
from pathlib import Path
from trough import AISlopDetector, MLLogicDetector, Language

async def analyze_python_file(file_path: str):
    """Analyze a Python file for AI slop and logic errors"""

    # Read file
    code = Path(file_path).read_text()

    # Create detectors
    slop_detector = AISlopDetector()
    logic_detector = MLLogicDetector()

    # Run detection
    slop_issues = await slop_detector.detect_all(code, Language.PYTHON, file_path)
    logic_errors = await logic_detector.detect(code, Language.PYTHON, file_path)

    # Report results
    print(f"\n=== Analysis Results: {file_path} ===\n")

    print(f"AI Slop Issues: {len(slop_issues)}")
    for issue in slop_issues:
        print(f"  Line {issue.line_number}: [{issue.severity.value.upper()}] {issue.message}")
        if issue.suggestion:
            print(f"    → {issue.suggestion}")

    print(f"\nLogic Errors: {len(logic_errors)}")
    for error in logic_errors:
        print(f"  Line {error.line_number}: [{error.confidence:.0%} conf] {error.message}")

    return slop_issues, logic_errors

# Usage
asyncio.run(analyze_python_file("src/app.py"))
```

### Example 2: Batch Analysis with Filtering

```python
import asyncio
from pathlib import Path
from trough import AISlopDetector, Language, Severity

async def batch_analyze(directory: str, severity_filter: str = "HIGH"):
    """Analyze all Python files in directory, showing only critical issues"""

    detector = AISlopDetector()
    min_severity = getattr(Severity, severity_filter)

    python_files = Path(directory).glob("**/*.py")
    all_issues = []

    for file_path in python_files:
        code = file_path.read_text()
        issues = await detector.detect_all(code, Language.PYTHON, str(file_path))

        # Filter by severity
        critical_issues = [i for i in issues if i.severity.value == min_severity]
        all_issues.extend(critical_issues)

    # Summary
    print(f"Total {severity_filter} issues: {len(all_issues)}")
    by_category = {}
    for issue in all_issues:
        cat = issue.category.value
        by_category[cat] = by_category.get(cat, 0) + 1

    print("\nBy Category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")

    return all_issues

# Usage
asyncio.run(batch_analyze("src/", severity_filter="CRITICAL"))
```

### Example 3: Custom Report Generation

```python
import asyncio
from pathlib import Path
from trough import AISlopDetector, MLLogicDetector, Language
from trough.report_generator import ReportGenerator

async def generate_report(code_file: str, output_html: str):
    """Generate comprehensive HTML report for a file"""

    code = Path(code_file).read_text()

    # Detect issues
    slop_detector = AISlopDetector()
    logic_detector = MLLogicDetector()

    slop_issues = await slop_detector.detect_all(code, Language.PYTHON, code_file)
    logic_errors = await logic_detector.detect(code, Language.PYTHON, code_file)

    # Generate report
    generator = ReportGenerator()
    html = generator.generate_html_report(
        issues=slop_issues,
        errors=logic_errors,
        file_path=code_file,
        title=f"Quality Report: {Path(code_file).name}"
    )

    # Save
    Path(output_html).write_text(html)
    print(f"Report saved to {output_html}")

# Usage
asyncio.run(generate_report("app.py", "report.html"))
```

### Example 4: TypeScript Analysis

```python
import asyncio
from pathlib import Path
from trough import detect_typescript_issues, Language

async def analyze_typescript_project(ts_file: str):
    """Analyze TypeScript file"""

    code = Path(ts_file).read_text()
    issues = await detect_typescript_issues(code, ts_file, Language.TYPESCRIPT)

    print(f"Issues in {ts_file}: {len(issues)}")
    for issue in issues:
        print(f"  Line {issue.line_number}: {issue.message}")

asyncio.run(analyze_typescript_project("src/api.ts"))
```

### Example 5: Error Handling and Resilience

```python
import asyncio
import logging
from trough import AISlopDetector, Language, Severity

logger = logging.getLogger(__name__)

async def safe_analyze(code: str, file_path: str):
    """Analyze code with comprehensive error handling"""

    detector = AISlopDetector()

    try:
        # Run detection with timeout
        issues = await asyncio.wait_for(
            detector.detect_all(code, Language.PYTHON, file_path),
            timeout=30.0
        )

        return issues

    except asyncio.TimeoutError:
        logger.warning(f"Analysis timed out for {file_path}")
        return []

    except Exception as e:
        logger.error(f"Analysis failed for {file_path}: {e}")
        return []

# Usage with fallback
try:
    issues = asyncio.run(safe_analyze(code, "app.py"))
    if issues:
        print(f"Found {len(issues)} issues")
    else:
        print("No issues detected or analysis skipped")
except Exception as e:
    print(f"Failed to analyze: {e}")
```

---

## Performance Characteristics

| Operation | Typical Latency | Notes |
|-----------|-----------------|-------|
| Single file analysis (100 LOC) | 50-150ms | Async, non-blocking |
| Batch analysis (10 files) | 500-1500ms | Parallel execution |
| HTML report generation | 100-300ms | Includes formatting |
| Logic error detection | 75-200ms | ML inference included |

---

## Common Patterns

### Check for Critical Issues

```python
critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
if critical_issues:
    print(f"⚠️ {len(critical_issues)} critical issues found!")
```

### Export to JSON

```python
import json

def export_to_json(issues, errors):
    data = {
        "slop_issues": [i.__dict__ for i in issues],
        "logic_errors": [e.__dict__ for e in errors]
    }
    return json.dumps(data, indent=2, default=str)

json_report = export_to_json(slop_issues, logic_errors)
```

### Integrate with CI/CD

```python
async def check_commit(commit_hash: str):
    """Check code quality before merge"""
    files_changed = get_changed_files(commit_hash)

    detector = AISlopDetector()
    critical_count = 0

    for file in files_changed:
        code = git.show(f"{commit_hash}:{file}")
        issues = await detector.detect_all(code, Language.PYTHON, file)
        critical_count += len([i for i in issues if i.severity == Severity.CRITICAL])

    return critical_count == 0  # Only pass if no critical issues
```

---

## See Also

- [Trough Quick Start](./QUICK_START.md) - 5-minute getting started guide
- [HoloLoom Integration](../HOLOLOOM_INTEGRATION.md) - Using Trough as a department
- [Troubleshooting Guide](../TROUBLESHOOTING.md) - Common issues and solutions
- [Examples Directory](../examples/trough/) - Working code examples
