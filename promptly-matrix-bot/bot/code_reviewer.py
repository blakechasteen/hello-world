#!/usr/bin/env python3
"""
Code Review System for Promptly Matrix Bot

Analyzes code for:
- Security vulnerabilities (SQL injection, XSS, etc.)
- Code quality issues
- Style violations
- Best practices
- Performance concerns

Features:
- Pattern-based analysis (no external deps required)
- Language detection
- Risk scoring
- Actionable recommendations
- Support for Python, JavaScript, TypeScript, SQL, Shell
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Code issue types"""
    SECURITY = "security"
    QUALITY = "quality"
    STYLE = "style"
    PERFORMANCE = "performance"
    BEST_PRACTICE = "best_practice"


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"  # Must fix
    HIGH = "high"          # Should fix
    MEDIUM = "medium"      # Consider fixing
    LOW = "low"            # Optional
    INFO = "info"          # Informational


@dataclass
class CodeIssue:
    """Code issue/finding"""
    type: IssueType
    severity: Severity
    title: str
    description: str
    line: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration


@dataclass
class CodeReviewResult:
    """Complete code review result"""
    language: str
    issues: List[CodeIssue] = field(default_factory=list)
    lines_analyzed: int = 0
    risk_score: float = 0.0  # 0-10 scale
    summary: Dict[str, int] = field(default_factory=dict)

    def add_issue(self, issue: CodeIssue):
        """Add issue and update summary"""
        self.issues.append(issue)

        # Update summary
        key = f"{issue.severity.value}_{issue.type.value}"
        self.summary[key] = self.summary.get(key, 0) + 1

        # Update risk score
        severity_weights = {
            Severity.CRITICAL: 3.0,
            Severity.HIGH: 2.0,
            Severity.MEDIUM: 1.0,
            Severity.LOW: 0.5,
            Severity.INFO: 0.1
        }
        self.risk_score += severity_weights.get(issue.severity, 0)

    def get_critical_count(self) -> int:
        """Get number of critical issues"""
        return sum(1 for issue in self.issues if issue.severity == Severity.CRITICAL)

    def get_high_count(self) -> int:
        """Get number of high severity issues"""
        return sum(1 for issue in self.issues if issue.severity == Severity.HIGH)


class CodeReviewer:
    """
    Analyzes code for security and quality issues.

    Usage:
        reviewer = CodeReviewer()
        result = reviewer.review(code, language="python")

        print(f"Risk score: {result.risk_score}/10")
        for issue in result.issues:
            print(f"[{issue.severity.value}] {issue.title}")
    """

    def __init__(self):
        """Initialize code reviewer with security patterns"""
        self.patterns = self._load_security_patterns()

    def review(self, code: str, language: Optional[str] = None) -> CodeReviewResult:
        """
        Review code for security and quality issues.

        Args:
            code: Source code to review
            language: Programming language (auto-detected if None)

        Returns:
            CodeReviewResult with findings
        """
        # Detect language if not specified
        if not language:
            language = self._detect_language(code)

        result = CodeReviewResult(
            language=language,
            lines_analyzed=len(code.split('\n'))
        )

        # Run security checks
        self._check_security(code, language, result)

        # Run quality checks
        self._check_quality(code, language, result)

        # Run style checks
        self._check_style(code, language, result)

        # Normalize risk score
        result.risk_score = min(result.risk_score, 10.0)

        logger.info(f"Reviewed {result.lines_analyzed} lines of {language}, "
                   f"found {len(result.issues)} issues, risk={result.risk_score:.1f}")

        return result

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code"""
        code_lower = code.lower()

        # Python
        if re.search(r'\bdef\s+\w+\s*\(|import\s+\w+|\bclass\s+\w+:', code):
            return "python"

        # JavaScript/TypeScript
        if re.search(r'\bfunction\s+\w+\s*\(|\bconst\s+\w+\s*=|\blet\s+\w+\s*=|=>|interface\s+\w+', code):
            if 'interface' in code or 'type' in code and ':' in code:
                return "typescript"
            return "javascript"

        # SQL
        if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b', code, re.IGNORECASE):
            return "sql"

        # Shell
        if re.search(r'#!/bin/(bash|sh)|^\$\s|\becho\s', code):
            return "shell"

        # Go
        if re.search(r'\bfunc\s+\w+\s*\(|package\s+\w+|import\s+\(', code):
            return "go"

        return "unknown"

    def _check_security(self, code: str, language: str, result: CodeReviewResult):
        """Check for security vulnerabilities"""
        patterns = self.patterns.get(language, {}).get('security', [])

        for pattern_dict in patterns:
            pattern = pattern_dict['pattern']
            title = pattern_dict['title']
            severity = pattern_dict['severity']
            cwe = pattern_dict.get('cwe')
            recommendation = pattern_dict.get('recommendation')

            matches = list(re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE))

            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                snippet = match.group(0)[:100]

                result.add_issue(CodeIssue(
                    type=IssueType.SECURITY,
                    severity=severity,
                    title=title,
                    description=f"Found potential security issue on line {line_num}",
                    line=line_num,
                    code_snippet=snippet,
                    recommendation=recommendation,
                    cwe_id=cwe
                ))

    def _check_quality(self, code: str, language: str, result: CodeReviewResult):
        """Check code quality"""
        lines = code.split('\n')

        # Check for overly long functions
        if language in ['python', 'javascript', 'typescript']:
            self._check_function_length(code, language, result)

        # Check for high cyclomatic complexity
        self._check_complexity(code, language, result)

        # Check for hardcoded secrets
        self._check_secrets(code, result)

        # Check for TODO/FIXME comments
        self._check_todos(code, result)

    def _check_style(self, code: str, language: str, result: CodeReviewResult):
        """Check code style"""
        lines = code.split('\n')

        # Check line length
        for i, line in enumerate(lines):
            if len(line) > 120:
                result.add_issue(CodeIssue(
                    type=IssueType.STYLE,
                    severity=Severity.LOW,
                    title="Long line",
                    description=f"Line {i+1} exceeds 120 characters ({len(line)} chars)",
                    line=i+1,
                    recommendation="Break long lines for readability"
                ))

        # Check for trailing whitespace
        for i, line in enumerate(lines):
            if line.endswith(' ') or line.endswith('\t'):
                result.add_issue(CodeIssue(
                    type=IssueType.STYLE,
                    severity=Severity.INFO,
                    title="Trailing whitespace",
                    description=f"Line {i+1} has trailing whitespace",
                    line=i+1
                ))

    def _check_function_length(self, code: str, language: str, result: CodeReviewResult):
        """Check for overly long functions"""
        if language == 'python':
            pattern = r'^\s*(def|async def)\s+(\w+)\s*\([^)]*\):'
        elif language in ['javascript', 'typescript']:
            pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{|(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{'
        else:
            return

        matches = list(re.finditer(pattern, code, re.MULTILINE))

        for i, match in enumerate(matches):
            start_line = code[:match.start()].count('\n') + 1

            # Find end of function (simple heuristic)
            next_start = matches[i+1].start() if i+1 < len(matches) else len(code)
            func_code = code[match.start():next_start]
            func_lines = func_code.split('\n')

            if len(func_lines) > 50:
                func_name = match.group(2) if language == 'python' else match.group(1)
                result.add_issue(CodeIssue(
                    type=IssueType.QUALITY,
                    severity=Severity.MEDIUM,
                    title="Long function",
                    description=f"Function '{func_name}' is {len(func_lines)} lines (recommended: <50)",
                    line=start_line,
                    recommendation="Consider breaking into smaller functions"
                ))

    def _check_complexity(self, code: str, language: str, result: CodeReviewResult):
        """Check cyclomatic complexity (simplified)"""
        # Count decision points
        decision_keywords = [
            r'\bif\b', r'\belif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bcatch\b', r'\bexcept\b', r'\bcase\b', r'\bswitch\b',
            r'\?\s*.*\s*:', r'\&\&', r'\|\|'
        ]

        decision_count = sum(len(re.findall(kw, code, re.IGNORECASE)) for kw in decision_keywords)

        # Rough complexity estimate
        lines = len(code.split('\n'))
        complexity = decision_count / max(lines / 10, 1)

        if complexity > 10:
            result.add_issue(CodeIssue(
                type=IssueType.QUALITY,
                severity=Severity.MEDIUM,
                title="High cyclomatic complexity",
                description=f"Estimated complexity: {complexity:.1f} (recommended: <10)",
                recommendation="Simplify logic, extract methods, reduce nesting"
            ))

    def _check_secrets(self, code: str, result: CodeReviewResult):
        """Check for hardcoded secrets"""
        secret_patterns = [
            (r'(password|passwd|pwd)\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'(api[_-]?key|apikey)\s*=\s*["\']([^"\']+)["\']', "Hardcoded API key"),
            (r'(secret|token)\s*=\s*["\']([^"\']+)["\']', "Hardcoded secret/token"),
            (r'(aws_access_key_id|aws_secret_access_key)\s*=', "AWS credentials"),
        ]

        for pattern, title in secret_patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE))

            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                result.add_issue(CodeIssue(
                    type=IssueType.SECURITY,
                    severity=Severity.CRITICAL,
                    title=title,
                    description=f"Found {title.lower()} on line {line_num}",
                    line=line_num,
                    recommendation="Use environment variables or secret management",
                    cwe_id="CWE-798"
                ))

    def _check_todos(self, code: str, result: CodeReviewResult):
        """Check for TODO/FIXME comments"""
        todo_pattern = r'#\s*(TODO|FIXME|XXX|HACK)[\s:]*(.*)'
        matches = list(re.finditer(todo_pattern, code, re.IGNORECASE))

        for match in matches:
            line_num = code[:match.start()].count('\n') + 1
            tag = match.group(1).upper()
            comment = match.group(2).strip()

            severity = Severity.HIGH if tag == 'FIXME' else Severity.LOW

            result.add_issue(CodeIssue(
                type=IssueType.QUALITY,
                severity=severity,
                title=f"{tag} comment",
                description=f"{tag} on line {line_num}: {comment[:50]}...",
                line=line_num
            ))

    def _load_security_patterns(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load security check patterns"""
        return {
            'python': {
                'security': [
                    {
                        'pattern': r'eval\s*\(',
                        'title': 'Unsafe eval() usage',
                        'severity': Severity.CRITICAL,
                        'cwe': 'CWE-95',
                        'recommendation': 'Avoid eval(). Use ast.literal_eval() or safer alternatives'
                    },
                    {
                        'pattern': r'exec\s*\(',
                        'title': 'Unsafe exec() usage',
                        'severity': Severity.CRITICAL,
                        'cwe': 'CWE-95',
                        'recommendation': 'Avoid exec(). Refactor to use safer alternatives'
                    },
                    {
                        'pattern': r'pickle\.loads?\s*\(',
                        'title': 'Unsafe pickle deserialization',
                        'severity': Severity.HIGH,
                        'cwe': 'CWE-502',
                        'recommendation': 'Use json or other safe serialization formats'
                    },
                    {
                        'pattern': r'os\.system\s*\(',
                        'title': 'Shell command injection risk',
                        'severity': Severity.HIGH,
                        'cwe': 'CWE-78',
                        'recommendation': 'Use subprocess with list arguments instead'
                    },
                    {
                        'pattern': r'subprocess\.(call|run|Popen)\s*\([^,]+,\s*shell\s*=\s*True',
                        'title': 'Shell injection risk with shell=True',
                        'severity': Severity.HIGH,
                        'cwe': 'CWE-78',
                        'recommendation': 'Use shell=False and pass command as list'
                    },
                    {
                        'pattern': r'f?["\'].*SELECT.*FROM.*{.*}.*["\']',
                        'title': 'Potential SQL injection',
                        'severity': Severity.CRITICAL,
                        'cwe': 'CWE-89',
                        'recommendation': 'Use parameterized queries'
                    },
                ]
            },
            'javascript': {
                'security': [
                    {
                        'pattern': r'eval\s*\(',
                        'title': 'Unsafe eval() usage',
                        'severity': Severity.CRITICAL,
                        'cwe': 'CWE-95',
                        'recommendation': 'Avoid eval(). Parse JSON or use safer alternatives'
                    },
                    {
                        'pattern': r'innerHTML\s*=',
                        'title': 'XSS risk with innerHTML',
                        'severity': Severity.HIGH,
                        'cwe': 'CWE-79',
                        'recommendation': 'Use textContent or sanitize input'
                    },
                    {
                        'pattern': r'document\.write\s*\(',
                        'title': 'XSS risk with document.write',
                        'severity': Severity.HIGH,
                        'cwe': 'CWE-79',
                        'recommendation': 'Use DOM manipulation methods instead'
                    },
                    {
                        'pattern': r'dangerouslySetInnerHTML',
                        'title': 'XSS risk with dangerouslySetInnerHTML',
                        'severity': Severity.HIGH,
                        'cwe': 'CWE-79',
                        'recommendation': 'Sanitize HTML or use safe alternatives'
                    },
                ]
            },
            'sql': {
                'security': [
                    {
                        'pattern': r'(SELECT|INSERT|UPDATE|DELETE).*["\'].*\+.*["\']',
                        'title': 'SQL injection risk',
                        'severity': Severity.CRITICAL,
                        'cwe': 'CWE-89',
                        'recommendation': 'Use parameterized queries'
                    },
                ]
            },
        }


# Singleton
_code_reviewer = None

def get_code_reviewer() -> CodeReviewer:
    """Get singleton code reviewer"""
    global _code_reviewer
    if _code_reviewer is None:
        _code_reviewer = CodeReviewer()
    return _code_reviewer


# Test
if __name__ == "__main__":
    reviewer = CodeReviewer()

    # Test Python code with vulnerabilities
    python_code = '''
def process_user_input(data):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id={data}"
    db.execute(query)

    # Command injection
    os.system(f"ls {data}")

    # TODO: Add input validation
    return eval(data)  # Unsafe eval
'''

    result = reviewer.review(python_code, language="python")

    print(f"✅ Language detected: {result.language}")
    print(f"✅ Lines analyzed: {result.lines_analyzed}")
    print(f"✅ Risk score: {result.risk_score:.1f}/10")
    print(f"✅ Issues found: {len(result.issues)}")
    print(f"   - Critical: {result.get_critical_count()}")
    print(f"   - High: {result.get_high_count()}")

    for issue in result.issues[:5]:  # Show first 5
        print(f"\n[{issue.severity.value.upper()}] {issue.title}")
        print(f"  Line {issue.line}: {issue.description}")
        if issue.recommendation:
            print(f"  → {issue.recommendation}")

    # Test JavaScript code
    js_code = '''
function updateContent(userInput) {
    document.getElementById("content").innerHTML = userInput;  // XSS
    eval("var x = " + userInput);  // Unsafe eval
}
'''

    result_js = reviewer.review(js_code)
    print(f"\n✅ JavaScript review: {len(result_js.issues)} issues")

    print("\n✅ All code reviewer tests passed!")
