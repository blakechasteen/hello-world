"""
Security Scanning Ability - Tier 2 Plugin

Provides built-in security scanning capabilities for detecting common vulnerabilities
in code without requiring external tools.

Status: Production Ready
Location: HoloLoom/departments/proto/abilities/core/security_scan.py
Version: 1.0.0
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..protocol import (
    AbilityManifest,
    AbilityResult,
    AbilityTier,
    AbilityTrustLevel,
    BaseAbility,
)


class AbilityError(Exception):
    """Custom exception for ability errors."""
    pass


class Severity(str, Enum):
    """Security finding severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScanCategory(str, Enum):
    """Security scan categories."""
    SECRETS = "secrets"
    INJECTION = "injection"
    CRYPTO = "crypto"
    DANGEROUS_FUNCTIONS = "dangerous_functions"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_DESERIALIZATION = "insecure_deserialization"


@dataclass
class SecurityFinding:
    """A security vulnerability finding."""
    severity: Severity
    category: ScanCategory
    file_path: str
    line_number: int
    match: str
    description: str
    context: str | None = None
    fix_suggestion: str | None = None


class SecurityScanAbility(BaseAbility):
    """
    Security scanning ability for detecting common vulnerabilities.

    Provides built-in pattern matching for:
    - Hardcoded secrets (API keys, passwords, tokens)
    - Injection vulnerabilities (SQL, command, LDAP, XPath)
    - Weak cryptography
    - Dangerous functions (eval, exec, pickle)
    - Path traversal
    - Insecure deserialization

    Examples:
        # Scan a single file
        result = await ability.execute({
            "target": "app.py",
            "scan_types": ["secrets", "injection"],
            "severity_threshold": "medium"
        })

        # Scan a directory with exclusions
        result = await ability.execute({
            "target": "src/",
            "exclude_patterns": ["*test*", "*.pyc"],
            "include_line_context": True
        })
    """

    # Comprehensive regex patterns for security scanning
    PATTERNS = {
        # Hardcoded Secrets
        ScanCategory.SECRETS: {
            "aws_access_key": (
                r'(?i)(aws_access_key_id|aws_secret_access_key)\s*[=:]\s*["\']?([A-Z0-9]{20})["\']?',
                Severity.CRITICAL,
                "Hardcoded AWS access key detected",
                "Store AWS credentials in environment variables or AWS Secrets Manager"
            ),
            "private_key": (
                r'-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----',
                Severity.CRITICAL,
                "Private key found in code",
                "Store private keys in secure key management systems, never in code"
            ),
            "password_assignment": (
                r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{4,})["\']',
                Severity.HIGH,
                "Hardcoded password detected",
                "Use environment variables or secure credential management"
            ),
            "api_key": (
                r'(?i)(api[_-]?key|apikey|api[_-]?secret)\s*[=:]\s*["\']([a-zA-Z0-9_\-]{16,})["\']',
                Severity.HIGH,
                "Hardcoded API key detected",
                "Use environment variables or secure configuration management"
            ),
            "bearer_token": (
                r'(?i)(bearer|token|auth[_-]?token)\s*[=:]\s*["\']([a-zA-Z0-9_\-\.]{20,})["\']',
                Severity.HIGH,
                "Hardcoded authentication token detected",
                "Store tokens securely using environment variables or secret management"
            ),
            "github_token": (
                r'(?i)gh[pousr]_[a-zA-Z0-9]{36}',
                Severity.CRITICAL,
                "GitHub personal access token detected",
                "Revoke the token immediately and use GitHub Secrets"
            ),
            "slack_token": (
                r'xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[a-zA-Z0-9]{24,}',
                Severity.CRITICAL,
                "Slack token detected",
                "Revoke the token and use environment variables"
            ),
        },

        # Injection Vulnerabilities
        ScanCategory.INJECTION: {
            "sql_injection": (
                r'(?i)(execute|exec|query)\s*\(\s*["\'].*%s.*["\'].*%',
                Severity.CRITICAL,
                "Potential SQL injection vulnerability",
                "Use parameterized queries or prepared statements"
            ),
            "sql_format": (
                r'(?i)(execute|exec|query)\s*\(\s*.*\.format\(',
                Severity.CRITICAL,
                "SQL query using string formatting (injection risk)",
                "Use parameterized queries instead of string formatting"
            ),
            "sql_concatenation": (
                r'(?i)(SELECT|INSERT|UPDATE|DELETE).*\+.*["\']',
                Severity.HIGH,
                "SQL query using string concatenation (injection risk)",
                "Use parameterized queries or ORM methods"
            ),
            "command_injection": (
                r'(?i)(os\.system|subprocess\.call|subprocess\.run).*shell\s*=\s*True',
                Severity.CRITICAL,
                "Command injection vulnerability (shell=True)",
                "Use shell=False and pass arguments as list, validate all inputs"
            ),
            "eval_user_input": (
                r'(?i)eval\s*\(\s*(request|input|raw_input)',
                Severity.CRITICAL,
                "eval() called on user input",
                "Never use eval() on user input; use safer alternatives like ast.literal_eval()"
            ),
            "ldap_injection": (
                r'(?i)ldap.*search.*\(\s*.*\+.*["\']',
                Severity.HIGH,
                "Potential LDAP injection vulnerability",
                "Sanitize and escape user input before LDAP queries"
            ),
            "xpath_injection": (
                r'(?i)xpath.*\(\s*.*\+.*["\']',
                Severity.HIGH,
                "Potential XPath injection vulnerability",
                "Use parameterized XPath queries"
            ),
        },

        # Weak Cryptography
        ScanCategory.CRYPTO: {
            "weak_hash": (
                r'(?i)(md5|sha1)\s*\(',
                Severity.MEDIUM,
                "Weak cryptographic hash algorithm (MD5/SHA1)",
                "Use SHA-256 or stronger algorithms for security-sensitive operations"
            ),
            "weak_cipher": (
                r'(?i)(DES|RC4|Blowfish)\.new',
                Severity.HIGH,
                "Weak encryption algorithm detected",
                "Use AES-256 or other modern encryption algorithms"
            ),
            "insecure_random": (
                r'(?i)random\.(random|randint|choice)',
                Severity.MEDIUM,
                "Insecure random number generation",
                "Use secrets module for cryptographic purposes"
            ),
            "hardcoded_key": (
                r'(?i)(encryption_key|secret_key|aes_key)\s*=\s*["\']([a-zA-Z0-9+/=]{16,})["\']',
                Severity.CRITICAL,
                "Hardcoded encryption key",
                "Generate keys dynamically and store securely"
            ),
        },

        # Dangerous Functions
        ScanCategory.DANGEROUS_FUNCTIONS: {
            "eval": (
                r'\beval\s*\(',
                Severity.HIGH,
                "Use of eval() function",
                "Avoid eval(); use safer alternatives like ast.literal_eval() for data"
            ),
            "exec": (
                r'\bexec\s*\(',
                Severity.HIGH,
                "Use of exec() function",
                "Avoid exec(); refactor code to eliminate dynamic execution"
            ),
            "pickle_loads": (
                r'pickle\.loads?\s*\(',
                Severity.HIGH,
                "Insecure deserialization with pickle",
                "Use JSON or other safe serialization formats for untrusted data"
            ),
            "yaml_unsafe": (
                r'yaml\.load\s*\(\s*[^,)]+\s*\)',
                Severity.HIGH,
                "Unsafe YAML loading (allows arbitrary code execution)",
                "Use yaml.safe_load() instead of yaml.load()"
            ),
            "input_python2": (
                r'\binput\s*\(',
                Severity.MEDIUM,
                "Use of input() (unsafe in Python 2)",
                "Use raw_input() in Python 2, or ensure Python 3"
            ),
        },

        # Path Traversal
        ScanCategory.PATH_TRAVERSAL: {
            "path_traversal_open": (
                r'open\s*\(\s*[^)]*\+',
                Severity.HIGH,
                "Potential path traversal vulnerability",
                "Validate and sanitize file paths, use Path.resolve() to prevent traversal"
            ),
            "path_traversal_join": (
                r'os\.path\.join\s*\([^)]*user|request',
                Severity.HIGH,
                "User input in path join (traversal risk)",
                "Validate that resulting path is within expected directory"
            ),
        },

        # Insecure Deserialization
        ScanCategory.INSECURE_DESERIALIZATION: {
            "marshal_loads": (
                r'marshal\.loads?\s*\(',
                Severity.HIGH,
                "Insecure deserialization with marshal",
                "Avoid marshal for untrusted data; use JSON or msgpack"
            ),
            "shelve": (
                r'shelve\.open\s*\(',
                Severity.MEDIUM,
                "Shelve uses pickle internally (insecure)",
                "Use database or JSON for untrusted data storage"
            ),
        },
    }

    @property
    def manifest(self) -> AbilityManifest:
        """Return ability manifest."""
        return AbilityManifest(
            name="security_scan",
            version="1.0.0",
            tier=AbilityTier.PLUGIN,
            trust_level=AbilityTrustLevel.VERIFIED,
            description="Built-in security scanning for detecting common vulnerabilities",
            permissions=["read_file"],
            parameters={
                "target": {
                    "type": "string",
                    "required": True,
                    "description": "File or directory path to scan"
                },
                "scan_types": {
                    "type": "array",
                    "required": False,
                    "description": "Scan categories to run (default: all)",
                    "default": ["secrets", "injection", "crypto", "dangerous_functions",
                               "path_traversal", "insecure_deserialization"]
                },
                "severity_threshold": {
                    "type": "string",
                    "required": False,
                    "description": "Minimum severity to report (low, medium, high, critical)",
                    "default": "low"
                },
                "exclude_patterns": {
                    "type": "array",
                    "required": False,
                    "description": "File patterns to exclude from scanning",
                    "default": []
                },
                "include_line_context": {
                    "type": "boolean",
                    "required": False,
                    "description": "Include surrounding code lines in findings",
                    "default": True
                }
            },
            tags=["security", "vulnerability", "audit", "scan", "analysis"]
        )

    def _should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned."""
        # Only scan text files (common code extensions)
        code_extensions = {
            '.py', '.js', '.ts', '.java', '.go', '.rb', '.php', '.cs',
            '.c', '.cpp', '.h', '.hpp', '.rs', '.swift', '.kt', '.scala',
            '.sh', '.bash', '.yml', '.yaml', '.json', '.xml', '.sql',
            '.env', '.config', '.ini', '.properties'
        }

        return file_path.suffix.lower() in code_extensions

    def _get_line_context(self, lines: list[str], line_num: int,
                         context_lines: int = 2) -> str:
        """Get surrounding lines for context."""
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines + 1)

        context = []
        for i in range(start, end):
            prefix = "→ " if i == line_num else "  "
            context.append(f"{prefix}{i+1:4d} | {lines[i].rstrip()}")

        return "\n".join(context)

    def _scan_file(
        self,
        file_path: Path,
        scan_types: set[ScanCategory],
        severity_threshold: Severity,
        include_context: bool
    ) -> list[SecurityFinding]:
        """Scan a single file for security issues."""
        findings = []

        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                content = ''.join(lines)

            # Scan each enabled category
            for category in scan_types:
                if category not in self.PATTERNS:
                    continue

                patterns = self.PATTERNS[category]

                # Check each pattern in the category
                for pattern_name, (regex, severity, description, fix) in patterns.items():
                    # Skip if below severity threshold
                    severity_levels = {
                        Severity.LOW: 0,
                        Severity.MEDIUM: 1,
                        Severity.HIGH: 2,
                        Severity.CRITICAL: 3
                    }
                    if severity_levels[severity] < severity_levels[severity_threshold]:
                        continue

                    # Find all matches
                    for match in re.finditer(regex, content, re.MULTILINE):
                        # Calculate line number
                        line_num = content[:match.start()].count('\n')

                        # Get context if requested
                        context = None
                        if include_context:
                            context = self._get_line_context(lines, line_num)

                        findings.append(SecurityFinding(
                            severity=severity,
                            category=category,
                            file_path=str(file_path),
                            line_number=line_num + 1,
                            match=match.group(0)[:100],  # Truncate long matches
                            description=description,
                            context=context,
                            fix_suggestion=fix
                        ))

        except Exception as e:
            # Log error but continue scanning other files
            self.logger.warning(f"Error scanning {file_path}: {e}")

        return findings

    def _matches_pattern(self, path: Path, patterns: list[str]) -> bool:
        """Check if path matches any exclusion pattern."""
        path_str = str(path)
        for pattern in patterns:
            if '*' in pattern:
                # Simple glob matching
                pattern_regex = pattern.replace('*', '.*')
                if re.search(pattern_regex, path_str):
                    return True
            else:
                if pattern in path_str:
                    return True
        return False

    async def preflight(self, params: dict[str, Any]) -> None:
        """
        Validate parameters before execution.

        Checks:
        - Target path exists and is accessible
        - Scan types are valid
        - Severity threshold is valid
        """
        target = params.get("target")
        if not target:
            raise AbilityError("Target path required")

        target_path = Path(target)
        if not target_path.exists():
            raise AbilityError(f"Target path does not exist: {target}")

        # Validate scan types
        scan_types = params.get("scan_types", [])
        if scan_types:
            valid_types = {cat.value for cat in ScanCategory}
            invalid = [st for st in scan_types if st not in valid_types]
            if invalid:
                raise AbilityError(
                    f"Invalid scan types: {invalid}. "
                    f"Valid: {sorted(valid_types)}"
                )

        # Validate severity threshold
        threshold = params.get("severity_threshold", "low")
        valid_severities = {sev.value for sev in Severity}
        if threshold not in valid_severities:
            raise AbilityError(
                f"Invalid severity threshold: {threshold}. "
                f"Valid: {sorted(valid_severities)}"
            )

    async def execute(self, params: dict[str, Any]) -> AbilityResult:
        """
        Execute security scan.

        Scans target file or directory for security vulnerabilities using
        built-in pattern matching.

        Returns:
            AbilityResult with findings, summary, and recommendations
        """
        target = Path(params["target"])

        # Parse parameters
        scan_types_list = params.get("scan_types", [cat.value for cat in ScanCategory])
        scan_types = {ScanCategory(st) for st in scan_types_list}

        severity_threshold = Severity(params.get("severity_threshold", "low"))
        exclude_patterns = params.get("exclude_patterns", [])
        include_context = params.get("include_line_context", True)

        # Collect files to scan
        files_to_scan = []
        if target.is_file():
            files_to_scan = [target]
        else:
            for root, dirs, files in os.walk(target):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file

                    # Skip if matches exclusion pattern
                    if self._matches_pattern(file_path, exclude_patterns):
                        continue

                    # Only scan code files
                    if self._should_scan_file(file_path):
                        files_to_scan.append(file_path)

        self.logger.info(f"Scanning {len(files_to_scan)} files...")

        # Scan all files
        all_findings = []
        for file_path in files_to_scan:
            findings = self._scan_file(
                file_path,
                scan_types,
                severity_threshold,
                include_context
            )
            all_findings.extend(findings)

        # Generate summary
        summary = {
            "total_findings": len(all_findings),
            "files_scanned": len(files_to_scan),
            "by_severity": {
                "critical": len([f for f in all_findings if f.severity == Severity.CRITICAL]),
                "high": len([f for f in all_findings if f.severity == Severity.HIGH]),
                "medium": len([f for f in all_findings if f.severity == Severity.MEDIUM]),
                "low": len([f for f in all_findings if f.severity == Severity.LOW]),
            },
            "by_category": {}
        }

        for category in ScanCategory:
            count = len([f for f in all_findings if f.category == category])
            if count > 0:
                summary["by_category"][category.value] = count

        # Generate recommendations
        recommendations = []
        if summary["by_severity"]["critical"] > 0:
            recommendations.append(
                "⚠️ CRITICAL: Address critical vulnerabilities immediately "
                "(secrets, SQL injection, command injection)"
            )
        if summary["by_severity"]["high"] > 0:
            recommendations.append(
                "⚠️ HIGH: Review and fix high-severity issues "
                "(dangerous functions, weak crypto)"
            )
        if any(f.category == ScanCategory.SECRETS for f in all_findings):
            recommendations.append(
                "🔑 Rotate all exposed credentials immediately and audit access logs"
            )
        if len(all_findings) == 0:
            recommendations.append(
                "✅ No security issues detected in scan"
            )

        # Format findings for output
        findings_output = []
        for finding in all_findings:
            finding_dict = {
                "severity": finding.severity.value,
                "category": finding.category.value,
                "file": finding.file_path,
                "line": finding.line_number,
                "match": finding.match,
                "description": finding.description,
                "fix_suggestion": finding.fix_suggestion
            }
            if finding.context:
                finding_dict["context"] = finding.context
            findings_output.append(finding_dict)

        return AbilityResult(
            success=True,
            data={
                "findings": findings_output,
                "summary": summary,
                "recommendations": recommendations,
                "scan_config": {
                    "target": str(target),
                    "scan_types": [st.value for st in scan_types],
                    "severity_threshold": severity_threshold.value,
                    "files_scanned": len(files_to_scan)
                }
            },
            message=f"Scan complete: {len(all_findings)} findings in {len(files_to_scan)} files"
        )

    async def verify(self, result: AbilityResult) -> bool:
        """
        Verify scan results are valid.

        Checks:
        - Result has findings list
        - Summary has required fields
        - Findings have required attributes
        """
        if not result.success:
            return True  # Errors are valid results

        data = result.data

        # Check required fields
        if "findings" not in data:
            self.logger.error("Missing findings in result")
            return False

        if "summary" not in data:
            self.logger.error("Missing summary in result")
            return False

        summary = data["summary"]
        required_summary_fields = ["total_findings", "files_scanned", "by_severity"]
        for field in required_summary_fields:
            if field not in summary:
                self.logger.error(f"Missing {field} in summary")
                return False

        # Validate findings structure
        for finding in data["findings"]:
            required_fields = ["severity", "category", "file", "line",
                             "description", "fix_suggestion"]
            for field in required_fields:
                if field not in finding:
                    self.logger.error(f"Missing {field} in finding")
                    return False

        return True

    async def cleanup(self) -> None:
        """Cleanup after scan (no resources to clean)."""
        pass
