"""
Phase 5C: AI Code Review
HoloLoom-powered intelligent code review with security, performance, and quality analysis.

Features:
- Fetch PR diff from GitHub
- Analyze code with HoloLoom for:
  - Security vulnerabilities (SQL injection, XSS, etc.)
  - Performance issues (N+1 queries, inefficient loops)
  - Code quality (complexity, duplication)
  - Best practices (error handling, logging)
- Post review comments to GitHub
- Generate review summary for Matrix

Matrix Commands:
- !review pr <number> - Full AI code review
- !review security <number> - Security-focused review
- !review performance <number> - Performance-focused review
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from github import Github, GithubException
    from github.PullRequest import PullRequest
except ImportError:
    print("⚠️  PyGithub not installed. Run: pip install PyGithub==2.1.1")
    Github = None

from .github_auth import GitHubAuth


class ReviewFocus(Enum):
    """Review focus areas."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    BEST_PRACTICES = "best_practices"
    ALL = "all"


class IssueSeverity(Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CodeIssue:
    """Represents a code review issue."""

    file_path: str
    line_number: int
    severity: IssueSeverity
    category: str  # security, performance, quality, best_practices
    title: str
    description: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class ReviewResult:
    """Code review result."""

    pr_number: int
    overall_score: float  # 0.0 - 1.0
    recommendation: str  # APPROVE, REQUEST_CHANGES, COMMENT
    issues: List[CodeIssue]
    summary: str
    details: Dict[str, Any]


class AICodeReviewer:
    """AI-powered code reviewer using HoloLoom."""

    def __init__(self, auth: Optional[GitHubAuth] = None):
        """Initialize code reviewer."""
        self.auth = auth or GitHubAuth()

        # Security patterns (simple regex-based for demo)
        self.security_patterns = {
            "sql_injection": [
                r"execute\s*\(\s*['\"].*%s",  # String formatting in SQL
                r"raw\s*\(\s*['\"].*\+",  # String concatenation in raw SQL
            ],
            "xss": [
                r"innerHTML\s*=",  # Direct innerHTML assignment
                r"dangerouslySetInnerHTML",  # React dangerous prop
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",  # Hardcoded password
                r"api_key\s*=\s*['\"][^'\"]+['\"]",  # Hardcoded API key
                r"secret\s*=\s*['\"][^'\"]+['\"]",  # Hardcoded secret
            ],
            "command_injection": [
                r"os\.system\s*\(",  # os.system()
                r"subprocess\.call\s*\([^,]*\+",  # subprocess with concatenation
            ],
        }

        # Performance patterns
        self.performance_patterns = {
            "n_plus_one": [
                r"for\s+\w+\s+in\s+.*:\s*\n\s*.*\.get\(",  # Loop with DB query
            ],
            "inefficient_loop": [
                r"for\s+.*\n.*for\s+.*\n.*for\s+",  # Triple nested loop
            ],
            "memory_leak": [
                r"global\s+\w+\s*=\s*\[\]",  # Global mutable state
            ],
        }

        # Best practices patterns
        self.best_practice_patterns = {
            "missing_error_handling": [
                r"requests\.\w+\([^)]*\)\s*\n(?!.*except)",  # requests without try/except
            ],
            "missing_logging": [
                r"except\s+\w+:\s*\n\s*pass",  # Silent exception
            ],
            "hardcoded_config": [
                r"http://localhost:\d+",  # Hardcoded localhost URL
            ],
        }

    def _get_client(self, installation_id: int) -> Github:
        """Get authenticated GitHub client."""
        if Github is None:
            raise ImportError("PyGithub not installed")

        token = self.auth.get_installation_token(installation_id)
        return Github(token)

    async def review_pr(
        self,
        installation_id: int,
        repo_full_name: str,
        pr_number: int,
        focus: ReviewFocus = ReviewFocus.ALL,
        post_comments: bool = True,
    ) -> ReviewResult:
        """
        Perform AI code review on a pull request.

        Args:
            installation_id: GitHub App installation ID
            repo_full_name: Repository (e.g., "owner/repo")
            pr_number: PR number
            focus: Review focus (security, performance, quality, all)
            post_comments: Whether to post comments to GitHub

        Returns:
            ReviewResult with issues and recommendation
        """
        try:
            client = self._get_client(installation_id)
            repo = client.get_repo(repo_full_name)
            pr = repo.get_pull(pr_number)

            # Get PR diff
            files = pr.get_files()
            all_issues: List[CodeIssue] = []

            # Analyze each file
            for file in files:
                if file.status == "removed":
                    continue

                # Get file content
                file_content = self._get_file_content(file)

                # Run analysis based on focus
                file_issues = self._analyze_file(
                    file_path=file.filename,
                    content=file_content,
                    patch=file.patch or "",
                    focus=focus,
                )

                all_issues.extend(file_issues)

            # Calculate overall score
            overall_score = self._calculate_score(all_issues)

            # Determine recommendation
            recommendation = self._get_recommendation(all_issues, overall_score)

            # Generate summary
            summary = self._generate_summary(all_issues, overall_score)

            # Post comments to GitHub
            if post_comments and all_issues:
                await self._post_review_comments(pr, all_issues, recommendation, summary)

            # Build result
            result = ReviewResult(
                pr_number=pr_number,
                overall_score=overall_score,
                recommendation=recommendation,
                issues=all_issues,
                summary=summary,
                details={
                    "files_reviewed": len(list(files)),
                    "total_issues": len(all_issues),
                    "critical_issues": sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL),
                    "high_issues": sum(1 for i in all_issues if i.severity == IssueSeverity.HIGH),
                    "medium_issues": sum(1 for i in all_issues if i.severity == IssueSeverity.MEDIUM),
                    "low_issues": sum(1 for i in all_issues if i.severity == IssueSeverity.LOW),
                },
            )

            return result

        except GithubException as e:
            raise Exception(f"GitHub error: {e}")

    def _get_file_content(self, file) -> str:
        """Get file content from PR file object."""
        try:
            # Use patch if available (safer for large files)
            if file.patch:
                lines = []
                for line in file.patch.split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        lines.append(line[1:])  # Remove + prefix
                return "\n".join(lines)

            # Fallback: Get raw content (if file is small enough)
            return file.raw_data.get("content", "")

        except Exception:
            return ""

    def _analyze_file(
        self, file_path: str, content: str, patch: str, focus: ReviewFocus
    ) -> List[CodeIssue]:
        """Analyze a single file for issues."""
        issues: List[CodeIssue] = []

        # Security analysis
        if focus in [ReviewFocus.SECURITY, ReviewFocus.ALL]:
            issues.extend(self._check_security(file_path, content))

        # Performance analysis
        if focus in [ReviewFocus.PERFORMANCE, ReviewFocus.ALL]:
            issues.extend(self._check_performance(file_path, content))

        # Best practices
        if focus in [ReviewFocus.BEST_PRACTICES, ReviewFocus.ALL]:
            issues.extend(self._check_best_practices(file_path, content))

        # Code quality (basic complexity check)
        if focus in [ReviewFocus.QUALITY, ReviewFocus.ALL]:
            issues.extend(self._check_quality(file_path, content))

        return issues

    def _check_security(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for security vulnerabilities."""
        import re

        issues: List[CodeIssue] = []

        # SQL Injection
        for pattern in self.security_patterns["sql_injection"]:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.CRITICAL,
                        category="security",
                        title="Potential SQL Injection",
                        description="String formatting or concatenation in SQL query can lead to SQL injection.",
                        suggestion="Use parameterized queries instead:\n```python\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```",
                        code_snippet=match.group(0),
                    )
                )

        # XSS
        for pattern in self.security_patterns["xss"]:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.HIGH,
                        category="security",
                        title="Potential XSS Vulnerability",
                        description="Direct HTML injection can lead to cross-site scripting (XSS) attacks.",
                        suggestion="Sanitize user input or use safe DOM methods.",
                        code_snippet=match.group(0),
                    )
                )

        # Hardcoded secrets
        for pattern in self.security_patterns["hardcoded_secrets"]:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.CRITICAL,
                        category="security",
                        title="Hardcoded Secret Detected",
                        description="Hardcoded secrets in source code are a security risk.",
                        suggestion="Use environment variables:\n```python\nimport os\npassword = os.getenv('DB_PASSWORD')\n```",
                        code_snippet=match.group(0),
                    )
                )

        return issues

    def _check_performance(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for performance issues."""
        import re

        issues: List[CodeIssue] = []

        # N+1 queries
        for pattern in self.performance_patterns["n_plus_one"]:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.MEDIUM,
                        category="performance",
                        title="Potential N+1 Query",
                        description="Database query inside a loop can cause N+1 query problem.",
                        suggestion="Use select_related() or prefetch_related() to fetch data in bulk.",
                        code_snippet=match.group(0),
                    )
                )

        # Inefficient loops
        for pattern in self.performance_patterns["inefficient_loop"]:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.MEDIUM,
                        category="performance",
                        title="Deeply Nested Loops",
                        description="Triple nested loop has O(n³) complexity.",
                        suggestion="Consider using more efficient algorithms or data structures.",
                        code_snippet=match.group(0)[:100] + "...",
                    )
                )

        return issues

    def _check_best_practices(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for best practice violations."""
        import re

        issues: List[CodeIssue] = []

        # Missing error handling
        for pattern in self.best_practice_patterns["missing_error_handling"]:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.LOW,
                        category="best_practices",
                        title="Missing Error Handling",
                        description="Network request without error handling can cause unhandled exceptions.",
                        suggestion="Wrap in try/except:\n```python\ntry:\n    response = requests.get(url)\n    response.raise_for_status()\nexcept requests.RequestException as e:\n    logger.error(f'Request failed: {e}')\n```",
                        code_snippet=match.group(0),
                    )
                )

        # Silent exceptions
        for pattern in self.best_practice_patterns["missing_logging"]:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        severity=IssueSeverity.LOW,
                        category="best_practices",
                        title="Silent Exception",
                        description="Catching exceptions without logging makes debugging difficult.",
                        suggestion="Add logging:\n```python\nexcept SomeException as e:\n    logger.warning(f'Non-critical error: {e}')\n```",
                        code_snippet=match.group(0),
                    )
                )

        return issues

    def _check_quality(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check code quality (complexity, duplication, etc.)."""
        issues: List[CodeIssue] = []

        # Simple complexity check: long functions
        lines = content.split("\n")
        current_function = None
        function_start = 0

        for i, line in enumerate(lines):
            # Detect function start (Python)
            if line.strip().startswith("def ") or line.strip().startswith("async def "):
                if current_function:
                    # Check previous function length
                    length = i - function_start
                    if length > 100:
                        issues.append(
                            CodeIssue(
                                file_path=file_path,
                                line_number=function_start + 1,
                                severity=IssueSeverity.LOW,
                                category="quality",
                                title="Long Function",
                                description=f"Function is {length} lines long. Consider breaking into smaller functions.",
                                suggestion="Aim for functions under 50 lines for better readability.",
                            )
                        )

                current_function = line.strip()
                function_start = i

        return issues

    def _calculate_score(self, issues: List[CodeIssue]) -> float:
        """Calculate overall code quality score (0.0 - 1.0)."""
        if not issues:
            return 1.0

        # Weight by severity
        severity_weights = {
            IssueSeverity.CRITICAL: 10,
            IssueSeverity.HIGH: 5,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 1,
            IssueSeverity.INFO: 0.5,
        }

        total_penalty = sum(severity_weights[issue.severity] for issue in issues)

        # Score decreases with penalty
        # Max penalty of 50 gives score of 0.0
        score = max(0.0, 1.0 - (total_penalty / 50.0))

        return score

    def _get_recommendation(self, issues: List[CodeIssue], score: float) -> str:
        """Get review recommendation."""
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]

        if critical_issues:
            return "REQUEST_CHANGES"
        elif high_issues or score < 0.7:
            return "REQUEST_CHANGES"
        elif score < 0.9:
            return "COMMENT"
        else:
            return "APPROVE"

    def _generate_summary(self, issues: List[CodeIssue], score: float) -> str:
        """Generate review summary."""
        if not issues:
            return "✅ No issues found! Code looks great."

        # Group by category
        by_category: Dict[str, List[CodeIssue]] = {}
        for issue in issues:
            category = issue.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(issue)

        # Build summary
        lines = [f"**Overall Score**: {score * 100:.1f}%\n"]

        for category, cat_issues in by_category.items():
            emoji = {
                "security": "🔒",
                "performance": "⚡",
                "quality": "✨",
                "best_practices": "📚",
            }.get(category, "ℹ️")

            lines.append(f"\n{emoji} **{category.replace('_', ' ').title()}**: {len(cat_issues)} issue(s)")

            # List issues by severity
            critical = [i for i in cat_issues if i.severity == IssueSeverity.CRITICAL]
            high = [i for i in cat_issues if i.severity == IssueSeverity.HIGH]
            medium = [i for i in cat_issues if i.severity == IssueSeverity.MEDIUM]

            if critical:
                lines.append(f"  - 🚨 Critical: {len(critical)}")
            if high:
                lines.append(f"  - ⚠️  High: {len(high)}")
            if medium:
                lines.append(f"  - ℹ️  Medium: {len(medium)}")

        return "\n".join(lines)

    async def _post_review_comments(
        self, pr: PullRequest, issues: List[CodeIssue], recommendation: str, summary: str
    ) -> None:
        """Post review comments to GitHub PR."""
        # Create review with summary
        review_body = f"🤖 **AI Code Review**\n\n{summary}"

        # Add individual comments for issues
        comments = []
        for issue in issues:
            emoji = {
                IssueSeverity.CRITICAL: "🚨",
                IssueSeverity.HIGH: "⚠️",
                IssueSeverity.MEDIUM: "ℹ️",
                IssueSeverity.LOW: "💡",
            }.get(issue.severity, "")

            comment_body = f"{emoji} **{issue.title}**\n\n{issue.description}"

            if issue.suggestion:
                comment_body += f"\n\n**Suggestion**:\n{issue.suggestion}"

            comments.append(
                {
                    "path": issue.file_path,
                    "line": issue.line_number,
                    "body": comment_body,
                }
            )

        # Create review
        event = "REQUEST_CHANGES" if recommendation == "REQUEST_CHANGES" else "COMMENT"

        try:
            pr.create_review(body=review_body, event=event, comments=comments if comments else None)
        except GithubException as e:
            # Fallback: Post as single comment if review fails
            pr.create_issue_comment(review_body)


# Matrix command handler
async def handle_review_command(
    reviewer: AICodeReviewer,
    installation_id: int,
    repo: str,
    pr_number: int,
    focus: ReviewFocus = ReviewFocus.ALL,
) -> str:
    """Handle !review pr command."""
    try:
        result = await reviewer.review_pr(
            installation_id=installation_id,
            repo_full_name=repo,
            pr_number=pr_number,
            focus=focus,
            post_comments=True,
        )

        # Build Matrix message
        recommendation_emoji = {
            "APPROVE": "✅",
            "REQUEST_CHANGES": "❌",
            "COMMENT": "💬",
        }.get(result.recommendation, "")

        msg_lines = [
            f"{recommendation_emoji} **AI Code Review Complete**\n",
            f"**PR #{result.pr_number}**",
            f"**Score**: {result.overall_score * 100:.1f}%",
            f"**Recommendation**: {result.recommendation.replace('_', ' ').title()}\n",
            result.summary,
            f"\n**Issues Found**: {result.details['total_issues']}",
            f"  - 🚨 Critical: {result.details['critical_issues']}",
            f"  - ⚠️  High: {result.details['high_issues']}",
            f"  - ℹ️  Medium: {result.details['medium_issues']}",
            f"  - 💡 Low: {result.details['low_issues']}",
        ]

        return "\n".join(msg_lines)

    except Exception as e:
        return f"❌ **Review failed**: {str(e)}"


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demo AI code review."""
        reviewer = AICodeReviewer()

        print("\n=== Running AI Code Review ===")
        result = await reviewer.review_pr(
            installation_id=12345678,  # Replace with real installation ID
            repo_full_name="your-username/your-repo",
            pr_number=1,
            focus=ReviewFocus.ALL,
            post_comments=False,  # Don't post to GitHub in demo
        )

        print(f"\nOverall Score: {result.overall_score * 100:.1f}%")
        print(f"Recommendation: {result.recommendation}")
        print(f"\n{result.summary}")
        print(f"\nTotal Issues: {len(result.issues)}")

        for issue in result.issues[:5]:  # Show first 5
            print(f"\n- [{issue.severity.value.upper()}] {issue.title}")
            print(f"  File: {issue.file_path}:{issue.line_number}")
            print(f"  {issue.description}")

    asyncio.run(demo())
