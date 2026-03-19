"""
Security Screen — Prevent sensitive data from entering memory store (borrowed from Repomix).

Scans content against a set of regex patterns for common secrets (API keys,
tokens, private keys, SSNs). Blocks storage if any pattern matches.

Integration: optional `security_screen` parameter on LiteMemoryBus.__init__.
"""

import re
from collections.abc import Callable

__all__ = [
    "SecurityScreenError",
    "DEFAULT_PATTERNS",
    "default_security_screen",
    "create_security_screen",
]

# Patterns: (name, regex, description)
DEFAULT_PATTERNS: list[tuple[str, str, str]] = [
    ("aws_key", r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    ("openai_key", r"sk-[a-zA-Z0-9]{20,}", "OpenAI/Anthropic API Key"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9_]{36,}", "GitHub Token"),
    ("generic_secret", r"(?i)(password|secret|token)\s*[=:]\s*['\"][^'\"]{8,}", "Generic Secret"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "US Social Security Number"),
    ("private_key", r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----", "Private Key"),
]

# Pre-compiled for performance (called on every store())
_COMPILED_DEFAULT = [(name, re.compile(pattern), desc) for name, pattern, desc in DEFAULT_PATTERNS]


class SecurityScreenError(ValueError):
    """Raised when content fails security screening."""

    def __init__(self, pattern_name: str, description: str):
        super().__init__(f"Security screen blocked: {description} ({pattern_name})")
        self.pattern_name = pattern_name
        self.description = description


def default_security_screen(content: str) -> bool:
    """Returns True if content is safe, False if it contains sensitive patterns."""
    for name, compiled, desc in _COMPILED_DEFAULT:
        if compiled.search(content):
            return False
    return True


def create_security_screen(
    extra_patterns: list[tuple[str, str, str]] | None = None,
    disabled_patterns: list[str] | None = None,
) -> Callable[[str], bool]:
    """Factory for customized security screens.

    Args:
        extra_patterns: Additional (name, regex, description) tuples to check.
        disabled_patterns: Pattern names to skip from DEFAULT_PATTERNS.

    Returns:
        A callable(content) → bool screen function.
    """
    patterns = [
        (name, regex, desc) for name, regex, desc in DEFAULT_PATTERNS
        if name not in (disabled_patterns or [])
    ]
    if extra_patterns:
        patterns.extend(extra_patterns)

    compiled = [(name, re.compile(regex), desc) for name, regex, desc in patterns]

    def _screen(content: str) -> bool:
        for name, pattern, desc in compiled:
            if pattern.search(content):
                return False
        return True

    return _screen
