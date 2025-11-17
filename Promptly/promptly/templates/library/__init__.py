"""
Built-in Template Library

Categories:
- base: Base templates for common patterns
- roles: Role-based message templates (system, user, assistant)
- domains: Domain-specific templates (summarization, QA, coding, etc.)
- patterns: Advanced patterns (few-shot, chain-of-thought, ReAct)
"""

from pathlib import Path

LIBRARY_PATH = Path(__file__).parent


def get_library_path() -> str:
    """Get the absolute path to the template library"""
    return str(LIBRARY_PATH)


__all__ = ['LIBRARY_PATH', 'get_library_path']
