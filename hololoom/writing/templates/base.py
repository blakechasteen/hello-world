"""
Template Base Protocol
======================

Base protocol for content templates.
"""

from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.protocol import MemoryShard, WritingContext


class TemplateType(Enum):
    """Template types."""
    EMAIL = "email"
    REPORT = "report"
    ESSAY = "essay"
    LETTER = "letter"
    MEMO = "memo"


@dataclass
class TemplateContext:
    """Context for template filling."""
    template_type: TemplateType
    variables: Dict[str, Any]  # Template variables (recipient, subject, etc.)
    memories: List[MemoryShard]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TemplateResult:
    """Result of template generation."""
    content: str
    template_type: TemplateType
    variables_used: List[str]
    sections: Dict[str, str]  # Section name -> content
    metadata: Dict[str, Any]


class TemplateProtocol(Protocol):
    """
    Protocol for content templates.

    Templates provide structured generation with predefined sections.
    """

    async def generate(
        self,
        context: TemplateContext
    ) -> TemplateResult:
        """
        Generate content from template.

        Args:
            context: Template context with variables and memories

        Returns:
            Template result with filled content
        """
        ...

    def get_required_variables(self) -> List[str]:
        """
        Get list of required template variables.

        Returns:
            List of required variable names
        """
        ...

    def get_optional_variables(self) -> List[str]:
        """
        Get list of optional template variables.

        Returns:
            List of optional variable names
        """
        ...

    @property
    def template_type(self) -> TemplateType:
        """Template type this template handles."""
        ...
