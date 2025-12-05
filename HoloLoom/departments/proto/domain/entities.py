"""Core domain entities for Proto code agent.

Entities:
    - IntentType: Enumeration of user intent categories
    - CodeContext: Context for code-related operations
    - ProtoIntent: User intent with context
    - ProtoAction: Action to be executed
    - ProtoResponse: Response from Proto

Implemented: 2025-12-01
Status: Production ready
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class IntentType(Enum):
    """User intent categories for code-related tasks.

    Categories:
        ASK: Ask a question about code
        EXPLAIN: Explain code or concepts
        REVIEW: Review code quality
        REFACTOR: Suggest refactoring
        TEST: Generate tests
        DEBUG: Debug an issue
        GENERATE: Generate code
        SECURITY: Security analysis
        DOCUMENT: Generate documentation
        UNKNOWN: Unknown or unclassified intent
    """
    ASK = "ask"
    EXPLAIN = "explain"
    REVIEW = "review"
    REFACTOR = "refactor"
    TEST = "test"
    DEBUG = "debug"
    GENERATE = "generate"
    SECURITY = "security"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass
class CodeContext:
    """Context for code-related operations.

    Attributes:
        language: Programming language (python, typescript, go, etc.)
        file_path: Absolute path to file
        content: File content
        selection: Selected text (if applicable)
        cursor_position: (line, column) of cursor
        working_directory: Current working directory
        project_root: Project root directory
        git_branch: Current git branch
        metadata: Additional context metadata
    """
    language: Optional[str] = None
    file_path: Optional[str] = None
    content: Optional[str] = None
    selection: Optional[str] = None
    cursor_position: Optional[Tuple[int, int]] = None
    working_directory: Optional[str] = None
    project_root: Optional[str] = None
    git_branch: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_content(self) -> bool:
        """Check if context has file content."""
        return self.content is not None and len(self.content) > 0

    def has_selection(self) -> bool:
        """Check if context has selected text."""
        return self.selection is not None and len(self.selection) > 0


@dataclass
class ProtoIntent:
    """User intent with context.

    Represents a user request or question directed at Proto, including
    the intent type, query text, and relevant code context.

    Attributes:
        intent_id: Unique intent identifier
        intent_type: Category of intent
        query: User's question or request
        code_context: Code context (if applicable)
        created_at: Timestamp when intent was created
        metadata: Additional intent metadata
    """
    intent_id: UUID = field(default_factory=uuid4)
    intent_type: IntentType = IntentType.UNKNOWN
    query: str = ""
    code_context: Optional[CodeContext] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_code_related(self) -> bool:
        """Check if intent is code-related (has context)."""
        return self.code_context is not None

    def get_summary(self) -> str:
        """Get short summary of intent."""
        return f"{self.intent_type.value}: {self.query[:50]}..."


@dataclass
class ProtoAction:
    """Action to be executed by Proto.

    Represents an executable action derived from an intent, such as
    running analysis, generating code, or fetching documentation.

    Attributes:
        action_id: Unique action identifier
        action_type: Type of action (ability name)
        parameters: Action parameters
        intent_id: ID of intent that triggered this action
        created_at: Timestamp when action was created
        executed_at: Timestamp when action was executed
        result: Result of action execution
        error: Error message if action failed
        confidence: Confidence score (0.0-1.0)
    """
    action_id: UUID = field(default_factory=uuid4)
    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    intent_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    confidence: float = 0.0

    def is_executed(self) -> bool:
        """Check if action has been executed."""
        return self.executed_at is not None

    def is_success(self) -> bool:
        """Check if action executed successfully."""
        return self.is_executed() and self.error is None

    def get_duration_ms(self) -> Optional[float]:
        """Get execution duration in milliseconds."""
        if not self.is_executed():
            return None
        return (self.executed_at - self.created_at).total_seconds() * 1000


@dataclass
class ProtoResponse:
    """Response from Proto.

    Contains the final response to a user intent, including the main
    content, actions taken, confidence score, and suggestions.

    Attributes:
        response_id: Unique response identifier
        content: Main response content
        actions_taken: List of actions executed
        confidence: Confidence score (0.0-1.0)
        sources: List of source references
        suggestions: List of suggested follow-up actions
        created_at: Timestamp when response was created
        metadata: Additional response metadata
    """
    response_id: UUID = field(default_factory=uuid4)
    content: str = ""
    actions_taken: List[ProtoAction] = field(default_factory=list)
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if response confidence exceeds threshold."""
        return self.confidence >= threshold

    def has_actions(self) -> bool:
        """Check if response includes executed actions."""
        return len(self.actions_taken) > 0

    def get_successful_actions(self) -> List[ProtoAction]:
        """Get list of successfully executed actions."""
        return [a for a in self.actions_taken if a.is_success()]

    def get_failed_actions(self) -> List[ProtoAction]:
        """Get list of failed actions."""
        return [a for a in self.actions_taken if not a.is_success()]
