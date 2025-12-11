"""
Agentic Protocol - Shared Types
================================

Shared enums and data types for the agentic module.
Centralizes types to avoid circular imports between modules.

Author: Claude Code
Date: 2025-12-09
"""

from enum import Enum


class AgentCapability(Enum):
    """Agent specialization capabilities."""
    # Research & Analysis
    RESEARCH = "research"           # Deep exploration, multi-hop reasoning
    ANALYSIS = "analysis"           # Data analysis, pattern finding
    VERIFICATION = "verification"   # Fact checking, claim validation
    DATA_ANALYSIS = "data_analysis" # Statistical data analysis

    # Generation & Synthesis
    SUMMARIZATION = "summarization" # Compress and summarize content
    GENERATION = "generation"       # Create new content
    SYNTHESIS = "synthesis"         # Combine multiple sources
    CONTENT_CREATION = "content_creation"  # Create content

    # Code & Technical
    CODE_REVIEW = "code_review"     # Review and analyze code
    CODE_GENERATION = "code_generation"  # Write code
    DEBUGGING = "debugging"         # Find and fix bugs

    # Planning & Coordination
    PLANNING = "planning"           # Create plans and strategies
    COORDINATION = "coordination"   # Orchestrate multi-agent workflows
    DECISION = "decision"           # Make decisions, select actions

    # Reasoning
    REASONING = "reasoning"         # General reasoning capability

    # Domain-Specific
    MATH = "math"                   # Mathematical reasoning
    WRITING = "writing"            # Creative and technical writing
    QA = "qa"                      # Quality assurance

    # General
    GENERAL = "general"            # General-purpose agent


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"             # Ask agent to perform task
    RESPONSE = "response"           # Response to a request
    CONTEXT = "context"             # Share context/knowledge
    STATUS = "status"               # Status update
    ERROR = "error"                 # Error notification
    BROADCAST = "broadcast"         # Message to all agents
    HANDOFF = "handoff"             # Transfer responsibility
    QUERY = "query"                 # Simple query
    RESULT = "result"               # Task result
    DELEGATE = "delegate"           # Delegate task to another agent


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


class AgentStatus(Enum):
    """Agent operational status."""
    IDLE = "idle"                   # Ready for work
    BUSY = "busy"                   # Processing task
    WAITING = "waiting"             # Waiting for input
    ERROR = "error"                 # Error state
    OFFLINE = "offline"             # Not available


class SelectionStrategy(Enum):
    """Agent selection strategies for expert routing."""
    FASTEST = "fastest"       # Prioritize lowest latency
    RELIABLE = "reliable"     # Prioritize highest success rate
    BALANCED = "balanced"     # Balance speed and reliability
    THOMPSON = "thompson"     # Pure Thompson Sampling (exploration)
    RANDOM = "random"         # Random selection


class QueryComplexity(Enum):
    """Query complexity levels (aligned with routing.query_classifier)."""
    TRIVIAL = "trivial"      # <10ms: "hi", "thanks"
    SIMPLE = "simple"        # <50ms: "what is X?"
    MODERATE = "moderate"    # <100ms: "explain X"
    COMPLEX = "complex"      # <150ms: "compare X and Y"
    RESEARCH = "research"    # No limit: "analyze all tradeoffs"


class EnsembleStrategy(Enum):
    """Aggregation strategies for ensemble decisions."""
    MAJORITY_VOTE = "majority"         # Most common answer wins
    CONFIDENCE_WEIGHTED = "weighted"   # Average weighted by confidence
    BEST_CONFIDENCE = "best"           # Take highest-confidence result
    UNANIMOUS = "unanimous"            # All must agree (fallback to weighted)


__all__ = [
    "AgentCapability",
    "MessageType",
    "MessagePriority",
    "AgentStatus",
    "SelectionStrategy",
    "QueryComplexity",
    "EnsembleStrategy",
]
