"""
HoloLoom Shared Types
=====================
Core data contracts used across all modules.
These are the "wool" - the raw data structures that flow through the loom.

Module Philosophy:
- Pure data structures only (dataclasses)
- No business logic
- No dependencies on other modules
- Imported by everyone (orchestrator, policy, all core modules)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ============================================================================
# Type Aliases
# ============================================================================
Vector = List[float]  # Embedding vector representation

# ============================================================================
# Pattern Detection
# ============================================================================
@dataclass
class Motif:
    """
    A detected pattern or motif in query/context with location.
    Used by motif detection modules.
    """
    pattern: str                    # The pattern that was detected
    span: Tuple[int, int]          # Where in text (start, end indices)
    score: float                    # Confidence/relevance score (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Query → Features → Response Pipeline
# ============================================================================
@dataclass
class Query:
    """
    User input query with metadata.
    Entry point to the HoloLoom pipeline.
    """
    text: str                                    # The actual query text
    metadata: Dict[str, Any] = field(default_factory=dict)  # User context, session info, etc.

@dataclass
class Features:
    """
    Extracted features from a query.
    Output of feature extraction phase, input to policy/response.
    
    This is the "warp" - the structured representation that the shuttle works with.
    """
    psi: Vector                     # Ψ - The embedding/vector representation
    motifs: List[Motif]            # Detected patterns/motifs in the query
    metadata: Dict[str, Any] = field(default_factory=dict)  # Feature extraction metadata

@dataclass
class Context:
    """
    Context information accompanying a response.
    Provides provenance and confidence metrics.
    """
    sources: List[str]                           # Where information came from
    confidence: float                            # Overall confidence (0.0 to 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

@dataclass
class Response:
    """
    Final output response with context.
    End product of the HoloLoom pipeline.
    
    This is the "fabric" - the woven result of the loom's work.
    """
    text: str                # The response content
    context: Context         # Provenance and confidence information
    metadata: Dict[str, Any] = field(default_factory=dict)  # Response generation metadata

# ============================================================================
# Memory & Retrieval
# ============================================================================
@dataclass
class MemoryRecord:
    """
    Vector database record for mem0ai integration.
    """
    id: str
    vector: Optional[Vector]                     # Embedding (None for metadata-only records)
    payload: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Tool Execution
# ============================================================================
@dataclass
class ToolCall:
    """Request to execute a tool."""
    name: str
    args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolResult:
    """Result from tool execution."""
    ok: bool
    data: Any
    error: Optional[str] = None

# ============================================================================
# Policy & Decision Making
# ============================================================================
@dataclass
class PolicyAction:
    """
    Action decided by policy engine.
    Maps to which tools/modules to invoke and why.
    """
    kind: str                       # "retrieve", "search", "tool", "respond"
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for the action
    reasoning: Optional[str] = None  # Why this action was chosen
    confidence: float = 1.0         # Decision confidence (0.0 to 1.0)

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class ModeConfig:
    """Operational mode configuration."""
    name: str                      # "bare" | "fast" | "fused"
    enable_ts: bool = False        # Enable Thompson sampling for exploration
    seed: Optional[int] = None