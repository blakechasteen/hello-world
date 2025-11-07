"""
Agent system type definitions.

Data structures for specialized agent bots with working memory.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
import numpy as np


class AgentDomain(Enum):
    """Agent specialization domains"""
    BUDGET = "budget"
    ARCHITECTURE = "architecture"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    PLANNING = "planning"
    GENERAL = "general"


@dataclass
class WorkingMemoryState:
    """
    Three complementary views of agent's current focus:
    - Semantic: WHERE in 244D space (geometric)
    - Graph: WHICH nodes are activated (network)
    - Computational: WHAT's ready for math (warp space)
    """

    # LEVEL 1: SEMANTIC (Geometric)
    focus_vector: np.ndarray  # 244D position
    attention_radius: float = 0.3
    momentum: float = 0.7  # How sticky is focus

    # LEVEL 2: GRAPH (Activation Dynamics)
    activation_map: Dict[str, float] = field(default_factory=dict)  # node_id → activation
    propagation_decay: float = 0.85

    # LEVEL 3: COMPUTATIONAL (Warp Space)
    tensioned_threads: Set[str] = field(default_factory=set)  # Currently in warp
    tension_profile: Dict[str, float] = field(default_factory=dict)  # Persistent tensions

    def copy(self) -> 'WorkingMemoryState':
        """Deep copy of state"""
        return WorkingMemoryState(
            focus_vector=self.focus_vector.copy(),
            attention_radius=self.attention_radius,
            momentum=self.momentum,
            activation_map=self.activation_map.copy(),
            propagation_decay=self.propagation_decay,
            tensioned_threads=self.tensioned_threads.copy(),
            tension_profile=self.tension_profile.copy()
        )


@dataclass
class WorkingMemorySnapshot:
    """Snapshot of working memory at decision time (for learning)"""
    timestamp: float
    query_text: str

    # Semantic state
    focus_vector: np.ndarray
    attention_radius: float
    top_semantic_dimensions: List[str]

    # Graph state
    activation_map: Dict[str, float]
    highly_activated: List[str]  # nodes > 0.7

    # Computational state
    tensioned_threads: List[str]
    tension_profile: Dict[str, float]
    warp_operations: List[str]  # What math was done

    # Outcome
    confidence: float
    success: bool  # confidence > threshold
    tool_used: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize (handle numpy arrays)"""
        d = asdict(self)
        d['focus_vector'] = self.focus_vector.tolist()
        d['tensioned_threads'] = list(self.tensioned_threads)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'WorkingMemorySnapshot':
        """Deserialize"""
        d['focus_vector'] = np.array(d['focus_vector'])
        return cls(**d)


@dataclass
class LearnedPattern:
    """What worked in the past - learned from successful queries"""
    pattern_id: str

    # Context (what the pattern looks like)
    semantic_region: np.ndarray  # Centroid of successful focus vectors
    typical_activation_pattern: Dict[str, float]  # Average activations
    critical_threads: List[str]  # Threads present in >80% of successes

    # Performance (how well it works)
    success_count: int
    total_count: int
    avg_confidence: float

    # Meta
    first_seen: float
    last_used: float

    def success_rate(self) -> float:
        """Success rate of this pattern"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

    def to_dict(self) -> Dict:
        """Serialize"""
        return {
            'pattern_id': self.pattern_id,
            'semantic_region': self.semantic_region.tolist(),
            'typical_activation_pattern': self.typical_activation_pattern,
            'critical_threads': self.critical_threads,
            'success_count': self.success_count,
            'total_count': self.total_count,
            'avg_confidence': self.avg_confidence,
            'first_seen': self.first_seen,
            'last_used': self.last_used
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'LearnedPattern':
        """Deserialize"""
        d['semantic_region'] = np.array(d['semantic_region'])
        return cls(**d)


@dataclass
class AgentProfile:
    """Configuration for a specialized agent bot"""
    agent_id: str
    name: str
    domain: AgentDomain

    # Custom instructions
    system_prompt: str
    priorities: List[str] = field(default_factory=list)  # ["accuracy", "speed", "cost_efficiency"]
    semantic_dimensions: List[str] = field(default_factory=list)  # Which 244D dimensions to emphasize

    # Tool preferences
    preferred_tools: List[str] = field(default_factory=list)
    tool_thresholds: Dict[str, float] = field(default_factory=dict)  # Confidence thresholds per tool

    # Memory configuration
    context_window_size: int = 10
    heat_decay_rate: float = 0.05  # Per hour

    # Learning configuration
    enable_reflection: bool = True
    enable_learning: bool = True
    refinement_threshold: float = 0.75
    success_threshold: float = 0.75  # What counts as "success"

    # Working memory configuration
    attention_radius: float = 0.3
    momentum: float = 0.7
    propagation_decay: float = 0.85


@dataclass
class AgentStats:
    """Runtime statistics for an agent"""
    total_queries: int = 0
    successful_queries: int = 0  # confidence > threshold
    avg_confidence: float = 0.0
    total_learning_events: int = 0
    patterns_learned: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def success_rate(self) -> float:
        """Overall success rate"""
        return self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0

    def cache_hit_rate(self) -> float:
        """Cache effectiveness"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
