"""
mythRL Core Architecture Types
==============================
Fundamental types for the mythRL Shuttle-centric architecture.

These types support:
- Progressive complexity scaling (3-5-7-9)
- Full computational provenance
- Spacetime tracing
- Result aggregation

Author: mythRL Team
Date: 2025-10-27 (Phase 1 Protocol Standardization)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


# ============================================================================
# Complexity Levels
# ============================================================================

class ComplexityLevel(Enum):
    """
    Progressive complexity levels for mythRL Shuttle.
    
    Philosophy: 3-5-7-9 Progressive Complexity System
    - LITE (3 steps): Essential operations only - Extract → Route → Execute
    - FAST (5 steps): + Pattern Selection + Temporal Windows
    - FULL (7 steps): + Decision Engine + Synthesis Bridge
    - RESEARCH (9 steps): + Advanced WarpSpace + Full Tracing
    
    Performance Targets:
    - LITE: <50ms - Perfect for simple queries, greetings
    - FAST: <150ms - Search patterns with temporal awareness
    - FULL: <300ms - Complex analysis with decision engine
    - RESEARCH: No limit - Maximum capability deployment
    
    Examples:
        >>> ComplexityLevel.LITE.value
        3
        >>> ComplexityLevel.RESEARCH.value
        9
    """
    LITE = 3      # Essential: Extract → Route → Execute
    FAST = 5      # + Pattern Selection + Temporal Windows  
    FULL = 7      # + Decision Engine + Synthesis Bridge
    RESEARCH = 9  # + Advanced WarpSpace + Full Tracing


# ============================================================================
# Provenance Tracing
# ============================================================================

@dataclass
class ProvenceTrace:
    """
    Full computational provenance for mythRL operations.
    
    Tracks the complete execution history including:
    - All protocol invocations with timing
    - Shuttle internal events
    - Performance metrics
    - Synthesis chain (how patterns were combined)
    - Temporal contexts (time-aware processing)
    
    This enables:
    - Complete auditability
    - Performance analysis
    - Debugging and optimization
    - Research into decision-making patterns
    
    Attributes:
        operation_id: Unique identifier for this operation
        complexity_level: Complexity level used (LITE/FAST/FULL/RESEARCH)
        start_time: perf_counter timestamp when operation started
        modules_invoked: List of module names called
        protocol_calls: List of protocol method invocations with timing
        shuttle_events: List of shuttle internal events
        performance_metrics: Dict of performance measurements
        synthesis_chain: List of pattern synthesis operations
        temporal_contexts: List of temporal window activations
    
    Example:
        >>> trace = ProvenceTrace(
        ...     operation_id="weave_1234567890",
        ...     complexity_level=ComplexityLevel.FULL,
        ...     start_time=time.perf_counter()
        ... )
        >>> trace.add_protocol_call('memory_backend', 'retrieve', 1.5, '10 results')
        >>> len(trace.protocol_calls)
        1
    """
    operation_id: str
    complexity_level: ComplexityLevel
    start_time: float
    modules_invoked: List[str] = field(default_factory=list)
    protocol_calls: List[Dict] = field(default_factory=list)
    shuttle_events: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    synthesis_chain: List[Dict] = field(default_factory=list)
    temporal_contexts: List[Dict] = field(default_factory=list)
    
    def add_protocol_call(
        self, 
        protocol: str, 
        method: str, 
        duration_ms: float, 
        result_summary: str
    ):
        """
        Record a protocol method invocation.
        
        Args:
            protocol: Protocol name (e.g., 'memory_backend')
            method: Method name (e.g., 'retrieve')
            duration_ms: Execution time in milliseconds
            result_summary: Brief description of result
        
        Example:
            >>> trace.add_protocol_call(
            ...     'memory_backend', 
            ...     'retrieve', 
            ...     1.5, 
            ...     'Retrieved 10 shards'
            ... )
        """
        self.protocol_calls.append({
            'timestamp': time.perf_counter(),
            'protocol': protocol,
            'method': method,
            'duration_ms': duration_ms,
            'result_summary': result_summary
        })
    
    def add_shuttle_event(
        self, 
        event_type: str, 
        description: str, 
        data: Optional[Dict] = None
    ):
        """
        Record shuttle internal events.
        
        Args:
            event_type: Event category (e.g., 'weave_start', 'synthesis')
            description: Human-readable description
            data: Optional structured data about the event
        
        Example:
            >>> trace.add_shuttle_event(
            ...     'synthesis', 
            ...     'Combined 3 patterns', 
            ...     {'patterns': ['search', 'filter', 'aggregate']}
            ... )
        """
        self.shuttle_events.append({
            'timestamp': time.perf_counter(),
            'event_type': event_type,
            'description': description,
            'data': data or {}
        })
    
    def add_synthesis_step(
        self,
        step_name: str,
        inputs: List[str],
        output: str,
        duration_ms: float
    ):
        """
        Record a pattern synthesis step.
        
        Args:
            step_name: Name of synthesis operation
            inputs: Input pattern/module names
            output: Resulting synthesized pattern
            duration_ms: Synthesis time
        """
        self.synthesis_chain.append({
            'timestamp': time.perf_counter(),
            'step': step_name,
            'inputs': inputs,
            'output': output,
            'duration_ms': duration_ms
        })
    
    def add_temporal_context(
        self,
        window_start: float,
        window_end: float,
        context_type: str,
        metadata: Optional[Dict] = None
    ):
        """
        Record temporal window activation.
        
        Args:
            window_start: Start timestamp
            window_end: End timestamp  
            context_type: Type of temporal context
            metadata: Optional metadata
        """
        self.temporal_contexts.append({
            'window_start': window_start,
            'window_end': window_end,
            'context_type': context_type,
            'metadata': metadata or {}
        })
    
    def get_total_duration_ms(self) -> float:
        """Get total operation duration in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000
    
    def get_protocol_summary(self) -> Dict[str, int]:
        """Get summary of protocol usage."""
        summary = {}
        for call in self.protocol_calls:
            protocol = call['protocol']
            summary[protocol] = summary.get(protocol, 0) + 1
        return summary


# ============================================================================
# Result Types
# ============================================================================

@dataclass 
class MythRLResult:
    """
    Result object with full provenance.
    
    Every mythRL operation returns this standardized result containing:
    - The actual output
    - Confidence score
    - Complexity level used
    - Complete provenance trace
    - Spacetime coordinates (for navigation)
    
    Attributes:
        query: Original query string
        output: Result output (can be any type)
        confidence: Confidence score (0.0 to 1.0)
        complexity_level: Complexity level used
        provenance: Full provenance trace
        spacetime_coordinates: Dict with spacetime navigation data
    
    Example:
        >>> result = MythRLResult(
        ...     query="What is HoloLoom?",
        ...     output="Neural decision-making system",
        ...     confidence=0.95,
        ...     complexity_level=ComplexityLevel.FAST,
        ...     provenance=trace,
        ...     spacetime_coordinates={'x': 0.5, 'y': 0.3, 't': 1234567890}
        ... )
        >>> result.get_performance_summary()
        {'total_duration_ms': 145.2, 'complexity_level': 'FAST', ...}
    """
    query: str
    output: Any
    confidence: float
    complexity_level: ComplexityLevel
    provenance: ProvenceTrace
    spacetime_coordinates: Dict
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary.
        
        Returns:
            Dict with:
            - total_duration_ms: Total execution time
            - complexity_level: Complexity level name
            - modules_used: Number of unique modules
            - protocol_calls: Number of protocol invocations
            - shuttle_events: Number of shuttle events
        """
        total_time = self.provenance.get_total_duration_ms()
        return {
            'total_duration_ms': total_time,
            'complexity_level': self.complexity_level.name,
            'modules_used': len(self.provenance.modules_invoked),
            'protocol_calls': len(self.provenance.protocol_calls),
            'shuttle_events': len(self.provenance.shuttle_events)
        }
    
    def get_confidence_breakdown(self) -> Dict:
        """
        Get breakdown of confidence sources.
        
        Returns:
            Dict with confidence contributors
        """
        # This can be extended to track confidence from different sources
        return {
            'overall': self.confidence,
            'complexity_bonus': 0.1 * (self.complexity_level.value / 9),
            'protocol_diversity': min(1.0, len(self.provenance.get_protocol_summary()) / 5)
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'ComplexityLevel',
    'ProvenceTrace',
    'MythRLResult',
]
