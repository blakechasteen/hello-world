"""
Provenance Tracking - Full computational lineage

Tracks the complete decision-making process from input to output,
enabling debugging, auditing, and understanding of AI decisions.

Integrates with HoloLoom's Spacetime fabric for complete lineage.

Research:
- Gehani & Tariq (2012): SPADE: Support for Provenance Auditing in Distributed Environments
- Moreau & Groth (2013): Provenance: An Introduction to PROV
- Herschel et al. (2017): A survey on provenance: What for? What form? What from?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime


class ProvenanceEvent(Enum):
    """Types of provenance events"""
    INPUT = "input"  # Input received
    TRANSFORM = "transform"  # Data transformation
    RETRIEVE = "retrieve"  # Memory retrieval
    COMPUTE = "compute"  # Computation
    DECISION = "decision"  # Decision made
    OUTPUT = "output"  # Output generated
    ERROR = "error"  # Error occurred


@dataclass
class ComputationalTrace:
    """Single step in computational lineage"""
    event_type: ProvenanceEvent
    timestamp: datetime
    stage: str  # Which stage of pipeline (e.g., "extraction", "decision")
    inputs: Dict[str, Any]  # Inputs to this step
    outputs: Dict[str, Any]  # Outputs from this step
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    duration_ms: float = 0.0  # Duration of this step
    dependencies: List[str] = field(default_factory=list)  # IDs of preceding steps

    def __repr__(self) -> str:
        return f"{self.event_type.value.upper()} @ {self.stage} ({self.duration_ms:.1f}ms)"


@dataclass
class LineageGraph:
    """Directed acyclic graph of computational lineage"""
    traces: List[ComputationalTrace] = field(default_factory=list)
    root_id: Optional[str] = None
    final_output: Optional[Any] = None

    def add_trace(self, trace: ComputationalTrace, trace_id: Optional[str] = None):
        """Add a trace to the lineage"""
        self.traces.append(trace)
        if not self.root_id and trace.event_type == ProvenanceEvent.INPUT:
            self.root_id = trace_id or f"trace_{len(self.traces)}"

    def get_critical_path(self) -> List[ComputationalTrace]:
        """
        Get critical path (longest path from input to output).

        This shows the bottleneck in the pipeline.
        """
        if not self.traces:
            return []

        # Find input and output traces
        input_traces = [t for t in self.traces if t.event_type == ProvenanceEvent.INPUT]
        output_traces = [t for t in self.traces if t.event_type == ProvenanceEvent.OUTPUT]

        if not input_traces or not output_traces:
            return self.traces

        # Simple heuristic: return all traces in order
        # More sophisticated: build DAG and find longest path
        return self.traces

    def total_duration(self) -> float:
        """Total duration of all steps"""
        return sum(t.duration_ms for t in self.traces)

    def bottleneck_stages(self, threshold: float = 0.3) -> List[ComputationalTrace]:
        """
        Find bottleneck stages (taking >threshold of total time).

        Args:
            threshold: Fraction of total time (0.3 = 30%)

        Returns:
            Traces that are bottlenecks
        """
        total = self.total_duration()
        threshold_duration = total * threshold

        return [t for t in self.traces if t.duration_ms > threshold_duration]

    def __repr__(self) -> str:
        return f"LineageGraph({len(self.traces)} steps, {self.total_duration():.1f}ms total)"


class ProvenanceTracker:
    """
    Track provenance (lineage) of AI decisions.

    Records the full computational path from input to output,
    enabling debugging, auditing, and explainability.
    """

    def __init__(
        self,
        track_intermediate: bool = True,
        track_timing: bool = True
    ):
        """
        Args:
            track_intermediate: Track intermediate steps (not just input/output)
            track_timing: Track timing information
        """
        self.track_intermediate = track_intermediate
        self.track_timing = track_timing
        self.lineage = LineageGraph()
        self._start_times: Dict[str, float] = {}  # Track step start times

    def record_input(
        self,
        inputs: Dict[str, Any],
        stage: str = "input"
    ):
        """Record input to system"""
        trace = ComputationalTrace(
            event_type=ProvenanceEvent.INPUT,
            timestamp=datetime.now(),
            stage=stage,
            inputs=inputs,
            outputs={}
        )
        self.lineage.add_trace(trace)

    def record_transform(
        self,
        stage: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ):
        """Record data transformation"""
        if not self.track_intermediate:
            return

        trace = ComputationalTrace(
            event_type=ProvenanceEvent.TRANSFORM,
            timestamp=datetime.now(),
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata or {},
            dependencies=dependencies or []
        )
        self.lineage.add_trace(trace)

    def record_retrieval(
        self,
        stage: str,
        query: Dict[str, Any],
        retrieved: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record memory retrieval"""
        trace = ComputationalTrace(
            event_type=ProvenanceEvent.RETRIEVE,
            timestamp=datetime.now(),
            stage=stage,
            inputs=query,
            outputs=retrieved,
            metadata=metadata or {}
        )
        self.lineage.add_trace(trace)

    def record_computation(
        self,
        stage: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record computation step"""
        if not self.track_intermediate:
            return

        trace = ComputationalTrace(
            event_type=ProvenanceEvent.COMPUTE,
            timestamp=datetime.now(),
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        self.lineage.add_trace(trace)

    def record_decision(
        self,
        stage: str,
        inputs: Dict[str, Any],
        decision: Any,
        confidence: float = 1.0,
        alternatives: Optional[List[Any]] = None
    ):
        """Record decision made"""
        metadata = {
            'confidence': confidence,
            'alternatives': alternatives or []
        }

        trace = ComputationalTrace(
            event_type=ProvenanceEvent.DECISION,
            timestamp=datetime.now(),
            stage=stage,
            inputs=inputs,
            outputs={'decision': decision},
            metadata=metadata
        )
        self.lineage.add_trace(trace)

    def record_output(
        self,
        outputs: Dict[str, Any],
        stage: str = "output"
    ):
        """Record final output"""
        trace = ComputationalTrace(
            event_type=ProvenanceEvent.OUTPUT,
            timestamp=datetime.now(),
            stage=stage,
            inputs={},
            outputs=outputs
        )
        self.lineage.add_trace(trace)
        self.lineage.final_output = outputs

    def record_error(
        self,
        stage: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record error"""
        metadata = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }

        trace = ComputationalTrace(
            event_type=ProvenanceEvent.ERROR,
            timestamp=datetime.now(),
            stage=stage,
            inputs=context or {},
            outputs={},
            metadata=metadata
        )
        self.lineage.add_trace(trace)

    def start_timing(self, stage: str):
        """Start timing a stage"""
        if self.track_timing:
            import time
            self._start_times[stage] = time.time() * 1000

    def end_timing(self, stage: str) -> float:
        """End timing a stage, return duration in ms"""
        if not self.track_timing or stage not in self._start_times:
            return 0.0

        import time
        current_time = time.time() * 1000
        duration = current_time - self._start_times[stage]
        del self._start_times[stage]
        return duration

    def get_lineage(self) -> LineageGraph:
        """Get complete lineage graph"""
        return self.lineage

    def visualize_lineage(self) -> str:
        """Generate text visualization of lineage"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPUTATIONAL LINEAGE")
        lines.append("=" * 80)
        lines.append(f"Total Duration: {self.lineage.total_duration():.1f}ms")
        lines.append(f"Total Steps: {len(self.lineage.traces)}")
        lines.append("")

        # Show critical path
        critical_path = self.lineage.get_critical_path()
        lines.append("Critical Path:")
        for i, trace in enumerate(critical_path):
            indent = "  " * (i % 3)
            arrow = "└─>" if i == len(critical_path) - 1 else "├─>"
            lines.append(f"{indent}{arrow} {trace}")

        lines.append("")

        # Show bottlenecks
        bottlenecks = self.lineage.bottleneck_stages(threshold=0.2)
        if bottlenecks:
            lines.append("Bottlenecks (>20% of total time):")
            for trace in bottlenecks:
                pct = (trace.duration_ms / self.lineage.total_duration()) * 100
                lines.append(f"  ⚠ {trace.stage}: {trace.duration_ms:.1f}ms ({pct:.0f}%)")

        return "\n".join(lines)

    def export_to_spacetime(self) -> Dict[str, Any]:
        """
        Export to HoloLoom Spacetime format.

        Integrates with existing Spacetime fabric for unified provenance.
        """
        return {
            'type': 'provenance_lineage',
            'total_duration_ms': self.lineage.total_duration(),
            'num_steps': len(self.lineage.traces),
            'critical_path': [
                {
                    'event': t.event_type.value,
                    'stage': t.stage,
                    'duration_ms': t.duration_ms,
                    'metadata': t.metadata
                }
                for t in self.lineage.get_critical_path()
            ],
            'bottlenecks': [
                {
                    'stage': t.stage,
                    'duration_ms': t.duration_ms,
                    'fraction_of_total': t.duration_ms / (self.lineage.total_duration() + 1e-10)
                }
                for t in self.lineage.bottleneck_stages()
            ]
        }


def trace_decision(
    inputs: Dict[str, Any],
    decision_function: callable,
    track_intermediate: bool = True
) -> tuple:
    """
    Convenience function to trace a decision.

    Args:
        inputs: Input to decision function
        decision_function: Function that makes decision
        track_intermediate: Track intermediate steps

    Returns:
        (decision, lineage_graph) tuple
    """
    tracker = ProvenanceTracker(track_intermediate=track_intermediate)

    # Record input
    tracker.record_input(inputs)

    # Start timing
    tracker.start_timing("decision")

    # Make decision
    try:
        decision = decision_function(inputs)

        # Record decision
        duration = tracker.end_timing("decision")
        tracker.record_decision(
            stage="decision",
            inputs=inputs,
            decision=decision
        )

        # Record output
        tracker.record_output({'decision': decision})

        return decision, tracker.get_lineage()

    except Exception as e:
        # Record error
        tracker.record_error("decision", e, context=inputs)
        raise
