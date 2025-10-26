"""
Fabric - Spacetime Woven Output
================================
The final woven fabric that emerges from the weaving process.

Philosophy:
When all threads converge and the shuttle completes its pass, the result
is a piece of "fabric" - a structured response with complete lineage.

Spacetime captures both the OUTPUT (what was woven) and the TRACE (how it
was woven), creating a 4-dimensional artifact that exists in both semantic
space and temporal sequence.

The fabric can be:
- Serialized for persistence
- Analyzed for quality
- Used as training signal for reflection
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Spacetime - Woven Fabric Output
# ============================================================================

@dataclass
class WeavingTrace:
    """
    Complete computational trace of the weaving process.

    Records every stage of the shuttle's journey through the loom,
    enabling full reconstruction and analysis.
    """
    # Temporal markers
    start_time: datetime
    end_time: datetime
    duration_ms: float

    # Stage timings
    stage_durations: Dict[str, float] = field(default_factory=dict)

    # Feature extraction trace
    motifs_detected: List[str] = field(default_factory=list)
    embedding_scales_used: List[int] = field(default_factory=list)
    spectral_features: Optional[Dict[str, Any]] = None

    # Memory retrieval trace
    threads_activated: List[str] = field(default_factory=list)
    context_shards_count: int = 0
    retrieval_mode: str = "unknown"

    # Decision trace
    policy_adapter: str = "unknown"
    tool_selected: str = "unknown"
    tool_confidence: float = 0.0
    bandit_statistics: Optional[Dict[str, Any]] = None

    # Warp space trace (if used)
    warp_operations: List[tuple] = field(default_factory=list)
    tensor_field_stats: Optional[Dict[str, Any]] = None

    # Analytical metrics (mathematical guarantees)
    analytical_metrics: Optional[Dict[str, Any]] = None

    # Error tracking
    errors: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Spacetime:
    """
    Woven fabric - structured output with complete lineage.

    Spacetime is the 4-dimensional output of the weaving process:
    - 3D semantic space: query embedding, context, response
    - 1D temporal dimension: computational trace over time

    Unlike a simple response dict, Spacetime captures:
    - WHAT was produced (response content)
    - HOW it was produced (weaving trace)
    - WHY decisions were made (policy reasoning)
    - WHEN each step occurred (temporal markers)

    This enables:
    - Quality analysis and debugging
    - Reflection learning from successful patterns
    - Reproducibility and audit trails
    - Evolution of the weaving process

    Usage:
        spacetime = Spacetime(
            query_text="What is Thompson Sampling?",
            response="Thompson Sampling is...",
            trace=weaving_trace,
            metadata={"execution_mode": "fused"}
        )
        spacetime.save("output/response_001.json")
    """

    # Core output
    query_text: str
    response: str
    tool_used: str
    confidence: float

    # Computational lineage
    trace: WeavingTrace

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Context preservation
    context_summary: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)

    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)

    # Quality metrics (can be added post-hoc)
    quality_score: Optional[float] = None
    user_feedback: Optional[str] = None

    def __post_init__(self):
        """Calculate derived metrics on initialization."""
        # Ensure created_at is set
        if self.created_at is None:
            self.created_at = datetime.now()

        # Derive metadata if not provided
        if 'duration_ms' not in self.metadata and self.trace:
            self.metadata['duration_ms'] = self.trace.duration_ms

        if 'threads_activated' not in self.metadata and self.trace:
            self.metadata['threads_activated'] = len(self.trace.threads_activated)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to serializable dictionary.

        Returns:
            Dict with all spacetime data (JSON-compatible)
        """
        return {
            'query_text': self.query_text,
            'response': self.response,
            'tool_used': self.tool_used,
            'confidence': self.confidence,
            'trace': asdict(self.trace),
            'metadata': self.metadata,
            'context_summary': self.context_summary,
            'sources_used': self.sources_used,
            'created_at': self.created_at.isoformat(),
            'quality_score': self.quality_score,
            'user_feedback': self.user_feedback
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Spacetime':
        """
        Create Spacetime from dictionary.

        Args:
            data: Dict from to_dict() or JSON

        Returns:
            Spacetime instance
        """
        # Parse trace
        trace_data = data.get('trace', {})
        trace = WeavingTrace(
            start_time=datetime.fromisoformat(trace_data['start_time']),
            end_time=datetime.fromisoformat(trace_data['end_time']),
            duration_ms=trace_data['duration_ms'],
            stage_durations=trace_data.get('stage_durations', {}),
            motifs_detected=trace_data.get('motifs_detected', []),
            embedding_scales_used=trace_data.get('embedding_scales_used', []),
            spectral_features=trace_data.get('spectral_features'),
            threads_activated=trace_data.get('threads_activated', []),
            context_shards_count=trace_data.get('context_shards_count', 0),
            retrieval_mode=trace_data.get('retrieval_mode', 'unknown'),
            policy_adapter=trace_data.get('policy_adapter', 'unknown'),
            tool_selected=trace_data.get('tool_selected', 'unknown'),
            tool_confidence=trace_data.get('tool_confidence', 0.0),
            bandit_statistics=trace_data.get('bandit_statistics'),
            warp_operations=trace_data.get('warp_operations', []),
            tensor_field_stats=trace_data.get('tensor_field_stats'),
            errors=trace_data.get('errors', []),
            warnings=trace_data.get('warnings', [])
        )

        # Parse created_at
        created_at = datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()

        return cls(
            query_text=data['query_text'],
            response=data['response'],
            tool_used=data['tool_used'],
            confidence=data['confidence'],
            trace=trace,
            metadata=data.get('metadata', {}),
            context_summary=data.get('context_summary'),
            sources_used=data.get('sources_used', []),
            created_at=created_at,
            quality_score=data.get('quality_score'),
            user_feedback=data.get('user_feedback')
        )

    def save(self, path: str) -> None:
        """
        Save spacetime to JSON file.

        Args:
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Spacetime saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'Spacetime':
        """
        Load spacetime from JSON file.

        Args:
            path: Input file path

        Returns:
            Spacetime instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        logger.info(f"Spacetime loaded from {path}")
        return cls.from_dict(data)

    def add_quality_score(self, score: float, feedback: Optional[str] = None) -> None:
        """
        Add quality assessment to fabric.

        Args:
            score: Quality score (0-1)
            feedback: Optional textual feedback
        """
        self.quality_score = score
        if feedback:
            self.user_feedback = feedback

        logger.info(f"Quality score added: {score}")

    def summarize(self) -> Dict[str, Any]:
        """
        Get concise summary of spacetime.

        Returns:
            Dict with key metrics
        """
        return {
            'query': self.query_text[:100] + '...' if len(self.query_text) > 100 else self.query_text,
            'tool': self.tool_used,
            'confidence': f"{self.confidence:.2f}",
            'duration_ms': f"{self.trace.duration_ms:.1f}",
            'threads_activated': len(self.trace.threads_activated),
            'motifs_detected': len(self.trace.motifs_detected),
            'quality_score': f"{self.quality_score:.2f}" if self.quality_score is not None else "N/A",
            'timestamp': self.created_at.isoformat()
        }

    def get_reflection_signal(self) -> Dict[str, Any]:
        """
        Extract signal for reflection learning.

        Returns training data for improving the weaving process:
        - Successful patterns to reinforce
        - Failed patterns to avoid
        - Timing characteristics to optimize

        Returns:
            Dict with reflection learning signals
        """
        # Determine success based on confidence and quality
        success = self.confidence >= 0.7
        if self.quality_score is not None:
            success = success and self.quality_score >= 0.7

        return {
            'success': success,
            'query_embedding': None,  # Would be populated with actual embedding
            'tool_selected': self.tool_used,
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'duration_ms': self.trace.duration_ms,
            'stage_durations': self.trace.stage_durations,
            'motifs_pattern': self.trace.motifs_detected,
            'retrieval_mode': self.trace.retrieval_mode,
            'threads_activated_count': len(self.trace.threads_activated),
            'errors': len(self.trace.errors) > 0,
            'warnings': len(self.trace.warnings) > 0,
            'bandit_statistics': self.trace.bandit_statistics
        }


# ============================================================================
# Fabric Collection
# ============================================================================

class FabricCollection:
    """
    Collection of woven fabrics for batch analysis.

    Manages multiple Spacetime instances, enabling:
    - Batch quality analysis
    - Pattern discovery across responses
    - Aggregate statistics
    - Reflection learning from corpus
    """

    def __init__(self, fabrics: Optional[List[Spacetime]] = None):
        """
        Initialize fabric collection.

        Args:
            fabrics: Optional initial list of Spacetime instances
        """
        self.fabrics: List[Spacetime] = fabrics or []
        self.logger = logging.getLogger(__name__)

    def add(self, fabric: Spacetime) -> None:
        """Add a fabric to the collection."""
        self.fabrics.append(fabric)

    def save_all(self, directory: str) -> None:
        """
        Save all fabrics to directory.

        Args:
            directory: Output directory path
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for idx, fabric in enumerate(self.fabrics):
            filename = f"spacetime_{idx:04d}_{fabric.created_at.strftime('%Y%m%d_%H%M%S')}.json"
            fabric.save(dir_path / filename)

        self.logger.info(f"Saved {len(self.fabrics)} fabrics to {directory}")

    @classmethod
    def load_all(cls, directory: str) -> 'FabricCollection':
        """
        Load all fabrics from directory.

        Args:
            directory: Input directory path

        Returns:
            FabricCollection with loaded fabrics
        """
        dir_path = Path(directory)
        fabrics = []

        for json_file in sorted(dir_path.glob("*.json")):
            try:
                fabric = Spacetime.load(str(json_file))
                fabrics.append(fabric)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(fabrics)} fabrics from {directory}")
        return cls(fabrics)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all fabrics.

        Returns:
            Dict with statistical summary
        """
        if not self.fabrics:
            return {"count": 0}

        # Tool distribution
        tool_counts = {}
        for fabric in self.fabrics:
            tool_counts[fabric.tool_used] = tool_counts.get(fabric.tool_used, 0) + 1

        # Confidence and quality scores
        confidences = [f.confidence for f in self.fabrics]
        quality_scores = [f.quality_score for f in self.fabrics if f.quality_score is not None]

        # Timing
        durations = [f.trace.duration_ms for f in self.fabrics]

        # Success rate
        successful = sum(1 for f in self.fabrics if f.confidence >= 0.7)

        return {
            'count': len(self.fabrics),
            'tool_distribution': tool_counts,
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else None,
            'avg_duration_ms': sum(durations) / len(durations),
            'success_rate': successful / len(self.fabrics),
            'date_range': {
                'start': min(f.created_at for f in self.fabrics).isoformat(),
                'end': max(f.created_at for f in self.fabrics).isoformat()
            }
        }

    def get_reflection_corpus(self) -> List[Dict[str, Any]]:
        """
        Extract reflection learning signals from all fabrics.

        Returns:
            List of reflection signals for training
        """
        return [fabric.get_reflection_signal() for fabric in self.fabrics]

    def filter_by_tool(self, tool: str) -> 'FabricCollection':
        """
        Filter fabrics by tool used.

        Args:
            tool: Tool name to filter by

        Returns:
            New FabricCollection with filtered fabrics
        """
        filtered = [f for f in self.fabrics if f.tool_used == tool]
        return FabricCollection(filtered)

    def filter_by_quality(self, min_quality: float) -> 'FabricCollection':
        """
        Filter fabrics by minimum quality score.

        Args:
            min_quality: Minimum quality threshold

        Returns:
            New FabricCollection with filtered fabrics
        """
        filtered = [
            f for f in self.fabrics
            if f.quality_score is not None and f.quality_score >= min_quality
        ]
        return FabricCollection(filtered)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Spacetime Fabric Demo")
    print("="*80 + "\n")

    # Create a weaving trace
    start = datetime.now()
    import time
    time.sleep(0.1)  # Simulate processing
    end = datetime.now()

    trace = WeavingTrace(
        start_time=start,
        end_time=end,
        duration_ms=(end - start).total_seconds() * 1000,
        stage_durations={
            'features': 25.3,
            'retrieval': 45.2,
            'decision': 15.1,
            'execution': 14.4
        },
        motifs_detected=["ALGORITHM", "OPTIMIZATION"],
        embedding_scales_used=[96, 192, 384],
        threads_activated=["thread_001", "thread_002"],
        context_shards_count=3,
        retrieval_mode="fused",
        policy_adapter="fused_adapter",
        tool_selected="answer",
        tool_confidence=0.87,
        bandit_statistics={'answer': {'pulls': 15, 'successes': 12}}
    )

    # Create spacetime fabric
    spacetime = Spacetime(
        query_text="What is Thompson Sampling?",
        response="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation.",
        tool_used="answer",
        confidence=0.87,
        trace=trace,
        metadata={
            "execution_mode": "fused",
            "user_id": "demo_user"
        },
        sources_used=["docs/algorithms.md", "papers/thompson_1933.pdf"]
    )

    # Add quality score
    spacetime.add_quality_score(0.92, feedback="Accurate and concise")

    # Print summary
    print("Spacetime Summary:")
    summary = spacetime.summarize()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    # Save to file
    spacetime.save("/tmp/demo_spacetime.json")
    print("Saved to /tmp/demo_spacetime.json")
    print()

    # Load back
    loaded = Spacetime.load("/tmp/demo_spacetime.json")
    print(f"Loaded spacetime: {loaded.query_text[:50]}...")
    print()

    # Get reflection signal
    reflection = spacetime.get_reflection_signal()
    print("Reflection Signal:")
    for key, value in reflection.items():
        if value is not None:
            print(f"  {key}: {value}")
    print()

    # Fabric collection demo
    print("="*80)
    print("Fabric Collection Demo")
    print("="*80 + "\n")

    collection = FabricCollection([spacetime])

    # Add another fabric
    trace2 = WeavingTrace(
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_ms=85.3,
        motifs_detected=["SEARCH"],
        tool_selected="search",
        tool_confidence=0.75
    )

    spacetime2 = Spacetime(
        query_text="Find recent papers on RL",
        response="Found 5 papers...",
        tool_used="search",
        confidence=0.75,
        trace=trace2
    )
    spacetime2.add_quality_score(0.80)

    collection.add(spacetime2)

    # Get statistics
    stats = collection.get_statistics()
    print("Collection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Demo complete!")
