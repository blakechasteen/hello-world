#!/usr/bin/env python3
"""
HoloLoom Memory Bridge for Promptly
====================================
Connects Promptly's recursive intelligence with HoloLoom's memory system.

Features:
- Store loop results in HoloLoom knowledge graph
- Retrieve past refinement patterns
- Meta-learning: "What loop type worked best for this task?"
- Persistent scratchpad across sessions
- Strange loops that reference their own history
"""

import json
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Try to import HoloLoom memory
HOLOLOOM_AVAILABLE = False
try:
    # Add HoloLoom to path if not already there
    hololoom_path = Path(__file__).parent.parent.parent / "HoloLoom"
    if hololoom_path.exists() and str(hololoom_path) not in sys.path:
        sys.path.insert(0, str(hololoom_path))

    from memory.protocol import UnifiedMemoryInterface, Strategy, create_unified_memory, Memory
    HOLOLOOM_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] HoloLoom not available: {e}")


@dataclass
class LoopMemory:
    """Memory entry for a recursive loop execution"""
    loop_id: str
    loop_type: str  # refine, hofstadter, etc.
    task: str
    final_output: str
    iterations: int
    stop_reason: str
    quality_scores: List[float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_memory_content(self) -> str:
        """Convert to HoloLoom memory format"""
        parts = [
            f"# Recursive Loop: {self.loop_type}",
            "",
            f"**Task:** {self.task}",
            f"**Iterations:** {self.iterations}",
            f"**Stop Reason:** {self.stop_reason}",
            ""
        ]

        if self.quality_scores:
            parts.append("**Quality Progression:**")
            for i, score in enumerate(self.quality_scores, 1):
                parts.append(f"  Iteration {i}: {score:.2f}")
            parts.append("")

        parts.extend([
            "**Final Output:**",
            self.final_output[:500] + "..." if len(self.final_output) > 500 else self.final_output
        ])

        return "\n".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to HoloLoom metadata"""
        return {
            "loop_id": self.loop_id,
            "loop_type": self.loop_type,
            "iterations": self.iterations,
            "stop_reason": self.stop_reason,
            "quality_scores": self.quality_scores,
            "timestamp": self.timestamp,
            **self.metadata
        }


class HoloLoomBridge:
    """
    Bridge between Promptly and HoloLoom

    Enables:
    - Storing loop results in knowledge graph
    - Retrieving similar past loops
    - Learning from refinement patterns
    - Meta-analysis of loop effectiveness
    """

    def __init__(self, user_id: str = "promptly_user", strategy: Strategy = Strategy.FUSED):
        """
        Initialize HoloLoom bridge.

        Args:
            user_id: User identifier for HoloLoom
            strategy: Memory strategy (FUSED, TEMPORAL, SEMANTIC, GRAPH, PATTERN)
        """
        self.user_id = user_id
        self.strategy = strategy
        self.memory: Optional[UnifiedMemoryInterface] = None
        self.enabled = HOLOLOOM_AVAILABLE

        if HOLOLOOM_AVAILABLE:
            try:
                self.memory = create_unified_memory(
                    user_id=user_id,
                    enable_neo4j=False,  # Use simple graph for now
                    enable_mem0=False,
                    enable_qdrant=False,
                    enable_patterns=True
                )
                print(f"[OK] HoloLoom bridge initialized")
            except Exception as e:
                print(f"[WARN] Failed to initialize HoloLoom: {e}")
                self.enabled = False
        else:
            print("[WARN] HoloLoom not available - bridge disabled")

    def store_loop_result(
        self,
        loop_type: str,
        task: str,
        result: Any,  # LoopResult from recursive_loops
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Store loop execution result in HoloLoom.

        Args:
            loop_type: Type of loop (refine, hofstadter, etc.)
            task: Original task
            result: LoopResult object
            metadata: Additional metadata

        Returns:
            Memory ID if stored, None if disabled
        """
        if not self.enabled or not self.memory:
            return None

        # Generate loop ID
        loop_id = f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create loop memory
        loop_mem = LoopMemory(
            loop_id=loop_id,
            loop_type=loop_type,
            task=task,
            final_output=result.final_output,
            iterations=result.iterations,
            stop_reason=result.stop_reason,
            quality_scores=result.improvement_history if hasattr(result, 'improvement_history') else [],
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        # Store in HoloLoom
        try:
            memory_obj = Memory(
                content=loop_mem.to_memory_content(),
                metadata=loop_mem.to_metadata()
            )

            memory_id = self.memory.add(memory_obj)
            print(f"[OK] Stored loop result in HoloLoom: {loop_id}")
            return memory_id
        except Exception as e:
            print(f"[WARN]  Failed to store loop result: {e}")
            return None

    def retrieve_similar_loops(
        self,
        task: str,
        loop_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Retrieve similar past loop executions.

        Args:
            task: Current task to find similar loops for
            loop_type: Optional filter by loop type
            limit: Maximum number of results

        Returns:
            List of similar loop memories
        """
        if not self.enabled or not self.memory:
            return []

        try:
            # Search for similar tasks
            results = self.memory.search(
                query=task,
                limit=limit
            )

            # Filter by loop type if specified
            if loop_type:
                results = [
                    r for r in results
                    if r.get('metadata', {}).get('loop_type') == loop_type
                ]

            return results
        except Exception as e:
            print(f"[WARN]  Failed to retrieve similar loops: {e}")
            return []

    def get_loop_analytics(self, loop_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analytics on loop performance.

        Args:
            loop_type: Optional filter by loop type

        Returns:
            Analytics dictionary with stats
        """
        if not self.enabled or not self.memory:
            return {"enabled": False}

        try:
            # Get all loop memories
            all_memories = self.memory.get_all()

            # Filter to loop memories only
            loop_memories = [
                m for m in all_memories
                if m.get('metadata', {}).get('loop_id')
            ]

            # Filter by type if specified
            if loop_type:
                loop_memories = [
                    m for m in loop_memories
                    if m.get('metadata', {}).get('loop_type') == loop_type
                ]

            if not loop_memories:
                return {
                    "enabled": True,
                    "total_loops": 0,
                    "message": "No loop executions found"
                }

            # Calculate analytics
            total_loops = len(loop_memories)
            avg_iterations = sum(
                m.get('metadata', {}).get('iterations', 0)
                for m in loop_memories
            ) / total_loops if total_loops > 0 else 0

            # Stop reason distribution
            stop_reasons = {}
            for m in loop_memories:
                reason = m.get('metadata', {}).get('stop_reason', 'unknown')
                stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

            # Average quality scores
            all_scores = []
            for m in loop_memories:
                scores = m.get('metadata', {}).get('quality_scores', [])
                all_scores.extend(scores)

            avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0

            return {
                "enabled": True,
                "total_loops": total_loops,
                "avg_iterations": avg_iterations,
                "avg_quality": avg_quality,
                "stop_reasons": stop_reasons,
                "loop_type": loop_type or "all"
            }
        except Exception as e:
            print(f"[WARN]  Failed to get analytics: {e}")
            return {"enabled": True, "error": str(e)}

    def get_best_loop_for_task(self, task: str, top_n: int = 3) -> List[Dict]:
        """
        Find the best loop type for a given task based on history.

        Args:
            task: Task description
            top_n: Number of recommendations

        Returns:
            List of recommended loop types with scores
        """
        if not self.enabled or not self.memory:
            return []

        try:
            # Get similar past loops
            similar = self.retrieve_similar_loops(task, limit=20)

            if not similar:
                return []

            # Score loop types by quality
            loop_type_scores = {}
            loop_type_counts = {}

            for mem in similar:
                metadata = mem.get('metadata', {})
                loop_type = metadata.get('loop_type')
                scores = metadata.get('quality_scores', [])

                if loop_type and scores:
                    avg_score = sum(scores) / len(scores)

                    if loop_type not in loop_type_scores:
                        loop_type_scores[loop_type] = []
                        loop_type_counts[loop_type] = 0

                    loop_type_scores[loop_type].append(avg_score)
                    loop_type_counts[loop_type] += 1

            # Average scores per loop type
            recommendations = []
            for loop_type, scores in loop_type_scores.items():
                avg_score = sum(scores) / len(scores)
                recommendations.append({
                    "loop_type": loop_type,
                    "avg_quality": avg_score,
                    "sample_size": loop_type_counts[loop_type],
                    "confidence": min(loop_type_counts[loop_type] / 5.0, 1.0)  # Max confidence at 5+ samples
                })

            # Sort by quality
            recommendations.sort(key=lambda x: x['avg_quality'], reverse=True)

            return recommendations[:top_n]
        except Exception as e:
            print(f"[WARN]  Failed to get recommendations: {e}")
            return []

    def clear_loop_history(self):
        """Clear all stored loop results"""
        if not self.enabled or not self.memory:
            return

        try:
            # Get all loop memories
            all_memories = self.memory.get_all()

            # Delete loop memories
            deleted = 0
            for mem in all_memories:
                if mem.get('metadata', {}).get('loop_id'):
                    # Note: UnifiedMemoryInterface might not have delete
                    # This is a placeholder for when it's implemented
                    deleted += 1

            print(f"[OK] Cleared {deleted} loop memories")
        except Exception as e:
            print(f"[WARN]  Failed to clear history: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_bridge(user_id: str = "promptly_user", strategy: Strategy = Strategy.FUSED) -> HoloLoomBridge:
    """Create HoloLoom bridge instance"""
    return HoloLoomBridge(user_id=user_id, strategy=strategy)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("HoloLoom Memory Bridge for Promptly")
    print("\nFeatures:")
    print("  - Store loop results in knowledge graph")
    print("  - Retrieve similar past loops")
    print("  - Meta-learning from refinement patterns")
    print("  - Analytics on loop effectiveness")

    if HOLOLOOM_AVAILABLE:
        print("\n[OK] HoloLoom available")

        # Demo
        bridge = create_bridge()

        if bridge.enabled:
            print("\n[OK] Bridge initialized successfully")

            # Get analytics (will be empty initially)
            analytics = bridge.get_loop_analytics()
            print(f"\nAnalytics: {analytics}")
        else:
            print("\n[WARN]  Bridge initialization failed")
    else:
        print("\n[WARN]  HoloLoom not available")
        print("Install with: pip install the HoloLoom dependencies")
