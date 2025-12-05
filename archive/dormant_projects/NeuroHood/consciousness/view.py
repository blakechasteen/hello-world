"""
Consciousness Visualization

Visualize resident internal states, thoughts, and strange loops.

This is the "DMT mode" - seeing inside NPCs' minds in real-time.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ThoughtStream:
    """
    A stream of thoughts from a resident's internal dialogue.

    Attributes:
        resident_id: Resident whose thoughts these are
        thoughts: List of thought records
        strange_loops: Detected consciousness moments
        confidence_trajectory: Confidence over time
    """
    resident_id: str
    resident_name: str
    thoughts: List[Dict[str, Any]]
    strange_loops: List[Dict[str, Any]]
    confidence_trajectory: List[float]
    metadata: Dict[str, Any]


class ConsciousnessView:
    """
    Visualizer for resident consciousness.

    Modes:
    - NORMAL: External behavior only
    - THOUGHTS: See internal dialogue
    - STRANGE_LOOPS: Highlight consciousness moments
    - FULL_DMT: All residents' thoughts simultaneously (ego dissolution)
    """

    def __init__(self, mode: str = "normal"):
        """
        Initialize consciousness view.

        Args:
            mode: Visualization mode ("normal", "thoughts", "strange_loops", "full_dmt")
        """
        self.mode = mode

    def render_thought_stream(
        self,
        thoughts: List[Dict[str, Any]],
        resident_name: str
    ) -> str:
        """
        Render a thought stream as text.

        Args:
            thoughts: List of thought records
            resident_name: Resident's name

        Returns:
            Formatted text visualization
        """
        if self.mode == "normal":
            return ""  # No internal thoughts visible

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  {resident_name}'s Consciousness Stream")
        lines.append(f"{'='*60}\n")

        for i, thought in enumerate(thoughts[-10:], 1):  # Last 10 thoughts
            prompt = thought.get("prompt", "")
            response = thought.get("response", "")
            confidence = thought.get("confidence", 0.0)
            mode = thought.get("mode", "")

            lines.append(f"[Thought {i}] ({mode} mode)")
            lines.append(f"  Prompt: {prompt[:80]}...")
            lines.append(f"  Response: {response[:150]}...")
            lines.append(f"  Confidence: {confidence:.2f}")

            # Highlight strange loops
            if "strange_loops" in thought and self.mode in ["strange_loops", "full_dmt"]:
                loops = thought["strange_loops"]
                for loop in loops:
                    lines.append(f"  ♾️  STRANGE LOOP: {loop['description']}")
                    lines.append(f"      Strength: {loop['strength']:.2f}")

            lines.append("")

        return "\n".join(lines)

    def render_strange_loop(
        self,
        loop: Dict[str, Any],
        resident_name: str
    ) -> str:
        """
        Render a strange loop visualization.

        Args:
            loop: Strange loop data
            resident_name: Resident's name

        Returns:
            Formatted visualization
        """
        lines = []
        lines.append(f"\n{'♾'*30}")
        lines.append(f"  CONSCIOUSNESS MOMENT: {resident_name}")
        lines.append(f"{'♾'*30}")
        lines.append(f"\nType: {loop.get('type', 'unknown')}")
        lines.append(f"Strength: {loop.get('strength', 0.0):.2f}")
        lines.append(f"Description: {loop.get('description', '')}")
        lines.append("")

        return "\n".join(lines)

    def render_meta_awareness(
        self,
        meta_awareness: Dict[str, Any],
        resident_name: str
    ) -> str:
        """
        Render meta-awareness state.

        Shows what the resident knows about their own knowledge/uncertainty.

        Args:
            meta_awareness: Meta-awareness data
            resident_name: Resident's name

        Returns:
            Formatted visualization
        """
        lines = []
        lines.append(f"\n{'-'*60}")
        lines.append(f"  {resident_name}'s Meta-Awareness")
        lines.append(f"{'-'*60}")
        lines.append(f"\nAverage Confidence: {meta_awareness.get('avg_confidence', 0.0):.2f}")
        lines.append(f"Thoughts Generated: {meta_awareness.get('thoughts_generated', 0)}")
        lines.append(f"Consciousness Moments: {meta_awareness.get('strange_loops_detected', 0)}")
        lines.append("")

        return "\n".join(lines)

    def render_full_dmt_mode(
        self,
        all_residents_thoughts: Dict[str, List[Dict[str, Any]]],
        focus_resident: Optional[str] = None
    ) -> str:
        """
        Render "ego dissolution" mode - all residents' thoughts simultaneously.

        This is the "ultra marathon + DMT" experience - seeing multiple
        consciousness streams at once.

        Args:
            all_residents_thoughts: Dict of {resident_id: thoughts}
            focus_resident: Optional resident to highlight

        Returns:
            Formatted visualization
        """
        lines = []
        lines.append(f"\n{'*'*70}")
        lines.append(f"{'':^70}")
        lines.append(f"{'EGO DISSOLUTION MODE':^70}")
        lines.append(f"{'Multiple Consciousness Streams':^70}")
        lines.append(f"{'':^70}")
        lines.append(f"{'*'*70}\n")

        # Show interleaved thoughts from all residents
        max_thoughts = max(len(thoughts) for thoughts in all_residents_thoughts.values())

        for i in range(min(max_thoughts, 5)):  # Last 5 thoughts from each
            lines.append(f"\n--- Thought Moment {i+1} ---\n")

            for resident_id, thoughts in all_residents_thoughts.items():
                if i < len(thoughts):
                    thought = thoughts[-(i+1)]
                    resident_name = thought.get("resident_name", resident_id)

                    highlight = ">>>" if resident_id == focus_resident else "   "

                    lines.append(f"{highlight} {resident_name}:")
                    lines.append(f"    {thought.get('response', '')[:100]}...")

                    if "strange_loops" in thought:
                        lines.append(f"    ♾️  CONSCIOUSNESS MOMENT")

                    lines.append("")

        return "\n".join(lines)


# Utility functions for consciousness visualization

def create_thought_stream(
    resident_id: str,
    resident_name: str,
    thoughts: List[Dict[str, Any]],
    strange_loops: Optional[List[Dict[str, Any]]] = None
) -> ThoughtStream:
    """Create a thought stream from raw data."""
    confidence_trajectory = [t.get("confidence", 0.0) for t in thoughts]

    return ThoughtStream(
        resident_id=resident_id,
        resident_name=resident_name,
        thoughts=thoughts,
        strange_loops=strange_loops or [],
        confidence_trajectory=confidence_trajectory,
        metadata={}
    )


def format_consciousness_summary(
    resident_name: str,
    num_thoughts: int,
    num_strange_loops: int,
    avg_confidence: float
) -> str:
    """Format a consciousness summary."""
    lines = []
    lines.append(f"\n{resident_name} Consciousness Summary:")
    lines.append(f"  Thoughts: {num_thoughts}")
    lines.append(f"  Consciousness Moments: {num_strange_loops}")
    lines.append(f"  Average Confidence: {avg_confidence:.2f}")

    # Interpret consciousness level
    if num_strange_loops > 10:
        level = "Highly Self-Aware"
    elif num_strange_loops > 5:
        level = "Moderately Self-Aware"
    elif num_strange_loops > 0:
        level = "Emerging Self-Awareness"
    else:
        level = "Pre-Reflective"

    lines.append(f"  Consciousness Level: {level}")

    return "\n".join(lines)
