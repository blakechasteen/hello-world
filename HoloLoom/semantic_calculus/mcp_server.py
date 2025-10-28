"""
Semantic Calculus MCP Server
=============================
Exposes semantic flow analysis via Model Context Protocol.

This allows Claude Desktop to perform real-time semantic analysis:
- analyze_semantic_flow: Compute velocity, acceleration, curvature
- predict_conversation_flow: Forecast conversation trajectory
- evaluate_conversation_ethics: Multi-objective virtue analysis

Usage:
    python -m HoloLoom.semantic_calculus.mcp_server

Configuration:
    Automatically uses optimized settings (caching enabled, JIT compilation)
"""

import asyncio
import logging
import json
from typing import Any, List, Dict, Optional, Sequence
from datetime import datetime

try:
    from mcp.server import Server
    from mcp.types import Resource, Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")

# Import semantic calculus components
try:
    from . import (
        SemanticFlowCalculus,
        SemanticSpectrum,
        EthicalSemanticPolicy,
        COMPASSIONATE_COMMUNICATION,
        SCIENTIFIC_DISCOURSE,
        THERAPEUTIC_DIALOGUE,
        HAS_NUMBA,
    )
    from ..embedding.spectral import create_embedder
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from semantic_calculus import (
        SemanticFlowCalculus,
        SemanticSpectrum,
        EthicalSemanticPolicy,
        COMPASSIONATE_COMMUNICATION,
        SCIENTIFIC_DISCOURSE,
        THERAPEUTIC_DIALOGUE,
        HAS_NUMBA,
    )
    from embedding.spectral import create_embedder

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global semantic calculus instance (initialized in main)
calculus: SemanticFlowCalculus = None
spectrum: SemanticSpectrum = None
ethical_policies: Dict[str, EthicalSemanticPolicy] = {}

# Create MCP server
if MCP_AVAILABLE:
    server = Server("hololoom-semantic-calculus")
else:
    server = None


# ============================================================================
# Helper Functions
# ============================================================================

def format_semantic_analysis(trajectory, spectrum_analysis=None, ethical_analysis=None) -> str:
    """Format semantic analysis into readable text."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("SEMANTIC FLOW ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    # Basic trajectory stats
    lines.append(f"Words analyzed: {len(trajectory.words)}")
    lines.append(f"Total semantic distance: {trajectory.total_distance():.4f}")
    lines.append("")

    # Velocity analysis
    speeds = [s.speed for s in trajectory.states]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    max_speed = max(speeds) if speeds else 0

    lines.append("VELOCITY ANALYSIS:")
    lines.append(f"  Average speed: {avg_speed:.4f}")
    lines.append(f"  Maximum speed: {max_speed:.4f}")
    lines.append(f"  Interpretation: {'Rapid semantic shifts' if avg_speed > 0.5 else 'Smooth, coherent flow'}")
    lines.append("")

    # Acceleration analysis
    accels = [s.acceleration_magnitude for s in trajectory.states]
    avg_accel = sum(accels) / len(accels) if accels else 0

    lines.append("ACCELERATION ANALYSIS:")
    lines.append(f"  Average acceleration: {avg_accel:.4f}")
    lines.append(f"  Interpretation: {'Changing topics/directions' if avg_accel > 0.1 else 'Consistent trajectory'}")
    lines.append("")

    # Curvature analysis (how much the path bends)
    curvatures = [trajectory.curvature(i) for i in range(len(trajectory.states))]
    avg_curvature = sum(curvatures) / len(curvatures) if curvatures else 0
    max_curvature_idx = curvatures.index(max(curvatures)) if curvatures else 0

    lines.append("CURVATURE ANALYSIS:")
    lines.append(f"  Average curvature: {avg_curvature:.4f}")
    lines.append(f"  Maximum curvature at: '{trajectory.words[max_curvature_idx]}'")
    lines.append(f"  Interpretation: {'Sharp semantic turns' if avg_curvature > 0.2 else 'Direct, linear flow'}")
    lines.append("")

    # Spectrum analysis (if provided)
    if spectrum_analysis:
        lines.append("SEMANTIC DIMENSION ANALYSIS:")
        dominant = spectrum_analysis.get('dominant_velocity', [])
        # dominant_velocity is a list of tuples: [(name, velocity), ...]
        for dim_name, velocity in dominant[:5]:  # Top 5
            direction = "increasing" if velocity > 0 else "decreasing"
            lines.append(f"  {dim_name}: {direction} ({abs(velocity):.3f})")
        lines.append("")

    # Ethical analysis (if provided)
    if ethical_analysis:
        lines.append("ETHICAL EVALUATION:")
        virtue_score = ethical_analysis.get('virtue_score', 0)
        manipulation_score = ethical_analysis.get('manipulation_score', 0)

        lines.append(f"  Virtue score: {virtue_score:.3f}")
        lines.append(f"  Manipulation score: {manipulation_score:.3f}")
        lines.append(f"  Overall assessment: {ethical_analysis.get('assessment', 'neutral')}")
        lines.append("")

    # Energy analysis
    kinetic_energies = [s.kinetic for s in trajectory.states if s.kinetic is not None]
    if kinetic_energies:
        avg_energy = sum(kinetic_energies) / len(kinetic_energies)
        lines.append("SEMANTIC ENERGY:")
        lines.append(f"  Average kinetic energy: {avg_energy:.4f}")
        lines.append(f"  Interpretation: {'High energy conversation' if avg_energy > 0.5 else 'Low energy, contemplative'}")
        lines.append("")

    # Performance stats
    if hasattr(calculus, 'get_cache_stats'):
        cache_stats = calculus.get_cache_stats()
        if cache_stats:
            lines.append("PERFORMANCE METRICS:")
            lines.append(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
            lines.append(f"  JIT compilation: {'enabled' if HAS_NUMBA else 'disabled (install numba for 10-50x speedup)'}")
            lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def predict_trajectory(trajectory, n_steps: int = 3) -> Dict:
    """Predict future semantic trajectory based on current velocity/acceleration."""
    # Use last state for prediction
    last_state = trajectory.states[-1]

    # Simple linear extrapolation: next_pos = pos + velocity * dt
    # With acceleration: next_pos = pos + velocity * dt + 0.5 * accel * dt^2
    dt = trajectory.dt

    predictions = []
    current_pos = last_state.position
    current_vel = last_state.velocity
    current_accel = last_state.acceleration

    for step in range(1, n_steps + 1):
        t = step * dt
        # Kinematic equation with constant acceleration
        predicted_pos = current_pos + current_vel * t + 0.5 * current_accel * (t ** 2)
        predicted_speed = np.linalg.norm(current_vel + current_accel * t)

        predictions.append({
            'step': step,
            'time': t,
            'predicted_speed': float(predicted_speed),
            'confidence': max(0.0, 1.0 - 0.2 * step)  # Decreases with distance
        })

    return {
        'predictions': predictions,
        'current_speed': float(last_state.speed),
        'current_acceleration': float(last_state.acceleration_magnitude),
        'interpretation': 'speeding up' if np.dot(last_state.velocity, last_state.acceleration) > 0 else 'slowing down'
    }


import numpy as np


# ============================================================================
# MCP Tools
# ============================================================================

if MCP_AVAILABLE:
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available semantic calculus tools."""
        return [
            Tool(
                name="analyze_semantic_flow",
                description=(
                    "Analyze semantic flow of conversation text. "
                    "Computes velocity (rate of meaning change), acceleration (change in direction), "
                    "and curvature (how much the conversation path bends). "
                    "Projects onto 16 interpretable semantic dimensions (Warmth, Formality, Certainty, etc.). "
                    "Returns detailed analysis with performance metrics."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Conversation text to analyze (will be tokenized into words)"
                        },
                        "include_dimensions": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include semantic dimension analysis (Warmth, Formality, etc.)"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "json"],
                            "default": "text",
                            "description": "Output format (text=human-readable, json=structured)"
                        }
                    },
                    "required": ["text"]
                }
            ),

            Tool(
                name="predict_conversation_flow",
                description=(
                    "Predict where conversation is heading based on current trajectory. "
                    "Uses velocity and acceleration to extrapolate future semantic direction. "
                    "Returns predictions for next N steps with confidence scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Current conversation text"
                        },
                        "n_steps": {
                            "type": "integer",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Number of future steps to predict"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "json"],
                            "default": "text",
                            "description": "Output format (text=human-readable, json=structured)"
                        }
                    },
                    "required": ["text"]
                }
            ),

            Tool(
                name="evaluate_conversation_ethics",
                description=(
                    "Evaluate conversation ethics using multi-objective optimization. "
                    "Analyzes: virtue score (alignment with ethical objectives), "
                    "manipulation detection (patterns like false urgency, charm offensive), "
                    "and provides recommendations for ethical improvement. "
                    "Supports multiple ethical frameworks: compassionate, scientific, therapeutic."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Conversation text to evaluate"
                        },
                        "framework": {
                            "type": "string",
                            "enum": ["compassionate", "scientific", "therapeutic"],
                            "default": "compassionate",
                            "description": "Ethical framework to use for evaluation"
                        },
                        "include_recommendations": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include ethical improvement recommendations"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "json"],
                            "default": "text",
                            "description": "Output format (text=human-readable, json=structured)"
                        }
                    },
                    "required": ["text"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
        """Handle tool calls."""
        try:
            if name == "analyze_semantic_flow":
                text = arguments["text"]
                include_dimensions = arguments.get("include_dimensions", True)
                output_format = arguments.get("format", "text")

                # Tokenize
                words = text.split()

                if len(words) < 2:
                    return [TextContent(
                        type="text",
                        text="Error: Need at least 2 words to analyze semantic flow"
                    )]

                # Compute trajectory
                trajectory = calculus.compute_trajectory(words)

                # Compute spectrum analysis if requested
                spectrum_analysis = None
                if include_dimensions and spectrum:
                    # Project trajectory onto semantic dimensions
                    spectrum_analysis = spectrum.analyze_semantic_forces(
                        trajectory.positions,
                        dt=trajectory.dt
                    )

                # Format output
                if output_format == "json":
                    # Build structured result
                    result = {
                        "words": words,
                        "n_states": len(trajectory.states),
                        "total_distance": float(trajectory.total_distance()),
                        "velocity": {
                            "average": float(np.mean([s.speed for s in trajectory.states])),
                            "maximum": float(np.max([s.speed for s in trajectory.states])),
                        },
                        "acceleration": {
                            "average": float(np.mean([s.acceleration_magnitude for s in trajectory.states])),
                        },
                        "curvature": {
                            "average": float(np.mean([trajectory.curvature(i) for i in range(len(trajectory.states))])),
                            "values": [float(trajectory.curvature(i)) for i in range(len(trajectory.states))]
                        },
                    }

                    # Add semantic dimensions if available
                    if spectrum_analysis:
                        dominant = spectrum_analysis.get('dominant_velocity', [])
                        result["semantic_dimensions"] = [
                            {"name": name, "velocity": float(vel)}
                            for name, vel in dominant[:10]  # Top 10
                        ]

                    # Add performance metrics
                    if hasattr(calculus, 'get_cache_stats'):
                        cache_stats = calculus.get_cache_stats()
                        if cache_stats:
                            total_requests = cache_stats['hits'] + cache_stats['misses']
                            result["performance"] = {
                                "cache_hit_rate": cache_stats['hit_rate'],
                                "cache_hits": cache_stats['hits'],
                                "cache_misses": cache_stats['misses'],
                                "cache_size": cache_stats['size'],
                                "cache_capacity": cache_stats['max_size'],
                                "total_requests": total_requests
                            }

                    output = json.dumps(result, indent=2)
                else:
                    output = format_semantic_analysis(trajectory, spectrum_analysis)

                return [TextContent(type="text", text=output)]

            elif name == "predict_conversation_flow":
                text = arguments["text"]
                n_steps = arguments.get("n_steps", 3)
                output_format = arguments.get("format", "text")

                # Tokenize and compute trajectory
                words = text.split()

                if len(words) < 3:
                    return [TextContent(
                        type="text",
                        text="Error: Need at least 3 words to predict conversation flow"
                    )]

                trajectory = calculus.compute_trajectory(words)

                # Predict future trajectory
                prediction = predict_trajectory(trajectory, n_steps)

                # Format output
                if output_format == "json":
                    # JSON format
                    result = {
                        "current_state": {
                            "interpretation": prediction['interpretation'],
                            "velocity_magnitude": prediction['current_speed'],
                            "acceleration_magnitude": prediction['current_acceleration']
                        },
                        "predictions": [
                            {
                                "step": pred['step'],
                                "time": pred['time'],
                                "predicted_speed": pred['predicted_speed'],
                                "confidence": pred['confidence'],
                                "distance_from_current": pred['predicted_speed'] * pred['time']
                            }
                            for pred in prediction['predictions']
                        ],
                        "trajectory_status": prediction['interpretation']
                    }
                    output = json.dumps(result, indent=2)
                else:
                    # Text format
                    lines = []
                    lines.append("CONVERSATION FLOW PREDICTION")
                    lines.append("=" * 70)
                    lines.append(f"\nTrajectory Status: {prediction['interpretation']}")
                    lines.append(f"Current speed: {prediction['current_speed']:.4f}")
                    lines.append(f"Current acceleration: {prediction['current_acceleration']:.4f}")
                    lines.append(f"\nPredictions for next {n_steps} steps:")

                    for pred in prediction['predictions']:
                        lines.append(f"\n  Step {pred['step']} (t={pred['time']:.1f}):")
                        lines.append(f"    Predicted speed: {pred['predicted_speed']:.4f}")
                        lines.append(f"    Confidence: {pred['confidence']:.1%}")

                    lines.append("\n" + "=" * 70)
                    output = "\n".join(lines)

                return [TextContent(type="text", text=output)]

            elif name == "evaluate_conversation_ethics":
                text = arguments["text"]
                framework = arguments.get("framework", "compassionate")
                include_recommendations = arguments.get("include_recommendations", True)
                output_format = arguments.get("format", "text")

                # Map framework name to policy
                framework_map = {
                    "compassionate": COMPASSIONATE_COMMUNICATION,
                    "scientific": SCIENTIFIC_DISCOURSE,
                    "therapeutic": THERAPEUTIC_DIALOGUE
                }

                objective = framework_map.get(framework, COMPASSIONATE_COMMUNICATION)

                # Get or create policy for this framework
                if framework not in ethical_policies:
                    dim_names = [dim.name for dim in spectrum.dimensions]
                    ethical_policies[framework] = EthicalSemanticPolicy(objective, dim_names)

                policy = ethical_policies[framework]

                # Tokenize and compute trajectory
                words = text.split()

                if len(words) < 2:
                    return [TextContent(
                        type="text",
                        text="Error: Need at least 2 words to evaluate ethics"
                    )]

                trajectory = calculus.compute_trajectory(words)

                # Project to semantic coordinates
                # spectrum.project_trajectory returns dict, but ethics code expects array
                projections_dict = spectrum.project_trajectory(trajectory.positions)
                # Convert dict to array: each row is [dim1, dim2, ..., dim16]
                dim_names = [dim.name for dim in spectrum.dimensions]
                q_semantic = np.column_stack([projections_dict[name] for name in dim_names])

                # Evaluate ethics
                ethical_analysis = policy.analyze_conversation_ethics(q_semantic)

                # Create assessment based on results
                mean_virtue = ethical_analysis['mean_virtue']
                max_manip = ethical_analysis['max_manipulation']
                is_ethical = ethical_analysis['is_ethical']

                if is_ethical and mean_virtue > 0.7:
                    assessment = "highly ethical"
                elif is_ethical and mean_virtue > 0.5:
                    assessment = "ethical"
                elif max_manip > 0.5:
                    assessment = "manipulative"
                elif mean_virtue < 0.3:
                    assessment = "problematic"
                else:
                    assessment = "neutral"

                # Format output
                if output_format == "json":
                    # JSON format
                    result = {
                        "framework": framework,
                        "virtue_score": float(mean_virtue),
                        "manipulation_score": float(max_manip),
                        "assessment": assessment,
                        "manipulation": {
                            "detected": bool(max_manip > 0.3),  # Convert to Python bool
                            "patterns": {}  # Could add specific pattern detection here
                        },
                        "ethics_range": {
                            "min_virtue": float(ethical_analysis['min_virtue']),
                            "max_virtue": float(ethical_analysis['max_virtue']),
                            "mean_manipulation": float(ethical_analysis['mean_manipulation'])
                        }
                    }

                    # Add recommendations if requested
                    if include_recommendations:
                        recs = []
                        if mean_virtue < 0.5:
                            recs.append("Consider increasing compassion and understanding")
                        if max_manip > 0.3:
                            recs.append("Reduce manipulative language patterns")
                        if len(ethical_analysis['constraint_violations']) > 0:
                            recs.append("Address ethical constraint violations")
                        result["recommendations"] = recs

                    output = json.dumps(result, indent=2)
                else:
                    # Text format
                    lines = []
                    lines.append(f"ETHICAL EVALUATION ({framework.upper()} FRAMEWORK)")
                    lines.append("=" * 70)
                    lines.append(f"\nFramework: {framework}")
                    lines.append(f"\nVirtue Score: {mean_virtue:.3f}")
                    lines.append(f"Manipulation Score: {max_manip:.3f}")
                    lines.append(f"\nAssessment: {assessment}")

                    # Manipulation detection
                    if max_manip > 0.3:
                        lines.append("\nManipulation Detection: DETECTED")
                        lines.append(f"Maximum manipulation score: {max_manip:.3f}")
                    else:
                        lines.append("\nManipulation Detection: None detected")

                    # Recommendations
                    if include_recommendations:
                        lines.append("\nRecommendations:")
                        if mean_virtue < 0.5:
                            lines.append("  • Consider increasing compassion and understanding")
                        if max_manip > 0.3:
                            lines.append("  • Reduce manipulative language patterns")
                        if len(ethical_analysis['constraint_violations']) > 0:
                            lines.append(f"  • Address {len(ethical_analysis['constraint_violations'])} ethical constraint violations")
                        if mean_virtue > 0.7 and max_manip < 0.3:
                            lines.append("  • Communication is ethical - continue current approach")

                    lines.append("\n" + "=" * 70)
                    output = "\n".join(lines)

                return [TextContent(type="text", text=output)]

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

        except Exception as e:
            logger.error(f"Error in tool {name}: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]


# ============================================================================
# Server Initialization
# ============================================================================

async def initialize_semantic_calculus():
    """Initialize semantic calculus components."""
    global calculus, spectrum, ethical_policies

    logger.info("Initializing semantic calculus MCP server...")

    # Create embedder
    logger.info("Creating embedder (384D)...")
    embed_model = create_embedder(sizes=[384])
    embed_fn = lambda words: embed_model.encode(words)

    # Create semantic flow calculus with caching
    logger.info("Creating semantic flow calculus (cache enabled)...")
    calculus = SemanticFlowCalculus(
        embed_fn,
        enable_cache=True,
        cache_size=10000
    )

    # Create semantic spectrum (16 dimensions)
    logger.info("Creating semantic spectrum (16D)...")
    spectrum = SemanticSpectrum()
    spectrum.learn_axes(embed_fn)

    logger.info("Semantic calculus initialized successfully!")
    logger.info(f"  JIT compilation: {'ENABLED' if HAS_NUMBA else 'DISABLED (pip install numba for 10-50x speedup)'}")
    logger.info(f"  Embedding cache: ENABLED (10K words)")
    logger.info(f"  Semantic dimensions: {len(spectrum.dimensions)}")


async def main():
    """Main entry point for MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP not available. Install with: pip install mcp")
        return

    # Initialize semantic calculus
    await initialize_semantic_calculus()

    # Run server
    logger.info("Starting MCP server...")
    logger.info("Server name: hololoom-semantic-calculus")
    logger.info("Tools available: analyze_semantic_flow, predict_conversation_flow, evaluate_conversation_ethics")

    # Import stdio transport
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
