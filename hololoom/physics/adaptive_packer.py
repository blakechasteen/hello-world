"""
Adaptive Packer - Extract Optimally Packed Context

High-level API for using fluid dynamics to pack context windows adaptively.
Combines ContextFlowEngine with reverse prompting for sparse regions.

Philosophy:
    "Context flows like water to fill available space optimally"

Usage:
    packer = AdaptivePacker(graph, max_tokens=8000)

    # Inject query
    packer.inject("ThompsonSampling", importance=0.95, tokens=50)

    # Pack adaptively
    context = await packer.pack(enable_reverse_prompting=True)
"""

from typing import Dict, List, Optional, Tuple, Callable, Awaitable
import asyncio
from dataclasses import dataclass

from .fluid_dynamics import ContextFlowEngine, FlowState

try:
    from ..memory.graph import KG
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


@dataclass
class PackedContext:
    """Result of adaptive context packing."""
    nodes: List[Tuple[str, float, Optional[str]]]  # (node, importance, text)
    tokens_used: int
    tokens_available: int
    flow_states: List[FlowState]  # History of flow states
    reverse_prompts_executed: List[str]  # Reverse prompts that were executed
    final_pressure_map: Dict[str, float]


class AdaptivePacker:
    """
    Adaptive context packer using fluid dynamics.

    Automatically packs context windows by:
        1. Injecting high-pressure context (user query)
        2. Propagating via Navier-Stokes
        3. Detecting sparse regions
        4. Reverse prompting to fill gaps (optional)
        5. Extracting optimally packed context

    Usage:
        packer = AdaptivePacker(graph, max_tokens=8000)

        # Inject query
        packer.inject("ThompsonSampling", importance=0.95, tokens=50)

        # Pack adaptively (synchronous)
        context = packer.pack_sync(max_iterations=10)

        # Pack with reverse prompting (async)
        context = await packer.pack(enable_reverse_prompting=True)
    """

    def __init__(
        self,
        graph: Optional["KG"] = None,
        max_tokens: int = 8000,
        viscosity: float = 0.01,
        sparse_threshold: float = 0.3,
        reverse_prompt_callback: Optional[
            Callable[[str], Awaitable[Tuple[int, str]]]
        ] = None
    ):
        """
        Initialize adaptive packer.

        Args:
            graph: Knowledge graph (optional)
            max_tokens: Maximum token budget
            viscosity: Fluid viscosity (lower = faster propagation)
            sparse_threshold: Pressure below this triggers reverse prompting
            reverse_prompt_callback: Async function to execute reverse prompts
                                    Should return (tokens, text) for the prompt
        """
        self.max_tokens = max_tokens
        self.engine = ContextFlowEngine(
            graph=graph,
            viscosity=viscosity,
            sparse_threshold=sparse_threshold
        )
        self.reverse_prompt_callback = reverse_prompt_callback
        self.flow_history: List[FlowState] = []
        self.reverse_prompts_executed: List[str] = []

    def inject(
        self,
        node: str,
        importance: float,
        tokens: int = 0,
        text: Optional[str] = None
    ):
        """
        Inject context at a node (creates high-pressure source).

        Args:
            node: Target node
            importance: Importance value (0.0 - 1.0)
            tokens: Token count
            text: Context text
        """
        self.engine.inject_context(node, importance, tokens, text)

    def add_edge(self, source: str, target: str):
        """Add edge to graph structure."""
        self.engine.add_edge(source, target)

    def pack_sync(
        self,
        max_iterations: int = 10,
        dt: float = 0.01
    ) -> PackedContext:
        """
        Pack context synchronously (no reverse prompting).

        Args:
            max_iterations: Number of Navier-Stokes iterations
            dt: Timestep

        Returns:
            Packed context
        """
        # Propagate for max_iterations
        for _ in range(max_iterations):
            self.engine.step(dt)
            self.flow_history.append(self.engine.get_state())

        # Extract packed context
        nodes = self.engine.extract_context(self.max_tokens)

        return PackedContext(
            nodes=nodes,
            tokens_used=self.engine.tokens_used,
            tokens_available=self.max_tokens,
            flow_states=self.flow_history,
            reverse_prompts_executed=self.reverse_prompts_executed,
            final_pressure_map=dict(self.engine.pressure.pressures)
        )

    async def pack(
        self,
        max_iterations: int = 10,
        dt: float = 0.01,
        enable_reverse_prompting: bool = True,
        reverse_prompt_budget: float = 0.2  # 20% of token budget
    ) -> PackedContext:
        """
        Pack context adaptively with optional reverse prompting.

        Args:
            max_iterations: Number of Navier-Stokes iterations
            dt: Timestep
            enable_reverse_prompting: Whether to fill sparse regions
            reverse_prompt_budget: Fraction of tokens for reverse prompts

        Returns:
            Packed context
        """
        reverse_tokens_budget = int(self.max_tokens * reverse_prompt_budget)
        reverse_tokens_used = 0

        for iteration in range(max_iterations):
            # Propagate
            self.engine.step(dt)
            state = self.engine.get_state()
            self.flow_history.append(state)

            # Check for sparse regions
            if enable_reverse_prompting and self.reverse_prompt_callback:
                sparse = self.engine.detect_sparse_regions()

                # Only fill sparse regions if we have token budget
                if sparse and reverse_tokens_used < reverse_tokens_budget:
                    # Fill one sparse region per iteration
                    node = sparse[0]
                    reverse_prompt = self.engine.generate_reverse_prompt(node)

                    if reverse_prompt:
                        # Execute reverse prompt
                        try:
                            tokens, text = await self.reverse_prompt_callback(
                                reverse_prompt
                            )

                            # Inject result as new pressure source
                            self.engine.inject_context(
                                node=node,
                                importance=0.6,  # Medium importance
                                tokens=tokens,
                                text=text
                            )

                            self.reverse_prompts_executed.append(reverse_prompt)
                            reverse_tokens_used += tokens

                        except Exception as e:
                            # Graceful degradation if reverse prompt fails
                            pass

        # Extract packed context
        nodes = self.engine.extract_context(self.max_tokens)

        return PackedContext(
            nodes=nodes,
            tokens_used=self.engine.tokens_used,
            tokens_available=self.max_tokens,
            flow_states=self.flow_history,
            reverse_prompts_executed=self.reverse_prompts_executed,
            final_pressure_map=dict(self.engine.pressure.pressures)
        )

    def get_flow_summary(self) -> Dict[str, any]:
        """Get summary of flow dynamics."""
        if not self.flow_history:
            return {
                "iterations": 0,
                "final_pressure_mean": 0.0,
                "final_velocity_mean": 0.0,
                "sparse_regions": 0,
                "reverse_prompts": 0
            }

        final_state = self.flow_history[-1]
        return {
            "iterations": len(self.flow_history),
            "final_pressure_mean": final_state.pressure_stats.get("mean", 0.0),
            "final_velocity_mean": final_state.velocity_stats.get("mean", 0.0),
            "sparse_regions": len(final_state.sparse_regions),
            "reverse_prompts": len(self.reverse_prompts_executed),
            "tokens_used": final_state.tokens_used
        }

    def visualize_pressure_field(self) -> str:
        """
        Create ASCII visualization of pressure field.

        Returns:
            ASCII art showing pressure distribution
        """
        if not self.engine.pressure.pressures:
            return "No pressure field to visualize"

        # Sort by pressure
        nodes = sorted(
            self.engine.pressure.pressures.items(),
            key=lambda x: x[1],
            reverse=True
        )

        lines = ["Pressure Field (High -> Low):"]
        lines.append("=" * 50)

        for node, pressure in nodes[:10]:  # Top 10
            bar_length = int(pressure * 40)
            bar = "#" * bar_length
            lines.append(f"{node:20s} {bar} {pressure:.2f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        summary = self.get_flow_summary()
        return (
            f"AdaptivePacker(max_tokens={self.max_tokens}, "
            f"iterations={summary['iterations']}, "
            f"reverse_prompts={summary['reverse_prompts']})"
        )
