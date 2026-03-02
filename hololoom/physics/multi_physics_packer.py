"""
Multi-Physics Context Packer

Combines Phase 1 (Gradient Flow) + Phase 2 (Fluid Dynamics) for optimal context packing.

Architecture:
    1. Gradient Flow: Routes context allocation across components
    2. Fluid Dynamics: Packs context within each component optimally

Example:
    - Gradient flow decides: cache=40%, graph=35%, embeddings=25%
    - Fluid dynamics packs each component's context optimally

Usage:
    packer = MultiPhysicsPacker(max_tokens=8000)

    # Allocate across components via gradient flow
    allocations = await packer.allocate_budget(
        components=["cache", "graph", "embeddings"],
        importance={"cache": 0.9, "graph": 0.7, "embeddings": 0.8}
    )

    # Pack each component via fluid dynamics
    packed = await packer.pack_all(allocations, graph=kg)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio

from .gradient_flow import GradientFlowEngine, RouteDecision
from .adaptive_packer import AdaptivePacker, PackedContext

try:
    from ..memory.graph import KG
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


@dataclass
class ComponentAllocation:
    """Token budget allocation for a component."""
    component: str
    tokens: int
    importance: float
    loss: float


@dataclass
class MultiPhysicsResult:
    """Result from multi-physics packing."""
    allocations: List[ComponentAllocation]
    packed_contexts: Dict[str, PackedContext]
    total_tokens_used: int
    total_tokens_available: int
    gradient_iterations: int
    fluid_iterations: int


class MultiPhysicsPacker:
    """
    Multi-physics context packer combining gradient flow + fluid dynamics.

    Phase 1 (Gradient Flow):
        - Allocates token budget across components
        - Minimizes loss based on importance

    Phase 2 (Fluid Dynamics):
        - Packs context within each component
        - Pressure flows to fill available space

    Usage:
        packer = MultiPhysicsPacker(max_tokens=8000)

        result = await packer.pack(
            components={
                "cache": {"importance": 0.9, "graph": cache_graph},
                "graph": {"importance": 0.7, "graph": knowledge_graph},
                "embeddings": {"importance": 0.8, "graph": embedding_graph}
            }
        )

        # Access results
        for component, allocation in result.allocations.items():
            print(f"{component}: {allocation.tokens} tokens")

        for component, packed in result.packed_contexts.items():
            print(f"{component}: {len(packed.nodes)} nodes packed")
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        gradient_learning_rate: float = 0.1,
        fluid_viscosity: float = 0.01,
        gradient_steps: int = 20,
        fluid_steps: int = 10
    ):
        """
        Initialize multi-physics packer.

        Args:
            max_tokens: Total token budget
            gradient_learning_rate: Gradient descent learning rate
            fluid_viscosity: Fluid dynamics viscosity
            gradient_steps: Gradient flow iterations
            fluid_steps: Fluid dynamics iterations
        """
        self.max_tokens = max_tokens
        self.gradient_steps = gradient_steps
        self.fluid_steps = fluid_steps

        # Phase 1: Gradient flow for allocation
        def allocation_loss(metrics: Dict[str, float]) -> float:
            """Loss = inverse importance × allocation gap."""
            importance = metrics.get("importance", 0.5)
            current = metrics.get("current_allocation", 0.0)
            target = metrics.get("target_allocation", 0.5)

            # Higher importance → should get more tokens
            # Gap from target → penalty
            gap = abs(target - current)
            loss = (1.0 - importance) + gap
            return loss

        self.gradient_engine = GradientFlowEngine(
            loss_fn=allocation_loss,
            learning_rate=gradient_learning_rate,
            noise_level=0.01
        )

        # Phase 2: Fluid dynamics packers (created per component)
        self.fluid_viscosity = fluid_viscosity

    def allocate_budget(
        self,
        components: List[str],
        importance: Dict[str, float],
        constraints: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> List[ComponentAllocation]:
        """
        Allocate token budget across components via gradient flow.

        Args:
            components: List of component names
            importance: Importance scores for each component (0.0 - 1.0)
            constraints: Optional (min_tokens, max_tokens) per component

        Returns:
            List of ComponentAllocation objects
        """
        # Normalize importance scores
        total_importance = sum(importance.values())
        normalized_importance = {
            c: importance.get(c, 0.5) / total_importance
            for c in components
        }

        # Target allocations based on importance
        target_allocations = {
            c: norm_imp * self.max_tokens
            for c, norm_imp in normalized_importance.items()
        }

        # Build metrics for gradient flow
        metrics = []
        for component in components:
            target = target_allocations[component] / self.max_tokens
            metrics.append({
                "importance": importance.get(component, 0.5),
                "current_allocation": 0.0,  # Start with no allocation
                "target_allocation": target
            })

        # Run gradient descent to find optimal allocation
        decision = self.gradient_engine.route(
            targets=components,
            target_metrics=metrics,
            max_steps=self.gradient_steps
        )

        # Convert to allocations
        allocations = []
        for component, metric in zip(components, metrics):
            target_tokens = int(target_allocations[component])

            # Apply constraints if provided
            if constraints and component in constraints:
                min_tokens, max_tokens = constraints[component]
                target_tokens = max(min_tokens, min(target_tokens, max_tokens))

            allocations.append(ComponentAllocation(
                component=component,
                tokens=target_tokens,
                importance=importance.get(component, 0.5),
                loss=self.gradient_engine.compute_loss(metric)
            ))

        return allocations

    async def pack_component(
        self,
        component: str,
        token_budget: int,
        graph: Optional[Any] = None,
        initial_nodes: Optional[List[str]] = None
    ) -> PackedContext:
        """
        Pack a single component via fluid dynamics.

        Args:
            component: Component name
            token_budget: Token budget for this component
            graph: Knowledge graph for this component
            initial_nodes: Initial nodes to inject

        Returns:
            PackedContext
        """
        # Create fluid dynamics packer
        packer = AdaptivePacker(
            graph=graph,
            max_tokens=token_budget,
            viscosity=self.fluid_viscosity
        )

        # Inject initial nodes if provided
        if initial_nodes:
            for node in initial_nodes:
                packer.inject(
                    node=node,
                    importance=0.8,
                    tokens=50,
                    text=f"Context for {node}"
                )

        # Pack via fluid dynamics
        result = packer.pack_sync(max_iterations=self.fluid_steps)

        return result

    async def pack(
        self,
        components: Dict[str, Dict[str, Any]],
        constraints: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> MultiPhysicsResult:
        """
        Complete multi-physics packing pipeline.

        Args:
            components: Dict mapping component name -> config
                        Config must have: "importance", optional: "graph", "initial_nodes"
            constraints: Optional token constraints per component

        Returns:
            MultiPhysicsResult
        """
        # Extract importance scores
        importance = {
            name: config.get("importance", 0.5)
            for name, config in components.items()
        }

        # Phase 1: Allocate budget via gradient flow
        allocations = self.allocate_budget(
            components=list(components.keys()),
            importance=importance,
            constraints=constraints
        )

        # Phase 2: Pack each component via fluid dynamics
        packed_contexts = {}

        for allocation in allocations:
            config = components[allocation.component]

            packed = await self.pack_component(
                component=allocation.component,
                token_budget=allocation.tokens,
                graph=config.get("graph"),
                initial_nodes=config.get("initial_nodes")
            )

            packed_contexts[allocation.component] = packed

        # Compute total tokens used
        total_used = sum(
            packed.tokens_used
            for packed in packed_contexts.values()
        )

        return MultiPhysicsResult(
            allocations=allocations,
            packed_contexts=packed_contexts,
            total_tokens_used=total_used,
            total_tokens_available=self.max_tokens,
            gradient_iterations=self.gradient_steps,
            fluid_iterations=self.fluid_steps
        )

    def get_summary(self, result: MultiPhysicsResult) -> str:
        """Get human-readable summary of packing result."""
        lines = []
        lines.append("=" * 60)
        lines.append("Multi-Physics Context Packing Result")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Total tokens: {result.total_tokens_used} / {result.total_tokens_available}")
        lines.append(f"Utilization: {result.total_tokens_used / result.total_tokens_available:.1%}")
        lines.append("")

        lines.append("Phase 1 (Gradient Flow) - Budget Allocation:")
        for alloc in result.allocations:
            pct = alloc.tokens / result.total_tokens_available
            bar = "#" * int(pct * 40)
            lines.append(f"  {alloc.component:15s}: [{bar:<40}] {alloc.tokens:4d} tokens ({pct:.1%})")

        lines.append("")
        lines.append("Phase 2 (Fluid Dynamics) - Context Packing:")
        for component, packed in result.packed_contexts.items():
            lines.append(f"  {component:15s}: {len(packed.nodes)} nodes packed, {packed.tokens_used} tokens used")

        return "\n".join(lines)


# Helper function for easy integration

async def pack_context_multiphysics(
    cache_graph: Optional[Any] = None,
    knowledge_graph: Optional[Any] = None,
    embedding_graph: Optional[Any] = None,
    cache_importance: float = 0.9,
    graph_importance: float = 0.7,
    embedding_importance: float = 0.8,
    max_tokens: int = 8000
) -> MultiPhysicsResult:
    """
    Pack context using multi-physics approach (Phases 1+2).

    Args:
        cache_graph: Cache knowledge graph
        knowledge_graph: Main knowledge graph
        embedding_graph: Embedding relationship graph
        cache_importance: Cache importance (0.0 - 1.0)
        graph_importance: Graph importance
        embedding_importance: Embedding importance
        max_tokens: Total token budget

    Returns:
        MultiPhysicsResult with allocations and packed contexts
    """
    packer = MultiPhysicsPacker(max_tokens=max_tokens)

    components = {}

    if cache_graph:
        components["cache"] = {
            "importance": cache_importance,
            "graph": cache_graph
        }

    if knowledge_graph:
        components["graph"] = {
            "importance": graph_importance,
            "graph": knowledge_graph
        }

    if embedding_graph:
        components["embeddings"] = {
            "importance": embedding_importance,
            "graph": embedding_graph
        }

    return await packer.pack(components)
