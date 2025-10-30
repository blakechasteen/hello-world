"""
Spring Dynamics Engine - Background Memory Degradation System
==============================================================
Physics-inspired living memory with recall strengthening and natural forgetting.

This module implements a continuous background process that models memory
as a spring-mass system in semantic space. Unlike the activation-based
spring_dynamics.py (which handles query-time spreading), this engine runs
continuously to manage memory lifecycle:

- **Recall strengthens**: Frequently accessed memories get stronger springs
- **Forgetting happens naturally**: Unused memories decay exponentially
- **Activation spreads**: Accessing one memory primes related ones

The engine maintains each memory node's position in semantic space, spring
strength (recall probability), and velocity (activation momentum).

Physics Model:
--------------
1. Spring Force (Hooke's Law): F = -k × (x - x0)
   - k: spring constant (recall strength)
   - x: current position
   - x0: rest position (initial embedding)

2. Velocity Update: v += (F / m) × dt
3. Position Update: x += v × dt
4. Damping: v *= damping_factor (energy loss)
5. Forgetting: k(t) = k0 × exp(-λt) (exponential decay)

Background Loop (100ms updates):
---------------------------------
Every 100ms:
1. Calculate spring forces for all active nodes
2. Update velocities (F = ma)
3. Update positions
4. Apply damping (energy loss)
5. Decay spring constants (forgetting)
6. Record state for retrieval scoring

Author: HoloLoom Memory Team
Date: October 30, 2025
"""

from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Node Data Structure
# ============================================================================

@dataclass
class MemoryNode:
    """
    A memory node in the spring-mass dynamics system.

    Each memory has:
    - Position in semantic space (continuous representation)
    - Spring constant k (recall strength - higher = easier to recall)
    - Velocity (activation momentum)
    - Lifecycle metadata (access count, timestamps)
    """

    # Identity
    id: str                           # Node ID (matches backend memory ID)
    content: str                      # Text content (for logging/debugging)

    # Physics State (in semantic space)
    position: np.ndarray              # Current position (x, y, z, ...) in embedding space
    rest_position: np.ndarray         # Rest position (initial embedding)
    velocity: np.ndarray              # Velocity (activation momentum)
    mass: float = 1.0                 # Mass (for future extensions)

    # Spring Properties (Recall Strength)
    spring_constant: float = 1.0      # k - recall strength (0.1-10.0)
    rest_length: float = 0.0          # Rest length (typically 0 for memory at rest)

    # Lifecycle Tracking
    access_count: int = 0             # How many times accessed
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

    # Decay Parameters
    decay_rate: float = 0.01          # λ: How fast spring weakens (0.001-0.1)
    min_spring_constant: float = 0.1  # Minimum k (never forget completely)
    max_spring_constant: float = 10.0 # Maximum k (cap reinforcement)

    # Neighbor Connections (for activation spreading)
    neighbors: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Ensure numpy arrays are correct shape."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.rest_position, np.ndarray):
            self.rest_position = np.array(self.rest_position, dtype=np.float32)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.zeros_like(self.position, dtype=np.float32)

    def get_displacement(self) -> np.ndarray:
        """Get displacement from rest position: (x - x0)."""
        return self.position - self.rest_position

    def get_spring_force(self) -> np.ndarray:
        """
        Calculate spring force: F = -k × (x - x0).

        This pulls the node back toward its rest position (initial embedding).
        """
        displacement = self.get_displacement()
        return -self.spring_constant * displacement

    def get_recall_strength(self) -> float:
        """
        Get normalized recall strength [0, 1].

        Based on spring constant relative to min/max range.
        """
        k_range = self.max_spring_constant - self.min_spring_constant
        return (self.spring_constant - self.min_spring_constant) / k_range if k_range > 0 else 0.5


# ============================================================================
# Spring Dynamics Engine Configuration
# ============================================================================

@dataclass
class SpringEngineConfig:
    """Configuration for background spring dynamics engine."""

    # Physics Parameters
    damping: float = 0.95              # Velocity damping (0.9-0.99, higher = less damping)
    dt: float = 0.1                    # Time step in seconds (100ms default)

    # Forgetting Parameters
    base_decay_rate: float = 0.01      # Base decay rate λ (per hour)
    decay_scale_factor: float = 1.0    # Scale factor for individual decay rates

    # Recall Strengthening
    access_boost: float = 0.5          # How much to boost k on access (0.1-1.0)
    velocity_boost: float = 2.0        # Velocity magnitude added on access

    # Activation Spreading
    propagation_factor: float = 0.5    # How much activation spreads to neighbors (0-1)
    propagation_decay: float = 0.8     # How much activation decays per hop
    max_propagation_hops: int = 2      # Maximum hops for activation spreading

    # Background Loop
    update_interval_ms: int = 100      # How often to update (milliseconds)
    max_active_nodes: int = 1000       # Maximum nodes to track in active set

    # Energy Thresholds
    min_velocity_threshold: float = 0.01   # Below this, node considered at rest
    sleep_after_inactive_hours: int = 24   # Put node to sleep after this long


# ============================================================================
# Spring Dynamics Engine - Background Memory Management
# ============================================================================

class SpringDynamicsEngine:
    """
    Continuous background process for memory lifecycle management.

    This engine runs asynchronously in the background, updating memory
    node physics every 100ms. It handles:

    1. Recall strengthening: Accessing memories increases spring constant
    2. Natural forgetting: Unused memories decay exponentially
    3. Activation spreading: Accessing one memory primes related ones
    4. Position dynamics: Nodes move in semantic space based on forces

    Usage:
        # Create engine
        engine = SpringDynamicsEngine(config)

        # Add memories
        engine.add_node('node_1', embedding, content='...', neighbors={'node_2'})

        # Start background loop
        await engine.start()

        # Access memory (strengthens it)
        engine.on_memory_accessed('node_1', query_embedding, pulse_strength=0.5)

        # Stop background loop
        await engine.stop()
    """

    def __init__(self, config: Optional[SpringEngineConfig] = None):
        """
        Initialize spring dynamics engine.

        Args:
            config: SpringEngineConfig (or None for defaults)
        """
        self.config = config or SpringEngineConfig()

        # Memory nodes: {node_id: MemoryNode}
        self.nodes: Dict[str, MemoryNode] = {}

        # Active set (nodes with significant velocity or recent access)
        self.active_nodes: Set[str] = set()

        # Background task
        self._task: Optional[asyncio.Task] = None
        self.running = False

        # Statistics
        self.total_updates = 0
        self.total_accesses = 0
        self.total_propagations = 0

        # Callbacks (for monitoring)
        self.on_node_activated: Optional[Callable[[str], None]] = None
        self.on_node_sleeping: Optional[Callable[[str], None]] = None

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def start(self):
        """Start background dynamics loop."""
        if self.running:
            logger.warning("SpringDynamicsEngine already running")
            return

        self.running = True
        self._task = asyncio.create_task(self._dynamics_loop())
        logger.info(f"SpringDynamicsEngine started (update interval: {self.config.update_interval_ms}ms)")

    async def stop(self):
        """Stop background dynamics loop."""
        if not self.running:
            return

        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"SpringDynamicsEngine stopped (total updates: {self.total_updates})")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    # ========================================================================
    # Background Dynamics Loop
    # ========================================================================

    async def _dynamics_loop(self):
        """
        Background loop: Update spring dynamics continuously.

        Every 100ms:
        1. Calculate spring forces for active nodes
        2. Update velocities (F = ma)
        3. Update positions
        4. Apply damping (energy loss)
        5. Decay spring constants (forgetting)
        6. Update active set
        """
        dt = self.config.dt

        while self.running:
            try:
                # Update physics for active nodes
                if self.active_nodes:
                    self._update_physics(dt)
                    self._apply_forgetting_decay()
                    self._update_active_set()
                    self.total_updates += 1

                # Sleep until next update
                await asyncio.sleep(self.config.update_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dynamics loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    def _update_physics(self, dt: float):
        """
        Update physics for all active nodes.

        Steps:
        1. Calculate forces (spring + damping)
        2. Update velocities (F = ma)
        3. Update positions (x += v × dt)
        4. Apply damping
        """
        # Calculate forces
        forces: Dict[str, np.ndarray] = {}

        for node_id in list(self.active_nodes):
            node = self.nodes.get(node_id)
            if not node:
                self.active_nodes.discard(node_id)
                continue

            # Spring force: F = -k × (x - x0)
            spring_force = node.get_spring_force()

            # Damping force: F_damp = -c × v (implicit in damping step)

            forces[node_id] = spring_force

        # Update velocities and positions
        for node_id, force in forces.items():
            node = self.nodes[node_id]

            # F = ma → a = F/m
            acceleration = force / node.mass

            # Update velocity: v += a × dt
            node.velocity += acceleration * dt

            # Update position: x += v × dt
            node.position += node.velocity * dt

            # Apply damping: v *= damping_factor
            node.velocity *= self.config.damping

    def _apply_forgetting_decay(self):
        """
        Apply exponential forgetting decay to spring constants.

        Uses Ebbinghaus forgetting curve:
        k(t) = k0 × exp(-λt)

        Where:
        - k(t): spring constant at time t
        - k0: initial spring constant
        - λ: decay rate
        - t: time since last access (in hours)
        """
        now = datetime.now()

        for node_id in list(self.active_nodes):
            node = self.nodes.get(node_id)
            if not node:
                continue

            # Time since last access (hours)
            hours_since_access = (now - node.last_accessed).total_seconds() / 3600.0

            # Exponential decay: k(t) = k0 × exp(-λt)
            decay_factor = np.exp(-node.decay_rate * hours_since_access)

            # Apply decay (but keep above minimum)
            node.spring_constant = max(
                node.min_spring_constant,
                node.spring_constant * decay_factor
            )

    def _update_active_set(self):
        """
        Update active set: Remove nodes at rest, add nodes with activity.

        A node is "at rest" if:
        - Velocity magnitude < threshold
        - Spring constant near minimum
        - Not accessed recently
        """
        now = datetime.now()
        inactive_threshold = timedelta(hours=self.config.sleep_after_inactive_hours)

        to_remove = set()

        for node_id in self.active_nodes:
            node = self.nodes.get(node_id)
            if not node:
                to_remove.add(node_id)
                continue

            # Check if at rest
            velocity_magnitude = np.linalg.norm(node.velocity)
            is_at_rest = (
                velocity_magnitude < self.config.min_velocity_threshold and
                node.spring_constant <= node.min_spring_constant * 1.1 and
                (now - node.last_accessed) > inactive_threshold
            )

            if is_at_rest:
                to_remove.add(node_id)
                if self.on_node_sleeping:
                    self.on_node_sleeping(node_id)

        # Remove inactive nodes
        self.active_nodes -= to_remove

        # Trim active set if too large
        if len(self.active_nodes) > self.config.max_active_nodes:
            # Keep most recently accessed nodes
            sorted_nodes = sorted(
                self.active_nodes,
                key=lambda nid: self.nodes[nid].last_accessed if nid in self.nodes else datetime.min,
                reverse=True
            )
            self.active_nodes = set(sorted_nodes[:self.config.max_active_nodes])

    # ========================================================================
    # Memory Node Management
    # ========================================================================

    def add_node(
        self,
        node_id: str,
        embedding: np.ndarray,
        content: str = "",
        neighbors: Optional[Set[str]] = None,
        initial_spring_constant: float = 1.0,
        decay_rate: Optional[float] = None
    ):
        """
        Add a memory node to the engine.

        Args:
            node_id: Unique node identifier
            embedding: Initial position in semantic space (rest position)
            content: Text content (for logging)
            neighbors: Set of neighbor node IDs (for activation spreading)
            initial_spring_constant: Starting recall strength (default 1.0)
            decay_rate: Custom decay rate (or None for base rate)
        """
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists, skipping add")
            return

        embedding = np.array(embedding, dtype=np.float32)

        node = MemoryNode(
            id=node_id,
            content=content,
            position=embedding.copy(),
            rest_position=embedding.copy(),
            velocity=np.zeros_like(embedding, dtype=np.float32),
            spring_constant=initial_spring_constant,
            decay_rate=decay_rate or self.config.base_decay_rate,
            neighbors=neighbors or set()
        )

        self.nodes[node_id] = node
        self.active_nodes.add(node_id)  # Start in active set

    def remove_node(self, node_id: str):
        """Remove a memory node from the engine."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.active_nodes.discard(node_id)

            # Remove from neighbors
            for node in self.nodes.values():
                node.neighbors.discard(node_id)

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a memory node by ID."""
        return self.nodes.get(node_id)

    # ========================================================================
    # Recall Strengthening (Memory Access)
    # ========================================================================

    def on_memory_accessed(
        self,
        node_id: str,
        query_embedding: Optional[np.ndarray] = None,
        pulse_strength: float = 0.5
    ):
        """
        Called when a memory is accessed (retrieved/used).

        This strengthens the memory:
        1. Increase spring constant k (recall reinforcement)
        2. Add velocity toward query point (if provided)
        3. Propagate activation to neighbors
        4. Update access metadata

        Args:
            node_id: Memory node that was accessed
            query_embedding: Query embedding (or None if no query context)
            pulse_strength: Strength of reinforcement (0-1, default 0.5)
        """
        node = self.nodes.get(node_id)
        if not node:
            logger.warning(f"Cannot access non-existent node: {node_id}")
            return

        # 1. Strengthen spring constant (recall reinforcement)
        boost = self.config.access_boost * pulse_strength
        node.spring_constant = min(
            node.max_spring_constant,
            node.spring_constant + boost
        )

        # 2. Add velocity toward query point (if provided)
        if query_embedding is not None:
            query_embedding = np.array(query_embedding, dtype=np.float32)

            # Direction from current position to query
            direction = query_embedding - node.position
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-6:
                # Add velocity in query direction
                velocity_magnitude = self.config.velocity_boost * pulse_strength
                node.velocity += velocity_magnitude * (direction / direction_norm)

        # 3. Propagate activation to neighbors
        self._propagate_activation(node_id, strength=pulse_strength)

        # 4. Update metadata
        node.access_count += 1
        node.last_accessed = datetime.now()

        # 5. Add to active set
        self.active_nodes.add(node_id)

        if self.on_node_activated:
            self.on_node_activated(node_id)

        self.total_accesses += 1

    def _propagate_activation(self, source_id: str, strength: float, depth: int = 0):
        """
        Propagate activation to neighbor nodes (priming effect).

        Args:
            source_id: Source node ID
            strength: Activation strength (decays with each hop)
            depth: Current recursion depth
        """
        if depth >= self.config.max_propagation_hops:
            return

        source_node = self.nodes.get(source_id)
        if not source_node:
            return

        # Propagate to neighbors
        propagated_strength = strength * self.config.propagation_factor * self.config.propagation_decay

        if propagated_strength < 0.05:  # Stop if activation too weak
            return

        for neighbor_id in source_node.neighbors:
            neighbor = self.nodes.get(neighbor_id)
            if not neighbor:
                continue

            # Boost neighbor spring constant (weaker than direct access)
            neighbor.spring_constant = min(
                neighbor.max_spring_constant,
                neighbor.spring_constant + propagated_strength * 0.2
            )

            # Add velocity toward source (attraction)
            direction = source_node.position - neighbor.position
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-6:
                neighbor.velocity += propagated_strength * (direction / direction_norm)

            # Add to active set
            self.active_nodes.add(neighbor_id)

            self.total_propagations += 1

            # Recursively propagate (with decay)
            self._propagate_activation(neighbor_id, propagated_strength, depth + 1)

    # ========================================================================
    # Retrieval via Beta Wave Activation Spreading
    # ========================================================================

    def retrieve_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        seed_strength: float = 1.0,
        activation_threshold: float = 0.1,
        max_iterations: int = 50
    ) -> 'BetaWaveRecallResult':
        """
        Retrieve memories via beta wave activation spreading.

        This is THE core retrieval mechanism. It models how beta wave
        oscillations create transient synchronization between brain regions,
        enabling both direct recall and creative associations.

        Process:
        1. Find seed nodes (semantically similar to query)
        2. Pulse seeds with activation
        3. Activation spreads through springs (k determines conductivity)
        4. High-activation nodes = recalled memories
        5. Distant activated nodes = creative insights

        The "call number" metaphor:
        - Each memory has an address (node ID)
        - Spring constant k = how fresh/accessible that call number is
        - Strong k = fresh, easy to recall
        - Weak k = faded, needs a strong reminder
        - Spreading = beta waves synchronizing brain regions

        Args:
            query_embedding: Query vector
            top_k: Number of memories to return
            seed_strength: Initial activation strength for seeds
            activation_threshold: Minimum activation to be "recalled"
            max_iterations: Maximum spreading iterations

        Returns:
            BetaWaveRecallResult with recalled memories and creative insights
        """
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # 1. Find seed nodes (most semantically similar)
        seed_nodes = self._find_seed_nodes(query_embedding, top_n=5)

        # 2. Initialize activation states
        activation = {node_id: 0.0 for node_id in self.nodes.keys()}
        for node_id, similarity in seed_nodes:
            activation[node_id] = seed_strength * similarity

        # 3. Spread activation through springs (beta wave propagation)
        for iteration in range(max_iterations):
            new_activation = activation.copy()

            for node_id, current_activation in activation.items():
                if current_activation < 0.01:  # Skip inactive nodes
                    continue

                node = self.nodes[node_id]

                # Spread to neighbors through springs
                for neighbor_id in node.neighbors:
                    if neighbor_id not in self.nodes:
                        continue

                    neighbor = self.nodes[neighbor_id]

                    # Conductivity = spring constant k (stronger springs conduct better)
                    conductivity = neighbor.spring_constant / neighbor.max_spring_constant

                    # Activation flows based on spring strength
                    transferred_activation = current_activation * conductivity * 0.3

                    new_activation[neighbor_id] += transferred_activation

                # Decay (beta wave damping)
                new_activation[node_id] *= 0.9

            # Check convergence
            max_change = max(abs(new_activation[nid] - activation[nid]) for nid in activation.keys())
            activation = new_activation

            if max_change < 0.001:
                break

        # 4. Extract recalled memories (high activation)
        recalled = [
            (node_id, act)
            for node_id, act in activation.items()
            if act >= activation_threshold
        ]
        recalled.sort(key=lambda x: x[1], reverse=True)

        # 5. Detect creative insights (distant but activated nodes)
        creative_insights = self._detect_creative_insights(
            seed_nodes=[s[0] for s in seed_nodes],
            activated_nodes=recalled,
            query_embedding=query_embedding
        )

        return BetaWaveRecallResult(
            recalled_memories=recalled[:top_k],
            all_activations=activation,
            creative_insights=creative_insights,
            seed_nodes=seed_nodes,
            iterations=iteration + 1
        )

    def _find_seed_nodes(
        self,
        query_embedding: np.ndarray,
        top_n: int = 5
    ) -> List[tuple[str, float]]:
        """
        Find seed nodes most semantically similar to query.

        Uses cosine similarity between query and rest_position (original embedding).

        Returns:
            List of (node_id, similarity) tuples
        """
        similarities = []

        for node_id, node in self.nodes.items():
            # Cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            rest_norm = np.linalg.norm(node.rest_position)

            if query_norm > 1e-6 and rest_norm > 1e-6:
                cosine_sim = np.dot(query_embedding, node.rest_position) / (query_norm * rest_norm)
                similarities.append((node_id, float(cosine_sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]

    def _detect_creative_insights(
        self,
        seed_nodes: List[str],
        activated_nodes: List[tuple[str, float]],
        query_embedding: np.ndarray
    ) -> List[tuple[str, float, str]]:
        """
        Detect creative insights: activated nodes that are semantically distant from query.

        These are memories that got activated through spreading but aren't
        directly similar to the query - cross-domain associations.

        Returns:
            List of (node_id, activation, insight_type) tuples
        """
        insights = []
        seed_set = set(seed_nodes)

        for node_id, activation in activated_nodes:
            if node_id in seed_set:
                continue  # Skip direct seeds

            node = self.nodes[node_id]

            # Check semantic distance from query
            distance = np.linalg.norm(node.rest_position - query_embedding)

            # If semantically distant but activated = creative insight
            if distance > 2.0 and activation > 0.15:
                # Classify insight type
                if activation > 0.5:
                    insight_type = "strong_association"  # Highly activated distant node
                elif len(node.neighbors & seed_set) > 0:
                    insight_type = "bridge_node"  # Connects to seeds
                else:
                    insight_type = "emergent_pattern"  # Multi-hop activation

                insights.append((node_id, activation, insight_type))

        # Sort by activation
        insights.sort(key=lambda x: x[1], reverse=True)

        return insights[:5]  # Top 5 creative insights

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict:
        """Get engine statistics."""
        active_node_list = list(self.active_nodes)

        return {
            'total_nodes': len(self.nodes),
            'active_nodes': len(self.active_nodes),
            'total_updates': self.total_updates,
            'total_accesses': self.total_accesses,
            'total_propagations': self.total_propagations,
            'running': self.running,
            'avg_spring_constant': np.mean([n.spring_constant for n in self.nodes.values()]) if self.nodes else 0.0,
            'avg_velocity': np.mean([np.linalg.norm(n.velocity) for n in self.nodes.values()]) if self.nodes else 0.0,
            'most_active_nodes': [
                {
                    'id': nid,
                    'access_count': self.nodes[nid].access_count,
                    'spring_constant': self.nodes[nid].spring_constant,
                    'velocity_magnitude': np.linalg.norm(self.nodes[nid].velocity)
                }
                for nid in active_node_list[:5]
            ] if active_node_list else []
        }


# ============================================================================
# Result Data Structures
# ============================================================================

@dataclass
class BetaWaveRecallResult:
    """
    Result of beta wave activation spreading retrieval.

    This captures both direct recalls and creative insights from spreading.
    """

    # Recalled memories (node_id, activation_level)
    recalled_memories: List[tuple[str, float]]

    # All node activations (for visualization/analysis)
    all_activations: Dict[str, float]

    # Creative insights (node_id, activation, insight_type)
    # These are semantically distant nodes activated through spreading
    creative_insights: List[tuple[str, float, str]]

    # Seed nodes (node_id, similarity_to_query)
    seed_nodes: List[tuple[str, float]]

    # Spreading metadata
    iterations: int  # How many iterations until convergence

    def get_direct_recalls(self) -> List[str]:
        """Get node IDs that were directly similar to query (seeds)."""
        return [node_id for node_id, _ in self.seed_nodes]

    def get_associative_recalls(self) -> List[str]:
        """Get node IDs recalled via spreading (not seeds)."""
        seed_ids = set(self.get_direct_recalls())
        return [
            node_id
            for node_id, _ in self.recalled_memories
            if node_id not in seed_ids
        ]

    def get_creative_insight_ids(self) -> List[str]:
        """Get node IDs identified as creative insights."""
        return [node_id for node_id, _, _ in self.creative_insights]

    def __str__(self) -> str:
        direct = len(self.get_direct_recalls())
        associative = len(self.get_associative_recalls())
        creative = len(self.creative_insights)
        return (
            f"BetaWaveRecall({len(self.recalled_memories)} recalled: "
            f"{direct} direct, {associative} associative, "
            f"{creative} creative insights, {self.iterations} iterations)"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'MemoryNode',
    'SpringEngineConfig',
    'SpringDynamicsEngine',
    'BetaWaveRecallResult',
]
