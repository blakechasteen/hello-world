"""
Multi-Wave Memory Engine - Complete Sleep-Wake Cycle with Streaming Ingestion
===============================================================================

This module implements the complete brain wave cycle for memory management:
- Beta waves (active retrieval)
- Alpha waves (relaxed filtering)
- Theta waves (light sleep consolidation)
- Delta waves (deep sleep pruning)
- REM sleep (dreaming, random replay)

Plus streaming ingestion from SpinningWheel data sources.

Author: HoloLoom Memory Team
Date: October 30, 2025
"""

from typing import Dict, List, Optional, Set, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np
import random
import logging

from .spring_dynamics_engine import (
    SpringDynamicsEngine,
    SpringEngineConfig,
    MemoryNode,
    BetaWaveRecallResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# Brain Wave Modes
# ============================================================================

class BrainWaveMode(Enum):
    """Brain wave frequency modes corresponding to different states."""
    BETA = "beta"        # 13-30 Hz - Active retrieval (awake)
    ALPHA = "alpha"      # 8-13 Hz - Relaxed filtering (awake, resting)
    THETA = "theta"      # 4-8 Hz - Light sleep consolidation
    DELTA = "delta"      # 0.5-4 Hz - Deep sleep reorganization
    REM = "rem"          # Mixed - Dreaming, random replay


# ============================================================================
# Theta Wave Consolidation (Light Sleep)
# ============================================================================

@dataclass
class ActivationPattern:
    """Record of which nodes were active together."""
    nodes: List[str]
    timestamp: datetime
    source: str  # 'query' or 'ingestion'


class ThetaWaveConsolidator:
    """
    Theta wave consolidation - strengthen frequently co-activated pairs.

    This runs during "light sleep" (system idle 30min-2hours) and learns
    from recent activation patterns to create permanent new connections.
    """

    def __init__(self, engine: 'MultiWaveMemoryEngine'):
        self.engine = engine
        self.co_activation_history: List[ActivationPattern] = []
        self.consolidation_threshold = 3  # Co-occur at least 3 times
        self.max_history = 100

    def record_activation_pattern(
        self,
        activations: Dict[str, float],
        source: str = 'query',
        threshold: float = 0.3
    ):
        """
        Record which nodes were active together.

        Args:
            activations: {node_id: activation_level}
            source: 'query' or 'ingestion'
            threshold: Minimum activation to be considered "active"
        """
        active_nodes = [
            node_id for node_id, act in activations.items()
            if act > threshold
        ]

        if len(active_nodes) >= 2:  # Need at least 2 nodes
            pattern = ActivationPattern(
                nodes=active_nodes,
                timestamp=datetime.now(),
                source=source
            )
            self.co_activation_history.append(pattern)

            # Keep last N patterns
            if len(self.co_activation_history) > self.max_history:
                self.co_activation_history.pop(0)

    def theta_consolidation_update(self):
        """
        Theta wave update: Strengthen connections between frequently
        co-activated nodes.

        This creates PERMANENT changes to the graph structure.
        """
        # Find node pairs that frequently appear together
        co_occurrence_counts: Dict[tuple, int] = {}

        for pattern in self.co_activation_history[-50:]:  # Last 50 patterns
            nodes = pattern.nodes

            # Count co-occurrences
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i+1:]:
                    pair = tuple(sorted([node_a, node_b]))
                    co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1

        # Strengthen frequently co-activated pairs
        consolidated = 0
        for (node_a, node_b), count in co_occurrence_counts.items():
            if count >= self.consolidation_threshold:
                # Strengthen connection
                if self._strengthen_connection(node_a, node_b, strength=count / 10.0):
                    consolidated += 1

        if consolidated > 0:
            logger.info(f"[THETA] Consolidated {consolidated} connections")

        return consolidated

    def _strengthen_connection(self, node_a: str, node_b: str, strength: float) -> bool:
        """
        Strengthen connection between two nodes.

        Returns:
            True if connection was strengthened
        """
        if node_a not in self.engine.nodes or node_b not in self.engine.nodes:
            return False

        n_a = self.engine.nodes[node_a]
        n_b = self.engine.nodes[node_b]

        # Add to neighbors (for future spreading)
        n_a.neighbors.add(node_b)
        n_b.neighbors.add(node_a)

        # Pull rest positions slightly closer (permanent change!)
        direction = n_b.rest_position - n_a.rest_position
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 1e-6:
            # Move rest positions closer
            pull_strength = strength * 0.05  # Small permanent shift
            n_a.rest_position += pull_strength * (direction / direction_norm)
            n_b.rest_position -= pull_strength * (direction / direction_norm)

        return True


# ============================================================================
# Delta Wave Pruning (Deep Sleep)
# ============================================================================

class DeltaWavePruner:
    """
    Delta wave pruning - remove weak connections, strengthen strong ones.

    This runs during "deep sleep" (system idle >2 hours) and aggressively
    prunes unused connections.
    """

    def __init__(self, engine: 'MultiWaveMemoryEngine'):
        self.engine = engine
        self.weak_threshold = 0.3
        self.strong_threshold = 5.0
        self.prune_after_hours = 72  # 3 days

    def delta_pruning_update(self):
        """
        Delta wave update: Prune weak connections, strengthen strong ones.

        This is aggressive optimization - removes noise accumulated during
        awake time.
        """
        now = datetime.now()
        weak_nodes = []
        pruned = 0
        strengthened = 0

        # 1. Identify weak connections
        for node_id, node in self.engine.nodes.items():
            if node.spring_constant < self.weak_threshold:
                hours_since_access = (now - node.last_accessed).total_seconds() / 3600

                if hours_since_access > self.prune_after_hours:
                    weak_nodes.append(node_id)

        # 2. Prune weak connections
        for node_id in weak_nodes:
            node = self.engine.nodes[node_id]

            # Remove from all neighbors
            for neighbor_id in list(node.neighbors):
                if neighbor_id in self.engine.nodes:
                    self.engine.nodes[neighbor_id].neighbors.discard(node_id)

            # Clear its neighbors
            node.neighbors.clear()

            # Reduce spring constant further (almost forgotten)
            node.spring_constant = max(
                node.min_spring_constant,
                node.spring_constant * 0.5
            )
            pruned += 1

        # 3. Strengthen strong patterns
        for node_id, node in self.engine.nodes.items():
            if node.spring_constant > self.strong_threshold:
                # This is important knowledge - consolidate it
                node.spring_constant = min(
                    node.max_spring_constant,
                    node.spring_constant * 1.05
                )

                # Increase resistance to decay
                node.decay_rate *= 0.95  # Decays 5% slower
                strengthened += 1

        # 4. Reset velocities (calm down activation momentum)
        for node in self.engine.nodes.values():
            node.velocity *= 0.1  # Dramatic damping

        if pruned > 0 or strengthened > 0:
            logger.info(f"[DELTA] Pruned {pruned} weak nodes, strengthened {strengthened} strong nodes")

        return pruned, strengthened


# ============================================================================
# REM Sleep / Dreaming
# ============================================================================

class REMDreamer:
    """
    REM sleep dreaming - random replay creates unexpected connections.

    This is where CREATIVITY happens - random activations lead to
    novel associations that wouldn't occur during normal retrieval.
    """

    def __init__(self, engine: 'MultiWaveMemoryEngine'):
        self.engine = engine
        self.dream_intensity = 0.8
        self.random_seed_count = 3
        self.bridge_distance_threshold = 1.5

    async def dream_cycle(self, duration_seconds: float = 10.0):
        """
        Run a dream cycle: random activation sequences.

        Args:
            duration_seconds: How long to dream (default 10s)
        """
        start_time = datetime.now()
        bridges_created = 0

        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            # 1. Pick random seed nodes
            if len(self.engine.nodes) < self.random_seed_count:
                break

            seed_nodes = random.sample(
                list(self.engine.nodes.keys()),
                k=min(self.random_seed_count, len(self.engine.nodes))
            )

            # 2. Activate with random intensities
            activations = {node_id: 0.0 for node_id in self.engine.nodes.keys()}
            for seed_id in seed_nodes:
                activations[seed_id] = random.uniform(0.5, 1.0) * self.dream_intensity

            # 3. Let activation spread (chaotic, like beta but random seeds)
            for iteration in range(20):
                new_activations = activations.copy()

                for node_id, current_activation in activations.items():
                    if current_activation < 0.01:
                        continue

                    node = self.engine.nodes[node_id]

                    # Spread to neighbors (more chaotic than awake)
                    for neighbor_id in node.neighbors:
                        if neighbor_id not in self.engine.nodes:
                            continue

                        neighbor = self.engine.nodes[neighbor_id]
                        conductivity = neighbor.spring_constant / neighbor.max_spring_constant

                        # Dream spreading is more chaotic (higher transfer)
                        transferred = current_activation * conductivity * 0.5
                        new_activations[neighbor_id] += transferred

                    # Faster decay in dreams
                    new_activations[node_id] *= 0.85

                activations = new_activations

            # 4. CREATIVE INSIGHT: Connect distant nodes activated together
            highly_activated = [
                node_id for node_id, act in activations.items()
                if act > 0.3
            ]

            # Create NEW connections between activated nodes (even if distant)
            if len(highly_activated) >= 2:
                for _ in range(min(3, len(highly_activated) // 2)):
                    node_a, node_b = random.sample(highly_activated, k=2)

                    # Check if they're semantically distant
                    if node_a in self.engine.nodes and node_b in self.engine.nodes:
                        distance = np.linalg.norm(
                            self.engine.nodes[node_a].rest_position -
                            self.engine.nodes[node_b].rest_position
                        )

                        if distance > self.bridge_distance_threshold:
                            # Create bridge! (This is creative insight)
                            self.engine.nodes[node_a].neighbors.add(node_b)
                            self.engine.nodes[node_b].neighbors.add(node_a)
                            bridges_created += 1

                            logger.info(f"[DREAM] Created bridge: {node_a[:20]}... ↔ {node_b[:20]}... (distance: {distance:.2f})")

            # Brief pause between dream sequences
            await asyncio.sleep(0.5)

        return bridges_created


# ============================================================================
# Multi-Wave Memory Engine (Complete System)
# ============================================================================

class MultiWaveMemoryEngine(SpringDynamicsEngine):
    """
    Complete sleep-wake cycle memory engine with streaming ingestion.

    Automatically switches between brain wave modes based on activity:
    - Beta: Active queries (<5 min idle)
    - Alpha: Relaxed filtering (5-30 min idle)
    - Theta: Light sleep consolidation (30 min-2 hours idle)
    - Delta/REM: Deep sleep cycle (>2 hours idle)

    Accepts streaming data from SpinningWheel sources.
    """

    def __init__(self, config: Optional[SpringEngineConfig] = None):
        super().__init__(config)

        # Current brain wave mode
        self.mode = BrainWaveMode.BETA
        self.last_query_time = datetime.now()

        # Mode-specific handlers
        self.theta_consolidator = ThetaWaveConsolidator(self)
        self.delta_pruner = DeltaWavePruner(self)
        self.rem_dreamer = REMDreamer(self)

        # Streaming ingestion
        self.ingestion_active = False
        self.total_ingested = 0

    # ========================================================================
    # Enhanced Dynamics Loop with Mode Switching
    # ========================================================================

    async def _dynamics_loop(self):
        """
        Enhanced dynamics loop with automatic mode switching.
        """
        while self.running:
            try:
                # Update mode based on time since last query
                self._update_mode()

                # Run mode-specific update
                if self.mode == BrainWaveMode.BETA:
                    # Fast active retrieval (100ms updates)
                    if self.active_nodes:
                        self._update_physics(dt=0.1)
                        self._apply_forgetting_decay()
                        self._update_active_set()
                    await asyncio.sleep(0.1)

                elif self.mode == BrainWaveMode.ALPHA:
                    # Relaxed filtering (125ms updates)
                    self._alpha_filtering_update()
                    await asyncio.sleep(0.125)

                elif self.mode == BrainWaveMode.THETA:
                    # Light sleep consolidation (250ms updates)
                    self.theta_consolidator.theta_consolidation_update()
                    await asyncio.sleep(0.25)

                elif self.mode == BrainWaveMode.DELTA:
                    # Deep sleep reorganization (1s updates)
                    self.delta_pruner.delta_pruning_update()
                    await asyncio.sleep(1.0)

                elif self.mode == BrainWaveMode.REM:
                    # Dreaming - random replay (10s cycles)
                    await self.rem_dreamer.dream_cycle(duration_seconds=10.0)

                self.total_updates += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dynamics loop: {e}")
                await asyncio.sleep(1.0)

    def _update_mode(self):
        """Switch brain wave modes based on time since last query."""
        # Don't change mode during active ingestion
        if self.ingestion_active:
            self.mode = BrainWaveMode.BETA
            return

        elapsed_minutes = (datetime.now() - self.last_query_time).total_seconds() / 60.0

        if elapsed_minutes < 5:
            new_mode = BrainWaveMode.BETA  # Active
        elif elapsed_minutes < 30:
            new_mode = BrainWaveMode.ALPHA  # Resting
        elif elapsed_minutes < 120:
            new_mode = BrainWaveMode.THETA  # Light sleep
        else:
            # Alternate between delta and REM (70% delta, 30% REM)
            new_mode = BrainWaveMode.DELTA if random.random() < 0.7 else BrainWaveMode.REM

        if new_mode != self.mode:
            logger.info(f"[MODE] Switching from {self.mode.value} to {new_mode.value}")
            self.mode = new_mode

    def _alpha_filtering_update(self):
        """Alpha wave filtering: suppress weak activations."""
        alpha_threshold = 0.2

        for node in list(self.active_nodes):
            n = self.nodes.get(node)
            if not n:
                self.active_nodes.discard(node)
                continue

            # Suppress weak activations (faster decay)
            velocity_magnitude = np.linalg.norm(n.velocity)

            if velocity_magnitude < alpha_threshold:
                n.velocity *= 0.9  # Faster damping
                n.spring_constant *= 0.99  # Slight weakening

            # Strengthen clear signals
            elif n.spring_constant > 3.0:
                n.spring_constant *= 1.005  # Slight boost

    # ========================================================================
    # Query Interface (Wake Up to Beta Mode)
    # ========================================================================

    def on_query(self, query_embedding: np.ndarray) -> BetaWaveRecallResult:
        """
        Called when user makes a query - wake up to beta mode!

        Returns:
            BetaWaveRecallResult with recalled memories
        """
        self.last_query_time = datetime.now()
        self.mode = BrainWaveMode.BETA

        # Beta wave retrieval
        result = self.retrieve_memories(query_embedding)

        # Record activation pattern for theta consolidation
        self.theta_consolidator.record_activation_pattern(
            result.all_activations,
            source='query'
        )

        return result

    # ========================================================================
    # Streaming Ingestion from SpinningWheel
    # ========================================================================

    async def ingest_stream(
        self,
        shard_stream: AsyncIterator,
        embedding_func: Callable[[str], np.ndarray]
    ):
        """
        Ingest streaming data from SpinningWheel.

        Args:
            shard_stream: AsyncIterator yielding MemoryShards
            embedding_func: Function to embed shard content
        """
        self.ingestion_active = True
        logger.info("[INGEST] Starting stream ingestion")

        try:
            async for shard in shard_stream:
                # Encode new memory (beta wave encoding)
                embedding = embedding_func(shard.content)

                self.encode_new_memory(
                    node_id=shard.id,
                    content=shard.content,
                    embedding=embedding,
                    initial_k=1.0  # Fresh memory
                )

                self.total_ingested += 1

                # Brief pause to not overwhelm system
                await asyncio.sleep(0.01)

        finally:
            self.ingestion_active = False
            logger.info(f"[INGEST] Stream completed ({self.total_ingested} total memories)")

    def encode_new_memory(
        self,
        node_id: str,
        content: str,
        embedding: np.ndarray,
        initial_k: float = 1.0
    ):
        """
        Encode new memory (beta wave encoding during ingestion).

        Args:
            node_id: Unique node identifier
            content: Text content
            embedding: Semantic embedding
            initial_k: Initial spring constant (default 1.0)
        """
        # Find semantically similar nodes (for neighbor connections)
        similarities = []
        for existing_id, existing_node in self.nodes.items():
            # Cosine similarity
            emb_norm = np.linalg.norm(embedding)
            exist_norm = np.linalg.norm(existing_node.rest_position)

            if emb_norm > 1e-6 and exist_norm > 1e-6:
                cosine_sim = np.dot(embedding, existing_node.rest_position) / (emb_norm * exist_norm)
                if cosine_sim > 0.7:  # High similarity threshold
                    similarities.append((existing_id, cosine_sim))

        # Connect to top-3 most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        neighbors = {nid for nid, _ in similarities[:3]}

        # Add node
        self.add_node(
            node_id=node_id,
            embedding=embedding,
            content=content,
            neighbors=neighbors,
            initial_spring_constant=initial_k
        )

        # Record activation pattern (for theta consolidation)
        if neighbors:
            activations = {node_id: 1.0}
            for neighbor_id in neighbors:
                activations[neighbor_id] = 0.5

            self.theta_consolidator.record_activation_pattern(
                activations,
                source='ingestion'
            )

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict:
        """Get enhanced statistics with mode info."""
        base_stats = super().get_statistics()

        base_stats.update({
            'mode': self.mode.value,
            'minutes_since_last_query': (datetime.now() - self.last_query_time).total_seconds() / 60.0,
            'total_ingested': self.total_ingested,
            'ingestion_active': self.ingestion_active,
            'consolidation_history_size': len(self.theta_consolidator.co_activation_history)
        })

        return base_stats


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'BrainWaveMode',
    'MultiWaveMemoryEngine',
    'ThetaWaveConsolidator',
    'DeltaWavePruner',
    'REMDreamer',
]
