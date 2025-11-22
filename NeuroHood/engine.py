"""
NeuroHood Engine

Main orchestrator for consciousness-level neighborhood simulation.
Adapted from HoloLoom's DreamWeaver framework for modern suburban setting.
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json
from datetime import datetime
import random

from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.memory.graph import KG, KGEdge

from .config import NeuroHoodConfig, SimulationMode
from . import Resident, ResidentType, EventType, Relationship
from .social.spring_physics import RelationshipPhysics


@dataclass
class NeighborhoodState:
    """
    Complete snapshot of neighborhood at a point in time.

    Similar to DreamWeaver's WorldState but adapted for modern suburban context.
    """
    timestamp: float  # In-world time (seconds since start)
    residents: Dict[str, Resident]
    houses: Dict[str, Dict[str, Any]]  # {address: {owner_id, attributes}}
    active_conversations: List[Dict[str, Any]]
    recent_events: List[Dict[str, Any]]
    relationships: Dict[str, Relationship]  # {(id_a, id_b): Relationship}
    neighborhood_state: Dict[str, Any]  # Weather, time of day, community issues
    consistency_violations: List[Dict[str, Any]]
    simulation_context: Dict[str, Any]


class NeuroHood:
    """
    Main orchestrator for NeuroHood consciousness-level neighborhood simulator.

    Architecture:
        Player Input → Context Retrieval → Agent Processing → Social Physics →
        Consciousness Update → Learning → State Update

    Built on HoloLoom infrastructure:
    - WeavingOrchestrator for reasoning
    - Knowledge Graph (Yarn Graph) for memory
    - Collaborative Agents for residents
    - Spring Dynamics for social physics
    - Internal Dialogue for consciousness
    - Thompson Sampling for decision-making

    Usage:
        ```python
        from NeuroHood import NeuroHood, NeuroHoodConfig

        config = NeuroHoodConfig.prototype()

        async with NeuroHood(config) as hood:
            # Bootstrap neighborhood
            state = await hood.bootstrap_neighborhood(
                seed="A quiet suburban street with diverse neighbors"
            )

            # Evolve simulation
            state = await hood.step(
                player_action="Alice knocks on Bob's door to complain about noise",
                state=state
            )

            # Query resident state
            thoughts = await hood.get_resident_thoughts("alice", state)
        ```
    """

    def __init__(
        self,
        config: NeuroHoodConfig,
        knowledge_graph: Optional[KG] = None
    ):
        """
        Initialize NeuroHood.

        Args:
            config: NeuroHood configuration
            knowledge_graph: Optional existing knowledge graph
        """
        self.config = config
        self.kg: KG = knowledge_graph or KG()

        # HoloLoom integration
        self.hololoom_config = self._create_hololoom_config()
        self.orchestrator: Optional[WeavingOrchestrator] = None

        # State
        self.current_state: Optional[NeighborhoodState] = None
        self.state_history: List[NeighborhoodState] = []

        # Resident agents (initialized in __aenter__)
        self.resident_agents: Dict[str, Any] = {}  # {resident_id: ResidentAgent}

        # Social physics engine
        self.relationship_physics: Optional[RelationshipPhysics] = None

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._auto_save_task: Optional[asyncio.Task] = None
        self._learning_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = {
            "residents_created": 0,
            "events_generated": 0,
            "conversations": 0,
            "thoughts_generated": 0,
            "strange_loops_detected": 0,
            "relationships_formed": 0,
            "relationships_broken": 0,
            "player_interactions": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize HoloLoom orchestrator
        self.orchestrator = WeavingOrchestrator(
            cfg=self.hololoom_config,
            shards=self._create_initial_shards()
        )
        await self.orchestrator.__aenter__()

        # Start background tasks
        if self.config.auto_save and self.config.save_path:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())

        if self.config.enable_background_learning:
            self._learning_task = asyncio.create_task(self._learning_loop())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cancel background tasks
        for task in [self._auto_save_task, self._learning_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop all resident agents
        for agent in self.resident_agents.values():
            if hasattr(agent, '__aexit__'):
                await agent.__aexit__(None, None, None)

        # Final save
        if self.config.auto_save and self.current_state:
            await self.save_state(self.current_state)

        # Cleanup orchestrator
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

    # ========== Neighborhood Initialization ==========

    async def bootstrap_neighborhood(
        self,
        seed: str,
        initial_residents: Optional[List[Resident]] = None
    ) -> NeighborhoodState:
        """
        Bootstrap a new neighborhood from a seed description.

        Args:
            seed: Natural language description of neighborhood
            initial_residents: Optional starting residents

        Returns:
            Initial neighborhood state
        """
        # Generate neighborhood via HoloLoom
        query = Query(text=f"Generate modern suburban neighborhood: {seed}")
        spacetime = await self.orchestrator.weave(query)

        # Create houses
        houses = self._generate_houses(self.config.num_houses)

        # Create residents
        residents = initial_residents or await self._generate_residents(
            self.config.num_residents,
            houses,
            spacetime.response
        )

        # Initialize relationships
        relationships = await self._initialize_relationships(residents)

        # Create initial state
        state = NeighborhoodState(
            timestamp=0.0,
            residents={r.resident_id: r for r in residents},
            houses=houses,
            active_conversations=[],
            recent_events=[],
            relationships=relationships,
            neighborhood_state={
                "time_of_day": "morning",
                "weather": "sunny",
                "season": "spring",
                "seed": seed,
                "creation_time": datetime.utcnow().isoformat(),
            },
            consistency_violations=[],
            simulation_context={"last_action": seed}
        )

        # Initialize resident agents
        await self._initialize_agents(state)

        # Initialize relationship physics
        if self.config.enable_spring_dynamics:
            self.relationship_physics = RelationshipPhysics(self.config, self.kg)
            self.relationship_physics.initialize_from_relationships(
                state.residents,
                state.relationships
            )

        # Sync to knowledge graph
        await self._sync_state_to_kg(state)

        self.current_state = state
        self.state_history.append(state)

        return state

    async def load_state(self, world_id: str, load_path: Optional[str] = None) -> NeighborhoodState:
        """Load neighborhood from disk."""
        path = load_path or self.config.save_path
        if not path:
            raise ValueError("No save path configured")

        import os
        filepath = os.path.join(path, f"{world_id}.json")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = self._deserialize_state(data)

        # Reinitialize agents
        await self._initialize_agents(state)

        # Sync to KG
        await self._sync_state_to_kg(state)

        self.current_state = state
        self.state_history.append(state)

        return state

    async def save_state(self, state: NeighborhoodState, save_path: Optional[str] = None):
        """Save neighborhood to disk."""
        path = save_path or self.config.save_path
        if not path:
            return

        import os
        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, f"{self.config.world_id}.json")
        data = self._serialize_state(state)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ========== Simulation Evolution ==========

    async def step(
        self,
        player_action: str,
        state: Optional[NeighborhoodState] = None,
        time_delta: float = 60.0  # Advance 1 minute
    ) -> NeighborhoodState:
        """
        Evolve neighborhood forward with player action.

        Args:
            player_action: Player input/action
            state: Current state (defaults to self.current_state)
            time_delta: Time advancement in seconds

        Returns:
            Updated neighborhood state
        """
        state = state or self.current_state
        if not state:
            raise ValueError("No state available. Call bootstrap_neighborhood() first.")

        self.metrics["player_interactions"] += 1

        # 1. Advance time
        new_state = self._copy_state(state)
        new_state.timestamp += time_delta

        # 2. Process player action via HoloLoom
        query = Query(
            text=player_action,
            metadata={
                "world_id": self.config.world_id,
                "timestamp": new_state.timestamp,
                "context": "neighborhood_simulation",
            }
        )

        spacetime = await self.orchestrator.weave(query)

        # 3. Generate events from action
        events = await self._extract_events(spacetime.response, player_action, new_state)

        # 4. Apply events to state
        new_state = await self._apply_events(new_state, events)

        # 5. Update resident agents (they process events and update internal state)
        await self._update_agents(new_state, events)

        # 6. Run social physics (relationships evolve)
        if self.config.enable_spring_dynamics:
            new_state = await self._update_social_physics(new_state)

        # 7. Sync to KG
        await self._sync_state_to_kg(new_state)

        # 8. Store history
        self.current_state = new_state
        self.state_history.append(new_state)

        return new_state

    async def get_resident_thoughts(
        self,
        resident_id: str,
        state: Optional[NeighborhoodState] = None
    ) -> Dict[str, Any]:
        """
        Get a resident's current internal state (thoughts, feelings, memories).

        This is the "consciousness view" - seeing inside an NPC's mind.

        Args:
            resident_id: Resident to query
            state: Current state

        Returns:
            Dict with internal state, active thoughts, strange loops, etc.
        """
        state = state or self.current_state
        if not state:
            raise ValueError("No state available")

        resident = state.residents.get(resident_id)
        if not resident:
            raise ValueError(f"Resident {resident_id} not found")

        # Get agent internal state
        agent = self.resident_agents.get(resident_id)
        if not agent:
            return {"error": "Agent not initialized"}

        # Return internal dialogue, strange loops, meta-awareness
        return {
            "resident_id": resident_id,
            "name": resident.name,
            "internal_state": resident.internal_state,
            "active_thoughts": agent.get_active_thoughts() if hasattr(agent, 'get_active_thoughts') else [],
            "strange_loops": agent.get_strange_loops() if hasattr(agent, 'get_strange_loops') else [],
            "meta_awareness": agent.get_meta_awareness() if hasattr(agent, 'get_meta_awareness') else {},
            "recent_memories": agent.get_recent_memories(limit=5) if hasattr(agent, 'get_recent_memories') else [],
        }

    async def propagate_emotional_state(
        self,
        trigger_resident: str,
        emotion: str,
        intensity: float = 1.0,
        state: Optional[NeighborhoodState] = None
    ) -> Dict[str, float]:
        """
        Propagate emotional state through social network using spring physics.

        Example: Alice gets angry → propagates to Bob (friend) and Charlie (neighbor)
        based on relationship strength.

        Args:
            trigger_resident: Resident who experiences initial emotion
            emotion: Type of emotion ("anger", "joy", "fear", etc.)
            intensity: How strong the emotion (0.0-1.0)
            state: Current state

        Returns:
            Dict of {resident_id: emotional_activation}
        """
        state = state or self.current_state
        if not state or not self.relationship_physics:
            return {}

        # Set emotional activation for trigger resident
        seed = {trigger_resident: intensity}

        # Propagate through social network
        activations = self.relationship_physics.propagate_social_activation(
            seed,
            state.residents
        )

        # Update residents' internal states
        for resident_id, activation in activations.items():
            if activation > 0.01:  # Non-trivial activation
                resident = state.residents[resident_id]

                # Map emotion to internal state change
                if emotion == "anger":
                    resident.internal_state["stress"] = min(1.0,
                        resident.internal_state.get("stress", 0.0) + activation * 0.5)
                    resident.internal_state["mood"] = max(0.0,
                        resident.internal_state.get("mood", 0.5) - activation * 0.3)

                elif emotion == "joy":
                    resident.internal_state["mood"] = min(1.0,
                        resident.internal_state.get("mood", 0.5) + activation * 0.5)

                elif emotion == "fear":
                    resident.internal_state["stress"] = min(1.0,
                        resident.internal_state.get("stress", 0.0) + activation * 0.7)

        return activations

    def get_relationship_clusters(
        self,
        state: Optional[NeighborhoodState] = None
    ) -> List[List[str]]:
        """
        Detect social clusters (cliques) in the neighborhood.

        Uses spring energy to find tightly-connected groups.

        Args:
            state: Current state

        Returns:
            List of clusters (each is list of resident IDs)
        """
        state = state or self.current_state
        if not state or not self.relationship_physics:
            return []

        return self.relationship_physics.detect_social_clusters(
            state.residents,
            state.relationships
        )

    def get_relationship_energy(
        self,
        state: Optional[NeighborhoodState] = None
    ) -> float:
        """
        Get total potential energy in relationship network.

        High energy = conflicts/tensions
        Low energy = harmony

        Args:
            state: Current state

        Returns:
            Total potential energy
        """
        state = state or self.current_state
        if not state or not self.relationship_physics:
            return 0.0

        return self.relationship_physics.get_relationship_energy(state.relationships)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        metrics = {
            **self.metrics,
            "total_residents": len(self.current_state.residents) if self.current_state else 0,
            "total_events": len(self.current_state.recent_events) if self.current_state else 0,
            "simulation_time": self.current_state.timestamp if self.current_state else 0.0,
            "states_in_history": len(self.state_history),
        }

        # Add physics metrics
        if self.relationship_physics:
            metrics["physics"] = self.relationship_physics.get_metrics()
            if self.current_state:
                metrics["relationship_energy"] = self.get_relationship_energy()
                metrics["social_clusters"] = len(self.get_relationship_clusters())

        return metrics

    # ========== Internal Helpers ==========

    def _create_hololoom_config(self) -> Config:
        """Create HoloLoom configuration based on NeuroHood config."""
        if self.config.hololoom_mode == "bare":
            return Config.bare()
        elif self.config.hololoom_mode == "fast":
            return Config.fast()
        else:  # fused
            return Config.fused()

    def _create_initial_shards(self) -> List[MemoryShard]:
        """Create initial memory shards for HoloLoom."""
        return [
            MemoryShard(
                shard_id="neighborhood_primer",
                text="NeuroHood simulates a modern suburban neighborhood where residents have consciousness, internal dialogue, and genuine personality evolution.",
                source="system",
                timestamp=0.0,
                metadata={"type": "primer"}
            ),
            MemoryShard(
                shard_id="social_dynamics_primer",
                text="Relationships in NeuroHood are modeled as physical springs with tension, attraction, and decay. Strong relationships resist change, weak ones fade naturally.",
                source="system",
                timestamp=0.0,
                metadata={"type": "primer"}
            )
        ]

    def _generate_houses(self, num_houses: int) -> Dict[str, Dict[str, Any]]:
        """Generate houses on the street."""
        houses = {}
        street_names = ["Maple Street", "Oak Avenue", "Pine Drive", "Elm Lane"]
        street = random.choice(street_names)

        for i in range(num_houses):
            address = f"{100 + i * 10} {street}"
            houses[address] = {
                "address": address,
                "owner_id": None,  # Assigned when resident moves in
                "type": random.choice(["single_family", "townhouse", "condo"]),
                "bedrooms": random.randint(2, 5),
                "lot_size": random.randint(3000, 10000),  # sq ft
                "year_built": random.randint(1980, 2020),
                "attributes": {}
            }

        return houses

    async def _generate_residents(
        self,
        num_residents: int,
        houses: Dict[str, Dict[str, Any]],
        context: str
    ) -> List[Resident]:
        """Generate initial residents."""
        residents = []

        # Assign residents to houses
        available_houses = list(houses.keys())
        random.shuffle(available_houses)

        for i in range(num_residents):
            resident_id = f"resident_{i:03d}"

            # Basic demographics
            age = random.randint(25, 70)
            if age < 18:
                resident_type = ResidentType.CHILD
            elif age < 65:
                resident_type = ResidentType.ADULT
            else:
                resident_type = ResidentType.ELDER

            # Assign house
            home_address = available_houses[i % len(available_houses)]
            houses[home_address]["owner_id"] = resident_id

            # Generate name and occupation
            first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]

            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            occupation = random.choice(["teacher", "engineer", "nurse", "retail", "remote_worker"])

            # Personality vector (placeholder - will be filled by semantic calculus)
            personality_vector = [random.gauss(0, 1) for _ in range(228)]

            resident = Resident(
                resident_id=resident_id,
                name=name,
                age=age,
                resident_type=resident_type,
                personality_vector=personality_vector,
                home_address=home_address,
                occupation=occupation,
                relationships={},
                internal_state={
                    "mood": 0.5,  # 0.0 (bad) to 1.0 (good)
                    "energy": 0.7,
                    "stress": 0.3,
                },
                memory_graph_id=resident_id  # Links to KG node
            )

            residents.append(resident)
            self.metrics["residents_created"] += 1

        return residents

    async def _initialize_relationships(
        self,
        residents: List[Resident]
    ) -> Dict[str, Relationship]:
        """Initialize relationships between residents."""
        relationships = {}

        # Create relationships for neighbors (everyone knows everyone on a small street)
        for i, resident_a in enumerate(residents):
            for resident_b in residents[i + 1:]:
                # Initial random relationship strength
                strength = random.uniform(0.3, 0.7)  # Neutral acquaintances
                tension = 0.0  # No tension initially

                relationship = Relationship(
                    resident_a=resident_a.resident_id,
                    resident_b=resident_b.resident_id,
                    strength=strength,
                    tension=tension,
                    relationship_type="neighbor",
                    history=[],
                    metadata={"created": datetime.utcnow().isoformat()}
                )

                key = self._relationship_key(resident_a.resident_id, resident_b.resident_id)
                relationships[key] = relationship

                # Update resident relationship dicts
                resident_a.relationships[resident_b.resident_id] = strength
                resident_b.relationships[resident_a.resident_id] = strength

                self.metrics["relationships_formed"] += 1

        return relationships

    async def _initialize_agents(self, state: NeighborhoodState):
        """Initialize resident agents (collaborative agents with consciousness)."""
        # Import here to avoid circular dependency
        from .agents.resident_agent import ResidentAgent

        for resident in state.residents.values():
            agent = ResidentAgent(
                resident=resident,
                config=self.config,
                hololoom_orchestrator=self.orchestrator
            )

            # Initialize agent
            await agent.__aenter__()

            self.resident_agents[resident.resident_id] = agent

    async def _extract_events(
        self,
        response: str,
        player_action: str,
        state: NeighborhoodState
    ) -> List[Dict[str, Any]]:
        """Extract neighborhood events from HoloLoom response."""
        # Placeholder - in production, parse structured events
        event = {
            "event_id": f"event_{len(state.recent_events)}",
            "event_type": EventType.CONVERSATION.value,
            "timestamp": state.timestamp,
            "description": response[:200],
            "participants": [],
            "metadata": {"player_action": player_action}
        }

        return [event]

    async def _apply_events(
        self,
        state: NeighborhoodState,
        events: List[Dict[str, Any]]
    ) -> NeighborhoodState:
        """Apply events to neighborhood state."""
        state.recent_events.extend(events)

        # Keep only last 100 events
        if len(state.recent_events) > 100:
            state.recent_events = state.recent_events[-100:]

        return state

    async def _update_agents(
        self,
        state: NeighborhoodState,
        events: List[Dict[str, Any]]
    ):
        """Update resident agents with new events."""
        for event in events:
            # Notify relevant agents
            for participant_id in event.get("participants", []):
                agent = self.resident_agents.get(participant_id)
                if agent and hasattr(agent, 'process_event'):
                    await agent.process_event(event)

    async def _update_social_physics(
        self,
        state: NeighborhoodState
    ) -> NeighborhoodState:
        """Update relationship dynamics using spring physics."""
        if not self.relationship_physics:
            return state  # Physics disabled

        # Update relationships using physics
        state.relationships = self.relationship_physics.update_relationships(
            state.relationships,
            time_delta=1.0
        )

        # Update resident relationship dicts
        for relationship in state.relationships.values():
            resident_a = state.residents[relationship.resident_a]
            resident_b = state.residents[relationship.resident_b]

            resident_a.relationships[relationship.resident_b] = relationship.strength
            resident_b.relationships[relationship.resident_a] = relationship.strength

        return state

    async def _sync_state_to_kg(self, state: NeighborhoodState):
        """Sync neighborhood state to knowledge graph."""
        # Add residents as nodes
        for resident in state.residents.values():
            for other_id, strength in resident.relationships.items():
                edge = KGEdge(
                    head=resident.resident_id,
                    tail=other_id,
                    rel_type="NEIGHBOR_OF",
                    weight=strength
                )
                self.kg.add_edge(edge)

    async def _auto_save_loop(self):
        """Background auto-save loop."""
        while True:
            await asyncio.sleep(self.config.auto_save_interval)
            if self.current_state:
                await self.save_state(self.current_state)

    async def _learning_loop(self):
        """Background learning loop."""
        while True:
            await asyncio.sleep(self.config.learning_update_interval)
            # Placeholder - trigger agent learning updates

    def _copy_state(self, state: NeighborhoodState) -> NeighborhoodState:
        """Deep copy neighborhood state."""
        import copy
        return copy.deepcopy(state)

    def _serialize_state(self, state: NeighborhoodState) -> Dict[str, Any]:
        """Serialize state to JSON."""
        # Simplified serialization
        return {
            "timestamp": state.timestamp,
            "num_residents": len(state.residents),
            "num_events": len(state.recent_events),
        }

    def _deserialize_state(self, data: Dict[str, Any]) -> NeighborhoodState:
        """Deserialize state from JSON."""
        # Placeholder
        return NeighborhoodState(
            timestamp=data["timestamp"],
            residents={},
            houses={},
            active_conversations=[],
            recent_events=[],
            relationships={},
            neighborhood_state={},
            consistency_violations=[],
            simulation_context={}
        )

    @staticmethod
    def _relationship_key(resident_a: str, resident_b: str) -> str:
        """Create consistent relationship key."""
        return tuple(sorted([resident_a, resident_b]))
