"""
DreamWeaver Core: Main orchestrator for world building

The DreamWeaver orchestrates the complete world building pipeline:
1. Accept user input (query, action, choice)
2. Retrieve relevant world context
3. Check consistency constraints
4. Generate new content (entities, events, descriptions)
5. Update world state
6. Learn from outcomes

Integration with HoloLoom:
- Uses WeavingOrchestrator for reasoning
- Stores world in Yarn Graph (KG)
- Uses Warp Space for narrative manifolds
- Thompson Sampling for narrative exploration
- Reflection Buffer for learning from player feedback
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json
from datetime import datetime

from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.memory.graph import KG, KGEdge

from . import (
    WorldEntity,
    NarrativeEvent,
    WorldState,
    ConsistencyRule,
    NarrativeThread,
    EntityType,
    EventType,
    ConsistencyLevel,
    GenerationMode,
    WorldBuilderProtocol,
    ConsistencyCheckerProtocol,
    GeneratorProtocol,
)


@dataclass
class DreamWeaverConfig:
    """Configuration for DreamWeaver."""

    # World persistence
    world_id: str = "default_world"
    save_path: Optional[str] = None
    auto_save: bool = True
    auto_save_interval: int = 300  # 5 minutes

    # Generation settings
    generation_mode: GenerationMode = GenerationMode.HYBRID
    consistency_level: ConsistencyLevel = ConsistencyLevel.BALANCED
    max_entities: int = 10000
    max_events: int = 100000

    # Narrative settings
    max_active_threads: int = 10
    thread_tension_decay: float = 0.1  # Per time unit
    event_causality_window: int = 20  # Look back N events for causal links

    # Generation parameters
    creativity_temperature: float = 0.8
    use_thompson_sampling: bool = True
    exploration_epsilon: float = 0.15

    # LLM settings (optional)
    llm_provider: Optional[str] = None  # "openai", "anthropic", "local"
    llm_model: Optional[str] = None
    llm_temperature: float = 0.8

    # Safety
    enable_content_filtering: bool = True
    track_authorship: bool = True  # Track human vs AI contributions
    enable_version_control: bool = True


class DreamWeaver:
    """
    Main orchestrator for open-source world building.

    The DreamWeaver transforms narrative threads into coherent, persistent worlds
    by integrating HoloLoom's reasoning engine with world-specific constraints.

    Architecture:
        User Input → Context Retrieval → Consistency Check → Generation →
        World Update → Reflection Learning

    Usage:
        ```python
        from hololoom.dreamweaving import DreamWeaver, DreamWeaverConfig
        from hololoom.config import Config

        config = DreamWeaverConfig(world_id="my_fantasy_world")
        holo_config = Config.fused()

        async with DreamWeaver(config, holo_config) as weaver:
            # Generate initial world
            world_state = await weaver.bootstrap_world(
                seed_prompt="A medieval fantasy world with magic and dragons"
            )

            # Evolve world with user input
            world_state = await weaver.step(
                user_input="I explore the ancient forest",
                world_state=world_state
            )

            # Query world state
            result = await weaver.query_world(
                "What factions exist in this world?",
                world_state
            )
        ```
    """

    def __init__(
        self,
        config: DreamWeaverConfig,
        hololoom_config: Config,
        knowledge_graph: Optional[KG] = None,
    ):
        """
        Initialize DreamWeaver.

        Args:
            config: DreamWeaver configuration
            hololoom_config: HoloLoom configuration (BARE/FAST/FUSED)
            knowledge_graph: Optional existing knowledge graph
        """
        self.config = config
        self.hololoom_config = hololoom_config

        # Core components (initialized in __aenter__)
        self.orchestrator: Optional[WeavingOrchestrator] = None
        self.kg: KG = knowledge_graph or KG()

        # World state
        self.current_world_state: Optional[WorldState] = None
        self.world_history: List[WorldState] = []

        # Consistency rules
        self.consistency_rules: List[ConsistencyRule] = []
        self._load_default_rules()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._auto_save_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = {
            "entities_generated": 0,
            "events_generated": 0,
            "consistency_checks": 0,
            "consistency_violations": 0,
            "user_interactions": 0,
            "llm_calls": 0,
            "cache_hits": 0,
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

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cancel background tasks
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass

        for task in self._background_tasks:
            task.cancel()

        # Final save
        if self.config.auto_save and self.current_world_state:
            await self.save_world(self.current_world_state)

        # Cleanup orchestrator
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

    # ========== World Initialization ==========

    async def bootstrap_world(
        self,
        seed_prompt: str,
        initial_entities: Optional[List[WorldEntity]] = None,
        initial_rules: Optional[List[ConsistencyRule]] = None
    ) -> WorldState:
        """
        Bootstrap a new world from a seed prompt.

        Args:
            seed_prompt: Natural language description of world
            initial_entities: Optional starting entities
            initial_rules: Optional consistency rules

        Returns:
            Initial world state
        """
        # Generate core world elements via HoloLoom
        query = Query(text=f"Generate world: {seed_prompt}")
        spacetime = await self.orchestrator.weave(query)

        # Extract entities from response
        entities = initial_entities or []
        if not entities:
            # Parse generated content for entities
            # (In production, use LLM to extract structured data)
            entities = await self._extract_entities_from_text(
                spacetime.response,
                seed_prompt
            )

        # Create initial world state
        world_state = WorldState(
            timestamp=0.0,
            entities={e.entity_id: e for e in entities},
            active_threads=[],
            recent_events=[],
            global_state={
                "seed_prompt": seed_prompt,
                "creation_time": datetime.utcnow().isoformat(),
            },
            consistency_violations=[],
            generation_context={"last_query": seed_prompt}
        )

        # Add custom rules
        if initial_rules:
            self.consistency_rules.extend(initial_rules)

        # Store in KG
        await self._sync_world_to_kg(world_state)

        self.current_world_state = world_state
        self.world_history.append(world_state)

        return world_state

    async def load_world(self, world_id: str, load_path: Optional[str] = None) -> WorldState:
        """
        Load existing world from disk.

        Args:
            world_id: World identifier
            load_path: Path to world file (defaults to config.save_path)

        Returns:
            Loaded world state
        """
        path = load_path or self.config.save_path
        if not path:
            raise ValueError("No save path configured")

        # Load from disk (JSON format)
        import os
        filepath = os.path.join(path, f"{world_id}.json")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        world_state = self._deserialize_world_state(data)

        # Sync to KG
        await self._sync_world_to_kg(world_state)

        self.current_world_state = world_state
        self.world_history.append(world_state)

        return world_state

    async def save_world(self, world_state: WorldState, save_path: Optional[str] = None):
        """
        Save world to disk.

        Args:
            world_state: World state to save
            save_path: Path to save file (defaults to config.save_path)
        """
        path = save_path or self.config.save_path
        if not path:
            return

        import os
        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, f"{self.config.world_id}.json")
        data = self._serialize_world_state(world_state)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ========== World Evolution ==========

    async def step(
        self,
        user_input: str,
        world_state: Optional[WorldState] = None,
        mode: Optional[GenerationMode] = None
    ) -> WorldState:
        """
        Evolve world state forward with user input.

        Args:
            user_input: User action, query, or choice
            world_state: Current world state (defaults to self.current_world_state)
            mode: Generation mode (defaults to config.generation_mode)

        Returns:
            Updated world state
        """
        world_state = world_state or self.current_world_state
        if not world_state:
            raise ValueError("No world state available. Call bootstrap_world() first.")

        mode = mode or self.config.generation_mode
        self.metrics["user_interactions"] += 1

        # 1. Retrieve relevant context
        context = await self._retrieve_world_context(user_input, world_state)

        # 2. Generate response via HoloLoom
        query = Query(
            text=user_input,
            metadata={
                "world_id": self.config.world_id,
                "context": context,
                "mode": mode.value,
            }
        )

        spacetime = await self.orchestrator.weave(query)

        # 3. Extract narrative events
        events = await self._extract_events_from_response(
            spacetime.response,
            user_input,
            world_state
        )

        # 4. Update world state
        new_world_state = await self._apply_events(world_state, events)

        # 5. Check consistency
        violations = await self._check_consistency(new_world_state)
        new_world_state.consistency_violations = violations

        # 6. Update KG
        await self._sync_world_to_kg(new_world_state)

        # 7. Store history
        self.current_world_state = new_world_state
        self.world_history.append(new_world_state)

        return new_world_state

    async def query_world(
        self,
        query: str,
        world_state: Optional[WorldState] = None
    ) -> str:
        """
        Query world state without modifying it.

        Args:
            query: Natural language query
            world_state: World state to query (defaults to current)

        Returns:
            Response text
        """
        world_state = world_state or self.current_world_state
        if not world_state:
            raise ValueError("No world state available")

        # Retrieve context
        context = await self._retrieve_world_context(query, world_state)

        # Query via HoloLoom
        holo_query = Query(
            text=query,
            metadata={
                "world_id": self.config.world_id,
                "context": context,
                "read_only": True,
            }
        )

        spacetime = await self.orchestrator.weave(holo_query)

        return spacetime.response

    # ========== Entity & Event Generation ==========

    async def generate_entity(
        self,
        entity_type: EntityType,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> WorldEntity:
        """
        Generate a new world entity.

        Args:
            entity_type: Type of entity to generate
            context: Context for generation
            constraints: Optional constraints

        Returns:
            Generated entity
        """
        # Build prompt
        prompt = self._build_entity_generation_prompt(entity_type, context, constraints)

        # Generate via HoloLoom
        query = Query(text=prompt)
        spacetime = await self.orchestrator.weave(query)

        # Parse entity from response
        entity = await self._parse_entity_from_text(
            spacetime.response,
            entity_type,
            context
        )

        self.metrics["entities_generated"] += 1

        return entity

    async def generate_event(
        self,
        event_type: EventType,
        participants: List[str],
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> NarrativeEvent:
        """
        Generate a narrative event.

        Args:
            event_type: Type of event
            participants: Entity IDs involved
            context: Context for generation
            constraints: Optional constraints

        Returns:
            Generated event
        """
        # Build prompt
        prompt = self._build_event_generation_prompt(
            event_type,
            participants,
            context,
            constraints
        )

        # Generate via HoloLoom
        query = Query(text=prompt)
        spacetime = await self.orchestrator.weave(query)

        # Parse event
        event = await self._parse_event_from_text(
            spacetime.response,
            event_type,
            participants,
            context
        )

        self.metrics["events_generated"] += 1

        return event

    # ========== Consistency Management ==========

    async def check_consistency(
        self,
        world_state: WorldState,
        rules: Optional[List[ConsistencyRule]] = None
    ) -> List[Dict[str, Any]]:
        """
        Check world state against consistency rules.

        Args:
            world_state: World state to check
            rules: Rules to check (defaults to self.consistency_rules)

        Returns:
            List of violations
        """
        return await self._check_consistency(world_state, rules)

    async def resolve_contradiction(
        self,
        world_state: WorldState,
        violation: Dict[str, Any]
    ) -> Optional[WorldState]:
        """
        Attempt to resolve a consistency violation.

        Args:
            world_state: World state with violation
            violation: Violation to resolve

        Returns:
            Fixed world state, or None if cannot resolve
        """
        # Build prompt for resolution
        prompt = f"""
        World consistency violation detected:
        Type: {violation['rule_type']}
        Description: {violation['description']}
        Severity: {violation['severity']}

        Suggest a resolution that maintains narrative coherence.
        """

        query = Query(text=prompt, metadata={"violation": violation})
        spacetime = await self.orchestrator.weave(query)

        # Apply suggested fix
        # (In production, parse structured fix from LLM response)

        return world_state  # Placeholder

    # ========== Narrative Thread Management ==========

    async def create_thread(
        self,
        thread_type: str,
        title: str,
        participants: List[str],
        world_state: Optional[WorldState] = None
    ) -> NarrativeThread:
        """Create a new narrative thread."""
        world_state = world_state or self.current_world_state

        thread = NarrativeThread(
            thread_id=f"thread_{len(world_state.active_threads)}",
            thread_type=thread_type,
            title=title,
            participants=participants,
            events=[],
            status="active",
            tension=0.5,
            foreshadowing=[],
            metadata={"created": datetime.utcnow().isoformat()}
        )

        world_state.active_threads.append(thread)

        return thread

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "world_states": len(self.world_history),
            "total_entities": len(self.current_world_state.entities) if self.current_world_state else 0,
            "total_events": len(self.current_world_state.recent_events) if self.current_world_state else 0,
            "active_threads": len(self.current_world_state.active_threads) if self.current_world_state else 0,
        }

    # ========== Internal Helpers ==========

    def _load_default_rules(self):
        """Load default consistency rules."""
        # Physical consistency
        self.consistency_rules.append(ConsistencyRule(
            rule_id="physical_causality",
            rule_type="physical",
            description="Events must respect causality (no time travel unless explicitly allowed)",
            check_function="check_temporal_ordering",
            severity=0.9,
            auto_fix=False
        ))

        # Logical consistency
        self.consistency_rules.append(ConsistencyRule(
            rule_id="entity_identity",
            rule_type="logical",
            description="Entities cannot be in two places at once",
            check_function="check_entity_location_consistency",
            severity=0.8,
            auto_fix=True
        ))

        # Narrative consistency
        self.consistency_rules.append(ConsistencyRule(
            rule_id="character_personality",
            rule_type="narrative",
            description="Character actions should align with established personality",
            check_function="check_character_consistency",
            severity=0.6,
            auto_fix=False
        ))

    def _create_initial_shards(self) -> List[MemoryShard]:
        """Create initial memory shards for HoloLoom."""
        # Seed with world building knowledge
        return [
            MemoryShard(
                id="worldbuilding_primer",
                text="World building involves creating consistent, engaging fictional worlds with rich history, geography, culture, and characters.",
                timestamp=0.0,
                metadata={"type": "primer", "source": "system"}
            )
        ]

    async def _retrieve_world_context(
        self,
        query: str,
        world_state: WorldState
    ) -> Dict[str, Any]:
        """Retrieve relevant world context for a query."""
        # Use KG to find relevant entities/events
        # (Simplified - in production, use full retrieval pipeline)

        return {
            "recent_events": world_state.recent_events[-5:],
            "active_threads": world_state.active_threads,
            "entity_count": len(world_state.entities),
        }

    async def _extract_entities_from_text(
        self,
        text: str,
        context: str
    ) -> List[WorldEntity]:
        """Extract entities from generated text."""
        # Placeholder - in production, use NER + LLM
        return []

    async def _extract_events_from_response(
        self,
        response: str,
        user_input: str,
        world_state: WorldState
    ) -> List[NarrativeEvent]:
        """Extract narrative events from response."""
        # Placeholder
        return []

    async def _apply_events(
        self,
        world_state: WorldState,
        events: List[NarrativeEvent]
    ) -> WorldState:
        """Apply events to world state."""
        # Clone state
        import copy
        new_state = copy.deepcopy(world_state)

        # Add events
        new_state.recent_events.extend(events)
        new_state.timestamp += 1.0

        return new_state

    async def _check_consistency(
        self,
        world_state: WorldState,
        rules: Optional[List[ConsistencyRule]] = None
    ) -> List[Dict[str, Any]]:
        """Check consistency rules."""
        rules = rules or self.consistency_rules
        violations = []

        self.metrics["consistency_checks"] += 1

        # Check each rule
        for rule in rules:
            # Placeholder - in production, execute check functions
            pass

        self.metrics["consistency_violations"] += len(violations)

        return violations

    async def _sync_world_to_kg(self, world_state: WorldState):
        """Sync world state to knowledge graph."""
        # Add entities as nodes
        for entity in world_state.entities.values():
            for rel in entity.relationships:
                edge = KGEdge(
                    head=entity.entity_id,
                    tail=rel["target_id"],
                    rel_type=rel["relation_type"],
                    weight=rel.get("strength", 1.0)
                )
                self.kg.add_edge(edge)

    async def _auto_save_loop(self):
        """Background auto-save loop."""
        while True:
            await asyncio.sleep(self.config.auto_save_interval)
            if self.current_world_state:
                await self.save_world(self.current_world_state)

    def _serialize_world_state(self, world_state: WorldState) -> Dict[str, Any]:
        """Serialize world state to JSON."""
        # Placeholder - use proper serialization
        return {
            "timestamp": world_state.timestamp,
            "entity_count": len(world_state.entities),
        }

    def _deserialize_world_state(self, data: Dict[str, Any]) -> WorldState:
        """Deserialize world state from JSON."""
        # Placeholder
        return WorldState(
            timestamp=data["timestamp"],
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

    def _build_entity_generation_prompt(
        self,
        entity_type: EntityType,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for entity generation."""
        return f"Generate a {entity_type.value} for the world."

    def _build_event_generation_prompt(
        self,
        event_type: EventType,
        participants: List[str],
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for event generation."""
        return f"Generate a {event_type.value} event involving {len(participants)} participants."

    async def _parse_entity_from_text(
        self,
        text: str,
        entity_type: EntityType,
        context: Dict[str, Any]
    ) -> WorldEntity:
        """Parse entity from generated text."""
        # Placeholder
        return WorldEntity(
            entity_id=f"entity_{self.metrics['entities_generated']}",
            entity_type=entity_type,
            name="Generated Entity",
            description=text[:200],
            attributes={},
            relationships=[],
            history=[],
            metadata={"generated": datetime.utcnow().isoformat()}
        )

    async def _parse_event_from_text(
        self,
        text: str,
        event_type: EventType,
        participants: List[str],
        context: Dict[str, Any]
    ) -> NarrativeEvent:
        """Parse event from generated text."""
        # Placeholder
        return NarrativeEvent(
            event_id=f"event_{self.metrics['events_generated']}",
            event_type=event_type,
            timestamp=self.current_world_state.timestamp if self.current_world_state else 0.0,
            participants=participants,
            location_id=None,
            description=text[:200],
            consequences={},
            causal_links={"causes": [], "effects": []},
            metadata={"generated": datetime.utcnow().isoformat()}
        )
