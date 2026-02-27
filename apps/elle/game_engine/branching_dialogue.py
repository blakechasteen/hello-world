"""
Branching Dialogue System for Elle Game Engine
===============================================

Implements dynamic branching dialogue trees with:
- DialogueNode: Individual dialogue entries with speaker, text, choices
- DialogueChoice: Player choices with conditions and effects
- DialogueTree: Complete conversation structure with traversal
- Condition System: Game state conditions for unlocking choices
- Effect System: World state changes from dialogue choices
- LLM Integration: Dynamic choice generation using HoloLoom

Features:
- Static + Dynamic dialogue (predefined trees + LLM-generated branches)
- Condition-gated choices (reputation, inventory, flags)
- Choice effects (mood changes, flag setting, relationship updates)
- History tracking in HoloLoom knowledge graph
- Memory-based dialogue variation (reference past conversations)

Created: 2025-12-03
Author: HoloLoom Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
import uuid
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Condition System
# ============================================================================

class ConditionType(Enum):
    """Types of conditions that can gate dialogue choices."""
    FLAG = auto()              # World flag is set/unset
    REPUTATION = auto()        # Player reputation threshold
    INVENTORY = auto()         # Player has item
    RELATIONSHIP = auto()      # NPC relationship level
    STAT = auto()              # Player stat check
    QUEST_STAGE = auto()       # Quest progress
    PREVIOUS_CHOICE = auto()   # Player made a previous choice
    CUSTOM = auto()            # Custom callback


@dataclass
class DialogueCondition:
    """
    A condition that must be met for a choice to be available.

    Examples:
        # Player must have hero reputation
        DialogueCondition(ConditionType.REPUTATION, key="reputation", value="hero")

        # Player must have the ancient_key item
        DialogueCondition(ConditionType.INVENTORY, key="ancient_key")

        # NPC trust level >= 5
        DialogueCondition(ConditionType.RELATIONSHIP, key="trust", value=5, op=">=")

        # Player previously chose to help the merchant
        DialogueCondition(ConditionType.PREVIOUS_CHOICE, key="helped_merchant")
    """
    type: ConditionType
    key: str
    value: Any = True  # Expected value
    op: str = "=="     # Comparison operator: ==, !=, >, <, >=, <=, in, not_in

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate this condition against game context.

        Args:
            context: Game state context with flags, inventory, stats, etc.

        Returns:
            True if condition is met
        """
        # Get actual value from context
        actual = self._get_value(context)

        # Handle None case
        if actual is None:
            return self.op == "!=" and self.value is not None

        # Evaluate based on operator
        if self.op == "==":
            return actual == self.value
        elif self.op == "!=":
            return actual != self.value
        elif self.op == ">":
            return actual > self.value
        elif self.op == "<":
            return actual < self.value
        elif self.op == ">=":
            return actual >= self.value
        elif self.op == "<=":
            return actual <= self.value
        elif self.op == "in":
            return actual in self.value
        elif self.op == "not_in":
            return actual not in self.value
        else:
            logger.warning(f"Unknown operator: {self.op}")
            return False

    def _get_value(self, context: Dict[str, Any]) -> Any:
        """Extract the relevant value from context based on condition type."""
        if self.type == ConditionType.FLAG:
            return context.get('flags', {}).get(self.key)
        elif self.type == ConditionType.REPUTATION:
            return context.get('player', {}).get('reputation')
        elif self.type == ConditionType.INVENTORY:
            inventory = context.get('player', {}).get('inventory', [])
            return self.key in inventory
        elif self.type == ConditionType.RELATIONSHIP:
            npcs = context.get('npc_relationships', {})
            npc_rel = npcs.get(context.get('current_npc_id', ''), {})
            return npc_rel.get(self.key, 0)
        elif self.type == ConditionType.STAT:
            return context.get('player', {}).get('stats', {}).get(self.key)
        elif self.type == ConditionType.QUEST_STAGE:
            quests = context.get('quests', {})
            return quests.get(self.key, {}).get('stage')
        elif self.type == ConditionType.PREVIOUS_CHOICE:
            return self.key in context.get('choice_history', set())
        else:
            return None


# ============================================================================
# Effect System
# ============================================================================

class EffectType(Enum):
    """Types of effects a dialogue choice can trigger."""
    SET_FLAG = auto()          # Set a world flag
    CLEAR_FLAG = auto()        # Clear a world flag
    ADD_ITEM = auto()          # Give player an item
    REMOVE_ITEM = auto()       # Remove item from player
    CHANGE_RELATIONSHIP = auto()  # Change NPC relationship
    CHANGE_MOOD = auto()       # Change NPC mood
    UPDATE_QUEST = auto()      # Progress quest
    TRIGGER_EVENT = auto()     # Trigger game event
    CUSTOM = auto()            # Custom callback


@dataclass
class DialogueEffect:
    """
    An effect that occurs when a dialogue choice is selected.

    Examples:
        # Set a flag that merchant was helped
        DialogueEffect(EffectType.SET_FLAG, key="helped_merchant")

        # Increase NPC trust by 2
        DialogueEffect(EffectType.CHANGE_RELATIONSHIP, key="trust", value=2)

        # Give player a reward item
        DialogueEffect(EffectType.ADD_ITEM, key="gold_coins", value=50)

        # Change NPC mood to grateful
        DialogueEffect(EffectType.CHANGE_MOOD, value="grateful")
    """
    type: EffectType
    key: str = ""
    value: Any = None
    target_npc: Optional[str] = None  # For NPC-specific effects

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply this effect to the game context.

        Args:
            context: Mutable game state context

        Returns:
            Updated context
        """
        if self.type == EffectType.SET_FLAG:
            context.setdefault('flags', {})[self.key] = self.value if self.value is not None else True

        elif self.type == EffectType.CLEAR_FLAG:
            context.setdefault('flags', {}).pop(self.key, None)

        elif self.type == EffectType.ADD_ITEM:
            inventory = context.setdefault('player', {}).setdefault('inventory', [])
            if isinstance(self.value, int):
                # Add quantity
                for _ in range(self.value):
                    inventory.append(self.key)
            else:
                inventory.append(self.key)

        elif self.type == EffectType.REMOVE_ITEM:
            inventory = context.get('player', {}).get('inventory', [])
            if self.key in inventory:
                inventory.remove(self.key)

        elif self.type == EffectType.CHANGE_RELATIONSHIP:
            npc_id = self.target_npc or context.get('current_npc_id', 'unknown')
            relationships = context.setdefault('npc_relationships', {})
            npc_rel = relationships.setdefault(npc_id, {})
            current = npc_rel.get(self.key, 0)
            npc_rel[self.key] = current + (self.value or 1)

        elif self.type == EffectType.CHANGE_MOOD:
            npc_id = self.target_npc or context.get('current_npc_id', 'unknown')
            moods = context.setdefault('npc_moods', {})
            moods[npc_id] = self.value

        elif self.type == EffectType.UPDATE_QUEST:
            quests = context.setdefault('quests', {})
            quest = quests.setdefault(self.key, {})
            quest['stage'] = self.value

        elif self.type == EffectType.TRIGGER_EVENT:
            events = context.setdefault('triggered_events', [])
            events.append({'event': self.key, 'data': self.value})

        return context


# ============================================================================
# Dialogue Models
# ============================================================================

@dataclass
class DialogueChoice:
    """
    A player choice within a dialogue node.

    Choices can be:
    - Always available (no conditions)
    - Conditionally available (conditions must be met)
    - Trigger effects when selected
    - Lead to another dialogue node or end conversation
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""                                  # Display text
    next_node_id: Optional[str] = None              # Node to go to (None = end conversation)
    conditions: List[DialogueCondition] = field(default_factory=list)
    effects: List[DialogueEffect] = field(default_factory=list)

    # Display hints
    hint: Optional[str] = None                      # Tooltip/hint for this choice
    requires_item: Optional[str] = None             # Item required (for display)
    personality_tag: Optional[str] = None           # "kind", "aggressive", "cunning", etc.

    def is_available(self, context: Dict[str, Any]) -> bool:
        """Check if this choice is available given the context."""
        return all(cond.evaluate(context) for cond in self.conditions)

    def select(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select this choice, applying all effects.

        Returns updated context.
        """
        # Record choice in history
        context.setdefault('choice_history', set()).add(self.id)

        # Apply all effects
        for effect in self.effects:
            context = effect.apply(context)

        return context


@dataclass
class DialogueNode:
    """
    A single node in the dialogue tree.

    Contains:
    - Speaker information
    - Dialogue text
    - Available choices
    - Metadata for display/animation
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    speaker_id: str = ""                            # NPC or "player" or "narrator"
    speaker_name: str = ""                          # Display name
    text: str = ""                                  # Dialogue text
    choices: List[DialogueChoice] = field(default_factory=list)

    # Presentation
    tone: Optional[str] = None                      # "warm", "stern", "cryptic", etc.
    emotion: Optional[str] = None                   # For animation triggers
    camera_hint: Optional[str] = None               # "close_up", "wide", etc.

    # Metadata
    tags: Set[str] = field(default_factory=set)     # For searching/filtering
    is_entry_point: bool = False                    # Can conversation start here?
    is_exit_point: bool = False                     # Does this end conversation?

    def get_available_choices(self, context: Dict[str, Any]) -> List[DialogueChoice]:
        """Get choices that are currently available based on context."""
        return [c for c in self.choices if c.is_available(context)]


@dataclass
class DialogueTree:
    """
    A complete dialogue tree for a conversation.

    Contains all nodes, tracks current state, and handles navigation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""                                  # Display name
    description: str = ""                           # Editor description
    npc_id: str = ""                                # Primary NPC for this tree
    nodes: Dict[str, DialogueNode] = field(default_factory=dict)
    entry_node_id: Optional[str] = None             # Default starting node

    # Tree metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)

    def add_node(self, node: DialogueNode):
        """Add a node to the tree."""
        self.nodes[node.id] = node
        if node.is_entry_point and self.entry_node_id is None:
            self.entry_node_id = node.id
        self.updated_at = datetime.now()

    def get_node(self, node_id: str) -> Optional[DialogueNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_entry_nodes(self) -> List[DialogueNode]:
        """Get all possible entry points."""
        return [n for n in self.nodes.values() if n.is_entry_point]

    def validate(self) -> List[str]:
        """
        Validate tree integrity.

        Returns list of error messages (empty = valid).
        """
        errors = []

        # Check entry point
        if not self.entry_node_id:
            errors.append("No entry node defined")
        elif self.entry_node_id not in self.nodes:
            errors.append(f"Entry node '{self.entry_node_id}' not found")

        # Check all choice targets exist
        for node in self.nodes.values():
            for choice in node.choices:
                if choice.next_node_id and choice.next_node_id not in self.nodes:
                    errors.append(f"Choice in '{node.id}' points to missing node '{choice.next_node_id}'")

        # Check for orphan nodes (not reachable from entry)
        if self.entry_node_id:
            reachable = self._get_reachable_nodes(self.entry_node_id)
            for node_id in self.nodes:
                if node_id not in reachable and not self.nodes[node_id].is_entry_point:
                    errors.append(f"Node '{node_id}' is not reachable from any entry point")

        return errors

    def _get_reachable_nodes(self, start_id: str) -> Set[str]:
        """Get all nodes reachable from start_id."""
        reachable = set()
        queue = [start_id]

        while queue:
            node_id = queue.pop(0)
            if node_id in reachable:
                continue
            reachable.add(node_id)

            node = self.nodes.get(node_id)
            if node:
                for choice in node.choices:
                    if choice.next_node_id and choice.next_node_id not in reachable:
                        queue.append(choice.next_node_id)

        return reachable

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'npc_id': self.npc_id,
            'entry_node_id': self.entry_node_id,
            'nodes': {
                node_id: {
                    'id': node.id,
                    'speaker_id': node.speaker_id,
                    'speaker_name': node.speaker_name,
                    'text': node.text,
                    'tone': node.tone,
                    'emotion': node.emotion,
                    'is_entry_point': node.is_entry_point,
                    'is_exit_point': node.is_exit_point,
                    'tags': list(node.tags),
                    'choices': [
                        {
                            'id': c.id,
                            'text': c.text,
                            'next_node_id': c.next_node_id,
                            'hint': c.hint,
                            'personality_tag': c.personality_tag,
                            'conditions': [
                                {'type': cond.type.name, 'key': cond.key, 'value': cond.value, 'op': cond.op}
                                for cond in c.conditions
                            ],
                            'effects': [
                                {'type': eff.type.name, 'key': eff.key, 'value': eff.value, 'target_npc': eff.target_npc}
                                for eff in c.effects
                            ],
                        }
                        for c in node.choices
                    ]
                }
                for node_id, node in self.nodes.items()
            },
            'tags': list(self.tags),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTree':
        """Deserialize from dictionary."""
        tree = cls(
            id=data['id'],
            name=data.get('name', ''),
            description=data.get('description', ''),
            npc_id=data.get('npc_id', ''),
            entry_node_id=data.get('entry_node_id'),
            tags=set(data.get('tags', [])),
        )

        for node_id, node_data in data.get('nodes', {}).items():
            choices = []
            for choice_data in node_data.get('choices', []):
                conditions = [
                    DialogueCondition(
                        type=ConditionType[c['type']],
                        key=c['key'],
                        value=c.get('value', True),
                        op=c.get('op', '=='),
                    )
                    for c in choice_data.get('conditions', [])
                ]
                effects = [
                    DialogueEffect(
                        type=EffectType[e['type']],
                        key=e.get('key', ''),
                        value=e.get('value'),
                        target_npc=e.get('target_npc'),
                    )
                    for e in choice_data.get('effects', [])
                ]
                choices.append(DialogueChoice(
                    id=choice_data['id'],
                    text=choice_data['text'],
                    next_node_id=choice_data.get('next_node_id'),
                    hint=choice_data.get('hint'),
                    personality_tag=choice_data.get('personality_tag'),
                    conditions=conditions,
                    effects=effects,
                ))

            node = DialogueNode(
                id=node_data['id'],
                speaker_id=node_data.get('speaker_id', ''),
                speaker_name=node_data.get('speaker_name', ''),
                text=node_data.get('text', ''),
                tone=node_data.get('tone'),
                emotion=node_data.get('emotion'),
                is_entry_point=node_data.get('is_entry_point', False),
                is_exit_point=node_data.get('is_exit_point', False),
                tags=set(node_data.get('tags', [])),
                choices=choices,
            )
            tree.nodes[node_id] = node

        return tree


# ============================================================================
# Dialogue Engine
# ============================================================================

@dataclass
class DialogueState:
    """Current state of an active dialogue."""
    tree: DialogueTree
    current_node_id: str
    context: Dict[str, Any]
    history: List[str] = field(default_factory=list)  # Node IDs visited
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def current_node(self) -> Optional[DialogueNode]:
        """Get the current node."""
        return self.tree.get_node(self.current_node_id)


class BranchingDialogueEngine:
    """
    Main engine for running branching dialogues.

    Features:
    - Static tree execution
    - Dynamic choice generation via LLM
    - HoloLoom integration for history tracking
    - Memory-based dialogue variation

    Usage:
        engine = BranchingDialogueEngine()
        engine.register_tree(merchant_dialogue)

        # Start conversation
        state = engine.start_dialogue("merchant_001", context=game_context)

        # Get current options
        node = state.current_node
        choices = engine.get_available_choices(state)

        # Player selects choice
        state = engine.select_choice(state, choice_id)

        # Continue until conversation ends
        while not engine.is_finished(state):
            ...
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        hololoom_kg: Optional[Any] = None,
    ):
        """
        Initialize the dialogue engine.

        Args:
            llm_client: Optional LLM client for dynamic generation
            hololoom_kg: Optional HoloLoom knowledge graph for history
        """
        self.llm_client = llm_client
        self.kg = hololoom_kg
        self._trees: Dict[str, DialogueTree] = {}
        self._active_dialogues: Dict[str, DialogueState] = {}

    def register_tree(self, tree: DialogueTree):
        """Register a dialogue tree."""
        errors = tree.validate()
        if errors:
            logger.warning(f"Dialogue tree '{tree.id}' has validation errors: {errors}")
        self._trees[tree.id] = tree

    def get_tree(self, tree_id: str) -> Optional[DialogueTree]:
        """Get a registered tree."""
        return self._trees.get(tree_id)

    def start_dialogue(
        self,
        tree_id: str,
        context: Dict[str, Any],
        entry_node_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> DialogueState:
        """
        Start a new dialogue conversation.

        Args:
            tree_id: ID of the dialogue tree to use
            context: Game state context
            entry_node_id: Override entry point (default: tree's entry_node_id)
            session_id: Optional session ID for tracking

        Returns:
            Initial dialogue state

        Raises:
            ValueError: If tree not found or entry node invalid
        """
        tree = self._trees.get(tree_id)
        if not tree:
            raise ValueError(f"Dialogue tree '{tree_id}' not found")

        entry_id = entry_node_id or tree.entry_node_id
        if not entry_id or entry_id not in tree.nodes:
            raise ValueError(f"Invalid entry node '{entry_id}'")

        # Create state
        state = DialogueState(
            tree=tree,
            current_node_id=entry_id,
            context=dict(context),  # Copy to avoid mutation
            history=[entry_id],
        )

        # Track in active dialogues
        session_id = session_id or str(uuid.uuid4())
        self._active_dialogues[session_id] = state

        # Record in HoloLoom if available
        if self.kg:
            self._record_dialogue_start(session_id, tree_id, entry_id)

        logger.info(f"Started dialogue '{tree_id}' at node '{entry_id}'")

        return state

    def get_available_choices(self, state: DialogueState) -> List[DialogueChoice]:
        """Get currently available choices for the dialogue state."""
        node = state.current_node
        if not node:
            return []
        return node.get_available_choices(state.context)

    def select_choice(
        self,
        state: DialogueState,
        choice_id: str,
    ) -> DialogueState:
        """
        Select a choice and advance the dialogue.

        Args:
            state: Current dialogue state
            choice_id: ID of the choice to select

        Returns:
            Updated dialogue state

        Raises:
            ValueError: If choice not found or not available
        """
        node = state.current_node
        if not node:
            raise ValueError("No current node")

        # Find the choice
        choice = next((c for c in node.choices if c.id == choice_id), None)
        if not choice:
            raise ValueError(f"Choice '{choice_id}' not found in node '{node.id}'")

        # Check availability
        if not choice.is_available(state.context):
            raise ValueError(f"Choice '{choice_id}' is not available")

        # Apply effects
        state.context = choice.select(state.context)

        # Move to next node
        if choice.next_node_id:
            state.current_node_id = choice.next_node_id
            state.history.append(choice.next_node_id)

        # Record in HoloLoom
        if self.kg:
            self._record_choice(state.tree.id, node.id, choice_id)

        logger.debug(f"Selected choice '{choice_id}', moved to node '{state.current_node_id}'")

        return state

    def is_finished(self, state: DialogueState) -> bool:
        """Check if the dialogue has ended."""
        node = state.current_node
        if not node:
            return True
        if node.is_exit_point:
            return True
        # No available choices = end of conversation
        return len(self.get_available_choices(state)) == 0

    def get_dialogue_summary(self, state: DialogueState) -> Dict[str, Any]:
        """Get a summary of the dialogue progression."""
        return {
            'tree_id': state.tree.id,
            'nodes_visited': len(state.history),
            'history': state.history,
            'choices_made': list(state.context.get('choice_history', set())),
            'effects_triggered': len(state.context.get('triggered_events', [])),
            'duration_seconds': (datetime.now() - state.started_at).total_seconds(),
        }

    # -------------------------------------------------------------------------
    # Dynamic Generation (LLM Integration)
    # -------------------------------------------------------------------------

    async def generate_dynamic_choices(
        self,
        state: DialogueState,
        num_choices: int = 3,
        personality_constraints: Optional[List[str]] = None,
    ) -> List[DialogueChoice]:
        """
        Generate dynamic dialogue choices using LLM.

        This is used when a dialogue node has no predefined choices,
        allowing for more natural, context-aware conversations.

        Args:
            state: Current dialogue state
            num_choices: Number of choices to generate
            personality_constraints: Optional personality tags to respect

        Returns:
            List of generated DialogueChoice objects
        """
        if not self.llm_client:
            logger.warning("No LLM client configured for dynamic generation")
            return []

        node = state.current_node
        if not node:
            return []

        # Build prompt for LLM
        prompt = self._build_choice_generation_prompt(
            node=node,
            context=state.context,
            history=state.history,
            num_choices=num_choices,
            personality_constraints=personality_constraints,
        )

        try:
            # Call LLM (implementation depends on client type)
            response = await self._call_llm(prompt)
            choices = self._parse_generated_choices(response)
            return choices[:num_choices]
        except Exception as e:
            logger.error(f"Failed to generate dynamic choices: {e}")
            return []

    def _build_choice_generation_prompt(
        self,
        node: DialogueNode,
        context: Dict[str, Any],
        history: List[str],
        num_choices: int,
        personality_constraints: Optional[List[str]],
    ) -> str:
        """Build prompt for choice generation."""
        prompt = f"""Generate {num_choices} player dialogue choices for this conversation.

Current NPC: {node.speaker_name or node.speaker_id}
NPC's line: {node.text}
NPC's tone: {node.tone or 'neutral'}

Player context:
- Reputation: {context.get('player', {}).get('reputation', 'unknown')}
- Recent topics: {', '.join(history[-3:])}

Generate diverse choices that:
1. Feel natural as player responses
2. Vary in personality/approach
3. Could lead the conversation in different directions

"""
        if personality_constraints:
            prompt += f"Personality options: {', '.join(personality_constraints)}\n"

        prompt += """
Output format (JSON array):
[
  {"text": "Choice text here", "personality": "kind/aggressive/curious/neutral", "hint": "optional hint"}
]"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM client. Override for specific implementations."""
        if hasattr(self.llm_client, 'generate'):
            return await self.llm_client.generate(prompt)
        elif hasattr(self.llm_client, '__call__'):
            return await self.llm_client(prompt)
        else:
            raise ValueError("LLM client must have 'generate' method or be callable")

    def _parse_generated_choices(self, response: str) -> List[DialogueChoice]:
        """Parse LLM response into DialogueChoice objects."""
        try:
            # Try to extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return [
                    DialogueChoice(
                        text=item.get('text', ''),
                        hint=item.get('hint'),
                        personality_tag=item.get('personality'),
                    )
                    for item in data
                ]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return []

    # -------------------------------------------------------------------------
    # HoloLoom Integration
    # -------------------------------------------------------------------------

    def _record_dialogue_start(self, session_id: str, tree_id: str, entry_node_id: str):
        """Record dialogue start in HoloLoom knowledge graph."""
        if not self.kg:
            return
        try:
            from hololoom.memory.graph import KGEdge
            self.kg.G.add_node(
                f"dialogue:{session_id}",
                node_type="dialogue_session",
                tree_id=tree_id,
                started_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning(f"Failed to record dialogue start: {e}")

    def _record_choice(self, tree_id: str, node_id: str, choice_id: str):
        """Record choice selection in HoloLoom."""
        if not self.kg:
            return
        try:
            self.kg.G.add_node(
                f"choice:{choice_id}",
                node_type="dialogue_choice",
                tree_id=tree_id,
                node_id=node_id,
                selected_at=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.warning(f"Failed to record choice: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_simple_dialogue(
    npc_id: str,
    npc_name: str,
    lines: List[str],
    final_choices: Optional[List[str]] = None,
) -> DialogueTree:
    """
    Create a simple linear dialogue with optional final choices.

    Args:
        npc_id: NPC identifier
        npc_name: NPC display name
        lines: List of dialogue lines
        final_choices: Optional final choices for player

    Returns:
        DialogueTree with linear progression
    """
    tree = DialogueTree(
        name=f"{npc_name} Dialogue",
        npc_id=npc_id,
    )

    prev_node_id = None
    for i, line in enumerate(lines):
        node = DialogueNode(
            speaker_id=npc_id,
            speaker_name=npc_name,
            text=line,
            is_entry_point=(i == 0),
        )

        if prev_node_id:
            # Link previous node to this one
            prev_node = tree.get_node(prev_node_id)
            if prev_node:
                prev_node.choices = [DialogueChoice(
                    text="Continue",
                    next_node_id=node.id,
                )]

        tree.add_node(node)
        prev_node_id = node.id

    # Add final choices if provided
    if final_choices and prev_node_id:
        final_node = tree.get_node(prev_node_id)
        if final_node:
            final_node.is_exit_point = True
            final_node.choices = [
                DialogueChoice(text=choice_text)
                for choice_text in final_choices
            ]

    return tree


__all__ = [
    # Conditions
    'ConditionType',
    'DialogueCondition',
    # Effects
    'EffectType',
    'DialogueEffect',
    # Models
    'DialogueChoice',
    'DialogueNode',
    'DialogueTree',
    'DialogueState',
    # Engine
    'BranchingDialogueEngine',
    # Helpers
    'create_simple_dialogue',
]
