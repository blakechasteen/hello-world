# conversation.py
# Multi-NPC Conversation System
# Version: 1.0.0
#
# Enables NPCs to converse with each other, creating dynamic group conversations
# and emergent narratives.

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from enum import Enum

from .models import (
    GameStateSnapshot,
    NPCState,
    PlayerIntent,
    ElleGameAction,
    DialogueLine
)
from .policy import GamePolicy
from .llm_client import LLMClientProtocol


class ConversationMode(Enum):
    """Conversation modes"""
    TWO_NPC = "two_npc"               # Two NPCs talking
    GROUP = "group"                    # 3+ NPCs in conversation
    PLAYER_MEDIATED = "player_mediated"  # Player facilitating NPC conversation


@dataclass
class ConversationTurn:
    """Single turn in multi-NPC conversation"""
    speaker_npc_id: str
    listener_npc_ids: List[str]
    text: str
    tone: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    is_player: bool = False


@dataclass
class ActiveConversation:
    """Active multi-NPC conversation"""
    conversation_id: str
    participant_npc_ids: Set[str]
    mode: ConversationMode
    turns: List[ConversationTurn] = field(default_factory=list)
    topic: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    max_turns: int = 10
    turn_timeout_seconds: float = 300.0  # 5 minutes

    def is_active(self) -> bool:
        """Check if conversation is still active"""
        if len(self.turns) >= self.max_turns:
            return False

        time_since_last = (datetime.now() - self.last_activity).total_seconds()
        return time_since_last < self.turn_timeout_seconds

    def add_turn(self, turn: ConversationTurn):
        """Add turn to conversation"""
        self.turns.append(turn)
        self.last_activity = datetime.now()

    def get_history_for_npc(self, npc_id: str, last_n: int = 5) -> List[ConversationTurn]:
        """Get conversation history relevant to NPC"""
        # Return last N turns where NPC was involved (speaker or listener)
        relevant_turns = [
            t for t in self.turns
            if t.speaker_npc_id == npc_id or npc_id in t.listener_npc_ids
        ]
        return relevant_turns[-last_n:]


class ConversationCoordinator:
    """
    Coordinates multi-NPC conversations.

    Features:
    - Two-NPC conversations
    - Group conversations (3+ NPCs)
    - Player-mediated conversations
    - Conversation context management
    - Turn taking logic
    - Conversation timeout/completion
    """

    def __init__(self, policy: GamePolicy):
        self.policy = policy
        self.active_conversations: Dict[str, ActiveConversation] = {}

    def start_conversation(
        self,
        npc_ids: List[str],
        topic: Optional[str] = None,
        mode: Optional[ConversationMode] = None
    ) -> str:
        """
        Start a new multi-NPC conversation

        Args:
            npc_ids: List of NPC IDs participating
            topic: Optional conversation topic
            mode: Optional conversation mode (auto-detected if None)

        Returns:
            conversation_id: Unique conversation identifier
        """
        if len(npc_ids) < 2:
            raise ValueError("Need at least 2 NPCs for conversation")

        # Auto-detect mode
        if mode is None:
            if len(npc_ids) == 2:
                mode = ConversationMode.TWO_NPC
            else:
                mode = ConversationMode.GROUP

        # Generate conversation ID
        conversation_id = f"conv_{datetime.now().timestamp()}_{len(self.active_conversations)}"

        # Create conversation
        conversation = ActiveConversation(
            conversation_id=conversation_id,
            participant_npc_ids=set(npc_ids),
            mode=mode,
            topic=topic
        )

        self.active_conversations[conversation_id] = conversation

        return conversation_id

    async def next_turn(
        self,
        conversation_id: str,
        game_state: GameStateSnapshot,
        player_input: Optional[str] = None
    ) -> ElleGameAction:
        """
        Get next turn in conversation

        Args:
            conversation_id: Conversation identifier
            game_state: Current game state
            player_input: Optional player input (for player-mediated conversations)

        Returns:
            ElleGameAction with next NPC's dialogue
        """
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        if not conversation.is_active():
            return self._end_conversation_action(conversation)

        # Determine next speaker
        next_speaker_id = self._determine_next_speaker(conversation, game_state)

        # Build context for speaker
        context = self._build_conversation_context(
            conversation,
            next_speaker_id,
            game_state,
            player_input
        )

        # Get NPC response via policy
        action = await self.policy.get_action(game_state, context)

        # Add turn to conversation
        if action.dialogue:
            turn = ConversationTurn(
                speaker_npc_id=next_speaker_id,
                listener_npc_ids=[
                    npc_id for npc_id in conversation.participant_npc_ids
                    if npc_id != next_speaker_id
                ],
                text=action.dialogue[0].text,
                tone=action.dialogue[0].tone
            )
            conversation.add_turn(turn)

        return action

    def get_conversation(self, conversation_id: str) -> Optional[ActiveConversation]:
        """Get conversation by ID"""
        return self.active_conversations.get(conversation_id)

    def end_conversation(self, conversation_id: str) -> bool:
        """End conversation and remove from active list"""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            return True
        return False

    def cleanup_inactive_conversations(self):
        """Remove conversations that have timed out or completed"""
        inactive = [
            conv_id for conv_id, conv in self.active_conversations.items()
            if not conv.is_active()
        ]
        for conv_id in inactive:
            del self.active_conversations[conv_id]

    def _determine_next_speaker(
        self,
        conversation: ActiveConversation,
        game_state: GameStateSnapshot
    ) -> str:
        """
        Determine which NPC speaks next

        Logic:
        1. If no turns yet, pick random/first NPC
        2. If two-NPC mode, alternate
        3. If group mode, use round-robin or intelligent selection
        """
        if not conversation.turns:
            # First turn - pick first NPC or most relevant
            return list(conversation.participant_npc_ids)[0]

        if conversation.mode == ConversationMode.TWO_NPC:
            # Alternate between two NPCs
            last_speaker = conversation.turns[-1].speaker_npc_id
            participants = list(conversation.participant_npc_ids)
            other_npc = [npc for npc in participants if npc != last_speaker][0]
            return other_npc

        # Group mode - round-robin
        last_speaker = conversation.turns[-1].speaker_npc_id
        participants = sorted(list(conversation.participant_npc_ids))
        last_index = participants.index(last_speaker)
        next_index = (last_index + 1) % len(participants)
        return participants[next_index]

    def _build_conversation_context(
        self,
        conversation: ActiveConversation,
        speaker_npc_id: str,
        game_state: GameStateSnapshot,
        player_input: Optional[str] = None
    ) -> PlayerIntent:
        """
        Build conversation context for NPC

        Creates a PlayerIntent that includes:
        - Conversation history
        - Other participants
        - Topic (if any)
        - Player input (if any)
        """
        # Get conversation history for this NPC
        history = conversation.get_history_for_npc(speaker_npc_id, last_n=5)

        # Build context prompt
        context_parts = []

        if conversation.topic:
            context_parts.append(f"Topic: {conversation.topic}")

        if history:
            context_parts.append("Recent conversation:")
            for turn in history:
                speaker = turn.speaker_npc_id
                text = turn.text
                context_parts.append(f"  {speaker}: {text}")

        if player_input:
            context_parts.append(f"Player says: {player_input}")

        # Build other participants list
        other_npcs = [
            npc_id for npc_id in conversation.participant_npc_ids
            if npc_id != speaker_npc_id
        ]

        context_parts.append(f"You are talking to: {', '.join(other_npcs)}")
        context_parts.append(f"It's your turn to speak ({speaker_npc_id}).")

        context_text = "\n".join(context_parts)

        # Create PlayerIntent
        return PlayerIntent(
            type="talk_to_npc",
            target_npc_id=speaker_npc_id,
            raw_input=context_text
        )

    def _end_conversation_action(self, conversation: ActiveConversation) -> ElleGameAction:
        """Create action indicating conversation has ended"""
        return ElleGameAction(
            mode="world_reaction",
            priority="low",
            world_reaction={
                "description": f"The conversation between {', '.join(conversation.participant_npc_ids)} comes to a natural conclusion.",
                "flag_changes": {
                    f"conversation_{conversation.conversation_id}_ended": True
                }
            }
        )


async def initiate_npc_conversation(
    coordinator: ConversationCoordinator,
    game_state: GameStateSnapshot,
    initiator_npc_id: str,
    target_npc_id: str,
    opening_line: Optional[str] = None
) -> tuple[str, ElleGameAction]:
    """
    Initiate conversation between two NPCs

    Args:
        coordinator: Conversation coordinator
        game_state: Current game state
        initiator_npc_id: NPC starting conversation
        target_npc_id: NPC being talked to
        opening_line: Optional opening line from initiator

    Returns:
        (conversation_id, initial_action)
    """
    # Start conversation
    conversation_id = coordinator.start_conversation(
        [initiator_npc_id, target_npc_id],
        mode=ConversationMode.TWO_NPC
    )

    conversation = coordinator.get_conversation(conversation_id)

    # Add opening turn if provided
    if opening_line:
        turn = ConversationTurn(
            speaker_npc_id=initiator_npc_id,
            listener_npc_ids=[target_npc_id],
            text=opening_line
        )
        conversation.add_turn(turn)

    # Get response from target NPC
    action = await coordinator.next_turn(conversation_id, game_state)

    return conversation_id, action


async def run_npc_conversation(
    coordinator: ConversationCoordinator,
    game_state: GameStateSnapshot,
    npc_ids: List[str],
    topic: Optional[str] = None,
    max_turns: int = 10
) -> List[ConversationTurn]:
    """
    Run complete NPC conversation automatically

    Args:
        coordinator: Conversation coordinator
        game_state: Current game state
        npc_ids: NPCs participating
        topic: Optional topic
        max_turns: Maximum turns

    Returns:
        List of conversation turns
    """
    # Start conversation
    conversation_id = coordinator.start_conversation(
        npc_ids,
        topic=topic
    )

    conversation = coordinator.get_conversation(conversation_id)
    conversation.max_turns = max_turns

    # Run conversation
    turns = []
    while conversation.is_active():
        action = await coordinator.next_turn(conversation_id, game_state)

        if action.dialogue:
            turns.append(conversation.turns[-1])

    # Cleanup
    coordinator.end_conversation(conversation_id)

    return turns
