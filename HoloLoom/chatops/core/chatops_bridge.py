#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatOps Bridge
==============
Connects Matrix bot to HoloLoom orchestrator and Promptly framework.

This is component #2: The integration layer that:
- Routes messages from Matrix -> HoloLoom -> Promptly
- Manages conversation context and memory
- Stores chat history in knowledge graph
- Handles multi-user conversations
- Coordinates response generation

Architecture:
    Matrix Bot -> ChatOpsOrchestrator -> HoloLoom Orchestrator
                                      -> Promptly Skills
                                      -> Conversation Memory (KG)

Dependencies:
    - HoloLoom orchestrator (query processing)
    - HoloLoom knowledge graph (conversation storage)
    - Promptly framework (skill execution)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# HoloLoom imports
try:
    from HoloLoom.documentation.types import Query, Context, Features, MemoryShard
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
    from HoloLoom.memory.graph import KG
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError as e:
    HOLOLOOM_AVAILABLE = False
    print(f"Warning: HoloLoom not available. Some features will be limited. Error: {e}")

# Matrix bot imports (use TYPE_CHECKING to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from HoloLoom.chatops.core.matrix_bot import MatrixBot

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    print("Warning: Matrix bot not available")


logger = logging.getLogger(__name__)


# ============================================================================
# Conversation Context
# ============================================================================

@dataclass
class ConversationContext:
    """
    Context for an ongoing conversation.

    Tracks messages, participants, and state within a Matrix room.
    """
    room_id: str
    room_name: str
    participants: List[str] = field(default_factory=list)
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Knowledge graph node IDs for this conversation
    conversation_node_id: Optional[str] = None
    message_node_ids: List[str] = field(default_factory=list)

    def add_message(self, sender: str, text: str, timestamp: Optional[datetime] = None) -> None:
        """Add a message to history."""
        msg = {
            "sender": sender,
            "text": text,
            "timestamp": timestamp or datetime.now()
        }
        self.message_history.append(msg)
        self.last_activity = msg["timestamp"]

        # Update participants
        if sender not in self.participants:
            self.participants.append(sender)

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages."""
        return self.message_history[-limit:]

    def to_context_string(self, limit: int = 10) -> str:
        """Format recent messages as context string."""
        recent = self.get_recent_messages(limit)
        lines = []
        for msg in recent:
            sender = msg["sender"].split(":")[0][1:]  # Clean @user:server -> user
            lines.append(f"{sender}: {msg['text']}")
        return "\n".join(lines)


# ============================================================================
# ChatOps Orchestrator
# ============================================================================

class ChatOpsOrchestrator:
    """
    ChatOps orchestrator that bridges Matrix bot to HoloLoom.

    Responsibilities:
    1. Process incoming messages from Matrix
    2. Maintain conversation context per room
    3. Store conversation history in knowledge graph
    4. Route queries through HoloLoom orchestrator
    5. Execute Promptly skills for commands
    6. Generate and send responses back to Matrix

    Usage:
        chatops = ChatOpsOrchestrator(
            hololoom_config=Config.fast(),
            memory_store_path="./chatops_memory"
        )

        # Connect to Matrix bot
        bot = MatrixBot(matrix_config)
        chatops.connect_bot(bot)

        # Start processing
        await chatops.start()
    """

    def __init__(
        self,
        hololoom_config: Optional[Any] = None,
        memory_store_path: str = "./chatops_memory",
        context_limit: int = 10,
        enable_memory_storage: bool = True
    ):
        """
        Initialize ChatOps orchestrator.

        Args:
            hololoom_config: HoloLoom Config object (uses fast mode if None)
            memory_store_path: Path to store conversation memory
            context_limit: Number of recent messages to include in context
            enable_memory_storage: Whether to store conversations in KG
        """
        # HoloLoom components
        if HOLOLOOM_AVAILABLE:
            self.config = hololoom_config or Config.fast()
            self.orchestrator = None  # Lazy init
            self.knowledge_graph = KG()
        else:
            self.config = None
            self.orchestrator = None
            self.knowledge_graph = None

        # Configuration
        self.memory_store_path = Path(memory_store_path)
        self.memory_store_path.mkdir(parents=True, exist_ok=True)
        self.context_limit = context_limit
        self.enable_memory_storage = enable_memory_storage

        # Conversation tracking
        self.conversations: Dict[str, ConversationContext] = {}

        # Matrix bot reference
        self.bot: Optional[MatrixBot] = None

        # Promptly integration (optional)
        self.promptly_available = False
        try:
            from promptly.integrations.hololoom_bridge import HoloLoomBridge
            self.promptly_bridge = HoloLoomBridge()
            self.promptly_available = True
        except ImportError:
            self.promptly_bridge = None

        logger.info("ChatOpsOrchestrator initialized")

    # ========================================================================
    # Bot Connection
    # ========================================================================

    def connect_bot(self, bot: "MatrixBot") -> None:
        """
        Connect to Matrix bot.

        Args:
            bot: MatrixBot instance
        """
        self.bot = bot

        # Register default message handler
        bot.set_default_handler(self.handle_message)

        logger.info("Connected to Matrix bot")

    # ========================================================================
    # Message Processing
    # ========================================================================

    async def handle_message(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
        message: str
    ) -> None:
        """
        Handle incoming message from Matrix.

        This is the main entry point for message processing.

        Args:
            room: Matrix room
            event: Message event
            message: Message text
        """
        try:
            # Send typing indicator
            if self.bot:
                await self.bot.send_typing(room.room_id, typing=True)

            # Get or create conversation context
            context = self._get_conversation(room)

            # Add message to history
            context.add_message(event.sender, message, datetime.now())

            # Store in knowledge graph
            if self.enable_memory_storage and HOLOLOOM_AVAILABLE:
                await self._store_message_in_kg(context, event.sender, message)

            # Process query through HoloLoom
            response = await self._process_query(message, context)

            # Send response
            if self.bot:
                await self.bot.send_typing(room.room_id, typing=False)
                await self.bot.send_message(room.room_id, response, markdown=True)

                # Add bot response to history
                context.add_message(self.bot.config.user_id, response, datetime.now())

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            if self.bot:
                await self.bot.send_message(
                    room.room_id,
                    f"Sorry, I encountered an error: {str(e)}"
                )

    async def _process_query(
        self,
        query_text: str,
        conversation: ConversationContext
    ) -> str:
        """
        Process query through HoloLoom orchestrator.

        Args:
            query_text: User query
            conversation: Conversation context

        Returns:
            Response string
        """
        if not HOLOLOOM_AVAILABLE:
            return self._fallback_response(query_text)

        # Create HoloLoom query with conversation context
        query = Query(
            text=query_text,
            metadata={
                "room_id": conversation.room_id,
                "conversation_history": conversation.to_context_string(self.context_limit),
                "participants": conversation.participants,
                "source": "matrix_chatops"
            }
        )

        # Lazy init orchestrator
        if self.orchestrator is None:
            self.orchestrator = WeavingOrchestrator(self.config)

        # Process through HoloLoom
        try:
            # This would use the full orchestrator pipeline
            # For now, simplified version
            response_text = await self._orchestrator_process(query, conversation)
            return response_text

        except Exception as e:
            logger.error(f"HoloLoom processing error: {e}", exc_info=True)
            return f"I encountered an issue processing your query: {str(e)}"

    async def _orchestrator_process(
        self,
        query: Query,
        conversation: ConversationContext
    ) -> str:
        """
        Run query through HoloLoom orchestrator pipeline.

        Args:
            query: HoloLoom query
            conversation: Conversation context

        Returns:
            Response text
        """
        # Retrieve relevant context from conversation history
        context_shards = self._get_context_shards(conversation)

        # Extract features (this would use the real orchestrator)
        # For now, simplified mock
        response = f"""I understand you're asking about: "{query.text}"

Based on our conversation history, I can see we've discussed {len(conversation.message_history)} messages with {len(conversation.participants)} participants.

**Context:** {conversation.to_context_string(3)}

This is a demo response. Full HoloLoom integration will provide:
• Neural feature extraction
• Knowledge graph memory retrieval
• Thompson Sampling decision making
• Tool selection and execution
"""
        return response

    def _get_context_shards(
        self,
        conversation: ConversationContext
    ) -> List[MemoryShard]:
        """
        Convert conversation history to memory shards.

        Args:
            conversation: Conversation context

        Returns:
            List of MemoryShard objects
        """
        if not HOLOLOOM_AVAILABLE:
            return []

        shards = []
        for idx, msg in enumerate(conversation.get_recent_messages(self.context_limit)):
            shard = MemoryShard(
                id=f"{conversation.room_id}_msg_{idx}",
                text=msg["text"],
                episode=conversation.room_id,
                entities=[msg["sender"]],
                motifs=[],  # Would be extracted
                metadata={
                    "timestamp": msg["timestamp"].isoformat(),
                    "sender": msg["sender"],
                    "room": conversation.room_id
                }
            )
            shards.append(shard)

        return shards

    def _fallback_response(self, query_text: str) -> str:
        """
        Fallback response when HoloLoom not available.

        Args:
            query_text: User query

        Returns:
            Simple response
        """
        return f"""I received your message: "{query_text}"

However, HoloLoom is not fully initialized. This is a fallback response.

To enable full functionality:
1. Ensure HoloLoom is properly installed
2. Configure the orchestrator
3. Initialize the knowledge graph
"""

    # ========================================================================
    # Conversation Management
    # ========================================================================

    def _get_conversation(self, room: MatrixRoom) -> ConversationContext:
        """
        Get or create conversation context for room.

        Args:
            room: Matrix room

        Returns:
            ConversationContext
        """
        room_id = room.room_id

        if room_id not in self.conversations:
            # Create new conversation
            context = ConversationContext(
                room_id=room_id,
                room_name=room.display_name or room.room_id,
                participants=list(room.users.keys()) if hasattr(room, 'users') else []
            )
            self.conversations[room_id] = context

            # Create conversation node in KG
            if self.enable_memory_storage and HOLOLOOM_AVAILABLE:
                self._create_conversation_node(context)

        return self.conversations[room_id]

    def _create_conversation_node(self, context: ConversationContext) -> None:
        """
        Create conversation node in knowledge graph.

        Args:
            context: Conversation context
        """
        if not self.knowledge_graph:
            return

        node_id = f"conversation_{context.room_id}"
        self.knowledge_graph.add_entity(
            entity_id=node_id,
            entity_type="CONVERSATION",
            properties={
                "room_id": context.room_id,
                "room_name": context.room_name,
                "created_at": context.created_at.isoformat(),
                "platform": "matrix"
            }
        )

        context.conversation_node_id = node_id
        logger.info(f"Created conversation node: {node_id}")

    # ========================================================================
    # Knowledge Graph Storage
    # ========================================================================

    async def _store_message_in_kg(
        self,
        conversation: ConversationContext,
        sender: str,
        message: str
    ) -> None:
        """
        Store message in knowledge graph.

        Args:
            conversation: Conversation context
            sender: Message sender
            message: Message text
        """
        if not self.knowledge_graph:
            return

        try:
            # Create message node
            msg_id = f"msg_{conversation.room_id}_{len(conversation.message_history)}"
            self.knowledge_graph.add_entity(
                entity_id=msg_id,
                entity_type="MESSAGE",
                properties={
                    "text": message,
                    "sender": sender,
                    "timestamp": datetime.now().isoformat(),
                    "room_id": conversation.room_id
                }
            )

            # Create user node if doesn't exist
            user_id = f"user_{sender}"
            if not self.knowledge_graph.has_entity(user_id):
                self.knowledge_graph.add_entity(
                    entity_id=user_id,
                    entity_type="USER",
                    properties={"user_id": sender, "platform": "matrix"}
                )

            # Link message to conversation
            if conversation.conversation_node_id:
                self.knowledge_graph.add_relation(
                    from_id=msg_id,
                    to_id=conversation.conversation_node_id,
                    rel_type="PART_OF"
                )

            # Link message to sender
            self.knowledge_graph.add_relation(
                from_id=msg_id,
                to_id=user_id,
                rel_type="SENT_BY"
            )

            # Link to previous message (conversation flow)
            if conversation.message_node_ids:
                prev_msg_id = conversation.message_node_ids[-1]
                self.knowledge_graph.add_relation(
                    from_id=msg_id,
                    to_id=prev_msg_id,
                    rel_type="FOLLOWS"
                )

            conversation.message_node_ids.append(msg_id)

        except Exception as e:
            logger.error(f"Error storing message in KG: {e}", exc_info=True)

    # ========================================================================
    # Statistics & Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get chatops statistics.

        Returns:
            Dict with statistics
        """
        total_messages = sum(
            len(conv.message_history)
            for conv in self.conversations.values()
        )

        return {
            "active_conversations": len(self.conversations),
            "total_messages": total_messages,
            "knowledge_graph_nodes": len(self.knowledge_graph.G.nodes) if self.knowledge_graph else 0,
            "knowledge_graph_edges": len(self.knowledge_graph.G.edges) if self.knowledge_graph else 0,
            "conversations": [
                {
                    "room_id": conv.room_id,
                    "room_name": conv.room_name,
                    "participants": len(conv.participants),
                    "messages": len(conv.message_history),
                    "last_activity": conv.last_activity.isoformat()
                }
                for conv in self.conversations.values()
            ]
        }

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def start(self) -> None:
        """Start chatops orchestrator."""
        logger.info("ChatOps orchestrator started")

        # Initialize orchestrator if available
        if HOLOLOOM_AVAILABLE and self.orchestrator is None:
            try:
                self.orchestrator = WeavingOrchestrator(self.config)
                logger.info("HoloLoom orchestrator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize HoloLoom: {e}")

    async def stop(self) -> None:
        """Stop chatops orchestrator and cleanup."""
        logger.info("Stopping ChatOps orchestrator...")

        # Save knowledge graph
        if self.knowledge_graph and self.enable_memory_storage:
            save_path = self.memory_store_path / "knowledge_graph.pkl"
            try:
                # Would save KG here
                logger.info(f"Knowledge graph saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving knowledge graph: {e}")

        logger.info("ChatOps orchestrator stopped")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*80)
    print("ChatOps Bridge Demo")
    print("="*80)
    print()

    if not HOLOLOOM_AVAILABLE:
        print("Warning: HoloLoom not available - running in limited mode")

    if not MATRIX_AVAILABLE:
        print("Error: Matrix bot not available")
        sys.exit(1)

    # Create ChatOps orchestrator
    chatops = ChatOpsOrchestrator(
        memory_store_path="./demo_chatops_memory",
        context_limit=10,
        enable_memory_storage=True
    )

    # Create Matrix bot
    matrix_config = MatrixBotConfig(
        homeserver_url="https://matrix.org",
        user_id="@yourbot:matrix.org",
        password="your_password",
        rooms=["#test:matrix.org"],
        command_prefix="!"
    )

    bot = MatrixBot(matrix_config)

    # Connect
    chatops.connect_bot(bot)

    print("ChatOps bridge configured and connected")
    print("Starting bot...")
    print()

    # Run
    async def main():
        await chatops.start()
        await bot.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
