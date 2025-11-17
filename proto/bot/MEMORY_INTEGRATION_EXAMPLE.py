#!/usr/bin/env python3
"""
Memory Integration Example for Proto Bot

This file demonstrates how to integrate Week 3 memory features into
the existing Proto/Promptly bot.

Add these snippets to proto/bot/promptly_bot.py to enable:
- Automatic conversation storage
- Memory command handling (@proto remember/recall/related/memories)
- Context-aware responses

Usage:
    1. Copy the __init__ additions to PromptlyBot.__init__
    2. Copy the message_callback additions to message handler
    3. Copy the command handlers to the bot class
    4. Test with @proto remember/recall/related/memories commands
"""


# ============================================================================
# STEP 1: Add to PromptlyBot.__init__() method
# ============================================================================

def __init_additions__(self):
    """
    Add these lines to PromptlyBot.__init__() method.

    Insert after existing integrations (git_handler, code_reviewer, etc.)
    """
    # Initialize Memory Integration (Week 3)
    try:
        from .memory_integration import MemoryIntegration
        from .memory_commands import MemoryCommandHandler

        self.memory = MemoryIntegration()
        self.memory_handler = MemoryCommandHandler(self.memory)
        self.memory_enabled = True

        logger.info("Memory integration initialized (Week 3)")

    except ImportError as e:
        logger.warning(f"Memory integration not available: {e}")
        self.memory = None
        self.memory_handler = None
        self.memory_enabled = False


# ============================================================================
# STEP 2: Add to message_callback() method
# ============================================================================

async def message_callback_additions(self, room, event):
    """
    Add these features to the existing message_callback() method.

    Insert at appropriate locations in the message handler.
    """

    # === AUTO-STORE CONVERSATIONS (Add at start of message_callback) ===
    # Store every message in knowledge graph (Week 3)
    if self.memory_enabled and self.memory:
        try:
            await self.memory.store_conversation(
                user_id=event.sender,
                message=event.body,
                room_id=room.room_id,
                metadata={
                    "event_id": event.event_id,
                    "timestamp": event.server_timestamp
                }
            )
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")

    # === MEMORY COMMAND HANDLING (Add in command routing section) ===
    # Handle memory commands (Week 3)
    if cmd['type'] == 'memory-remember':
        response = await self.handle_memory_remember(
            fact=cmd['fact'],
            user_id=event.sender,
            room_id=room.room_id
        )
        await self.send_message(room.room_id, response)
        return

    elif cmd['type'] == 'memory-recall':
        response = await self.handle_memory_recall(
            query=cmd['query'],
            room_id=room.room_id
        )
        await self.send_message(room.room_id, response)
        return

    elif cmd['type'] == 'memory-related':
        response = await self.handle_memory_related(
            topic=cmd['topic'],
            room_id=room.room_id
        )
        await self.send_message(room.room_id, response)
        return

    elif cmd['type'] == 'memory-stats':
        response = await self.handle_memory_stats(room_id=room.room_id)
        await self.send_message(room.room_id, response)
        return


# ============================================================================
# STEP 3: Add Memory Command Handlers to PromptlyBot class
# ============================================================================

class PromptlyBotMemoryMethods:
    """
    Add these methods to the PromptlyBot class.

    These handle memory-related commands via the memory_handler.
    """

    async def handle_memory_remember(
        self,
        fact: str,
        user_id: str,
        room_id: str
    ) -> str:
        """
        Handle @proto remember <fact> command.

        Args:
            fact: Fact to remember
            user_id: User who issued command
            room_id: Room where command was issued

        Returns:
            Formatted response message
        """
        if not self.memory_enabled:
            return "❌ Memory features not available (HoloLoom not found)"

        try:
            # Initialize if needed
            if not self.memory_handler.initialized:
                await self.memory_handler.initialize()

            # Handle command
            return await self.memory_handler.handle_remember(
                fact=fact,
                user_id=user_id,
                room_id=room_id
            )

        except Exception as e:
            logger.error(f"Memory remember failed: {e}")
            return f"❌ **Error**: {str(e)}"

    async def handle_memory_recall(
        self,
        query: str,
        room_id: str,
        verbose: bool = False
    ) -> str:
        """
        Handle @proto recall <query> command.

        Args:
            query: Search query
            room_id: Room filter
            verbose: Include full details

        Returns:
            Formatted response message
        """
        if not self.memory_enabled:
            return "❌ Memory features not available (HoloLoom not found)"

        try:
            # Initialize if needed
            if not self.memory_handler.initialized:
                await self.memory_handler.initialize()

            # Handle command
            return await self.memory_handler.handle_recall(
                query=query,
                room_id=room_id,
                verbose=verbose
            )

        except Exception as e:
            logger.error(f"Memory recall failed: {e}")
            return f"❌ **Error**: {str(e)}"

    async def handle_memory_related(
        self,
        topic: str,
        room_id: str
    ) -> str:
        """
        Handle @proto related <topic> command.

        Args:
            topic: Topic to search for
            room_id: Room filter

        Returns:
            Formatted response message
        """
        if not self.memory_enabled:
            return "❌ Memory features not available (HoloLoom not found)"

        try:
            # Initialize if needed
            if not self.memory_handler.initialized:
                await self.memory_handler.initialize()

            # Handle command
            return await self.memory_handler.handle_related(
                topic=topic,
                room_id=room_id
            )

        except Exception as e:
            logger.error(f"Memory related failed: {e}")
            return f"❌ **Error**: {str(e)}"

    async def handle_memory_stats(self, room_id: str) -> str:
        """
        Handle @proto memories command.

        Args:
            room_id: Room filter

        Returns:
            Formatted response message
        """
        if not self.memory_enabled:
            return "❌ Memory features not available (HoloLoom not found)"

        try:
            # Initialize if needed
            if not self.memory_handler.initialized:
                await self.memory_handler.initialize()

            # Handle command
            return await self.memory_handler.handle_memories(
                room_id=room_id
            )

        except Exception as e:
            logger.error(f"Memory stats failed: {e}")
            return f"❌ **Error**: {str(e)}"


# ============================================================================
# STEP 4: Add Context-Aware Response Enhancement (Optional)
# ============================================================================

async def enhance_response_with_context(self, query: str, room_id: str) -> str:
    """
    Optional: Enhance generic responses with team context.

    Call this before generating any bot response to enrich with
    relevant team knowledge.

    Args:
        query: User query/question
        room_id: Room ID for context filtering

    Returns:
        Context string to prepend to response
    """
    if not self.memory_enabled or not self.memory:
        return ""

    try:
        # Initialize if needed
        if not self.memory.initialized:
            await self.memory.initialize()

        # Get relevant context
        context = await self.memory.get_context(
            query=query,
            room_id=room_id,
            limit=3
        )

        # If we have relevant memories, format context
        if context.memories and context.confidence > 0.5:
            context_lines = ["**Based on team discussions:**\n"]

            for memory in context.memories[:2]:
                time_ago = self._format_time_ago(memory.timestamp)
                context_lines.append(
                    f"• {memory.message[:80]}... ({time_ago})"
                )

            context_lines.append("\n")
            return "\n".join(context_lines)

        return ""

    except Exception as e:
        logger.error(f"Context enhancement failed: {e}")
        return ""


# ============================================================================
# COMPLETE EXAMPLE: How to integrate into existing bot
# ============================================================================

"""
INTEGRATION CHECKLIST:

1. ✅ Install HoloLoom in parent directory (../HoloLoom)

2. ✅ Update proto/bot/promptly_bot.py:

   In __init__():
   ----------------
   # Add after git_handler initialization
   from .memory_integration import MemoryIntegration
   from .memory_commands import MemoryCommandHandler

   self.memory = MemoryIntegration()
   self.memory_handler = MemoryCommandHandler(self.memory)
   self.memory_enabled = True

   In message_callback():
   ----------------------
   # At start (auto-store all conversations)
   if self.memory_enabled:
       await self.memory.store_conversation(
           user_id=event.sender,
           message=event.body,
           room_id=room.room_id
       )

   # In command routing section
   if cmd['type'] == 'memory-remember':
       response = await self.handle_memory_remember(...)
       await self.send_message(room.room_id, response)
       return

   # Add handlers for memory-recall, memory-related, memory-stats

   Add methods to PromptlyBot class:
   ----------------------------------
   - handle_memory_remember()
   - handle_memory_recall()
   - handle_memory_related()
   - handle_memory_stats()

3. ✅ Update proto/bot/command_parser.py (ALREADY DONE):
   - Memory command patterns added
   - Extraction logic added

4. ✅ Test in Matrix:
   @proto remember We use PostgreSQL for auth
   @proto recall database
   @proto related authentication
   @proto memories

5. ✅ Read documentation:
   - proto/MEMORY_INTEGRATION.md (complete guide)
   - proto/bot/memory_integration.py (API reference)
   - proto/bot/memory_commands.py (command handlers)
"""


# ============================================================================
# MINIMAL WORKING EXAMPLE
# ============================================================================

"""
Minimal integration (just add to message_callback):

async def message_callback(self, room, event):
    # ... existing code ...

    # Initialize memory if needed
    if not hasattr(self, 'memory_handler'):
        from .memory_integration import MemoryIntegration
        from .memory_commands import MemoryCommandHandler

        self.memory = MemoryIntegration()
        self.memory_handler = MemoryCommandHandler(self.memory)
        await self.memory_handler.initialize()

    # Auto-store conversation
    await self.memory.store_conversation(
        user_id=event.sender,
        message=event.body,
        room_id=room.room_id
    )

    # Handle memory commands
    cmd = self.parser.parse(event.body)
    if cmd and cmd['type'].startswith('memory-'):
        if cmd['type'] == 'memory-remember':
            response = await self.memory_handler.handle_remember(
                fact=cmd['fact'],
                user_id=event.sender,
                room_id=room.room_id
            )
        elif cmd['type'] == 'memory-recall':
            response = await self.memory_handler.handle_recall(
                query=cmd['query'],
                room_id=room.room_id
            )
        elif cmd['type'] == 'memory-related':
            response = await self.memory_handler.handle_related(
                topic=cmd['topic'],
                room_id=room.room_id
            )
        elif cmd['type'] == 'memory-stats':
            response = await self.memory_handler.handle_memories(
                room_id=room.room_id
            )

        await self.send_message(room.room_id, response)
        return

    # ... rest of existing code ...
"""


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("INTEGRATION EXAMPLE - Copy snippets to promptly_bot.py")
    print("="*60)
    print("\nSee proto/MEMORY_INTEGRATION.md for complete documentation")
