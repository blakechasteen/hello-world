#!/usr/bin/env python3
"""
Memory Command Handlers for Proto Matrix Bot

Handles memory-related commands:
- @proto remember <fact> - Store fact in team knowledge base
- @proto recall <query> - Search team knowledge
- @proto related <topic> - Find connected discussions
- @proto memories - Show memory statistics

Integrates with HoloLoom knowledge graph for institutional memory.

Usage:
    from bot.memory_commands import MemoryCommandHandler

    handler = MemoryCommandHandler(memory_integration)
    await handler.initialize()

    # In bot message handler
    if "@proto remember" in message:
        response = await handler.handle_remember(
            fact="We use PostgreSQL for auth",
            user_id="@alice:matrix.org",
            room_id="!team:matrix.org"
        )
        await send_message(response)
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from .memory_integration import MemoryIntegration, ContextResult

logger = logging.getLogger(__name__)


class MemoryCommandHandler:
    """
    Memory command handler for Proto bot.

    Handles @proto memory-related commands with rich formatting
    for Matrix responses.

    Example:
        >>> handler = MemoryCommandHandler(memory)
        >>> await handler.initialize()
        >>>
        >>> response = await handler.handle_remember(
        ...     fact="Redis for caching",
        ...     user_id="@alice:matrix.org",
        ...     room_id="!team:matrix.org"
        ... )
        >>> print(response)
    """

    def __init__(self, memory: Optional[MemoryIntegration] = None):
        """
        Initialize memory command handler.

        Args:
            memory: Optional MemoryIntegration instance (creates new if None)
        """
        self.memory = memory or MemoryIntegration()
        self.initialized = False

    async def initialize(self):
        """Initialize memory integration"""
        if not self.initialized:
            await self.memory.initialize()
            self.initialized = True
            logger.info("Memory command handler initialized")

    async def handle_remember(
        self,
        fact: str,
        user_id: str,
        room_id: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Handle @proto remember <fact> command.

        Stores fact in team knowledge base with automatic entity extraction.

        Args:
            fact: Fact to remember
            user_id: User who issued command
            room_id: Room where command was issued
            tags: Optional tags

        Returns:
            Formatted response message

        Example:
            >>> response = await handler.handle_remember(
            ...     fact="PostgreSQL for auth service",
            ...     user_id="@alice:matrix.org",
            ...     room_id="!team:matrix.org"
            ... )
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Store the fact
            memory_id = await self.memory.remember_fact(
                fact=fact,
                user_id=user_id,
                room_id=room_id,
                tags=tags
            )

            # Get stored memory details
            context = await self.memory.get_context(
                query=fact,
                room_id=room_id,
                limit=1
            )

            if context.memories:
                stored = context.memories[0]
                entities = stored.entities
                topics = stored.topics
            else:
                # Fallback to extraction
                entities = self.memory._extract_entities(fact)
                topics = self.memory._detect_topics(fact)

            # Format response
            response_lines = [
                "🧠 **Stored in team memory:**",
                f'"{fact}"',
                "",
                "**Added to knowledge graph:**"
            ]

            if entities:
                response_lines.append(f"• **Entities**: {', '.join(entities[:5])}")

            if topics:
                response_lines.append(f"• **Topics**: {', '.join(topics[:5])}")

            if tags:
                response_lines.append(f"• **Tags**: {', '.join(tags)}")

            response_lines.extend([
                "",
                f"**Stored by**: {user_id}",
                f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "💡 *Use `@proto recall <query>` to retrieve this later*"
            ])

            logger.info(f"Stored fact from {user_id}: {fact[:100]}...")

            return "\n".join(response_lines)

        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return f"❌ **Error storing fact**: {str(e)}"

    async def handle_recall(
        self,
        query: str,
        room_id: Optional[str] = None,
        limit: int = 5,
        verbose: bool = False
    ) -> str:
        """
        Handle @proto recall <query> command.

        Searches team knowledge base and returns relevant memories.

        Args:
            query: Search query
            room_id: Optional room filter
            limit: Maximum results to return
            verbose: Include full context details

        Returns:
            Formatted response message

        Example:
            >>> response = await handler.handle_recall(
            ...     query="database decisions",
            ...     room_id="!team:matrix.org"
            ... )
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Search knowledge graph
            context = await self.memory.get_context(
                query=query,
                room_id=room_id,
                limit=limit,
                include_related=True
            )

            # Format response
            if not context.memories:
                return self._format_no_results(query)

            return self._format_recall_results(context, query, verbose)

        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return f"❌ **Error searching memories**: {str(e)}"

    async def handle_related(
        self,
        topic: str,
        room_id: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """
        Handle @proto related <topic> command.

        Finds discussions related to a topic via knowledge graph traversal.

        Args:
            topic: Topic to search for
            room_id: Optional room filter
            limit: Maximum results

        Returns:
            Formatted response message

        Example:
            >>> response = await handler.handle_related(
            ...     topic="authentication",
            ...     room_id="!team:matrix.org"
            ... )
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Find related conversations
            related = await self.memory.find_related(
                topic=topic,
                room_id=room_id,
                limit=limit
            )

            # Format response
            if not related:
                return f"🔍 **No related discussions found for**: {topic}"

            response_lines = [
                f"🔗 **Related discussions about**: {topic}",
                f"Found {len(related)} related conversation(s)",
                ""
            ]

            # Group by topic
            by_topic = {}
            for memory in related:
                for t in memory.topics:
                    if t not in by_topic:
                        by_topic[t] = []
                    by_topic[t].append(memory)

            # Show grouped results
            for topic_name, memories in list(by_topic.items())[:5]:
                response_lines.append(f"**{topic_name.title()}** ({len(memories)} mention(s)):")

                for mem in memories[:3]:  # Max 3 per topic
                    time_ago = self._format_time_ago(mem.timestamp)
                    response_lines.append(
                        f"  • {mem.user_id} ({time_ago}): {mem.message[:80]}..."
                    )

                response_lines.append("")

            # Add related topics footer
            all_topics = set()
            for mem in related:
                all_topics.update(mem.topics)

            if len(all_topics) > 1:
                other_topics = [t for t in all_topics if t != topic]
                if other_topics:
                    response_lines.extend([
                        "**Also mentioned**: " + ", ".join(other_topics[:5]),
                        ""
                    ])

            response_lines.append(
                "💡 *Use `@proto recall <query>` for detailed search*"
            )

            logger.info(f"Found {len(related)} related discussions for topic: {topic}")

            return "\n".join(response_lines)

        except Exception as e:
            logger.error(f"Failed to find related topics: {e}")
            return f"❌ **Error finding related topics**: {str(e)}"

    async def handle_memories(
        self,
        room_id: Optional[str] = None
    ) -> str:
        """
        Handle @proto memories command.

        Shows memory statistics for the room or globally.

        Args:
            room_id: Optional room filter

        Returns:
            Formatted response message

        Example:
            >>> response = await handler.handle_memories(
            ...     room_id="!team:matrix.org"
            ... )
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Get statistics
            stats = await self.memory.get_statistics(room_id=room_id)

            # Format response
            scope = f"room {room_id}" if room_id else "all rooms"

            response_lines = [
                f"📊 **Memory Statistics** ({scope})",
                "",
                f"**Total Memories**: {stats['total_memories']}",
                f"**Connections**: {stats['total_connections']}",
                f"**Active**: {stats['active_memories']}",
                f"**Cached**: {stats['cached_conversations']}",
                "",
                f"**Unique Topics**: {stats['unique_topics']}",
                f"**Unique Entities**: {stats['unique_entities']}",
                ""
            ]

            # Sample topics
            if stats['sample_topics']:
                response_lines.append("**Recent Topics**:")
                for topic in stats['sample_topics'][:8]:
                    response_lines.append(f"  • {topic}")
                response_lines.append("")

            # Sample entities
            if stats['sample_entities']:
                response_lines.append("**Mentioned Entities**:")
                for entity in stats['sample_entities'][:8]:
                    response_lines.append(f"  • {entity}")

            logger.info(f"Returned statistics for {scope}")

            return "\n".join(response_lines)

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return f"❌ **Error retrieving statistics**: {str(e)}"

    # =========================================================================
    # Private Formatting Methods
    # =========================================================================

    def _format_recall_results(
        self,
        context: ContextResult,
        query: str,
        verbose: bool
    ) -> str:
        """
        Format recall results for Matrix.

        Args:
            context: Context result from memory
            query: Original query
            verbose: Include full details

        Returns:
            Formatted message
        """
        response_lines = [
            f"🔍 **Search results for**: {query}",
            f"Found {len(context.memories)} relevant memory/memories (confidence: {context.confidence:.0%})",
            ""
        ]

        # Show results
        for i, memory in enumerate(context.memories, 1):
            time_ago = self._format_time_ago(memory.timestamp)

            # Basic format
            response_lines.extend([
                f"**{i}. From discussion on {memory.timestamp.strftime('%Y-%m-%d')}** ({time_ago}):",
                f"{memory.message}",
                ""
            ])

            # Verbose mode: show metadata
            if verbose:
                if memory.entities:
                    response_lines.append(f"  *Entities*: {', '.join(memory.entities[:5])}")
                if memory.topics:
                    response_lines.append(f"  *Topics*: {', '.join(memory.topics[:5])}")
                response_lines.append(f"  *User*: {memory.user_id}")
                response_lines.append("")

        # Footer with related topics
        if context.topics_covered:
            topics_str = ", ".join(list(context.topics_covered)[:5])
            response_lines.extend([
                f"**Related topics**: {topics_str}",
                ""
            ])

        # Time range
        start, end = context.time_range
        if start != end:
            response_lines.append(
                f"📅 **Time range**: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            )

        # Related users
        if context.related_users and len(context.related_users) > 1:
            users_str = ", ".join(list(context.related_users)[:5])
            response_lines.append(f"👥 **Participants**: {users_str}")

        return "\n".join(response_lines)

    def _format_no_results(self, query: str) -> str:
        """Format no results message"""
        return "\n".join([
            f"🔍 **No memories found for**: {query}",
            "",
            "💡 **Tips**:",
            "  • Try broader search terms",
            "  • Check if the topic has been discussed",
            "  • Use `@proto related <topic>` to find related discussions",
            "  • Use `@proto memories` to see available topics"
        ])

    def _format_time_ago(self, timestamp: datetime) -> str:
        """
        Format timestamp as human-readable time ago.

        Args:
            timestamp: Datetime to format

        Returns:
            Human-readable string (e.g., "2 hours ago")
        """
        now = datetime.now()

        # Handle timezone-aware vs naive
        if timestamp.tzinfo is not None and now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        elif timestamp.tzinfo is None and now.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=now.tzinfo)

        delta = now - timestamp

        if delta < timedelta(minutes=1):
            return "just now"
        elif delta < timedelta(hours=1):
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif delta < timedelta(days=1):
            hours = int(delta.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta < timedelta(days=7):
            days = delta.days
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif delta < timedelta(days=30):
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        else:
            months = delta.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"

    async def close(self):
        """Cleanup resources"""
        if self.memory:
            await self.memory.close()


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing Memory Command Handler...")

        handler = MemoryCommandHandler()

        try:
            # Initialize
            print("\n1. Initializing...")
            await handler.initialize()
            print("   ✓ Initialized")

            # Test remember
            print("\n2. Testing @proto remember...")
            response = await handler.handle_remember(
                fact="We decided to use PostgreSQL for authentication service",
                user_id="@alice:matrix.org",
                room_id="!test:matrix.org",
                tags=["decision", "database"]
            )
            print(response)
            print("   ✓ Remember command handled")

            # Store more context
            await handler.memory.store_conversation(
                user_id="@bob:matrix.org",
                message="The PostgreSQL database will use JWT tokens",
                room_id="!test:matrix.org"
            )

            await handler.memory.store_conversation(
                user_id="@carol:matrix.org",
                message="JWT access tokens expire after 15 minutes",
                room_id="!test:matrix.org"
            )

            # Test recall
            print("\n3. Testing @proto recall...")
            response = await handler.handle_recall(
                query="What database are we using?",
                room_id="!test:matrix.org",
                verbose=True
            )
            print(response)
            print("   ✓ Recall command handled")

            # Test related
            print("\n4. Testing @proto related...")
            response = await handler.handle_related(
                topic="authentication",
                room_id="!test:matrix.org"
            )
            print(response)
            print("   ✓ Related command handled")

            # Test memories
            print("\n5. Testing @proto memories...")
            response = await handler.handle_memories(
                room_id="!test:matrix.org"
            )
            print(response)
            print("   ✓ Memories command handled")

            print("\n✅ All Memory Command Handler tests passed!")

        except ImportError as e:
            print(f"\n⚠️  SKIPPED: HoloLoom not available ({e})")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await handler.close()

    asyncio.run(test())
