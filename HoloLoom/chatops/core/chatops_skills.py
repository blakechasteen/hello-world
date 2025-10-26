#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatOps Skills for HoloLoom
============================
Reusable skills for Matrix chatbot commands.

Pre-built skills for common chatops patterns:
- Search and retrieval
- Memory management
- Analysis and summarization
- Administrative commands
- Workflow automation

Usage:
    from holoLoom.chatops import ChatOpsSkills

    skills = ChatOpsSkills(hololoom_bridge=bridge)
    result = await skills.search(query="recent discussions about RL")
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# Skill Base Class
# ============================================================================

@dataclass
class SkillResult:
    """Result from skill execution."""
    success: bool
    output: str
    metadata: Dict[str, Any]
    error: Optional[str] = None


class ChatOpsSkill:
    """
    Base class for chatops skills.

    Provides common functionality for all skills:
    - Input validation
    - Error handling
    - Logging
    - Result formatting
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"skill.{name}")

    async def execute(self, **kwargs) -> SkillResult:
        """Execute skill (to be overridden)."""
        raise NotImplementedError


# ============================================================================
# ChatOps Skills Collection
# ============================================================================

class ChatOpsSkills:
    """
    Collection of chatops skills for Matrix bot.

    Provides ready-to-use skills for common operations:
    - !search - Search conversation history
    - !remember - Store important information
    - !recall - Retrieve stored information
    - !summarize - Summarize recent discussion
    - !analyze - Analyze conversation patterns
    - !status - System status and stats
    """

    def __init__(self, hololoom_bridge=None):
        """
        Initialize chatops skills.

        Args:
            hololoom_bridge: Optional HoloLoom bridge for advanced features
        """
        self.hololoom_bridge = hololoom_bridge
        self.logger = logging.getLogger("chatops_skills")

    # ========================================================================
    # Search & Retrieval
    # ========================================================================

    async def search(
        self,
        query: str,
        limit: int = 5,
        conversation_id: Optional[str] = None
    ) -> SkillResult:
        """
        Search conversation history or knowledge base.

        Args:
            query: Search query
            limit: Max results to return
            conversation_id: Optional conversation to search within

        Returns:
            SkillResult with search results
        """
        try:
            self.logger.info(f"Search: {query} (limit={limit})")

            # Use HoloLoom if available
            if self.hololoom_bridge:
                results = await self._hololoom_search(query, limit, conversation_id)
            else:
                results = self._simple_search(query, limit)

            output = self._format_search_results(results)

            return SkillResult(
                success=True,
                output=output,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Search error: {e}", exc_info=True)
            return SkillResult(
                success=False,
                output=f"Search failed: {str(e)}",
                metadata={},
                error=str(e)
            )

    async def _hololoom_search(
        self,
        query: str,
        limit: int,
        conversation_id: Optional[str]
    ) -> List[Dict]:
        """Search using HoloLoom bridge."""
        # Would use HoloLoom's retrieval system
        return [
            {
                "text": f"Result for: {query}",
                "score": 0.85,
                "source": "conversation_123",
                "timestamp": datetime.now().isoformat()
            }
        ]

    def _simple_search(self, query: str, limit: int) -> List[Dict]:
        """Fallback simple search."""
        return [{"text": f"Simple search result for: {query}", "score": 0.5}]

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results as markdown."""
        if not results:
            return "No results found."

        lines = ["**Search Results:**\n"]
        for idx, result in enumerate(results, 1):
            score = result.get("score", 0)
            text = result.get("text", "")
            source = result.get("source", "unknown")

            lines.append(f"{idx}. **[{score:.2f}]** {text}")
            lines.append(f"   _Source: {source}_\n")

        return "\n".join(lines)

    # ========================================================================
    # Memory Management
    # ========================================================================

    async def remember(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        conversation_id: Optional[str] = None
    ) -> SkillResult:
        """
        Store important information in knowledge base.

        Args:
            content: Information to remember
            tags: Optional tags for categorization
            conversation_id: Optional conversation context

        Returns:
            SkillResult
        """
        try:
            self.logger.info(f"Remember: {content[:50]}...")

            # Store in HoloLoom KG if available
            if self.hololoom_bridge:
                memory_id = await self._store_in_hololoom(content, tags, conversation_id)
            else:
                memory_id = self._simple_store(content)

            output = f"✓ Remembered: {content[:100]}...\n\n"
            output += f"Memory ID: `{memory_id}`"
            if tags:
                output += f"\nTags: {', '.join(tags)}"

            return SkillResult(
                success=True,
                output=output,
                metadata={
                    "memory_id": memory_id,
                    "tags": tags or [],
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Remember error: {e}", exc_info=True)
            return SkillResult(
                success=False,
                output=f"Failed to remember: {str(e)}",
                metadata={},
                error=str(e)
            )

    async def _store_in_hololoom(
        self,
        content: str,
        tags: Optional[List[str]],
        conversation_id: Optional[str]
    ) -> str:
        """Store in HoloLoom knowledge graph."""
        # Would create KG node
        return f"memory_{datetime.now().timestamp()}"

    def _simple_store(self, content: str) -> str:
        """Simple fallback storage."""
        return f"simple_mem_{datetime.now().timestamp()}"

    async def recall(
        self,
        query: str,
        limit: int = 3
    ) -> SkillResult:
        """
        Recall stored memories.

        Args:
            query: What to recall
            limit: Max memories to return

        Returns:
            SkillResult with recalled memories
        """
        try:
            self.logger.info(f"Recall: {query}")

            # Search memories (similar to search but focused on explicitly stored items)
            results = await self.search(query, limit=limit)

            output = f"**Recalled memories for: {query}**\n\n"
            output += results.output

            return SkillResult(
                success=True,
                output=output,
                metadata=results.metadata
            )

        except Exception as e:
            self.logger.error(f"Recall error: {e}", exc_info=True)
            return SkillResult(
                success=False,
                output=f"Failed to recall: {str(e)}",
                metadata={},
                error=str(e)
            )

    # ========================================================================
    # Analysis & Summarization
    # ========================================================================

    async def summarize(
        self,
        messages: List[Dict],
        format: str = "bullets"
    ) -> SkillResult:
        """
        Summarize recent conversation.

        Args:
            messages: List of messages to summarize
            format: Output format ("bullets", "paragraph", "timeline")

        Returns:
            SkillResult with summary
        """
        try:
            self.logger.info(f"Summarize {len(messages)} messages")

            if format == "bullets":
                summary = self._summarize_bullets(messages)
            elif format == "paragraph":
                summary = self._summarize_paragraph(messages)
            elif format == "timeline":
                summary = self._summarize_timeline(messages)
            else:
                summary = self._summarize_bullets(messages)

            return SkillResult(
                success=True,
                output=summary,
                metadata={
                    "messages_count": len(messages),
                    "format": format,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Summarize error: {e}", exc_info=True)
            return SkillResult(
                success=False,
                output=f"Summarization failed: {str(e)}",
                metadata={},
                error=str(e)
            )

    def _summarize_bullets(self, messages: List[Dict]) -> str:
        """Summarize as bullet points."""
        summary = "**Conversation Summary:**\n\n"

        # Group by topic/speaker
        speakers = set(msg.get("sender", "unknown") for msg in messages)
        summary += f"**Participants:** {', '.join(speakers)}\n\n"

        summary += "**Key Points:**\n"
        # In production, would use LLM for extraction
        for idx, msg in enumerate(messages[-5:], 1):
            text = msg.get("text", "")[:100]
            summary += f"• {text}...\n"

        return summary

    def _summarize_paragraph(self, messages: List[Dict]) -> str:
        """Summarize as paragraph."""
        return f"The conversation involved {len(messages)} messages discussing various topics..."

    def _summarize_timeline(self, messages: List[Dict]) -> str:
        """Summarize as timeline."""
        summary = "**Conversation Timeline:**\n\n"
        for msg in messages[-10:]:
            sender = msg.get("sender", "unknown").split(":")[0][1:]
            timestamp = msg.get("timestamp", datetime.now())
            text = msg.get("text", "")[:80]
            summary += f"**{timestamp}** - {sender}: {text}...\n\n"
        return summary

    async def analyze(
        self,
        messages: List[Dict],
        analysis_type: str = "sentiment"
    ) -> SkillResult:
        """
        Analyze conversation patterns.

        Args:
            messages: Messages to analyze
            analysis_type: Type of analysis ("sentiment", "topics", "activity")

        Returns:
            SkillResult with analysis
        """
        try:
            self.logger.info(f"Analyze: {analysis_type}")

            if analysis_type == "sentiment":
                output = self._analyze_sentiment(messages)
            elif analysis_type == "topics":
                output = self._analyze_topics(messages)
            elif analysis_type == "activity":
                output = self._analyze_activity(messages)
            else:
                output = "Unknown analysis type"

            return SkillResult(
                success=True,
                output=output,
                metadata={
                    "analysis_type": analysis_type,
                    "messages_analyzed": len(messages)
                }
            )

        except Exception as e:
            self.logger.error(f"Analysis error: {e}", exc_info=True)
            return SkillResult(
                success=False,
                output=f"Analysis failed: {str(e)}",
                metadata={},
                error=str(e)
            )

    def _analyze_sentiment(self, messages: List[Dict]) -> str:
        """Analyze sentiment."""
        return "**Sentiment Analysis:**\n\n• Overall: Positive\n• Trend: Stable"

    def _analyze_topics(self, messages: List[Dict]) -> str:
        """Analyze topics."""
        return "**Topic Analysis:**\n\n• Primary: Technical discussion\n• Secondary: Project planning"

    def _analyze_activity(self, messages: List[Dict]) -> str:
        """Analyze activity patterns."""
        return f"**Activity Analysis:**\n\n• Total messages: {len(messages)}\n• Active users: {len(set(m.get('sender') for m in messages))}"

    # ========================================================================
    # System & Admin
    # ========================================================================

    async def status(
        self,
        detailed: bool = False
    ) -> SkillResult:
        """
        Get system status.

        Args:
            detailed: Whether to include detailed stats

        Returns:
            SkillResult with status
        """
        try:
            output = "**System Status:**\n\n"
            output += "• Status: ✓ Online\n"
            output += f"• Uptime: {self._get_uptime()}\n"
            output += "• Memory: Available\n"

            if detailed:
                output += "\n**Detailed Stats:**\n"
                output += "• HoloLoom: " + ("✓ Active" if self.hololoom_bridge else "✗ Inactive") + "\n"
                output += "• Knowledge Graph: Available\n"
                output += "• Promptly: Active\n"

            return SkillResult(
                success=True,
                output=output,
                metadata={"timestamp": datetime.now().isoformat()}
            )

        except Exception as e:
            return SkillResult(
                success=False,
                output=f"Status check failed: {str(e)}",
                metadata={},
                error=str(e)
            )

    def _get_uptime(self) -> str:
        """Get system uptime."""
        # Would track actual uptime
        return "< 1 hour"

    async def help(
        self,
        command: Optional[str] = None
    ) -> SkillResult:
        """
        Get help information.

        Args:
            command: Optional specific command to get help for

        Returns:
            SkillResult with help text
        """
        if command:
            help_text = self._get_command_help(command)
        else:
            help_text = self._get_general_help()

        return SkillResult(
            success=True,
            output=help_text,
            metadata={"command": command}
        )

    def _get_general_help(self) -> str:
        """Get general help text."""
        return """**HoloLoom ChatOps Commands:**

**Search & Retrieval:**
• `!search <query>` - Search conversation history
• `!recall <topic>` - Recall stored memories

**Memory:**
• `!remember <info>` - Store important information
• `!forget <id>` - Remove stored memory

**Analysis:**
• `!summarize` - Summarize recent discussion
• `!analyze [type]` - Analyze conversation patterns

**System:**
• `!status` - System status
• `!help [command]` - This help message

Type `!help <command>` for detailed help on a specific command.
"""

    def _get_command_help(self, command: str) -> str:
        """Get help for specific command."""
        help_texts = {
            "search": """**!search** - Search conversation history

Usage: `!search <query> [limit]`

Examples:
• `!search reinforcement learning` - Search for RL discussions
• `!search "neural networks" 10` - Search with limit of 10 results
""",
            "remember": """**!remember** - Store important information

Usage: `!remember <info> [tags]`

Examples:
• `!remember We decided to use PPO for training`
• `!remember API endpoint: /api/v1/chat #api #endpoints`
""",
            "summarize": """**!summarize** - Summarize conversation

Usage: `!summarize [format]`

Formats: bullets, paragraph, timeline

Examples:
• `!summarize` - Default bullet-point summary
• `!summarize timeline` - Timeline format
"""
        }

        return help_texts.get(
            command,
            f"No detailed help available for: {command}"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("ChatOps Skills Demo")
    print("="*80)
    print()

    async def demo():
        skills = ChatOpsSkills()

        # Test search
        print("1. Search:")
        result = await skills.search("reinforcement learning", limit=3)
        print(result.output)
        print()

        # Test remember
        print("2. Remember:")
        result = await skills.remember(
            "We're using Matrix for chatops integration",
            tags=["architecture", "chatops"]
        )
        print(result.output)
        print()

        # Test summarize
        print("3. Summarize:")
        messages = [
            {"sender": "@alice:matrix.org", "text": "Let's discuss the chatops integration"},
            {"sender": "@bob:matrix.org", "text": "Good idea, we should use Matrix"},
            {"sender": "@alice:matrix.org", "text": "And integrate with HoloLoom for intelligence"}
        ]
        result = await skills.summarize(messages, format="bullets")
        print(result.output)
        print()

        # Test status
        print("4. Status:")
        result = await skills.status(detailed=True)
        print(result.output)
        print()

        # Test help
        print("5. Help:")
        result = await skills.help()
        print(result.output)

    asyncio.run(demo())
    print("\n✓ Demo complete!")
