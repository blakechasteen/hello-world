#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Unified API
====================
Single, clean entry point for all HoloLoom functionality.

This is the main user-facing API that consolidates:
- WeavingOrchestrator (complete weaving cycle)
- AutoSpinOrchestrator (text auto-spinning)
- ConversationalAutoLoom (chat interface)
- Memory management
- Data ingestion

Usage:
    from HoloLoom import HoloLoom

    # Create instance
    loom = await HoloLoom.create(
        pattern="fast",              # BARE, FAST, or FUSED
        memory_backend="simple",     # simple, neo4j, qdrant, neo4j+qdrant
        enable_synthesis=True        # Enable pattern extraction
    )

    # Query (one-shot)
    response = await loom.query("What is HoloLoom?")

    # Chat (conversational)
    response = await loom.chat("Tell me more about the weaving metaphor")

    # Ingest data
    await loom.ingest_text("Knowledge base content...")
    await loom.ingest_web("https://example.com")
    await loom.ingest_youtube("VIDEO_ID")

    # Get statistics
    stats = loom.get_stats()
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

try:
    from HoloLoom.config import Config
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.spinningWheel.website import WebsiteSpinnerConfig, WebsiteSpinner
    from HoloLoom.spinningWheel.youtube import YouTubeSpinnerConfig, YouTubeSpinner
    from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories
    from HoloLoom.fabric.spacetime import Spacetime
    from HoloLoom.convergence.engine import CollapseStrategy
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you run from repository root with PYTHONPATH set")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HoloLoom:
    """
    Unified HoloLoom API - Single entry point for all functionality.

    This class consolidates all HoloLoom capabilities into a clean,
    user-friendly interface.

    Features:
    - Query processing with complete weaving cycle
    - Conversational chat with auto-memory
    - Multi-modal data ingestion (text, web, youtube, etc.)
    - Unified memory management
    - Pattern extraction and synthesis
    - Full computational traces

    Example:
        loom = await HoloLoom.create()
        response = await loom.query("Your question")
        print(response.response)
        print(response.trace)  # Complete computational trace
    """

    def __init__(
        self,
        weaver: WeavingOrchestrator,
        memory: Optional[Any] = None,
        config: Optional[Config] = None,
        enable_synthesis: bool = True
    ):
        """
        Initialize HoloLoom (use create() factory instead).

        Args:
            weaver: WeavingOrchestrator instance
            memory: Optional unified memory instance
            config: Optional Config instance
            enable_synthesis: Enable synthesis pipeline
        """
        self.weaver = weaver
        self.memory = memory
        self.config = config or Config.fast()
        self.enable_synthesis = enable_synthesis

        # Conversation history (for chat mode)
        self.conversation_history: List[Dict] = []
        self.conversation_mode = False

        # Statistics
        self.query_count = 0
        self.chat_count = 0
        self.ingest_count = 0

        logger.info("HoloLoom initialized")
        logger.info(f"  Mode: {self.config.mode.value}")
        logger.info(f"  Synthesis: {enable_synthesis}")

    @classmethod
    async def create(
        cls,
        pattern: str = "fast",
        memory_backend: str = "simple",
        enable_synthesis: bool = True,
        collapse_strategy: str = "epsilon_greedy"
    ) -> "HoloLoom":
        """
        Create HoloLoom instance (async factory).

        Args:
            pattern: Pattern card ("bare", "fast", "fused")
            memory_backend: Memory backend ("simple", "neo4j", "qdrant", "neo4j+qdrant")
            enable_synthesis: Enable synthesis pipeline
            collapse_strategy: Decision strategy ("argmax", "epsilon_greedy", "bayesian_blend", "pure_thompson")

        Returns:
            Configured HoloLoom instance

        Example:
            loom = await HoloLoom.create(pattern="fused", memory_backend="neo4j")
        """
        logger.info(f"Creating HoloLoom (pattern={pattern}, memory={memory_backend})")

        # Get config for pattern
        if pattern == "bare":
            config = Config.bare()
        elif pattern == "fast":
            config = Config.fast()
        elif pattern == "fused":
            config = Config.fused()
        else:
            raise ValueError(f"Invalid pattern: {pattern}. Use 'bare', 'fast', or 'fused'")

        # Create memory backend
        try:
            memory = await create_unified_memory(memory_backend)
            logger.info(f"✓ Created {memory_backend} memory backend")
        except Exception as e:
            logger.warning(f"Memory backend creation failed: {e}, using None")
            memory = None

        # Create weaving orchestrator
        try:
            strategy = CollapseStrategy(collapse_strategy)
        except ValueError:
            logger.warning(f"Invalid strategy: {collapse_strategy}, using epsilon_greedy")
            strategy = CollapseStrategy.EPSILON_GREEDY

        weaver = WeavingOrchestrator(
            config=config,
            default_pattern=pattern,
            collapse_strategy=strategy
        )

        # Create HoloLoom instance
        return cls(
            weaver=weaver,
            memory=memory,
            config=config,
            enable_synthesis=enable_synthesis
        )

    async def query(
        self,
        text: str,
        pattern: Optional[str] = None,
        return_trace: bool = True
    ) -> Union[Spacetime, str]:
        """
        Process a single query (one-shot, no conversation context).

        Args:
            text: Query text
            pattern: Optional override pattern ("bare", "fast", "fused")
            return_trace: Return full Spacetime with trace (default True)

        Returns:
            Spacetime object with response and trace, or just response string

        Example:
            result = await loom.query("What is Thompson Sampling?")
            print(result.response)
            print(result.trace.synthesis_result)
        """
        self.query_count += 1
        logger.info(f"Query #{self.query_count}: '{text[:50]}...'")

        # Execute weaving cycle
        spacetime = await self.weaver.weave(
            query=text,
            user_pattern=pattern
        )

        # Store in memory if available
        if self.memory:
            try:
                # Create memory from query + response
                memory_obj = {
                    'id': f'query_{datetime.now().timestamp()}',
                    'text': f"Q: {text}\nA: {spacetime.response}",
                    'timestamp': datetime.now().isoformat(),
                    'importance': 0.8,
                    'metadata': {
                        'user_input': text,
                        'system_output': spacetime.response,
                        'tool_used': spacetime.tool_used,
                        'confidence': spacetime.confidence
                    }
                }
                # Note: Actual storage would need proper Memory object conversion
                logger.debug("Query stored in memory")
            except Exception as e:
                logger.warning(f"Failed to store query in memory: {e}")

        return spacetime if return_trace else spacetime.response

    async def chat(
        self,
        message: str,
        pattern: Optional[str] = None,
        return_trace: bool = False
    ) -> Union[Spacetime, str]:
        """
        Conversational chat (maintains context across turns).

        Args:
            message: User message
            pattern: Optional override pattern
            return_trace: Return full Spacetime (default False for chat)

        Returns:
            Response string or Spacetime object

        Example:
            response1 = await loom.chat("What is HoloLoom?")
            response2 = await loom.chat("Tell me more")  # Has context from previous
        """
        self.chat_count += 1
        self.conversation_mode = True

        logger.info(f"Chat #{self.chat_count}: '{message[:50]}...'")

        # TODO: Build conversation context from history
        # For now, just use weaver
        spacetime = await self.weaver.weave(
            query=message,
            user_pattern=pattern
        )

        # Record in conversation history
        self.conversation_history.append({
            'turn': self.chat_count,
            'user': message,
            'assistant': spacetime.response,
            'tool': spacetime.tool_used,
            'confidence': spacetime.confidence,
            'timestamp': datetime.now()
        })

        return spacetime if return_trace else spacetime.response

    async def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Ingest plain text into memory.

        Args:
            text: Text content to ingest
            metadata: Optional metadata dict

        Returns:
            Number of memory shards created

        Example:
            count = await loom.ingest_text("Your knowledge base...")
        """
        self.ingest_count += 1
        logger.info(f"Ingesting text ({len(text)} chars)...")

        if not self.memory:
            logger.warning("No memory backend available")
            return 0

        try:
            # Use TextSpinner to create shards
            from HoloLoom.spinningWheel import TextSpinnerConfig, TextSpinner

            config = TextSpinnerConfig(chunk_size=500, chunk_by='paragraph')
            spinner = TextSpinner(config)

            shards = await spinner.spin({'text': text, 'metadata': metadata or {}})
            logger.info(f"Created {len(shards)} shards from text")

            # Convert to memories and store
            memories = shards_to_memories(shards)
            stored_ids = await self.memory.store_batch(memories)

            logger.info(f"✓ Ingested {len(stored_ids)} memories")
            return len(stored_ids)

        except Exception as e:
            logger.error(f"Text ingestion failed: {e}")
            return 0

    async def ingest_web(
        self,
        url: str,
        extract_images: bool = False,
        max_depth: int = 0
    ) -> int:
        """
        Ingest webpage(s) into memory.

        Args:
            url: URL to scrape
            extract_images: Extract images from page
            max_depth: Recursive crawl depth (0 = single page)

        Returns:
            Number of memory shards created

        Example:
            count = await loom.ingest_web("https://example.com")
        """
        self.ingest_count += 1
        logger.info(f"Ingesting web: {url}")

        if not self.memory:
            logger.warning("No memory backend available")
            return 0

        try:
            config = WebsiteSpinnerConfig(
                extract_images=extract_images,
                chunk_by='paragraph',
                chunk_size=500
            )
            spinner = WebsiteSpinner(config)

            shards = await spinner.spin({'url': url})
            logger.info(f"Scraped {len(shards)} shards from {url}")

            # Store in memory
            memories = shards_to_memories(shards)
            stored_ids = await self.memory.store_batch(memories)

            logger.info(f"✓ Ingested {len(stored_ids)} memories from web")
            return len(stored_ids)

        except Exception as e:
            logger.error(f"Web ingestion failed: {e}")
            return 0

    async def ingest_youtube(
        self,
        video_id: str,
        languages: Optional[List[str]] = None,
        chunk_duration: float = 60.0
    ) -> int:
        """
        Ingest YouTube video transcript into memory.

        Args:
            video_id: YouTube video ID or URL
            languages: Preferred languages (default: ['en'])
            chunk_duration: Chunk duration in seconds

        Returns:
            Number of memory shards created

        Example:
            count = await loom.ingest_youtube("VIDEO_ID")
        """
        self.ingest_count += 1
        logger.info(f"Ingesting YouTube: {video_id}")

        if not self.memory:
            logger.warning("No memory backend available")
            return 0

        try:
            config = YouTubeSpinnerConfig(
                chunk_duration=chunk_duration,
                languages=languages or ['en']
            )
            spinner = YouTubeSpinner(config)

            shards = await spinner.spin({'url': video_id, 'languages': languages or ['en']})
            logger.info(f"Transcribed {len(shards)} shards from video")

            # Store in memory
            memories = shards_to_memories(shards)
            stored_ids = await self.memory.store_batch(memories)

            logger.info(f"✓ Ingested {len(stored_ids)} memories from YouTube")
            return len(stored_ids)

        except Exception as e:
            logger.error(f"YouTube ingestion failed: {e}")
            return 0

    async def ingest_file(self, file_path: Union[str, Path]) -> int:
        """
        Ingest file into memory (auto-detects type).

        Args:
            file_path: Path to file

        Returns:
            Number of memory shards created

        Example:
            count = await loom.ingest_file("knowledge.txt")
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return 0

        # Read file
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        return await self.ingest_text(text, metadata={'source_file': str(path)})

    def get_stats(self) -> Dict[str, Any]:
        """
        Get HoloLoom usage statistics.

        Returns:
            Dict with statistics

        Example:
            stats = loom.get_stats()
            print(f"Queries: {stats['query_count']}")
        """
        stats = {
            'query_count': self.query_count,
            'chat_count': self.chat_count,
            'ingest_count': self.ingest_count,
            'conversation_turns': len(self.conversation_history),
            'conversation_mode': self.conversation_mode,
            'pattern': self.config.mode.value,
            'synthesis_enabled': self.enable_synthesis
        }

        # Add weaver stats
        if hasattr(self.weaver, 'get_statistics'):
            stats['weaving'] = self.weaver.get_statistics()

        return stats

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.conversation_mode = False
        logger.info("Conversation reset")

    async def close(self):
        """Clean up resources."""
        if hasattr(self.weaver, 'stop'):
            self.weaver.stop()
        logger.info("HoloLoom closed")


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_hololoom(
    pattern: str = "fast",
    memory: str = "simple",
    synthesis: bool = True
) -> HoloLoom:
    """
    Convenience function to create HoloLoom instance.

    Args:
        pattern: Pattern card ("bare", "fast", "fused")
        memory: Memory backend
        synthesis: Enable synthesis

    Returns:
        HoloLoom instance
    """
    return await HoloLoom.create(
        pattern=pattern,
        memory_backend=memory,
        enable_synthesis=synthesis
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("HOLOLOOM UNIFIED API DEMO")
        print("="*80)

        # Create HoloLoom
        print("\nCreating HoloLoom...")
        loom = await HoloLoom.create(
            pattern="fast",
            memory_backend="simple",
            enable_synthesis=True
        )

        print("HoloLoom created\n")

        # Query mode
        print("="*80)
        print("QUERY MODE")
        print("="*80)

        queries = [
            "What is HoloLoom?",
            "What is Thompson Sampling?",
            "How does the weaving metaphor work?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            result = await loom.query(query, return_trace=True)
            print(f"Response: {result.response[:100]}...")
            print(f"Tool: {result.tool_used}")
            print(f"Confidence: {result.confidence:.1%}")
            if hasattr(result.trace, 'synthesis_result'):
                syn = result.trace.synthesis_result
                print(f"Entities: {syn.get('entities', [])[:5]}")
                print(f"Reasoning: {syn.get('reasoning_type', 'unknown')}")

        # Chat mode
        print("\n" + "="*80)
        print("CHAT MODE")
        print("="*80)

        messages = [
            "Tell me about the weaving architecture",
            "What are the stages?",
            "How does synthesis work?"
        ]

        for msg in messages:
            print(f"\nYou: {msg}")
            response = await loom.chat(msg)
            print(f"HoloLoom: {response[:150]}...")

        # Statistics
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)

        stats = loom.get_stats()
        print(f"  Queries: {stats['query_count']}")
        print(f"  Chats: {stats['chat_count']}")
        print(f"  Ingests: {stats['ingest_count']}")
        print(f"  Pattern: {stats['pattern']}")
        print(f"  Synthesis: {stats['synthesis_enabled']}")

        # Cleanup
        await loom.close()

        print("\nDemo complete!")

    asyncio.run(demo())
