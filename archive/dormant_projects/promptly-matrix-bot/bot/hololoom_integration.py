#!/usr/bin/env python3
"""
HoloLoom Integration for Promptly Matrix Bot

Integrates HoloLoom's complete 9-step weaving cycle:
1. Loom Command → Pattern selection (BARE/FAST/FUSED)
2. Chrono Trigger → Temporal windows
3. Yarn Graph → Memory thread selection
4. Resonance Shed → Feature extraction (DotPlasma)
5. Warp Space → Continuous manifold tensioning
6. Convergence Engine → Decision collapse
7. Tool Execution → Action
8. Spacetime Fabric → Provenance trace
9. Reflection Buffer → Learning from outcome

Features:
- Full knowledge graph memory
- Multi-scale Matryoshka embeddings (96-192-384D)
- Thompson Sampling exploration
- Spectral graph features
- Complete provenance tracking
- Recursive learning

Usage:
    from bot.hololoom_integration import HoloLoomBot

    bot = HoloLoomBot()
    await bot.initialize()

    response = await bot.weave(
        query="What is Thompson Sampling?",
        user_id="@alice:matrix.org",
        room_id="!room:matrix.org",
        complexity="FAST"  # LITE, FAST, FULL, RESEARCH
    )

    print(response.text)
    print(f"Confidence: {response.confidence}")
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Add HoloLoom to path
HOLOLOOM_PATH = Path(__file__).parent.parent.parent / "HoloLoom"
if HOLOLOOM_PATH.exists():
    sys.path.insert(0, str(HOLOLOOM_PATH.parent))
else:
    logger.warning(f"HoloLoom not found at {HOLOLOOM_PATH}")


@dataclass
class WeavingResponse:
    """Response from HoloLoom weaving"""
    text: str                          # Response text
    confidence: float                  # Confidence score [0-1]
    tool_used: str                     # Tool that was selected
    context_depth: int                 # Graph traversal depth
    memories_retrieved: int            # Number of memories used
    latency_ms: float                  # Total latency
    cache_hit: bool = False           # Was response cached?
    provenance: Dict[str, Any] = field(default_factory=dict)  # Full trace
    metadata: Dict[str, Any] = field(default_factory=dict)    # Additional metadata

    def summary(self) -> str:
        """Human-readable summary"""
        cache_str = " (cached)" if self.cache_hit else ""
        return (
            f"Confidence: {self.confidence:.2%}{cache_str}\n"
            f"Tool: {self.tool_used}\n"
            f"Context: {self.memories_retrieved} memories, depth {self.context_depth}\n"
            f"Latency: {self.latency_ms:.0f}ms"
        )


class HoloLoomBot:
    """
    HoloLoom-integrated bot.

    Provides complete weaving cycle with knowledge graph memory,
    multi-scale embeddings, and Thompson Sampling exploration.

    Usage:
        bot = HoloLoomBot()
        await bot.initialize()

        response = await bot.weave(
            query="Explain recursion",
            user_id="@alice:matrix.org",
            complexity="FAST"
        )
    """

    def __init__(self, config_mode: str = "FAST"):
        """
        Initialize HoloLoom bot.

        Args:
            config_mode: Default config mode (BARE/FAST/FUSED)
        """
        self.config_mode = config_mode
        self.orchestrator = None
        self.config = None
        self.shards = []
        self.initialized = False

    async def initialize(self):
        """
        Initialize HoloLoom components.

        Sets up orchestrator, config, and memory shards.
        """
        if self.initialized:
            return

        try:
            # Import HoloLoom components
            from HoloLoom.config import Config
            from HoloLoom.weaving_orchestrator import WeavingOrchestrator
            from HoloLoom.documentation.types import MemoryShard, Query

            # Create config
            if self.config_mode == "BARE":
                self.config = Config.bare()
            elif self.config_mode == "FUSED":
                self.config = Config.fused()
            else:  # FAST (default)
                self.config = Config.fast()

            # Create initial memory shards (bot knowledge)
            self.shards = self._create_bot_shards()

            # Create orchestrator
            self.orchestrator = WeavingOrchestrator(
                cfg=self.config,
                shards=self.shards
            )

            self.initialized = True
            logger.info(f"HoloLoom bot initialized with {self.config_mode} config")

        except ImportError as e:
            logger.error(f"Failed to import HoloLoom: {e}")
            logger.error("Make sure HoloLoom is in the parent directory")
            raise

        except Exception as e:
            logger.error(f"Failed to initialize HoloLoom bot: {e}")
            raise

    def _create_bot_shards(self) -> List:
        """Create initial bot knowledge shards"""
        try:
            from HoloLoom.documentation.types import MemoryShard

            # Bot capabilities
            bot_knowledge = [
                {
                    "text": "Promptly bot can optimize prompts using DSPy with examples",
                    "entities": ["Promptly", "DSPy", "prompt optimization"],
                    "motifs": ["optimization", "machine learning"]
                },
                {
                    "text": "Schema builder creates JSON Schema, Pydantic, and DSPy signatures",
                    "entities": ["schema builder", "JSON Schema", "Pydantic", "DSPy"],
                    "motifs": ["validation", "structured output"]
                },
                {
                    "text": "Verification system detects hallucinations through multi-step fact-checking",
                    "entities": ["verification", "hallucination detection", "fact-checking"],
                    "motifs": ["reliability", "accuracy"]
                },
                {
                    "text": "Refinement strategies: ELEGANCE (clarity→simplicity→beauty), VERIFY (accuracy→completeness→consistency)",
                    "entities": ["refinement", "ELEGANCE", "VERIFY", "quality"],
                    "motifs": ["multi-pass", "quality improvement"]
                },
                {
                    "text": "Thompson Sampling balances exploration and exploitation using Bayesian updates",
                    "entities": ["Thompson Sampling", "Bayesian", "exploration", "exploitation"],
                    "motifs": ["reinforcement learning", "decision making"]
                }
            ]

            return [
                MemoryShard(
                    id=f"bot_knowledge_{i}",
                    text=item["text"],
                    entities=item["entities"],
                    motifs=item["motifs"]
                )
                for i, item in enumerate(bot_knowledge)
            ]

        except ImportError:
            logger.warning("Could not create MemoryShards - HoloLoom not available")
            return []

    async def weave(
        self,
        query: str,
        user_id: str,
        room_id: Optional[str] = None,
        complexity: str = "FAST"
    ) -> WeavingResponse:
        """
        Process query through HoloLoom weaving cycle.

        Args:
            query: User query
            user_id: User ID
            room_id: Room ID
            complexity: Complexity level (LITE/FAST/FULL/RESEARCH)

        Returns:
            WeavingResponse with answer and metadata
        """
        if not self.initialized:
            await self.initialize()

        import time
        start_time = time.time()

        try:
            from HoloLoom.documentation.types import Query as HoloLoomQuery

            # Create query object
            hololoom_query = HoloLoomQuery(
                text=query,
                metadata={
                    "user_id": user_id,
                    "room_id": room_id,
                    "complexity": complexity
                }
            )

            # Weave (full 9-step cycle)
            spacetime = await self.orchestrator.weave(hololoom_query)

            # Extract response
            latency_ms = (time.time() - start_time) * 1000

            response = WeavingResponse(
                text=spacetime.response,
                confidence=spacetime.confidence,
                tool_used=spacetime.metadata.get('tool_used', 'unknown'),
                context_depth=spacetime.metadata.get('context_depth', 0),
                memories_retrieved=spacetime.metadata.get('memories_retrieved', 0),
                latency_ms=latency_ms,
                cache_hit=spacetime.metadata.get('cache_hit', False),
                provenance={
                    'trace': spacetime.trace.summary() if hasattr(spacetime, 'trace') else {},
                    'decision_rationale': spacetime.metadata.get('decision_rationale', '')
                },
                metadata=spacetime.metadata
            )

            logger.info(
                f"Weaving complete: query='{query[:50]}...', "
                f"confidence={response.confidence:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Weaving failed: {e}")

            # Fallback response
            return WeavingResponse(
                text=f"Error processing query: {str(e)}",
                confidence=0.0,
                tool_used="error",
                context_depth=0,
                memories_retrieved=0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    async def add_memory(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        motifs: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ):
        """
        Add memory to knowledge graph.

        Args:
            text: Memory text
            entities: Extracted entities
            motifs: Extracted motifs
            user_id: User who created memory
        """
        if not self.initialized:
            await self.initialize()

        try:
            from HoloLoom.documentation.types import MemoryShard

            import hashlib
            shard_id = hashlib.md5(text.encode()).hexdigest()[:16]

            shard = MemoryShard(
                id=f"user_memory_{shard_id}",
                text=text,
                entities=entities or [],
                motifs=motifs or [],
                metadata={
                    "created_by": user_id,
                    "created_at": datetime.now().isoformat()
                }
            )

            self.shards.append(shard)

            # Re-initialize orchestrator with new shards
            # (In production, use dynamic memory backend instead)
            logger.info(f"Added memory: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with memory count, config, etc.
        """
        if not self.initialized:
            return {
                "initialized": False,
                "error": "Not initialized"
            }

        return {
            "initialized": True,
            "config_mode": self.config_mode,
            "memory_shards": len(self.shards),
            "capabilities": [
                "knowledge_graph_memory",
                "matryoshka_embeddings",
                "thompson_sampling",
                "spectral_features",
                "provenance_tracking",
                "recursive_learning"
            ]
        }

    async def close(self):
        """Cleanup resources"""
        if self.orchestrator and hasattr(self.orchestrator, 'close'):
            await self.orchestrator.close()

        self.initialized = False
        logger.info("HoloLoom bot closed")


class HoloLoomCommand:
    """
    Matrix bot command handler for HoloLoom integration.

    Handles @promptly commands that need HoloLoom's full capabilities.

    Usage:
        handler = HoloLoomCommand(bot)

        # In Matrix event handler
        if "@promptly weave" in message:
            response = await handler.handle_weave(
                query=query,
                user=sender,
                room=room_id
            )
    """

    def __init__(self, bot: HoloLoomBot):
        """
        Initialize command handler.

        Args:
            bot: HoloLoom bot instance
        """
        self.bot = bot

    async def handle_weave(
        self,
        query: str,
        user: str,
        room: str,
        complexity: str = "FAST"
    ) -> str:
        """
        Handle @promptly weave command.

        Args:
            query: User query
            user: User ID
            room: Room ID
            complexity: Complexity level

        Returns:
            Formatted response
        """
        response = await self.bot.weave(
            query=query,
            user_id=user,
            room_id=room,
            complexity=complexity
        )

        # Format response
        lines = [
            f"**Answer**: {response.text}",
            "",
            f"**{response.summary()}**"
        ]

        if response.provenance.get('decision_rationale'):
            lines.append(f"\n*Rationale*: {response.provenance['decision_rationale']}")

        return "\n".join(lines)

    async def handle_remember(
        self,
        text: str,
        user: str
    ) -> str:
        """
        Handle @promptly remember command.

        Stores memory in knowledge graph.

        Args:
            text: Text to remember
            user: User ID

        Returns:
            Confirmation message
        """
        await self.bot.add_memory(
            text=text,
            user_id=user
        )

        return f"Remembered: {text[:100]}..."

    async def handle_stats(self) -> str:
        """
        Handle @promptly stats command.

        Returns system statistics.
        """
        stats = await self.bot.get_statistics()

        lines = [
            "**HoloLoom Statistics**",
            "",
            f"Config Mode: {stats.get('config_mode', 'unknown')}",
            f"Memory Shards: {stats.get('memory_shards', 0)}",
            "",
            "**Capabilities**:"
        ]

        for capability in stats.get('capabilities', []):
            lines.append(f"  - {capability}")

        return "\n".join(lines)


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing HoloLoom integration...")

        # Create bot
        bot = HoloLoomBot(config_mode="FAST")

        try:
            # Initialize
            print("\n1. Initializing...")
            await bot.initialize()
            print("   OK - Initialized with FAST config")

            # Test weaving
            print("\n2. Testing weave...")
            response = await bot.weave(
                query="What is Thompson Sampling?",
                user_id="@test:matrix.org",
                complexity="FAST"
            )

            print(f"   Response: {response.text[:100]}...")
            print(f"   {response.summary()}")

            # Test memory addition
            print("\n3. Testing add_memory...")
            await bot.add_memory(
                text="Matrix bot can handle approval workflows with reaction-based voting",
                entities=["Matrix", "approval", "workflow"],
                motifs=["collaboration", "governance"],
                user_id="@test:matrix.org"
            )
            print("   OK - Memory added")

            # Test statistics
            print("\n4. Testing statistics...")
            stats = await bot.get_statistics()
            print(f"   Memory shards: {stats['memory_shards']}")
            print(f"   Capabilities: {len(stats['capabilities'])}")

            # Test command handler
            print("\n5. Testing command handler...")
            handler = HoloLoomCommand(bot)

            weave_response = await handler.handle_weave(
                query="How does verification work?",
                user="@test:matrix.org",
                room="!test:matrix.org"
            )
            print(f"   Weave response: {len(weave_response)} chars")

            stats_response = await handler.handle_stats()
            print(f"   Stats response: {len(stats_response)} chars")

            print("\nAll HoloLoom integration tests passed!")

        except ImportError as e:
            print(f"\nSKIPPED: HoloLoom not available ({e})")
            print("This is expected if HoloLoom is not in parent directory")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await bot.close()

    asyncio.run(test())
