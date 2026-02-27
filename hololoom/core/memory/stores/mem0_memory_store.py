"""
Mem0 Memory Store
=================
MemoryStore implementation using Mem0 AI for intelligent entity extraction.

Mem0 automatically:
- Extracts entities (hives, beekeepers, equipment)
- Identifies preferences and patterns
- Builds semantic understanding
- Links related memories

Perfect for natural language beekeeping notes!

Usage:
    store = Mem0MemoryStore(
        api_key="your-mem0-api-key",  # or use Ollama locally
        use_ollama=True
    )

    # Store natural text - Mem0 extracts structure
    await store.store(Memory(
        text="Hive Jodi is doing great, Dennis's genetics are amazing"
    ))

    # Mem0 automatically knows:
    # - Hive: "Jodi"
    # - Source: "Dennis"
    # - Sentiment: positive
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

import sys
from pathlib import Path

# Import protocol types
protocol_path = Path(__file__).parent.parent / 'protocol.py'
import importlib.util
spec = importlib.util.spec_from_file_location("protocol", protocol_path)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

Memory = protocol.Memory
MemoryQuery = protocol.MemoryQuery
RetrievalResult = protocol.RetrievalResult
Strategy = protocol.Strategy


class Mem0MemoryStore:
    """
    Memory store backed by Mem0 AI.

    Mem0 provides:
    - Automatic entity extraction
    - Semantic search
    - Relationship inference
    - Preference learning

    This is the "smart" layer that understands:
    - "Hive Jodi" = entity
    - "Dennis's genetics" = source attribution
    - "doing great" = positive status

    Configuration:
        # Option 1: Mem0 Cloud (requires API key)
        store = Mem0MemoryStore(api_key="...")

        # Option 2: Local Ollama (free, private)
        store = Mem0MemoryStore(
            use_ollama=True,
            ollama_base_url="http://localhost:11434"
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_ollama: bool = True,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b"
    ):
        """
        Initialize Mem0 memory store.

        Args:
            api_key: Mem0 API key (if using cloud)
            use_ollama: Use local Ollama instead of Mem0 cloud
            ollama_base_url: Ollama server URL
            ollama_model: Ollama model to use
        """
        if not MEM0_AVAILABLE:
            raise ImportError("mem0ai package required. Install: pip install mem0ai")

        self.use_ollama = use_ollama

        if use_ollama:
            # Configure for local Ollama
            from mem0 import Memory as Mem0Memory

            config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": ollama_model,
                        "base_url": ollama_base_url,
                        "ollama_base_url": ollama_base_url
                    }
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text:latest",
                        "ollama_base_url": ollama_base_url
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "hololoom_beekeeping",
                        "path": "./mem0_qdrant_data"
                    }
                }
            }

            self.client = Mem0Memory(config)
            print("✓ Initialized Mem0 with local Ollama")

        else:
            # Use Mem0 cloud
            if not api_key:
                raise ValueError("api_key required for Mem0 cloud")

            self.client = MemoryClient(api_key=api_key)
            print("✓ Initialized Mem0 cloud client")

    def _generate_id(self, text: str) -> str:
        """Generate deterministic ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    async def store(self, memory: Memory) -> str:
        """
        Store a memory with Mem0.

        Mem0 will:
        1. Extract entities and relationships
        2. Identify key facts
        3. Link to existing knowledge
        4. Store semantically

        Args:
            memory: Memory object to store

        Returns:
            memory_id
        """
        # Generate ID if not set
        if not memory.id:
            memory.id = self._generate_id(memory.text)

        user_id = memory.metadata.get('user_id', 'default')

        # Prepare context for Mem0
        messages = [
            {
                "role": "user",
                "content": memory.text
            }
        ]

        # Add Mem0 memory with metadata
        result = self.client.add(
            messages=messages,
            user_id=user_id,
            metadata={
                **memory.context,
                **memory.metadata,
                'memory_id': memory.id,
                'timestamp': memory.timestamp.isoformat()
            }
        )

        print(f"  Mem0 extracted: {len(result.get('results', []))} facts")

        return memory.id

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using Mem0's semantic search.

        Mem0 understands:
        - "What hives are strong?" → finds positive status mentions
        - "Dennis's bees" → finds genetics from Dennis
        - "winter prep" → finds seasonal preparation notes

        Args:
            query: Memory query
            strategy: Retrieval strategy (Mem0 uses semantic by default)

        Returns:
            RetrievalResult with scored memories
        """
        # Search Mem0 memory
        results = self.client.search(
            query=query.text,
            user_id=query.user_id,
            limit=query.limit
        )

        memories = []
        scores = []

        for item in results.get('results', []):
            # Reconstruct Memory object
            mem = Memory(
                id=item.get('metadata', {}).get('memory_id', self._generate_id(item['memory'])),
                text=item['memory'],
                timestamp=datetime.fromisoformat(
                    item.get('metadata', {}).get('timestamp', datetime.now().isoformat())
                ),
                context=item.get('metadata', {}),
                metadata=item.get('metadata', {})
            )

            memories.append(mem)
            scores.append(item.get('score', 0.5))

        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used="mem0_semantic",
            metadata={
                'total_memories': len(memories),
                'mem0_results': len(results.get('results', []))
            }
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from Mem0."""
        try:
            # Mem0 delete by memory ID
            self.client.delete(memory_id=memory_id)
            return True
        except Exception as e:
            print(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Mem0 connection."""
        try:
            # Get all memories for default user to verify connection
            results = self.client.get_all(user_id="default")

            return {
                'status': 'healthy',
                'backend': 'mem0_ollama' if self.use_ollama else 'mem0_cloud',
                'memory_count': len(results.get('results', []))
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'mem0',
                'error': str(e)
            }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=== Mem0 Memory Store Demo ===\n")

        # Initialize with Ollama (local, free)
        store = Mem0MemoryStore(use_ollama=True)

        print("\nStoring natural language beekeeping notes...\n")

        # Store memories - Mem0 will extract entities automatically
        memories = [
            "Hive Jodi is incredibly strong - 15 frames of brood. Dennis's genetics are proving to be amazing.",
            "Applied thymol treatment to all hives. Used conservative dosing of 20 units for smaller colonies.",
            "Need to order more Hillco feeders before spring. Current stock: 3, need 5 total.",
        ]

        for text in memories:
            mem = Memory(
                id="",
                text=text,
                timestamp=datetime.now(),
                context={},
                metadata={'user_id': 'blake'}
            )

            memory_id = await store.store(mem)
            print(f"✓ Stored: {text[:60]}...")

        print("\n" + "="*60)
        print("Querying with semantic search...")
        print("="*60)

        # Query 1: Find strong hives
        print("\nQuery: 'Which hives are strong?'")
        query = MemoryQuery(text="strong hives", user_id="blake", limit=3)
        result = await store.retrieve(query)

        for mem, score in zip(result.memories, result.scores):
            print(f"  [{score:.2f}] {mem.text[:60]}...")

        # Query 2: Find genetics info
        print("\nQuery: 'Tell me about Dennis genetics'")
        query = MemoryQuery(text="Dennis genetics", user_id="blake", limit=3)
        result = await store.retrieve(query)

        for mem, score in zip(result.memories, result.scores):
            print(f"  [{score:.2f}] {mem.text[:60]}...")

        print("\n✓ Demo complete!")

    asyncio.run(demo())
