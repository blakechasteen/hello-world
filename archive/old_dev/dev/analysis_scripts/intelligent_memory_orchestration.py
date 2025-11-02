#!/usr/bin/env python3
"""
Intelligent Multi-Backend Memory Orchestration
==============================================
Leverages the best features of each memory backend strategically.

Architecture:
- Neo4j: Thread-based graph relationships and temporal navigation
- Qdrant: Multi-scale vector similarity and semantic search  
- Mem0: LLM-powered intelligent extraction and user-specific memories
- InMemory: Fast caching and temporary storage

Strategy: Route different memory operations to the optimal backend.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent / "HoloLoom"))

from HoloLoom.memory.protocol import Memory, MemoryQuery, Strategy, UnifiedMemoryInterface
from HoloLoom.memory.stores.neo4j_store import Neo4jMemoryStore
from HoloLoom.memory.stores.qdrant_store import QdrantMemoryStore  
from HoloLoom.memory.stores.mem0_store import Mem0MemoryStore
from HoloLoom.memory.stores.in_memory_store import InMemoryStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelligentMemoryOrchestrator:
    """
    Smart orchestrator that routes operations to optimal backends.
    
    Backend Specializations:
    - Neo4j: Relationship discovery, thread weaving, temporal patterns
    - Qdrant: Semantic similarity, content-based retrieval  
    - Mem0: User context, intelligent summarization, personalization
    - InMemory: Fast caching, session state, temporary processing
    """
    
    def __init__(self):
        self.backends: Dict[str, Any] = {}
        self.cache_store = InMemoryStore()
        self.initialized = False
    
    async def initialize(self):
        """Initialize available backends with graceful degradation."""
        logger.info("🔧 Initializing multi-backend memory orchestrator...")
        
        # Try Neo4j (for graph relationships)
        try:
            neo4j = Neo4jMemoryStore(
                uri="bolt://localhost:7687",
                password="hololoom123",  # Use correct password
                database="neo4j"
            )
            health = await neo4j.health_check()
            if 'count' in health:  # Neo4j is healthy
                self.backends['neo4j'] = neo4j
                logger.info("✅ Neo4j: Graph relationships & thread weaving")
            else:
                logger.warning("⚠️ Neo4j unhealthy, skipping")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j unavailable: {e}")
        
        # Try Qdrant (for vector similarity)
        try:
            qdrant = QdrantMemoryStore(url="http://localhost:6333")
            health = await qdrant.health_check()
            if health.get('status') == 'healthy':
                self.backends['qdrant'] = qdrant
                logger.info("✅ Qdrant: Multi-scale semantic similarity")
            else:
                logger.warning("⚠️ Qdrant unhealthy, skipping")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant unavailable: {e}")
        
        # Try Mem0 (for intelligent extraction)
        try:
            mem0 = Mem0MemoryStore(user_id="orchestrator")
            health = await mem0.health_check()
            if health.get('status') == 'available':
                self.backends['mem0'] = mem0
                logger.info("✅ Mem0: Intelligent extraction & personalization")
            else:
                logger.warning("⚠️ Mem0 unhealthy, skipping") 
        except Exception as e:
            logger.warning(f"⚠️ Mem0 unavailable: {e}")
        
        # Always have cache
        self.backends['cache'] = self.cache_store
        logger.info("✅ InMemory: Fast caching & session state")
        
        self.initialized = True
        logger.info(f"🎯 Orchestrator ready with {len(self.backends)} backends")
    
    async def smart_store(self, memory: Memory, user_context: Optional[Dict] = None) -> Dict[str, str]:
        """
        Intelligently store memory across optimal backends.
        
        Strategy:
        1. Cache: Always store for fast access
        2. Neo4j: Store if rich context (relationships, threads)
        3. Qdrant: Store for semantic similarity search
        4. Mem0: Store for intelligent user-specific extraction
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # 1. Always cache for fast access
        cache_id = await self.backends['cache'].store(memory)
        results['cache'] = cache_id
        logger.info(f"📦 Cached: {cache_id}")
        
        # 2. Neo4j: Store if rich relational context
        if 'neo4j' in self.backends:
            has_rich_context = any([
                memory.context.get('place'),
                memory.context.get('people'),
                memory.context.get('time'),
                memory.context.get('topics')
            ])
            
            if has_rich_context:
                neo4j_id = await self.backends['neo4j'].store(memory)
                results['neo4j'] = neo4j_id
                logger.info(f"🕸️ Neo4j (rich context): {neo4j_id}")
        
        # 3. Qdrant: Store for semantic similarity
        if 'qdrant' in self.backends and len(memory.text) > 20:  # Skip very short text
            try:
                # Fix ID format for Qdrant (needs integer or UUID)
                import uuid
                qdrant_memory = Memory(
                    id=str(uuid.uuid4()),  # Generate UUID for Qdrant
                    text=memory.text,
                    timestamp=memory.timestamp,
                    context=memory.context,
                    metadata=memory.metadata
                )
                qdrant_id = await self.backends['qdrant'].store(qdrant_memory)
                results['qdrant'] = qdrant_id
                logger.info(f"🔍 Qdrant (semantic): {qdrant_id}")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant storage failed: {e}")
        
        # 4. Mem0: Store for intelligent extraction
        if 'mem0' in self.backends:
            try:
                mem0_id = await self.backends['mem0'].store(memory)
                results['mem0'] = mem0_id  
                logger.info(f"🧠 Mem0 (intelligent): {mem0_id}")
            except Exception as e:
                logger.warning(f"⚠️ Mem0 storage failed: {e}")
        
        return results
    
    async def smart_retrieve(self, query: str, strategy: str = "best", limit: int = 5) -> Dict[str, List[Memory]]:
        """
        Intelligently retrieve from optimal backends based on query type.
        
        Strategies:
        - 'semantic': Use Qdrant for similarity search
        - 'relationship': Use Neo4j for graph traversal  
        - 'personal': Use Mem0 for user-specific context
        - 'recent': Use cache for fast recent access
        - 'best': Automatically choose optimal strategy
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        memory_query = MemoryQuery(text=query, limit=limit)
        
        # Determine optimal strategy
        if strategy == 'best':
            strategy = self._determine_optimal_strategy(query)
        
        logger.info(f"🎯 Using strategy: {strategy}")
        
        if strategy == 'semantic' and 'qdrant' in self.backends:
            # Qdrant: Multi-scale semantic similarity
            try:
                qdrant_results = await self.backends['qdrant'].retrieve(memory_query, Strategy.SEMANTIC)
                results['qdrant'] = qdrant_results.memories
                logger.info(f"🔍 Qdrant found {len(qdrant_results.memories)} semantic matches")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant retrieval failed: {e}")
                results['qdrant'] = []
        
        elif strategy == 'relationship' and 'neo4j' in self.backends:
            # Neo4j: Graph-based relationship discovery
            try:
                neo4j_results = await self.backends['neo4j'].retrieve(memory_query, Strategy.GRAPH)
                results['neo4j'] = neo4j_results.memories
                logger.info(f"🕸️ Neo4j found {len(neo4j_results.memories)} relationship matches")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j retrieval failed: {e}")
                results['neo4j'] = []
        
        elif strategy == 'personal' and 'mem0' in self.backends:
            # Mem0: User-specific intelligent extraction
            try:
                mem0_results = await self.backends['mem0'].retrieve(memory_query, Strategy.FUSED)
                results['mem0'] = mem0_results.memories
                logger.info(f"🧠 Mem0 found {len(mem0_results.memories)} personal matches")
            except Exception as e:
                logger.warning(f"⚠️ Mem0 retrieval failed: {e}")
                results['mem0'] = []
        
        elif strategy == 'recent':
            # Cache: Fast recent access
            cache_results = await self.backends['cache'].retrieve(memory_query, Strategy.TEMPORAL)
            results['cache'] = cache_results.memories
            logger.info(f"📦 Cache found {len(cache_results.memories)} recent matches")
        
        else:
            # Fallback: Query all available backends and fuse
            logger.info("🔄 Fallback: Querying all available backends")
            
            for backend_name, backend in self.backends.items():
                if backend_name == 'cache':
                    continue  # Skip cache in fallback
                try:
                    backend_results = await backend.retrieve(memory_query, Strategy.FUSED)
                    results[backend_name] = backend_results.memories
                    logger.info(f"✅ {backend_name}: {len(backend_results.memories)} matches")
                except Exception as e:
                    logger.warning(f"⚠️ {backend_name} failed: {e}")
                    results[backend_name] = []
        
        return results
    
    def _determine_optimal_strategy(self, query: str) -> str:
        """Automatically determine the best retrieval strategy."""
        query_lower = query.lower()
        
        # Relationship indicators
        if any(word in query_lower for word in ['related', 'connected', 'similar to', 'thread', 'pattern']):
            return 'relationship'
        
        # Personal context indicators  
        if any(word in query_lower for word in ['my', 'i', 'personal', 'remember', 'user']):
            return 'personal'
        
        # Recent/temporal indicators
        if any(word in query_lower for word in ['recent', 'latest', 'today', 'yesterday', 'last']):
            return 'recent'
        
        # Default to semantic for content-based queries
        return 'semantic'
    
    async def memory_fusion(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Advanced: Fuse results from all backends with intelligent ranking.
        
        Uses each backend's strengths and combines results optimally.
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"🔗 Memory fusion for: '{query}'")
        
        all_results = await self.smart_retrieve(query, strategy='fusion', limit=limit*2)  # Get more for fusion
        
        # Combine and deduplicate memories
        seen_texts = set()
        fused_memories = []
        
        # Prioritization: Neo4j (relationships) > Qdrant (semantics) > Mem0 (intelligence) > Cache
        priority_order = ['neo4j', 'qdrant', 'mem0', 'cache']
        
        for backend_name in priority_order:
            if backend_name in all_results:
                for memory in all_results[backend_name]:
                    # Simple deduplication by text similarity
                    if memory.text not in seen_texts:
                        seen_texts.add(memory.text)
                        fused_memories.append(memory)
                        
                        if len(fused_memories) >= limit:
                            break
                
                if len(fused_memories) >= limit:
                    break
        
        logger.info(f"🎯 Fused {len(fused_memories)} unique memories from {len(all_results)} backends")
        return fused_memories[:limit]


async def demo_intelligent_orchestration():
    """Demonstrate intelligent multi-backend orchestration."""
    print("🚀 Intelligent Memory Orchestration Demo")
    print("=" * 50)
    
    orchestrator = IntelligentMemoryOrchestrator()
    await orchestrator.initialize()
    
    # Test memories with different characteristics
    test_memories = [
        Memory(
            id="mem_001",
            text="Hive Jodi has 8 frames of brood and needs honey supers added",
            timestamp=datetime.now(),
            context={
                'place': 'apiary', 
                'people': ['Blake'], 
                'topics': ['beekeeping', 'hive_management'],
                'time': 'evening'
            },
            metadata={'user_id': 'blake', 'priority': 'high'}
        ),
        Memory(
            id="mem_002", 
            text="Machine learning model achieved 94% accuracy on image classification",
            timestamp=datetime.now() - timedelta(hours=2),
            context={'topics': ['ML', 'computer_vision']},
            metadata={'user_id': 'blake', 'project': 'research'}
        ),
        Memory(
            id="mem_003",
            text="Remember to pick up groceries: milk, eggs, bread",
            timestamp=datetime.now() - timedelta(minutes=30),
            context={'place': 'home', 'topics': ['personal', 'todo']},
            metadata={'user_id': 'blake', 'type': 'reminder'}
        )
    ]
    
    print("\n📝 Storing memories intelligently...")
    for memory in test_memories:
        results = await orchestrator.smart_store(memory)
        print(f"Memory '{memory.text[:30]}...' stored in: {list(results.keys())}")
    
    print("\n🔍 Testing retrieval strategies...")
    
    # Test different retrieval strategies
    queries = [
        ("beekeeping management", "semantic"),
        ("what did Blake do recently", "personal"), 
        ("machine learning research", "semantic"),
        ("home tasks", "relationship"),
        ("recent activities", "recent")
    ]
    
    for query, strategy in queries:
        print(f"\nQuery: '{query}' (strategy: {strategy})")
        results = await orchestrator.smart_retrieve(query, strategy=strategy, limit=3)
        
        for backend_name, memories in results.items():
            print(f"  {backend_name}: {len(memories)} results")
            for memory in memories[:2]:  # Show first 2
                print(f"    - {memory.text[:50]}...")
    
    print(f"\n🔗 Testing memory fusion...")
    fused_results = await orchestrator.memory_fusion("Blake's recent work", limit=5)
    print(f"Fusion found {len(fused_results)} memories:")
    for memory in fused_results:
        print(f"  - {memory.text[:60]}...")
    
    print(f"\n✅ Intelligent orchestration demo complete!")
    print(f"💡 Each backend used for its optimal strengths!")


if __name__ == "__main__":
    asyncio.run(demo_intelligent_orchestration())