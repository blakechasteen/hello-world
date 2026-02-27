"""
HoloLoom + Mem0 Integration Adapter
====================================
Hybrid memory system that combines:
- Mem0's intelligent extraction and user-specific memory
- HoloLoom's multi-scale retrieval and domain reasoning

This module provides adapters and managers to coordinate both systems.

Philosophy:
Mem0 decides WHAT to remember (intelligent extraction).
HoloLoom decides HOW to recall it (multi-scale, domain-aware retrieval).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

# HoloLoom imports
try:
    from holoLoom.documentation.types import Query, Context, Features, MemoryShard
    from holoLoom.memory.cache import MemoryManager, RetrieverMS
    from holoLoom.memory.graph import KG, KGEdge
except ImportError as e:
    print(f"Warning: Could not import HoloLoom modules: {e}")
    print("Make sure you're running from the repository root")

# Mem0 import (optional dependency)
try:
    from mem0 import Memory
    _HAVE_MEM0 = True
except ImportError:
    Memory = None
    _HAVE_MEM0 = False
    logging.warning("mem0ai not installed. Install with: pip install mem0ai")


logging.basicConfig(level=logging.INFO)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Mem0Config:
    """Configuration for Mem0 integration."""
    
    enabled: bool = True
    api_key: Optional[str] = None  # For mem0 managed platform
    
    # Feature flags
    extraction_enabled: bool = True  # Use mem0's LLM extraction
    graph_sync_enabled: bool = True  # Sync mem0 graph to HoloLoom KG
    user_tracking_enabled: bool = True  # Track user-specific memories
    
    # Fusion weights
    mem0_weight: float = 0.3  # 30% mem0 in fused retrieval
    hololoom_weight: float = 0.7  # 70% HoloLoom
    
    # Memory limits
    max_memories_per_query: int = 5
    memory_relevance_threshold: float = 0.5
    
    def validate(self):
        """Validate configuration."""
        if self.enabled and not _HAVE_MEM0:
            raise RuntimeError(
                "Mem0 integration enabled but mem0ai not installed. "
                "Install with: pip install mem0ai"
            )
        
        total_weight = self.mem0_weight + self.hololoom_weight
        if not (0.95 <= total_weight <= 1.05):
            logging.warning(
                f"Fusion weights sum to {total_weight:.3f}, expected ~1.0. "
                "Normalizing weights."
            )
            self.mem0_weight /= total_weight
            self.hololoom_weight /= total_weight


# ============================================================================
# Mem0 <-> HoloLoom Shard Converter
# ============================================================================

class Mem0ShardConverter:
    """
    Converts between mem0 memory format and HoloLoom MemoryShard format.
    
    This enables bidirectional synchronization between the two systems.
    """
    
    @staticmethod
    def mem0_to_shard(
        mem0_memory: Dict,
        user_id: str = "default"
    ) -> MemoryShard:
        """
        Convert a mem0 memory to a HoloLoom MemoryShard.
        
        Args:
            mem0_memory: Memory dict from mem0
            user_id: User identifier
            
        Returns:
            MemoryShard instance
        """
        return MemoryShard(
            id=f"mem0_{mem0_memory.get('id', 'unknown')}",
            text=mem0_memory.get('memory', ''),
            episode=f"user_{user_id}",
            entities=mem0_memory.get('entities', []),
            motifs=[],  # Mem0 doesn't have motifs; could extract later
            metadata={
                'source': 'mem0',
                'mem0_id': mem0_memory.get('id'),
                'mem0_score': mem0_memory.get('score', 0.0),
                'mem0_type': mem0_memory.get('memory_type', 'episodic'),
                'created_at': mem0_memory.get('created_at'),
                'user_id': user_id
            }
        )
    
    @staticmethod
    def shard_to_mem0_text(shard: MemoryShard) -> str:
        """
        Convert a HoloLoom shard to text suitable for mem0 storage.
        
        Args:
            shard: MemoryShard instance
            
        Returns:
            Text string for mem0
        """
        # Format: "In episode {episode}: {text}"
        # This gives mem0 context about where the memory came from
        return f"In {shard.episode}: {shard.text}"


# ============================================================================
# Graph Synchronization Engine
# ============================================================================

class GraphSyncEngine:
    """
    Synchronizes entity relationships between mem0's graph memory
    and HoloLoom's knowledge graph.
    
    Strategy:
    - Mem0 tracks cross-session entity co-occurrences
    - HoloLoom KG provides explicit typed relationships
    - We sync bidirectionally to enrich both systems
    """
    
    def __init__(self, kg: KG):
        self.kg = kg
        self.logger = logging.getLogger(__name__)
    
    async def sync_mem0_to_kg(
        self,
        mem0_client: 'Memory',
        user_id: str = "default"
    ):
        """
        Sync mem0's memories into HoloLoom's knowledge graph.
        
        Creates:
        - Entity nodes for all entities in mem0 memories
        - MENTIONED_IN edges from entities to memory nodes
        - CO_OCCURS edges between entities in same memory
        - Temporal connections using connect_entity_to_time()
        
        Args:
            mem0_client: Initialized mem0 Memory instance
            user_id: User to sync memories for
        """
        self.logger.info(f"Syncing mem0 memories to KG for user {user_id}")
        
        try:
            # Get all memories for user
            memories = mem0_client.get_all(user_id=user_id)
            
            for memory in memories.get('results', []):
                memory_id = f"mem0_{memory.get('id')}"
                memory_text = memory.get('memory', '')
                entities = memory.get('entities', [])
                timestamp = memory.get('created_at')
                
                # Create memory node
                if memory_id not in self.kg.G:
                    self.kg.G.add_node(
                        memory_id,
                        text=memory_text,
                        kind='mem0_memory',
                        user_id=user_id
                    )
                
                # Connect entities to memory
                for i, entity in enumerate(entities):
                    # Add entity node if doesn't exist
                    if entity not in self.kg.G:
                        self.kg.G.add_node(entity, kind='entity')
                    
                    # Entity -> Memory edge
                    self.kg.add_edge(KGEdge(
                        src=entity,
                        dst=memory_id,
                        type="MENTIONED_IN",
                        weight=1.0,
                        metadata={'memory_text': memory_text[:100]}
                    ))
                    
                    # Connect to temporal thread if timestamp available
                    if timestamp:
                        try:
                            self.kg.connect_entity_to_time(
                                entity,
                                timestamp,
                                edge_type="ACTIVE_AT",
                                weight=1.0
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to connect {entity} to time: {e}")
                    
                    # Create co-occurrence edges between entities
                    for other_entity in entities[i+1:]:
                        self.kg.add_edge(KGEdge(
                            src=entity,
                            dst=other_entity,
                            type="CO_OCCURS_WITH",
                            weight=0.5,
                            metadata={'context': memory_id}
                        ))
            
            self.logger.info(f"Synced {len(memories.get('results', []))} memories to KG")
            
        except Exception as e:
            self.logger.error(f"Error syncing mem0 to KG: {e}", exc_info=True)
    
    def get_entity_context_from_kg(
        self,
        entity: str,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Get rich context for an entity from the KG.
        
        Returns connected entities, relationships, and temporal info.
        
        Args:
            entity: Entity name
            max_hops: Maximum relationship hops
            
        Returns:
            Dict with entity context
        """
        if entity not in self.kg.G:
            return {'entity': entity, 'found': False}
        
        # Get neighbors
        neighbors = self.kg.get_neighbors(entity, max_hops=max_hops)
        
        # Get relationships
        relationships = []
        for neighbor in list(neighbors)[:10]:  # Limit to top 10
            edge_types = self.kg.get_edge_types(entity, neighbor)
            if edge_types:
                relationships.append({
                    'target': neighbor,
                    'types': edge_types
                })
        
        # Get temporal info (if connected to time threads)
        time_neighbors = [
            n for n in neighbors
            if n.startswith('time::')
        ]
        
        return {
            'entity': entity,
            'found': True,
            'neighbor_count': len(neighbors),
            'relationships': relationships,
            'temporal_threads': time_neighbors[:5],  # Top 5 time connections
        }


# ============================================================================
# Hybrid Memory Manager
# ============================================================================

@dataclass
class HybridMemoryManager:
    """
    Unified memory manager that coordinates HoloLoom and mem0.
    
    Architecture:
    - Mem0: User-specific extraction, filtering, decay
    - HoloLoom: Multi-scale retrieval, domain reasoning, KG
    
    Usage:
        config = Mem0Config()
        hololoom_memory = create_memory_manager(...)
        hybrid = HybridMemoryManager(hololoom_memory, config)
        
        # Store memory
        await hybrid.store(query, results, user_id="blake")
        
        # Retrieve memory
        context = await hybrid.retrieve(query, user_id="blake")
    """
    
    hololoom: MemoryManager
    config: Mem0Config
    kg: Optional[KG] = None
    
    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.config.validate()
        
        # Initialize mem0 if enabled
        if self.config.enabled:
            if self.config.api_key:
                self.mem0 = Memory(api_key=self.config.api_key)
            else:
                self.mem0 = Memory()  # Uses local/default config
            
            self.converter = Mem0ShardConverter()
            
            # Initialize graph sync if KG provided
            if self.kg and self.config.graph_sync_enabled:
                self.graph_sync = GraphSyncEngine(self.kg)
            else:
                self.graph_sync = None
        else:
            self.mem0 = None
            self.converter = None
            self.graph_sync = None
        
        self.logger.info(
            f"HybridMemoryManager initialized "
            f"(mem0_enabled={self.config.enabled})"
        )
    
    async def store(
        self,
        query: Query,
        results: Dict,
        features: Features,
        user_id: str = "default"
    ):
        """
        Store memory in both HoloLoom and mem0.
        
        Pipeline:
        1. HoloLoom persists raw query/results
        2. Mem0 extracts important facts/preferences (if enabled)
        3. Sync mem0 entities to KG (if enabled)
        
        Args:
            query: Query object
            results: Results/actions taken
            features: Extracted features
            user_id: User identifier for personalization
        """
        # Always persist to HoloLoom (baseline)
        await self.hololoom.persist(query, results, features)
        
        # Optionally use mem0's intelligent extraction
        if self.mem0 and self.config.extraction_enabled:
            try:
                # Format as conversation for mem0
                messages = [
                    {"role": "user", "content": query.text},
                    {"role": "assistant", "content": results.get('response', '')}
                ]
                
                # Let mem0 extract what's important
                mem0_result = self.mem0.add(messages, user_id=user_id)
                
                self.logger.info(
                    f"Mem0 extracted {len(mem0_result.get('results', []))} memories"
                )
                
                # Optionally sync to KG
                if self.graph_sync and self.config.graph_sync_enabled:
                    await self.graph_sync.sync_mem0_to_kg(self.mem0, user_id)
                
            except Exception as e:
                self.logger.error(f"Mem0 extraction failed: {e}", exc_info=True)
    
    async def retrieve(
        self,
        query: Query,
        user_id: str = "default",
        k: int = 6,
        kg_sub = None
    ) -> Context:
        """
        Retrieve memory using fused HoloLoom + mem0 approach.
        
        Pipeline:
        1. Get HoloLoom's multi-scale retrieval results
        2. Get mem0's user-specific memories (if enabled)
        3. Fuse results using configured weights
        4. Return unified Context
        
        Args:
            query: Query object
            user_id: User identifier
            k: Number of results
            kg_sub: Knowledge graph subgraph (optional)
            
        Returns:
            Context with fused memories
        """
        # Get HoloLoom's retrieval (baseline)
        hololoom_context = await self.hololoom.retrieve(query, kg_sub)
        
        # If mem0 disabled, return HoloLoom results
        if not self.mem0:
            return hololoom_context
        
        # Get mem0's memories
        try:
            mem0_results = self.mem0.search(
                query=query.text,
                user_id=user_id,
                limit=self.config.max_memories_per_query
            )
            
            # Convert mem0 memories to shards
            mem0_shards = []
            for mem in mem0_results.get('results', []):
                score = mem.get('score', 0.0)
                if score >= self.config.memory_relevance_threshold:
                    shard = self.converter.mem0_to_shard(mem, user_id)
                    mem0_shards.append((shard, score))
            
            # Fuse results
            fused_context = self._fuse_contexts(
                hololoom_context,
                mem0_shards,
                query
            )
            
            return fused_context
            
        except Exception as e:
            self.logger.error(f"Mem0 retrieval failed: {e}", exc_info=True)
            # Fall back to HoloLoom only
            return hololoom_context
    
    def _fuse_contexts(
        self,
        hololoom_context: Context,
        mem0_shards: List[Tuple[MemoryShard, float]],
        query: Query
    ) -> Context:
        """
        Fuse HoloLoom and mem0 retrieval results.
        
        Strategy:
        - Weight scores by configured weights
        - Combine and re-rank
        - Remove duplicates (by text similarity)
        
        Args:
            hololoom_context: Context from HoloLoom retrieval
            mem0_shards: Shards from mem0 with scores
            query: Original query
            
        Returns:
            Fused Context
        """
        # Get HoloLoom hits
        hololoom_hits = hololoom_context.hits if hasattr(hololoom_context, 'hits') else []
        
        # Reweight scores
        weighted_hits = []
        
        # HoloLoom hits (70% weight by default)
        for shard, score in hololoom_hits:
            weighted_hits.append((
                shard,
                score * self.config.hololoom_weight,
                'hololoom'
            ))
        
        # Mem0 hits (30% weight by default)
        for shard, score in mem0_shards:
            weighted_hits.append((
                shard,
                score * self.config.mem0_weight,
                'mem0'
            ))
        
        # Sort by weighted score
        weighted_hits.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates (simple text-based deduplication)
        seen_texts = set()
        unique_hits = []
        
        for shard, score, source in weighted_hits:
            text_lower = shard.text.lower().strip()
            if text_lower not in seen_texts:
                seen_texts.add(text_lower)
                unique_hits.append((shard, score))
        
        # Take top k
        top_hits = unique_hits[:len(hololoom_hits)]  # Keep same count as HoloLoom
        
        # Build fused context
        fused_shards = [s for s, _ in top_hits]
        shard_texts = [s.text for s in fused_shards]
        relevance = float(np.mean([score for _, score in top_hits])) if top_hits else 0.0
        
        # Create new context with fused results
        return Context(
            shards=fused_shards,
            hits=top_hits,
            shard_texts=shard_texts,
            relevance=relevance,
            query=query,
            kg_sub=hololoom_context.kg_sub if hasattr(hololoom_context, 'kg_sub') else None,
            metadata={
                'fusion_method': 'weighted',
                'mem0_weight': self.config.mem0_weight,
                'hololoom_weight': self.config.hololoom_weight,
                'mem0_count': len([h for h in weighted_hits if h[2] == 'mem0']),
                'hololoom_count': len([h for h in weighted_hits if h[2] == 'hololoom']),
            }
        )
    
    async def get_user_profile(self, user_id: str = "default") -> Dict:
        """
        Get user profile from mem0.
        
        Returns aggregated user memories and preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user profile data
        """
        if not self.mem0:
            return {'user_id': user_id, 'memories': [], 'available': False}
        
        try:
            all_memories = self.mem0.get_all(user_id=user_id)
            
            return {
                'user_id': user_id,
                'memory_count': len(all_memories.get('results', [])),
                'memories': all_memories.get('results', [])[:10],  # Top 10
                'available': True
            }
        except Exception as e:
            self.logger.error(f"Failed to get user profile: {e}")
            return {'user_id': user_id, 'memories': [], 'available': False, 'error': str(e)}
    
    async def shutdown(self):
        """Graceful shutdown of both memory systems."""
        # Shutdown HoloLoom memory
        await self.hololoom.shutdown()
        
        # Mem0 doesn't require explicit shutdown in current version
        self.logger.info("HybridMemoryManager shutdown complete")


# ============================================================================
# Factory Functions
# ============================================================================

def create_hybrid_memory(
    hololoom_memory: MemoryManager,
    mem0_config: Optional[Mem0Config] = None,
    kg: Optional[KG] = None
) -> HybridMemoryManager:
    """
    Factory function to create a hybrid memory manager.
    
    Args:
        hololoom_memory: Initialized HoloLoom MemoryManager
        mem0_config: Optional Mem0 configuration (uses defaults if None)
        kg: Optional knowledge graph for entity sync
        
    Returns:
        Configured HybridMemoryManager
    """
    if mem0_config is None:
        mem0_config = Mem0Config()
    
    return HybridMemoryManager(
        hololoom=hololoom_memory,
        config=mem0_config,
        kg=kg
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=== Hybrid Memory Demo ===\n")
        
        # Note: This demo requires HoloLoom components
        # Run from repository root with: python -m HoloLoom.memory.mem0_adapter
        
        print("Demo would show:")
        print("1. Initialize HoloLoom memory")
        print("2. Initialize mem0 client")
        print("3. Create HybridMemoryManager")
        print("4. Store query/results in both systems")
        print("5. Retrieve using fused approach")
        print("6. Show entity sync to KG")
        
        print("\nTo run full demo, ensure:")
        print("- mem0ai is installed: pip install mem0ai")
        print("- HoloLoom is properly configured")
        print("- Run from repository root")
        
        # Pseudo-code for actual demo:
        """
        from holoLoom.memory.cache import create_memory_manager
        from holoLoom.embedding.spectral import MatryoshkaEmbeddings
        from holoLoom.memory.graph import KG
        
        # Create components
        shards = [...]  # Sample shards
        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        hololoom_memory = create_memory_manager(shards, emb)
        kg = KG()
        
        # Create hybrid manager
        mem0_config = Mem0Config(enabled=True)
        hybrid = create_hybrid_memory(hololoom_memory, mem0_config, kg)
        
        # Store
        query = Query(text="How should I prepare my hives for winter?")
        results = {"response": "Ensure adequate honey stores..."}
        features = Features(...)
        await hybrid.store(query, results, features, user_id="blake")
        
        # Retrieve
        context = await hybrid.retrieve(query, user_id="blake")
        print(f"Retrieved {len(context.shards)} shards")
        print(f"Fusion metadata: {context.metadata}")
        
        # Shutdown
        await hybrid.shutdown()
        """
    
    asyncio.run(demo())
