"""
In-Memory Store - Simple Implementation for Testing
===================================================
No external dependencies. Pure Python dict-based storage.
"""

from typing import Dict, List
from datetime import datetime
import hashlib

try:
    from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy
except ImportError:
    # Standalone mode
    from protocol import Memory, MemoryQuery, RetrievalResult, Strategy


class InMemoryStore:
    """
    Simple in-memory store using Python dict.
    
    Good for:
    - Testing
    - Development
    - Small workloads
    
    Not suitable for:
    - Production (no persistence)
    - Large datasets (no indexing)
    """
    
    def __init__(self):
        self._memories: Dict[str, Memory] = {}
        self._user_index: Dict[str, List[str]] = {}  # user_id -> [memory_ids]
    
    async def store(self, memory: Memory) -> str:
        """Store memory in dict."""
        # Generate ID if not provided
        if not memory.id:
            memory.id = self._generate_id(memory.text)

        self._memories[memory.id] = memory

        # Index by user
        user_id = memory.metadata.get('user_id', 'default')
        if user_id not in self._user_index:
            self._user_index[user_id] = []
        self._user_index[user_id].append(memory.id)

        return memory.id

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """
        Store multiple memories (batch operation).

        Args:
            memories: List of Memory objects

        Returns:
            List of memory IDs
        """
        ids = []
        for memory in memories:
            mem_id = await self.store(memory)
            ids.append(mem_id)
        return ids

    async def get_by_id(self, memory_id: str) -> Memory:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Unique identifier

        Returns:
            Memory object or None if not found
        """
        return self._memories.get(memory_id)

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using simple text matching.
        
        Strategy mapping:
        - TEMPORAL: Sort by timestamp
        - SEMANTIC: Text substring matching (simplified)
        - GRAPH: Return all connected (simplified)
        - PATTERN: Random selection (no real patterns)
        - FUSED: Combine temporal + semantic
        """
        # Get user's memories
        user_mems = self._user_index.get(query.user_id, [])
        candidates = [self._memories[mid] for mid in user_mems if mid in self._memories]
        
        if not candidates:
            return RetrievalResult(
                memories=[],
                scores=[],
                strategy_used=strategy.value,
                metadata={'total_memories': 0}
            )
        
        # Apply strategy
        if strategy == Strategy.TEMPORAL:
            scored = self._score_temporal(candidates, query)
        elif strategy == Strategy.SEMANTIC:
            scored = self._score_semantic(candidates, query)
        elif strategy == Strategy.GRAPH:
            scored = self._score_graph(candidates, query)
        elif strategy == Strategy.PATTERN:
            scored = self._score_pattern(candidates, query)
        else:  # FUSED
            scored = self._score_fused(candidates, query)
        
        # Sort and limit
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:query.limit]
        
        return RetrievalResult(
            memories=[mem for mem, _ in top_k],
            scores=[score for _, score in top_k],
            strategy_used=strategy.value,
            metadata={
                'total_memories': len(candidates),
                'matched': len(scored)
            }
        )
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self._memories:
            mem = self._memories[memory_id]
            user_id = mem.metadata.get('user_id', 'default')
            
            # Remove from storage
            del self._memories[memory_id]
            
            # Remove from index
            if user_id in self._user_index:
                self._user_index[user_id].remove(memory_id)
            
            return True
        return False
    
    async def health_check(self) -> Dict:
        """Check store health."""
        return {
            'status': 'healthy',
            'backend': 'in-memory',
            'memory_count': len(self._memories),
            'user_count': len(self._user_index),
            'latency_ms': 0.1
        }
    
    # ========================================================================
    # Internal Scoring Methods
    # ========================================================================
    
    def _score_temporal(self, memories: List[Memory], query: MemoryQuery) -> List[tuple]:
        """Score by recency."""
        now = datetime.now()
        scored = []
        for mem in memories:
            # Time delta in seconds
            delta = (now - mem.timestamp).total_seconds()
            # Score: newer = higher (exponential decay)
            score = 1.0 / (1.0 + delta / 3600)  # Decay over hours
            scored.append((mem, score))
        return scored
    
    def _score_semantic(self, memories: List[Memory], query: MemoryQuery) -> List[tuple]:
        """Score by text similarity (simplified substring matching)."""
        query_lower = query.text.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for mem in memories:
            mem_lower = mem.text.lower()
            mem_words = set(mem_lower.split())
            
            # Jaccard similarity
            if len(query_words) == 0:
                score = 0.0
            else:
                intersection = len(query_words & mem_words)
                union = len(query_words | mem_words)
                score = intersection / union if union > 0 else 0.0
            
            # Bonus for substring match
            if query_lower in mem_lower:
                score += 0.2
            
            scored.append((mem, min(score, 1.0)))
        return scored
    
    def _score_graph(self, memories: List[Memory], query: MemoryQuery) -> List[tuple]:
        """Score by graph connectivity (simplified - just return all)."""
        # In real implementation, would use Neo4j traversal
        # For now, uniform scores
        return [(mem, 0.5) for mem in memories]
    
    def _score_pattern(self, memories: List[Memory], query: MemoryQuery) -> List[tuple]:
        """Score by patterns (simplified - random)."""
        # In real implementation, would use Hofstadter resonance
        import random
        return [(mem, random.random()) for mem in memories]
    
    def _score_fused(self, memories: List[Memory], query: MemoryQuery) -> List[tuple]:
        """Fused scoring: 50% temporal + 50% semantic."""
        temporal_scores = {mem.id: score for mem, score in self._score_temporal(memories, query)}
        semantic_scores = {mem.id: score for mem, score in self._score_semantic(memories, query)}
        
        scored = []
        for mem in memories:
            t_score = temporal_scores.get(mem.id, 0.0)
            s_score = semantic_scores.get(mem.id, 0.0)
            fused_score = 0.5 * t_score + 0.5 * s_score
            scored.append((mem, fused_score))
        return scored
    
    def _generate_id(self, text: str) -> str:
        """Generate deterministic ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:16]
