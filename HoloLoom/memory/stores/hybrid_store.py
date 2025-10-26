"""
Hybrid Memory Store - Multi-Backend Fusion
==========================================
Combines multiple memory backends with weighted fusion.

Architecture:
    Query → [Mem0, Neo4j, Qdrant] → Score Fusion → Ranked Results

Fusion Strategies:
- Weighted: Each backend gets a weight (e.g., 30% Mem0, 30% Neo4j, 40% Qdrant)
- Max: Take max score across backends
- Mean: Average scores
- Rank: Reciprocal rank fusion
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy, MemoryStore


@dataclass
class BackendConfig:
    """Configuration for a backend in hybrid store."""
    store: MemoryStore
    weight: float = 1.0
    enabled: bool = True
    name: str = "backend"


class HybridMemoryStore:
    """
    Hybrid store that fuses results from multiple backends.
    
    Example:
        hybrid = HybridMemoryStore(
            backends=[
                BackendConfig(Mem0MemoryStore(), weight=0.3, name="mem0"),
                BackendConfig(Neo4jMemoryStore(), weight=0.3, name="neo4j"),
                BackendConfig(QdrantMemoryStore(), weight=0.4, name="qdrant")
            ],
            fusion_method="weighted"
        )
    
    Fusion Methods:
    - weighted: Sum of (backend_score * weight)
    - max: Max score across backends
    - mean: Average of backend scores
    - rrf: Reciprocal Rank Fusion (1 / (k + rank))
    """
    
    def __init__(
        self,
        backends: List[BackendConfig],
        fusion_method: str = "weighted",
        rrf_k: int = 60
    ):
        self.backends = [b for b in backends if b.enabled]
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.logger = logging.getLogger(__name__)
        
        # Normalize weights
        if fusion_method == "weighted":
            total_weight = sum(b.weight for b in self.backends)
            if total_weight > 0:
                for b in self.backends:
                    b.weight = b.weight / total_weight
        
        self.logger.info(
            f"Hybrid store initialized with {len(self.backends)} backends: "
            f"{[b.name for b in self.backends]}"
        )
    
    async def store(self, memory: Memory) -> str:
        """Store in ALL backends (parallel writes)."""
        import asyncio
        
        # Store in each backend
        tasks = [
            backend.store.store(memory)
            for backend in self.backends
        ]
        
        # Wait for all
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return first successful ID (they should all be the same)
        for result in results:
            if isinstance(result, str):
                self.logger.info(f"Stored memory {result} in {len(self.backends)} backends")
                return result
        
        # Fallback
        self.logger.warning("No backends successfully stored memory")
        return memory.id or "unknown"
    
    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve from all backends and fuse results.
        
        Process:
        1. Query all backends in parallel
        2. Collect results
        3. Fuse scores using configured method
        4. De-duplicate by memory ID
        5. Sort and return top-k
        """
        import asyncio
        
        # Query all backends in parallel
        tasks = [
            backend.store.retrieve(query, strategy)
            for backend in self.backends
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, RetrievalResult):
                valid_results.append((self.backends[i], result))
            else:
                self.logger.warning(
                    f"Backend {self.backends[i].name} failed: {result}"
                )
        
        if not valid_results:
            return RetrievalResult(
                memories=[],
                scores=[],
                strategy_used=strategy.value,
                metadata={'error': 'all_backends_failed'}
            )
        
        # Fuse results
        if self.fusion_method == "weighted":
            fused = self._fuse_weighted(valid_results)
        elif self.fusion_method == "max":
            fused = self._fuse_max(valid_results)
        elif self.fusion_method == "mean":
            fused = self._fuse_mean(valid_results)
        elif self.fusion_method == "rrf":
            fused = self._fuse_rrf(valid_results)
        else:
            fused = self._fuse_weighted(valid_results)  # Default
        
        # Sort and limit
        fused.sort(key=lambda x: x[1], reverse=True)
        top_k = fused[:query.limit]
        
        return RetrievalResult(
            memories=[mem for mem, _ in top_k],
            scores=[score for _, score in top_k],
            strategy_used=f"hybrid_{self.fusion_method}",
            metadata={
                'backends_used': [b.name for b, _ in valid_results],
                'fusion_method': self.fusion_method,
                'total_candidates': len(fused)
            }
        )
    
    def _fuse_weighted(
        self,
        results: List[tuple[BackendConfig, RetrievalResult]]
    ) -> List[tuple[Memory, float]]:
        """Weighted fusion: score = sum(backend_score * weight)."""
        memory_scores = {}
        
        for backend, result in results:
            for mem, score in zip(result.memories, result.scores):
                weighted_score = score * backend.weight
                
                if mem.id not in memory_scores:
                    memory_scores[mem.id] = {
                        'memory': mem,
                        'score': weighted_score,
                        'sources': [backend.name]
                    }
                else:
                    memory_scores[mem.id]['score'] += weighted_score
                    memory_scores[mem.id]['sources'].append(backend.name)
        
        # Add source metadata
        for mem_id, data in memory_scores.items():
            data['memory'].metadata['fusion_sources'] = data['sources']
        
        return [(data['memory'], data['score']) for data in memory_scores.values()]
    
    def _fuse_max(
        self,
        results: List[tuple[BackendConfig, RetrievalResult]]
    ) -> List[tuple[Memory, float]]:
        """Max fusion: score = max(backend_scores)."""
        memory_scores = {}
        
        for backend, result in results:
            for mem, score in zip(result.memories, result.scores):
                if mem.id not in memory_scores:
                    memory_scores[mem.id] = {
                        'memory': mem,
                        'score': score,
                        'source': backend.name
                    }
                else:
                    if score > memory_scores[mem.id]['score']:
                        memory_scores[mem.id]['score'] = score
                        memory_scores[mem.id]['source'] = backend.name
        
        return [(data['memory'], data['score']) for data in memory_scores.values()]
    
    def _fuse_mean(
        self,
        results: List[tuple[BackendConfig, RetrievalResult]]
    ) -> List[tuple[Memory, float]]:
        """Mean fusion: score = mean(backend_scores)."""
        memory_scores = {}
        
        for backend, result in results:
            for mem, score in zip(result.memories, result.scores):
                if mem.id not in memory_scores:
                    memory_scores[mem.id] = {
                        'memory': mem,
                        'scores': [score],
                        'sources': [backend.name]
                    }
                else:
                    memory_scores[mem.id]['scores'].append(score)
                    memory_scores[mem.id]['sources'].append(backend.name)
        
        # Compute means
        fused = []
        for data in memory_scores.values():
            mean_score = sum(data['scores']) / len(data['scores'])
            data['memory'].metadata['fusion_sources'] = data['sources']
            fused.append((data['memory'], mean_score))
        
        return fused
    
    def _fuse_rrf(
        self,
        results: List[tuple[BackendConfig, RetrievalResult]]
    ) -> List[tuple[Memory, float]]:
        """
        Reciprocal Rank Fusion.
        
        RRF Score = sum(1 / (k + rank)) across backends
        where k is a constant (default 60), rank is position in results
        """
        memory_scores = {}
        
        for backend, result in results:
            for rank, mem in enumerate(result.memories):
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                
                if mem.id not in memory_scores:
                    memory_scores[mem.id] = {
                        'memory': mem,
                        'score': rrf_score,
                        'sources': [backend.name]
                    }
                else:
                    memory_scores[mem.id]['score'] += rrf_score
                    memory_scores[mem.id]['sources'].append(backend.name)
        
        return [(data['memory'], data['score']) for data in memory_scores.values()]
    
    async def delete(self, memory_id: str) -> bool:
        """Delete from ALL backends."""
        import asyncio
        
        tasks = [
            backend.store.delete(memory_id)
            for backend in self.backends
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return True if any backend succeeded
        return any(r is True for r in results)
    
    async def health_check(self) -> Dict:
        """Check health of all backends."""
        import asyncio
        
        tasks = [
            backend.store.health_check()
            for backend in self.backends
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        backend_health = {}
        for backend, result in zip(self.backends, results):
            if isinstance(result, dict):
                backend_health[backend.name] = result
            else:
                backend_health[backend.name] = {
                    'status': 'error',
                    'error': str(result)
                }
        
        # Overall status
        all_healthy = all(
            h.get('status') == 'healthy'
            for h in backend_health.values()
        )
        
        return {
            'status': 'healthy' if all_healthy else 'degraded',
            'backend': 'hybrid',
            'fusion_method': self.fusion_method,
            'backends': backend_health,
            'total_backends': len(self.backends),
            'healthy_backends': sum(
                1 for h in backend_health.values()
                if h.get('status') == 'healthy'
            )
        }
