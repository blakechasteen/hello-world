"""
Mem0 Memory Store - User-Specific Intelligent Extraction
========================================================
Wraps mem0ai for LLM-based memory extraction and filtering.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy

# Optional mem0 import
try:
    from mem0 import Memory as Mem0Client
    _HAVE_MEM0 = True
except ImportError:
    Mem0Client = None
    _HAVE_MEM0 = False


class Mem0MemoryStore:
    """
    Mem0-backed memory store.
    
    Features:
    - LLM-based extraction (decides what's important)
    - User-specific memories (personalization)
    - Temporal decay (memories fade over time)
    - Multi-level memory (user/session/agent)
    
    Requires: pip install mem0ai
    """
    
    def __init__(
        self,
        user_id: str = "default",
        api_key: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        if not _HAVE_MEM0:
            raise RuntimeError(
                "mem0ai not installed. Install with: pip install mem0ai"
            )

        self.user_id = user_id
        self.logger = logging.getLogger(__name__)

        # Initialize mem0 client
        if api_key:
            self.client = Mem0Client.from_config({"api_key": api_key})
        elif config:
            self.client = Mem0Client.from_config(config)
        else:
            # Use local Ollama (NO API KEY REQUIRED!)
            # Using Qdrant for vector storage to avoid dimension mismatches
            default_config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "llama3.2:3b",
                        "temperature": 0.1,
                        "max_tokens": 1500,
                    }
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text:latest",
                        "embedding_dims": 768
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": "localhost",
                        "port": 6333,
                        "collection_name": "hololoom_mem0_768"
                    }
                },
                "version": "v1.1"
            }
            self.client = Mem0Client.from_config(default_config)

        self.logger.info(f"Mem0 store initialized for user: {user_id} (using Ollama)")
    
    async def store(self, memory: Memory) -> str:
        """
        Store memory via mem0's intelligent extraction.
        
        Mem0 will:
        1. Extract important facts/entities
        2. Store in user's memory space
        3. Build relationships between memories
        """
        user_id = memory.metadata.get('user_id', self.user_id)
        
        # Format as conversation for mem0
        messages = [
            {
                "role": "user",
                "content": memory.text
            }
        ]
        
        # Add memory
        result = self.client.add(messages, user_id=user_id, metadata=memory.metadata)
        
        # Extract mem0's assigned ID
        if isinstance(result, dict) and 'results' in result:
            results_list = result['results']
            if results_list and len(results_list) > 0:
                mem0_id = results_list[0].get('id', memory.id or 'unknown')
            else:
                mem0_id = memory.id or 'unknown'
        else:
            mem0_id = memory.id or 'unknown'
        
        self.logger.info(f"Stored memory {mem0_id} for user {user_id}")
        return f"mem0_{mem0_id}"

    async def store_many(self, memories: List[Memory]) -> List[str]:
        """Store multiple memories (batch operation)."""
        memory_ids = []
        for memory in memories:
            memory_id = await self.store(memory)
            memory_ids.append(memory_id)
        return memory_ids

    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        # Extract mem0 ID from our prefixed format
        if memory_id.startswith('mem0_'):
            mem0_id = memory_id[5:]
        else:
            mem0_id = memory_id
        
        try:
            # Get memory from mem0
            result = self.client.get(memory_id=mem0_id, user_id=self.user_id)
            
            if result and isinstance(result, dict):
                # Extract memory data
                memory_text = result.get('memory', result.get('text', ''))
                timestamp = datetime.now()  # mem0 doesn't store original timestamp
                
                # Try to parse timestamp if available
                if 'created_at' in result:
                    try:
                        timestamp = datetime.fromisoformat(result['created_at'].replace('Z', '+00:00'))
                    except:
                        pass
                
                return Memory(
                    id=memory_id,
                    text=memory_text,
                    timestamp=timestamp,
                    context=result.get('metadata', {}).get('context', {}),
                    metadata=result.get('metadata', {})
                )
        except Exception as e:
            self.logger.warning(f"Failed to get memory {memory_id}: {e}")
        
        return None
    
    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories using mem0's search.
        
        Mem0 handles:
        - Semantic similarity
        - User-specific filtering
        - Temporal relevance
        """
        user_id = query.user_id or self.user_id
        
        # Search mem0
        results = self.client.search(
            query=query.text,
            user_id=user_id,
            limit=query.limit
        )
        
        # Convert to Memory objects
        memories = []
        scores = []
        
        for item in results.get('results', []):
            mem = Memory(
                id=f"mem0_{item.get('id', 'unknown')}",
                text=item.get('memory', ''),
                timestamp=self._parse_timestamp(item.get('created_at')),
                context=item.get('metadata', {}),
                metadata={
                    'source': 'mem0',
                    'mem0_id': item.get('id'),
                    'user_id': user_id,
                    'hash': item.get('hash')
                }
            )
            memories.append(mem)
            scores.append(item.get('score', 0.0))
        
        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used='mem0_search',
            metadata={
                'backend': 'mem0',
                'user_id': user_id,
                'total_results': len(results.get('results', []))
            }
        )
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from mem0."""
        # Extract mem0 ID
        if memory_id.startswith('mem0_'):
            mem0_id = memory_id[5:]
        else:
            mem0_id = memory_id
        
        try:
            self.client.delete(mem0_id)
            self.logger.info(f"Deleted memory {mem0_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete {mem0_id}: {e}")
            return False
    
    async def health_check(self) -> Dict:
        """Check mem0 connection."""
        try:
            # Try to get all memories (limited)
            result = self.client.get_all(user_id=self.user_id)
            memory_count = len(result.get('results', []))
            
            return {
                'status': 'healthy',
                'backend': 'mem0',
                'user_id': self.user_id,
                'memory_count': memory_count,
                'features': ['llm_extraction', 'user_tracking', 'temporal_decay']
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'mem0',
                'error': str(e)
            }
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse mem0 timestamp string."""
        if not timestamp_str:
            return datetime.now()
        
        try:
            # mem0 uses ISO format
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            return datetime.now()
