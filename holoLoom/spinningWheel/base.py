# -*- coding: utf-8 -*-
"""
Base Spinner
============
Lightweight interface for input adapters.

All spinners convert raw input -> MemoryShard list.

Design Philosophy:
- Spinners are thin adapters, not full pipelines
- They normalize input into MemoryShards
- Optional pre-enrichment adds context (Ollama, Neo4j, mem0)
- Heavy processing happens in the Orchestrator

Inheritance:
    BaseSpinner (ABC)
        ├── AudioSpinner
        ├── TextSpinner (future)
        └── CodeSpinner (future)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import holoLoom types
try:
    from holoLoom.documentation.types import MemoryShard
except ImportError:
    # Fallback if types not available
    from dataclasses import dataclass
    
    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: str
        entities: List[str]
        motifs: List[str]
        metadata: Dict[str, Any] = None


@dataclass
class SpinnerConfig:
    """
    Base configuration for spinners.
    
    Attributes:
        enable_enrichment: Enable Ollama/Neo4j/mem0 pre-enrichment
        ollama_model: Ollama model for context extraction
        neo4j_conn: Neo4j connection (optional)
        mem0_client: mem0ai client (optional)
    """
    enable_enrichment: bool = False
    ollama_model: str = "llama3.2:3b"  # Lightweight local model
    neo4j_conn: Any = None
    mem0_client: Any = None


class BaseSpinner(ABC):
    """
    Lightweight adapter that converts raw input -> MemoryShards.
    
    Philosophy:
    - Keep it simple: parse, normalize, optionally enrich
    - Output standardized MemoryShards
    - Let the Orchestrator do the heavy lifting (embeddings, spectral, policy)
    
    Subclasses must implement:
    - spin(raw_data) -> List[MemoryShard]
    """
    
    def __init__(self, config: SpinnerConfig):
        self.config = config
        
        # Initialize enrichment services if enabled
        if config.enable_enrichment:
            self._init_enrichment()
    
    def _init_enrichment(self):
        """Initialize optional enrichment services."""
        try:
            from .enrichment import OllamaEnricher
            self.ollama = OllamaEnricher(model=self.config.ollama_model)
        except ImportError:
            self.ollama = None
        
        # Neo4j and mem0 enrichers (optional)
        self.neo4j = None  # TODO: Implement Neo4jEnricher
        self.mem0 = None   # TODO: Implement Mem0Enricher
    
    @abstractmethod
    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert raw input -> MemoryShards.
        
        Args:
            raw_data: Modality-specific input
            
        Returns:
            List of MemoryShard objects ready for Orchestrator
        """
        pass
    
    async def enrich(self, text: str) -> Dict[str, Any]:
        """
        Optional: Pre-enrich text with context using Ollama/Neo4j/mem0.
        
        This is lightweight - just extract entities, basic motifs, sentiment.
        Heavy processing (embeddings, spectral features) happens in Orchestrator.
        
        Args:
            text: Text to enrich
            
        Returns:
            Dict with enrichment data (entities, motifs, sentiment)
        """
        if not self.config.enable_enrichment:
            return {}
        
        enrichment = {}
        
        # Ollama: Extract entities and basic context
        if hasattr(self, 'ollama') and self.ollama:
            try:
                enrichment['ollama'] = await self.ollama.extract_context(text)
            except Exception as e:
                enrichment['ollama_error'] = str(e)
        
        # Neo4j: Lookup related entities (if available)
        if hasattr(self, 'neo4j') and self.neo4j:
            try:
                enrichment['neo4j'] = await self.neo4j.lookup_entities(text)
            except Exception:
                pass
        
        # mem0: Retrieve similar memories (if available)
        if hasattr(self, 'mem0') and self.mem0:
            try:
                enrichment['mem0'] = await self.mem0.search(text, limit=3)
            except Exception:
                pass
        
        return enrichment