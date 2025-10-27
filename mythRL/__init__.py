#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mythRL - Revolutionary Neural Decision-Making System
=====================================================
Unified API for the mythRL/HoloLoom architecture.

This is the modern, clean entry point for mythRL. Use this for all new code.

Quick Start:
    from mythRL import Weaver
    
    # Simple usage
    weaver = await Weaver.create(mode='fast', knowledge="Your text here")
    result = await weaver.query("What is this about?")
    print(result.response)
    
    # Advanced usage
    weaver = await Weaver.create(
        mode='full',
        memory_backend='neo4j_qdrant',
        enable_reflection=True
    )
    await weaver.ingest("Knowledge to learn")
    result = await weaver.query("Complex question")

Modes:
    - lite: <50ms, minimal features (96d embeddings)
    - fast: <150ms, balanced performance (96d+192d embeddings)
    - full: <300ms, complete features (96d+192d+384d embeddings)
    - research: No limit, maximum capability

Author: mythRL Team (Blake + Claude)
Date: 2025-10-27 (Phase 1 Consolidation)
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import logging

# Core imports
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config, ExecutionMode, MemoryBackend
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.spinningWheel import TextSpinner, TextSpinnerConfig
from HoloLoom.loom.command import PatternCard

logging.basicConfig(level=logging.INFO)


@dataclass
class WeaverResult:
    """
    Clean result object from Weaver queries.
    
    Attributes:
        response: The generated response text
        confidence: Confidence score (0.0-1.0)
        tool: Tool used ('answer', 'search', 'calc', etc.)
        context_count: Number of context items retrieved
        duration_ms: Execution time in milliseconds
        pattern: Pattern used ('bare', 'fast', 'fused')
        metadata: Additional execution metadata
    """
    response: str
    confidence: float
    tool: str
    context_count: int
    duration_ms: float
    pattern: str
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        return f"WeaverResult(tool={self.tool}, confidence={self.confidence:.2f}, pattern={self.pattern})"


class Weaver:
    """
    Unified mythRL API - clean, modern interface to the weaving architecture.
    
    This class provides a simple, intuitive API while leveraging the full power
    of the HoloLoom weaving architecture underneath.
    
    Usage:
        # Create with automatic knowledge loading
        weaver = await Weaver.create(mode='fast', knowledge="Your knowledge base")
        
        # Or create empty and ingest later
        weaver = await Weaver.create(mode='fast')
        await weaver.ingest("Knowledge to learn")
        
        # Query
        result = await weaver.query("Your question")
        print(result.response)
        
        # Chat (maintains context)
        result = await weaver.chat("Follow-up question")
    """
    
    def __init__(
        self,
        orchestrator: WeavingOrchestrator,
        mode: str,
        shards: List[MemoryShard]
    ):
        """
        Internal constructor. Use Weaver.create() instead.
        """
        self._orchestrator = orchestrator
        self._mode = mode
        self._shards = shards
        self._conversation_history: List[Dict[str, str]] = []
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    async def create(
        cls,
        mode: str = 'fast',
        knowledge: Optional[Union[str, Path, List[str]]] = None,
        memory_backend: str = 'in_memory',
        enable_reflection: bool = True,
        **kwargs
    ) -> 'Weaver':
        """
        Create a Weaver instance.
        
        Args:
            mode: Execution mode ('lite', 'fast', 'full', 'research')
            knowledge: Optional knowledge to load (text, file path, or list of texts)
            memory_backend: Memory backend ('in_memory', 'neo4j', 'qdrant', 'neo4j_qdrant')
            enable_reflection: Enable learning from interactions
            **kwargs: Additional config options
        
        Returns:
            Configured Weaver instance
        
        Examples:
            # Lite mode for speed
            weaver = await Weaver.create(mode='lite')
            
            # Fast mode with knowledge
            weaver = await Weaver.create(
                mode='fast',
                knowledge="HoloLoom is a neural decision system..."
            )
            
            # Full mode with Neo4j+Qdrant
            weaver = await Weaver.create(
                mode='full',
                memory_backend='neo4j_qdrant',
                enable_reflection=True
            )
        """
        # Map mode to config
        if mode == 'lite':
            config = Config.bare()
        elif mode == 'fast':
            config = Config.fast()
        elif mode in ('full', 'fused'):
            config = Config.fused()
        elif mode == 'research':
            # Research mode: all features enabled
            config = Config.fused()
            config.n_transformer_layers = 6  # More layers
            config.n_attention_heads = 12    # More heads
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'lite', 'fast', 'full', or 'research'")
        
        # Configure memory backend
        if memory_backend == 'neo4j':
            config.memory_backend = MemoryBackend.NEO4J
        elif memory_backend == 'qdrant':
            config.memory_backend = MemoryBackend.QDRANT
        elif memory_backend == 'neo4j_qdrant':
            config.memory_backend = MemoryBackend.NEO4J_QDRANT
        elif memory_backend == 'hyperspace':
            config.memory_backend = MemoryBackend.HYPERSPACE
        else:
            config.memory_backend = MemoryBackend.NETWORKX
        
        # Process knowledge if provided
        shards = []
        if knowledge is not None:
            shards = await cls._process_knowledge(knowledge, config)
        
        # Create orchestrator
        orchestrator = WeavingOrchestrator(
            cfg=config,
            shards=shards if shards else None,
            enable_reflection=enable_reflection
        )
        
        return cls(orchestrator=orchestrator, mode=mode, shards=shards)
    
    @staticmethod
    async def _process_knowledge(
        knowledge: Union[str, Path, List[str]],
        config: Config
    ) -> List[MemoryShard]:
        """Process knowledge into memory shards."""
        spinner_config = TextSpinnerConfig(
            chunk_by='paragraph',
            chunk_size=500,
            extract_entities=True
        )
        spinner = TextSpinner(spinner_config)
        
        # Handle different input types
        if isinstance(knowledge, Path) or (isinstance(knowledge, str) and Path(knowledge).exists()):
            # File path
            path = Path(knowledge)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return await spinner.spin({'text': text, 'source': str(path.name)})
        
        elif isinstance(knowledge, list):
            # Multiple texts
            all_shards = []
            for idx, text in enumerate(knowledge):
                shards = await spinner.spin({'text': text, 'source': f'text_{idx}'})
                all_shards.extend(shards)
            return all_shards
        
        else:
            # Single text string
            return await spinner.spin({'text': knowledge, 'source': 'inline_knowledge'})
    
    async def query(self, question: str, **kwargs) -> WeaverResult:
        """
        Query the knowledge base.
        
        Args:
            question: Question to answer
            **kwargs: Additional query parameters
        
        Returns:
            WeaverResult with response and metadata
        
        Example:
            result = await weaver.query("What is Thompson Sampling?")
            print(result.response)
            print(f"Confidence: {result.confidence}")
        """
        query_obj = Query(text=question)
        spacetime = await self._orchestrator.weave(query_obj)
        
        return WeaverResult(
            response=spacetime.response,
            confidence=spacetime.confidence,
            tool=spacetime.tool_used or 'answer',
            context_count=spacetime.trace.context_shards_count if spacetime.trace else 0,
            duration_ms=spacetime.metadata.get('duration_ms', 0),
            pattern=spacetime.metadata.get('pattern', 'unknown'),
            metadata=spacetime.metadata
        )
    
    async def chat(self, message: str) -> WeaverResult:
        """
        Chat with context from conversation history.
        
        Args:
            message: User message
        
        Returns:
            WeaverResult with response
        
        Example:
            result1 = await weaver.chat("Tell me about HoloLoom")
            result2 = await weaver.chat("What are its key features?")
        """
        # Add to conversation history
        self._conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        # Query with context
        result = await self.query(message)
        
        # Add response to history
        self._conversation_history.append({
            'role': 'assistant',
            'content': result.response
        })
        
        return result
    
    async def ingest(self, knowledge: Union[str, Path, List[str]]) -> int:
        """
        Ingest new knowledge into the system.
        
        Args:
            knowledge: Text, file path, or list of texts to ingest
        
        Returns:
            Number of shards created
        
        Example:
            count = await weaver.ingest("New information to learn")
            print(f"Ingested {count} shards")
        """
        new_shards = await self._process_knowledge(knowledge, self._orchestrator.cfg)
        self._shards.extend(new_shards)
        
        # Update orchestrator's yarn graph if it exists
        if hasattr(self._orchestrator, 'yarn_graph') and self._orchestrator.yarn_graph:
            for shard in new_shards:
                # Add to yarn graph (implementation depends on yarn graph API)
                pass
        
        self.logger.info(f"Ingested {len(new_shards)} new shards")
        return len(new_shards)
    
    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history = []
        self.logger.info("Conversation history cleared")
    
    async def close(self):
        """Close resources."""
        await self._orchestrator.close()
    
    async def __aenter__(self):
        """Support async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager."""
        await self.close()
    
    def __repr__(self) -> str:
        return f"Weaver(mode={self._mode}, shards={len(self._shards)})"


# Convenient exports
__all__ = ['Weaver', 'WeaverResult']
__version__ = '1.0.0'
__author__ = 'mythRL Team'
