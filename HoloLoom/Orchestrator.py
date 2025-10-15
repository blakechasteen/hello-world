#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Orchestrator
=====================
The central "shuttle" that weaves together all components.

This is the only module that imports from other HoloLoom modules.
All cross-module coordination happens here.

Architecture:
- Composes motif detection, embedding, memory, and policy
- Implements the main processing pipeline: Query -> Features -> Decision -> Response
- Handles execution modes (bare, fast, fused)
- Manages async coordination and error handling

Philosophy:
The orchestrator is the "shuttle" moving across the "warp threads" (modules),
weaving them into finished "fabric" (responses).
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Absolute package imports (works when running module from repository root)
try:
    from holoLoom.documentation.types import Query, Context, Features, MemoryShard
    from holoLoom.config import Config
    from holoLoom.motif.base import create_motif_detector
    from holoLoom.embedding.spectral import MatryoshkaEmbeddings, SpectralFusion
    from holoLoom.memory.base import create_retriever
    from holoLoom.policy.unified import UnifiedPolicy
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you run this module from the repository root or use: python -m holoLoom.orchestrator")
    raise

logging.basicConfig(level=logging.INFO)


# ============================================================================
# Tool Execution
# ============================================================================

class ToolExecutor:
    """
    Executes tools based on policy decisions.
    
    In a real system, this would call actual APIs, databases, etc.
    For now, it's a stub that returns structured results.
    """
    
    def __init__(self):
        self.tools = ["answer", "search", "notion_write", "calc"]
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, tool: str, query: Query, context: Context) -> Dict:
        """
        Execute a tool based on the policy decision.
        
        Args:
            tool: Tool name from ActionPlan
            query: Original query
            context: Retrieved context
            
        Returns:
            Dict with execution results
        """
        self.logger.info(f"Executing tool: {tool}")
        
        # Stub implementation - returns mock results based on tool type
        if tool == "answer":
            return {
                "tool": "answer",
                "result": f"Generated answer for: {query.text}",
                "confidence": 0.85,
                "sources": len(context.shards) if context and hasattr(context, 'shards') else 0
            }
        elif tool == "search":
            return {
                "tool": "search",
                "result": "Search results based on query",
                "sources": ["source1", "source2", "source3"],
                "count": 3
            }
        elif tool == "notion_write":
            return {
                "tool": "notion_write",
                "result": "Successfully wrote to Notion database",
                "status": "success",
                "page_id": "mock_page_123"
            }
        elif tool == "calc":
            return {
                "tool": "calc",
                "result": "Calculation completed",
                "value": 42,
                "expression": "mock_calculation"
            }
        else:
            return {
                "tool": tool,
                "result": "Unknown tool",
                "error": f"Tool '{tool}' not implemented",
                "status": "error"
            }


# ============================================================================
# Main Orchestrator
# ============================================================================

class HoloLoomOrchestrator:
    """
    The main orchestrator that coordinates all HoloLoom components.
    
    This is the "shuttle" that weaves together:
    - Motif detection (patterns in queries)
    - Embedding (semantic representation)
    - Memory (context retrieval)
    - Policy (decision making)
    - Tools (action execution)
    
    Usage:
        config = Config.fused()
        orchestrator = HoloLoomOrchestrator(cfg=config, shards=memory_shards)
        response = await orchestrator.process(Query(text="What is Thompson Sampling?"))
    """
    
    def __init__(self, cfg: Config, shards: List[MemoryShard]):
        self.cfg = cfg
        self.shards = shards
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing HoloLoom components...")
        
        # Motif detection
        self.motif_detector = create_motif_detector(mode=cfg.motif_mode)
        
        # Embeddings
        self.embedder = MatryoshkaEmbeddings(
            sizes=cfg.scales,
            base_model_name=cfg.base_encoder
        )
        
        # Spectral fusion (if in fused mode)
        if cfg.execution_mode == "fused":
            self.spectral_fusion = SpectralFusion(scales=cfg.scales)
        else:
            self.spectral_fusion = None
        
        # Memory retrieval
        self.retriever = create_retriever(
            mode=cfg.retrieval_mode,
            shards=shards,
            embedder=self.embedder
        )
        
        # Policy
        self.policy = UnifiedPolicy(config=cfg)
        
        # Tool executor
        self.tool_executor = ToolExecutor()
        
        self.logger.info(f"HoloLoom initialized in '{cfg.execution_mode}' mode")
    
    async def process(self, query: Query) -> Dict[str, Any]:
        """
        Main processing pipeline: Query -> Features -> Context -> Decision -> Response
        
        Args:
            query: User query
            
        Returns:
            Dict with response, metadata, and execution trace
        """
        self.logger.info(f"Processing query: {query.text}")
        
        try:
            # Stage 1: Feature Extraction
            features = await self._extract_features(query)
            
            # Stage 2: Memory Retrieval
            context = await self._retrieve_context(query, features)
            
            # Stage 3: Policy Decision
            decision = await self._make_decision(query, features, context)
            
            # Stage 4: Tool Execution
            result = await self._execute_tool(decision, query, context)
            
            # Stage 5: Response Assembly
            response = self._assemble_response(query, features, context, decision, result)
            
            self.logger.info(f"Query processed successfully: tool={decision.get('tool')}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "query": query.text
            }
    
    async def _extract_features(self, query: Query) -> Features:
        """Extract features from query (motifs, embeddings, spectral)."""
        self.logger.debug("Extracting features...")
        
        # Detect motifs
        motifs = await self.motif_detector.detect(query.text)
        
        # Generate embeddings
        embeddings = await self.embedder.encode([query.text])
        
        # Spectral features (if available)
        spectral = None
        if self.spectral_fusion:
            spectral = await self.spectral_fusion.compute([query.text])
        
        features = Features(
            motifs=[m.pattern for m in motifs] if motifs else [],
            embeddings=embeddings,
            spectral=spectral,
            metadata={"query_length": len(query.text)}
        )
        
        return features
    
    async def _retrieve_context(self, query: Query, features: Features) -> Context:
        """Retrieve relevant context from memory."""
        self.logger.debug("Retrieving context...")
        
        # Use retriever to find relevant shards
        retrieved_shards = await self.retriever.retrieve(
            query=query.text,
            top_k=self.cfg.top_k
        )
        
        context = Context(
            shards=retrieved_shards,
            query=query,
            features=features
        )
        
        self.logger.debug(f"Retrieved {len(retrieved_shards)} context shards")
        return context
    
    async def _make_decision(self, query: Query, features: Features, context: Context) -> Dict:
        """Use policy to decide which tool to use."""
        self.logger.debug("Making policy decision...")
        
        # Policy decides which tool to use
        decision = await self.policy.decide(
            query=query,
            features=features,
            context=context
        )
        
        self.logger.debug(f"Policy decision: tool={decision.get('tool')}, confidence={decision.get('confidence')}")
        return decision
    
    async def _execute_tool(self, decision: Dict, query: Query, context: Context) -> Dict:
        """Execute the tool selected by policy."""
        tool = decision.get("tool", "answer")
        return await self.tool_executor.execute(tool, query, context)
    
    def _assemble_response(
        self, 
        query: Query, 
        features: Features, 
        context: Context, 
        decision: Dict, 
        result: Dict
    ) -> Dict[str, Any]:
        """Assemble final response with all metadata."""
        return {
            "status": "success",
            "query": query.text,
            "response": result.get("result", "No response generated"),
            "tool": decision.get("tool"),
            "confidence": decision.get("confidence", 0.0),
            "context_shards": len(context.shards) if context and hasattr(context, 'shards') else 0,
            "motifs": features.motifs if features and hasattr(features, 'motifs') else [],
            "metadata": {
                "execution_mode": self.cfg.execution_mode,
                "retrieval_mode": self.cfg.retrieval_mode,
                "motif_mode": self.cfg.motif_mode
            },
            "trace": {
                "features": features.__dict__ if features else {},
                "decision": decision,
                "tool_result": result
            }
        }


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

async def main():
    """Example usage of HoloLoomOrchestrator."""
    print("\n" + "="*80)
    print("HoloLoom Orchestrator - Test Run")
    print("="*80 + "\n")
    
    # Create some sample memory shards
    shards = [
        MemoryShard(
            id="shard_001",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            episode="docs",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["ALGORITHM", "OPTIMIZATION"]
        ),
        MemoryShard(
            id="shard_002",
            text="The algorithm balances exploration and exploitation by sampling from posterior distributions.",
            episode="docs",
            entities=["exploration", "exploitation", "posterior"],
            motifs=["ALGORITHM", "PROBABILITY"]
        ),
        MemoryShard(
            id="shard_003",
            text="Hive Jodi has 8 frames of brood and is very active with goldenrod flow.",
            episode="inspection_2025_10_13",
            entities=["Hive Jodi", "brood", "goldenrod"],
            motifs=["HIVE_INSPECTION", "SEASONAL"]
        )
    ]
    
    # Create config
    config = Config.fused()
    
    # Create orchestrator
    print("Initializing orchestrator...")
    orchestrator = HoloLoomOrchestrator(cfg=config, shards=shards)
    print("Orchestrator ready!\n")
    
    # Process a query
    query = Query(text="What is Thompson Sampling?")
    print(f"Processing query: '{query.text}'")
    print("-" * 80)
    
    response = await orchestrator.process(query)
    
    # Print response
    print("\n" + "="*80)
    print("RESPONSE")
    print("="*80)
    print(f"Status: {response['status']}")
    print(f"Query: {response['query']}")
    print(f"Tool: {response['tool']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Context Shards Used: {response['context_shards']}")
    print(f"Motifs Detected: {response['motifs']}")
    print(f"\nResponse Text:")
    print(f"  {response['response']}")
    print("\nMetadata:")
    for key, value in response['metadata'].items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())