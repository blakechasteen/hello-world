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

Major Refactoring (2025-10-26):
- Consolidated type definitions (using holoLoom.documentation.types exclusively)
- Simplified policy initialization (removed fallback wrapper)
- Extracted helper methods for mode derivation and config parsing
- Improved error handling with consistent response structure
- Enhanced documentation and type hints
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Shared types (canonical location)
from HoloLoom.Documentation.types import Query, Context, Features, MemoryShard

# Core modules
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.motif.base import create_motif_detector
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings, SpectralFusion
from HoloLoom.memory.base import create_retriever
from HoloLoom.policy.unified import create_policy

logging.basicConfig(level=logging.INFO)


# ============================================================================
# Helper Functions
# ============================================================================

def derive_execution_mode(cfg: Config) -> str:
    """
    Derive execution mode string from config.

    Handles both ExecutionMode enum and string values for backward compatibility.

    Args:
        cfg: Configuration object

    Returns:
        Execution mode string ('bare', 'fast', or 'fused')
    """
    mode_val = getattr(cfg, 'mode', None)

    if mode_val is None:
        return 'fused'  # Default

    if hasattr(mode_val, 'value'):
        return mode_val.value  # ExecutionMode enum

    return str(mode_val)  # String value


def derive_motif_mode(cfg: Config, execution_mode: str) -> str:
    """
    Derive motif detection mode from config.

    Allows explicit override via cfg.motif_mode, otherwise infers from execution mode.

    Args:
        cfg: Configuration object
        execution_mode: Execution mode string

    Returns:
        Motif mode string ('regex' or 'hybrid')
    """
    motif_mode = getattr(cfg, 'motif_mode', None)

    if motif_mode is not None:
        return motif_mode

    # Infer from execution mode
    if execution_mode == 'bare':
        return 'regex'
    else:
        return 'hybrid'


def derive_retrieval_mode(cfg: Config, execution_mode: str) -> str:
    """
    Derive retrieval mode from config.

    Args:
        cfg: Configuration object
        execution_mode: Execution mode string

    Returns:
        Retrieval mode string ('fast' or 'fused')
    """
    retrieval_mode = getattr(cfg, 'retrieval_mode', None)

    if retrieval_mode is not None:
        return retrieval_mode

    # Infer from execution mode
    return 'fused' if execution_mode == 'fused' else 'fast'


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
            tool: Tool name from policy decision
            query: Original query
            context: Retrieved context

        Returns:
            Dict with execution results
        """
        self.logger.info(f"Executing tool: {tool}")

        # Tool implementations (stubs - replace with real implementations)
        tool_handlers = {
            "answer": self._handle_answer,
            "search": self._handle_search,
            "notion_write": self._handle_notion_write,
            "calc": self._handle_calc
        }

        handler = tool_handlers.get(tool, self._handle_unknown)
        return await handler(query, context)

    async def _handle_answer(self, query: Query, context: Context) -> Dict:
        """Generate an answer based on context."""
        return {
            "tool": "answer",
            "result": f"Generated answer for: {query.text}",
            "confidence": 0.85,
            "sources": len(context.shards) if context and hasattr(context, 'shards') else 0
        }

    async def _handle_search(self, query: Query, context: Context) -> Dict:
        """Perform a search."""
        return {
            "tool": "search",
            "result": "Search results based on query",
            "sources": ["source1", "source2", "source3"],
            "count": 3
        }

    async def _handle_notion_write(self, query: Query, context: Context) -> Dict:
        """Write to Notion database."""
        return {
            "tool": "notion_write",
            "result": "Successfully wrote to Notion database",
            "status": "success",
            "page_id": "mock_page_123"
        }

    async def _handle_calc(self, query: Query, context: Context) -> Dict:
        """Perform calculation."""
        return {
            "tool": "calc",
            "result": "Calculation completed",
            "value": 42,
            "expression": "mock_calculation"
        }

    async def _handle_unknown(self, query: Query, context: Context) -> Dict:
        """Handle unknown tool."""
        return {
            "tool": "unknown",
            "result": "Unknown tool",
            "error": "Tool not implemented",
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
        """
        Initialize orchestrator with configuration and memory shards.

        Args:
            cfg: Configuration object
            shards: List of memory shards for retrieval
        """
        self.cfg = cfg
        self.shards = shards
        self.logger = logging.getLogger(__name__)

        # Derive operational modes
        self.execution_mode = derive_execution_mode(cfg)
        self.motif_mode = derive_motif_mode(cfg, self.execution_mode)
        self.retrieval_mode = derive_retrieval_mode(cfg, self.execution_mode)

        self.logger.info(f"Initializing HoloLoom in '{self.execution_mode}' mode...")
        self._initialize_components()
        self.logger.info("HoloLoom initialization complete")

    def _initialize_components(self):
        """Initialize all system components based on configuration."""

        # 1. Motif Detection
        self.motif_detector = create_motif_detector(mode=self.motif_mode)
        self.logger.debug(f"Motif detector: {self.motif_mode} mode")

        # 2. Embeddings
        self.embedder = MatryoshkaEmbeddings(
            sizes=self.cfg.scales,
            base_model_name=self.cfg.base_model_name
        )
        self.logger.debug(f"Embeddings: scales={self.cfg.scales}")

        # 3. Spectral Fusion (only in fused mode)
        if self.execution_mode == 'fused':
            self.spectral_fusion = SpectralFusion()
            self.logger.debug("Spectral fusion: enabled")
        else:
            self.spectral_fusion = None
            self.logger.debug("Spectral fusion: disabled")

        # 4. Memory Retrieval
        self.retriever = create_retriever(
            shards=self.shards,
            emb=self.embedder,
            fusion_weights=self.cfg.fusion_weights
        )
        self.logger.debug(f"Retriever: {self.retrieval_mode} mode")

        # 5. Policy Engine
        mem_dim = max(self.cfg.scales)
        self.policy = create_policy(
            mem_dim=mem_dim,
            emb=self.embedder,
            scales=self.cfg.scales,
            device=None,  # Auto-detect
            n_layers=self.cfg.n_transformer_layers,
            n_heads=self.cfg.n_attention_heads,
            bandit_strategy=self.cfg.bandit_strategy,
            epsilon=self.cfg.epsilon
        )
        self.logger.debug(f"Policy: mem_dim={mem_dim}, "
                         f"layers={self.cfg.n_transformer_layers}, "
                         f"heads={self.cfg.n_attention_heads}")

        # 6. Tool Executor
        self.tool_executor = ToolExecutor()

    async def process(self, query: Query) -> Dict[str, Any]:
        """
        Main processing pipeline: Query -> Features -> Context -> Decision -> Response

        Pipeline stages:
        1. Feature Extraction: Extract motifs, embeddings, and spectral features
        2. Memory Retrieval: Find relevant context from memory
        3. Policy Decision: Select appropriate tool based on features and context
        4. Tool Execution: Execute the selected tool
        5. Response Assembly: Package results with metadata

        Args:
            query: User query

        Returns:
            Dict with response, metadata, and execution trace

        Response Structure (Success):
            {
                "status": "success",
                "query": str,
                "response": str,
                "tool": str,
                "confidence": float,
                "context_shards": int,
                "motifs": List[str],
                "metadata": {...},
                "trace": {...}
            }

        Response Structure (Error):
            {
                "status": "error",
                "query": str,
                "error": str,
                "error_type": str,
                "trace": {...}
            }
        """
        self.logger.info(f"Processing query: {query.text}")

        try:
            # Stage 1: Feature Extraction
            features = await self._extract_features(query)
            self.logger.debug(f"Features extracted: {len(features.motifs)} motifs, "
                            f"embedding_dim={len(features.psi)}")

            # Stage 2: Memory Retrieval
            context = await self._retrieve_context(query, features)
            self.logger.debug(f"Context retrieved: {len(context.shards)} shards")

            # Stage 3: Policy Decision
            decision = await self._make_decision(query, features, context)
            self.logger.debug(f"Decision: tool={decision.get('tool')}, "
                            f"confidence={decision.get('confidence', 0):.2f}")

            # Stage 4: Tool Execution
            result = await self._execute_tool(decision, query, context)
            self.logger.debug(f"Tool executed: {result.get('tool')}")

            # Stage 5: Response Assembly
            response = self._assemble_response(query, features, context, decision, result)

            self.logger.info(f"Query processed successfully: tool={decision.get('tool')}")
            return response

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return self._assemble_error_response(query, e)

    async def _extract_features(self, query: Query) -> Features:
        """
        Extract features from query.

        Extracts:
        - Motifs: Detected patterns in query text
        - Embeddings: Vector representations (Ψ)
        - Spectral: Graph and topic features (if enabled)

        Args:
            query: Query object

        Returns:
            Features object with motifs, embeddings, and metadata
        """
        self.logger.debug("Extracting features...")

        # Detect motifs
        motifs = await self.motif_detector.detect(query.text)

        # Generate embeddings
        embeddings = self.embedder.encode([query.text])

        # Extract spectral features if available
        spectral = None
        if self.spectral_fusion:
            import networkx as nx
            kg_sub = nx.MultiDiGraph()  # Empty graph for single-query spectral
            spectral, _metrics = await self.spectral_fusion.features(
                kg_sub, [query.text], self.embedder
            )

        # Convert embeddings to list format
        try:
            psi_vec = embeddings[0].tolist() if hasattr(embeddings, '__iter__') else embeddings
        except Exception:
            psi_vec = []

        # Build metrics dict
        metrics = {}
        if spectral is not None:
            metrics['spectral'] = (
                spectral.tolist() if hasattr(spectral, 'tolist') else spectral
            )

        # Construct Features object
        features = Features(
            psi=psi_vec,
            motifs=motifs if motifs else [],
            metrics=metrics,
            metadata={"query_length": len(query.text)}
        )

        return features

    async def _retrieve_context(self, query: Query, features: Features) -> Context:
        """
        Retrieve relevant context from memory.

        Uses configured retrieval mode (fast vs fused) to find relevant shards.

        Args:
            query: Original query
            features: Extracted features

        Returns:
            Context object with retrieved shards and metadata
        """
        self.logger.debug("Retrieving context...")

        # Get retrieval parameters from config
        top_k = self.cfg.retrieval_k
        fast_flag = self.cfg.fast_mode

        # Perform retrieval
        hits = await self.retriever.search(
            query=query.text,
            k=top_k,
            fast=fast_flag
        )

        # Extract shards and texts
        shards = [shard for shard, _ in hits]
        shard_texts = [shard.text for shard in shards]

        # Construct Context object
        context = Context(
            shards=shards,
            hits=hits,
            shard_texts=shard_texts,
            query=query,
            features=features
        )

        self.logger.debug(f"Retrieved {len(hits)} context shards")
        return context

    async def _make_decision(
        self,
        query: Query,
        features: Features,
        context: Context
    ) -> Dict:
        """
        Use policy to decide which tool to use.

        Args:
            query: Original query
            features: Extracted features
            context: Retrieved context

        Returns:
            Decision dict with tool, confidence, and metadata
        """
        self.logger.debug("Making policy decision...")

        # Policy.decide() takes only features and context (not query)
        action_plan = await self.policy.decide(
            features=features,
            context=context
        )

        # Convert ActionPlan to dict for backward compatibility
        decision = {
            "tool": action_plan.chosen_tool,
            "confidence": action_plan.tool_probs.get(action_plan.chosen_tool, 0.0),
            "adapter": action_plan.adapter,
            "tool_probs": action_plan.tool_probs,
            "metadata": action_plan.metadata
        }

        self.logger.debug(
            f"Policy decision: tool={decision.get('tool')}, "
            f"confidence={decision.get('confidence', 0):.2f}"
        )

        return decision

    async def _execute_tool(
        self,
        decision: Dict,
        query: Query,
        context: Context
    ) -> Dict:
        """
        Execute the tool selected by policy.

        Args:
            decision: Policy decision
            query: Original query
            context: Retrieved context

        Returns:
            Tool execution result
        """
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
        """
        Assemble final response with all metadata.

        Args:
            query: Original query
            features: Extracted features
            context: Retrieved context
            decision: Policy decision
            result: Tool execution result

        Returns:
            Complete response dict
        """
        return {
            "status": "success",
            "query": query.text,
            "response": result.get("result", "No response generated"),
            "tool": decision.get("tool"),
            "confidence": decision.get("confidence", 0.0),
            "context_shards": len(context.shards) if context and hasattr(context, 'shards') else 0,
            "motifs": features.motifs if features and hasattr(features, 'motifs') else [],
            "metadata": {
                "execution_mode": self.execution_mode,
                "retrieval_mode": self.retrieval_mode,
                "motif_mode": self.motif_mode
            },
            "trace": {
                "features": features.__dict__ if features else {},
                "decision": decision,
                "tool_result": result
            }
        }

    def _assemble_error_response(self, query: Query, error: Exception) -> Dict[str, Any]:
        """
        Assemble error response with consistent structure.

        Args:
            query: Original query
            error: Exception that occurred

        Returns:
            Error response dict
        """
        return {
            "status": "error",
            "query": query.text,
            "error": str(error),
            "error_type": type(error).__name__,
            "metadata": {
                "execution_mode": self.execution_mode,
                "retrieval_mode": self.retrieval_mode,
                "motif_mode": self.motif_mode
            },
            "trace": {
                "error_details": repr(error)
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

    # Create sample memory shards
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

    if response['status'] == 'success':
        print(f"Tool: {response.get('tool')}")
        conf = response.get('confidence', 0)
        print(f"Confidence: {conf:.2f}")
        print(f"Context Shards Used: {response['context_shards']}")
        print(f"Motifs Detected: {response['motifs']}")
        print(f"\nResponse Text:")
        print(f"  {response['response']}")
        print("\nMetadata:")
        for key, value in response['metadata'].items():
            print(f"  {key}: {value}")
    else:
        print(f"Error: {response.get('error')}")
        print(f"Error Type: {response.get('error_type')}")

    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
