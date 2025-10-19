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
from typing import Dict, List, Any, Optional, Sequence
from dataclasses import dataclass

# Track optional-import fallbacks so we can surface them once logging is configured
_IMPORT_WARNINGS: List[str] = []

# Absolute package imports (works when running module from repository root)
try:
    from holoLoom.documentation.types import Query, Context, Features, MemoryShard
    from holoLoom.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you run this module from the repository root or use: python -m holoLoom.orchestrator")
    raise


# ---------------------------------------------------------------------------
# Optional dependency fallbacks
# ---------------------------------------------------------------------------

def _record_warning(msg: str) -> None:
    """Buffer fallback warnings until the logger is ready."""
    _IMPORT_WARNINGS.append(msg)


try:  # Motif detection
    from holoLoom.motif.base import create_motif_detector as _create_motif_detector
except Exception as exc:  # pragma: no cover - exercised when optional deps missing
    _record_warning(f"Using simple motif detector fallback (failed to import holoLoom.motif.base: {exc})")

    class _SimpleMotifDetector:
        """Very small async detector that tags capitalised tokens as motifs."""

        def __init__(self, mode: Optional[str] = None) -> None:
            self.mode = mode or "simple"

        async def detect(self, text: str) -> List[str]:
            if not text:
                return []
            tokens = [tok.strip(".,!?") for tok in text.split()]
            motifs = [tok.upper() for tok in tokens if tok and tok[0].isupper()]
            # Provide deterministic ordering for tests
            return motifs

    def create_motif_detector(mode: Optional[str] = None):
        return _SimpleMotifDetector(mode=mode)

else:
    create_motif_detector = _create_motif_detector


try:  # Embeddings / spectral fusion
    from holoLoom.embedding.spectral import MatryoshkaEmbeddings as _MatryoshkaEmbeddings
    from holoLoom.embedding.spectral import SpectralFusion as _SpectralFusion
except Exception as exc:  # pragma: no cover - exercised when optional deps missing
    _record_warning(f"Using simple embedding fallback (failed to import holoLoom.embedding.spectral: {exc})")

    class _SimpleVector(list):
        """List-like container that also exposes ``tolist`` for compatibility."""

        def tolist(self) -> List[float]:  # noqa: D401 - simple delegation helper
            return list(self)

    class MatryoshkaEmbeddings:
        """Minimal embedding stub that encodes text length as a single feature."""

        def __init__(self, sizes: Optional[Sequence[int]] = None, base_model_name: Optional[str] = None) -> None:
            self.sizes = list(sizes or [1])
            self.base_model_name = base_model_name or "stub"

        def encode(self, texts: Sequence[str]) -> List[_SimpleVector]:
            vectors: List[_SimpleVector] = []
            for text in texts:
                length = float(len(text))
                vectors.append(_SimpleVector([length]))
            return vectors

    class SpectralFusion:
        """Async stub returning deterministic spectral-like metrics."""

        async def features(self, graph: Any, texts: Sequence[str], embedder: Any) -> Any:
            values = [float(len(t)) % 10 for t in texts]
            return values, {"mode": "stub"}

else:
    MatryoshkaEmbeddings = _MatryoshkaEmbeddings
    SpectralFusion = _SpectralFusion


try:  # Memory retrieval
    from holoLoom.memory.base import create_retriever as _create_retriever
except Exception as exc:  # pragma: no cover - exercised when optional deps missing
    _record_warning(f"Using simple retriever fallback (failed to import holoLoom.memory.base: {exc})")

    class _SimpleRetriever:
        def __init__(self, shards: Sequence[MemoryShard], emb: Any, fusion_weights: Optional[Any] = None) -> None:
            self._shards = list(shards)

        async def search(self, query: str, k: int = 5, fast: bool = False):
            scored = []
            for idx, shard in enumerate(self._shards):
                score = 1.0 / (idx + 1)
                scored.append((shard, score))
            return scored[:k]

    def create_retriever(shards: Sequence[MemoryShard], emb: Any, fusion_weights: Optional[Any] = None):
        return _SimpleRetriever(shards=shards, emb=emb, fusion_weights=fusion_weights)

else:
    create_retriever = _create_retriever


try:  # Policy engine
    from holoLoom.policy.unified import UnifiedPolicy as _UnifiedPolicy
    from holoLoom.policy.unified import create_policy as _create_policy
except Exception as exc:  # pragma: no cover - exercised when optional deps missing
    _record_warning(f"Using simple policy fallback (failed to import holoLoom.policy.unified: {exc})")

    class UnifiedPolicy:
        """Lightweight policy stub used when torch/numpy are unavailable."""

        def __init__(self, tools: Optional[Sequence[str]] = None):
            self.tools = list(tools or ["answer", "search", "notion_write", "calc"])

        async def decide(self, query: Optional[Query] = None, features: Optional[Features] = None, context: Optional[Context] = None) -> Dict[str, Any]:
            tool_idx = 0
            if context and getattr(context, "shards", None):
                tool_idx = len(context.shards) % len(self.tools)
            elif features and getattr(features, "psi", None):
                tool_idx = len(features.psi) % len(self.tools)
            tool = self.tools[tool_idx]
            confidence = 0.5 + 0.1 * tool_idx
            return {"tool": tool, "confidence": min(confidence, 0.95)}

    def create_policy(**kwargs):
        tools = kwargs.get("tools") or ["answer", "search", "notion_write", "calc"]
        return UnifiedPolicy(tools=tools)

else:
    UnifiedPolicy = _UnifiedPolicy
    create_policy = _create_policy

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

        # Emit any deferred import warnings now that logging is ready
        for msg in _IMPORT_WARNINGS:
            self.logger.warning(msg)
        _IMPORT_WARNINGS.clear()

        # Initialize components
        self.logger.info("Initializing HoloLoom components...")
        
        # Derive execution mode string
        mode_val = getattr(cfg, 'mode', None)
        if hasattr(mode_val, 'value'):
            execution_mode = mode_val.value
        else:
            execution_mode = str(mode_val) if mode_val is not None else 'fused'
        self.execution_mode = execution_mode

        # Motif detection: allow cfg to explicitly set motif_mode, otherwise derive from exec mode
        motif_mode = getattr(cfg, 'motif_mode', None)
        if motif_mode is None:
            if execution_mode == 'bare':
                motif_mode = 'regex'
            elif execution_mode == 'fast':
                motif_mode = 'hybrid'
            else:
                motif_mode = 'hybrid'
        self.motif_mode = motif_mode
        self.motif_detector = create_motif_detector(mode=self.motif_mode)

        # Embeddings (Config uses base_model_name)
        base_model_name = getattr(cfg, 'base_model_name', None)
        self.embedder = MatryoshkaEmbeddings(
            sizes=cfg.scales,
            base_model_name=base_model_name
        )

        # Spectral fusion (enabled for fused mode)
        if execution_mode == 'fused':
            # SpectralFusion does not accept scales at init; use default params
            self.spectral_fusion = SpectralFusion()
        else:
            self.spectral_fusion = None

        # Memory retrieval: map top_k/retrieval_k and retrieval_mode
        retrieval_mode = getattr(cfg, 'retrieval_mode', None)
        if retrieval_mode is None:
            retrieval_mode = 'fused' if execution_mode == 'fused' else 'fast'
        self.retrieval_mode = retrieval_mode

        self.retriever = create_retriever(
            shards=shards,
            emb=self.embedder,
            fusion_weights=getattr(cfg, 'fusion_weights', None)
        )
        
        # Policy - create using factory to match expected constructor
        mem_dim = max(cfg.scales) if getattr(cfg, 'scales', None) else 384
        try:
            self.policy = create_policy(
                mem_dim=mem_dim,
                emb=self.embedder,
                scales=cfg.scales,
                device=None,
                n_layers=getattr(cfg, 'n_transformer_layers', 2),
                n_heads=getattr(cfg, 'n_attention_heads', 4),
                bandit_strategy=getattr(cfg, 'bandit_strategy', None) or None,
                epsilon=getattr(cfg, 'epsilon', 0.1)
            )
        except TypeError:
            # Fallback: create a SimpleUnifiedPolicy (exported as UnifiedPolicy)
            import numpy as _np
            import torch as _torch

            n_tools = getattr(cfg, 'n_tools', 4)
            # Instantiate a categorical policy to get action_probs
            policy_nn = UnifiedPolicy(
                input_dim=mem_dim,
                action_dim=n_tools,
                policy_type='categorical',
                hidden_dims=[128, 128]
            )

            # Wrap to provide async decide(features, context) -> dict
            class PolicyWrapper:
                def __init__(self, nn_module):
                    self.nn = nn_module
                    self.tools = ["answer", "search", "notion_write", "calc"]

                async def decide(self, query=None, features=None, context=None):
                    # Build input from features.psi; fallback to zeros
                    psi = getattr(features, 'psi', None)
                    if psi is None:
                        x = _torch.zeros(1, mem_dim, dtype=_torch.float32)
                    else:
                        arr = _np.array(psi, dtype=_np.float32)
                        x = _torch.tensor(arr[None, ...], dtype=_torch.float32)

                    out = self.nn.forward(x)
                    probs = out.get('action_probs') if isinstance(out, dict) else None
                    if probs is None:
                        # Fallback: use mean logits
                        probs = _torch.softmax(out.get('logits', _torch.randn(1, n_tools)), dim=-1)

                    probs_np = probs.detach().cpu().numpy()[0]
                    idx = int(_np.argmax(probs_np))
                    tool = self.tools[idx % len(self.tools)]
                    return {"tool": tool, "confidence": float(probs_np[idx])}

            self.policy = PolicyWrapper(policy_nn)
        
        # Tool executor
        self.tool_executor = ToolExecutor()
        
        self.logger.info(f"HoloLoom initialized in '{self.execution_mode}' mode")
    
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
        embeddings = self.embedder.encode([query.text])
        
        # Spectral features (if available)
        spectral = None
        if self.spectral_fusion:
            # Build a minimal empty graph for single-text spectral features
            try:
                import networkx as nx  # type: ignore
            except ImportError:
                nx = None  # type: ignore

            kg_sub = nx.MultiDiGraph() if nx else None
            spectral, _metrics = await self.spectral_fusion.features(kg_sub, [query.text], self.embedder)
        
        # Map embedding output to Features.psi (Vector) and include spectral in metrics
        try:
            psi_vec = embeddings[0].tolist() if hasattr(embeddings, '__iter__') else embeddings
        except Exception:
            psi_vec = []

        metrics = {}
        if spectral is not None:
            metrics['spectral'] = spectral.tolist() if hasattr(spectral, 'tolist') else spectral

        features = Features(
            psi=psi_vec,
            motifs=motifs if motifs else [],
            metrics=metrics,
            metadata={"query_length": len(query.text)}
        )
        
        return features
    
    async def _retrieve_context(self, query: Query, features: Features) -> Context:
        """Retrieve relevant context from memory."""
        self.logger.debug("Retrieving context...")
        
        # Use retriever to find relevant shards (Config uses retrieval_k)
        top_k = getattr(self.cfg, 'retrieval_k', None)
        if top_k is None:
            top_k = getattr(self.cfg, 'top_k', 6)

        # Determine fast flag from config
        fast_flag = getattr(self.cfg, 'fast_mode', False)

        # RetrieverMS exposes .search(query, k, fast)
        hits = await self.retriever.search(query=query.text, k=top_k, fast=fast_flag)

        # hits: List[Tuple[MemoryShard, float]]
        shards = [s for s, _ in hits]
        shard_texts = [s.text for s in shards]

        context = Context(
            shards=shards,
            hits=hits,
            shard_texts=shard_texts,
            query=query,
            features=features
        )

        self.logger.debug(f"Retrieved {len(hits)} context shards")
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
        # Ensure decision is a dict-like object
        if decision is None:
            decision = {}

        # Ensure result is a dict-like object
        if result is None:
            result = {}

        return {
            "status": "success",
            "query": query.text,
            "response": result.get("result", "No response generated"),
            "tool": decision.get("tool"),
            "confidence": decision.get("confidence", 0.0),
            "context_shards": len(context.shards) if context and hasattr(context, 'shards') else 0,
            "motifs": features.motifs if features and hasattr(features, 'motifs') else [],
            "metadata": {
                "execution_mode": getattr(self, 'execution_mode', getattr(self.cfg, 'mode', None)),
                "retrieval_mode": getattr(self, 'retrieval_mode', getattr(self.cfg, 'retrieval_mode', None)),
                "motif_mode": getattr(self, 'motif_mode', None)
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
    print(f"Tool: {response.get('tool')}")
    conf = response.get('confidence')
    try:
        print(f"Confidence: {conf:.2f}")
    except Exception:
        print(f"Confidence: {conf}")
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
