"""
HoloLoom Orchestrator
=====================
The central "shuttle" that weaves together all components.

This is the only module that imports from other HoloLoom modules.
All cross-module coordination happens here.

Architecture:
- Composes motif detection, embedding, memory, and policy
- Implements the main processing pipeline: Query → Features → Decision → Response
- Handles execution modes (bare, fast, fused)
- Manages async coordination and error handling

Philosophy:
The orchestrator is the "shuttle" moving across the "warp threads" (modules),
weaving them into finished "fabric" (responses).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Import from our clean modular architecture!
from types import Query, Response, Features, Context, MemoryShard
from config import Config, ExecutionMode, BanditStrategy

from motif.base import create_motif_detector
from embedding.spectral import MatryoshkaEmbeddings, SpectralFusion
from memory.cache import create_memory_manager
from memory.graph import KG, KGEdge, extract_entities_simple
from policy.unified import create_policy


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
        
        if tool == "answer":
            # Direct answer from context
            return {
                "type": "answer",
                "direct": True,
                "context_used": len(context.hits) > 0
            }
        
        elif tool == "search":
            # Would call external search API
            return {
                "type": "search",
                "search_performed": True,
                "results_count": 5
            }
        
        elif tool == "notion_write":
            # Would write to Notion
            return {
                "type": "notion_write",
                "written": True,
                "page_id": "mock_page_123"
            }
        
        elif tool == "calc":
            # Would perform calculation
            return {
                "type": "calc",
                "computed": True,
                "result": "mock_calculation"
            }
        
        else:
            return {"type": "unknown", "error": f"Unknown tool: {tool}"}


# ============================================================================
# Main Orchestrator
# ============================================================================

@dataclass
class HoloLoomOrchestrator:
    """
    Main orchestrator that coordinates all HoloLoom components.
    
    This is the "shuttle" - the only module that knows about all other modules.
    It weaves together motif detection, embeddings, memory, and policy into
    a cohesive decision-making pipeline.
    
    Execution Modes:
    - Bare: Minimal processing (regex motifs, no spectral features, fast retrieval)
    - Fast: Balanced (hybrid motifs, spectral features, fast retrieval)
    - Fused: Full quality (all features, multi-scale retrieval, neural policy)
    """
    
    cfg: Config
    shards: List[MemoryShard]
    logger: Optional[logging.Logger] = None
    
    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing HoloLoom in {self.cfg.mode} mode")
        
        # Initialize embeddings
        self.emb = MatryoshkaEmbeddings(
            sizes=self.cfg.scales,
            base_model_name=self.cfg.base_model_name
        )
        
        # Initialize motif detector (mode-dependent)
        force_regex = (self.cfg.mode == ExecutionMode.BARE)
        self.motif_detector = create_motif_detector(
            mode="bare" if force_regex else "hybrid"
        )
        
        # Initialize spectral fusion
        self.spectral = SpectralFusion()
        
        # Initialize memory manager
        self.memory = create_memory_manager(
            shards=self.shards,
            emb=self.emb,
            fusion_weights=self.cfg.fusion_weights,
            root=self.cfg.memory_path or "data"
        )
        
        # Initialize knowledge graph
        self.kg = KG()
        
        # Initialize policy
        self.mem_dim = max(self.cfg.scales) if not self.cfg.fast_mode else min(self.cfg.scales)
        self.policy = create_policy(
            mem_dim=self.mem_dim,
            emb=self.emb,
            scales=self.cfg.scales,
            n_layers=2,
            n_heads=4,
            bandit_strategy=self.cfg.bandit_strategy,
            epsilon=self.cfg.epsilon
        )
        
        # Initialize tool executor
        self.tools = ToolExecutor()
        
        # Execution settings
        self.fast_retrieval = (self.cfg.mode != ExecutionMode.FUSED)
        
        self.logger.info("✓ HoloLoom initialized successfully")
    
    async def process(self, query: Query) -> Response:
        """
        Main processing pipeline: Query → Response
        
        Pipeline stages:
        1. Feature extraction (motifs + Ψ)
        2. Context retrieval (memory + KG)
        3. Policy decision (tool selection)
        4. Tool execution
        5. Response generation
        
        Args:
            query: Input query
            
        Returns:
            Structured response with context and metadata
        """
        try:
            # Timeout for full pipeline
            async with asyncio.timeout(5.0):
                return await self._optimal_path(query)
        except asyncio.TimeoutError:
            self.logger.warning("Pipeline timeout, falling back to fast path")
            return await self._fast_path(query)
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            return self._safe_fallback(query, e)
    
    async def _optimal_path(self, query: Query) -> Response:
        """
        Full pipeline with all features enabled.
        
        This is the main "weaving" process - all modules work together.
        """
        self.logger.debug(f"Processing query: {query.text[:50]}...")
        
        # Stage 1: Extract motifs (async)
        motif_task = asyncio.create_task(self.motif_detector.detect(query.text))
        
        # Stage 2: Build KG context
        entities = extract_entities_simple(query.text)
        for entity in entities[:3]:  # Limit to avoid huge graphs
            self.kg.add_edge(KGEdge(
                src=entity,
                dst="query",
                type="MENTIONS",
                weight=1.0
            ))
        kg_sub = self.kg.subgraph_for_entities(entities[:3])
        
        # Stage 3: Retrieve memory
        context = await self.memory.retrieve(query, kg_sub, fast=self.fast_retrieval)
        
        # Wait for motifs
        motif_objects = await motif_task
        motifs = [m.pattern for m in motif_objects]
        
        # Stage 4: Extract features
        if self.cfg.mode == ExecutionMode.BARE:
            # Bare mode: skip expensive spectral computation
            import numpy as np
            psi = np.zeros(6)
            metrics = {'coherence': 0.0, 'fiedler': 0.0, 'topic_var': 0.0}
        else:
            # Fast/Fused modes: compute spectral features
            psi, metrics = await self.spectral.features(
                kg_sub, 
                context.shard_texts, 
                self.emb
            )
        
        features = Features(
            psi=psi,
            motifs=motifs,
            metrics=metrics,
            confidence=self._compute_confidence(motifs, metrics, context)
        )
        
        # Stage 5: Policy decision
        plan = await self.policy.decide(features, context)
        
        # Stage 6: Execute tool
        if plan.chosen_tool == 'answer':
            results = {
                'direct': True,
                'chosen_tool': plan.chosen_tool,
                'probs': plan.tool_probs
            }
        else:
            results = await self.tools.execute(plan.chosen_tool, query, context)
        
        # Stage 7: Generate response
        response = self._generate_response(query, features, context, plan, results)
        
        # Stage 8: Persist to memory (async, non-blocking)
        asyncio.create_task(self.memory.persist(query, results, features))
        
        self.logger.debug(f"✓ Query processed: tool={plan.chosen_tool}, confidence={features.confidence:.3f}")
        
        return response
    
    async def _fast_path(self, query: Query) -> Response:
        """
        Fast path for timeouts or low-complexity queries.
        
        Skips spectral features and uses cached motifs only.
        """
        self.logger.debug("Using fast path")
        
        # Minimal feature extraction
        motif_objects = await self.motif_detector.detect(query.text)
        motifs = [m.pattern for m in motif_objects]
        
        features = Features.empty()
        features.motifs = motifs
        
        context = Context.empty()
        
        # Simple plan
        from types import ActionPlan
        plan = ActionPlan(
            chosen_tool="answer",
            adapter="general",
            tool_probs={"answer": 1.0, "search": 0.0, "notion_write": 0.0, "calc": 0.0}
        )
        
        results = await self.tools.execute(plan.chosen_tool, query, context)
        
        return self._generate_response(query, features, context, plan, results)
    
    def _safe_fallback(self, query: Query, error: Exception) -> Response:
        """
        Safety fallback for catastrophic failures.
        
        Returns a valid response with error metadata.
        """
        self.logger.error(f"Fallback triggered: {error}")
        
        return Response(
            query=query.text,
            motifs=[],
            entities=[],
            hits=[],
            psi=[],
            psi_metrics={},
            tool_probs={},
            chosen_tool="error",
            adapter="none",
            metadata={"error": str(error)}
        )
    
    def _compute_confidence(
        self,
        motifs: List[str],
        metrics: Dict,
        context: Context
    ) -> float:
        """
        Compute overall confidence score.
        
        Combines:
        - Number of motifs detected
        - Spectral coherence
        - Context relevance
        """
        motif_score = min(len(motifs) * 0.15, 0.4)
        coherence_score = metrics.get('coherence', 0.0) * 0.3
        relevance_score = context.relevance * 0.3
        
        return float(min(motif_score + coherence_score + relevance_score, 1.0))
    
    def _generate_response(
        self,
        query: Query,
        features: Features,
        context: Context,
        plan,
        results: Dict
    ) -> Response:
        """
        Generate structured response from pipeline outputs.
        
        Packages everything into a Response object for the client.
        """
        # Summarize hits
        hits_summary = [
            (h[0].id, h[0].text[:60] + ("…" if len(h[0].text) > 60 else ""))
            for h in context.hits
        ]
        
        # Extract entities
        entities = extract_entities_simple(query.text)
        
        return Response(
            query=query.text,
            motifs=features.motifs,
            entities=entities[:10],
            hits=hits_summary,
            psi=features.psi.tolist() if hasattr(features.psi, 'tolist') else list(features.psi),
            psi_metrics=features.metrics,
            tool_probs=plan.tool_probs,
            chosen_tool=plan.chosen_tool,
            adapter=plan.adapter,
            metadata={
                'confidence': features.confidence,
                'mode': self.cfg.mode.value,
                'fast_retrieval': self.fast_retrieval,
                **results
            }
        )
    
    async def shutdown(self):
        """Graceful shutdown - flush memory queues."""
        self.logger.info("Shutting down HoloLoom")
        await self.memory.shutdown()
        self.logger.info("✓ Shutdown complete")


# ============================================================================
# Demo / Example Usage
# ============================================================================

async def main():
    """Demo showing how to use the orchestrator."""
    import json
    
    print("=" * 70)
    print("HoloLoom Orchestrator Demo")
    print("=" * 70)
    
    # Create sample memory shards
    demo_shards = [
        MemoryShard(
            id="s1",
            text="Explain multi-head self-attention using a joke with double meaning.",
            episode="MC_2025-10-10",
            entities=["self-attention"],
            motifs=["setup→twist"]
        ),
        MemoryShard(
            id="s2",
            text="ColBERT uses late interaction where query and doc tokens interact via MaxSim.",
            episode="MC_2025-10-10",
            entities=["ColBERT"],
            motifs=["contrast"]
        ),
        MemoryShard(
            id="s3",
            text="Farm low-ABV compute: pin hot shards in RAM and use PQ/IVF compression.",
            episode="FarmOps",
            entities=["Farm"],
            motifs=["goal→constraint"]
        ),
    ]
    
    # Create config
    cfg = Config(
        scales=[96, 192, 384],
        fusion_weights={96: 0.25, 192: 0.35, 384: 0.40},
        memory_path="data/demo",
        mode=ExecutionMode.FUSED,
        fast_mode=False
    )
    
    # Initialize orchestrator
    orchestrator = HoloLoomOrchestrator(cfg=cfg, shards=demo_shards)
    
    # Process queries
    queries = [
        "How does multi-head attention resolve a pun compared to ColBERT?",
        "Design a frost plan for fig trees and low-ABV compute scheduling.",
        "Why did my cider get sulfur and how does SNA help?",
    ]
    
    for q_text in queries:
        query = Query(text=q_text)
        
        print(f"\n{'=' * 70}")
        print(f"QUERY: {q_text}")
        print('=' * 70)
        
        response = await orchestrator.process(query)
        
        print(f"\n📊 Response:")
        print(f"  Tool: {response.chosen_tool}")
        print(f"  Adapter: {response.adapter}")
        print(f"  Confidence: {response.metadata.get('confidence', 0):.3f}")
        print(f"\n🧵 Motifs: {response.motifs}")
        print(f"\n🎯 Entities: {response.entities}")
        print(f"\n📈 Ψ metrics:")
        for k, v in response.psi_metrics.items():
            print(f"    {k}: {v:.3f}")
        print(f"\n🔧 Tool probabilities:")
        for tool, prob in response.tool_probs.items():
            print(f"    {tool}: {prob:.3f}")
    
    # Shutdown
    await orchestrator.shutdown()
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())