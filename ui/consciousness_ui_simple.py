"""
Consciousness Stack Web UI - Standalone Demo Version

Simplified Gradio interface that demonstrates the consciousness stack concepts
without requiring the full HoloLoom import chain.

Run: python ui/consciousness_ui_simple.py
"""

import gradio as gr
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


# Minimal dataclass replications
@dataclass
class AwarenessContext:
    """Simplified awareness context"""
    confidence: float
    uncertainty: float
    domain: str
    subdomain: str
    phrase_type: str
    is_question: bool
    cache_status: str = "MISS"
    knowledge_gap: bool = False


@dataclass
class MemoryNode:
    """Simplified memory node"""
    content: str
    relevance: float
    composite_score: float
    retrieval_depth: int
    timestamp: str
    item_id: str = ""  # Track original item ID for graph traversal


@dataclass
class PackedContext:
    """Simplified packed context"""
    total_tokens: int
    elements_included: int
    elements_compressed: int
    elements_excluded: int
    avg_importance: float
    min_importance: float
    packing_time_ms: float
    formatted_text: str
    compression_stats: Dict


# Demo memory backend
class DemoMemoryBackend:
    """Demo memory with quantum computing knowledge"""
    
    def __init__(self):
        self.knowledge_base = {
            'quantum_1': {
                'id': 'quantum_1',
                'content': 'Quantum entanglement is a physical phenomenon where pairs or groups of particles interact in ways such that the quantum state of each particle cannot be described independently.',
                'relevance': 0.95,
                'timestamp': datetime.now().isoformat(),
                'related': ['computing_1', 'teleport_1']
            },
            'computing_1': {
                'id': 'computing_1',
                'content': 'Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform computations that would be infeasible on classical computers.',
                'relevance': 0.92,
                'timestamp': datetime.now().isoformat(),
                'related': ['quantum_1', 'algorithms_1', 'applications_1']
            },
            'teleport_1': {
                'id': 'teleport_1',
                'content': 'Quantum teleportation is a technique for transferring quantum states from one location to another, using entanglement and classical communication.',
                'relevance': 0.88,
                'timestamp': datetime.now().isoformat(),
                'related': ['quantum_1', 'communication_1']
            },
            'algorithms_1': {
                'id': 'algorithms_1',
                'content': 'Quantum algorithms like Shor\'s factoring algorithm and Grover\'s search algorithm demonstrate exponential speedups over classical algorithms.',
                'relevance': 0.85,
                'timestamp': datetime.now().isoformat(),
                'related': ['computing_1', 'applications_1']
            },
            'applications_1': {
                'id': 'applications_1',
                'content': 'Quantum computing applications include cryptography, drug discovery, financial modeling, optimization problems, and machine learning.',
                'relevance': 0.90,
                'timestamp': datetime.now().isoformat(),
                'related': ['computing_1', 'algorithms_1']
            },
            'communication_1': {
                'id': 'communication_1',
                'content': 'Quantum communication protocols enable secure transmission of information using principles like quantum key distribution (QKD).',
                'relevance': 0.82,
                'timestamp': datetime.now().isoformat(),
                'related': ['teleport_1']
            },
            'hardware_1': {
                'id': 'hardware_1',
                'content': 'Quantum hardware challenges include maintaining coherence, error correction, scaling qubit counts, and operating at near-absolute-zero temperatures.',
                'relevance': 0.86,
                'timestamp': datetime.now().isoformat(),
                'related': ['computing_1']
            }
        }
    
    async def retrieve_with_threshold(self, query: str, threshold: float, limit: int = 10):
        """Retrieve items matching query above threshold"""
        query_lower = query.lower()
        results = []
        for item in self.knowledge_base.values():
            # Simple keyword matching
            if any(word in item['content'].lower() for word in query_lower.split()):
                if item['relevance'] >= threshold:
                    results.append(item)
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:limit]
    
    async def get_related(self, item_id: str, limit: int = 10):
        """Get related items for graph traversal"""
        if item_id not in self.knowledge_base:
            return []
        item = self.knowledge_base[item_id]
        results = []
        for rid in item.get('related', [])[:limit]:
            if rid in self.knowledge_base:
                results.append(self.knowledge_base[rid])
        return results


# Consciousness Stack Components
class SimpleAwareness:
    """Simplified awareness layer"""
    
    async def analyze(self, query: str) -> AwarenessContext:
        """Analyze query and generate awareness context"""
        query_lower = query.lower()
        
        # Domain detection
        domain = "general"
        subdomain = "unknown"
        if "quantum" in query_lower:
            domain = "science"
            subdomain = "quantum_physics"
        elif "computer" in query_lower or "computing" in query_lower:
            domain = "technology"
            subdomain = "computer_science"
        
        # Structure analysis
        is_question = query.strip().endswith("?") or any(
            q in query_lower for q in ["what", "how", "why", "when", "where", "who"]
        )
        phrase_type = "question" if is_question else "statement"
        
        # Confidence based on length and structure
        confidence = min(0.95, 0.5 + len(query.split()) * 0.05)
        
        return AwarenessContext(
            confidence=confidence,
            uncertainty=1.0 - confidence,
            domain=domain,
            subdomain=subdomain,
            phrase_type=phrase_type,
            is_question=is_question,
            cache_status="MISS",
            knowledge_gap=len(query.split()) < 3
        )


class SimpleMemoryFusion:
    """Simplified memory fusion with multipass crawling"""
    
    def __init__(self, backend: DemoMemoryBackend, max_passes: int = 3):
        self.backend = backend
        self.max_passes = max_passes
        self.thresholds = [0.60, 0.75, 0.85, 0.90][:max_passes]
    
    async def retrieve_with_fusion(self, query: str, max_results: int = 10) -> List[MemoryNode]:
        """Multipass retrieval with graph crawling"""
        seen_ids = set()
        all_nodes = []
        
        for pass_idx, threshold in enumerate(self.thresholds):
            # Initial retrieval or expansion
            if pass_idx == 0:
                items = await self.backend.retrieve_with_threshold(query, threshold, max_results)
            else:
                # Expand from previous pass using actual item IDs
                items = []
                for node in all_nodes:
                    if node.item_id:  # Use tracked item ID for graph traversal
                        related = await self.backend.get_related(node.item_id, limit=3)
                        items.extend(related)
            
            # Process items
            for item in items:
                item_id = item.get('id', item['content'][:20])
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    
                    # Calculate composite score (relevance + depth penalty)
                    depth_weight = 1.0 / (1.0 + pass_idx * 0.3)
                    composite_score = item['relevance'] * depth_weight
                    
                    if composite_score >= threshold:
                        node = MemoryNode(
                            content=item['content'],
                            relevance=item['relevance'],
                            composite_score=composite_score,
                            retrieval_depth=pass_idx,
                            timestamp=item['timestamp'],
                            item_id=item_id  # Store ID for graph traversal
                        )
                        all_nodes.append(node)
            
            # Early stopping if no new items
            if not items:
                break
        
        # Sort by composite score and limit
        all_nodes.sort(key=lambda n: n.composite_score, reverse=True)
        return all_nodes[:max_results]


class SimpleContextPacker:
    """Simplified context packer"""
    
    def __init__(self, token_budget: int = 4000):
        self.token_budget = token_budget
    
    async def pack(self, query: str, awareness: AwarenessContext, memories: List[MemoryNode]) -> PackedContext:
        """Pack context with token budget"""
        start = time.perf_counter()
        
        # Reserve tokens
        query_tokens = len(query.split()) * 1.3
        response_tokens = self.token_budget * 0.25
        available = self.token_budget - query_tokens - response_tokens
        
        # Build context
        elements = []
        total_tokens = 0
        compressed_count = 0
        
        # Add awareness
        awareness_text = f"[Domain: {awareness.domain}/{awareness.subdomain}, Confidence: {awareness.confidence:.0%}]"
        awareness_tokens = len(awareness_text.split()) * 1.3
        elements.append(awareness_text)
        total_tokens += awareness_tokens
        
        # Add memories
        for memory in memories:
            memory_tokens = len(memory.content.split()) * 1.3
            
            if total_tokens + memory_tokens < available:
                elements.append(f"- [{memory.composite_score:.2f}] {memory.content}")
                total_tokens += memory_tokens
            elif memory.composite_score > 0.8:
                # Compress high-importance memory
                compressed = memory.content[:100] + "..."
                compressed_tokens = len(compressed.split()) * 1.3
                if total_tokens + compressed_tokens < available:
                    elements.append(f"- [{memory.composite_score:.2f}] {compressed}")
                    total_tokens += compressed_tokens
                    compressed_count += 1
        
        formatted = "\n".join(elements)
        pack_time = (time.perf_counter() - start) * 1000
        
        importance_scores = [m.composite_score for m in memories if m.composite_score > 0]
        
        return PackedContext(
            total_tokens=int(total_tokens),
            elements_included=len(elements) - compressed_count,
            elements_compressed=compressed_count,
            elements_excluded=len(memories) - len(elements) + 1,
            avg_importance=sum(importance_scores) / len(importance_scores) if importance_scores else 0,
            min_importance=min(importance_scores) if importance_scores else 0,
            packing_time_ms=pack_time,
            formatted_text=formatted,
            compression_stats={
                "included_uncompressed": len(elements) - compressed_count,
                "included_compressed": compressed_count,
                "excluded": len(memories) - len(elements) + 1
            }
        )


class SimpleDualStream:
    """Simplified dual-stream generator"""
    
    async def generate(self, query: str, context: str) -> tuple[str, str, float]:
        """Generate internal and external responses"""
        start = time.perf_counter()
        
        # Internal reasoning
        internal = f"""Looking at: "{query}"
Got {len(context)} chars of context
Strategy: Use what we know to give a helpful answer
Confidence: Pretty good based on the context quality"""
        
        # External response (more casual, less formal)
        external = f"""Hey! So about {query.lower().rstrip('?')} - here's what I found:

{context[:400]}...

That's the gist of what I know about this. Let me know if you want me to dive deeper into any part!"""
        
        gen_time = (time.perf_counter() - start) * 1000
        
        return internal, external, gen_time


# Initialize components
memory_backend = DemoMemoryBackend()
awareness = SimpleAwareness()


async def process_query_async(
    query: str,
    complexity: str,
    use_fusion: bool,
    max_memories: int,
    token_budget: int
):
    """Process query through consciousness stack"""
    
    if not query.strip():
        return "Please enter a query.", "", "", "", "", "", "{}"
    
    results = {
        'query': query,
        'complexity': complexity,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Awareness Analysis
    awareness_ctx = await awareness.analyze(query)
    
    awareness_output = f"""## 🔍 Awareness Analysis

**Confidence**: {awareness_ctx.confidence:.2%}
**Uncertainty**: {awareness_ctx.uncertainty:.2%}
**Cache Status**: {awareness_ctx.cache_status}
**Knowledge Gap**: {'Yes' if awareness_ctx.knowledge_gap else 'No'}

**Structure**:
- Type: {awareness_ctx.phrase_type}
- Is Question: {awareness_ctx.is_question}

**Pattern Recognition**:
- Domain: {awareness_ctx.domain}/{awareness_ctx.subdomain}
"""
    
    results['awareness'] = {
        'confidence': awareness_ctx.confidence,
        'domain': awareness_ctx.domain
    }
    
    # 2. Memory Fusion
    fusion_output = ""
    memories = []
    
    if use_fusion:
        max_passes = {"LITE": 1, "FAST": 2, "FULL": 3, "RESEARCH": 4}.get(complexity, 3)
        fusion = SimpleMemoryFusion(memory_backend, max_passes=max_passes)
        
        memory_nodes = await fusion.retrieve_with_fusion(query, max_results=max_memories)
        
        fusion_output = f"""## 🕷️ Memory Fusion

**Retrieved**: {len(memory_nodes)} memories
**Max Depth**: {max((n.retrieval_depth for n in memory_nodes), default=0)}
**Avg Score**: {sum(n.composite_score for n in memory_nodes) / len(memory_nodes) if memory_nodes else 0:.3f}
**Passes**: {max_passes}

**Top Memories**:
"""
        for i, node in enumerate(memory_nodes[:5], 1):
            fusion_output += f"{i}. [Depth {node.retrieval_depth}, Score {node.composite_score:.3f}] {node.content[:80]}...\n"
        
        memories = memory_nodes
        results['fusion'] = {
            'count': len(memory_nodes),
            'max_depth': max((n.retrieval_depth for n in memory_nodes), default=0)
        }
    else:
        fusion_output = "## 🕷️ Memory Fusion\n\n*Disabled*"
        results['fusion'] = None
    
    # 3. Context Packing
    packer = SimpleContextPacker(token_budget=token_budget)
    packed = await packer.pack(query, awareness_ctx, memories)
    
    packing_output = f"""## 📦 Context Packing

**Total Tokens**: {packed.total_tokens}/{int(token_budget * 0.65)}
**Elements**: {packed.elements_included} included, {packed.elements_compressed} compressed, {packed.elements_excluded} excluded
**Avg Importance**: {packed.avg_importance:.2%}
**Min Importance**: {packed.min_importance:.2%}
**Packing Time**: {packed.packing_time_ms:.2f}ms

**Compression Stats**: {json.dumps(packed.compression_stats, indent=2)}
"""
    
    results['packing'] = {
        'total_tokens': packed.total_tokens,
        'elements_included': packed.elements_included
    }
    
    # 4. LLM Context
    llm_context = f"""# Query
{query}

# Awareness
{awareness_output}

# Knowledge
{packed.formatted_text}
"""
    
    # 5. Dual-Stream Generation
    generator = SimpleDualStream()
    internal, external, gen_time = await generator.generate(query, packed.formatted_text)
    
    generation_output = f"""## 🎭 Dual-Stream Generation

**Generation Time**: {gen_time:.2f}ms

### Internal Reasoning
```
{internal}
```

### External Response
```
{external}
```
"""
    
    # Performance Summary
    total_time = packed.packing_time_ms + gen_time
    
    performance_output = f"""## ⚡ Performance Summary

**Total Time**: {total_time:.2f}ms
- Awareness: <1ms
- Memory Fusion: <2ms
- Context Packing: {packed.packing_time_ms:.2f}ms
- Generation: {gen_time:.2f}ms

**Efficiency**:
- Token Usage: {(packed.total_tokens / (token_budget * 0.65) * 100):.1f}%
- Quality: {packed.avg_importance:.2%} importance
- Compression: {(packed.elements_compressed / max(packed.elements_included, 1) * 100):.0f}% compressed
"""
    
    return awareness_output, fusion_output, packing_output, llm_context, generation_output, performance_output, json.dumps(results, indent=2)


def process_query(query, complexity, use_fusion, max_memories, token_budget):
    """Sync wrapper"""
    return asyncio.run(process_query_async(query, complexity, use_fusion, max_memories, token_budget))


# Build Gradio UI
with gr.Blocks(title="Consciousness Stack UI", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 Consciousness Stack Interactive UI
    
    Explore the complete consciousness infrastructure with real-time visualizations.
    
    **Features**: Compositional Awareness • Memory Fusion • Smart Context Packing • Dual-Stream Generation
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input Configuration")
            
            query_input = gr.Textbox(
                label="Query",
                placeholder="Enter your question...",
                lines=3,
                value="What are the applications of quantum computing?"
            )
            
            complexity_input = gr.Radio(
                choices=["LITE", "FAST", "FULL", "RESEARCH"],
                label="Complexity Level",
                value="FULL",
                info="Controls multipass depth and thresholds"
            )
            
            use_fusion_input = gr.Checkbox(
                label="Enable Memory Fusion",
                value=True,
                info="Use multipass graph crawling"
            )
            
            max_memories_input = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="Max Memories",
                info="Maximum memories to retrieve"
            )
            
            token_budget_input = gr.Slider(
                minimum=1000,
                maximum=8000,
                value=4000,
                step=500,
                label="Token Budget",
                info="Total tokens available for context"
            )
            
            submit_btn = gr.Button("🚀 Process Query", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("### 📊 Raw Results (JSON)")
            json_output = gr.JSON(label="Complete Results")
    
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Pipeline Stages")
            
            with gr.Tab("1️⃣ Awareness"):
                awareness_output = gr.Markdown()
            
            with gr.Tab("2️⃣ Memory Fusion"):
                fusion_output = gr.Markdown()
            
            with gr.Tab("3️⃣ Context Packing"):
                packing_output = gr.Markdown()
            
            with gr.Tab("4️⃣ LLM Context"):
                llm_context_output = gr.Code(label="Formatted for LLM", language="markdown")
            
            with gr.Tab("5️⃣ Generation"):
                generation_output = gr.Markdown()
            
            with gr.Tab("⚡ Performance"):
                performance_output = gr.Markdown()
    
    # Connect button
    submit_btn.click(
        fn=process_query,
        inputs=[query_input, complexity_input, use_fusion_input, max_memories_input, token_budget_input],
        outputs=[awareness_output, fusion_output, packing_output, llm_context_output, generation_output, performance_output, json_output]
    )
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### 💡 Example Queries")
    
    gr.Examples(
        examples=[
            ["What are the applications of quantum computing?", "FULL", True, 10, 4000],
            ["Explain quantum entanglement", "FAST", True, 8, 3000],
            ["How does quantum teleportation work?", "LITE", False, 5, 2000],
            ["What are the challenges in building quantum computers?", "RESEARCH", True, 15, 6000],
        ],
        inputs=[query_input, complexity_input, use_fusion_input, max_memories_input, token_budget_input],
    )


if __name__ == "__main__":
    print("\n" + "🧠" * 40)
    print("CONSCIOUSNESS STACK WEB UI".center(80))
    print("🧠" * 40 + "\n")
    print("Starting Gradio server...")
    print("Open your browser to: http://localhost:7860\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
