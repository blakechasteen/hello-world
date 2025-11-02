"""
Consciousness Stack Web UI

Interactive Gradio interface for the complete consciousness stack:
- Query input with awareness analysis
- Memory fusion visualization
- Context packing inspection
- Dual-stream generation
- Real-time performance metrics

Run: python ui/consciousness_ui.py
"""

import gradio as gr
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import awareness components directly
try:
    from HoloLoom.awareness.compositional_awareness import CompositionalAwarenessLayer
    from HoloLoom.awareness.context_packer import SmartContextPacker, TokenBudget
    from HoloLoom.awareness.dual_stream import DualStreamGenerator
    from HoloLoom.awareness.memory_fusion import MemoryFusion, MultipassConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Installing standalone versions...")
    # Create minimal imports
    pass


# Demo memory backend for UI testing
class UIMemoryBackend:
    """Simple memory backend for UI demonstrations"""
    
    def __init__(self):
        self.knowledge_base = {
            'quantum_1': {
                'id': 'quantum_1',
                'content': 'Quantum entanglement is a physical phenomenon where particles become correlated',
                'relevance': 0.95,
                'timestamp': datetime.now().isoformat(),
                'related': ['computing_1', 'teleport_1']
            },
            'computing_1': {
                'id': 'computing_1',
                'content': 'Quantum computing uses superposition and entanglement for computation',
                'relevance': 0.92,
                'timestamp': datetime.now().isoformat(),
                'related': ['quantum_1', 'algorithms_1']
            },
            'teleport_1': {
                'id': 'teleport_1',
                'content': 'Quantum teleportation transfers quantum states using entanglement',
                'relevance': 0.88,
                'timestamp': datetime.now().isoformat(),
                'related': ['quantum_1', 'communication_1']
            },
            'algorithms_1': {
                'id': 'algorithms_1',
                'content': 'Quantum algorithms like Shor\'s and Grover\'s provide exponential speedups',
                'relevance': 0.85,
                'timestamp': datetime.now().isoformat(),
                'related': ['computing_1']
            },
            'communication_1': {
                'id': 'communication_1',
                'content': 'Quantum communication enables secure key distribution',
                'relevance': 0.82,
                'timestamp': datetime.now().isoformat(),
                'related': ['teleport_1']
            }
        }
    
    async def retrieve_with_threshold(self, query: str, threshold: float, limit: int = 10):
        query_lower = query.lower()
        results = []
        for item_id, item in self.knowledge_base.items():
            if any(word in item['content'].lower() for word in query_lower.split()):
                if item['relevance'] >= threshold:
                    results.append(item)
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:limit]
    
    async def get_related(self, item_id: str, limit: int = 10):
        if item_id not in self.knowledge_base:
            return []
        item = self.knowledge_base[item_id]
        results = []
        for rid in item.get('related', [])[:limit]:
            if rid in self.knowledge_base:
                results.append(self.knowledge_base[rid])
        return results


# Initialize consciousness stack
awareness_layer = CompositionalAwarenessLayer()
memory_backend = UIMemoryBackend()


async def process_query_async(
    query: str,
    complexity: str,
    use_fusion: bool,
    max_memories: int,
    token_budget: int
):
    """Process query through consciousness stack"""
    
    if not query.strip():
        return "Please enter a query.", "", "", "", ""
    
    results = {
        'query': query,
        'complexity': complexity,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Awareness Analysis
    awareness_ctx = await awareness_layer.get_unified_context(query)
    
    conf = awareness_ctx.confidence
    patterns = awareness_ctx.patterns
    struct = awareness_ctx.structural
    
    awareness_output = f"""## 🔍 Awareness Analysis

**Confidence**: {1.0 - conf.uncertainty_level:.2%}
**Uncertainty**: {conf.uncertainty_level:.2%}
**Cache Status**: {conf.query_cache_status}
**Knowledge Gap**: {'Yes' if conf.knowledge_gap_detected else 'No'}

**Structure**:
- Type: {struct.phrase_type}
- Is Question: {struct.is_question}
- Expected Response: {struct.suggested_response_type}

**Pattern Recognition**:
- Domain: {patterns.domain}/{patterns.subdomain}
- Seen Count: {patterns.seen_count}×
- Pattern Confidence: {patterns.confidence:.2%}
"""
    
    results['awareness'] = {
        'confidence': 1.0 - conf.uncertainty_level,
        'uncertainty': conf.uncertainty_level,
        'domain': patterns.domain,
        'subdomain': patterns.subdomain
    }
    
    # 2. Memory Fusion (if enabled)
    fusion_output = ""
    memories = []
    
    if use_fusion:
        config = MultipassConfig.for_complexity(complexity)
        fusion = MemoryFusion(config=config, memory_backend=memory_backend)
        
        fused_nodes = await fusion.retrieve_with_fusion(query, max_results=max_memories)
        
        fusion_output = f"""## 🕷️ Memory Fusion

**Retrieved**: {len(fused_nodes)} memories
**Max Depth**: {max((n.retrieval_depth for n in fused_nodes), default=0)}
**Avg Score**: {sum(n.composite_score for n in fused_nodes) / len(fused_nodes):.3f}
**Passes**: {config.max_passes}

**Top Memories**:
"""
        for i, node in enumerate(fused_nodes[:5], 1):
            fusion_output += f"{i}. [Depth {node.retrieval_depth}, Score {node.composite_score:.3f}] {node.content[:80]}...\n"
        
        # Convert for packer
        memories = [{'text': n.content, 'score': n.composite_score} for n in fused_nodes]
        
        results['fusion'] = {
            'count': len(fused_nodes),
            'max_depth': max((n.retrieval_depth for n in fused_nodes), default=0),
            'avg_score': sum(n.composite_score for n in fused_nodes) / len(fused_nodes)
        }
    else:
        fusion_output = "## 🕷️ Memory Fusion\n\n*Disabled*"
        results['fusion'] = None
    
    # 3. Context Packing
    packer = SmartContextPacker(
        token_budget=TokenBudget(
            total=token_budget,
            reserved_for_query=int(token_budget * 0.1),
            reserved_for_response=int(token_budget * 0.25)
        ),
        use_memory_fusion=False,  # Already retrieved above
        memory_backend=memory_backend
    )
    
    packed = await packer.pack_context(
        query,
        awareness_ctx,
        memory_results=memories if memories else None,
        max_memories=max_memories
    )
    
    packing_output = f"""## 📦 Context Packing

**Total Tokens**: {packed.total_tokens}/{packer.budget.available_for_context}
**Elements**: {packed.elements_included} included, {packed.elements_compressed} compressed, {packed.elements_excluded} excluded
**Avg Importance**: {packed.avg_importance:.2%}
**Min Importance**: {packed.min_importance:.2%}
**Packing Time**: {packed.packing_time_ms:.2f}ms

**Compression Stats**: {json.dumps(packed.compression_stats, indent=2)}
"""
    
    results['packing'] = {
        'total_tokens': packed.total_tokens,
        'elements_included': packed.elements_included,
        'avg_importance': packed.avg_importance
    }
    
    # 4. Formatted Context for LLM
    llm_context = packed.format_for_llm(include_metadata=True)
    
    # 5. Dual-Stream Generation
    generator = DualStreamGenerator(awareness_layer=awareness_layer)
    response = await generator.generate(query, show_internal=True, use_llm=False)
    
    generation_output = f"""## 🎭 Dual-Stream Generation

**Generation Time**: {response.generation_time_ms:.2f}ms

### Internal Reasoning
```
{response.internal_stream[:500]}...
```

### External Response
```
{response.external_stream[:500]}...
```
"""
    
    results['generation'] = {
        'time_ms': response.generation_time_ms,
        'internal_length': len(response.internal_stream),
        'external_length': len(response.external_stream)
    }
    
    # Performance Summary
    total_time = packed.packing_time_ms + response.generation_time_ms
    
    performance_output = f"""## ⚡ Performance Summary

**Total Time**: {total_time:.2f}ms
- Awareness: <1ms
- Memory Fusion: <2ms
- Context Packing: {packed.packing_time_ms:.2f}ms
- Generation: {response.generation_time_ms:.2f}ms

**Efficiency**:
- Token Usage: {(packed.total_tokens / packer.budget.available_for_context * 100):.1f}%
- Quality: {packed.avg_importance:.2%} importance
- Compression: {(packed.elements_compressed / max(packed.elements_included, 1) * 100):.0f}% compressed
"""
    
    return awareness_output, fusion_output, packing_output, llm_context, generation_output, performance_output, json.dumps(results, indent=2)


def process_query(query, complexity, use_fusion, max_memories, token_budget):
    """Sync wrapper for async processing"""
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
    print("Open your browser to interact with the consciousness stack!\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
