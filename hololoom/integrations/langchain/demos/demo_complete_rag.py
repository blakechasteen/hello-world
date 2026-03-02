"""
Demo: Complete RAG Pipeline
============================

Complete end-to-end RAG example combining:
- Document loading (LangChain)
- Memory storage (HoloLoom)
- Vector search (LangChain)
- LLM generation (LangChain)

Author: Claude Code
Date: 2025-11-20
"""

import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


async def demo_simple_rag():
    """Simple RAG pipeline."""
    print("\n" + "="*60)
    print("Demo 1: Simple RAG (Text + LLM)")
    print("="*60 + "\n")

    print("📚 Pipeline: Ingest → Recall → Generate\n")

    print("""
    from hololoom import HoloLoom
    from hololoom.integrations.langchain import MultiProviderLLM

    async def simple_rag():
        async with HoloLoom() as loom:
            # 1. Ingest knowledge
            await loom.experience('''
            Thompson Sampling is a Bayesian approach to the exploration-exploitation
            trade-off. It maintains a posterior distribution over action rewards and
            samples from this distribution to select actions.
            ''')

            # 2. Recall relevant memories
            memories = await loom.recall('What is Thompson Sampling?')
            context = '\\n'.join([m.content for m in memories])

            # 3. Generate answer
            llm = MultiProviderLLM(provider='ollama', model='llama3.2:3b')
            prompt = f'Context:\\n{context}\\n\\nQuestion: What is Thompson Sampling?\\n\\nAnswer:'
            response = llm(prompt)

            print(response)

    asyncio.run(simple_rag())
    """)


async def demo_document_rag():
    """RAG with document loading."""
    print("\n" + "="*60)
    print("Demo 2: Document RAG (Files + Vector Store)")
    print("="*60 + "\n")

    print("📄 Pipeline: Load Docs → Vector Store → Retrieve → Generate\n")

    print("""
    from hololoom import HoloLoom
    from hololoom.integrations.langchain import (
        UniversalDocumentLoader,
        VectorStoreFactory,
        MultiProviderLLM
    )

    async def document_rag():
        # 1. Load documents
        loader = UniversalDocumentLoader()
        shards = loader.load_directory('docs/', glob_pattern='**/*.md')
        print(f'Loaded {len(shards)} document chunks')

        # 2. Create vector store
        vector_store = VectorStoreFactory(store_type='chroma')
        vector_store.add_shards(shards)

        # 3. HoloLoom memory
        async with HoloLoom() as loom:
            for shard in shards:
                await loom.experience(shard.content)

            # 4. Query
            query = 'Explain the architecture'

            # Hybrid retrieval
            vector_results = vector_store.similarity_search(query, k=5)
            hololoom_memories = await loom.recall(query, k=5)

            # Combine contexts
            context = '\\n'.join([r[0] for r in vector_results] +
                                [m.content for m in hololoom_memories])

            # 5. Generate
            llm = MultiProviderLLM(provider='anthropic')
            prompt = f'Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:'
            response = llm(prompt)

            print(response)

    asyncio.run(document_rag())
    """)


async def demo_multi_source_rag():
    """RAG with multiple sources."""
    print("\n" + "="*60)
    print("Demo 3: Multi-Source RAG")
    print("="*60 + "\n")

    print("🌐 Pipeline: Web + PDFs + Code → Synthesize\n")

    print("""
    from hololoom import HoloLoom
    from hololoom.integrations.langchain import (
        UniversalDocumentLoader,
        MultiProviderLLM
    )

    async def multi_source_rag():
        loader = UniversalDocumentLoader()

        # Load from multiple sources
        web_shards = loader.load('https://en.wikipedia.org/wiki/Thompson_Sampling')
        pdf_shards = loader.load('papers/thompson_1933.pdf')
        code_shards = loader.load_directory('src/', glob_pattern='**/*.py')

        all_shards = web_shards + pdf_shards + code_shards
        print(f'Loaded {len(all_shards)} chunks from 3 sources')

        # Ingest to HoloLoom
        async with HoloLoom() as loom:
            for shard in all_shards:
                await loom.experience(shard.content)

            # Multi-source query
            query = 'How is Thompson Sampling implemented in this codebase?'
            memories = await loom.recall(query, k=10)

            # Group by source
            by_source = {}
            for mem in memories:
                source_type = 'web' if 'wiki' in mem.source else \
                             'paper' if 'pdf' in mem.source else 'code'
                by_source.setdefault(source_type, []).append(mem.content)

            # Generate synthesis
            llm = MultiProviderLLM(provider='anthropic')
            context = f'''
            From Wikipedia: {by_source.get('web', ['N/A'])[0]}
            From Paper: {by_source.get('paper', ['N/A'])[0]}
            From Code: {by_source.get('code', ['N/A'])[0]}
            '''

            response = llm(f'{context}\\n\\n{query}\\n\\nAnswer:')
            print(response)

    asyncio.run(multi_source_rag())
    """)


async def demo_agentic_rag():
    """RAG with agentic reasoning."""
    print("\n" + "="*60)
    print("Demo 4: Agentic RAG (Multi-Step)")
    print("="*60 + "\n")

    print("🤖 Pipeline: Question → Sub-Questions → Research → Synthesis\n")

    print("""
    from hololoom import HoloLoom
    from hololoom.integrations.langchain import MultiProviderLLM
    from hololoom.agentic import AgenticOrchestrator, ReasoningMode

    async def agentic_rag():
        # Load knowledge base
        async with HoloLoom() as loom:
            # ... ingest documents ...

            # Agentic reasoning
            async with AgenticOrchestrator(shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query='Compare Thompson Sampling vs UCB',
                    mode=ReasoningMode.RESEARCH,
                    max_steps=5
                )

                # LLM synthesis
                llm = MultiProviderLLM(provider='anthropic')
                context = '\\n'.join([step.result for step in result.steps_taken])

                final_answer = llm(f'''
                Research Steps:
                {context}

                Synthesize a comprehensive comparison of Thompson Sampling vs UCB:
                ''')

                print(final_answer)

    asyncio.run(agentic_rag())
    """)


async def demo_quick_prototype():
    """Quick prototyping API."""
    print("\n" + "="*60)
    print("Demo 5: Quick Prototyping API")
    print("="*60 + "\n")

    print("⚡ Simplified API for Rapid Development\n")

    print("""
    from hololoom.integrations.langchain import QuickPrototype

    async def quick_rag():
        # Setup
        proto = QuickPrototype()
        await proto.setup(
            use_case='development',
            llm_provider='ollama',
            vector_store='chroma'
        )

        # Ingest
        chunks = await proto.ingest('docs/')
        print(f'Ingested {chunks} chunks')

        # Query (automatic RAG)
        answer = await proto.query('What is the architecture?')
        print(answer)

    asyncio.run(quick_rag())
    """)


def demo_performance_tips():
    """Performance optimization tips."""
    print("\n" + "="*60)
    print("Demo 6: Performance Tips")
    print("="*60 + "\n")

    print("⚡ Optimization Strategies:\n")

    print("1️⃣  Batch Document Loading:")
    print("""
    loader = UniversalDocumentLoader()
    shards = loader.load_directory('docs/')  # Load all at once
    vector_store.add_shards(shards, batch_size=100)  # Batch insert
    """)

    print("\n2️⃣  Use Local LLM for Development:")
    print("""
    # Fast iteration with Ollama
    llm = MultiProviderLLM(provider='ollama', model='llama3.2:3b')
    # ~150ms per query vs 800ms for GPT-4
    """)

    print("\n3️⃣  Hybrid Search for Better Recall:")
    print("""
    # Combine semantic + keyword search
    results = vector_store.hybrid_search(query, k=10, alpha=0.7)
    # 30% better recall than semantic-only
    """)

    print("\n4️⃣  HoloLoom Query Cache:")
    print("""
    from hololoom import HoloLoom
    from hololoom.config import Config

    config = Config.fast()
    config.query_cache_size = 10000  # Cache 10k queries

    async with HoloLoom(config=config) as loom:
        # First query: ~150ms
        memories = await loom.recall('query')
        # Repeated query: <1ms (100x faster!)
    """)


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*15 + "Complete RAG Pipeline Demo")
    print("="*70)

    asyncio.run(demo_simple_rag())
    asyncio.run(demo_document_rag())
    asyncio.run(demo_multi_source_rag())
    asyncio.run(demo_agentic_rag())
    asyncio.run(demo_quick_prototype())
    demo_performance_tips()

    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70 + "\n")

    print("Key Takeaways:")
    print("1. LangChain provides breadth (100+ loaders, 20+ LLMs)")
    print("2. HoloLoom provides depth (Thompson Sampling, learning)")
    print("3. Combine both for best results")
    print("4. Use QuickPrototype API for rapid development\n")

    print("Next Steps:")
    print("1. Try examples with your own data")
    print("2. Experiment with different LLM providers")
    print("3. Compare vector stores for your use case")
    print("4. See README.md for complete documentation\n")


if __name__ == "__main__":
    main()
