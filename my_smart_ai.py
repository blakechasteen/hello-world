"""
YOUR Smart AI - Load your own data and query it
Run: PYTHONPATH=. python my_smart_ai.py
"""
import asyncio
from pathlib import Path
from HoloLoom.rag import SimpleRAG
from HoloLoom.config import Config

async def main():
    print("🧠 YOUR Smart AI - HoloLoom RAG System\n")

    # Configuration
    config = Config.fast()  # Fast mode = good balance

    # Create RAG system
    async with SimpleRAG(config=config) as rag:
        print("✓ RAG system initialized\n")

        # ========================================
        # STEP 1: INGEST YOUR DATA
        # ========================================
        print("📥 STEP 1: Ingesting your data...\n")

        # Example 1: Ingest text directly
        texts = [
            """
            HoloLoom is a neural decision-making system that combines multi-scale
            embeddings, knowledge graphs, and reinforcement learning. It uses
            Thompson Sampling for exploration/exploitation balance.
            """,
            """
            The weaving orchestrator implements a 9-step cycle:
            1. Loom Command (pattern selection)
            2. Chrono Trigger (temporal control)
            3. Yarn Graph (memory selection)
            4. Resonance Shed (feature extraction)
            5. DotPlasma (feature fusion)
            6. Warp Space (continuous manifold)
            7. Convergence Engine (decision collapse)
            8. Tool Execution (action)
            9. Reflection Buffer (learning)
            """,
            """
            SimpleRAG provides zero-config RAG with 4 reasoning modes:
            - DIRECT: Fast single-pass answers
            - VERIFY: Answer + verification
            - RESEARCH: Multi-query exploration
            - PLAN_EXECUTE: Goal decomposition
            """,
        ]

        for i, text in enumerate(texts, 1):
            await rag.ingest(text)
            print(f"  ✓ Ingested document {i}/{len(texts)}")

        # Example 2: Ingest from files (CUSTOMIZE THIS!)
        # Uncomment and add your file paths:
        """
        my_files = [
            "path/to/your/notes.txt",
            "path/to/your/docs.md",
            "path/to/your/research.txt",
        ]

        for file_path in my_files:
            if Path(file_path).exists():
                content = Path(file_path).read_text(encoding='utf-8')
                await rag.ingest(content)
                print(f"  ✓ Ingested {file_path}")
        """

        print(f"\n✓ Ingestion complete!\n")

        # ========================================
        # STEP 2: QUERY YOUR DATA
        # ========================================
        print("🔍 STEP 2: Querying your AI...\n")

        queries = [
            "What is HoloLoom?",
            "How does the weaving cycle work?",
            "What reasoning modes are available?",
        ]

        for query in queries:
            print(f"❓ Query: {query}")

            result = await rag.query(
                query,
                mode="verify"  # Use VERIFY mode for quality answers
            )

            print(f"💡 Answer ({result.confidence:.0%} confidence):")
            print(f"   {result.response}\n")
            print(f"   Sources: {len(result.sources)} retrieved")
            print(f"   Mode: {result.reasoning_mode}")
            print("-" * 80 + "\n")

        # ========================================
        # STEP 3: INTERACTIVE MODE
        # ========================================
        print("\n🎮 STEP 3: Interactive mode - Ask anything!\n")
        print("Type your questions (or 'quit' to exit):\n")

        while True:
            try:
                user_query = input("You: ").strip()

                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if not user_query:
                    continue

                # Query with verification
                result = await rag.query(user_query, mode="verify")

                print(f"\n🤖 AI ({result.confidence:.0%} confidence):")
                print(f"{result.response}\n")

                if result.confidence < 0.7:
                    print("⚠️  Low confidence - answer may be uncertain\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    print("=" * 80)
    asyncio.run(main())
    print("=" * 80)
