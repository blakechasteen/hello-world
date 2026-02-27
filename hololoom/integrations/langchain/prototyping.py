"""
HoloLoom + LangChain Quick Prototyping CLI
===========================================

Interactive CLI for rapid prototyping with HoloLoom + LangChain integration.

Features:
- Quick setup wizard
- Interactive query mode
- Document ingestion (100+ formats)
- LLM provider selection
- Vector store selection
- Example workflows

Author: Claude Code
Date: 2025-11-20
"""

import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

try:
    from .document_loaders import UniversalDocumentLoader
    from .llm_providers import MultiProviderLLM, create_best_available_llm
    from .vector_stores import VectorStoreFactory, get_recommended_store
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


class HoloLoomCLI:
    """
    Interactive CLI for prototyping with HoloLoom + LangChain.

    Provides quick setup, document ingestion, and interactive querying.
    """

    def __init__(self):
        """Initialize CLI."""
        self.hololoom = None
        self.llm = None
        self.vector_store = None
        self.documents = []

    async def start(self):
        """Start interactive CLI."""
        print("\n" + "="*60)
        print("HoloLoom + LangChain Prototyping CLI")
        print("="*60 + "\n")

        if not INTEGRATION_AVAILABLE:
            print("❌ LangChain integration not available.")
            print("   Install with: pip install langchain langchain-community")
            return

        # Setup wizard
        await self.setup_wizard()

        # Main loop
        await self.interactive_loop()

    async def setup_wizard(self):
        """Interactive setup wizard."""
        print("📋 Setup Wizard\n")

        # 1. HoloLoom configuration
        print("1️⃣  HoloLoom Configuration")
        use_case = self._prompt(
            "Use case (development/production/research)",
            default="development"
        )

        # Import HoloLoom
        try:
            from hololoom import hololoom
            from hololoom.config import Config

            if use_case == "production":
                config = Config.fused()
            elif use_case == "research":
                config = Config.fused()
                config.enable_alignment = True
            else:
                config = Config.fast()

            self.hololoom = HoloLoom(config=config)
            print(f"   ✅ HoloLoom initialized ({use_case} mode)\n")
        except ImportError:
            print("   ⚠️  HoloLoom not available, using standalone mode\n")

        # 2. LLM provider
        print("2️⃣  LLM Provider")
        print("   Available: anthropic, openai, cohere, ollama (local)")

        provider = self._prompt(
            "Provider (or 'auto' for best available)",
            default="auto"
        )

        if provider == "auto":
            self.llm = create_best_available_llm()
            print(f"   ✅ Using best available LLM\n")
        else:
            model = self._prompt(f"Model name (or Enter for default)")
            if model:
                self.llm = MultiProviderLLM(provider=provider, model=model)
            else:
                self.llm = MultiProviderLLM(provider=provider)
            print(f"   ✅ {provider.upper()} initialized\n")

        # 3. Vector store
        print("3️⃣  Vector Store")
        recommended = get_recommended_store(use_case)
        print(f"   Recommended for {use_case}: {recommended}")

        store_type = self._prompt(
            "Store type (qdrant/chroma/faiss/pinecone)",
            default=recommended
        )

        try:
            self.vector_store = VectorStoreFactory(store_type=store_type)
            print(f"   ✅ {store_type.upper()} vector store initialized\n")
        except Exception as e:
            print(f"   ⚠️  Vector store error: {e}")
            print(f"   Falling back to in-memory\n")
            self.vector_store = VectorStoreFactory(store_type="faiss")

        print("✅ Setup complete!\n")

    async def interactive_loop(self):
        """Main interactive loop."""
        print("🚀 Interactive Mode")
        print("   Commands: query, ingest, stats, examples, help, exit\n")

        while True:
            command = input("hololoom> ").strip().lower()

            if not command:
                continue

            if command == "exit":
                print("\n👋 Goodbye!")
                break

            elif command == "help":
                self.show_help()

            elif command.startswith("query "):
                query_text = command[6:].strip()
                await self.handle_query(query_text)

            elif command.startswith("ingest "):
                path = command[7:].strip()
                await self.handle_ingest(path)

            elif command == "stats":
                self.show_stats()

            elif command == "examples":
                self.show_examples()

            else:
                print(f"❓ Unknown command: {command}")
                print("   Type 'help' for available commands\n")

    async def handle_query(self, query_text: str):
        """Handle query command."""
        print(f"\n🔍 Query: {query_text}\n")

        try:
            # Use HoloLoom if available
            if self.hololoom:
                memories = await self.hololoom.recall(query_text)
                context = "\n".join([m.content for m in memories[:3]])

                # Generate response with LLM
                prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
                response = self.llm(prompt)

                print(f"💡 Answer: {response}\n")

                # Show sources
                print("📚 Sources:")
                for i, mem in enumerate(memories[:3], 1):
                    print(f"   {i}. {mem.source}")
                print()

            else:
                # Standalone mode (no HoloLoom)
                response = self.llm(query_text)
                print(f"💡 Answer: {response}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")

    async def handle_ingest(self, path: str):
        """Handle document ingestion."""
        print(f"\n📄 Ingesting: {path}\n")

        try:
            # Load documents
            loader = UniversalDocumentLoader()
            shards = loader.load(path)

            print(f"   Found {len(shards)} document chunks")

            # Add to HoloLoom if available
            if self.hololoom:
                for shard in shards:
                    await self.hololoom.experience(shard.content)
                print(f"   ✅ Added to HoloLoom memory")

            # Add to vector store
            if self.vector_store:
                ids = self.vector_store.add_shards(shards)
                print(f"   ✅ Added to vector store ({len(ids)} vectors)")

            self.documents.extend(shards)
            print()

        except Exception as e:
            print(f"❌ Error: {e}\n")

    def show_stats(self):
        """Show statistics."""
        print("\n📊 Statistics\n")

        print(f"Documents ingested: {len(self.documents)}")

        if self.hololoom:
            metrics = self.hololoom.get_metrics()
            print(f"Active memories: {metrics['activation']['active_nodes']}")
            print(f"Memory coherence: {metrics['coherence']['global_coherence']:.2f}")

        if self.llm:
            stats = self.llm.get_usage_stats()
            print(f"LLM provider: {stats['provider']}")
            print(f"Total tokens: {stats['total_tokens']}")

        print()

    def show_examples(self):
        """Show example workflows."""
        print("\n📖 Example Workflows\n")

        print("1️⃣  Quick Q&A:")
        print("   hololoom> ingest docs/research.pdf")
        print("   hololoom> query What are the main findings?\n")

        print("2️⃣  Multi-document research:")
        print("   hololoom> ingest docs/")
        print("   hololoom> query Compare approaches across papers\n")

        print("3️⃣  Code analysis:")
        print("   hololoom> ingest src/")
        print("   hololoom> query Explain the architecture\n")

        print("4️⃣  Web research:")
        print("   hololoom> ingest https://example.com/article")
        print("   hololoom> query Summarize this article\n")

    def show_help(self):
        """Show help."""
        print("\n📘 Available Commands\n")
        print("  query <text>    - Ask a question")
        print("  ingest <path>   - Ingest documents (file/directory/URL)")
        print("  stats           - Show usage statistics")
        print("  examples        - Show example workflows")
        print("  help            - Show this help")
        print("  exit            - Exit CLI\n")

    def _prompt(self, message: str, default: Optional[str] = None) -> str:
        """Prompt user for input."""
        if default:
            response = input(f"   {message} [{default}]: ").strip()
            return response if response else default
        else:
            return input(f"   {message}: ").strip()


# Convenience function

def quick_start():
    """
    Quick start function for immediate prototyping.

    Usage:
        >>> from hololoom.integrations.langchain import quick_start
        >>> quick_start()
    """
    cli = HoloLoomCLI()
    asyncio.run(cli.start())


# Programmatic API

class QuickPrototype:
    """
    Programmatic API for quick prototyping.

    Example:
        >>> proto = QuickPrototype()
        >>> await proto.setup(use_case="development")
        >>> await proto.ingest("docs/")
        >>> answer = await proto.query("What are the main topics?")
    """

    def __init__(self):
        self.hololoom = None
        self.llm = None
        self.vector_store = None

    async def setup(
        self,
        use_case: str = "development",
        llm_provider: str = "auto",
        vector_store: str = "chroma"
    ):
        """
        Setup prototype environment.

        Args:
            use_case: Use case (development/production/research)
            llm_provider: LLM provider (auto/openai/anthropic/ollama)
            vector_store: Vector store type
        """
        # HoloLoom
        try:
            from hololoom import hololoom
            from hololoom.config import Config

            config = Config.fast() if use_case == "development" else Config.fused()
            self.hololoom = HoloLoom(config=config)
        except ImportError:
            pass

        # LLM
        if llm_provider == "auto":
            self.llm = create_best_available_llm()
        else:
            self.llm = MultiProviderLLM(provider=llm_provider)

        # Vector store
        self.vector_store = VectorStoreFactory(store_type=vector_store)

    async def ingest(self, source: str) -> int:
        """
        Ingest documents.

        Args:
            source: File path, directory, or URL

        Returns:
            Number of chunks ingested
        """
        loader = UniversalDocumentLoader()
        shards = loader.load(source)

        # Add to HoloLoom
        if self.hololoom:
            for shard in shards:
                await self.hololoom.experience(shard.content)

        # Add to vector store
        if self.vector_store:
            self.vector_store.add_shards(shards)

        return len(shards)

    async def query(self, question: str, mode: str = "rag") -> str:
        """
        Query with context.

        Args:
            question: Query text
            mode: Query mode (rag/direct)

        Returns:
            Answer text
        """
        if mode == "rag" and self.hololoom:
            # RAG mode with HoloLoom memory
            memories = await self.hololoom.recall(question)
            context = "\n".join([m.content for m in memories[:3]])

            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            return self.llm(prompt)

        else:
            # Direct mode
            return self.llm(question)


# Main entry point

if __name__ == "__main__":
    quick_start()
