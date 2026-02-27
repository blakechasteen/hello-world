"""
Smart AI for YOUR Creative Writing
Ingests your stories, chapters, notes and makes them searchable
Run: PYTHONPATH=. python ingest_my_writing.py
"""
import asyncio
from pathlib import Path
from hololoom.rag import SimpleRAG
from hololoom.config import Config

async def ingest_creative_writing():
    print("📚 Creative Writing AI - Loading your stories...\n")

    config = Config.fast()
    config.enable_smart_routing = True  # ⚡ Phase 1: 15x speedup for simple queries!

    async with SimpleRAG(config=config) as rag:
        print("⚡ Smart routing enabled\n")
        # ========================================
        # CUSTOMIZE THESE PATHS TO YOUR FILES!
        # ========================================
        your_writing_folders = [
            r"c:\Users\blake\OneDrive\Documents\school\creativewriting\SpeakForMe",
            # Add more folders here:
            # r"c:\Users\blake\OneDrive\Documents\school\creativewriting\OtherProject",
        ]

        file_extensions = ['.txt', '.md', '.doc', '.docx']

        # Scan and ingest
        total_files = 0
        total_chars = 0

        for folder in your_writing_folders:
            folder_path = Path(folder)

            if not folder_path.exists():
                print(f"⚠️  Folder not found: {folder}")
                continue

            print(f"\n📁 Scanning: {folder_path.name}")

            # Find all text files
            for file_path in folder_path.rglob('*'):
                if file_path.suffix.lower() in file_extensions or file_path.suffix == '':
                    try:
                        # Read file
                        content = file_path.read_text(encoding='utf-8', errors='ignore')

                        if len(content.strip()) < 50:  # Skip tiny files
                            continue

                        # Ingest with metadata
                        await rag.ingest(content)

                        total_files += 1
                        total_chars += len(content)

                        print(f"  ✓ {file_path.name} ({len(content):,} characters)")

                    except Exception as e:
                        print(f"  ✗ Failed to read {file_path.name}: {e}")

        print(f"\n✅ Ingestion complete!")
        print(f"   Files: {total_files}")
        print(f"   Characters: {total_chars:,}")
        print(f"   ~Words: {total_chars // 5:,}\n")

        # ========================================
        # NOW QUERY YOUR WRITING
        # ========================================
        print("🔍 Now you can ask questions about your writing!\n")

        example_queries = [
            "What are the main themes in my writing?",
            "Tell me about the characters",
            "What happens in chapter 5?",
            "Summarize the plot",
            "What writing style do I use?",
        ]

        print("Example queries you could try:")
        for q in example_queries:
            print(f"  • {q}")

        print("\n" + "=" * 80 + "\n")

        # Interactive mode
        while True:
            try:
                query = input("Ask about your writing (or 'quit'): ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    break

                if not query:
                    continue

                print("\n🤖 Analyzing your writing...")

                result = await rag.query(query, mode="research")  # Research mode for creative analysis

                print(f"\n💡 Answer ({result.confidence:.0%} confidence):")
                print(f"{result.response}\n")
                print(f"📄 Sources: {len(result.sources)} passages analyzed\n")
                print("-" * 80 + "\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

        print("\n👋 Thanks for using your Creative Writing AI!")

if __name__ == "__main__":
    asyncio.run(ingest_creative_writing())