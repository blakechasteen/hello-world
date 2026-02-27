#!/usr/bin/env python3
"""
Chat with HoloLoom Documentation Demo
=====================================
Loads the README.md documentation into memory and lets you ask questions about it.

Usage:
    $env:PYTHONPATH = "."
    python demos/chat_with_docs_demo.py
"""

import asyncio
import sys
import os
import uuid

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hololoom.config import Config, MemoryBackend
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.terminal_ui import TerminalUI
from hololoom.protocols.types import MemoryShard, Query


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a sentence or paragraph
        if end < len(text):
            # Look for a good break point
            for sep in ['\n\n', '\n', '. ', ', ']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]  # Filter empty chunks


def extract_entities_simple(text: str) -> list[str]:
    """Extract likely entities from text (simple heuristic)."""
    entities = []
    # Look for capitalized phrases
    words = text.split()
    i = 0
    while i < len(words):
        word = words[i].strip('.,;:()[]{}')
        if word and word[0].isupper() and len(word) > 2:
            # Check for multi-word entity
            entity_words = [word]
            j = i + 1
            while j < len(words):
                next_word = words[j].strip('.,;:()[]{}')
                if next_word and next_word[0].isupper():
                    entity_words.append(next_word)
                    j += 1
                else:
                    break
            entity = ' '.join(entity_words)
            if len(entity) > 3 and entity not in ['The', 'This', 'That', 'These', 'When', 'What', 'How', 'Why']:
                entities.append(entity)
            i = j
        else:
            i += 1
    # Also extract code identifiers (e.g., WeavingOrchestrator, MemoryShard)
    import re
    code_patterns = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    entities.extend(code_patterns)
    return list(set(entities))[:10]  # Limit to 10 entities


def extract_motifs_simple(text: str) -> list[str]:
    """Extract topic motifs from text (simple heuristic)."""
    motifs = []
    text_lower = text.lower()
    # Check for common topics
    topic_keywords = {
        'memory': ['memory', 'storage', 'cache', 'retrieval'],
        'weaving': ['weave', 'weaving', 'orchestrator', 'pipeline'],
        'learning': ['learn', 'training', 'ppo', 'policy'],
        'embedding': ['embedding', 'vector', 'matryoshka'],
        'graph': ['graph', 'knowledge', 'yarn', 'neo4j'],
        'api': ['api', 'endpoint', 'server', 'fastapi'],
        'configuration': ['config', 'configuration', 'setting'],
        'testing': ['test', 'testing', 'pytest'],
        'documentation': ['documentation', 'readme', 'guide'],
    }
    for motif, keywords in topic_keywords.items():
        if any(kw in text_lower for kw in keywords):
            motifs.append(motif)
    return motifs[:5]


def load_documentation(file_path: str) -> list[MemoryShard]:
    """Load a documentation file and convert to memory shards."""
    print(f"Loading documentation from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"  File size: {len(content):,} characters")

    # Chunk the content
    chunks = chunk_text(content, chunk_size=800, overlap=100)
    print(f"  Created {len(chunks)} chunks")

    # Convert chunks to memory shards
    shards = []
    for i, chunk in enumerate(chunks):
        entities = extract_entities_simple(chunk)
        motifs = extract_motifs_simple(chunk)

        shard = MemoryShard(
            id=str(uuid.uuid4()),
            text=chunk,
            entities=entities,
            motifs=motifs,
            timestamp=float(i),  # Use chunk index as timestamp
            metadata={
                "source": file_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "confidence": 0.9
            }
        )
        shards.append(shard)

    print(f"  Created {len(shards)} memory shards")
    return shards


async def main():
    print("=" * 70)
    print("HoloLoom Chat with Documentation Demo")
    print("=" * 70)
    print()

    # Load documentation
    doc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "README.md"
    )

    if not os.path.exists(doc_path):
        print(f"Error: Could not find {doc_path}")
        return 1

    shards = load_documentation(doc_path)

    # Create config
    print("\nInitializing HoloLoom...")
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY

    # Create orchestrator
    print("Creating orchestrator with documentation memory...")
    try:
        orchestrator = WeavingOrchestrator(
            cfg=config,
            shards=shards,
            enable_complexity_auto_detect=True
        )
        print("  Orchestrator created successfully!")
    except Exception as e:
        print(f"  Error creating orchestrator: {e}")
        return 1

    # Create UI
    ui = TerminalUI(orchestrator)

    print("\n" + "=" * 70)
    print("Ready! You can now ask questions about HoloLoom.")
    print("The system has loaded the README.md documentation into memory.")
    print("=" * 70)
    print("\nExample questions:")
    print("  - What is the weaving cycle?")
    print("  - How does Thompson Sampling work?")
    print("  - What memory backends are supported?")
    print("  - Explain the RAG system")
    print("\nType 'quit' or 'exit' to stop.")
    print("Type 'stats' to see session statistics.")
    print("Type 'history' to see conversation history.")
    print()

    # Process queries
    while True:
        try:
            query_text = input("\nYou: ").strip()

            if not query_text:
                continue

            if query_text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if query_text.lower() == 'stats':
                ui.show_stats()
                continue

            if query_text.lower() == 'history':
                ui.show_history()
                continue

            # Process query
            await ui.weave_with_display(query_text, show_trace=False)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
