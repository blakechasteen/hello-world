"""
Workspace Indexer Demo - HoloLoom Integration

Demonstrates the complete code → knowledge graph pipeline:
1. Scan workspace for code files
2. Parse code structure (classes, functions, imports)
3. Index into HoloLoom's knowledge graph
4. Semantic search over code

Created: 2025-11-17
"""

import asyncio
from pathlib import Path
from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.spinningWheel import WorkspaceSpinner


async def main():
    print("=" * 80)
    print("Workspace Indexer Demo - HoloLoom Integration")
    print("=" * 80)
    print()

    # Create HoloLoom instance
    config = Config.fast()
    async with HoloLoom(config=config) as loom:
        # Create workspace spinner
        spinner = WorkspaceSpinner()

        # Index the HoloLoom project itself (meta!)
        workspace_path = Path(__file__).parent.parent / "HoloLoom" / "spinningWheel"

        print(f"📁 Indexing workspace: {workspace_path}")
        print()

        # First index (full scan)
        print("🔍 Phase 1: Initial Index (Full Scan)")
        print("-" * 80)

        stats1 = await spinner.index_to_hololoom(
            workspace_path=workspace_path,
            hololoom_instance=loom,
            incremental=True,
            languages=["python"]  # Only Python files
        )

        print(f"✅ Files indexed: {stats1['files']}")
        print(f"✅ Entities created: {stats1['entities']}")
        print(f"✅ Edges created: {stats1['edges']}")
        print(f"✅ Errors: {stats1['errors']}")
        print()

        # Show knowledge graph stats
        print("📊 Knowledge Graph Statistics:")
        print("-" * 80)
        kg = loom.graph
        print(f"Total nodes: {kg.number_of_nodes()}")
        print(f"Total edges: {kg.number_of_edges()}")

        # Show entity types
        file_entities = [n for n in kg.nodes() if kg.nodes[n].get('type') == 'file']
        code_entities = [n for n in kg.nodes() if kg.nodes[n].get('type') == 'entity']

        print(f"File entities: {len(file_entities)}")
        print(f"Code entities: {len(code_entities)}")
        print()

        # Show sample entities
        print("📝 Sample Entities:")
        print("-" * 80)
        for entity in list(code_entities)[:5]:
            node_data = kg.nodes[entity]
            print(f"  - {node_data.get('name')} ({node_data.get('type')})")
            print(f"    File: {node_data.get('file_path')}")
        print()

        # Semantic search over code
        print("🔎 Phase 2: Semantic Search Over Code")
        print("-" * 80)

        queries = [
            "workspace scanner",
            "code parsing",
            "file processing"
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            memories = await loom.recall(query, limit=3)

            if memories:
                for i, memory in enumerate(memories, 1):
                    file_path = memory.context.get('file_path', 'unknown')
                    element_count = memory.context.get('element_count', 0)
                    print(f"  {i}. {file_path} ({element_count} elements)")
            else:
                print("  No results found")

        print()

        # Incremental update (no changes)
        print("🔄 Phase 3: Incremental Update (No Changes)")
        print("-" * 80)

        stats2 = await spinner.index_to_hololoom(
            workspace_path=workspace_path,
            hololoom_instance=loom,
            incremental=True,
            languages=["python"]
        )

        print(f"✅ Files re-indexed: {stats2['files']}")
        print(f"✅ Files skipped (unchanged): {stats2.get('skipped', 'N/A')}")
        print()

        # Show index metadata
        print("💾 Index Metadata:")
        print("-" * 80)
        index_file = workspace_path / ".hololoom_index.json"
        if index_file.exists():
            print(f"Location: {index_file}")
            import json
            with open(index_file) as f:
                metadata = json.load(f)
            print(f"Files tracked: {len(metadata)}")

            # Show sample metadata
            if metadata:
                sample_file = list(metadata.keys())[0]
                sample_data = metadata[sample_file]
                print(f"\nSample entry: {sample_file}")
                print(f"  Hash: {sample_data.get('hash', 'N/A')[:16]}...")
                print(f"  Indexed at: {sample_data.get('indexed_at', 'N/A')}")
                print(f"  Entities: {sample_data.get('entities', 0)}")
                print(f"  Edges: {sample_data.get('edges', 0)}")
        else:
            print("No index metadata file found")

        print()
        print("=" * 80)
        print("Demo Complete!")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
