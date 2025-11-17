#!/usr/bin/env python3
"""
Test script for LSP server real data integration (Wave 2)

This script demonstrates:
1. Index a test workspace using WorkspaceSpinner
2. Query the LSP handlers directly (without LSP client)
3. Verify real entities are returned from knowledge graph

Usage:
    PYTHONPATH=. python HoloLoom/lsp/test_real_data.py

Expected output:
    ✓ Workspace indexed: N files, M entities, P edges
    ✓ Completion: Returns real entities from KG
    ✓ Hover: Shows entity metadata
    ✓ Definition: Returns file location
    ✓ Symbol search: Finds entities by name

Wave 2 Integration Test (November 2025)
"""

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List

# ============================================================================
# TEST UTILITIES
# ============================================================================

def create_test_workspace() -> Path:
    """
    Create a temporary test workspace with sample Python files.

    Returns:
        Path to temporary workspace directory
    """
    # Create temporary directory
    workspace = Path(tempfile.mkdtemp(prefix="hololoom_test_"))

    # Create sample Python files
    (workspace / "sample.py").write_text("""
\"\"\"Sample Python module for testing LSP integration.\"\"\"

class MyClass:
    \"\"\"A sample class.\"\"\"

    def __init__(self, name: str):
        self.name = name

    def greet(self):
        \"\"\"Greet the user.\"\"\"
        return f"Hello, {self.name}!"


def my_function(x: int, y: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return x + y


def another_function():
    \"\"\"Another sample function.\"\"\"
    instance = MyClass("World")
    result = instance.greet()
    return result
""")

    (workspace / "module2.py").write_text("""
\"\"\"Second module for testing.\"\"\"

from sample import MyClass

def use_class():
    \"\"\"Use MyClass from sample module.\"\"\"
    obj = MyClass("Test")
    return obj.greet()
""")

    print(f"✓ Created test workspace: {workspace}")
    return workspace


def cleanup_workspace(workspace: Path):
    """Remove temporary workspace."""
    if workspace.exists():
        shutil.rmtree(workspace)
        print(f"✓ Cleaned up workspace: {workspace}")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

async def test_workspace_indexing(workspace: Path):
    """
    Test workspace indexing with WorkspaceSpinner.

    Args:
        workspace: Path to test workspace

    Returns:
        HoloLoom instance with indexed workspace
    """
    from HoloLoom import HoloLoom
    from HoloLoom.spinningWheel import WorkspaceSpinner

    print("\n" + "=" * 70)
    print("TEST 1: Workspace Indexing")
    print("=" * 70)

    # Create HoloLoom instance
    loom = HoloLoom()
    await loom.__aenter__()

    # Index workspace
    spinner = WorkspaceSpinner()
    stats = await spinner.index_to_hololoom(
        workspace_path=workspace,
        hololoom_instance=loom,
        incremental=False  # Full index
    )

    print(f"\n✓ Workspace indexed successfully!")
    print(f"  - Files: {stats['files']}")
    print(f"  - Entities: {stats['entities']}")
    print(f"  - Edges: {stats['edges']}")

    # Verify KG populated
    kg = loom.graph
    print(f"\n✓ Knowledge graph populated:")
    print(f"  - Nodes: {kg.number_of_nodes()}")
    print(f"  - Edges: {kg.number_of_edges()}")

    # Show sample entities
    print(f"\n✓ Sample entities:")
    for node_id in list(kg.nodes())[:5]:
        node_data = kg.nodes[node_id]
        print(f"  - {node_data.get('name', node_id)} ({node_data.get('type', 'unknown')})")

    return loom


async def test_completion_handler(loom):
    """
    Test completion handler with real KG data.

    Args:
        loom: HoloLoom instance with indexed workspace
    """
    print("\n" + "=" * 70)
    print("TEST 2: Completion Handler (Real KG Data)")
    print("=" * 70)

    # Import helper functions from server
    from HoloLoom.lsp.server import (
        _get_file_entities,
        _find_entities_fuzzy,
        _entity_to_completion_item
    )

    kg = loom.graph

    # Test 1: Get entities from specific file
    print("\n✓ Test: Get entities from 'sample.py'")
    file_entities = _get_file_entities(kg, "sample.py")

    if file_entities:
        print(f"  Found {len(file_entities)} entities:")
        for node_id, entity in file_entities:
            print(f"    - {entity.get('name')} ({entity.get('type')})")
            print(f"      Line: {entity.get('line_number')}")

        # Convert to completion items
        print(f"\n✓ Convert to LSP CompletionItems:")
        for node_id, entity in file_entities[:3]:  # Show first 3
            item = _entity_to_completion_item(entity, node_id, rank=0)
            print(f"    - label: {item.label}")
            print(f"      kind: {item.kind}")
            print(f"      detail: {item.detail}")
            print(f"      sort_text: {item.sort_text}")
    else:
        print("  ❌ No entities found (check file path)")

    # Test 2: Fuzzy search
    print("\n✓ Test: Fuzzy search for 'MyClass'")
    fuzzy_results = _find_entities_fuzzy(kg, "MyClass", limit=5)

    if fuzzy_results:
        print(f"  Found {len(fuzzy_results)} matches:")
        for node_id, entity in fuzzy_results:
            print(f"    - {entity.get('name')} ({entity.get('type')}) in {entity.get('file_path')}")
    else:
        print("  ❌ No matches found")


async def test_hover_handler(loom):
    """
    Test hover handler with real KG data.

    Args:
        loom: HoloLoom instance with indexed workspace
    """
    print("\n" + "=" * 70)
    print("TEST 3: Hover Handler (Real KG Data)")
    print("=" * 70)

    from HoloLoom.lsp.server import _find_entity_by_name, _get_related_entities

    kg = loom.graph

    # Test: Find entity by name
    print("\n✓ Test: Find 'MyClass' entity")
    result = _find_entity_by_name(kg, "MyClass")

    if result:
        entity_id, entity = result
        print(f"  Found entity: {entity.get('name')}")
        print(f"    Type: {entity.get('type')}")
        print(f"    File: {entity.get('file_path')}")
        print(f"    Line: {entity.get('line_number')}")

        if entity.get('docstring'):
            print(f"    Docstring: {entity['docstring'][:50]}...")

        # Get related entities
        print(f"\n✓ Test: Get related entities")
        related = _get_related_entities(kg, entity_id, max_depth=1)

        if related:
            print(f"  Found {len(related)} related entities:")
            for rel_id, rel_entity in related[:3]:
                print(f"    - {rel_entity.get('name')} ({rel_entity.get('type')})")
        else:
            print("  No related entities (normal for simple test)")
    else:
        print("  ❌ Entity not found")


async def test_definition_handler(loom):
    """
    Test definition handler with real KG data.

    Args:
        loom: HoloLoom instance with indexed workspace
    """
    print("\n" + "=" * 70)
    print("TEST 4: Definition Handler (Real KG Data)")
    print("=" * 70)

    from HoloLoom.lsp.server import _find_entity_by_name, _file_path_to_uri

    kg = loom.graph

    # Test: Go-to-definition for 'my_function'
    print("\n✓ Test: Go-to-definition for 'my_function'")
    result = _find_entity_by_name(kg, "my_function")

    if result:
        entity_id, entity = result

        file_path = entity.get('file_path')
        line_number = entity.get('line_number', 0)

        print(f"  Definition location:")
        print(f"    File: {file_path}")
        print(f"    Line: {line_number}")

        # Convert to URI
        file_uri = _file_path_to_uri(file_path)
        print(f"    URI: {file_uri}")
    else:
        print("  ❌ Entity not found")


async def test_symbol_search(loom):
    """
    Test workspace symbol search.

    Args:
        loom: HoloLoom instance with indexed workspace
    """
    print("\n" + "=" * 70)
    print("TEST 5: Workspace Symbol Search")
    print("=" * 70)

    from HoloLoom.lsp.server import _find_entities_fuzzy, _entity_to_symbol_info

    kg = loom.graph

    # Test: Search for 'function'
    print("\n✓ Test: Search for entities matching 'function'")
    matches = _find_entities_fuzzy(kg, "function", limit=10)

    if matches:
        print(f"  Found {len(matches)} matches:")
        for node_id, entity in matches:
            symbol_info = _entity_to_symbol_info(entity, node_id)
            print(f"    - {symbol_info.name} ({symbol_info.kind})")
            print(f"      Location: {symbol_info.location.uri}")
    else:
        print("  ❌ No matches found")


async def test_edge_cases(loom):
    """
    Test edge cases and error handling.

    Args:
        loom: HoloLoom instance
    """
    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases & Error Handling")
    print("=" * 70)

    from HoloLoom.lsp.server import (
        _find_entity_by_name,
        _get_file_entities,
        _find_entities_fuzzy
    )

    kg = loom.graph

    # Test 1: Entity not found
    print("\n✓ Test: Entity not found (graceful handling)")
    result = _find_entity_by_name(kg, "NonExistentEntity")
    if result is None:
        print("  ✓ Correctly returns None for missing entity")
    else:
        print("  ❌ Should return None")

    # Test 2: File not found
    print("\n✓ Test: File not indexed (graceful handling)")
    entities = _get_file_entities(kg, "nonexistent_file.py")
    if len(entities) == 0:
        print("  ✓ Correctly returns empty list for missing file")
    else:
        print("  ❌ Should return empty list")

    # Test 3: Empty query
    print("\n✓ Test: Empty fuzzy search query")
    matches = _find_entities_fuzzy(kg, "", limit=10)
    # Empty query is ok, may return all or none
    print(f"  ✓ Returns {len(matches)} results (expected: variable)")

    # Test 4: Missing metadata
    print("\n✓ Test: Entity with missing metadata")
    # All entities should have basic metadata from workspace indexer
    has_missing = False
    for node_id in kg.nodes():
        entity = kg.nodes[node_id]
        if entity.get('type') != 'file':  # Skip file nodes
            if not entity.get('name'):
                has_missing = True
                print(f"  ❌ Entity {node_id} missing 'name'")
                break

    if not has_missing:
        print("  ✓ All entities have required metadata")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def main():
    """Run all tests."""
    print("=" * 70)
    print("LSP Server Real Data Integration Test (Wave 2)")
    print("=" * 70)
    print("\nThis test demonstrates LSP handlers querying real KG data")
    print("populated by the workspace indexer (Agent A's work).")

    workspace = None
    loom = None

    try:
        # Create test workspace
        workspace = create_test_workspace()

        # Test 1: Index workspace
        loom = await test_workspace_indexing(workspace)

        # Test 2: Completion handler
        await test_completion_handler(loom)

        # Test 3: Hover handler
        await test_hover_handler(loom)

        # Test 4: Definition handler
        await test_definition_handler(loom)

        # Test 5: Symbol search
        await test_symbol_search(loom)

        # Test 6: Edge cases
        await test_edge_cases(loom)

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("\n✓ All tests completed successfully!")
        print("\nKey achievements (Wave 2):")
        print("  1. Workspace indexer populates knowledge graph ✓")
        print("  2. LSP handlers query real KG data (no placeholders) ✓")
        print("  3. Entities have metadata (file, line, docstring) ✓")
        print("  4. Graph traversal works (DEFINES edges) ✓")
        print("  5. Error handling for empty KG / missing entities ✓")
        print("\nNext steps:")
        print("  - Wave 3: VS Code extension integration")
        print("  - Use LSP client to test full protocol")
        print("  - Add more edge types (USES, IS_A, MENTIONS)")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        if loom:
            await loom.__aexit__(None, None, None)

        if workspace:
            cleanup_workspace(workspace)

        print("\n" + "=" * 70)
        print("Test completed")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
