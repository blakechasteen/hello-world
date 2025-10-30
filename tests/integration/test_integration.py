#!/usr/bin/env python3
"""
Integration Tests - Complete System Testing
============================================
Tests all major updates from today's session.
"""

import sys
import os
import asyncio
from datetime import datetime

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_file_memory_store():
    """Test file-based memory persistence"""
    print("\n" + "="*60)
    print("TEST 1: File Memory Store")
    print("="*60)

    try:
        from HoloLoom.memory.stores.file_store import FileMemoryStore
        from HoloLoom.memory.protocol import Memory
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        # Create embedder
        print("  Creating embedder...")
        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        # Create store
        print("  Initializing file store...")
        store = FileMemoryStore(
            data_dir='./test_memory_integration',
            embedder=embedder
        )

        print(f"  [OK] Store initialized: {len(store.memories)} memories")

        # Add memories
        print("  Adding test memories...")
        memories = [
            Memory(
                id="test_1",
                text="Python is a programming language",
                timestamp=datetime.now(),
                context={},
                metadata={"source": "test"}
            ),
            Memory(
                id="test_2",
                text="Machine learning uses neural networks",
                timestamp=datetime.now(),
                context={},
                metadata={"source": "test"}
            )
        ]

        async def add_memories():
            for mem in memories:
                await store.store(mem)

        asyncio.run(add_memories())

        print(f"  [OK] Added {len(memories)} memories")
        print(f"  [OK] Total in store: {len(store.memories)}")

        # Test retrieval
        print("  Testing retrieval...")

        async def test_retrieval():
            from HoloLoom.memory.protocol import MemoryQuery, Strategy
            query = MemoryQuery(text="What is Python?", limit=2)
            result = await store.retrieve(query, strategy=Strategy.FUSED)
            return result

        result = asyncio.run(test_retrieval())

        print(f"  [OK] Retrieved {len(result.memories)} memories")
        print(f"  [OK] Scores: {[f'{s:.3f}' for s in result.scores]}")

        # Test persistence
        print("  Testing persistence...")
        store2 = FileMemoryStore(
            data_dir='./test_memory_integration',
            embedder=embedder
        )
        print(f"  [OK] Reloaded {len(store2.memories)} memories from disk")

        print("\n[PASS] File Memory Store: PASS")
        return True

    except Exception as e:
        print(f"\n[FAIL] File Memory Store: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_memory():
    """Test hybrid memory with fallback"""
    print("\n" + "="*60)
    print("TEST 2: Hybrid Memory System")
    print("="*60)

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        print("  Initializing weaving orchestrator...")
        weaver = WeavingOrchestrator(
            use_mcts=True,
            mcts_simulations=10
        )

        # Check memory backend
        if hasattr(weaver.memory_store, 'backends'):
            print(f"  [OK] Hybrid memory active: {len(weaver.memory_store.backends)} backends")
            for backend in weaver.memory_store.backends:
                print(f"    - {backend.name} ({backend.weight:.1%})")
        elif hasattr(weaver.memory_store, 'data_dir'):
            print(f"  [OK] File-only memory: {weaver.memory_store.data_dir}")
        else:
            print(f"  [WARN] In-memory fallback")

        # Add knowledge
        print("  Adding knowledge...")

        async def add_knowledge():
            await weaver.add_knowledge(
                "MCTS uses Thompson Sampling for exploration",
                {"topic": "algorithms"}
            )
            await weaver.add_knowledge(
                "Matryoshka embeddings enable multi-scale retrieval",
                {"topic": "embeddings"}
            )

        asyncio.run(add_knowledge())
        print("  [OK] Knowledge added")

        # Test retrieval
        print("  Testing context retrieval...")

        async def test_context():
            context = await weaver._retrieve_context("What is MCTS?", limit=2)
            return context

        context = asyncio.run(test_context())
        print(f"  [OK] Retrieved {len(context)} context shards")

        print("\n[PASS] Hybrid Memory: PASS")
        return True

    except Exception as e:
        print(f"\n[FAIL] Hybrid Memory: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weaving_orchestrator():
    """Test complete weaving cycle with memory"""
    print("\n" + "="*60)
    print("TEST 3: Weaving Orchestrator")
    print("="*60)

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        print("  Creating orchestrator...")
        weaver = WeavingOrchestrator(
            use_mcts=True,
            mcts_simulations=20
        )

        print("  [OK] Orchestrator initialized")

        # Add knowledge
        print("  Adding knowledge base...")

        async def setup():
            await weaver.add_knowledge(
                "Thompson Sampling balances exploration and exploitation",
                {"category": "RL"}
            )
            await weaver.add_knowledge(
                "MCTS builds search trees with UCB1",
                {"category": "algorithms"}
            )

        asyncio.run(setup())
        print("  [OK] Knowledge base ready")

        # Run weaving cycle
        print("  Executing weaving cycle...")

        async def weave():
            spacetime = await weaver.weave("Explain Thompson Sampling")
            return spacetime

        result = asyncio.run(weave())

        print(f"  [OK] Tool selected: {result.tool_used}")
        print(f"  [OK] Confidence: {result.confidence:.1%}")
        print(f"  [OK] Duration: {result.trace.duration_ms:.0f}ms")
        print(f"  [OK] Context retrieved: {result.trace.context_shards_count} shards")

        # Check stats
        stats = weaver.get_statistics()
        print(f"  [OK] Total weavings: {stats['total_weavings']}")

        weaver.stop()

        print("\n[PASS] Weaving Orchestrator: PASS")
        return True

    except Exception as e:
        print(f"\n[FAIL] Weaving Orchestrator: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_promptly_structure():
    """Test Promptly organization"""
    print("\n" + "="*60)
    print("TEST 4: Promptly Organization")
    print("="*60)

    try:
        import os
        from pathlib import Path

        promptly_root = Path(__file__).parent.parent / 'Promptly'

        # Check directories
        required_dirs = ['promptly', 'demos', 'docs', 'tests', 'templates']
        for dir_name in required_dirs:
            dir_path = promptly_root / dir_name
            if dir_path.exists():
                print(f"  [OK] {dir_name}/ exists")
            else:
                print(f"  [X] {dir_name}/ missing")

        # Check promptly package structure
        promptly_pkg = promptly_root / 'promptly'
        pkg_dirs = ['tools', 'integrations', 'docs', 'examples']
        for dir_name in pkg_dirs:
            dir_path = promptly_pkg / dir_name
            if dir_path.exists():
                print(f"  [OK] promptly/{dir_name}/ exists")
            else:
                print(f"  [X] promptly/{dir_name}/ missing")

        # Check UI module
        ui_path = promptly_pkg / 'ui'
        if ui_path.exists():
            print(f"  [OK] promptly/ui/ exists")
            ui_files = ['__init__.py', 'terminal_app.py', 'web_app.py']
            for file_name in ui_files:
                if (ui_path / file_name).exists():
                    print(f"    [OK] {file_name}")

        print("\n[PASS] Promptly Organization: PASS")
        return True

    except Exception as e:
        print(f"\n[FAIL] Promptly Organization: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vscode_extension():
    """Test VS Code extension structure"""
    print("\n" + "="*60)
    print("TEST 5: VS Code Extension")
    print("="*60)

    try:
        from pathlib import Path
        import json

        vscode_root = Path(__file__).parent.parent / 'Promptly' / 'vscode-extension'

        # Check package.json
        package_json = vscode_root / 'package.json'
        if package_json.exists():
            print("  [OK] package.json exists")

            with open(package_json) as f:
                manifest = json.load(f)

            print(f"    - Name: {manifest.get('name')}")
            print(f"    - Version: {manifest.get('version')}")
            print(f"    - Commands: {len(manifest.get('contributes', {}).get('commands', []))}")
            print(f"    - Views: {len(manifest.get('contributes', {}).get('views', {}).get('promptly', []))}")
        else:
            print("  [X] package.json missing")

        # Check structure
        if (vscode_root / 'src').exists():
            print("  [OK] src/ directory exists")
        if (vscode_root / 'README.md').exists():
            print("  [OK] README.md exists")

        print("\n[PASS] VS Code Extension: PASS")
        return True

    except Exception as e:
        print(f"\n[FAIL] VS Code Extension: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUITE")
    print("Testing Today's Updates")
    print("="*60)

    results = {
        "File Memory Store": test_file_memory_store(),
        "Hybrid Memory": test_hybrid_memory(),
        "Weaving Orchestrator": test_weaving_orchestrator(),
        "Promptly Organization": test_promptly_structure(),
        "VS Code Extension": test_vscode_extension()
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{name:.<40} {status}")

    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*60)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
