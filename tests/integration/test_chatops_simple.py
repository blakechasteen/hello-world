#!/usr/bin/env python3
"""
Simple ChatOps Component Test
==============================
Test individual ChatOps components before running the full bot.
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_imports():
    """Test all ChatOps imports."""
    print("="*80)
    print("ChatOps Import Test")
    print("="*80)
    print()

    # Test 1: Matrix Bot
    print("1. Testing MatrixBot imports...")
    try:
        from HoloLoom.chatops.core.matrix_bot import MatrixBot, MatrixBotConfig
        print("   [OK] MatrixBot imported successfully")
    except Exception as e:
        print(f"   [FAIL] MatrixBot import failed: {e}")
        return False

    # Test 2: ChatOps Bridge
    print("2. Testing ChatOpsOrchestrator imports...")
    try:
        from HoloLoom.chatops.core.chatops_bridge import ChatOpsOrchestrator, ConversationContext
        print("   [OK] ChatOpsOrchestrator imported successfully")
    except Exception as e:
        print(f"   [FAIL] ChatOpsOrchestrator import failed: {e}")
        return False

    # Test 3: Conversation Memory
    print("3. Testing ConversationMemory imports...")
    try:
        from HoloLoom.chatops.core.conversation_memory import ConversationMemory
        print("   [OK] ConversationMemory imported successfully")
    except Exception as e:
        print(f"   [FAIL] ConversationMemory import failed: {e}")
        return False

    # Test 4: HoloLoom Config
    print("4. Testing HoloLoom config...")
    try:
        from HoloLoom.config import Config
        cfg = Config.fast()
        print(f"   [OK] HoloLoom Config created: mode={cfg.mode}")
    except Exception as e:
        print(f"   [FAIL] HoloLoom config failed: {e}")
        return False

    print()
    return True


async def test_memory_backend():
    """Test memory backend creation."""
    print("="*80)
    print("Memory Backend Test")
    print("="*80)
    print()

    try:
        from HoloLoom.config import Config, MemoryBackend
        from HoloLoom.memory.backend_factory import create_memory_backend
        from HoloLoom.memory.protocol import Memory, MemoryQuery

        # Test NetworkX backend
        print("Testing NetworkX (in-memory) backend...")
        config = Config.fast()
        config.memory_backend = MemoryBackend.INMEMORY

        memory = await create_memory_backend(config)
        print("[OK] Memory backend created")

        # Test store
        from datetime import datetime
        test_mem = Memory(
            id="chatops_test_1",
            text="Testing ChatOps memory integration",
            timestamp=datetime.now(),
            context={"room": "test_room"},
            metadata={"source": "test", "type": "chatops"}
        )
        await memory.store(test_mem)
        print("[OK] Memory stored")

        # Test recall
        query = MemoryQuery(text="ChatOps memory")
        result = await memory.recall(query, limit=5)
        print(f"[OK] Recall successful: {len(result.memories)} memories found")

        print()
        return True

    except Exception as e:
        print(f"[FAIL] Memory backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chatops_orchestrator():
    """Test ChatOps orchestrator creation."""
    print("="*80)
    print("ChatOps Orchestrator Test")
    print("="*80)
    print()

    try:
        from HoloLoom.chatops.core.chatops_bridge import ChatOpsOrchestrator
        from HoloLoom.config import Config

        print("Creating ChatOpsOrchestrator...")
        config = Config.fast()

        chatops = ChatOpsOrchestrator(
            hololoom_config=config,
            memory_store_path="./test_chatops_memory",
            context_limit=10,
            enable_memory_storage=True
        )
        print("[OK] ChatOpsOrchestrator created")

        # Test conversation context
        print("Testing conversation context...")
        from HoloLoom.chatops.core.chatops_bridge import ConversationContext

        ctx = ConversationContext(
            room_id="!test:matrix.org",
            room_name="Test Room"
        )
        ctx.add_message("@alice:matrix.org", "Hello, bot!")
        ctx.add_message("@bob:matrix.org", "How are you?")

        recent = ctx.get_recent_messages(5)
        print(f"[OK] Conversation context: {len(recent)} messages")

        context_str = ctx.to_context_string(5)
        print(f"[OK] Context string:\n{context_str}")

        print()
        return True

    except Exception as e:
        print(f"[FAIL] ChatOps orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_matrix_config():
    """Test Matrix bot configuration."""
    print("="*80)
    print("Matrix Configuration Test")
    print("="*80)
    print()

    try:
        from HoloLoom.chatops.core.matrix_bot import MatrixBotConfig

        print("Creating test Matrix config...")
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@test_bot:matrix.org",
            password="test_password",
            device_id="CHATOPS_TEST",
            rooms=[],
            command_prefix="!",
            respond_to_mentions=True,
            respond_to_dm=True
        )
        print(f"[OK] Matrix config created")
        print(f"  Homeserver: {config.homeserver_url}")
        print(f"  User: {config.user_id}")
        print(f"  Command prefix: {config.command_prefix}")

        print()
        return True

    except Exception as e:
        print(f"[FAIL] Matrix config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print()
    print("=" + "="*78 + "=")
    print(" "*25 + "ChatOps Component Tests")
    print("=" + "="*78 + "=")
    print()

    results = []

    # Run tests
    results.append(("Imports", await test_imports()))
    results.append(("Memory Backend", await test_memory_backend()))
    results.append(("ChatOps Orchestrator", await test_chatops_orchestrator()))
    results.append(("Matrix Configuration", await test_matrix_config()))

    # Summary
    print()
    print("="*80)
    print("Test Summary")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print()
        print(" All tests passed! ChatOps components are ready.")
        print()
        print("Next steps:")
        print("1. Set Matrix credentials (MATRIX_USER_ID, MATRIX_PASSWORD)")
        print("2. Run: PYTHONPATH=. python HoloLoom/chatops/run_chatops.py")
        return True
    else:
        print()
        print("[WARNING] Some tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)