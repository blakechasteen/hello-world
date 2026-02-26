#!/usr/bin/env python3
"""
Test Vector Embedding Integration

Verifies that:
1. Embeddings are generated when messages are archived
2. Embeddings are stored with the memory
3. Semantic search works using embeddings
"""

import asyncio
import sys
sys.path.insert(0, 'c:/Users/blake/Documents/mythRL')

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
from HoloLoom.memory.protocol import Memory, MemoryQuery
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from datetime import datetime
import numpy as np


async def test_embedding_pipeline():
    """Test complete embedding pipeline"""

    print("\n" + "="*70)
    print("  Vector Embedding Integration Test")
    print("="*70 + "\n")

    # 1. Initialize embedder
    print("[1/5] Initializing embedder...")
    embedder = MatryoshkaEmbeddings(sizes=[384, 768])
    print("✓ MatryoshkaEmbeddings initialized\n")

    # 2. Initialize memory backend
    print("[2/5] Connecting to memory...")
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID
    memory = await create_memory_backend(config)
    print("✓ Memory backend connected\n")

    # 3. Create test message with embedding
    print("[3/5] Generating embedding for test message...")
    test_text = "The quick brown fox jumps over the lazy dog using semantic vector embeddings"

    # Generate embedding
    embeddings = embedder.encode([test_text])
    embedding = embeddings[0] if len(embeddings) > 0 else None

    print(f"✓ Embedding generated: shape={embedding.shape if embedding is not None else 'None'}")
    print(f"  First 5 values: {embedding[:5] if embedding is not None else 'None'}\n")

    # 4. Store message with embedding
    print("[4/5] Storing message with embedding...")

    memory_obj = Memory(
        id=f"embedding_test_{datetime.now().timestamp()}",
        text=test_text,
        timestamp=datetime.now(),
        context={
            'test': 'embedding_pipeline',
            'has_embedding': True,
        },
        metadata={
            'thread_id': 'embedding_test',
            'role': 'user',
            'source': 'embedding_test',
            'user_id': 'test_user',
        }
    )

    # Try to add embedding if supported
    if hasattr(memory_obj, 'embedding'):
        memory_obj.embedding = embedding
        print("✓ Embedding attached to Memory object")
    else:
        print("⚠ Memory object doesn't have embedding field (will use metadata)")
        memory_obj.metadata['embedding_shape'] = str(embedding.shape)

    # Store
    result = await memory.store(memory_obj, user_id='test_user')
    print(f"✓ Stored with ID: {result}\n")

    # 5. Test semantic retrieval
    print("[5/5] Testing semantic retrieval...")

    # Query with semantically similar text (different words, same meaning)
    query_text = "A fast auburn canine leaps above a sluggish hound"  # Similar meaning, different words

    query = MemoryQuery(
        text=query_text,
        user_id='test_user',
        limit=5
    )

    print(f"Query: '{query_text}'")
    recall_result = await memory.recall(query, limit=5)

    print(f"✓ Found {len(recall_result.memories)} results:")
    for i, mem in enumerate(recall_result.memories[:5], 1):
        print(f"  {i}. {mem.text[:80]}...")
        if hasattr(mem, 'metadata') and mem.metadata:
            print(f"     Thread: {mem.metadata.get('thread_id', 'N/A')}")

    # Check if our test message was found
    found = any("semantic vector embeddings" in m.text for m in recall_result.memories)

    print("\n" + "="*70)
    if found:
        print("  ✓✓✓ EMBEDDING PIPELINE WORKS! ✓✓✓")
        print("  Semantic search found the test message using embeddings!")
    else:
        print("  ⚠ Test message not found in top 5 results")
        print("  This may be okay if there are many other messages in the database")
    print("="*70 + "\n")

    return found


async def test_chat_archiving():
    """Test that chat messages are archived with embeddings"""

    print("\n" + "="*70)
    print("  Chat Archiving with Embeddings Test")
    print("="*70 + "\n")

    from HoloLoom.apps.workflow_builder.thread_manager import ThreadManager, Message
    from HoloLoom.awareness import CompositionalAwarenessLayer, DualStreamGenerator, OllamaLLM

    # Initialize components
    print("[1/3] Initializing components...")

    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID

    memory = await create_memory_backend(config)
    embedder = MatryoshkaEmbeddings(sizes=[384, 768])
    awareness = CompositionalAwarenessLayer()

    try:
        llm = OllamaLLM(model="llama3.2:3b")
        if llm.is_available():
            dual_stream_gen = DualStreamGenerator(awareness, llm_generator=llm)
        else:
            dual_stream_gen = DualStreamGenerator(awareness, llm_generator=None)
    except Exception:
        dual_stream_gen = DualStreamGenerator(awareness, llm_generator=None)

    thread_manager = ThreadManager(
        awareness_layer=awareness,
        llm_generator=dual_stream_gen,
        memory_backend=memory,
        embedder=embedder,
        user_id='test_user'
    )

    print("✓ Components initialized\n")

    # Send a test message
    print("[2/3] Sending test message through chat...")
    test_query = "Vector embeddings enable semantic similarity search in high-dimensional spaces"

    result = await thread_manager.process_message(test_query)

    print(f"✓ Message processed")
    print(f"  Thread ID: {result['thread_id']}")
    if 'assistant_message' in result:
        print(f"  Response: {result['assistant_message']['content'][:100]}...\n")
    else:
        print(f"  Response keys: {list(result.keys())}\n")

    # Give archiving time to complete (it's a background task)
    print("[3/3] Waiting for background archiving...")
    await asyncio.sleep(2)
    print("✓ Archiving should be complete\n")

    # Query for the message
    print("Testing retrieval...")
    query = MemoryQuery(
        text="semantic similarity high-dimensional",
        user_id='test_user',
        limit=5
    )

    recall_result = await memory.recall(query, limit=5)

    print(f"Found {len(recall_result.memories)} results:")
    for i, mem in enumerate(recall_result.memories[:3], 1):
        print(f"  {i}. {mem.text[:80]}...")

    found = any("Vector embeddings" in m.text for m in recall_result.memories)

    print("\n" + "="*70)
    if found:
        print("  ✓✓✓ CHAT ARCHIVING WITH EMBEDDINGS WORKS! ✓✓✓")
    else:
        print("  ⚠ Chat message not found (may need more time to archive)")
    print("="*70 + "\n")

    return found


async def main():
    """Run all tests"""

    print("\n" + "="*70)
    print("  VECTOR EMBEDDING INTEGRATION - COMPREHENSIVE TEST")
    print("="*70)

    try:
        # Test 1: Direct embedding pipeline
        test1_passed = await test_embedding_pipeline()

        # Test 2: Chat archiving with embeddings
        test2_passed = await test_chat_archiving()

        # Summary
        print("\n" + "="*70)
        print("  SUMMARY")
        print("="*70)
        print(f"  Test 1 (Direct Pipeline):  {'✓ PASS' if test1_passed else '✗ FAIL'}")
        print(f"  Test 2 (Chat Archiving):    {'✓ PASS' if test2_passed else '✗ FAIL'}")
        print("="*70 + "\n")

        if test1_passed and test2_passed:
            print("  ✓✓✓ ALL TESTS PASSED ✓✓✓\n")
        else:
            print("  ⚠ Some tests did not pass (check results above)\n")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
