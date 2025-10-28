"""
Isolated test for TextSpinner - tests the spinner without full HoloLoom imports
"""

import asyncio
import sys
from pathlib import Path

# Add the directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import just the text spinner module
from HoloLoom.spinning_wheel.text import TextSpinner, TextSpinnerConfig, spin_text


async def test_basic_functionality():
    """Test basic TextSpinner functionality."""
    print("Testing TextSpinner basic functionality...")

    text = """
    HoloLoom is a neural decision-making system. It combines multi-scale embeddings
    with knowledge graph memory. The system uses Thompson Sampling for exploration.
    """

    # Test 1: Single shard
    print("\n1. Testing single shard (no chunking)...")
    config = TextSpinnerConfig(chunk_by=None, extract_entities=True)
    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'test.txt'})

    print(f"   Created {len(shards)} shard(s)")
    print(f"   Shard ID: {shards[0].id}")
    print(f"   Text length: {len(shards[0].text)}")
    print(f"   Entities: {shards[0].entities}")
    assert len(shards) == 1
    print("   [OK] Single shard test passed")

    # Test 2: Paragraph chunking
    print("\n2. Testing paragraph chunking...")
    long_text = """
    First paragraph about embeddings and representations.

    Second paragraph about memory systems and knowledge graphs.

    Third paragraph about policy engines and decision making.
    """
    config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=100)
    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': long_text, 'source': 'multi.txt'})

    print(f"   Created {len(shards)} shard(s)")
    for i, shard in enumerate(shards):
        print(f"   Shard {i}: {len(shard.text)} chars, chunk_index={shard.metadata.get('chunk_index')}")
    assert len(shards) >= 2
    print("   [OK] Paragraph chunking test passed")

    # Test 3: Sentence chunking
    print("\n3. Testing sentence chunking...")
    text = "First sentence. Second sentence here. Third sentence follows. Fourth one."
    config = TextSpinnerConfig(chunk_by='sentence', chunk_size=40, min_chunk_size=10)
    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'sentences.txt'})

    print(f"   Created {len(shards)} shard(s)")
    assert len(shards) >= 1
    print("   [OK] Sentence chunking test passed")

    # Test 4: Character chunking
    print("\n4. Testing character chunking...")
    text = "A" * 500
    config = TextSpinnerConfig(chunk_by='character', chunk_size=100, min_chunk_size=10)
    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'chars.txt'})

    print(f"   Created {len(shards)} shard(s)")
    assert len(shards) == 5  # 500 / 100 = 5 chunks
    print("   [OK] Character chunking test passed")

    # Test 5: Convenience function
    print("\n5. Testing convenience function...")
    shards = await spin_text("Quick test text", source="quick", chunk_by=None)
    print(f"   Created {len(shards)} shard(s)")
    assert len(shards) == 1
    print("   [OK] Convenience function test passed")

    # Test 6: Metadata propagation
    print("\n6. Testing metadata propagation...")
    custom_meta = {'author': 'Test', 'version': 1}
    config = TextSpinnerConfig(chunk_by=None)
    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': 'Test',
        'source': 'meta.txt',
        'metadata': custom_meta
    })
    assert shards[0].metadata['author'] == 'Test'
    assert shards[0].metadata['version'] == 1
    assert 'char_count' in shards[0].metadata
    print(f"   Metadata: {shards[0].metadata}")
    print("   [OK] Metadata propagation test passed")

    # Test 7: Entity extraction
    print("\n7. Testing entity extraction...")
    text = "Stanford University and MIT collaborate. Google Brain and OpenAI research."
    config = TextSpinnerConfig(extract_entities=True)
    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'entities.txt'})
    print(f"   Entities extracted: {shards[0].entities}")
    assert len(shards[0].entities) > 0
    print("   [OK] Entity extraction test passed")

    # Test 8: Error handling
    print("\n8. Testing error handling...")
    try:
        await spinner.spin({'source': 'no_text.txt'})  # Missing 'text' key
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Correctly caught error: {e}")
        print("   [OK] Error handling test passed")

    print("\n" + "="*60)
    print("All tests passed! [SUCCESS]")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(test_basic_functionality())
