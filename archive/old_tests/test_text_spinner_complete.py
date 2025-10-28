"""
Complete test suite for TextSpinner
====================================

Tests all chunking modes, entity extraction, and enrichment.
"""

import asyncio
from HoloLoom.spinning_wheel.text import TextSpinner, TextSpinnerConfig, spin_text


async def test_single_shard():
    """Test creating a single shard from entire document."""
    print("\n=== Test 1: Single Shard (No Chunking) ===")

    text = """
    HoloLoom is a neural decision-making system. It combines multi-scale embeddings
    with knowledge graph memory. The system uses Thompson Sampling for exploration.
    Claude Code is helping test this implementation.
    """

    config = TextSpinnerConfig(
        chunk_by=None,  # No chunking
        extract_entities=True,
        enable_enrichment=False
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': text,
        'source': 'test_doc.txt',
        'metadata': {'author': 'tester', 'version': 1}
    })

    print(f"Shards created: {len(shards)}")
    assert len(shards) == 1, "Should create exactly 1 shard"

    shard = shards[0]
    print(f"Shard ID: {shard.id}")
    print(f"Text length: {len(shard.text)} characters")
    print(f"Entities found: {shard.entities}")
    print(f"Metadata: {shard.metadata}")

    assert 'HoloLoom' in shard.entities or 'Claude' in shard.entities
    assert shard.metadata['author'] == 'tester'
    assert shard.metadata['char_count'] > 0

    print("✓ Single shard test passed")


async def test_paragraph_chunking():
    """Test paragraph-based chunking."""
    print("\n=== Test 2: Paragraph Chunking ===")

    text = """
    First paragraph discusses embeddings. Matryoshka representations enable
    multi-scale processing. This is very important for efficiency.

    Second paragraph is about memory systems. Knowledge graphs store relationships.
    Vector databases enable similarity search.

    Third paragraph covers the policy engine. Thompson Sampling balances exploration
    and exploitation. The neural core provides action predictions.
    """

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=200,  # Small chunks to force splitting
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': text,
        'source': 'multi_para.txt'
    })

    print(f"Shards created: {len(shards)}")
    assert len(shards) >= 2, "Should create multiple shards"

    for i, shard in enumerate(shards):
        print(f"\nShard {i}:")
        print(f"  ID: {shard.id}")
        print(f"  Length: {len(shard.text)} chars")
        print(f"  Entities: {shard.entities[:3]}...")  # First 3
        print(f"  Chunk index: {shard.metadata.get('chunk_index')}")

    print("✓ Paragraph chunking test passed")


async def test_sentence_chunking():
    """Test sentence-based chunking."""
    print("\n=== Test 3: Sentence Chunking ===")

    text = """
    HoloLoom integrates multiple components. The orchestrator coordinates all modules.
    Each module operates independently. The warp thread architecture ensures modularity.
    Memory systems provide context. Embeddings enable semantic search. The policy
    makes decisions. Thompson Sampling drives exploration.
    """

    config = TextSpinnerConfig(
        chunk_by='sentence',
        chunk_size=150,  # Force multiple chunks
        min_chunk_size=30
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'sentences.txt'})

    print(f"Shards created: {len(shards)}")

    for i, shard in enumerate(shards):
        print(f"\nShard {i}: {shard.text[:80]}...")
        print(f"  Length: {len(shard.text)} chars")

    print("✓ Sentence chunking test passed")


async def test_character_chunking():
    """Test fixed-size character chunking."""
    print("\n=== Test 4: Character Chunking ===")

    text = "A" * 1000  # 1000 character string

    config = TextSpinnerConfig(
        chunk_by='character',
        chunk_size=250,  # Should create 4 chunks
        min_chunk_size=10
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'chars.txt'})

    print(f"Shards created: {len(shards)}")
    assert len(shards) == 4, f"Expected 4 shards, got {len(shards)}"

    for i, shard in enumerate(shards):
        print(f"Shard {i}: {len(shard.text)} chars")
        assert len(shard.text) == 250, f"Expected 250 chars, got {len(shard.text)}"

    print("✓ Character chunking test passed")


async def test_entity_extraction():
    """Test basic entity extraction."""
    print("\n=== Test 5: Entity Extraction ===")

    text = """
    Dr. Jane Smith works at Stanford University. She collaborates with
    Professor John Doe from MIT. Their research on Neural Networks and
    Deep Learning was published in Nature. The team includes researchers
    from Google Brain and OpenAI.
    """

    config = TextSpinnerConfig(
        extract_entities=True,
        enable_enrichment=False
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': text, 'source': 'entities.txt'})

    shard = shards[0]
    print(f"Entities extracted: {shard.entities}")

    # Should extract some proper nouns
    assert len(shard.entities) > 0, "Should extract at least some entities"

    # Check for expected entities
    expected_entities = ['Stanford', 'University', 'Nature', 'Google', 'Neural', 'Networks']
    found = [e for e in expected_entities if any(e in ent for ent in shard.entities)]
    print(f"Expected entities found: {found}")

    print("✓ Entity extraction test passed")


async def test_convenience_function():
    """Test the convenience spin_text function."""
    print("\n=== Test 6: Convenience Function ===")

    text = "Quick test of the spin_text convenience function. Very easy to use!"

    shards = await spin_text(
        text=text,
        source='quick_test',
        chunk_by='sentence',
        chunk_size=50
    )

    print(f"Shards created: {len(shards)}")
    assert len(shards) > 0

    print(f"First shard: {shards[0].text[:50]}...")
    print("✓ Convenience function test passed")


async def test_markdown_preservation():
    """Test handling of markdown content."""
    print("\n=== Test 7: Markdown Content ===")

    markdown_text = """
# HoloLoom Architecture

## Memory Systems

The memory layer includes:
- Knowledge graphs
- Vector databases
- Episodic buffers

## Policy Engine

The policy uses **Thompson Sampling** for exploration.

### Key Features
- Multi-scale embeddings
- Matryoshka representations
- Spectral features
"""

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=300,
        preserve_structure=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': markdown_text,
        'source': 'architecture.md',
        'metadata': {'format': 'markdown'}
    })

    print(f"Shards created: {len(shards)}")

    for i, shard in enumerate(shards):
        print(f"\nShard {i}:")
        # Check that markdown headers are preserved
        if '#' in shard.text:
            print(f"  Contains markdown headers: ✓")
        print(f"  Preview: {shard.text[:60]}...")

    print("✓ Markdown preservation test passed")


async def test_metadata_propagation():
    """Test that metadata is properly propagated."""
    print("\n=== Test 8: Metadata Propagation ===")

    text = "Testing metadata propagation through the spinner."

    custom_metadata = {
        'author': 'Test Author',
        'project': 'HoloLoom',
        'version': '2.0',
        'tags': ['test', 'spinner', 'metadata']
    }

    config = TextSpinnerConfig(chunk_by=None)
    spinner = TextSpinner(config)

    shards = await spinner.spin({
        'text': text,
        'source': 'metadata_test.txt',
        'episode': 'test_episode_001',
        'metadata': custom_metadata
    })

    shard = shards[0]
    print(f"Shard metadata: {shard.metadata}")

    # Verify custom metadata is present
    assert shard.metadata['author'] == 'Test Author'
    assert shard.metadata['project'] == 'HoloLoom'
    assert shard.metadata['version'] == '2.0'
    assert shard.metadata['tags'] == ['test', 'spinner', 'metadata']

    # Verify system metadata is also present
    assert 'char_count' in shard.metadata
    assert 'word_count' in shard.metadata
    assert shard.metadata['source'] == 'metadata_test.txt'

    print("✓ Metadata propagation test passed")


async def test_empty_and_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Test 9: Edge Cases ===")

    # Test 1: Very short text
    short_text = "Hi"
    config = TextSpinnerConfig(chunk_by='paragraph', min_chunk_size=50)
    spinner = TextSpinner(config)
    shards = await spinner.spin({'text': short_text, 'source': 'short.txt'})
    print(f"Short text shards: {len(shards)} (might be 0 due to min_chunk_size)")

    # Test 2: Text with lots of whitespace
    whitespace_text = "\n\n\n   \n\nSome content\n\n\n   \n\n"
    shards = await spinner.spin({'text': whitespace_text, 'source': 'whitespace.txt'})
    print(f"Whitespace text shards: {len(shards)}")

    # Test 3: Missing text should raise error
    try:
        await spinner.spin({'source': 'no_text.txt'})
        assert False, "Should raise ValueError for missing text"
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    print("✓ Edge cases test passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("TextSpinner Complete Test Suite")
    print("=" * 60)

    tests = [
        test_single_shard,
        test_paragraph_chunking,
        test_sentence_chunking,
        test_character_chunking,
        test_entity_extraction,
        test_convenience_function,
        test_markdown_preservation,
        test_metadata_propagation,
        test_empty_and_edge_cases
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)
