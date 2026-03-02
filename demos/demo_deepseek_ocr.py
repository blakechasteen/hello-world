"""
DeepSeek OCR Demo
=================

Demonstrates DeepSeek OCR integration with HoloLoom.

Shows:
1. Protocol-based OCR architecture
2. Automatic backend fallback
3. Multiple output formats
4. Batch processing
5. HoloLoom memory integration

Usage:
    python demos/demo_deepseek_ocr.py
"""

import asyncio
from pathlib import Path


# ============================================================================
# Demo 1: Backend Chain with Fallback
# ============================================================================

async def demo_backend_chain():
    """Demonstrate automatic backend fallback."""
    print("=" * 70)
    print("Demo 1: OCR Backend Chain with Automatic Fallback")
    print("=" * 70)

    from hololoom.spinningWheel.ocr_backends import get_all_available_backends

    # Get chain of all available backends
    chain = get_all_available_backends()

    print(f"\nActive backend: {chain.get_name()}")
    print(f"Quality: {chain.get_quality().value}")
    print(f"Available: {chain.is_available()}")

    # Show all backends in chain
    print(f"\nBackend chain ({len(chain.available_backends)} available):")
    for i, backend in enumerate(chain.available_backends):
        print(f"  {i+1}. {backend.get_name():12s} - {backend.get_quality().value}")

    print("\n" + "-" * 70)


# ============================================================================
# Demo 2: Extract Text from Image
# ============================================================================

async def demo_extract_image():
    """Extract text from sample image."""
    print("\n" + "=" * 70)
    print("Demo 2: Extract Text from Image")
    print("=" * 70)

    from hololoom.spinningWheel.ocr_backends import get_best_available_backend
    from hololoom.spinningWheel.ocr_protocol import OCROutputFormat
    from PIL import Image, ImageDraw, ImageFont

    # Create sample image with text
    print("\nCreating sample image...")
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)

    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Draw title
    draw.text((50, 50), "Sample Document", fill='black', font=font)

    # Draw body text
    body_text = """
    This is a sample document for OCR testing.

    Key points:
    - High-quality text extraction
    - Markdown formatting support
    - Batch processing capability
    """

    draw.text((50, 120), body_text, fill='black', font=small_font)

    # Save sample
    sample_path = Path("temp_sample.png")
    img.save(sample_path)

    # Extract text
    print("\nExtracting text...")
    backend = get_best_available_backend()

    # Try different formats
    for format_type in [OCROutputFormat.TEXT, OCROutputFormat.MARKDOWN]:
        print(f"\n{format_type.value.upper()} format:")
        print("-" * 70)

        result = await backend.extract_text(sample_path, output_format=format_type)

        print(f"Backend: {result.backend}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        print(f"\nExtracted text:\n{result.text}")

    # Cleanup
    sample_path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 3: Batch Processing
# ============================================================================

async def demo_batch_processing():
    """Demonstrate batch OCR processing."""
    print("\n" + "=" * 70)
    print("Demo 3: Batch Processing")
    print("=" * 70)

    from hololoom.spinningWheel.ocr_backends import get_best_available_backend
    from PIL import Image, ImageDraw, ImageFont
    import time

    # Create multiple sample images
    print("\nCreating batch of sample images...")
    sample_paths = []

    for i in range(3):
        img = Image.new('RGB', (600, 200), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            font = ImageFont.load_default()

        draw.text((50, 80), f"Document {i+1}", fill='black', font=font)

        path = Path(f"temp_sample_{i}.png")
        img.save(path)
        sample_paths.append(path)

    # Batch extract
    print(f"\nBatch extracting from {len(sample_paths)} images...")
    backend = get_best_available_backend()

    start = time.time()
    results = await backend.batch_extract(sample_paths)
    elapsed = (time.time() - start) * 1000

    print(f"\nBatch processing complete:")
    print(f"  Total time: {elapsed:.1f}ms")
    print(f"  Per-image: {elapsed/len(sample_paths):.1f}ms")
    print(f"  Throughput: {len(sample_paths)/(elapsed/1000):.1f} images/sec")

    for i, result in enumerate(results):
        print(f"\n  Image {i+1}: {result.text.strip()[:50]}")
        print(f"    Confidence: {result.confidence:.2f}")

    # Cleanup
    for path in sample_paths:
        path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 4: Spinner Integration
# ============================================================================

async def demo_spinner_integration():
    """Demonstrate OCR spinner integration."""
    print("\n" + "=" * 70)
    print("Demo 4: DeepSeek OCR Spinner Integration")
    print("=" * 70)

    from hololoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRSpinner
    from PIL import Image, ImageDraw, ImageFont

    # Create sample document
    print("\nCreating sample document...")
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        body_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = title_font

    # Draw title
    draw.text((50, 50), "Research Paper", fill='black', font=title_font)

    # Draw abstract
    abstract = """Abstract:

This paper explores the integration of vision-language
models for document understanding. We demonstrate a
10x compression ratio while maintaining 97% accuracy.

Key findings:
1. Multi-scale processing improves quality
2. Structured output enables downstream tasks
3. Batch processing achieves 2500 tokens/sec"""

    draw.text((50, 150), abstract, fill='black', font=body_font)

    sample_path = Path("temp_paper.png")
    img.save(sample_path)

    # Process with spinner
    print("\nProcessing with DeepSeekOCRSpinner...")
    spinner = DeepSeekOCRSpinner()

    result = await spinner.spin(sample_path)

    print(f"\nSpinner result:")
    print(f"  Success: {result.success}")
    print(f"  Shards: {result.shard_count}")
    print(f"  Entities: {result.entity_count}")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")
    print(f"  Average importance: {result.avg_importance:.2f}")

    if result.shards:
        shard = result.shards[0]
        print(f"\nFirst shard:")
        print(f"  Text: {shard.text[:200]}...")
        print(f"  Backend: {shard.metadata.get('backend')}")
        print(f"  Confidence: {shard.metadata.get('ocr_confidence', 0):.2f}")

    # Cleanup
    sample_path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 5: HoloLoom Memory Integration
# ============================================================================

async def demo_memory_integration():
    """Demonstrate integration with HoloLoom memory."""
    print("\n" + "=" * 70)
    print("Demo 5: HoloLoom Memory Integration")
    print("=" * 70)

    from hololoom import HoloLoom
    from hololoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRSpinner
    from PIL import Image, ImageDraw, ImageFont

    # Create sample documents
    print("\nCreating sample documents...")
    documents = [
        ("Machine Learning Basics", "Neural networks learn patterns from data through backpropagation."),
        ("Thompson Sampling", "Thompson Sampling balances exploration and exploitation using Bayesian priors."),
        ("Computer Vision", "Vision transformers process images as sequences of patches.")
    ]

    sample_paths = []

    for i, (title, content) in enumerate(documents):
        img = Image.new('RGB', (700, 300), color='white')
        draw = ImageDraw.Draw(img)

        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            body_font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            title_font = ImageFont.load_default()
            body_font = title_font

        draw.text((50, 50), title, fill='black', font=title_font)
        draw.text((50, 120), content, fill='black', font=body_font)

        path = Path(f"temp_doc_{i}.png")
        img.save(path)
        sample_paths.append(path)

    # Process with OCR
    print("\nExtracting text from documents...")
    spinner = DeepSeekOCRSpinner()

    all_shards = []
    for path in sample_paths:
        result = await spinner.spin(path)
        all_shards.extend(result.shards)

    print(f"Extracted {len(all_shards)} shards")

    # Store in HoloLoom memory
    print("\nStoring in HoloLoom memory...")
    async with HoloLoom() as loom:
        # Experience all documents
        for shard in all_shards:
            await loom.experience(shard.text)

        # Query memory
        print("\nQuerying memory...")
        queries = [
            "What is Thompson Sampling?",
            "How do neural networks learn?",
            "Tell me about vision transformers"
        ]

        for query in queries:
            memories = await loom.recall(query)

            print(f"\nQuery: {query}")
            print(f"  Retrieved {len(memories)} memories")

            if memories:
                best_match = memories[0]
                print(f"  Best match: {best_match.text[:100]}...")

    # Cleanup
    for path in sample_paths:
        path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("DeepSeek OCR + HoloLoom Integration Demo")
    print("=" * 70)

    try:
        await demo_backend_chain()
        await demo_extract_image()
        await demo_batch_processing()
        await demo_spinner_integration()
        await demo_memory_integration()

        print("\n" + "=" * 70)
        print("All demos complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())