"""
Specialized OCR Spinners Demo
==============================

Demonstrates all specialized OCR spinners:
1. Handwritten Spinner - Handwritten notes
2. Receipt Spinner - Receipt parsing
3. DeepSeek OCR Spinner - General documents
4. Integration with HoloLoom memory

Shows:
- Protocol-based OCR architecture
- Automatic backend fallback
- Structured data extraction
- Task detection (handwritten)
- Financial parsing (receipts)
- Memory integration

Usage:
    python demos/demo_specialized_ocr_spinners.py

Author: Claude Code
Date: January 2025
"""

import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# Demo 1: Handwritten Notes
# ============================================================================

async def demo_handwritten_notes():
    """Demonstrate handwritten note extraction with task detection."""
    print("=" * 70)
    print("Demo 1: Handwritten Notes with Task Detection")
    print("=" * 70)

    from HoloLoom.spinningWheel import HandwrittenSpinner

    # Create sample handwritten note (simulated with print text)
    print("\nCreating sample handwritten note...")
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    # Simulate handwritten style (all content as if handwritten)
    note_content = """Meeting Notes - Jan 15, 2025

Attendees:
- John Smith
- Sarah Johnson
- Mike Davis

TODO:
[ ] Review project proposal
[ ] Schedule follow-up meeting
[ ] Send summary email

Key Points:
• Budget approval needed by Feb 1
• Team expanding to 5 members
• New office location: Building 3

Action Items:
1. John to finalize budget
2. Sarah to hire developers
3. Mike to find office space

/s/ John Smith"""

    draw.text((50, 50), note_content, fill='black', font=font)

    sample_path = Path("temp_handwritten_note.png")
    img.save(sample_path)

    # Process with HandwrittenSpinner
    print("\nProcessing with HandwrittenSpinner...")
    spinner = HandwrittenSpinner(detect_tasks=True)

    result = await spinner.spin(sample_path)

    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Shards: {result.shard_count}")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")

    if result.shards:
        shard = result.shards[0]

        print(f"\nNote Analysis:")
        print(f"  Has title: {shard.metadata.get('has_title')}")
        print(f"  Has bullets: {shard.metadata.get('has_bullets')}")
        print(f"  Has numbering: {shard.metadata.get('has_numbering')}")
        print(f"  Has signatures: {shard.metadata.get('has_signatures')}")
        print(f"  Sections: {shard.metadata.get('section_count')}")

        print(f"\nDetected Tasks ({shard.metadata.get('task_count')}):")
        for task in shard.metadata.get('detected_tasks', []):
            print(f"    - {task}")

        print(f"\nExtracted Entities:")
        for entity in shard.entities[:5]:
            print(f"    - {entity}")

        print(f"\nImportance: {shard.metadata.get('importance_score'):.2f}")
        print(f"  Reason: {shard.metadata.get('importance_reason')}")

    # Cleanup
    sample_path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 2: Receipt Parsing
# ============================================================================

async def demo_receipt_parsing():
    """Demonstrate receipt parsing with structured data extraction."""
    print("\n" + "=" * 70)
    print("Demo 2: Receipt Parsing with Structured Data")
    print("=" * 70)

    from HoloLoom.spinningWheel import ReceiptSpinner

    # Create sample receipt
    print("\nCreating sample receipt...")
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 20)
    except:
        title_font = ImageFont.load_default()
        body_font = title_font

    receipt_content = """WHOLE FOODS MARKET
123 Main Street
San Francisco, CA 94102

01/15/2025    14:23

Organic Bananas         $3.99
Almond Milk             $4.49
Whole Wheat Bread       $5.99
Free Range Eggs         $6.99
Greek Yogurt            $7.99
Fresh Salmon            $15.99

Subtotal               $45.44
Tax (8.5%)              $3.86
Total                  $49.30

VISA ending in 4242
Thank you for shopping!"""

    y = 50
    for line in receipt_content.split('\n'):
        if line.strip().isupper() and len(line.strip()) < 30:
            draw.text((50, y), line, fill='black', font=title_font)
        else:
            draw.text((50, y), line, fill='black', font=body_font)
        y += 28

    sample_path = Path("temp_receipt.png")
    img.save(sample_path)

    # Process with ReceiptSpinner
    print("\nProcessing with ReceiptSpinner...")
    spinner = ReceiptSpinner(
        verify_calculations=True,
        categorize=True
    )

    result = await spinner.spin(sample_path)

    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")

    if result.shards:
        shard = result.shards[0]
        receipt_data = shard.metadata.get('receipt_data', {})

        print(f"\nReceipt Details:")
        print(f"  Merchant: {receipt_data.get('merchant')}")
        print(f"  Date: {receipt_data.get('date')}")
        print(f"  Time: {receipt_data.get('time')}")
        print(f"  Category: {receipt_data.get('category')}")

        print(f"\nItems ({receipt_data.get('item_count')}):")
        for item in receipt_data.get('items', [])[:5]:
            price = item.get('total_price')
            price_str = f"${price:.2f}" if price else "?"
            print(f"    - {item.get('name')}: {price_str}")

        print(f"\nFinancial Summary:")
        print(f"  Subtotal: ${receipt_data.get('subtotal', 0):.2f}")
        print(f"  Tax: ${receipt_data.get('tax', 0):.2f}")
        print(f"  Total: ${receipt_data.get('total', 0):.2f}")
        print(f"  Payment: {receipt_data.get('payment_method', 'N/A')}")

        print(f"\nCalculations verified: {shard.metadata.get('calculation_verified')}")
        print(f"Importance: {shard.metadata.get('importance_score'):.2f}")

    # Cleanup
    sample_path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 3: Batch Processing
# ============================================================================

async def demo_batch_processing():
    """Demonstrate batch processing of mixed document types."""
    print("\n" + "=" * 70)
    print("Demo 3: Batch Processing - Mixed Documents")
    print("=" * 70)

    from HoloLoom.spinningWheel import HandwrittenSpinner, ReceiptSpinner
    from HoloLoom.spinningWheel.ocr_backends import get_all_available_backends
    import time

    # Create multiple sample documents
    print("\nCreating batch of mixed documents...")
    sample_paths = []

    # Create 2 notes and 2 receipts
    for i in range(2):
        # Note
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        draw.text((50, 50), f"Note {i+1}\n\nTODO: Task {i+1}\n\nImportant reminder", fill='black', font=font)

        path = Path(f"temp_note_{i}.png")
        img.save(path)
        sample_paths.append(('note', path))

        # Receipt
        img = Image.new('RGB', (600, 400), color='white')
        draw.text((50, 50), f"STORE {i+1}\n\nItem A: $10.00\nItem B: $15.00\n\nTotal: $25.00", fill='black', font=font)

        path = Path(f"temp_receipt_{i}.png")
        img.save(path)
        sample_paths.append(('receipt', path))

    # Get OCR backend once (shared across spinners)
    ocr_chain = get_all_available_backends()
    print(f"\nUsing OCR backend: {ocr_chain.get_name()} ({ocr_chain.get_quality().value})")

    # Process all documents
    print(f"\nProcessing {len(sample_paths)} documents...")

    start = time.time()
    all_shards = []

    for doc_type, path in sample_paths:
        if doc_type == 'note':
            spinner = HandwrittenSpinner()
        else:
            spinner = ReceiptSpinner()

        result = await spinner.spin(path)
        all_shards.extend(result.shards)

    elapsed = (time.time() - start) * 1000

    print(f"\nBatch Processing Complete:")
    print(f"  Total documents: {len(sample_paths)}")
    print(f"  Total shards: {len(all_shards)}")
    print(f"  Total time: {elapsed:.1f}ms")
    print(f"  Per-document: {elapsed/len(sample_paths):.1f}ms")
    print(f"  Throughput: {len(sample_paths)/(elapsed/1000):.1f} docs/sec")

    print(f"\nProcessed:")
    for i, shard in enumerate(all_shards):
        doc_type = shard.metadata.get('note_type') or shard.metadata.get('document_type')
        importance = shard.metadata.get('importance_score', 0)
        print(f"  {i+1}. {doc_type}: importance={importance:.2f}")

    # Cleanup
    for _, path in sample_paths:
        path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 4: HoloLoom Memory Integration
# ============================================================================

async def demo_memory_integration():
    """Demonstrate integration with HoloLoom memory system."""
    print("\n" + "=" * 70)
    print("Demo 4: HoloLoom Memory Integration")
    print("=" * 70)

    from HoloLoom import HoloLoom
    from HoloLoom.spinningWheel import HandwrittenSpinner, ReceiptSpinner

    # Create sample documents
    print("\nCreating sample documents for memory...")

    documents = []

    # Handwritten note
    img = Image.new('RGB', (700, 500), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    note_text = """Project Planning - Q1 2025

Team Goals:
- Launch new product feature
- Improve performance by 30%
- Expand user base to 1M

TODO:
[ ] Complete design mockups
[ ] Set up CI/CD pipeline
[ ] Hire 2 engineers"""

    draw.text((50, 50), note_text, fill='black', font=font)

    note_path = Path("temp_project_note.png")
    img.save(note_path)
    documents.append(('note', note_path, HandwrittenSpinner()))

    # Receipt
    img = Image.new('RGB', (600, 700), color='white')
    draw = ImageDraw.Draw(img)

    receipt_text = """TECH SUPPLY CO.
01/15/2025

Laptop Stand         $45.99
USB-C Hub            $29.99
Wireless Mouse       $24.99

Subtotal            $100.97
Tax                   $8.58
Total               $109.55

VISA ending in 1234"""

    y = 50
    for line in receipt_text.split('\n'):
        draw.text((50, y), line, fill='black', font=font)
        y += 30

    receipt_path = Path("temp_tech_receipt.png")
    img.save(receipt_path)
    documents.append(('receipt', receipt_path, ReceiptSpinner()))

    # Process and store in memory
    print("\nProcessing documents and storing in HoloLoom...")

    async with HoloLoom() as loom:
        # Extract and experience
        for doc_type, path, spinner in documents:
            result = await spinner.spin(path)

            for shard in result.shards:
                await loom.experience(shard.text)

            print(f"  ✓ Processed {doc_type}")

        # Query memory
        print("\nQuerying memory...")

        queries = [
            "What are the project goals?",
            "What tasks need to be done?",
            "What was purchased from Tech Supply?",
            "How much was spent?"
        ]

        for query in queries:
            memories = await loom.recall(query)

            print(f"\nQuery: {query}")
            if memories:
                print(f"  Found {len(memories)} memories")
                best = memories[0]
                preview = best.text[:100].replace('\n', ' ')
                print(f"  Best match: {preview}...")
            else:
                print(f"  No memories found")

        # Get metrics
        metrics = loom.get_metrics()
        print(f"\nMemory System Metrics:")
        print(f"  Active nodes: {metrics.get('activation', {}).get('active_nodes', 0)}")
        print(f"  Total memories: {metrics.get('graph_stats', {}).get('node_count', 0)}")

    # Cleanup
    for _, path, _ in documents:
        path.unlink()

    print("\n" + "-" * 70)


# ============================================================================
# Demo 5: OCR Backend Comparison
# ============================================================================

async def demo_backend_comparison():
    """Compare different OCR backends."""
    print("\n" + "=" * 70)
    print("Demo 5: OCR Backend Quality Comparison")
    print("=" * 70)

    from HoloLoom.spinningWheel.ocr_backends import get_all_available_backends

    # Get backend chain
    chain = get_all_available_backends()

    print(f"\nActive Backend: {chain.get_name()}")
    print(f"Quality: {chain.get_quality().value}")

    print(f"\nAvailable Backends in Chain:")
    for i, backend in enumerate(chain.available_backends):
        print(f"  {i+1}. {backend.get_name():15s} - {backend.get_quality().value:10s}")

    print("\nBackend Selection Logic:")
    print("  1. Try DeepSeek OCR (excellent quality, CUDA required)")
    print("  2. Fall back to Tesseract (good quality, CPU works)")
    print("  3. Final fallback to filename extraction (poor quality)")

    print("\nRecommendations:")
    print("  - Production: Use DeepSeek for best results")
    print("  - Development: Tesseract works well for testing")
    print("  - Handwriting: DeepSeek or cloud APIs (Azure, Google)")
    print("  - Receipts: Any backend works, DeepSeek best")

    print("\n" + "-" * 70)


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("Specialized OCR Spinners Demo")
    print("=" * 70)
    print("\nDemonstrating:")
    print("  1. Handwritten Notes - Task detection, entity extraction")
    print("  2. Receipt Parsing - Structured data, financial verification")
    print("  3. Batch Processing - Mixed document types")
    print("  4. Memory Integration - HoloLoom integration")
    print("  5. Backend Comparison - OCR quality comparison")

    try:
        await demo_handwritten_notes()
        await demo_receipt_parsing()
        await demo_batch_processing()
        await demo_memory_integration()
        await demo_backend_comparison()

        print("\n" + "=" * 70)
        print("All Demos Complete!")
        print("=" * 70)

        print("\nKey Takeaways:")
        print("  ✓ Protocol-based OCR enables multiple backends")
        print("  ✓ Automatic fallback ensures system never crashes")
        print("  ✓ Specialized spinners extract structured data")
        print("  ✓ Seamless integration with HoloLoom memory")
        print("  ✓ Task detection, financial parsing, entity extraction")

        print("\nNext Steps:")
        print("  - Add more OCR backends (Azure, AWS, Google)")
        print("  - Implement table extraction for complex documents")
        print("  - Add form field detection")
        print("  - Create document classification spinner")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
