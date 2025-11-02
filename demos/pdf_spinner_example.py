"""
PDF Spinner Usage Examples
===========================

Demonstrates how to use PDFSpinner to ingest PDF documents
into HoloLoom memory.

Requirements:
    pip install PyPDF2

Optional (for advanced features):
    pip install pdfplumber  # Better table extraction
    pip install pytesseract  # OCR support

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
from typing import List

from HoloLoom.spinningWheel.pdf_spinner import (
    PDFSpinner,
    spin_pdf,
    create_pdf_scorer
)
from HoloLoom.spinningWheel.protocol import SpinResult
from HoloLoom.documentation.types import MemoryShard


# =============================================================================
# Example 1: Basic PDF Ingestion
# =============================================================================

async def example_1_basic_pdf():
    """
    Basic example: Spin a single PDF into MemoryShards

    Use case: Import research paper for query/search
    """
    print("=" * 60)
    print("Example 1: Basic PDF Ingestion")
    print("=" * 60)

    # Initialize spinner
    spinner = PDFSpinner(
        importance_threshold=0.3,  # Filter out low-importance pages
        chunk_by_page=False  # Treat whole document as single shard
    )

    try:
        # Spin a single PDF
        pdf_path = Path("./data/research_paper.pdf")
        result: SpinResult = await spinner.spin(pdf_path)

        print(f"\nProcessed: {result.items_processed} pages")
        print(f"Filtered: {result.items_filtered} low-importance pages")
        print(f"Shards created: {len(result.shards)}")

        # Inspect first shard
        if result.shards:
            shard = result.shards[0]
            print(f"\nFirst shard preview:")
            print(f"  ID: {shard.id}")
            print(f"  Episode: {shard.episode}")
            print(f"  Text: {shard.text[:100]}...")
            print(f"  Entities: {shard.entities}")
            print(f"  Motifs: {shard.motifs}")
            print(f"  Importance: {shard.metadata.get('importance_score', 'N/A')}")

    except FileNotFoundError:
        print(f"\nNote: Create {pdf_path} to run this example")


# =============================================================================
# Example 2: Page-Based vs Document-Based Chunking
# =============================================================================

async def example_2_chunking_modes():
    """
    Demonstrate page-based vs document-based chunking

    Use case: Choose granularity for retrieval
    """
    print("=" * 60)
    print("Example 2: Chunking Modes")
    print("=" * 60)

    pdf_path = Path("./data/textbook.pdf")

    # Mode 1: Page-based (one shard per page)
    spinner_page = PDFSpinner(
        importance_threshold=0.0,  # Include all
        chunk_by_page=True
    )

    try:
        result_page = await spinner_page.spin(pdf_path)
        print(f"\nPage-based mode:")
        print(f"  Shards created: {len(result_page.shards)}")
        print(f"  Pages processed: {result_page.items_processed}")

        # Mode 2: Document-based (one shard for whole document)
        spinner_doc = PDFSpinner(
            importance_threshold=0.0,
            chunk_by_page=False
        )

        result_doc = await spinner_doc.spin(pdf_path)
        print(f"\nDocument-based mode:")
        print(f"  Shards created: {len(result_doc.shards)}")
        print(f"  Total text length: {len(result_doc.shards[0].text) if result_doc.shards else 0}")

    except FileNotFoundError:
        print(f"\nNote: Create {pdf_path} to run this example")


# =============================================================================
# Example 3: Importance Filtering
# =============================================================================

async def example_3_importance_filtering():
    """
    Demonstrate importance filtering at different thresholds

    Use case: Quality vs. quantity tradeoff
    """
    print("=" * 60)
    print("Example 3: Importance Filtering")
    print("=" * 60)

    pdf_path = Path("./data/technical_report.pdf")
    thresholds = [0.2, 0.5, 0.8]

    for threshold in thresholds:
        spinner = PDFSpinner(
            importance_threshold=threshold,
            chunk_by_page=True  # Per-page for granular filtering
        )

        try:
            result = await spinner.spin(pdf_path)

            print(f"\nThreshold {threshold}:")
            print(f"  Pages processed: {result.items_processed}")
            print(f"  Shards created: {len(result.shards)}")
            print(f"  Filter rate: {result.items_filtered / result.items_processed * 100:.1f}%")

            # Show importance distribution
            if result.shards:
                scores = [s.metadata.get('importance_score', 0) for s in result.shards]
                avg_score = sum(scores) / len(scores)
                print(f"  Average importance: {avg_score:.3f}")

        except FileNotFoundError:
            print(f"\nNote: Create {pdf_path} to run this example")
            break


# =============================================================================
# Example 4: Directory Batch Processing
# =============================================================================

async def example_4_batch_processing():
    """
    Process entire directory of PDFs

    Use case: Bulk import of document library
    """
    print("=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)

    spinner = PDFSpinner(
        importance_threshold=0.4,  # Higher threshold for bulk
        chunk_by_page=False  # Document-level for efficiency
    )

    docs_dir = Path("./data/papers/")

    try:
        # Process all PDFs in directory
        result = await spinner.spin_directory(docs_dir, recursive=True)

        print(f"\nProcessed {result.items_processed} documents")
        print(f"Total shards: {len(result.shards)}")

        # Group shards by document
        docs = {}
        for shard in result.shards:
            doc_name = shard.metadata.get('file_name', 'Unknown')
            docs[doc_name] = docs.get(doc_name, 0) + 1

        print("\nShards per document:")
        for doc_name, count in sorted(docs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_name}: {count}")

    except FileNotFoundError:
        print(f"\nNote: Create {docs_dir} with PDFs to run this example")


# =============================================================================
# Example 5: Streaming Large PDFs
# =============================================================================

async def example_5_streaming():
    """
    Stream MemoryShards for memory-efficient processing

    Use case: Large PDFs that don't fit in memory
    """
    print("=" * 60)
    print("Example 5: Streaming Large PDFs")
    print("=" * 60)

    spinner = PDFSpinner(
        importance_threshold=0.3,
        chunk_by_page=True  # Stream page-by-page
    )

    pdf_path = Path("./data/large_manual.pdf")

    try:
        shard_count = 0

        # Process shards one at a time
        async for shard in spinner.spin_stream(pdf_path, batch_size=10):
            shard_count += 1

            # Process shard (e.g., add to memory, index, etc.)
            # await memory.add_shard(shard)

            # Progress indicator
            if shard_count % 10 == 0:
                print(f"  Processed {shard_count} pages...")

        print(f"\nTotal pages streamed: {shard_count}")

    except FileNotFoundError:
        print(f"\nNote: Create {pdf_path} to run this example")


# =============================================================================
# Example 6: Custom Importance Scoring
# =============================================================================

async def example_6_custom_scoring():
    """
    Customize importance scoring for domain-specific needs

    Use case: Prioritize specific document sections or keywords
    """
    print("=" * 60)
    print("Example 6: Custom Importance Scoring")
    print("=" * 60)

    # Create custom scorer for scientific papers
    scorer = create_pdf_scorer()
    scorer.add_technical_terms({
        'hypothesis', 'methodology', 'results', 'conclusion',
        'p-value', 'statistical', 'correlation', 'regression',
        'experiment', 'dataset', 'baseline', 'performance'
    })

    spinner = PDFSpinner(
        importance_threshold=0.3,
        chunk_by_page=True
    )

    # Override scorer
    spinner.importance_scorer = scorer

    pdf_path = Path("./data/scientific_paper.pdf")

    try:
        result = await spinner.spin(pdf_path)

        print(f"\nProcessed: {result.items_processed} pages")
        print(f"High-priority pages: {len(result.shards)}")

        # Show methodology/results pages
        important_pages = [
            s for s in result.shards
            if any(term in s.text.lower()
                   for term in ['methodology', 'results', 'conclusion'])
        ]

        print(f"\nKey sections found: {len(important_pages)}")
        for shard in important_pages[:3]:  # Show first 3
            section = shard.text.split('\n')[0][:50]  # First line
            print(f"  - Page {shard.metadata.get('page_number', '?')}: {section}...")

    except FileNotFoundError:
        print(f"\nNote: Create {pdf_path} to run this example")


# =============================================================================
# Example 7: OCR Support for Scanned PDFs
# =============================================================================

async def example_7_ocr():
    """
    OCR support for scanned/image-based PDFs

    Use case: Extract text from scanned documents

    Requires: pip install pytesseract
    """
    print("=" * 60)
    print("Example 7: OCR Support")
    print("=" * 60)

    spinner = PDFSpinner(
        importance_threshold=0.3,
        chunk_by_page=True,
        enable_ocr=True  # Enable OCR for scanned pages
    )

    pdf_path = Path("./data/scanned_document.pdf")

    try:
        result = await spinner.spin(pdf_path)

        print(f"\nProcessed: {result.items_processed} pages")
        print(f"OCR applied to: {result.metadata.get('ocr_pages', 0)} pages")
        print(f"Shards created: {len(result.shards)}")

        # Check which pages needed OCR
        ocr_shards = [
            s for s in result.shards
            if s.metadata.get('used_ocr', False)
        ]

        print(f"\nPages requiring OCR: {len(ocr_shards)}")
        if ocr_shards:
            print(f"  Sample OCR text: {ocr_shards[0].text[:100]}...")

    except ImportError:
        print("\nNote: Install pytesseract for OCR support:")
        print("  pip install pytesseract")
        print("  # Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract")
    except FileNotFoundError:
        print(f"\nNote: Create {pdf_path} to run this example")


# =============================================================================
# Main Runner
# =============================================================================

async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("PDF Spinner Examples")
    print("=" * 60 + "\n")

    # NOTE: Update these with your actual PDF paths
    print("SETUP REQUIRED:")
    print("1. Install PyPDF2: pip install PyPDF2")
    print("2. Optional: pip install pdfplumber (better table extraction)")
    print("3. Optional: pip install pytesseract (OCR support)")
    print("4. Create ./data/ directory with sample PDFs\n")

    examples = [
        ("Basic PDF Ingestion", example_1_basic_pdf),
        ("Chunking Modes", example_2_chunking_modes),
        ("Importance Filtering", example_3_importance_filtering),
        ("Batch Processing", example_4_batch_processing),
        ("Streaming Large PDFs", example_5_streaming),
        ("Custom Importance Scoring", example_6_custom_scoring),
        ("OCR Support", example_7_ocr),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nUncomment examples in main() to run them individually.")

    # Uncomment to run examples:
    # await example_1_basic_pdf()
    # await example_2_chunking_modes()
    # await example_3_importance_filtering()
    # await example_4_batch_processing()
    # await example_5_streaming()
    # await example_6_custom_scoring()
    # await example_7_ocr()


if __name__ == "__main__":
    asyncio.run(main())
