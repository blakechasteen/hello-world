"""
PDF Spinner Demo
================

Demonstrates the PDFSpinner for ingesting PDF documents.

Features shown:
- Text extraction from pages
- Metadata extraction
- Multiple chunking strategies
- Importance scoring
- Section detection

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.spinningWheel.pdf_spinner import PDFSpinner, PDFPage, PDFMetadata


async def demo_pdf_spinner():
    """Run PDF spinner demo."""
    print("=" * 70)
    print("PDF Spinner Demo")
    print("=" * 70)

    # Create spinners with different strategies
    spinner_page = PDFSpinner(chunk_strategy="page")
    spinner_section = PDFSpinner(chunk_strategy="section")

    # Check availability
    if not spinner_page.is_available():
        print("Error: PyPDF2 not available")
        print("Install with: pip install PyPDF2")
        return

    print("\nSpinner capabilities:")
    caps = spinner_page.get_capabilities()
    print(f"  - Basic processing: {caps.basic_processing}")
    print(f"  - Streaming: {caps.streaming}")
    print(f"  - Importance scoring: {caps.importance_scoring}")
    print(f"  - Multimodal: {caps.multimodal}")

    print("\nChunking strategies available:")
    print("  - page: One shard per page")
    print("  - section: Group pages by detected sections")
    print("  - custom: Custom grouping (advanced)")

    print("\nUsage examples:")
    print()
    print("# Extract by page:")
    print("spinner = PDFSpinner(chunk_strategy='page')")
    print("result = await spinner.spin('/path/to/document.pdf')")
    print()
    print("# Extract by section:")
    print("spinner = PDFSpinner(chunk_strategy='section')")
    print("result = await spinner.spin('/path/to/document.pdf')")
    print()
    print("# Limit pages:")
    print("spinner = PDFSpinner(max_pages=10)")
    print("result = await spinner.spin('/path/to/document.pdf')")


async def demo_importance_scoring():
    """Demonstrate page importance scoring."""
    print("\n" + "=" * 70)
    print("Page Importance Scoring Demo")
    print("=" * 70)

    spinner = PDFSpinner()

    # Example metadata
    metadata = PDFMetadata(
        title="Machine Learning: A Probabilistic Perspective",
        author="Kevin P. Murphy",
        num_pages=1024
    )

    # Example pages
    pages = [
        PDFPage(
            page_number=1,
            text="Title page content",
            has_images=True,
            image_count=1
        ),
        PDFPage(
            page_number=2,
            text="Table of Contents",
            has_images=False
        ),
        PDFPage(
            page_number=3,
            text="Chapter 1: Introduction\n\n" + ("Lorem ipsum dolor sit amet. " * 50),
            has_images=False,
            table_count=1
        ),
        PDFPage(
            page_number=4,
            text="Mathematical notation and formulas " * 50,
            has_images=True,
            image_count=3,
            table_count=2
        ),
    ]

    print(f"\nDocument: {metadata.title}")
    print(f"Author: {metadata.author}")
    print(f"Total pages: {metadata.num_pages}")
    print("\nScoring example pages:\n")

    for page in pages:
        score = spinner._score_page_importance(page, metadata)
        print(f"Page {page.page_number}:")
        print(f"  Content length: {len(page.text)} chars")
        print(f"  Has images: {page.has_images} ({page.image_count})")
        print(f"  Has tables: {page.table_count > 0} ({page.table_count})")
        print(f"  Importance Score: {score.score:.2f}")
        print(f"  Reason: {score.reason}")
        print()


async def demo_section_detection():
    """Demonstrate section detection."""
    print("\n" + "=" * 70)
    print("Section Detection Demo")
    print("=" * 70)

    spinner = PDFSpinner(chunk_strategy="section")

    # Example document structure
    pages = [
        PDFPage(page_number=1, text="CHAPTER 1: INTRODUCTION\n\nContent of chapter 1"),
        PDFPage(page_number=2, text="More content from chapter 1"),
        PDFPage(page_number=3, text="CHAPTER 2: METHODOLOGY\n\nContent of chapter 2"),
        PDFPage(page_number=4, text="More content from chapter 2"),
        PDFPage(page_number=5, text="CHAPTER 3: RESULTS\n\nContent of chapter 3"),
    ]

    print("\nDetecting sections in document:")
    sections = spinner._detect_sections(pages)

    for i, section in enumerate(sections):
        print(f"\nSection {i+1}: Pages {section[0].page_number}-{section[-1].page_number}")
        print(f"  Contains {len(section)} page(s)")
        print(f"  First 50 chars: {section[0].text[:50]}...")


async def demo_metadata_extraction():
    """Demonstrate metadata extraction."""
    print("\n" + "=" * 70)
    print("Metadata Extraction Demo")
    print("=" * 70)

    print("\nPDF Metadata Example:")
    print()

    metadata = PDFMetadata(
        title="Deep Learning",
        author="Ian Goodfellow, Yoshua Bengio, Aaron Courville",
        subject="Machine Learning",
        creator="LaTeX",
        producer="pdfTeX",
        num_pages=800
    )

    print(f"  Title: {metadata.title}")
    print(f"  Author: {metadata.author}")
    print(f"  Subject: {metadata.subject}")
    print(f"  Creator: {metadata.creator}")
    print(f"  Producer: {metadata.producer}")
    print(f"  Pages: {metadata.num_pages}")

    print("\nThis metadata is automatically extracted when spinning a PDF")
    print("and included in each shard's metadata dictionary.")


async def main():
    """Run all demos."""
    await demo_pdf_spinner()
    await demo_importance_scoring()
    await demo_section_detection()
    await demo_metadata_extraction()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNotes:")
    print("  - Install PyPDF2: pip install PyPDF2")
    print("  - For image extraction: pip install Pillow")
    print("  - For OCR on scanned PDFs: pip install pytesseract")
    print("  - Point to actual PDF files to process them")


if __name__ == "__main__":
    asyncio.run(main())
