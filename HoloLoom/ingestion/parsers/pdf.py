"""
PDF Parser - Extract Text from PDF Files
=========================================
Parse PDF files with optional OCR for image-based PDFs.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        logger.warning("pypdf/PyPDF2 not installed. PDF parsing unavailable.")
        PYPDF_AVAILABLE = False
        PdfReader = None


async def parse_pdf(file_path: str, enable_ocr: bool = False) -> str:
    """
    Extract text from PDF file.

    Args:
        file_path: Path to PDF file
        enable_ocr: Enable OCR for image-based PDFs (requires pytesseract)

    Returns:
        Extracted text

    Raises:
        RuntimeError: If PDF parsing unavailable
    """
    if not PYPDF_AVAILABLE:
        raise RuntimeError("PDF parsing requires pypdf or PyPDF2: pip install pypdf")

    path = Path(file_path)
    text_parts = []

    try:
        # Open PDF
        reader = PdfReader(str(path))

        # Extract text from each page
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                # Add page marker
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            else:
                # Empty page or image-based
                if enable_ocr:
                    # TODO: Add OCR with pytesseract
                    text_parts.append(f"[Page {page_num + 1}]\n[Image content - OCR not implemented]")
                else:
                    logger.debug(f"Page {page_num + 1} has no extractable text")

        # Combine pages
        full_text = "\n\n".join(text_parts)

        logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")

        return full_text

    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path}: {e}")
        raise


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("PDF Parser Demo")
        print("="*80 + "\n")

        if not PYPDF_AVAILABLE:
            print("pypdf/PyPDF2 not available")
            print("Install with: pip install pypdf")
            return

        # Create sample PDF (if pypdf available for writing)
        print("PDF parser is ready")
        print("Usage: text = await parse_pdf('document.pdf')")
        print("\n✓ Demo complete!")

    asyncio.run(demo())
