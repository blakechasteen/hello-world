"""
Image Parser - Extract Content from Image Files
================================================
Parse image files and extract text, metadata, and embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not installed. Image parsing unavailable.")
    PIL_AVAILABLE = False
    Image = None

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    logger.warning("pytesseract not installed. OCR unavailable.")
    OCR_AVAILABLE = False
    pytesseract = None


async def parse_image(
    file_path: str,
    enable_ocr: bool = True,
    ocr_lang: str = "eng"
) -> str:
    """
    Extract text and metadata from image file.

    Args:
        file_path: Path to image file
        enable_ocr: Enable OCR text extraction
        ocr_lang: OCR language code

    Returns:
        Extracted text (or metadata string if no text found)

    Raises:
        RuntimeError: If PIL not available
        FileNotFoundError: If image file not found
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Image parsing requires PIL: pip install pillow")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    text_parts = []

    try:
        # Open image
        img = Image.open(str(path))

        # Add metadata
        text_parts.append(f"[Image: {path.name}]")
        text_parts.append(f"Format: {img.format}")
        text_parts.append(f"Size: {img.size[0]}x{img.size[1]}")
        text_parts.append(f"Mode: {img.mode}")

        # Extract EXIF metadata if available
        exif_data = img.getexif()
        if exif_data:
            text_parts.append("\nMetadata:")
            for tag_id, value in exif_data.items():
                # Only include common tags
                if tag_id in [271, 272, 274, 282, 283, 296, 306]:  # Make, Model, etc.
                    text_parts.append(f"  Tag {tag_id}: {value}")

        # OCR text extraction
        if enable_ocr and OCR_AVAILABLE:
            try:
                ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)
                if ocr_text and ocr_text.strip():
                    text_parts.append("\nExtracted Text:")
                    text_parts.append(ocr_text.strip())
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                text_parts.append("\n[OCR unavailable]")
        elif enable_ocr and not OCR_AVAILABLE:
            text_parts.append("\n[OCR requested but pytesseract not installed]")

        # Combine all parts
        full_text = "\n".join(text_parts)

        logger.info(f"Parsed image {path.name}: {len(full_text)} characters")

        return full_text

    except Exception as e:
        logger.error(f"Failed to parse image {file_path}: {e}")
        raise


async def get_image_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from image file.

    Args:
        file_path: Path to image file

    Returns:
        Dict with metadata (format, size, mode, exif)
    """
    if not PIL_AVAILABLE:
        return {}

    path = Path(file_path)
    if not path.exists():
        return {}

    try:
        img = Image.open(str(path))

        metadata = {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "width": img.size[0],
            "height": img.size[1]
        }

        # EXIF data
        exif_data = img.getexif()
        if exif_data:
            metadata["exif"] = dict(exif_data)

        return metadata

    except Exception as e:
        logger.error(f"Failed to extract metadata from {file_path}: {e}")
        return {}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 80)
        print("Image Parser Demo")
        print("=" * 80 + "\n")

        if not PIL_AVAILABLE:
            print("PIL not available")
            print("Install with: pip install pillow")
            return

        print("Image parser is ready")
        print("Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF")
        print("\nUsage:")
        print("  text = await parse_image('photo.jpg', enable_ocr=True)")
        print("  metadata = await get_image_metadata('photo.jpg')")
        print("\n✓ Demo complete!")

    asyncio.run(demo())
