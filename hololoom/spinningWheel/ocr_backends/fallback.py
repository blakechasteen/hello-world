"""
Fallback OCR Backend
====================

Last-resort OCR backend that extracts filename/metadata only.

Always available, provides minimal information when no OCR is available.
"""

from pathlib import Path
from typing import Any

from hololoom.spinningWheel.ocr_protocol import (
    BaseOCRBackend,
    OCROutputFormat,
    OCRQuality,
    OCRResult,
)


class FallbackOCRBackend(BaseOCRBackend):
    """
    Fallback OCR backend.

    Extracts filename and basic metadata when no real OCR is available.
    Always available, always returns success (with low confidence).
    """

    def __init__(self):
        """Initialize fallback backend."""
        super().__init__(name="fallback")

    def is_available(self) -> bool:
        """Always available."""
        return True

    def get_quality(self) -> OCRQuality:
        """Poor quality (no actual OCR)."""
        return OCRQuality.POOR

    async def _extract_text_impl(
        self,
        image: Any,
        output_format: OCROutputFormat,
        **kwargs
    ) -> OCRResult:
        """
        Extract minimal information from image.

        Returns filename and metadata only.
        """
        # Try to get filename
        filename = "unknown"

        if hasattr(image, 'filename'):
            filename = Path(image.filename).name
        elif isinstance(kwargs.get('source'), (str, Path)):
            filename = Path(kwargs['source']).name

        # Get image size
        image_size = (0, 0)
        if hasattr(image, 'size'):
            image_size = image.size

        # Generate placeholder text
        text = f"[Image: {filename}]"

        if output_format == OCROutputFormat.MARKDOWN:
            text = f"![{filename}]({filename})"

        return OCRResult(
            text=text,
            confidence=0.1,  # Very low confidence
            image_size=image_size,
            metadata={
                'fallback': True,
                'filename': filename
            }
        )
