"""
Tesseract OCR Backend
=====================

Basic OCR backend using pytesseract.

Good baseline quality, works on CPU, widely available.
"""

from typing import Union, Any, List
import warnings

from HoloLoom.spinningWheel.ocr_protocol import (
    BaseOCRBackend,
    OCRResult,
    OCRQuality,
    OCROutputFormat,
    OCRBoundingBox
)


class TesseractOCRBackend(BaseOCRBackend):
    """
    Tesseract OCR backend.

    Uses pytesseract for text extraction.
    Provides basic-to-good quality OCR.
    """

    def __init__(self):
        """Initialize tesseract backend."""
        super().__init__(name="tesseract")

        # Lazy load tesseract
        self._tesseract = None

    def is_available(self) -> bool:
        """Check if pytesseract is available."""
        try:
            import pytesseract
            # Try to get version (will fail if tesseract not installed)
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def get_quality(self) -> OCRQuality:
        """Good quality with preprocessing."""
        return OCRQuality.GOOD

    async def _extract_text_impl(
        self,
        image: Any,
        output_format: OCROutputFormat,
        **kwargs
    ) -> OCRResult:
        """
        Extract text using tesseract.

        Args:
            image: PIL Image
            output_format: Desired format
            **kwargs: Tesseract options

        Returns:
            OCRResult
        """
        # Lazy load
        if self._tesseract is None:
            import pytesseract
            self._tesseract = pytesseract

        # Get image size
        image_size = image.size if hasattr(image, 'size') else (0, 0)

        # Extract text based on format
        if output_format == OCROutputFormat.JSON:
            # Get structured data with bounding boxes
            data = self._tesseract.image_to_data(image, output_type='dict')
            return self._process_structured_result(data, image_size)

        elif output_format == OCROutputFormat.MARKDOWN:
            # Get text with markdown formatting hints
            text = self._tesseract.image_to_string(image)
            text = self._format_as_markdown(text)

        else:  # TEXT
            # Simple text extraction
            text = self._tesseract.image_to_string(image)

        # Estimate confidence
        confidence = self._estimate_confidence(text)

        # Detect language
        try:
            osd = self._tesseract.image_to_osd(image, output_type='dict')
            detected_language = osd.get('script', 'unknown')
            language_confidence = osd.get('script_conf', 0.0) / 100.0
        except Exception:
            detected_language = None
            language_confidence = None

        return OCRResult(
            text=text,
            confidence=confidence,
            image_size=image_size,
            detected_language=detected_language,
            language_confidence=language_confidence
        )

    async def extract_text_structured(
        self,
        image: Union[Any],
        **kwargs
    ) -> OCRResult:
        """Extract text with bounding boxes."""
        return await self._extract_text_impl(
            image,
            OCROutputFormat.JSON,
            **kwargs
        )

    def _process_structured_result(
        self,
        data: dict,
        image_size: tuple[int, int]
    ) -> OCRResult:
        """
        Process tesseract structured output.

        Args:
            data: Tesseract output dict
            image_size: (width, height)

        Returns:
            OCRResult with bounding boxes
        """
        # Extract bounding boxes
        bounding_boxes = []
        texts = []

        for i in range(len(data['text'])):
            conf = int(data['conf'][i])
            text = data['text'][i].strip()

            # Skip low-confidence or empty
            if conf < 0 or not text:
                continue

            # Create bounding box
            bbox = OCRBoundingBox(
                x=data['left'][i],
                y=data['top'][i],
                width=data['width'][i],
                height=data['height'][i],
                text=text,
                confidence=conf / 100.0
            )

            bounding_boxes.append(bbox)
            texts.append(text)

        # Combine text
        full_text = ' '.join(texts)

        # Average confidence
        confidences = [bb.confidence for bb in bounding_boxes]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            image_size=image_size
        )

    def _format_as_markdown(self, text: str) -> str:
        """
        Format text as markdown (basic heuristics).

        Args:
            text: Raw OCR text

        Returns:
            Markdown-formatted text
        """
        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                formatted_lines.append('')
                continue

            # Detect headings (all caps, short)
            if stripped.isupper() and len(stripped) < 50:
                formatted_lines.append(f"## {stripped.title()}")

            # Detect bullet points
            elif stripped.startswith(('•', '·', '-', '*')):
                formatted_lines.append(f"- {stripped[1:].strip()}")

            # Detect numbered lists
            elif stripped[0].isdigit() and '.' in stripped[:3]:
                formatted_lines.append(stripped)

            else:
                formatted_lines.append(stripped)

        return '\n'.join(formatted_lines)

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate overall confidence from output.

        Heuristics:
        - Longer text = higher confidence
        - Proper words = higher confidence
        - Few special chars = higher confidence
        """
        if not text or len(text) < 10:
            return 0.3

        # Length factor
        length_factor = min(1.0, len(text) / 500)

        # Word factor (alphabetic content)
        words = text.split()
        alpha_words = sum(1 for w in words if w.isalpha())
        word_factor = alpha_words / len(words) if words else 0.5

        # Special char factor
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,!?-')
        special_ratio = special_chars / len(text)
        special_factor = max(0.0, 1.0 - special_ratio * 5)

        # Combine
        confidence = length_factor * 0.3 + word_factor * 0.5 + special_factor * 0.2

        return max(0.0, min(1.0, confidence))