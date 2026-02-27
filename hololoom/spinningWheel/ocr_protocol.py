"""
OCR Protocol
============

Protocol-based abstraction for OCR backends.

Enables multiple OCR implementations (DeepSeek, Tesseract, Azure, AWS, Google)
to be used interchangeably by spinners (PDF, image, multimodal).

Philosophy:
    "One protocol, many backends"

Architecture:
    Spinner → OCRProtocol → Backend (DeepSeek/Tesseract/Cloud)

Benefits:
- Spinners don't depend on specific OCR implementations
- Easy to add new OCR backends
- Graceful fallback through backend chain
- Consistent API across all backends

Author: Claude Code
Date: January 2025
"""

from typing import Protocol, List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time


# ============================================================================
# OCR Types
# ============================================================================

class OCRQuality(Enum):
    """OCR quality levels."""
    EXCELLENT = "excellent"  # DeepSeek, cloud APIs
    GOOD = "good"           # Tesseract with preprocessing
    BASIC = "basic"         # Basic Tesseract
    POOR = "poor"           # Fallback methods


class OCROutputFormat(Enum):
    """OCR output format options."""
    MARKDOWN = "markdown"   # Structured markdown
    TEXT = "text"          # Plain text
    JSON = "json"          # Structured JSON with bounding boxes
    HTML = "html"          # HTML with formatting


@dataclass
class OCRBoundingBox:
    """Bounding box for OCR'd text region."""
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


@dataclass
class OCRResult:
    """
    Result of OCR operation.

    Contains extracted text plus metadata about quality and processing.
    """
    text: str
    confidence: float  # 0.0-1.0

    # Optional structured output
    bounding_boxes: Optional[List[OCRBoundingBox]] = None

    # Metadata
    backend: str = "unknown"
    quality: OCRQuality = OCRQuality.BASIC
    processing_time_ms: float = 0.0
    image_size: tuple[int, int] = (0, 0)

    # Language detection
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None

    # Additional metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Validate result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'backend': self.backend,
            'quality': self.quality.value,
            'processing_time_ms': self.processing_time_ms,
            'image_size': self.image_size,
            'detected_language': self.detected_language,
            'language_confidence': self.language_confidence,
            'bounding_boxes': [
                {
                    'x': bb.x,
                    'y': bb.y,
                    'width': bb.width,
                    'height': bb.height,
                    'text': bb.text,
                    'confidence': bb.confidence
                }
                for bb in (self.bounding_boxes or [])
            ],
            'metadata': self.metadata
        }


# ============================================================================
# OCR Protocol
# ============================================================================

class OCRProtocol(Protocol):
    """
    Protocol that all OCR backends must implement.

    Minimal interface for OCR operations:
    - extract_text(): Extract text from image/bytes
    - get_name(): Backend name
    - is_available(): Check if backend is usable
    - get_quality(): Quality level of this backend

    Optional methods:
    - extract_text_structured(): Extract with bounding boxes
    - batch_extract(): Process multiple images efficiently
    """

    def get_name(self) -> str:
        """
        Get backend name.

        Returns:
            Name (e.g., 'deepseek', 'tesseract', 'azure')
        """
        ...

    def is_available(self) -> bool:
        """
        Check if backend is available.

        Returns:
            True if backend can be used
        """
        ...

    def get_quality(self) -> OCRQuality:
        """
        Get quality level of this backend.

        Returns:
            OCRQuality enum
        """
        ...

    async def extract_text(
        self,
        image: Union[Path, bytes, Any],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> OCRResult:
        """
        Extract text from image.

        Args:
            image: Image path, bytes, or PIL Image
            output_format: Desired output format
            **kwargs: Backend-specific options

        Returns:
            OCRResult with extracted text and metadata
        """
        ...

    # Optional: Structured extraction
    async def extract_text_structured(
        self,
        image: Union[Path, bytes, Any],
        **kwargs
    ) -> OCRResult:
        """
        Extract text with bounding boxes.

        Optional method for backends that support structured output.

        Args:
            image: Image path, bytes, or PIL Image
            **kwargs: Backend-specific options

        Returns:
            OCRResult with bounding_boxes populated
        """
        raise NotImplementedError(f"{self.get_name()} doesn't support structured extraction")

    # Optional: Batch processing
    async def batch_extract(
        self,
        images: List[Union[Path, bytes, Any]],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> List[OCRResult]:
        """
        Extract text from multiple images efficiently.

        Optional method for backends that support batching.

        Args:
            images: List of images
            output_format: Desired output format
            **kwargs: Backend-specific options

        Returns:
            List of OCRResults
        """
        # Default: Process sequentially
        results = []
        for image in images:
            result = await self.extract_text(image, output_format, **kwargs)
            results.append(result)
        return results


# ============================================================================
# Base OCR Backend
# ============================================================================

class BaseOCRBackend(ABC):
    """
    Abstract base class for OCR backends.

    Provides common functionality:
    - Error handling
    - Performance tracking
    - Result validation

    Subclasses must implement:
    - _extract_text_impl(): Core OCR logic
    """

    def __init__(self, name: str):
        """
        Initialize base backend.

        Args:
            name: Backend name
        """
        self.name = name

    # ========================================================================
    # Protocol Implementation
    # ========================================================================

    def get_name(self) -> str:
        """Get backend name."""
        return self.name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available (must be implemented)."""
        ...

    @abstractmethod
    def get_quality(self) -> OCRQuality:
        """Get quality level (must be implemented)."""
        ...

    async def extract_text(
        self,
        image: Union[Path, bytes, Any],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> OCRResult:
        """
        Extract text with error handling.

        Delegates to _extract_text_impl() for actual OCR.
        """
        start_time = time.time()

        try:
            # Check availability
            if not self.is_available():
                raise RuntimeError(f"{self.name} backend not available")

            # Load image if needed
            image_obj = self._load_image(image)

            # Delegate to implementation
            result = await self._extract_text_impl(image_obj, output_format, **kwargs)

            # Add metadata
            result.backend = self.name
            result.quality = self.get_quality()
            result.processing_time_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            # Return error result
            processing_time_ms = (time.time() - start_time) * 1000

            return OCRResult(
                text=f"[OCR Error: {str(e)}]",
                confidence=0.0,
                backend=self.name,
                quality=OCRQuality.POOR,
                processing_time_ms=processing_time_ms,
                metadata={'error': str(e), 'failed': True}
            )

    async def extract_text_structured(
        self,
        image: Union[Path, bytes, Any],
        **kwargs
    ) -> OCRResult:
        """Default: Not implemented."""
        raise NotImplementedError(f"{self.name} doesn't support structured extraction")

    async def batch_extract(
        self,
        images: List[Union[Path, bytes, Any]],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> List[OCRResult]:
        """Default: Sequential processing."""
        results = []
        for image in images:
            result = await self.extract_text(image, output_format, **kwargs)
            results.append(result)
        return results

    # ========================================================================
    # Abstract Methods
    # ========================================================================

    @abstractmethod
    async def _extract_text_impl(
        self,
        image: Any,
        output_format: OCROutputFormat,
        **kwargs
    ) -> OCRResult:
        """
        Core OCR implementation.

        Subclasses must implement this method.

        Args:
            image: PIL Image object
            output_format: Desired output format
            **kwargs: Backend-specific options

        Returns:
            OCRResult with extracted text
        """
        ...

    # ========================================================================
    # Utilities
    # ========================================================================

    def _load_image(self, image: Union[Path, bytes, Any]) -> Any:
        """
        Load image from various formats.

        Args:
            image: Path, bytes, or PIL Image

        Returns:
            PIL Image object
        """
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("PIL (pillow) not installed")

        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            import io
            return Image.open(io.BytesIO(image)).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


# ============================================================================
# OCR Backend Chain
# ============================================================================

class OCRBackendChain:
    """
    Chain of OCR backends with automatic fallback.

    Tries backends in order until one succeeds:
    1. DeepSeek (best quality)
    2. Cloud APIs (Azure, AWS, Google)
    3. Tesseract (basic)
    4. Fallback (filename extraction)

    Example:
        >>> chain = OCRBackendChain([
        ...     DeepSeekOCRBackend(),
        ...     TesseractOCRBackend(),
        ...     FallbackOCRBackend()
        ... ])
        >>> result = await chain.extract_text("document.png")
    """

    def __init__(self, backends: List[OCRProtocol]):
        """
        Initialize backend chain.

        Args:
            backends: List of backends in priority order
        """
        self.backends = backends

        # Filter to only available backends
        self.available_backends = [
            backend for backend in backends
            if backend.is_available()
        ]

        if not self.available_backends:
            print("Warning: No OCR backends available")

    def get_name(self) -> str:
        """Get name of active backend."""
        if self.available_backends:
            return self.available_backends[0].get_name()
        return "none"

    def is_available(self) -> bool:
        """Check if any backend is available."""
        return len(self.available_backends) > 0

    def get_quality(self) -> OCRQuality:
        """Get quality of best available backend."""
        if self.available_backends:
            return self.available_backends[0].get_quality()
        return OCRQuality.POOR

    async def extract_text(
        self,
        image: Union[Path, bytes, Any],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> OCRResult:
        """
        Extract text, trying backends in order.

        Args:
            image: Image path, bytes, or PIL Image
            output_format: Desired output format
            **kwargs: Backend-specific options

        Returns:
            OCRResult from first successful backend
        """
        last_error = None

        for backend in self.available_backends:
            try:
                result = await backend.extract_text(image, output_format, **kwargs)

                # Check if result is valid (not an error)
                if result.confidence > 0 and not result.metadata.get('failed'):
                    return result

            except Exception as e:
                last_error = e
                continue

        # All backends failed
        return OCRResult(
            text=f"[All OCR backends failed: {last_error}]",
            confidence=0.0,
            backend="none",
            quality=OCRQuality.POOR,
            metadata={'error': str(last_error), 'all_failed': True}
        )

    async def extract_text_structured(
        self,
        image: Union[Path, bytes, Any],
        **kwargs
    ) -> OCRResult:
        """Extract text with bounding boxes (first backend that supports it)."""
        for backend in self.available_backends:
            try:
                return await backend.extract_text_structured(image, **kwargs)
            except NotImplementedError:
                continue

        raise NotImplementedError("No backends support structured extraction")

    async def batch_extract(
        self,
        images: List[Union[Path, bytes, Any]],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> List[OCRResult]:
        """Batch extract using best available backend."""
        if self.available_backends:
            return await self.available_backends[0].batch_extract(images, output_format, **kwargs)

        # All backends unavailable
        return [
            OCRResult(
                text="[No OCR backend available]",
                confidence=0.0,
                backend="none",
                quality=OCRQuality.POOR
            )
            for _ in images
        ]


# ============================================================================
# Utility Functions
# ============================================================================

def create_ocr_backend_chain(
    prefer_deepseek: bool = True,
    prefer_cloud: bool = False,
    enable_fallback: bool = True
) -> OCRBackendChain:
    """
    Create OCR backend chain with sensible defaults.

    Args:
        prefer_deepseek: Try DeepSeek first (best quality)
        prefer_cloud: Try cloud APIs (Azure, AWS, Google)
        enable_fallback: Include pytesseract fallback

    Returns:
        OCRBackendChain ready to use

    Example:
        >>> chain = create_ocr_backend_chain()
        >>> result = await chain.extract_text("doc.png")
    """
    backends = []

    # Add DeepSeek (best quality)
    if prefer_deepseek:
        try:
            from HoloLoom.spinningWheel.ocr_backends.deepseek import DeepSeekOCRBackend
            backends.append(DeepSeekOCRBackend())
        except ImportError:
            pass

    # Add cloud APIs
    if prefer_cloud:
        # Future: Azure, AWS, Google OCR backends
        pass

    # Add Tesseract fallback
    if enable_fallback:
        try:
            from HoloLoom.spinningWheel.ocr_backends.tesseract import TesseractOCRBackend
            backends.append(TesseractOCRBackend())
        except ImportError:
            pass

    # Last resort: Filename extraction
    from HoloLoom.spinningWheel.ocr_backends.fallback import FallbackOCRBackend
    backends.append(FallbackOCRBackend())

    return OCRBackendChain(backends)