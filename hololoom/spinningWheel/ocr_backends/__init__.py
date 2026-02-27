"""
OCR Backends
============

Protocol-based OCR backend implementations.

Available backends:
- deepseek: DeepSeek OCR (best quality, requires CUDA)
- tesseract: Pytesseract (basic quality, works everywhere)
- fallback: Filename extraction (last resort)

Usage:
    from HoloLoom.spinningWheel.ocr_backends import get_best_available_backend

    backend = get_best_available_backend()
    result = await backend.extract_text("document.png")
"""

from typing import Optional
from HoloLoom.spinningWheel.ocr_protocol import OCRProtocol, OCRBackendChain


def get_best_available_backend() -> OCRProtocol:
    """
    Get best available OCR backend.

    Priority order:
    1. DeepSeek (best quality)
    2. Tesseract (basic quality)
    3. Fallback (filename only)

    Returns:
        OCRProtocol implementation
    """
    # Try DeepSeek first
    try:
        from .deepseek import DeepSeekOCRBackend
        backend = DeepSeekOCRBackend()
        if backend.is_available():
            return backend
    except ImportError:
        pass

    # Try Tesseract
    try:
        from .tesseract import TesseractOCRBackend
        backend = TesseractOCRBackend()
        if backend.is_available():
            return backend
    except ImportError:
        pass

    # Last resort: Fallback
    from .fallback import FallbackOCRBackend
    return FallbackOCRBackend()


def get_all_available_backends() -> OCRBackendChain:
    """
    Get chain of all available backends.

    Returns:
        OCRBackendChain with all available backends
    """
    backends = []

    # Try DeepSeek
    try:
        from .deepseek import DeepSeekOCRBackend
        backend = DeepSeekOCRBackend()
        if backend.is_available():
            backends.append(backend)
    except ImportError:
        pass

    # Try Tesseract
    try:
        from .tesseract import TesseractOCRBackend
        backend = TesseractOCRBackend()
        if backend.is_available():
            backends.append(backend)
    except ImportError:
        pass

    # Always include fallback
    from .fallback import FallbackOCRBackend
    backends.append(FallbackOCRBackend())

    return OCRBackendChain(backends)


__all__ = [
    'get_best_available_backend',
    'get_all_available_backends'
]
