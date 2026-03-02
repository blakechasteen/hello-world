"""
Visual Q&A Engine - OCR + CLIP Integration

Provides visual question answering capabilities:
- Extract text from images (OCR)
- Compute image embeddings (CLIP)
- Image similarity search
- Graceful degradation if dependencies unavailable

Architecture:
    VisualQAEngine
    ├── OCR Backend (DeepSeekOCRSpinner or pytesseract fallback)
    ├── CLIP Encoder (for image embeddings)
    └── Similarity Search (cosine similarity)

Usage:
    engine = VisualQAEngine(loom, config)
    await engine.initialize()

    # Extract text from image
    text = await engine.extract_text_from_image("document.png")

    # Compute image embedding
    embedding = engine.compute_image_embedding("image.png")

Author: Agent D (Claude Code)
Date: January 2025
"""

import logging
from typing import Union, Optional, Any
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

# Optional dependencies (graceful degradation)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class VisualQAEngine:
    """
    Visual question answering engine using OCR + CLIP.

    Responsibilities:
    - Extract text from images (OCR)
    - Compute image embeddings (CLIP)
    - Image similarity search
    - Graceful degradation if dependencies unavailable

    Features:
    - Primary: DeepSeekOCRSpinner (best quality)
    - Fallback: pytesseract (basic OCR)
    - Last resort: Return empty string

    Usage:
        engine = VisualQAEngine(loom, config)
        await engine.initialize()

        # Extract text
        text = await engine.extract_text_from_image("doc.png")

        # Compute embedding
        emb = engine.compute_image_embedding("img.png")
    """

    def __init__(self, loom, config):
        """
        Initialize with HoloLoom instance and config.

        Args:
            loom: HoloLoom instance
            config: Configuration object
        """
        self.loom = loom
        self.config = config
        self.ocr_spinner = None
        self.clip_encoder = None
        self.fallback_ocr = None

        # Check availability
        self.deepseek_available = self._check_deepseek_available()
        self.fallback_available = self._check_fallback_available()

    async def initialize(self):
        """
        Initialize OCR and CLIP (lazy loading).

        Loads OCR backend (DeepSeek or pytesseract) and CLIP encoder.
        """
        # Try to load DeepSeek OCR spinner (preferred)
        if self.deepseek_available:
            try:
                from hololoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRSpinner, DeepSeekOCRConfig

                # Create config for OCR
                ocr_config = DeepSeekOCRConfig(
                    backend="transformers",  # More compatible than vLLM
                    resolution=1024,
                    output_format="text",
                    enable_enrichment=False
                )

                self.ocr_spinner = DeepSeekOCRSpinner(ocr_config)
                logger.info("✓ DeepSeek OCR initialized")
            except Exception as e:
                logger.warning(f"DeepSeek OCR initialization failed: {e}")
                self.ocr_spinner = None
                self.deepseek_available = False

        # Fall back to pytesseract if DeepSeek unavailable
        if not self.ocr_spinner and self.fallback_available:
            try:
                import pytesseract
                self.fallback_ocr = pytesseract
                logger.info("✓ Fallback OCR (pytesseract) initialized")
            except ImportError:
                logger.warning("pytesseract not available")
                self.fallback_ocr = None

        # Try to load CLIP encoder
        try:
            from hololoom.memory.photo_tokens import PhotoTokenMemory

            # Create temporary photo memory to access CLIP
            temp_photo_memory = PhotoTokenMemory(enable_clip=True)
            await temp_photo_memory._initialize()

            if temp_photo_memory.clip_model is not None:
                self.clip_encoder = temp_photo_memory
                logger.info("✓ CLIP encoder initialized")
            else:
                logger.warning("CLIP unavailable")
                self.clip_encoder = None

        except Exception as e:
            logger.warning(f"CLIP encoder initialization failed: {e}")
            self.clip_encoder = None

    # ========================================================================
    # OCR - Text Extraction
    # ========================================================================

    async def extract_text_from_image(
        self,
        image: Union[str, Path, bytes, 'Image.Image']
    ) -> str:
        """
        Extract text from image using OCR.

        Tries backends in order:
        1. DeepSeekOCRSpinner (best quality)
        2. pytesseract (fallback)
        3. Empty string (last resort)

        Args:
            image: Image in various formats (path, bytes, PIL Image)

        Returns:
            Extracted text (empty string if OCR unavailable)

        Example:
            text = await engine.extract_text_from_image("document.png")
            print(f"Extracted: {text}")
        """
        # Try DeepSeek OCR first
        if self.ocr_spinner:
            try:
                return await self._extract_text_deepseek(image)
            except Exception as e:
                logger.warning(f"DeepSeek OCR failed: {e}")

        # Fall back to pytesseract
        if self.fallback_ocr:
            try:
                return await self._extract_text_fallback(image)
            except Exception as e:
                logger.warning(f"Fallback OCR failed: {e}")

        # Last resort: empty string
        logger.warning("No OCR backend available, returning empty text")
        return ""

    async def _extract_text_deepseek(
        self,
        image: Union[str, Path, bytes, 'Image.Image']
    ) -> str:
        """
        Extract text using DeepSeek OCR.

        Args:
            image: Image to process

        Returns:
            Extracted text
        """
        # Convert to path if needed
        image_path = self._image_to_path(image)

        # Run OCR through spinner
        result = await self.ocr_spinner.spin(image_path)

        # Extract text from result
        if result.success and result.shards:
            text = result.shards[0].text
            logger.debug(f"DeepSeek OCR extracted {len(text)} characters")
            return text
        else:
            raise RuntimeError(f"DeepSeek OCR failed: {result.error_message}")

    async def _extract_text_fallback(
        self,
        image: Union[str, Path, bytes, 'Image.Image']
    ) -> str:
        """
        Extract text using pytesseract fallback.

        Args:
            image: Image to process

        Returns:
            Extracted text
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available for fallback OCR")

        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            from io import BytesIO
            pil_image = Image.open(BytesIO(image))
        elif hasattr(image, 'mode'):  # PIL Image
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Run OCR
        try:
            data = self.fallback_ocr.image_to_data(pil_image, output_type='dict')

            # Extract text with confidence > 0
            texts = data['text']
            confidences = data['conf']

            valid_words = [
                text for text, conf in zip(texts, confidences)
                if conf > 0 and text.strip()
            ]

            text = ' '.join(valid_words)
            logger.debug(f"Pytesseract extracted {len(text)} characters")
            return text

        except Exception as e:
            raise RuntimeError(f"Pytesseract OCR failed: {e}")

    # ========================================================================
    # CLIP - Image Embeddings
    # ========================================================================

    def compute_image_embedding(
        self,
        image: Union[str, Path, 'Image.Image']
    ) -> Optional['np.ndarray']:
        """
        Compute CLIP embedding for image.

        Args:
            image: Image (path or PIL Image)

        Returns:
            CLIP embedding (512D) or None if CLIP unavailable

        Example:
            emb = engine.compute_image_embedding("image.png")
            print(f"Embedding shape: {emb.shape}")  # (512,)

        Raises:
            RuntimeError: If CLIP encoder unavailable
        """
        if not self.clip_encoder:
            raise RuntimeError("CLIP encoder unavailable")

        # Load image if path
        if isinstance(image, (str, Path)):
            if not PIL_AVAILABLE:
                raise RuntimeError("PIL not available")
            pil_image = Image.open(image).convert('RGB')
        elif hasattr(image, 'mode'):  # PIL Image
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Encode with CLIP
        try:
            # Use photo memory's CLIP encoder
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = loop.run_until_complete(
                self.clip_encoder._encode_image_clip(pil_image)
            )
            return embedding

        except Exception as e:
            raise RuntimeError(f"CLIP encoding failed: {e}")

    def compute_text_embedding(
        self,
        text: str
    ) -> Optional['np.ndarray']:
        """
        Compute CLIP text embedding.

        Args:
            text: Text to embed

        Returns:
            CLIP embedding (512D) or None if CLIP unavailable

        Example:
            emb = engine.compute_text_embedding("architecture diagram")
            print(f"Embedding shape: {emb.shape}")  # (512,)

        Raises:
            RuntimeError: If CLIP encoder unavailable
        """
        if not self.clip_encoder:
            raise RuntimeError("CLIP encoder unavailable")

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = loop.run_until_complete(
                self.clip_encoder._encode_text_clip(text)
            )
            return embedding

        except Exception as e:
            raise RuntimeError(f"CLIP encoding failed: {e}")

    # ========================================================================
    # Utilities
    # ========================================================================

    def _check_deepseek_available(self) -> bool:
        """Check if DeepSeek OCR is available."""
        try:
            from hololoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRSpinner
            import transformers
            import torch
            return True
        except ImportError:
            return False

    def _check_fallback_available(self) -> bool:
        """Check if fallback OCR (pytesseract) is available."""
        try:
            import pytesseract
            from PIL import Image
            return True
        except ImportError:
            return False

    def _image_to_path(
        self,
        image: Union[str, Path, bytes, 'Image.Image']
    ) -> Path:
        """
        Convert image to file path (saves to temp file if needed).

        Args:
            image: Image in various formats

        Returns:
            Path to image file
        """
        if isinstance(image, (str, Path)):
            return Path(image)

        # Need to save to temp file
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available to convert image")

        # Convert to PIL Image
        if isinstance(image, bytes):
            from io import BytesIO
            pil_image = Image.open(BytesIO(image))
        elif hasattr(image, 'mode'):  # Already PIL Image
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            pil_image.save(f, format='PNG')
            return Path(f.name)

    def is_available(self) -> bool:
        """
        Check if visual Q&A is available.

        Returns:
            True if at least one OCR backend is available
        """
        return self.ocr_spinner is not None or self.fallback_ocr is not None

    def get_status(self) -> Dict[str, bool]:
        """
        Get status of visual Q&A components.

        Returns:
            Dictionary with component availability
        """
        return {
            'deepseek_ocr': self.ocr_spinner is not None,
            'fallback_ocr': self.fallback_ocr is not None,
            'clip_encoder': self.clip_encoder is not None,
            'any_ocr': self.is_available()
        }
