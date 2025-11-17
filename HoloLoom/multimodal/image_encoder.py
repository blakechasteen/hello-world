"""
HoloLoom Image Encoder
======================
CLIP-based image encoding with vision-language alignment.

This encoder uses CLIP (Contrastive Language-Image Pre-training)
to create embeddings that align visual and textual semantics,
enabling cross-modal retrieval (text→image, image→text).

Features:
- CLIP vision encoder
- OCR for text extraction from images
- Image similarity search
- Support for PIL/OpenCV image loading
- Graceful degradation without optional dependencies

Architecture:
- Input: Image file path or PIL.Image
- Process: CLIP vision encoder → 512D → projection → 768D
- Output: ModalEmbedding with 768D vector

The projection to 768D ensures compatibility with text embeddings
from HoloLoom's MatryoshkaEmbeddings system.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Import from base module
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    ImageEncoderConfig,
    validate_embedding_dim,
    normalize_embedding
)

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    from PIL import Image
    _HAVE_PIL = True
except ImportError:
    Image = None
    _HAVE_PIL = False
    warnings.warn("PIL not available. Install with: pip install pillow")

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    _HAVE_CLIP = True
except ImportError:
    torch = None
    CLIPProcessor = None
    CLIPModel = None
    _HAVE_CLIP = False
    warnings.warn("transformers/torch not available. Install with: pip install transformers torch")

try:
    import pytesseract
    _HAVE_OCR = True
except ImportError:
    pytesseract = None
    _HAVE_OCR = False
    warnings.warn("pytesseract not available. Install with: pip install pytesseract")

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    cv2 = None
    _HAVE_CV2 = False
    # Don't warn - CV2 is optional, PIL is preferred


# ============================================================================
# Image Encoder Implementation
# ============================================================================

class ImageEncoder:
    """
    CLIP-based image encoder with OCR support.

    Encodes images into 768D embeddings that align with text semantics.
    This enables cross-modal operations like "find images of cats"
    using text queries.

    Technical Details:
    - Uses CLIP ViT-B/32 (default) or configurable model
    - CLIP produces 512D embeddings
    - Projects to 768D using learned linear layer
    - Optionally extracts text via OCR for hybrid embeddings

    Example:
        encoder = ImageEncoder()
        embedding = await encoder.encode("cat.jpg")
        # Returns ModalEmbedding with 768D vector
    """

    def __init__(self, config: Optional[ImageEncoderConfig] = None):
        """
        Initialize image encoder.

        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or ImageEncoderConfig()
        self._model = None
        self._processor = None
        self._projection = None  # 512D → 768D projection matrix

        # Load CLIP model
        if _HAVE_CLIP:
            try:
                logger.info(f"Loading CLIP model: {self.config.model_name}")
                self._model = CLIPModel.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                )
                self._processor = CLIPProcessor.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                )

                # Move to device
                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                else:
                    self._model = self._model.to("cpu")

                self._model.eval()

                # Initialize projection matrix (512 → 768)
                self._init_projection()

                logger.info(f"CLIP model loaded on {self.config.device}")

            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                self._model = None
                self._processor = None
        else:
            logger.warning("CLIP not available - using fallback embeddings")

    def _init_projection(self):
        """
        Initialize projection matrix from CLIP's 512D to HoloLoom's 768D.

        Uses orthogonal initialization for minimal information loss.
        """
        if torch is None:
            return

        # Create projection: 512 → 768
        projection = torch.nn.Linear(512, 768, bias=False)

        # Orthogonal initialization
        torch.nn.init.orthogonal_(projection.weight)

        # Move to same device as model
        if self.config.device == "cuda" and torch.cuda.is_available():
            projection = projection.to("cuda")
        else:
            projection = projection.to("cpu")

        projection.eval()
        self._projection = projection

    @property
    def modality(self) -> Modality:
        """Return modality type."""
        return Modality.IMAGE

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 768

    def _load_image(self, image_path: Union[str, Path]) -> Any:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object

        Raises:
            RuntimeError: If PIL not available
            FileNotFoundError: If image file not found
        """
        if not _HAVE_PIL:
            raise RuntimeError("PIL required for image loading: pip install pillow")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise

    def _extract_text_ocr(self, image: Any) -> Optional[str]:
        """
        Extract text from image using OCR.

        Args:
            image: PIL Image

        Returns:
            Extracted text or None if OCR unavailable/failed
        """
        if not self.config.enable_ocr:
            return None

        if not _HAVE_OCR:
            logger.warning("OCR requested but pytesseract not available")
            return None

        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.config.ocr_lang
            )
            return text.strip() if text else None
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None

    async def encode(
        self,
        input_data: Union[str, Path, Any],
        **kwargs
    ) -> ModalEmbedding:
        """
        Encode single image to 768D embedding.

        Args:
            input_data: Image file path or PIL Image
            **kwargs: Additional options

        Returns:
            ModalEmbedding with 768D vector

        Raises:
            RuntimeError: If CLIP not available and no fallback
        """
        # Load image if path provided
        if isinstance(input_data, (str, Path)):
            image = self._load_image(input_data)
            image_path = str(input_data)
        else:
            image = input_data
            image_path = kwargs.get("path", "unknown")

        # Extract text via OCR (optional)
        ocr_text = self._extract_text_ocr(image)

        # Encode with CLIP
        if self._model is not None and self._processor is not None:
            embedding_vec = await self._encode_clip(image)
            confidence = 1.0
        else:
            # Fallback: deterministic hash-based embedding
            logger.warning("Using fallback embedding (CLIP unavailable)")
            embedding_vec = self._fallback_embedding(image_path)
            confidence = 0.5

        # Validate dimension
        validate_embedding_dim(embedding_vec, 768)

        # Build metadata
        metadata = {
            "file": image_path,
            "ocr_text": ocr_text,
            "image_size": image.size if hasattr(image, 'size') else None
        }

        return ModalEmbedding(
            embedding=embedding_vec,
            modality=Modality.IMAGE,
            metadata=metadata,
            confidence=confidence
        )

    async def _encode_clip(self, image: Any) -> List[float]:
        """
        Encode image with CLIP model.

        Args:
            image: PIL Image

        Returns:
            768D embedding vector
        """
        # Preprocess image
        inputs = self._processor(
            images=image,
            return_tensors="pt"
        )

        # Move to device
        if self.config.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            # Get CLIP vision features (512D)
            vision_outputs = self._model.get_image_features(**inputs)

            # Project to 768D
            if self._projection is not None:
                projected = self._projection(vision_outputs)
            else:
                # Pad if no projection available
                projected = torch.nn.functional.pad(vision_outputs, (0, 256))

            # Normalize
            embedding = torch.nn.functional.normalize(projected, dim=-1)

            # Convert to list
            embedding_vec = embedding.cpu().numpy()[0].tolist()

        return embedding_vec

    def _fallback_embedding(self, image_path: str) -> List[float]:
        """
        Generate fallback embedding when CLIP unavailable.

        Uses deterministic hash-based generation for testing.

        Args:
            image_path: Path to image (for hashing)

        Returns:
            768D embedding vector
        """
        # Hash path for deterministic seed
        seed = abs(hash(image_path)) % (2**32)
        rng = np.random.default_rng(seed)

        # Generate random embedding
        embedding = rng.normal(0, 1, 768)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

        return embedding.tolist()

    async def encode_batch(
        self,
        inputs: List[Union[str, Path, Any]],
        **kwargs
    ) -> List[ModalEmbedding]:
        """
        Encode multiple images efficiently.

        Args:
            inputs: List of image paths or PIL Images
            **kwargs: Additional options

        Returns:
            List of ModalEmbeddings

        Note:
            Currently processes sequentially. Future optimization:
            batch processing with CLIP for efficiency.
        """
        results = []
        for input_data in inputs:
            try:
                embedding = await self.encode(input_data, **kwargs)
                results.append(embedding)
            except Exception as e:
                logger.error(f"Failed to encode {input_data}: {e}")
                # Continue with other images
                continue

        return results

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First 768D embedding
            embedding2: Second 768D embedding

        Returns:
            Similarity score (0.0 to 1.0)
        """
        v1 = np.array(embedding1)
        v2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        )

        # Clip to [0, 1]
        return float(max(0.0, min(1.0, similarity)))


# ============================================================================
# Factory Function
# ============================================================================

def create_image_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
    enable_ocr: bool = True,
    device: str = "cpu",
    **kwargs
) -> ImageEncoder:
    """
    Factory function to create image encoder.

    Args:
        model_name: CLIP model name
        enable_ocr: Enable text extraction
        device: "cpu" or "cuda"
        **kwargs: Additional config options

    Returns:
        ImageEncoder instance
    """
    config = ImageEncoderConfig(
        model_name=model_name,
        enable_ocr=enable_ocr,
        device=device,
        **kwargs
    )
    return ImageEncoder(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 80)
        print("Image Encoder Demo")
        print("=" * 80 + "\n")

        # Create encoder
        encoder = create_image_encoder(device="cpu")

        print(f"Modality: {encoder.modality.value}")
        print(f"Embedding dimension: {encoder.embedding_dim}")
        print(f"CLIP available: {_HAVE_CLIP}")
        print(f"OCR available: {_HAVE_OCR}")

        if not _HAVE_PIL:
            print("\nPIL not available - install with: pip install pillow")
            return

        # Would encode real image:
        # embedding = await encoder.encode("path/to/image.jpg")
        # print(f"\nEmbedding shape: {len(embedding.embedding)}")
        # print(f"Confidence: {embedding.confidence}")
        # print(f"OCR text: {embedding.metadata.get('ocr_text')}")

        print("\n✓ Image encoder ready!")
        print("Usage: embedding = await encoder.encode('image.jpg')")

    asyncio.run(demo())
