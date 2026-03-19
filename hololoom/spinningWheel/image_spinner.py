"""
Image Spinner - OCR-Powered Image Text Extraction
==================================================

Extracts text from images using protocol-based OCR backends.

Features:
- Automatic backend selection (DeepSeek → Tesseract → Fallback)
- Multiple output formats (TEXT, MARKDOWN, JSON)
- Batch processing
- Entity/motif extraction
- Importance scoring
- Image metadata extraction

Supports:
- PNG, JPG, JPEG, WebP, TIFF, BMP
- Single images or batches
- Various quality levels

Usage:
    from hololoom.spinningWheel import ImageSpinner

    spinner = ImageSpinner()
    result = await spinner.spin("document.png")

    # Batch processing
    result = await spinner.spin(["img1.png", "img2.jpg"])

Author: Claude Code
Date: January 2025
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hololoom.protocols.types import MemoryShard
from hololoom.spinningWheel.ocr_backends import get_all_available_backends
from hololoom.spinningWheel.ocr_protocol import OCROutputFormat
from hololoom.spinningWheel.protocol import (
    BaseSpinner,
    ImportanceScore,
    ImportanceSignals,
    SpinnerCapabilities,
    SpinResult,
)


@dataclass
class ImageMetadata:
    """Image file metadata"""
    file_path: Path
    file_size: int
    width: int
    height: int
    format: str
    mode: str  # RGB, RGBA, L, etc.


class ImageSpinner(BaseSpinner):
    """
    Spinner for extracting text from images using OCR.

    Uses protocol-based OCR backends with automatic fallback:
    1. DeepSeek (best quality, CUDA required)
    2. Tesseract (good quality, CPU)
    3. Fallback (filename only)
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        output_format: str = "markdown",
        enable_enrichment: bool = False,
        batch_size: int = 8
    ):
        """
        Initialize ImageSpinner.

        Args:
            importance_threshold: Min importance score (0.0-1.0)
            output_format: OCR output format ("text", "markdown", "json")
            enable_enrichment: Extract entities/motifs via Ollama
            batch_size: Batch size for multi-image processing
        """
        super().__init__(name="image", importance_threshold=importance_threshold)

        self.output_format = OCROutputFormat(output_format)
        self.enable_enrichment = enable_enrichment
        self.batch_size = batch_size

        # Get OCR backend chain
        self.ocr_chain = get_all_available_backends()

        print(f"[ImageSpinner] Using OCR backend: {self.ocr_chain.get_name()} ({self.ocr_chain.get_quality().value})")

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=self.enable_enrichment,
            motif_extraction=self.enable_enrichment,
            embeddings=False,
            multimodal=False,
            supported_formats=['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.gif']
        )

    def is_available(self) -> bool:
        """Check if OCR backend is available."""
        return self.ocr_chain.is_available()

    async def _spin_impl(
        self,
        source: Any,
        **kwargs
    ) -> list[MemoryShard]:
        """
        Extract text from image(s).

        Args:
            source: Image path(s) - str, Path, or list
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        # Normalize to list of paths
        if isinstance(source, (str, Path)):
            image_paths = [Path(source)]
        elif isinstance(source, list):
            image_paths = [Path(p) for p in source]
        else:
            raise ValueError(f"source must be path or list of paths, got {type(source)}")

        # Validate paths
        for path in image_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

        # Extract text via OCR
        all_shards = []

        # Process in batches for efficiency
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]

            # Batch OCR extraction
            ocr_results = await self.ocr_chain.batch_extract(
                batch,
                output_format=self.output_format
            )

            # Convert to shards
            for path, ocr_result in zip(batch, ocr_results):
                shard = await self._create_shard_from_ocr(path, ocr_result)
                all_shards.append(shard)

        return all_shards

    async def _create_shard_from_ocr(
        self,
        image_path: Path,
        ocr_result: Any
    ) -> MemoryShard:
        """
        Create MemoryShard from OCR result.

        Args:
            image_path: Path to image
            ocr_result: OCRResult from backend

        Returns:
            MemoryShard
        """
        # Extract image metadata
        metadata_obj = self._extract_image_metadata(image_path)

        # Score importance
        importance = self._score_image_importance(
            ocr_result.text,
            ocr_result.confidence,
            metadata_obj
        )

        # Extract entities/motifs if enabled
        entities = []
        motifs = []

        if self.enable_enrichment and len(ocr_result.text) > 50:
            entities, motifs = await self._enrich_content(ocr_result.text)

        # Create shard
        shard = self._create_shard(
            id_suffix=f"{image_path.stem}_{int(time.time())}",
            text=ocr_result.text,
            episode=f"image_{image_path.stem}",
            entities=entities,
            motifs=motifs,
            metadata={
                'file_path': str(image_path),
                'file_name': image_path.name,
                'file_size': metadata_obj.file_size,
                'image_width': metadata_obj.width,
                'image_height': metadata_obj.height,
                'image_format': metadata_obj.format,
                'image_mode': metadata_obj.mode,
                'ocr_backend': ocr_result.backend,
                'ocr_confidence': ocr_result.confidence,
                'ocr_quality': ocr_result.quality.value,
                'ocr_processing_time_ms': ocr_result.processing_time_ms,
                'output_format': self.output_format.value,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'detected_language': ocr_result.detected_language,
                'language_confidence': ocr_result.language_confidence
            }
        )

        return shard

    def _extract_image_metadata(self, image_path: Path) -> ImageMetadata:
        """Extract image metadata."""
        try:
            from PIL import Image
            img = Image.open(image_path)

            return ImageMetadata(
                file_path=image_path,
                file_size=image_path.stat().st_size,
                width=img.width,
                height=img.height,
                format=img.format or 'unknown',
                mode=img.mode
            )
        except Exception:
            # Fallback metadata
            return ImageMetadata(
                file_path=image_path,
                file_size=image_path.stat().st_size if image_path.exists() else 0,
                width=0,
                height=0,
                format='unknown',
                mode='unknown'
            )

    def _score_image_importance(
        self,
        text: str,
        confidence: float,
        metadata: ImageMetadata
    ) -> ImportanceScore:
        """
        Score image importance.

        Factors:
        - Text length (longer = more substantial)
        - OCR confidence (higher = better quality)
        - Image resolution (higher = better quality)
        - Text structure (headings, lists)
        """
        signals = ImportanceSignals()

        # Length score
        text_length = len(text)
        signals.length_score = min(1.0, text_length / 1000)

        # Technical score (basic keyword detection)
        technical_keywords = [
            'algorithm', 'data', 'system', 'method', 'analysis',
            'diagram', 'chart', 'figure', 'table', 'graph'
        ]
        technical_count = sum(1 for kw in technical_keywords if kw.lower() in text.lower())
        signals.technical_score = min(1.0, technical_count / 3)

        # Structural score (markdown indicators)
        structure_markers = text.count('\n#') + text.count('\n-') + text.count('\n*')
        signals.structural_score = min(1.0, structure_markers / 10)

        # Authority score (OCR confidence proxy)
        signals.authority_score = confidence

        # Custom signals
        signals.custom_signals = {
            'ocr_confidence': confidence,
            'high_resolution': 1.0 if (metadata.width * metadata.height) > 1000000 else 0.5,
            'file_size': min(1.0, metadata.file_size / (5 * 1024 * 1024))  # Normalize to 5MB
        }

        # Noise penalty (very short or mostly symbols)
        if text_length < 20:
            signals.noise_penalty = -0.5
        elif text_length < 10:
            signals.noise_penalty = -1.0

        # Compute total
        score = signals.compute_total()

        # Generate reason
        reasons = []
        if confidence > 0.8:
            reasons.append("high OCR confidence")
        if text_length > 500:
            reasons.append("substantial text")
        if signals.structural_score > 0.5:
            reasons.append("structured content")
        if technical_count > 0:
            reasons.append("technical content")

        reason = " + ".join(reasons) if reasons else "standard image"

        return ImportanceScore(
            score=score,
            signals=signals,
            reason=reason
        )

    async def _enrich_content(self, text: str) -> tuple[list[str], list[str]]:
        """
        Extract entities and motifs using Ollama.

        Returns:
            (entities, motifs)
        """
        try:
            import ollama
        except ImportError:
            return [], []

        try:
            # Extract entities
            entity_prompt = f"""Extract key entities (people, organizations, locations, concepts) from this text.
Return only a comma-separated list of entities.

Text: {text[:2000]}

Entities:"""

            entity_response = ollama.generate(
                model="llama3.2:3b",
                prompt=entity_prompt
            )

            entities = [
                e.strip()
                for e in entity_response['response'].split(',')
                if e.strip()
            ][:20]

            # Extract motifs/topics
            motif_prompt = f"""Extract key topics and themes from this text.
Return only a comma-separated list of topics.

Text: {text[:2000]}

Topics:"""

            motif_response = ollama.generate(
                model="llama3.2:3b",
                prompt=motif_prompt
            )

            motifs = [
                m.strip()
                for m in motif_response['response'].split(',')
                if m.strip()
            ][:10]

            return entities, motifs

        except Exception as e:
            print(f"Warning: Enrichment failed: {e}")
            return [], []


# Convenience function

async def spin_image(
    image_path: str,
    output_format: str = "markdown",
    importance_threshold: float = 0.3
) -> SpinResult:
    """
    Convenience function to extract text from image.

    Args:
        image_path: Path to image
        output_format: Output format ("text", "markdown", "json")
        importance_threshold: Min importance score

    Returns:
        SpinResult with MemoryShards

    Example:
        >>> result = await spin_image("document.png")
        >>> print(result.shards[0].text)
    """
    spinner = ImageSpinner(
        importance_threshold=importance_threshold,
        output_format=output_format
    )

    result = await spinner.spin(image_path)

    return result
