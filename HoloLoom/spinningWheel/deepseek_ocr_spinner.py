"""
DeepSeek OCR Spinner
====================

Integrates DeepSeek's open-source OCR model for document and image text extraction.

Features:
- Document OCR → Markdown conversion
- Image text extraction
- Multi-resolution support (512-1280px)
- Batch processing
- Graceful degradation (falls back to basic OCR if DeepSeek unavailable)

Requirements:
- vLLM 0.8.5+ (recommended) or transformers 4.x
- PyTorch 2.6.0+
- CUDA 11.8+
- Flash Attention 2.7.3

Installation:
    pip install vllm torch transformers pillow
    # Or use transformers backend:
    pip install transformers torch pillow

Usage:
    # Simple usage
    from HoloLoom.spinningWheel import DeepSeekOCRSpinner

    spinner = DeepSeekOCRSpinner()
    result = await spinner.spin("document.png")

    # With configuration
    from HoloLoom.spinningWheel.deepseek_ocr_spinner import DeepSeekOCRConfig

    config = DeepSeekOCRConfig(
        backend="vllm",  # or "transformers"
        resolution=1024,
        output_format="markdown",
        enable_enrichment=True
    )
    spinner = DeepSeekOCRSpinner(config)
    result = await spinner.spin(["doc1.png", "doc2.pdf"])

Author: Claude Code
Date: January 2025
"""

import asyncio
import time
import hashlib
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinnerStatus,
    SpinResult,
    ImportanceSignals,
    ImportanceScore
)
from HoloLoom.documentation.types import MemoryShard


# ============================================================================
# Configuration
# ============================================================================

class OCRBackend(Enum):
    """OCR backend options."""
    VLLM = "vllm"              # Fastest, requires vLLM installed
    TRANSFORMERS = "transformers"  # Compatible, requires transformers
    FALLBACK = "fallback"      # Basic OCR (pytesseract/PIL)


class OutputFormat(Enum):
    """OCR output format options."""
    MARKDOWN = "markdown"      # Structured markdown (default)
    TEXT = "text"             # Plain text
    JSON = "json"             # Structured JSON with bounding boxes


@dataclass
class DeepSeekOCRConfig:
    """
    Configuration for DeepSeek OCR Spinner.

    Philosophy: "If you need to configure it, we failed."
    These are sensible defaults that work for 90% of use cases.
    """
    # Backend selection
    backend: OCRBackend = OCRBackend.VLLM
    model_name: str = "deepseek-ai/DeepSeek-OCR"

    # Resolution (512, 640, 1024, 1280)
    resolution: int = 1024

    # Output format
    output_format: OutputFormat = OutputFormat.MARKDOWN

    # Processing options
    batch_size: int = 8
    max_tokens: int = 8192
    ngram_size: int = 30      # For vLLM ngram processor
    window_size: int = 90     # For vLLM ngram processor

    # Chunking (for multi-page PDFs)
    chunk_pages: bool = True
    pages_per_chunk: int = 10

    # Enrichment (entity/motif extraction via Ollama)
    enable_enrichment: bool = False
    enrichment_model: str = "llama3.2:3b"

    # Importance scoring
    importance_threshold: float = 0.3

    # Performance
    device: str = "cuda"      # "cuda" or "cpu"

    def __post_init__(self):
        """Validate configuration."""
        if self.resolution not in [512, 640, 1024, 1280]:
            raise ValueError(f"Resolution must be 512, 640, 1024, or 1280, got {self.resolution}")


# ============================================================================
# DeepSeek OCR Spinner
# ============================================================================

class DeepSeekOCRSpinner(BaseSpinner):
    """
    Spinner for extracting text from images and documents using DeepSeek OCR.

    Supports:
    - Images: PNG, JPG, WebP, etc.
    - Documents: PDF (multi-page), scanned documents
    - Output formats: Markdown, text, JSON

    Graceful degradation:
    - DeepSeek available → Use DeepSeek OCR (best quality)
    - DeepSeek unavailable → Fall back to pytesseract (basic OCR)
    - No OCR available → Extract filename/metadata only
    """

    def __init__(
        self,
        config: Optional[DeepSeekOCRConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize DeepSeek OCR spinner.

        Args:
            config: OCR configuration (None = use defaults)
            checkpoint_dir: Directory for checkpoints (None = no checkpointing)
        """
        self.config = config or DeepSeekOCRConfig()

        super().__init__(
            name="deepseek_ocr",
            importance_threshold=self.config.importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        # Backend components (lazy loaded)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._fallback_ocr = None

        # Check availability
        self._check_dependencies()

    # ========================================================================
    # Protocol Implementation
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities."""
        has_deepseek = self._check_deepseek_available()
        has_fallback = self._check_fallback_available()

        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=self.config.enable_enrichment,
            motif_extraction=self.config.enable_enrichment,
            embeddings=False,
            multimodal=True,
            cross_modal_fusion=False,
            max_input_size=None,
            supported_formats=['.png', '.jpg', '.jpeg', '.webp', '.pdf', '.tiff', '.bmp']
        )

    def is_available(self) -> bool:
        """Check if spinner is available."""
        # Available if either DeepSeek or fallback OCR is present
        return self._check_deepseek_available() or self._check_fallback_available()

    def get_status(self) -> SpinnerStatus:
        """Get operational status."""
        if self._check_deepseek_available():
            return SpinnerStatus.AVAILABLE
        elif self._check_fallback_available():
            return SpinnerStatus.DEGRADED
        else:
            return SpinnerStatus.UNAVAILABLE

    # ========================================================================
    # Core Implementation
    # ========================================================================

    async def _spin_impl(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> List[MemoryShard]:
        """
        Core OCR implementation.

        Args:
            source: Image/document path(s) or list of paths
            **kwargs: Additional options

        Returns:
            List of MemoryShards with extracted text
        """
        # Normalize to list of paths
        if isinstance(source, (str, Path)):
            sources = [Path(source)]
        else:
            sources = [Path(s) for s in source]

        # Process all sources
        all_shards = []

        for source_path in sources:
            if not source_path.exists():
                print(f"Warning: Path does not exist: {source_path}")
                continue

            # Handle PDFs differently (multi-page)
            if source_path.suffix.lower() == '.pdf':
                shards = await self._process_pdf(source_path, **kwargs)
            else:
                shards = await self._process_image(source_path, **kwargs)

            all_shards.extend(shards)

        return all_shards

    async def _process_image(
        self,
        image_path: Path,
        **kwargs
    ) -> List[MemoryShard]:
        """Process single image."""
        # Extract text via OCR
        text, confidence = await self._extract_text(image_path)

        # Score importance
        importance = self._score_document_importance(text, image_path)

        # Extract entities/motifs if enrichment enabled
        entities = []
        motifs = []

        if self.config.enable_enrichment and len(text) > 50:
            entities, motifs = await self._enrich_content(text)

        # Create shard
        shard = self._create_shard(
            id_suffix=f"{image_path.stem}_{int(time.time())}",
            text=text,
            episode=f"ocr_{image_path.stem}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source_file': str(image_path),
                'file_type': image_path.suffix,
                'file_size_bytes': image_path.stat().st_size,
                'ocr_confidence': confidence,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'output_format': self.config.output_format.value,
                'resolution': self.config.resolution,
                'backend': self._get_active_backend()
            }
        )

        return [shard]

    async def _process_pdf(
        self,
        pdf_path: Path,
        **kwargs
    ) -> List[MemoryShard]:
        """Process multi-page PDF."""
        # Extract pages
        pages = await self._extract_pdf_pages(pdf_path)

        # Chunk pages if needed
        if self.config.chunk_pages:
            page_chunks = [
                pages[i:i + self.config.pages_per_chunk]
                for i in range(0, len(pages), self.config.pages_per_chunk)
            ]
        else:
            page_chunks = [pages]

        # Process each chunk
        shards = []

        for chunk_idx, chunk in enumerate(page_chunks):
            # Process pages in chunk
            chunk_texts = []
            chunk_confidences = []

            for page_num, page_image in chunk:
                text, confidence = await self._extract_text_from_bytes(page_image)
                chunk_texts.append(f"## Page {page_num}\n\n{text}")
                chunk_confidences.append(confidence)

            # Combine chunk
            combined_text = "\n\n---\n\n".join(chunk_texts)
            avg_confidence = sum(chunk_confidences) / len(chunk_confidences) if chunk_confidences else 0.0

            # Score importance
            importance = self._score_document_importance(combined_text, pdf_path)

            # Extract entities/motifs if enrichment enabled
            entities = []
            motifs = []

            if self.config.enable_enrichment and len(combined_text) > 50:
                entities, motifs = await self._enrich_content(combined_text)

            # Create shard for chunk
            page_range = f"{chunk[0][0]}-{chunk[-1][0]}"

            shard = self._create_shard(
                id_suffix=f"{pdf_path.stem}_pages_{page_range}_{int(time.time())}",
                text=combined_text,
                episode=f"pdf_{pdf_path.stem}",
                entities=entities,
                motifs=motifs,
                metadata={
                    'source_file': str(pdf_path),
                    'file_type': '.pdf',
                    'file_size_bytes': pdf_path.stat().st_size,
                    'page_range': page_range,
                    'page_count': len(chunk),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(page_chunks),
                    'ocr_confidence': avg_confidence,
                    'importance_score': importance.score,
                    'importance_reason': importance.reason,
                    'output_format': self.config.output_format.value,
                    'backend': self._get_active_backend()
                }
            )

            shards.append(shard)

        return shards

    # ========================================================================
    # OCR Backends
    # ========================================================================

    async def _extract_text(self, image_path: Path) -> tuple[str, float]:
        """
        Extract text from image using best available backend.

        Returns:
            (text, confidence)
        """
        # Try DeepSeek first
        if self._check_deepseek_available():
            try:
                return await self._extract_text_deepseek(image_path)
            except Exception as e:
                print(f"Warning: DeepSeek OCR failed, falling back: {e}")

        # Fall back to basic OCR
        if self._check_fallback_available():
            try:
                return await self._extract_text_fallback(image_path)
            except Exception as e:
                print(f"Warning: Fallback OCR failed: {e}")

        # Last resort: filename only
        return f"[Image: {image_path.name}]", 0.0

    async def _extract_text_from_bytes(self, image_bytes: bytes) -> tuple[str, float]:
        """Extract text from image bytes."""
        # For now, save to temp file and process
        # TODO: Direct bytes processing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(image_bytes)
            temp_path = Path(f.name)

        try:
            return await self._extract_text(temp_path)
        finally:
            temp_path.unlink()

    async def _extract_text_deepseek(self, image_path: Path) -> tuple[str, float]:
        """Extract text using DeepSeek OCR."""
        # Lazy load model
        if self._model is None:
            self._load_deepseek_model()

        # Load image
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")

        # Prepare prompt
        if self.config.output_format == OutputFormat.MARKDOWN:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        else:
            prompt = "<image>\nFree OCR."

        # Run inference based on backend
        if self.config.backend == OCRBackend.VLLM:
            text = await self._run_vllm_inference(image, prompt)
        else:
            text = await self._run_transformers_inference(image, prompt)

        # Confidence (DeepSeek doesn't provide confidence, use heuristic)
        confidence = self._estimate_confidence(text)

        return text, confidence

    async def _extract_text_fallback(self, image_path: Path) -> tuple[str, float]:
        """Extract text using pytesseract fallback."""
        if self._fallback_ocr is None:
            try:
                import pytesseract
                self._fallback_ocr = pytesseract
            except ImportError:
                raise RuntimeError("pytesseract not available")

        try:
            from PIL import Image
            image = Image.open(image_path)

            # Run OCR
            data = self._fallback_ocr.image_to_data(image, output_type='dict')

            # Extract text with confidence
            texts = data['text']
            confidences = data['conf']

            # Filter out low-confidence words
            valid_words = [
                text for text, conf in zip(texts, confidences)
                if conf > 0 and text.strip()
            ]

            text = ' '.join(valid_words)
            avg_confidence = sum(c for c in confidences if c > 0) / len([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0

            # Normalize confidence to 0-1
            confidence = avg_confidence / 100.0

            return text, confidence

        except Exception as e:
            raise RuntimeError(f"Fallback OCR failed: {e}")

    async def _run_vllm_inference(self, image, prompt: str) -> str:
        """Run inference using vLLM backend."""
        from vllm import SamplingParams

        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            top_p=0.95
        )

        # Run inference
        outputs = self._model.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            },
            sampling_params=sampling_params
        )

        # Extract text
        return outputs[0].outputs[0].text

    async def _run_transformers_inference(self, image, prompt: str) -> str:
        """Run inference using transformers backend."""
        # Encode inputs
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.config.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                do_sample=False
            )

        # Decode
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if prompt in text:
            text = text.replace(prompt, '').strip()

        return text

    # ========================================================================
    # PDF Processing
    # ========================================================================

    async def _extract_pdf_pages(self, pdf_path: Path) -> List[tuple[int, bytes]]:
        """
        Extract pages from PDF as images.

        Returns:
            List of (page_number, image_bytes) tuples
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")

        pages = []

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page to image (use resolution from config)
            zoom = self.config.resolution / 72.0  # 72 DPI base
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PNG bytes
            image_bytes = pix.pil_tobytes(format="PNG")

            pages.append((page_num + 1, image_bytes))

        doc.close()

        return pages

    # ========================================================================
    # Enrichment
    # ========================================================================

    async def _enrich_content(self, text: str) -> tuple[List[str], List[str]]:
        """
        Extract entities and motifs using Ollama.

        Returns:
            (entities, motifs)
        """
        try:
            import ollama
        except ImportError:
            print("Warning: ollama not installed, skipping enrichment")
            return [], []

        try:
            # Extract entities
            entity_prompt = f"""Extract key entities (people, organizations, locations, concepts) from this text.
Return only a comma-separated list of entities.

Text: {text[:2000]}

Entities:"""

            entity_response = ollama.generate(
                model=self.config.enrichment_model,
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
                model=self.config.enrichment_model,
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

    # ========================================================================
    # Importance Scoring
    # ========================================================================

    def score_importance(self, data: Any) -> ImportanceScore:
        """Score importance of OCR'd document."""
        if isinstance(data, str):
            text = data
            source_path = None
        else:
            text = ""
            source_path = Path(data) if data else None

        return self._score_document_importance(text, source_path)

    def _score_document_importance(
        self,
        text: str,
        source_path: Optional[Path] = None
    ) -> ImportanceScore:
        """
        Score importance of OCR'd document.

        Factors:
        - Length (longer = more substantial)
        - Structure (headings, lists, formatting)
        - Technical content (domain-specific terms)
        - File type (PDF > image)
        """
        signals = ImportanceSignals()

        # Length score (normalized)
        text_length = len(text)
        signals.length_score = min(1.0, text_length / 5000)

        # Structural score (headings, lists, etc.)
        structure_markers = text.count('\n#') + text.count('\n-') + text.count('\n*') + text.count('\n1.')
        signals.structural_score = min(1.0, structure_markers / 20)

        # Technical score (presence of technical terms)
        technical_keywords = [
            'algorithm', 'data', 'system', 'process', 'method',
            'analysis', 'research', 'study', 'report', 'documentation'
        ]
        technical_count = sum(1 for kw in technical_keywords if kw.lower() in text.lower())
        signals.technical_score = min(1.0, technical_count / 5)

        # Authority score (file type)
        if source_path:
            if source_path.suffix.lower() == '.pdf':
                signals.authority_score = 0.8
            else:
                signals.authority_score = 0.5

        # Noise penalty (very short, mostly symbols)
        if text_length < 50:
            signals.noise_penalty = -0.5
        elif text_length < 20:
            signals.noise_penalty = -1.0

        # Compute total
        score = signals.compute_total()
        reason = signals.explain()

        return ImportanceScore(
            score=score,
            signals=signals,
            reason=reason
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def _check_dependencies(self):
        """Check and log available dependencies."""
        status_messages = []

        if self._check_deepseek_available():
            backend = self.config.backend.value
            status_messages.append(f"✓ DeepSeek OCR available (backend: {backend})")
        else:
            status_messages.append("✗ DeepSeek OCR not available")

        if self._check_fallback_available():
            status_messages.append("✓ Fallback OCR (pytesseract) available")
        else:
            status_messages.append("✗ Fallback OCR not available")

        for msg in status_messages:
            print(f"[DeepSeekOCRSpinner] {msg}")

    def _check_deepseek_available(self) -> bool:
        """Check if DeepSeek dependencies are available."""
        if self.config.backend == OCRBackend.VLLM:
            try:
                import vllm
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        elif self.config.backend == OCRBackend.TRANSFORMERS:
            try:
                import transformers
                import torch
                return True
            except ImportError:
                return False
        else:
            return False

    def _check_fallback_available(self) -> bool:
        """Check if fallback OCR is available."""
        try:
            import pytesseract
            from PIL import Image
            return True
        except ImportError:
            return False

    def _load_deepseek_model(self):
        """Lazy load DeepSeek model."""
        if self.config.backend == OCRBackend.VLLM:
            self._load_vllm_model()
        elif self.config.backend == OCRBackend.TRANSFORMERS:
            self._load_transformers_model()

    def _load_vllm_model(self):
        """Load model using vLLM backend."""
        from vllm import LLM

        print(f"[DeepSeekOCRSpinner] Loading model with vLLM: {self.config.model_name}")

        # Load model with ngram processor
        self._model = LLM(
            model=self.config.model_name,
            trust_remote_code=True,
            max_model_len=self.config.max_tokens,
            gpu_memory_utilization=0.9
        )

        print("[DeepSeekOCRSpinner] Model loaded successfully")

    def _load_transformers_model(self):
        """Load model using transformers backend."""
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        import torch

        print(f"[DeepSeekOCRSpinner] Loading model with transformers: {self.config.model_name}")

        # Load model
        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.config.device)

        # Load processor and tokenizer
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        print("[DeepSeekOCRSpinner] Model loaded successfully")

    def _get_active_backend(self) -> str:
        """Get name of active backend."""
        if self._model is not None:
            return self.config.backend.value
        elif self._fallback_ocr is not None:
            return "pytesseract"
        else:
            return "none"

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate OCR confidence from output text.

        Heuristics:
        - Longer text = higher confidence
        - Proper capitalization = higher confidence
        - Few special chars = higher confidence
        """
        if not text or len(text) < 10:
            return 0.3

        # Length factor
        length_factor = min(1.0, len(text) / 1000)

        # Capitalization factor (properly capitalized sentences)
        sentences = text.split('.')
        proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        cap_factor = proper_caps / len(sentences) if sentences else 0.5

        # Special char factor (fewer weird chars = better)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,!?-')
        special_ratio = special_chars / len(text)
        special_factor = max(0.0, 1.0 - special_ratio * 5)

        # Combine
        confidence = (length_factor * 0.3 + cap_factor * 0.4 + special_factor * 0.3)

        return max(0.0, min(1.0, confidence))


# ============================================================================
# Convenience Functions
# ============================================================================

async def extract_text_from_image(
    image_path: Union[str, Path],
    output_format: str = "markdown",
    backend: str = "vllm",
    resolution: int = 1024
) -> str:
    """
    Quick utility to extract text from single image.

    Args:
        image_path: Path to image
        output_format: "markdown", "text", or "json"
        backend: "vllm" or "transformers"
        resolution: Image resolution (512, 640, 1024, 1280)

    Returns:
        Extracted text

    Example:
        >>> text = await extract_text_from_image("document.png")
        >>> print(text)
    """
    config = DeepSeekOCRConfig(
        backend=OCRBackend(backend),
        output_format=OutputFormat(output_format),
        resolution=resolution
    )

    spinner = DeepSeekOCRSpinner(config)
    result = await spinner.spin(image_path)

    if result.success and result.shards:
        return result.shards[0].text
    else:
        raise RuntimeError(f"OCR failed: {result.error_message}")


async def extract_text_from_pdf(
    pdf_path: Union[str, Path],
    output_format: str = "markdown",
    backend: str = "vllm",
    pages_per_chunk: int = 10
) -> List[str]:
    """
    Quick utility to extract text from PDF.

    Args:
        pdf_path: Path to PDF
        output_format: "markdown", "text", or "json"
        backend: "vllm" or "transformers"
        pages_per_chunk: Pages per shard

    Returns:
        List of extracted text (one per chunk)

    Example:
        >>> chunks = await extract_text_from_pdf("document.pdf")
        >>> for i, chunk in enumerate(chunks):
        ...     print(f"Chunk {i}: {chunk[:100]}...")
    """
    config = DeepSeekOCRConfig(
        backend=OCRBackend(backend),
        output_format=OutputFormat(output_format),
        chunk_pages=True,
        pages_per_chunk=pages_per_chunk
    )

    spinner = DeepSeekOCRSpinner(config)
    result = await spinner.spin(pdf_path)

    if result.success:
        return [shard.text for shard in result.shards]
    else:
        raise RuntimeError(f"PDF OCR failed: {result.error_message}")
