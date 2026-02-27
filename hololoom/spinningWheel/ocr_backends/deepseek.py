"""
DeepSeek OCR Backend
====================

High-quality OCR backend using DeepSeek's vision-language model.

Features:
- Best-in-class quality
- 10x compression (1000 words → 100 tokens)
- Multiple resolution support
- Structured markdown output
- vLLM or transformers backend

Requirements:
- vLLM 0.8.5+ OR transformers 4.x
- PyTorch 2.6.0+
- CUDA 11.8+
"""

from typing import Union, Any, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from HoloLoom.spinningWheel.ocr_protocol import (
    BaseOCRBackend,
    OCRResult,
    OCRQuality,
    OCROutputFormat
)


class DeepSeekBackendType(Enum):
    """DeepSeek backend options."""
    VLLM = "vllm"              # Fastest
    TRANSFORMERS = "transformers"  # Compatible


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek backend."""
    backend_type: DeepSeekBackendType = DeepSeekBackendType.VLLM
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    resolution: int = 1024  # 512, 640, 1024, or 1280
    max_tokens: int = 8192
    device: str = "cuda"


class DeepSeekOCRBackend(BaseOCRBackend):
    """
    DeepSeek OCR backend.

    Uses DeepSeek's vision-language model for high-quality OCR.
    Best quality, requires CUDA.
    """

    def __init__(self, config: DeepSeekConfig = None):
        """
        Initialize DeepSeek backend.

        Args:
            config: DeepSeek configuration (None = use defaults)
        """
        super().__init__(name="deepseek")

        self.config = config or DeepSeekConfig()

        # Lazy load model
        self._model = None
        self._processor = None
        self._tokenizer = None

    def is_available(self) -> bool:
        """Check if DeepSeek dependencies are available."""
        try:
            if self.config.backend_type == DeepSeekBackendType.VLLM:
                import vllm
                import torch
                return torch.cuda.is_available()
            else:
                import transformers
                import torch
                return True
        except ImportError:
            return False

    def get_quality(self) -> OCRQuality:
        """Excellent quality."""
        return OCRQuality.EXCELLENT

    async def _extract_text_impl(
        self,
        image: Any,
        output_format: OCROutputFormat,
        **kwargs
    ) -> OCRResult:
        """
        Extract text using DeepSeek OCR.

        Args:
            image: PIL Image
            output_format: Desired format
            **kwargs: Additional options

        Returns:
            OCRResult with high-quality text
        """
        # Lazy load model
        if self._model is None:
            self._load_model()

        # Get image size
        image_size = image.size if hasattr(image, 'size') else (0, 0)

        # Prepare prompt based on format
        if output_format == OCROutputFormat.MARKDOWN:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        else:
            prompt = "<image>\nFree OCR."

        # Run inference
        if self.config.backend_type == DeepSeekBackendType.VLLM:
            text = await self._run_vllm_inference(image, prompt)
        else:
            text = await self._run_transformers_inference(image, prompt)

        # Estimate confidence (DeepSeek doesn't provide confidence scores)
        confidence = self._estimate_confidence(text)

        return OCRResult(
            text=text,
            confidence=confidence,
            image_size=image_size,
            metadata={
                'backend_type': self.config.backend_type.value,
                'resolution': self.config.resolution,
                'model': self.config.model_name
            }
        )

    async def batch_extract(
        self,
        images: List[Any],
        output_format: OCROutputFormat = OCROutputFormat.TEXT,
        **kwargs
    ) -> List[OCRResult]:
        """
        Batch extract (vLLM supports efficient batching).

        Args:
            images: List of PIL Images
            output_format: Desired format
            **kwargs: Additional options

        Returns:
            List of OCRResults
        """
        # Lazy load model
        if self._model is None:
            self._load_model()

        # vLLM supports batching, transformers doesn't (yet)
        if self.config.backend_type == DeepSeekBackendType.VLLM:
            return await self._batch_extract_vllm(images, output_format, **kwargs)
        else:
            # Fall back to sequential for transformers
            return await super().batch_extract(images, output_format, **kwargs)

    # ========================================================================
    # Model Loading
    # ========================================================================

    def _load_model(self):
        """Load DeepSeek model."""
        if self.config.backend_type == DeepSeekBackendType.VLLM:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_vllm(self):
        """Load model with vLLM backend."""
        from vllm import LLM

        print(f"[DeepSeekOCRBackend] Loading model with vLLM: {self.config.model_name}")

        self._model = LLM(
            model=self.config.model_name,
            trust_remote_code=True,
            max_model_len=self.config.max_tokens,
            gpu_memory_utilization=0.9
        )

        print("[DeepSeekOCRBackend] Model loaded successfully")

    def _load_transformers(self):
        """Load model with transformers backend."""
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        import torch

        print(f"[DeepSeekOCRBackend] Loading model with transformers: {self.config.model_name}")

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

        print("[DeepSeekOCRBackend] Model loaded successfully")

    # ========================================================================
    # Inference
    # ========================================================================

    async def _run_vllm_inference(self, image: Any, prompt: str) -> str:
        """Run inference using vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            top_p=0.95
        )

        outputs = self._model.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            },
            sampling_params=sampling_params
        )

        return outputs[0].outputs[0].text

    async def _run_transformers_inference(self, image: Any, prompt: str) -> str:
        """Run inference using transformers."""
        import torch

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

    async def _batch_extract_vllm(
        self,
        images: List[Any],
        output_format: OCROutputFormat,
        **kwargs
    ) -> List[OCRResult]:
        """Batch extract using vLLM (efficient batching)."""
        from vllm import SamplingParams

        # Prepare prompts
        if output_format == OCROutputFormat.MARKDOWN:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        else:
            prompt = "<image>\nFree OCR."

        # Prepare batch
        batch_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            }
            for image in images
        ]

        # Run batch inference
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            top_p=0.95
        )

        outputs = self._model.generate(batch_inputs, sampling_params=sampling_params)

        # Create results
        results = []

        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            confidence = self._estimate_confidence(text)
            image_size = images[i].size if hasattr(images[i], 'size') else (0, 0)

            result = OCRResult(
                text=text,
                confidence=confidence,
                image_size=image_size,
                metadata={
                    'batch_index': i,
                    'backend_type': 'vllm',
                    'resolution': self.config.resolution
                }
            )

            results.append(result)

        return results

    # ========================================================================
    # Utilities
    # ========================================================================

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate OCR confidence from output text.

        DeepSeek doesn't provide confidence scores, so we use heuristics:
        - Longer text = higher confidence
        - Proper formatting = higher confidence
        - Few errors = higher confidence
        """
        if not text or len(text) < 10:
            return 0.5

        # Length factor
        length_factor = min(1.0, len(text) / 1000)

        # Markdown structure (if markdown format)
        if '\n#' in text or '\n-' in text or '\n*' in text:
            structure_factor = 0.9
        else:
            structure_factor = 0.7

        # Capitalization (properly capitalized sentences)
        sentences = text.split('.')
        proper_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        cap_factor = proper_caps / len(sentences) if sentences else 0.7

        # Combine (weighted toward structure since DeepSeek is high quality)
        confidence = (
            length_factor * 0.2 +
            structure_factor * 0.5 +
            cap_factor * 0.3
        )

        # DeepSeek is high quality, boost baseline
        confidence = max(0.75, confidence)

        return min(1.0, confidence)