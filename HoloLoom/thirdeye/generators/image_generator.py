"""
ThirdEye Image Generator
========================

High-level API for generating 2D images from ThirdEye scenes.

Usage:
    from HoloLoom.thirdeye.generators import ImageGenerator

    generator = ImageGenerator()

    # From scene dictionary
    image = await generator.generate_from_scene(scene_dict)

    # From text description
    image = await generator.generate_from_text("A mystical forest...")

    # Save result
    image.save("output.png")

Philosophy: "See your thoughts become images."

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import io

from .prompt_builder import SceneToPrompt, build_prompt_from_scene
from .sd_backend import (
    StableDiffusionBackend,
    ComfyUIBackend,
    A1111Backend,
    GenerationParams,
    GenerationResult,
)


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    # Image dimensions
    width: int = 512
    height: int = 512

    # Generation quality
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: str = "euler"

    # Seed (-1 = random)
    seed: int = -1

    # Model (empty = default)
    model: str = ""

    # Style overrides
    additional_style: str = ""
    additional_negative: str = ""

    # Backend settings
    backend: str = "auto"  # auto, comfyui, a1111
    comfyui_port: int = 8188
    a1111_port: int = 7860
    timeout: float = 120.0


@dataclass
class GeneratedImage:
    """A generated image with metadata."""
    data: bytes
    width: int
    height: int
    prompt: str
    negative_prompt: str
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> Path:
        """Save image to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data)
        return path

    def to_pil(self):
        """Convert to PIL Image (requires pillow)."""
        try:
            from PIL import Image
            return Image.open(io.BytesIO(self.data))
        except ImportError:
            raise ImportError("PIL not available. Run: pip install pillow")

    def to_base64(self) -> str:
        """Convert to base64 string."""
        import base64
        return base64.b64encode(self.data).decode('utf-8')


class ImageGenerator:
    """
    High-level image generator for ThirdEye scenes.

    Automatically:
    - Converts scenes to optimized prompts
    - Detects available image generation backends
    - Handles errors gracefully
    - Returns clean result objects
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._backend: Optional[StableDiffusionBackend] = None
        self._prompt_builder = SceneToPrompt()

    async def _get_backend(self) -> StableDiffusionBackend:
        """Get or create the image generation backend."""
        if self._backend is None:
            if self.config.backend == "comfyui":
                self._backend = ComfyUIBackend(port=self.config.comfyui_port)
            elif self.config.backend == "a1111":
                self._backend = A1111Backend(port=self.config.a1111_port)
            else:
                # Auto-detect
                self._backend = StableDiffusionBackend(
                    comfyui_port=self.config.comfyui_port,
                    a1111_port=self.config.a1111_port,
                )
        return self._backend

    async def check_backend(self) -> Dict[str, Any]:
        """Check if image generation backend is available."""
        backend = await self._get_backend()
        available = await backend.health_check()

        return {
            "available": available,
            "backend": type(backend._active_backend).__name__ if hasattr(backend, '_active_backend') and backend._active_backend else "none",
            "message": (
                "Ready to generate images" if available
                else "No backend available. Start ComfyUI or AUTOMATIC1111."
            ),
        }

    async def generate_from_scene(
        self,
        scene: Dict[str, Any],
        config: Optional[GenerationConfig] = None,
    ) -> Union[GeneratedImage, Dict[str, str]]:
        """
        Generate an image from a ThirdEye scene.

        Args:
            scene: Scene dictionary from enhanced composer
            config: Optional config override

        Returns:
            GeneratedImage on success, or error dict on failure
        """
        cfg = config or self.config

        # Convert scene to prompts
        positive, negative = build_prompt_from_scene(
            scene,
            additional_style=cfg.additional_style,
            additional_negative=cfg.additional_negative.split(",") if cfg.additional_negative else None,
        )

        return await self._generate(positive, negative, cfg)

    async def generate_from_text(
        self,
        text: str,
        style: Optional[str] = None,
        negative: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Union[GeneratedImage, Dict[str, str]]:
        """
        Generate an image from text description.

        Args:
            text: Text description / prompt
            style: Additional style keywords
            negative: Negative prompt
            config: Optional config override

        Returns:
            GeneratedImage on success, or error dict on failure
        """
        cfg = config or self.config

        positive = text
        if style:
            positive = f"{text}, {style}"
        if cfg.additional_style:
            positive = f"{positive}, {cfg.additional_style}"

        neg = negative or ""
        if cfg.additional_negative:
            neg = f"{neg}, {cfg.additional_negative}" if neg else cfg.additional_negative

        # Add default negatives if empty
        if not neg:
            neg = "blurry, low quality, distorted, deformed, ugly, watermark, text"

        return await self._generate(positive, neg, cfg)

    async def _generate(
        self,
        positive: str,
        negative: str,
        config: GenerationConfig,
    ) -> Union[GeneratedImage, Dict[str, str]]:
        """Internal generation method."""
        backend = await self._get_backend()

        # Check availability
        if not await backend.health_check():
            return {
                "error": "No image generation backend available",
                "hint": (
                    "Start ComfyUI (port 8188) or AUTOMATIC1111 (port 7860). "
                    "See: https://github.com/comfyanonymous/ComfyUI"
                ),
                "prompt": positive,
            }

        # Build params
        params = GenerationParams(
            prompt=positive,
            negative_prompt=negative,
            width=config.width,
            height=config.height,
            steps=config.steps,
            cfg_scale=config.cfg_scale,
            sampler=config.sampler,
            seed=config.seed,
            model=config.model,
        )

        # Generate
        result = await backend.generate(params)

        if not result.success:
            return {
                "error": result.error or "Generation failed",
                "prompt": positive,
            }

        if not result.images:
            return {
                "error": "No images generated",
                "prompt": positive,
            }

        # Return first image
        seed = result.seeds[0] if result.seeds else -1
        return GeneratedImage(
            data=result.images[0],
            width=config.width,
            height=config.height,
            prompt=positive,
            negative_prompt=negative,
            seed=seed,
            metadata={
                "backend": result.metadata.get("backend", "unknown"),
                "generation_params": {
                    "steps": config.steps,
                    "cfg_scale": config.cfg_scale,
                    "sampler": config.sampler,
                },
            },
        )

    async def generate_batch(
        self,
        scenes: List[Dict[str, Any]],
        config: Optional[GenerationConfig] = None,
    ) -> List[Union[GeneratedImage, Dict[str, str]]]:
        """
        Generate images for multiple scenes.

        Args:
            scenes: List of scene dictionaries
            config: Optional config override

        Returns:
            List of GeneratedImage or error dicts
        """
        results = []
        for scene in scenes:
            result = await self.generate_from_scene(scene, config)
            results.append(result)
        return results

    async def close(self):
        """Close the generator and release resources."""
        if self._backend:
            await self._backend.close()
            self._backend = None


# Convenience function for quick generation
async def generate_image(
    source: Union[str, Dict[str, Any]],
    width: int = 512,
    height: int = 512,
    steps: int = 20,
    seed: int = -1,
) -> Union[GeneratedImage, Dict[str, str]]:
    """
    Quick image generation from text or scene.

    Args:
        source: Text prompt or scene dictionary
        width: Image width
        height: Image height
        steps: Generation steps
        seed: Random seed (-1 = random)

    Returns:
        GeneratedImage or error dict
    """
    config = GenerationConfig(
        width=width,
        height=height,
        steps=steps,
        seed=seed,
    )

    generator = ImageGenerator(config)
    try:
        if isinstance(source, str):
            return await generator.generate_from_text(source)
        else:
            return await generator.generate_from_scene(source)
    finally:
        await generator.close()


__all__ = [
    'GenerationConfig',
    'GeneratedImage',
    'ImageGenerator',
    'generate_image',
]
