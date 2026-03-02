"""
ThirdEye Image Generators
=========================

Integration with open-source image generation backends:
- Stable Diffusion (ComfyUI, AUTOMATIC1111)
- Flux (Black Forest Labs)
- Local/API-based generation

Converts ThirdEye scene descriptions to optimized prompts
and generates actual 2D images.

Created: 2025-12-01
Author: HoloLoom Team
"""

from .prompt_builder import SceneToPrompt, build_prompt_from_scene
from .sd_backend import StableDiffusionBackend, ComfyUIBackend, A1111Backend
from .image_generator import ImageGenerator, GenerationConfig, GeneratedImage

__all__ = [
    'SceneToPrompt',
    'build_prompt_from_scene',
    'StableDiffusionBackend',
    'ComfyUIBackend',
    'A1111Backend',
    'ImageGenerator',
    'GenerationConfig',
    'GeneratedImage',
]
