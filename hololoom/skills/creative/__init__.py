"""
Creative Skills Package
=======================
Skills for visual, audio, and document generation.

Available Skills:
- stable_diffusion: AI image generation
- ffmpeg: Video/audio processing
- latex_compiler: Professional document generation
- graphviz: Architecture diagram generation
- image_processing: Image processing with PIL/Pillow
"""

from .ffmpeg_skill import FFmpegSkill
from .graphviz_skill import GraphvizSkill
from .image_processing_skill import ImageProcessingSkill
from .latex_compiler import LaTeXCompilerSkill
from .stable_diffusion import StableDiffusionSkill

__all__ = ['StableDiffusionSkill', 'FFmpegSkill', 'LaTeXCompilerSkill', 'GraphvizSkill', 'ImageProcessingSkill']
