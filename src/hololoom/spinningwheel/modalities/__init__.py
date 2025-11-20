"""
HoloLoom Modality Spinners
===========================
Specialized spinners for different input modalities.

Modality spinners handle:
- Audio transcripts (audio.py)
- Code repositories (code.py)
- Images and receipts (image.py)
- Text documents (text.py)
- Website content (website.py)
- YouTube videos (youtube.py)

All spinners implement the SpinnerProtocol for consistency.
"""

from .audio import AudioSpinner
from .code import CodeSpinner
from .image import ImageSpinner
from .text import TextSpinner
from .website import WebsiteSpinner
from .youtube import YouTubeSpinner

__all__ = [
    'AudioSpinner',
    'CodeSpinner',
    'ImageSpinner',
    'TextSpinner',
    'WebsiteSpinner',
    'YouTubeSpinner',
]
