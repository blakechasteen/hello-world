"""
SpinningWheel
=============
Lightweight input adapters for HoloLoom.

Converts raw modality data -> MemoryShards -> Orchestrator

Philosophy:
- Keep spinners simple and focused
- Standardize output (MemoryShards)
- Optional pre-enrichment with context
- Let the Orchestrator do heavy lifting

Available Spinners:
- AudioSpinner: Converts audio transcripts + metadata -> MemoryShards
- YouTubeSpinner: Extracts YouTube video transcripts -> MemoryShards
- TextSpinner: Plain text/markdown -> MemoryShards
- CodeSpinner: (Future) Code/git diffs -> MemoryShards
"""

from .base import BaseSpinner, SpinnerConfig
from .audio import AudioSpinner
from .youtube import YouTubeSpinner, YouTubeSpinnerConfig, transcribe_youtube
from .text import TextSpinner, TextSpinnerConfig, spin_text
from .code import CodeSpinner, CodeSpinnerConfig, spin_code_file, spin_git_diff, spin_repository

__all__ = [
    "BaseSpinner",
    "SpinnerConfig",
    "AudioSpinner",
    "YouTubeSpinner",
    "YouTubeSpinnerConfig",
    "transcribe_youtube",
    "TextSpinner",
    "TextSpinnerConfig",
    "spin_text",
    "CodeSpinner",
    "CodeSpinnerConfig",
    "spin_code_file",
    "spin_git_diff",
    "spin_repository",
    "create_spinner"
]

__version__ = "0.1.0"


def create_spinner(modality: str, config: SpinnerConfig = None):
    """
    Factory function to create spinners.
    
    Args:
        modality: 'audio', 'text', 'code', etc.
        config: SpinnerConfig (optional)
        
    Returns:
        Spinner instance
        
    Example:
        spinner = create_spinner('audio', SpinnerConfig(enable_enrichment=True))
        shards = await spinner.spin(raw_data)
    """
    if config is None:
        config = SpinnerConfig()
    
    spinners = {
        'audio': AudioSpinner,
        'youtube': YouTubeSpinner,
        'text': TextSpinner,
        'code': CodeSpinner,
        # Future spinners:
        # 'video': VideoSpinner,
    }
    
    spinner_class = spinners.get(modality)
    if not spinner_class:
        available = ', '.join(spinners.keys())
        raise ValueError(f"Unknown modality: {modality}. Available: {available}")
    
    return spinner_class(config)