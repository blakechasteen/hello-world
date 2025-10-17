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
- TextSpinner: (Future) Plain text/markdown -> MemoryShards
- CodeSpinner: (Future) Code/git diffs -> MemoryShards
"""

from .base import BaseSpinner, SpinnerConfig
from .audio import AudioSpinner

__all__ = ["BaseSpinner", "SpinnerConfig", "AudioSpinner", "create_spinner"]

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
        # Future spinners:
        # 'text': TextSpinner,
        # 'code': CodeSpinner,
        # 'video': VideoSpinner,
    }
    
    spinner_class = spinners.get(modality)
    if not spinner_class:
        available = ', '.join(spinners.keys())
        raise ValueError(f"Unknown modality: {modality}. Available: {available}")
    
    return spinner_class(config)