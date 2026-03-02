"""
HoloLoom Synthesis
==================
Extract patterns and synthesize training data from high-quality memories.

Transforms filtered conversations into:
- Q&A pairs
- Reasoning chains
- Knowledge triples
- Training datasets

The gold mine: Your filtered signal, ready for learning.
"""

from .enriched_memory import EnrichedMemory, ReasoningType, MemoryEnricher
from .pattern_extractor import PatternExtractor, Pattern, PatternType
from .data_synthesizer import DataSynthesizer, TrainingExample, SynthesisConfig

__all__ = [
    'EnrichedMemory',
    'ReasoningType',
    'MemoryEnricher',
    'PatternExtractor',
    'Pattern',
    'PatternType',
    'DataSynthesizer',
    'TrainingExample',
    'SynthesisConfig'
]

__version__ = '0.1.0'
