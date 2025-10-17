"""
Resonance - Feature Interference Zone
======================================
The shed where feature extraction threads interfere and combine.

Exports:
- ResonanceShed: Multi-modal feature extraction
- FeatureThread: Individual feature extraction thread
- create_resonance_shed: Factory function
"""

from .shed import ResonanceShed, FeatureThread, create_resonance_shed

__all__ = ["ResonanceShed", "FeatureThread", "create_resonance_shed"]
