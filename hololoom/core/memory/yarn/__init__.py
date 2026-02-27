"""
Yarn - Lineage and Memory Tracking
==================================

Logs ancestry, perturbation seeds, rewards, embeddings.
Generates lineage graphs and analyzes centrality.

Moved to hololoom/memory/yarn/ in December 2025 consolidation.
"""

from .eggroll_yarn import Yarn

# Alias for backward compatibility
EggrollYarn = Yarn

__all__ = [
    "Yarn",
    "EggrollYarn",  # Backward compatibility alias
]
