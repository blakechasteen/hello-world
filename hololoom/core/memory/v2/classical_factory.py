"""
Classical AI Factory — One-call setup for the full classical AI stack.

Creates a SourceRegistry pre-loaded with:
  - TruthMaintenanceSystem (justification + retraction)
  - ChunkCompiler (SOAR-style compiled production learning)
  - CBRSystem (case-based reasoning)
  - DemonSystem (pandemonium pattern monitors)

Usage:
    from hololoom.memory.v2.classical_factory import create_classical_sources

    sources = create_classical_sources(activation_adapter=adapter)
    orchestrator = MatryoshkaOrchestrator(..., sources=sources)
"""

from typing import Optional

from .activation import ActivationAdapter
from .knowledge_source import SourceRegistry
from .tms import TruthMaintenanceSystem, TMSConfig
from .chunk_compiler import ChunkCompiler, ChunkConfig
from .cbr import CBRSystem, CBRConfig
from .demons import DemonSystem, DemonConfig


def create_classical_sources(
    activation_adapter: Optional[ActivationAdapter] = None,
    tms_config: Optional[TMSConfig] = None,
    chunk_config: Optional[ChunkConfig] = None,
    cbr_config: Optional[CBRConfig] = None,
    demon_config: Optional[DemonConfig] = None,
    activation_threshold: float = 0.1,
) -> SourceRegistry:
    """Create a SourceRegistry with all classical AI knowledge sources.

    Each source implements the KnowledgeSource protocol and fires at
    specific phases in the Matryoshka orchestrator pipeline:

      TMS:           POST_NAVIGATE, POST_CONFIDENCE
      ChunkCompiler: PRE_SHELL, POST_LOOP
      CBR:           POST_NAVIGATE, POST_LOOP
      Demons:        POST_GENERATE

    All sources are independently enable/disable-able via their configs.
    An empty SourceRegistry (no sources) preserves existing behavior.

    Args:
        activation_adapter: ACT-R activation for TMS retraction propagation
        tms_config: TMS tuning (trace depth, retraction penalty)
        chunk_config: ChunkCompiler tuning (cluster size, margin)
        cbr_config: CBR tuning (feature weights, max cases)
        demon_config: Demon thresholds (grounding, brevity, hedging)
        activation_threshold: Minimum activation for sources to fire

    Returns:
        SourceRegistry with all 4 classical AI sources registered
    """
    registry = SourceRegistry(activation_threshold=activation_threshold)

    registry.register(TruthMaintenanceSystem(
        activation_adapter=activation_adapter,
        config=tms_config or TMSConfig(),
    ))

    registry.register(ChunkCompiler(
        config=chunk_config or ChunkConfig(),
    ))

    registry.register(CBRSystem(
        config=cbr_config or CBRConfig(),
    ))

    registry.register(DemonSystem(
        config=demon_config or DemonConfig(),
    ))

    return registry
