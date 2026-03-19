"""
Memory Bus v2 — Matryoshka Architecture with Cognitive Core

Memory drives the model, not model drives memory.

Components:
  navigator.py        — PPR graph traversal for context construction
  formatter.py        — Fixed-budget context packer
  confidence.py       — Structural + narrative confidence estimation
  orchestrator.py     — Shell loop: PRIME → VERIFY → FLAG with conditional execution
  training.py         — Audit-driven optimization (bandit + refiner + DSPy)
  blackboard.py       — Shared cognitive workspace (SOAR + Blackboard + GWT)
  knowledge_source.py — Classical AI composition layer (KnowledgeSource protocol)
  tms.py              — Truth Maintenance System (justification + retraction)
  chunk_compiler.py   — SOAR-style compiled production learning
  cbr.py              — Case-Based Reasoning (retrieve + reuse past experiences)
  demons.py           — Pandemonium pattern monitors (grounding, brevity, hedging)
"""

from .activation import (
    ActivationAdapter,
    ActivationConfig,
)
from .blackboard import (
    Blackboard,
    ProductionRule,
    WorkingMemory,
    compute_ppr_entropy,
    default_skip_flag_rules,
    default_skip_verify_rules,
    extract_seed_sources,
)
from .cbr import (
    Case,
    CBRConfig,
    CBRSystem,
)
from .chunk_compiler import (
    ChunkAction,
    ChunkCompiler,
    ChunkConfig,
    CompiledChunk,
    ExperienceRecord,
)
from .classical_factory import create_classical_sources
from .cold_start import (
    ColdStartConfig,
    ColdStartHandler,
)
from .confidence import (
    ConfidenceConfig,
    ConfidenceEstimator,
    DualConfidence,
    NarrativeConfidence,
    StructuralConfidence,
    find_contradicting_pairs,
)
from .demons import (
    DemonAlert,
    DemonConfig,
    DemonSystem,
    check_brevity,
    check_grounding,
    check_hedging,
    check_repetition,
)
from .formatter import (
    ContextBlock,
    FormatConfig,
    FormattedItem,
    Formatter,
    ResolutionThresholds,
)
from .hierarchy import (
    ClusterInfo,
    HierarchyConfig,
    HierarchyManager,
)
from .knowledge_source import (
    KnowledgeSource,
    Phase,
    SourceRegistry,
)
from .navigator import (
    GraphBackend,
    LiteBusGraph,
    Navigator,
    NavigatorResult,
    NetworkXGraph,
    PPRConfig,
    SeedNode,
    extract_seeds,
    personalized_pagerank,
)
from .orchestrator import (
    DEFAULT_SHELLS,
    FLAG_SCHEMA,
    SHELL_SCHEMAS,
    VERIFY_SCHEMA,
    LoopResult,
    MatryoshkaOrchestrator,
    ShellConfig,
    ShellResult,
    ShellType,
    parse_structured_response,
)
from .tms import (
    JustificationRecord,
    Retraction,
    TMSConfig,
    TruthMaintenanceSystem,
)
from .training import (
    CONFIG_PRESETS,
    AuditCollector,
    AuditSummary,
    BanditArm,
    ContextualBanditAdapter,
    OptimizationResult,
    ParameterBandit,
    PromptRefiner,
    RefinementResult,
    RewardJudge,
    Signature,
    SignatureOptimizer,
    TrainingExample,
    TrainingPipeline,
)

__all__ = [
    # Navigator
    "Navigator", "NavigatorResult", "PPRConfig", "SeedNode",
    "LiteBusGraph", "NetworkXGraph", "GraphBackend",
    "extract_seeds", "personalized_pagerank",
    # Formatter
    "Formatter", "FormatConfig", "ContextBlock", "FormattedItem",
    "ResolutionThresholds",
    # Hierarchy
    "HierarchyManager", "HierarchyConfig", "ClusterInfo",
    # Confidence
    "ConfidenceEstimator", "ConfidenceConfig", "DualConfidence",
    "StructuralConfidence", "NarrativeConfidence",
    "find_contradicting_pairs",
    # Orchestrator
    "MatryoshkaOrchestrator", "ShellType", "ShellConfig",
    "ShellResult", "LoopResult", "DEFAULT_SHELLS",
    "VERIFY_SCHEMA", "FLAG_SCHEMA", "SHELL_SCHEMAS",
    "parse_structured_response",
    # Cold Start
    "ColdStartHandler", "ColdStartConfig",
    # Training
    "TrainingPipeline", "AuditCollector", "AuditSummary",
    "ParameterBandit", "BanditArm",
    "PromptRefiner", "RefinementResult",
    "SignatureOptimizer", "Signature",
    "OptimizationResult", "TrainingExample",
    "RewardJudge",
    # Blackboard (Cognitive Architecture)
    "Blackboard", "WorkingMemory", "ProductionRule",
    "default_skip_verify_rules", "default_skip_flag_rules",
    "compute_ppr_entropy", "extract_seed_sources",
    # Activation (ACT-R)
    "ActivationAdapter", "ActivationConfig",
    # Contextual Bandits
    "ContextualBanditAdapter", "CONFIG_PRESETS",
    # KnowledgeSource (Classical AI Composition)
    "KnowledgeSource", "Phase", "SourceRegistry",
    # Truth Maintenance System
    "TruthMaintenanceSystem", "TMSConfig",
    "JustificationRecord", "Retraction",
    # Chunk Compiler (SOAR)
    "ChunkCompiler", "ChunkConfig", "ChunkAction",
    "CompiledChunk", "ExperienceRecord",
    # Case-Based Reasoning
    "CBRSystem", "CBRConfig", "Case",
    # Demons (Pattern Monitors)
    "DemonSystem", "DemonConfig", "DemonAlert",
    "check_grounding", "check_brevity",
    "check_repetition", "check_hedging",
    # Classical AI Factory
    "create_classical_sources",
]
