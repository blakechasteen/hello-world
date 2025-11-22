"""
HoloLoom Shuttle - MCTS-Powered Warp↔Yarn Intersection

A Monte Carlo Tree Search system for intelligently combining:
- Warp: Vector/semantic search (Qdrant)
- Yarn: Knowledge graph traversal (Neo4j)

Uses Thompson Sampling to learn which traversal trajectories work best.

NOTE: Renamed "policies" to "trajectories" to avoid collision with HoloLoom's
existing policy system (tool selection policies in policy/unified.py).

Quick Start:
    from HoloLoom.shuttle import create_hololoom_shuttle

    shuttle = create_hololoom_shuttle()
    result = shuttle.intersect("What's blocking us?")

    print(result.fuzzy_evidence)        # Warp search results
    print(result.structural_claims)     # Yarn graph context
    print(result.trajectory_used)       # Which trajectory was selected

Components:
    - Trajectories: Different graph traversal strategies
    - Trajectory Bandit: Thompson Sampling for trajectory selection
    - MCTS: Monte Carlo Tree Search for optimal graph expansion
    - Orchestrator: Main coordinator
    - Config: Configuration system with graceful degradation
    - Entity Extraction: Three-tier extraction (spaCy/Regex/Payload)
    - Exceptions: Comprehensive error handling

Version: 2.0 (Production Ready - with Pre-Integration Moonshot Complete)
"""

# Configuration and error handling
from .config import (
    ShuttleConfig,
    ShuttleMode,
    validate_config,
)

from .exceptions import (
    ShuttleError,
    ConfigurationError,
    BackendUnavailableError,
    TimeoutError,
    EntityExtractionError,
    WarpSearchError,
    YarnTraversalError,
)

# Entity extraction
from .entity_extraction import (
    Anchor,
    EntityExtractor,
    PayloadExtractor,
    RegexExtractor,
    SpacyExtractor,
    EntityExtractionFactory,
)

# Trajectory strategies (renamed from "policies")
from .trajectories import (
    TraversalConfig,
    TrajectoryStrategy,
    ProjectBlockersTrajectory,
    OwnershipTrajectory,
    TimelineTrajectory,
    ConceptualTrajectory,
    HierarchicalTrajectory,
    ExploratoryTrajectory,
    ALL_TRAJECTORIES,
    TRAJECTORY_BY_NAME,
)

# Trajectory bandit (renamed from "PolicyBandit")
from .trajectory_bandit import (
    TrajectoryStats,
    TrajectoryBandit,
    TrajectorySelector,
    RewardCalculator,
)

# MCTS
from .mcts import (
    MCTSState,
    MCTSNode,
    MCTS,
    MCTSResult,
    NeighborMap,
    run_mcts_search,
)

# Orchestrator (v2 with error handling)
from .orchestrator_v2 import (
    WarpInterface,
    YarnInterface,
    WeaveResult,
    Shuttle,
)

# HoloLoom-specific adapters
from .hololoom_adapters import (
    HoloLoomWarp,
    HoloLoomYarn,
    create_hololoom_warp,
    create_hololoom_yarn,
    create_hololoom_shuttle,
)

# Weaving Orchestrator Integration (Step 3 replacement)
from .weaving_integration import (
    HoloLoomWarpAdapter,
    HoloLoomYarnAdapter,
    ShuttleStage,
    create_shuttle_stage,
)

__all__ = [
    # Configuration
    'ShuttleConfig',
    'ShuttleMode',
    'validate_config',

    # Exceptions
    'ShuttleError',
    'ConfigurationError',
    'BackendUnavailableError',
    'TimeoutError',
    'EntityExtractionError',
    'WarpSearchError',
    'YarnTraversalError',

    # Entity Extraction
    'Anchor',
    'EntityExtractor',
    'PayloadExtractor',
    'RegexExtractor',
    'SpacyExtractor',
    'EntityExtractionFactory',

    # Trajectories (formerly "Policies")
    'TraversalConfig',
    'TrajectoryStrategy',
    'ProjectBlockersTrajectory',
    'OwnershipTrajectory',
    'TimelineTrajectory',
    'ConceptualTrajectory',
    'HierarchicalTrajectory',
    'ExploratoryTrajectory',
    'ALL_TRAJECTORIES',
    'TRAJECTORY_BY_NAME',

    # Trajectory Bandit (formerly "PolicyBandit")
    'TrajectoryStats',
    'TrajectoryBandit',
    'TrajectorySelector',
    'RewardCalculator',

    # MCTS
    'MCTSState',
    'MCTSNode',
    'MCTS',
    'MCTSResult',
    'NeighborMap',
    'run_mcts_search',

    # Orchestrator
    'WarpInterface',
    'YarnInterface',
    'WeaveResult',
    'Shuttle',

    # HoloLoom Adapters
    'HoloLoomWarp',
    'HoloLoomYarn',
    'create_hololoom_warp',
    'create_hololoom_yarn',
    'create_hololoom_shuttle',

    # Weaving Integration
    'HoloLoomWarpAdapter',
    'HoloLoomYarnAdapter',
    'ShuttleStage',
    'create_shuttle_stage',
]

__version__ = '2.0.0'
__author__ = 'Claude + Blake'
__status__ = 'Production'
