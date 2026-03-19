"""
HoloLoom Domain Harness - Strands Layer
=======================================

Symbolic weaving layer for the domain harness.

Transforms practical worker actions into symbolic weave motions:
- Tension tracking per feature
- Weave motion decoding
- Loom state visualization
- Git auto-commit integration

The Strands layer gives the harness a story.

Components:
- tension_model: Computes tension from feature state
- weave_decoder: Maps actions to symbolic motions
- visualizer: ASCII/Rich/Matplotlib visualization
- update_strands: Integration with domain harness
- git_autocommit: Automated git commits

Usage:
    from hololoom.domain_harness.strands import (
        StrandsUpdater,
        LoomState,
        WeaveEvent,
        integrate_strands_with_worker
    )
    
    # After worker run
    updater = StrandsUpdater(Path("./domain_memory"))
    event = updater.update_after_worker(feature, action_result)
    
    # Visualize
    from hololoom.domain_harness.strands.visualizer import ASCIIRenderer
    renderer = ASCIIRenderer()
    print(renderer.render_loom(features, loom_state))
"""

# Core tension model
# Git
from hololoom.domain_harness.strands.git_autocommit import (
    GitAutoCommit,
    git_commit_all,
    git_push,
)
from hololoom.domain_harness.strands.tension_model import (
    DEPENDENCY_BLOCKED_MODIFIER,
    REGRESSION_MODIFIER,
    STALE_MODIFIER,
    STATUS_TENSION,
    LoomState,
    StrandState,
    TensionModel,
    TensionUpdater,
)

# Integration
from hololoom.domain_harness.strands.update_strands import (
    StrandsUpdater,
    integrate_strands_with_worker,
)

# Weave decoder
from hololoom.domain_harness.strands.weave_decoder import (
    MOTION_EFFECTS,
    WeaveDecoder,
    WeaveEvent,
    WeaveHistory,
    WeaveMotion,
    WeaveNarrative,
)

__all__ = [
    # Tension Model
    'StrandState',
    'LoomState',
    'TensionModel',
    'TensionUpdater',
    'STATUS_TENSION',
    'DEPENDENCY_BLOCKED_MODIFIER',
    'REGRESSION_MODIFIER',
    'STALE_MODIFIER',

    # Weave Decoder
    'WeaveMotion',
    'WeaveEvent',
    'WeaveDecoder',
    'WeaveHistory',
    'WeaveNarrative',
    'MOTION_EFFECTS',

    # Update Integration
    'StrandsUpdater',
    'integrate_strands_with_worker',

    # Git
    'GitAutoCommit',
    'git_commit_all',
    'git_push',
]
