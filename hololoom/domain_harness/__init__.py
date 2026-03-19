"""
HoloLoom Domain Harness
=======================

Implements the Initializer/Worker pattern for persistent domain memory.

The key insight: The agent is not the model.
The agent is the transformation of one memory state into another.

Layers:
1. Domain Harness (practical): Features, tests, progress
2. Strands (symbolic): Tension, weave motions, narrative

Two Backends:
1. Flat files (features.json, state.json, progress.log) - works anywhere
2. Neo4j graph (Feature nodes, relationships) - for HoloLoom integration

CLI:
    python -m HoloLoom.domain_harness.run_worker init "Build a REST API"
    python -m HoloLoom.domain_harness.run_worker run
    python -m HoloLoom.domain_harness.run_worker status
    
    python -m HoloLoom.domain_harness.strands.visualizer --daemon

Usage (Flat Files):
    from hololoom.domain_harness import FlatFileGenerator, FlatFileBackend
    
    generator = FlatFileGenerator(Path("./domain_memory"))
    generator.generate(features, "MyProject")
    
    backend = FlatFileBackend(Path("./domain_memory"))
    features = backend.get_actionable_features()

Usage (with Strands):
    from hololoom.domain_harness import DomainWorker, ActionResult
    from hololoom.domain_harness.strands import StrandsUpdater, integrate_strands_with_worker
    
    # After worker run
    integrate_strands_with_worker(domain_dir, feature, result)
"""

# =============================================================================
# Schema
# =============================================================================
# =============================================================================
# Initializer
# =============================================================================
from hololoom.domain_harness.initializer import (
    EXAMPLE_PROMPT,
    DependencyResolver,
    DomainInitializer,
    FeatureParser,
    InitializationResult,
    TestScaffolder,
)

# =============================================================================
# Protocol
# =============================================================================
from hololoom.domain_harness.protocol import (
    CompletionChecker,
    DomainProtocol,
    Neo4jDomainProtocol,
    WorkerRunner,
)
from hololoom.domain_harness.schema import (
    ActionResult,
    DependsOn,
    DomainContext,
    DomainCypher,
    DomainState,
    Feature,
    FeatureStatus,
    ProgressAction,
    ProgressEntry,
    RecordsProgress,
    TestResult,
    create_feature,
    create_progress_entry,
)

# =============================================================================
# Templates (Flat Files)
# =============================================================================
from hololoom.domain_harness.templates import (
    FlatFileBackend,
    FlatFileGenerator,
    append_progress_entry,
    generate_conftest,
    generate_features_json,
    generate_progress_log_header,
    generate_state_json,
    generate_test_file,
    generate_worker_readme,
    parse_features_json,
    parse_state_json,
)

# =============================================================================
# Worker
# =============================================================================
from hololoom.domain_harness.worker import (
    AgentDomainWorker,
    DomainWorker,
    ImplementationResult,
    LLMImplementer,
    WorkerConfig,
    create_worker,
    run_worker_loop,
    stub_implementation,
    stub_test,
)

__all__ = [
    # Schema
    'Feature',
    'FeatureStatus',
    'ProgressEntry',
    'ProgressAction',
    'DomainState',
    'TestResult',
    'DependsOn',
    'RecordsProgress',
    'DomainContext',
    'ActionResult',
    'DomainCypher',
    'create_feature',
    'create_progress_entry',

    # Protocol
    'DomainProtocol',
    'Neo4jDomainProtocol',
    'CompletionChecker',
    'WorkerRunner',

    # Initializer
    'DomainInitializer',
    'InitializationResult',
    'FeatureParser',
    'DependencyResolver',
    'TestScaffolder',
    'EXAMPLE_PROMPT',

    # Worker
    'DomainWorker',
    'AgentDomainWorker',
    'WorkerConfig',
    'ImplementationResult',
    'run_worker_loop',
    'create_worker',
    'stub_implementation',
    'stub_test',
    'LLMImplementer',

    # Templates
    'generate_features_json',
    'generate_state_json',
    'generate_progress_log_header',
    'append_progress_entry',
    'generate_test_file',
    'generate_conftest',
    'generate_worker_readme',
    'parse_features_json',
    'parse_state_json',
    'FlatFileGenerator',
    'FlatFileBackend',
]
