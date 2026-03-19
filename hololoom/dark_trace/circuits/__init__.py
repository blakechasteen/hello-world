"""
Dark Trace Circuits: Mechanistic Circuit Discovery

This module provides infrastructure for tracing computational circuits
through model internals, enabling mechanistic interpretability.

Key Capabilities:
- Circuit tracing (input → features → output)
- Graph-based circuit representation
- Pattern detection and matching
- Vulnerability analysis (safety-critical paths)
- Circuit visualization

Research Basis:
- ACDC (Conmy et al., 2023): Automated Circuit Discovery (NeurIPS)
- Causal Tracing (Meng et al., 2022): Locating factual knowledge
- Activation Patching: Isolating causal mechanisms
- IOI Circuit (Wang et al., 2022): Indirect Object Identification

Usage:
    from hololoom.dark_trace.circuits import (
        CircuitTracer,
        TracerConfig,
        CircuitTrace,
        CircuitGraph,
        CircuitNode,
        CircuitEdge,
        CircuitPatternMatcher,
        KnownPattern,
        PatternMatch,
        VulnerabilityAnalyzer,
        VulnerabilityReport,
        SafetyCriticalPath,
        CircuitVisualizer,
        CircuitDiagram,
    )

    # Trace a circuit from input to output
    tracer = CircuitTracer(model, probe)
    trace = tracer.trace_forward(input_tokens, target_layer=11)
    print(f"Circuit has {len(trace.nodes)} nodes")

    # Build circuit graph
    graph = CircuitGraph.from_trace(trace)
    print(f"Critical path: {graph.get_critical_path()}")

    # Find known patterns
    matcher = CircuitPatternMatcher()
    matches = matcher.find_patterns(graph)
    for match in matches:
        print(f"Found pattern: {match.pattern.name}")

    # Analyze vulnerabilities
    analyzer = VulnerabilityAnalyzer(graph)
    report = analyzer.analyze()
    for vuln in report.vulnerabilities:
        print(f"Vulnerability: {vuln.description}")

    # Visualize circuit
    viz = CircuitVisualizer(graph)
    diagram = viz.create_diagram()
    diagram.save("circuit.html")
"""

# Circuit tracing
# Circuit graph representation
from hololoom.dark_trace.circuits.graph import (
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
    CircuitPath,
    EdgeType,
    NodeType,
)

# Pattern detection
from hololoom.dark_trace.circuits.patterns import (
    AnomalyDetector,
    CircuitPatternMatcher,
    KnownPattern,
    PatternLibrary,
    PatternMatch,
)
from hololoom.dark_trace.circuits.tracer import (
    CircuitTrace,
    CircuitTracer,
    TraceDirection,
    TraceEdge,
    TraceNode,
    TracerConfig,
)

# Visualization
from hololoom.dark_trace.circuits.visualizer import (
    CircuitDiagram,
    CircuitVisualizationConfig,
    CircuitVisualizer,
)

# Vulnerability analysis
from hololoom.dark_trace.circuits.vulnerability import (
    SafetyCriticalPath,
    SinglePointOfFailure,
    Vulnerability,
    VulnerabilityAnalyzer,
    VulnerabilityReport,
    VulnerabilityType,
)

__all__ = [
    # Tracer
    "CircuitTracer",
    "TracerConfig",
    "CircuitTrace",
    "TraceNode",
    "TraceEdge",
    "TraceDirection",
    # Graph
    "CircuitGraph",
    "CircuitNode",
    "CircuitEdge",
    "NodeType",
    "EdgeType",
    "CircuitPath",
    # Patterns
    "CircuitPatternMatcher",
    "KnownPattern",
    "PatternMatch",
    "PatternLibrary",
    "AnomalyDetector",
    # Vulnerability
    "VulnerabilityAnalyzer",
    "VulnerabilityReport",
    "Vulnerability",
    "VulnerabilityType",
    "SafetyCriticalPath",
    "SinglePointOfFailure",
    # Visualization
    "CircuitVisualizer",
    "CircuitDiagram",
    "CircuitVisualizationConfig",
]
