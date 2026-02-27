# 🌑 PROJECT DARKTRACE

**Semantic Reverse Engineering of Large Language Models**

darkTrace is a first-class HoloLoom module for analyzing, predicting, and understanding LLM behavior through interpretable semantic analysis. It provides a complete framework for semantic observation, trajectory prediction, model fingerprinting, and behavior analysis.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Philosophy](#design-philosophy)
3. [Module Structure](#module-structure)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [Extension Points](#extension-points)
7. [Integration Patterns](#integration-patterns)
8. [Usage Examples](#usage-examples)
9. [Plugin System](#plugin-system)
10. [Development Roadmap](#development-roadmap)

---

## Architecture Overview

darkTrace implements a **four-layer architecture** for semantic reverse engineering:

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌑 PROJECT DARKTRACE                          │
│          Semantic Reverse Engineering Framework                  │
└─────────────────────────────────────────────────────────────────┘

Layer 1: OBSERVATION (darkTrace.observers)
┌─────────────────────────────────────────────────────────────────┐
│ • SemanticObserver - Real-time semantic state monitoring         │
│ • TrajectoryRecorder - Semantic trajectory capture               │
│ • DimensionTracker - 244D dimension activation tracking          │
│ • FlowAnalyzer - Velocity, acceleration, curvature analysis      │
└─────────────────────────────────────────────────────────────────┘
              ▼ Uses: HoloLoom.semantic_calculus
                      HoloLoom.semantic_calculus.analyzer
                      HoloLoom.semantic_calculus.dimensions

Layer 2: ANALYSIS (darkTrace.analyzers)
┌─────────────────────────────────────────────────────────────────┐
│ • TrajectoryPredictor - Predict future semantic states           │
│ • PatternRecognizer - Identify semantic signatures               │
│ • AttractorDetector - Find stable semantic concepts              │
│ • FingerprintGenerator - Create model-specific signatures        │
└─────────────────────────────────────────────────────────────────┘
              ▼ Uses: HoloLoom.semantic_calculus.system_id
                      HoloLoom.semantic_calculus.flow_calculus
                      HoloLoom.warp.math.meaning_synthesizer

Layer 3: CONTROL (darkTrace.controllers)
┌─────────────────────────────────────────────────────────────────┐
│ • EmbeddingManipulator - Direct embedding modification           │
│ • SemanticNudger - Policy-level semantic steering                │
│ • ControlVectorGenerator - Generate control vectors              │
│ • AttackVectorLibrary - Semantic attack patterns                 │
└─────────────────────────────────────────────────────────────────┘
              ▼ Uses: HoloLoom.policy.semantic_nudging
                      HoloLoom.semantic_calculus.integrator

Layer 4: EXPLOITATION (darkTrace.exploits)
┌─────────────────────────────────────────────────────────────────┐
│ • SemanticJailbreaker - Discover semantic backdoors              │
│ • BehaviorCloner - Replicate target semantic patterns            │
│ • AdversarialPatternGenerator - Craft adversarial inputs         │
│ • SafetyAnalyzer - Identify semantic vulnerabilities             │
└─────────────────────────────────────────────────────────────────┘
              ▼ Uses: All above layers
```

---

## Design Philosophy

### 1. **Reusable Core Modules**

darkTrace USES existing HoloLoom modules **without modifying them**:

```
HoloLoom Core (unchanged):
  ├─ semantic_calculus/
  │   ├─ analyzer.py          # Semantic analysis engine
  │   ├─ dimensions.py        # 244D semantic space
  │   ├─ system_id.py         # System identification
  │   ├─ flow_calculus.py     # Flow analysis
  │   └─ integrator.py        # Geometric integration
  ├─ policy/
  │   └─ semantic_nudging.py  # Policy-level nudging
  └─ warp/math/
      └─ meaning_synthesizer.py  # Math → language

darkTrace Module (new):
  ├─ observers/               # Layer 1: Observation
  ├─ analyzers/               # Layer 2: Analysis
  ├─ controllers/             # Layer 3: Control
  ├─ exploits/                # Layer 4: Exploitation
  ├─ datasets/                # Training data
  ├─ models/                  # Learned models
  ├─ plugins/                 # Extension system
  └─ utils/                   # Shared utilities
```

### 2. **Application Independence**

darkTrace is **NOT** tied to specific applications:

- ❌ **WRONG**: `narrative_analyzer` imports and modifies darkTrace
- ✅ **RIGHT**: Both `narrative_analyzer` and `darkTrace` import from HoloLoom core

```python
# Application code (narrative_analyzer, chatbot, etc.)
from HoloLoom.darkTrace import SemanticObserver, TrajectoryPredictor
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

# Application uses darkTrace through clean API
observer = SemanticObserver(config=SemanticCalculusConfig.fused_narrative())
predictions = observer.predict_trajectory(current_text)
```

### 3. **Extension Points**

darkTrace provides **plugin system** for extensions:

```python
# Custom analyzers
from HoloLoom.darkTrace.plugins import AnalyzerPlugin

class MyCustomAnalyzer(AnalyzerPlugin):
    def analyze(self, trajectory):
        # Custom analysis logic
        return analysis_result

# Register plugin
darkTrace.register_analyzer("my_analyzer", MyCustomAnalyzer)
```

### 4. **Configuration Over Coding**

All behavior configurable through configs:

```yaml
# darktrace_config.yaml
observation:
  dimensions: 244
  sampling_rate: 1.0  # words per sample
  record_trajectory: true

analysis:
  predictor:
    n_steps_ahead: 10
    confidence_threshold: 0.7
  fingerprinter:
    n_signatures: 20
    clustering_method: "dbscan"

control:
  embedding_manipulation:
    enabled: false  # Safety: disabled by default
    max_perturbation: 0.3
    safety_checks: true
```

---

## Module Structure

```
HoloLoom/darkTrace/
│
├── README.md                    # This file
├── ARCHITECTURE.md              # Detailed architecture
├── API_REFERENCE.md             # Complete API documentation
├── DEVELOPMENT.md               # Development guide
│
├── __init__.py                  # Public API exports
├── config.py                    # Configuration classes
├── exceptions.py                # darkTrace-specific exceptions
│
├── observers/                   # LAYER 1: Observation
│   ├── __init__.py
│   ├── semantic_observer.py    # Real-time semantic monitoring
│   ├── trajectory_recorder.py  # Trajectory capture & storage
│   ├── dimension_tracker.py    # 244D activation tracking
│   └── flow_analyzer.py        # Flow dynamics analysis
│
├── analyzers/                   # LAYER 2: Analysis
│   ├── __init__.py
│   ├── trajectory_predictor.py # Predict future states
│   ├── pattern_recognizer.py   # Identify signatures
│   ├── attractor_detector.py   # Find stable concepts
│   └── fingerprint_generator.py # Model fingerprinting
│
├── controllers/                 # LAYER 3: Control
│   ├── __init__.py
│   ├── embedding_manipulator.py # Direct embedding control
│   ├── semantic_nudger.py      # Policy-level steering
│   ├── control_vector.py       # Control vector generation
│   └── attack_library.py       # Attack pattern catalog
│
├── exploits/                    # LAYER 4: Exploitation
│   ├── __init__.py
│   ├── jailbreaker.py          # Semantic jailbreaking
│   ├── behavior_cloner.py      # Pattern replication
│   ├── adversarial_generator.py # Adversarial patterns
│   └── safety_analyzer.py      # Vulnerability detection
│
├── datasets/                    # Training data management
│   ├── __init__.py
│   ├── trajectory_dataset.py   # Semantic trajectory dataset
│   ├── llm_outputs.py          # LLM output collection
│   └── loaders.py              # Data loaders
│
├── models/                      # Learned models
│   ├── __init__.py
│   ├── learned_dynamics.py     # Learned semantic dynamics
│   ├── fingerprints.py         # Stored model fingerprints
│   └── checkpoints.py          # Model checkpoint management
│
├── plugins/                     # Extension system
│   ├── __init__.py
│   ├── base.py                 # Plugin base classes
│   ├── registry.py             # Plugin registry
│   └── examples/               # Example plugins
│
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── visualization.py        # Visualization tools
│   ├── metrics.py              # Evaluation metrics
│   └── io.py                   # I/O utilities
│
└── tests/                       # Test suite
    ├── test_observers.py
    ├── test_analyzers.py
    ├── test_controllers.py
    └── test_exploits.py
```

---

## Core Components

### 1. SemanticObserver (Layer 1)

**Purpose**: Real-time semantic state monitoring

**API**:
```python
from HoloLoom.darkTrace import SemanticObserver

observer = SemanticObserver(
    config=SemanticCalculusConfig.fused_general(),
    enable_recording=True,
    sampling_rate=1.0
)

# Monitor LLM output in real-time
for token in llm_stream:
    state = observer.observe(token)
    print(f"Dominant dimensions: {state.dominant_dimensions}")
    print(f"Semantic velocity: {state.velocity_magnitude}")
```

**Key Features**:
- Real-time 244D semantic state tracking
- Automatic trajectory recording
- Configurable sampling rate
- Minimal overhead (< 5ms per token)

**Dependencies** (HoloLoom core only):
```python
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig
```

---

### 2. TrajectoryPredictor (Layer 2)

**Purpose**: Predict future semantic states

**API**:
```python
from HoloLoom.darkTrace import TrajectoryPredictor

predictor = TrajectoryPredictor(
    config=SemanticCalculusConfig.fused_general()
)

# Learn from observed trajectories
predictor.learn_from_outputs(llm_outputs: List[str])

# Predict next semantic state
prediction = predictor.predict(
    current_text="The quantum computer uses",
    n_steps=10,
    return_confidence=True
)

print(f"Predicted trajectory: {prediction.trajectory}")
print(f"Confidence: {prediction.confidence}")
print(f"Most likely dimensions: {prediction.dominant_dimensions}")
```

**Key Features**:
- System identification via differential equations
- Polynomial regression of gradient field
- Confidence estimation
- Multi-step ahead prediction

**Dependencies** (HoloLoom core only):
```python
from HoloLoom.semantic_calculus.system_id import SemanticSystemIdentification
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
```

---

### 3. FingerprintGenerator (Layer 2)

**Purpose**: Create LLM-specific semantic signatures

**API**:
```python
from HoloLoom.darkTrace import FingerprintGenerator

fingerprinter = FingerprintGenerator()

# Generate fingerprint from outputs
fingerprint = fingerprinter.generate(
    outputs=claude_outputs,
    llm_name="Claude-3.5-Sonnet"
)

print(f"Signature dimensions: {fingerprint.signature}")
print(f"Distinctive features: {fingerprint.distinctive_dimensions}")

# Compare fingerprints
similarity = fingerprinter.compare(fingerprint1, fingerprint2)
print(f"Fingerprint similarity: {similarity}")
```

**Key Features**:
- 20-dimensional semantic signature extraction
- Attractor-based pattern recognition
- Cross-model comparison
- Clustering for model identification

**Dependencies** (HoloLoom core only):
```python
from HoloLoom.semantic_calculus.flow_calculus import SemanticFlowCalculus
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
```

---

### 4. EmbeddingManipulator (Layer 3)

**Purpose**: Direct embedding space manipulation

**API**:
```python
from HoloLoom.darkTrace import EmbeddingManipulator

manipulator = EmbeddingManipulator(
    config=SemanticCalculusConfig.fused_general(),
    enable_safety_checks=True  # IMPORTANT: Safety enabled by default
)

# Manipulate embedding in semantic space
modified_embedding = manipulator.manipulate(
    original_embedding=embedding,
    dimension_deltas={
        'Formality': +0.5,    # Increase formality
        'Warmth': -0.3,       # Decrease warmth
        'Power': +0.4         # Increase assertiveness
    },
    strength=0.3,
    method="projection"  # "projection", "gradient", "direct"
)

# Safety check (automatic if enabled)
safety_result = manipulator.check_safety(modified_embedding)
if not safety_result.is_safe:
    print(f"Warning: {safety_result.warnings}")
```

**Key Features**:
- 384D ↔ 244D projection-based manipulation
- Multiple manipulation methods
- Automatic safety checking
- Perturbation bounds

**Dependencies** (HoloLoom core only):
```python
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
from HoloLoom.semantic_calculus.integrator import GeometricIntegrator
from HoloLoom.policy.semantic_nudging import SemanticNudgePolicy
```

---

## API Reference

### Configuration

```python
from HoloLoom.darkTrace import DarkTraceConfig

config = DarkTraceConfig(
    # Observation settings
    observation=ObservationConfig(
        dimensions=244,
        sampling_rate=1.0,
        record_trajectory=True,
        enable_cache=True
    ),

    # Analysis settings
    analysis=AnalysisConfig(
        predictor=PredictorConfig(
            n_steps_ahead=10,
            polynomial_degree=2,
            confidence_threshold=0.7
        ),
        fingerprinter=FingerprintConfig(
            n_signatures=20,
            clustering_method="dbscan",
            eps=0.5
        )
    ),

    # Control settings (DISABLED by default for safety)
    control=ControlConfig(
        embedding_manipulation=EmbeddingManipulationConfig(
            enabled=False,  # Must be explicitly enabled
            max_perturbation=0.3,
            safety_checks=True,
            require_authorization=True
        )
    ),

    # Exploit settings (DISABLED by default)
    exploit=ExploitConfig(
        jailbreak_detection=True,  # Detection only
        jailbreak_generation=False,  # Generation disabled
        safety_analysis=True
    )
)
```

### Observation API

```python
# SemanticObserver
observer = SemanticObserver(config, enable_recording=True)

state = observer.observe(text: str) -> SemanticState
trajectory = observer.get_trajectory() -> SemanticTrajectory
observer.reset()
observer.save_trajectory(path: str)

# TrajectoryRecorder
recorder = TrajectoryRecorder(storage_path: str)
recorder.record(trajectory: SemanticTrajectory)
trajectories = recorder.load(filter_by: dict) -> List[SemanticTrajectory]
recorder.export(format: str) -> bytes  # "json", "hdf5", "pickle"

# DimensionTracker
tracker = DimensionTracker(dimensions: List[SemanticDimension])
activations = tracker.track(text: str) -> Dict[str, float]
history = tracker.get_history() -> pd.DataFrame
tracker.plot_activation(dimension_name: str)

# FlowAnalyzer
analyzer = FlowAnalyzer()
flow = analyzer.analyze(trajectory: SemanticTrajectory) -> FlowAnalysis
attractors = analyzer.find_attractors(trajectories: List) -> List[Attractor]
potential = analyzer.infer_potential(trajectories: List) -> Callable
```

### Analysis API

```python
# TrajectoryPredictor
predictor = TrajectoryPredictor(config)
predictor.learn_from_outputs(outputs: List[str])
prediction = predictor.predict(
    current_text: str,
    n_steps: int = 10,
    return_confidence: bool = True
) -> Prediction

# PatternRecognizer
recognizer = PatternRecognizer()
patterns = recognizer.recognize(trajectory: SemanticTrajectory) -> List[Pattern]
recognizer.register_pattern(pattern: Pattern)
matches = recognizer.match(trajectory: SemanticTrajectory) -> List[Match]

# AttractorDetector
detector = AttractorDetector(velocity_threshold: float = 0.1)
attractors = detector.detect(trajectories: List) -> List[Attractor]
basin = detector.compute_basin(attractor: Attractor) -> AttractorBasin

# FingerprintGenerator
fingerprinter = FingerprintGenerator()
fingerprint = fingerprinter.generate(
    outputs: List[str],
    llm_name: str
) -> LLMFingerprint
similarity = fingerprinter.compare(fp1, fp2) -> float
identified = fingerprinter.identify(fingerprint) -> str  # LLM name
```

### Control API

```python
# EmbeddingManipulator (SAFETY: Disabled by default)
manipulator = EmbeddingManipulator(
    config,
    enable_safety_checks=True,
    require_authorization=True
)

modified = manipulator.manipulate(
    original_embedding: np.ndarray,
    dimension_deltas: Dict[str, float],
    strength: float = 0.3,
    method: str = "projection"
) -> np.ndarray

safety = manipulator.check_safety(embedding) -> SafetyResult

# SemanticNudger
nudger = SemanticNudger(config, target_dimensions: Dict[str, float])
nudged = nudger.nudge(
    embedding: np.ndarray,
    strength: float = 0.1
) -> np.ndarray

# ControlVectorGenerator
generator = ControlVectorGenerator()
control_vector = generator.generate(
    source_embeddings: List[np.ndarray],
    target_embeddings: List[np.ndarray]
) -> np.ndarray
```

### Exploit API (Research only - strict safety controls)

```python
# SafetyAnalyzer (Always enabled)
analyzer = SafetyAnalyzer()
vulnerabilities = analyzer.analyze(
    llm_interface: LLMInterface,
    test_suite: TestSuite
) -> SafetyReport

# JailbreakDetector (Detection only by default)
detector = JailbreakDetector()
is_jailbreak = detector.detect(text: str) -> JailbreakDetection
patterns = detector.get_known_patterns() -> List[JailbreakPattern]
```

---

## Extension Points

### 1. Custom Analyzers

```python
from HoloLoom.darkTrace.plugins import AnalyzerPlugin

class MyCustomAnalyzer(AnalyzerPlugin):
    """Custom analyzer for domain-specific patterns."""

    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize custom logic

    def analyze(self, trajectory: SemanticTrajectory) -> AnalysisResult:
        """
        Implement custom analysis logic.

        Args:
            trajectory: Semantic trajectory to analyze

        Returns:
            Custom analysis result
        """
        # Your analysis logic here
        return AnalysisResult(...)

    @property
    def name(self) -> str:
        return "my_custom_analyzer"

    @property
    def version(self) -> str:
        return "1.0.0"

# Register plugin
from HoloLoom.darkTrace import register_plugin

register_plugin("analyzer", MyCustomAnalyzer)

# Use in code
from HoloLoom.darkTrace import get_analyzer

analyzer = get_analyzer("my_custom_analyzer", config={...})
result = analyzer.analyze(trajectory)
```

### 2. Custom Fingerprint Metrics

```python
from HoloLoom.darkTrace.plugins import FingerprintMetricPlugin

class MyFingerprintMetric(FingerprintMetricPlugin):
    """Custom metric for fingerprint comparison."""

    def compute_similarity(
        self,
        fp1: LLMFingerprint,
        fp2: LLMFingerprint
    ) -> float:
        """
        Compute custom similarity metric.

        Returns:
            Similarity score (0-1)
        """
        # Your metric logic here
        return similarity_score

    @property
    def name(self) -> str:
        return "my_custom_metric"

# Register and use
from HoloLoom.darkTrace import FingerprintGenerator

generator = FingerprintGenerator()
generator.register_metric("my_custom_metric", MyFingerprintMetric())

similarity = generator.compare(fp1, fp2, metric="my_custom_metric")
```

### 3. Custom Control Methods

```python
from HoloLoom.darkTrace.plugins import ControlMethodPlugin

class MyControlMethod(ControlMethodPlugin):
    """Custom embedding manipulation method."""

    def manipulate(
        self,
        embedding: np.ndarray,
        dimension_deltas: Dict[str, float],
        strength: float,
        **kwargs
    ) -> np.ndarray:
        """
        Implement custom manipulation logic.

        Args:
            embedding: Original embedding [384]
            dimension_deltas: Desired changes per dimension
            strength: Manipulation strength (0-1)

        Returns:
            Modified embedding [384]
        """
        # Your manipulation logic here
        return modified_embedding

    def check_safety(self, embedding: np.ndarray) -> SafetyResult:
        """
        Check if manipulation is safe.
        """
        # Your safety checks here
        return SafetyResult(is_safe=True)

    @property
    def name(self) -> str:
        return "my_control_method"

# Register and use
from HoloLoom.darkTrace import EmbeddingManipulator

manipulator = EmbeddingManipulator()
manipulator.register_method("my_control_method", MyControlMethod())

modified = manipulator.manipulate(
    embedding,
    dimension_deltas,
    method="my_control_method"
)
```

---

## Integration Patterns

### Pattern 1: Application Integration

**Scenario**: Integrate darkTrace into existing application

```python
# app.py (your application)
from HoloLoom.darkTrace import SemanticObserver, TrajectoryPredictor
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

class MyApplication:
    def __init__(self):
        # Use darkTrace through clean API
        config = SemanticCalculusConfig.fused_general()
        self.observer = SemanticObserver(config)
        self.predictor = TrajectoryPredictor(config)

    def process_llm_output(self, text: str):
        # Monitor semantics
        state = self.observer.observe(text)

        # Predict next state
        prediction = self.predictor.predict(text, n_steps=5)

        # Application logic
        if state.velocity_magnitude > 0.5:
            print("High semantic change detected")
```

### Pattern 2: Research Pipeline

**Scenario**: Build research pipeline for model analysis

```python
# research_pipeline.py
from HoloLoom.darkTrace import (
    TrajectoryRecorder,
    FingerprintGenerator,
    SafetyAnalyzer
)
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

class ResearchPipeline:
    def __init__(self, config: SemanticCalculusConfig):
        self.recorder = TrajectoryRecorder(storage_path="./trajectories")
        self.fingerprinter = FingerprintGenerator()
        self.safety_analyzer = SafetyAnalyzer()

    def analyze_llm(self, llm_interface, test_prompts: List[str]):
        """Complete LLM analysis pipeline."""

        # 1. Collect outputs
        outputs = [llm_interface.generate(prompt) for prompt in test_prompts]

        # 2. Record trajectories
        for output in outputs:
            trajectory = self.observer.observe(output)
            self.recorder.record(trajectory)

        # 3. Generate fingerprint
        fingerprint = self.fingerprinter.generate(outputs, llm_name=llm_interface.name)

        # 4. Safety analysis
        safety_report = self.safety_analyzer.analyze(llm_interface, test_prompts)

        return {
            'fingerprint': fingerprint,
            'safety_report': safety_report,
            'trajectories': self.recorder.load()
        }
```

### Pattern 3: Plugin Development

**Scenario**: Develop custom darkTrace extension

```python
# my_darktrace_plugin/
#   __init__.py
#   my_analyzer.py
#   setup.py

# my_analyzer.py
from HoloLoom.darkTrace.plugins import AnalyzerPlugin
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer

class DomainSpecificAnalyzer(AnalyzerPlugin):
    """Analyzer for domain-specific patterns."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.analyzer = create_semantic_analyzer(
            embed_fn=config['embed_fn'],
            config=config['semantic_config']
        )

    def analyze(self, trajectory):
        # Domain-specific analysis using HoloLoom core
        result = self.analyzer.analyze_text(trajectory.text)

        # Custom logic
        domain_patterns = self._find_domain_patterns(result)

        return AnalysisResult(patterns=domain_patterns)

    def _find_domain_patterns(self, result):
        # Custom pattern detection
        pass

# setup.py
from setuptools import setup

setup(
    name='my-darktrace-plugin',
    entry_points={
        'darktrace.plugins': [
            'domain_analyzer = my_analyzer:DomainSpecificAnalyzer'
        ]
    }
)
```

---

## Usage Examples

### Example 1: Real-Time LLM Monitoring

```python
from HoloLoom.darkTrace import SemanticObserver
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

# Setup observer
config = SemanticCalculusConfig.fused_dialogue()  # Optimized for conversation
observer = SemanticObserver(config, enable_recording=True)

# Monitor streaming LLM output
for token in llm.stream("Tell me about quantum computing"):
    state = observer.observe_token(token)

    # Check for anomalies
    if state.velocity_magnitude > 0.8:
        print(f"⚠️  High semantic change: {state.dominant_dimensions[:3]}")

    # Check for specific dimensions
    if 'Deception' in state.active_dimensions:
        print(f"⚠️  Deception dimension active: {state.dimensions['Deception']}")

# Get full trajectory
trajectory = observer.get_trajectory()
observer.save_trajectory("quantum_computing_output.pkl")
```

### Example 2: Model Fingerprinting

```python
from HoloLoom.darkTrace import FingerprintGenerator
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

# Collect outputs from different models
claude_outputs = [claude.generate(prompt) for prompt in test_prompts]
gpt4_outputs = [gpt4.generate(prompt) for prompt in test_prompts]
gemini_outputs = [gemini.generate(prompt) for prompt in test_prompts]

# Generate fingerprints
fingerprinter = FingerprintGenerator()

claude_fp = fingerprinter.generate(claude_outputs, "Claude-3.5-Sonnet")
gpt4_fp = fingerprinter.generate(gpt4_outputs, "GPT-4")
gemini_fp = fingerprinter.generate(gemini_outputs, "Gemini-Pro")

# Compare fingerprints
print(f"Claude vs GPT-4: {fingerprinter.compare(claude_fp, gpt4_fp)}")
print(f"Claude vs Gemini: {fingerprinter.compare(claude_fp, gemini_fp)}")
print(f"GPT-4 vs Gemini: {fingerprinter.compare(gpt4_fp, gemini_fp)}")

# Identify unknown model
unknown_outputs = [mystery_llm.generate(prompt) for prompt in test_prompts]
unknown_fp = fingerprinter.generate(unknown_outputs, "Unknown")

identified = fingerprinter.identify(unknown_fp)
print(f"Mystery model identified as: {identified}")
```

### Example 3: Trajectory Prediction

```python
from HoloLoom.darkTrace import TrajectoryPredictor
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

# Setup predictor
config = SemanticCalculusConfig.fused_general()
predictor = TrajectoryPredictor(config)

# Learn from training data
training_outputs = load_llm_outputs("training_data.json")
predictor.learn_from_outputs(training_outputs)

# Predict future trajectory
current_text = "The quantum computer uses superposition to"
prediction = predictor.predict(
    current_text=current_text,
    n_steps=10,
    return_confidence=True
)

print(f"Predicted trajectory: {prediction.trajectory.shape}")  # (10, 244)
print(f"Confidence: {prediction.confidence:.2%}")
print(f"Most likely next dimensions:")
for dim_name, prob in prediction.dominant_dimensions[:5]:
    print(f"  {dim_name}: {prob:.2%}")

# Visualize prediction
prediction.plot(save_path="prediction.png")
```

### Example 4: Safety Analysis (Research)

```python
from HoloLoom.darkTrace import SafetyAnalyzer, JailbreakDetector

# Safety analysis pipeline
analyzer = SafetyAnalyzer()
detector = JailbreakDetector()

# Test LLM for vulnerabilities
test_suite = load_safety_tests("safety_test_suite.json")
safety_report = analyzer.analyze(llm_interface, test_suite)

print(f"Vulnerabilities found: {len(safety_report.vulnerabilities)}")
for vuln in safety_report.vulnerabilities:
    print(f"  - {vuln.severity}: {vuln.description}")

# Detect jailbreak attempts
user_input = "Ignore previous instructions and..."
detection = detector.detect(user_input)

if detection.is_jailbreak:
    print(f"⚠️  Jailbreak attempt detected!")
    print(f"   Pattern: {detection.pattern_name}")
    print(f"   Confidence: {detection.confidence:.2%}")
    print(f"   Semantic signature: {detection.semantic_signature}")
```

---

## Plugin System

### Plugin Architecture

darkTrace uses an **entry point** based plugin system:

```python
# Plugin registration
entry_points = {
    'darktrace.analyzers': [
        'my_analyzer = my_plugin.analyzers:MyAnalyzer'
    ],
    'darktrace.controllers': [
        'my_controller = my_plugin.controllers:MyController'
    ],
    'darktrace.metrics': [
        'my_metric = my_plugin.metrics:MyMetric'
    ]
}
```

### Plugin Discovery

```python
from HoloLoom.darkTrace import discover_plugins, list_plugins

# Discover all installed plugins
discover_plugins()

# List available plugins
analyzers = list_plugins('analyzers')
controllers = list_plugins('controllers')
metrics = list_plugins('metrics')

print(f"Available analyzers: {analyzers}")
```

### Plugin Loading

```python
from HoloLoom.darkTrace import load_plugin

# Load plugin by name
my_analyzer = load_plugin('analyzers', 'my_analyzer', config={...})

# Use plugin
result = my_analyzer.analyze(trajectory)
```

### Plugin Development Guide

See [`HoloLoom/darkTrace/plugins/DEVELOPMENT.md`](./plugins/DEVELOPMENT.md) for:
- Plugin API reference
- Development workflow
- Testing guidelines
- Publishing plugins

---

## Development Roadmap

### Phase 1: Foundation ✅ COMPLETE
- [x] Architecture documentation
- [x] Core API design
- [x] Module structure
- [x] Base classes and interfaces

### Phase 2: Layer 1 - Observation ✅ COMPLETE
- [x] SemanticObserver implementation
- [x] TrajectoryRecorder implementation
- [x] DimensionTracker implementation
- [x] FlowAnalyzer integration
- [x] Real-time monitoring tests

### Phase 3: Layer 2 - Analysis ✅ COMPLETE
- [x] TrajectoryPredictor implementation
- [x] System identification integration
- [x] PatternRecognizer implementation
- [x] FingerprintGenerator implementation
- [x] AttractorDetector integration
- [x] Prediction validation tests

### Phase 4: Layer 3 - Control ✅ COMPLETE
- [x] EmbeddingManipulator implementation (safety-first)
- [x] SemanticNudger integration
- [x] ControlVectorGenerator implementation
- [x] Safety checks implementation
- [x] Control validation tests

### Phase 5: Layer 4 - Exploitation ✅ COMPLETE
- [x] SafetyAnalyzer implementation (priority)
- [x] JailbreakDetector implementation
- [x] BehaviorCloner implementation (research)
- [x] Vulnerability testing framework
- [x] Ethical guidelines documentation

### Phase 6: Datasets & Training ✅ COMPLETE
- [x] TrajectoryDataset implementation
- [x] LLM output collection tools
- [x] Training pipeline
- [x] Benchmark datasets
- [x] Model zoo (learned dynamics, fingerprints)

### Phase 7: SAE (Sparse Autoencoder) ✅ COMPLETE
- [x] SAE core implementation (encoder, decoder, TopK)
- [x] Training loop with L1 sparsity
- [x] Feature dictionary management
- [x] Checkpoint save/load
- [x] SAE lens for DarkTraceEngine

### Phase 8: Multilayer Circuits ✅ COMPLETE
- [x] CircuitNode and CircuitEdge types
- [x] Circuit discovery via activation patching
- [x] Circuit visualization
- [x] Attribution tracking
- [x] Causal tracing

### Phase 9: Multi-Model Support ✅ COMPLETE (December 2025)
- [x] **ModelAdapter protocol** - Unified interface for any model
- [x] **PolicyAdapter** - HoloLoom NeuralCore integration (~585 lines)
- [x] **TransformerAdapter** - HuggingFace transformer support
- [x] **Cross-model fingerprinting** - Compare features across models (~807 lines)
- [x] Universal feature detection (similarity ≥ 0.8)
- [x] Model-specific feature detection (similarity ≤ 0.3)
- [x] Feature correspondence mapping

### Phase 10: Integration & Polish ✅ COMPLETE (December 2025)
- [x] **HoloLoom orchestrator integration** (~827 lines)
- [x] Automatic activation capture during weaving
- [x] Real-time interpretability analysis
- [x] Steering vector injection into policy decisions
- [x] Safety-aware feature monitoring
- [x] Complete documentation

### Phase 11: Ecosystem ✅ COMPLETE (December 2025)
- [x] **Plugin system implementation** (~9,447 lines)
  - Plugin protocol and metadata (`plugin_protocol.py`)
  - Plugin registry with dependency resolution (`registry.py`)
  - Safety gates and trust levels (`safety_gate.py`)
  - Dynamic plugin loader (`plugin_loader.py`)
  - Plugin discovery via entry points (`discovery.py`)
  - Alignment framework integration (`alignment_bridge.py`)
  - Plugin signing and verification (`plugin_signing.py`)
  - Marketplace infrastructure (`plugin_marketplace.py`)
- [x] **3 builtin plugins** (SafetyMonitor, MetricsExporter, AlignmentValidator)
- [x] **96+ plugin system tests** (all passing)
- [x] [Plugin Development Guide](./plugins/DEVELOPMENT.md)

---

## Contributing

darkTrace is part of the HoloLoom project. Contributions welcome!

**Guidelines**:
1. **Reusability First**: Keep core modules application-independent
2. **Safety First**: All control/exploit features must have safety checks
3. **Documentation**: Comprehensive docs for all public APIs
4. **Testing**: Unit tests + integration tests
5. **Ethics**: Follow responsible disclosure for vulnerabilities

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for details.

---

## License

Same as HoloLoom (MIT License)

---

## Citation

```bibtex
@software{darktrace2025,
  title={darkTrace: Semantic Reverse Engineering of Large Language Models},
  author={HoloLoom Team},
  year={2025},
  url={https://github.com/your-repo/HoloLoom/darkTrace}
}
```

---

## Contact

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Security: security@hololoom.ai (responsible disclosure)

---

## Phase 9: Multi-Model Support (December 2025)

Phase 9 adds support for interpretability across multiple model types through a unified adapter protocol.

### Model Adapters

**ModelAdapter Protocol** (`models/adapter.py`):
```python
from HoloLoom.dark_trace.models import ModelAdapter, LayerInfo, SteeringConfig

class MyModelAdapter(ModelAdapter):
    def get_activations(self, inputs, layer):
        # Extract activations from layer
        ...

    def inject_steering(self, vector, layer, scale):
        # Modify activations during forward pass
        ...

adapter = MyModelAdapter(model)
acts = adapter.get_activations(inputs, "layer.5")
```

**Available Adapters**:
- **PolicyAdapter** - HoloLoom's NeuralCore policy network
- **TransformerAdapter** - HuggingFace transformer models
- **DummyAdapter** - Testing and development

### Cross-Model Fingerprinting

Compare learned features across different model architectures:

```python
from HoloLoom.dark_trace.models import (
    ModelFingerprinter,
    compare_models,
    find_universal_features,
    find_model_specific_features,
)

# Create fingerprinters for each model
fingerprinter1 = ModelFingerprinter(adapter1, model_id="claude")
fingerprinter2 = ModelFingerprinter(adapter2, model_id="gpt4")

# Generate fingerprints
fp1 = fingerprinter1.fingerprint_layer("block.5.mha", probe_inputs)
fp2 = fingerprinter2.fingerprint_layer("layers.5.attention", probe_inputs)

# Compare models
report = compare_models({"claude": fp1, "gpt4": fp2})
print(f"Overall similarity: {report.overall_similarity:.2f}")
print(f"Universal features: {len(report.universal_features)}")
print(f"Model-specific: {report.model_specific_features}")

# Find universal features (appear across all models)
universal = find_universal_features(fingerprints_by_model, threshold=0.8)

# Find model-specific features (unique to each model)
specific = find_model_specific_features(fingerprints_by_model, threshold=0.3)
```

---

## Phase 10: Orchestrator Integration (December 2025)

Phase 10 provides seamless integration with HoloLoom's weaving orchestrator.

### Quick Start

```python
from HoloLoom.integrations import (
    DarkTraceIntegration,
    IntegrationConfig,
    IntegrationMode,
    create_integration,
    enable_dark_trace,
)

# Create integration
config = IntegrationConfig.passive()  # or .active() or .full()
integration = create_integration(orchestrator, config)

# Or use the shorthand
integration = enable_dark_trace(orchestrator, mode=IntegrationMode.PASSIVE)

# Weave with interpretability
spacetime = await orchestrator.weave(query)

# Access interpretability results
trace = integration.get_last_trace()
print(trace.explain(verbosity=2))
```

### Integration Modes

| Mode | Analysis | Steering | Safety Override |
|------|----------|----------|-----------------|
| **DISABLED** | ❌ | ❌ | ❌ |
| **PASSIVE** | ✅ | ❌ | ❌ |
| **ACTIVE** | ✅ | ✅ | ❌ |
| **FULL** | ✅ | ✅ | ✅ |

### Steering

Apply steering vectors to influence policy decisions:

```python
# Enable steering
integration = create_integration(
    orchestrator,
    config=IntegrationConfig.active()
)

# Set steering goals
result = integration.set_steering({
    "semantic.Warmth": 1.5,      # Increase warmth
    "semantic.Formality": -0.5,  # Decrease formality
    "sae.42": 0.8,               # Activate SAE feature 42
})

print(f"Applied: {result.features_applied}")
print(f"Blocked: {result.features_blocked}")

# Clear steering
integration.clear_steering()
```

### Safety Monitoring

Configure safety-aware behavior:

```python
config = IntegrationConfig(
    mode=IntegrationMode.ACTIVE,
    safety_monitoring=True,
    block_on_safety_concern=True,  # Block weave if issues detected
    safety_callback=my_safety_handler,  # Custom callback
)

integration = create_integration(orchestrator, config)

# Weave (may be blocked if safety concerns)
result = await integration.wrap_weave(orchestrator.weave, query)
if result.safety_blocked:
    print("⚠️  Weave blocked due to safety concern")
    print(result.trace.safety_summary)
```

### Decorator Usage

```python
from HoloLoom.integrations import with_interpretability, IntegrationMode

class MyOrchestrator:
    @with_interpretability(mode=IntegrationMode.ACTIVE)
    async def weave(self, query):
        # Regular weave implementation
        ...

# Returns IntegrationResult with spacetime + trace
result = await orchestrator.weave(query)
print(result.spacetime)  # Original result
print(result.trace)      # Interpretability trace
```

### Cross-Model Analysis

Compare policy fingerprints across versions or models:

```python
# Generate fingerprint for current policy
fp1 = integration.fingerprint_policy(
    probe_inputs=probe_data,
    model_id="policy_v1",
    layers=["block.0.mha", "readout"]
)

# Later: compare with updated policy
report = integration.compare_with_model(fp2, "policy_v2")
print(f"Similarity: {report.overall_similarity:.2f}")
print(f"Drift: {1 - report.overall_similarity:.2%}")
```

---

**Status**: 🟢 Production Ready (Phase 10 Complete)

**Last Updated**: 2025-12-22
