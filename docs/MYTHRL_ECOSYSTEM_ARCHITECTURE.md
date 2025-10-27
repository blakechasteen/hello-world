# 🎯 mythRL ECOSYSTEM ARCHITECTURE

**Complete Architecture for Intelligent Multi-App System**

Version: 1.0
Date: 2025-10-27
Status: Design Phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Principles](#core-principles)
4. [Module Structure](#module-structure)
5. [Apps Ecosystem](#apps-ecosystem)
6. [Smart Dashboard Architecture](#smart-dashboard-architecture)
7. [Integration Patterns](#integration-patterns)
8. [Data Flow](#data-flow)
9. [Development Roadmap](#development-roadmap)
10. [Technical Specifications](#technical-specifications)

---

## Executive Summary

mythRL is an **intelligent ecosystem** for semantic analysis, prompt management, and LLM research. It consists of:

- **HoloLoom Core**: Reusable semantic analysis primitives
- **Three Specialized Apps**: darkTrace (security research), Promptly (prompt management), narrative_analyzer (literary analysis)
- **Smart Dashboard**: Intelligent UI that learns and adapts

**Key Innovation**: The dashboard itself is an **intelligent agent** that:
- Semantically understands user intent
- Predicts what users need proactively
- Learns from interactions over time
- Adapts layout and insights dynamically

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    mythRL ECOSYSTEM                              │
│          Intelligent Multi-App Analysis System                   │
└─────────────────────────────────────────────────────────────────┘

                           ┌──────────────────┐
                           │  Smart Dashboard │  ← Learns & Adapts
                           │   (mythRL_ui)    │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
            │  darkTrace   │ │  Promptly  │ │ narrative_ │
            │   Security   │ │   Prompt   │ │  analyzer  │
            │   Research   │ │   Mgmt     │ │  Literary  │
            └───────┬──────┘ └─────┬──────┘ └─────┬──────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   HoloLoom Core     │
                         │  (Reusable Modules) │
                         └─────────────────────┘
```

---

## Core Principles

### 1. **Clean Dependency Hierarchy**

```
mythRL_ui (Smart Dashboard)
    ↓ depends on
Apps (darkTrace, Promptly, narrative_analyzer)
    ↓ depends on
HoloLoom Core (semantic_calculus, policy, warp, etc.)

❌ NEVER: Core depends on Apps
❌ NEVER: Apps depend on Dashboard
❌ NEVER: Apps depend on each other
✅ ALWAYS: Clear upward dependencies only
```

### 2. **Apps are Standalone**

Each app must work independently:

```bash
# Each app works alone
cd apps/darkTrace && pip install -e . && darktrace --help
cd apps/Promptly && pip install -e . && promptly --help
cd apps/narrative_analyzer && pip install -e . && narrative-analyzer --help

# Dashboard is optional integration layer
cd mythRL_ui && pip install -e .  # Installs all apps
```

### 3. **Intelligence at Every Layer**

- **Core**: Semantic analysis, learning primitives
- **Apps**: Domain-specific intelligence
- **Dashboard**: Meta-intelligence (orchestrates apps)

### 4. **Learning from Interactions**

Every component learns:
- **Core**: Pattern recognition via reflection
- **Apps**: Domain-specific patterns
- **Dashboard**: User workflow patterns

---

## Module Structure

```
mythRL/
│
├─ HoloLoom/                           # 🔧 Core Library
│   ├─ semantic_calculus/              # Semantic analysis engine
│   │   ├─ analyzer.py                 # Semantic state analysis
│   │   ├─ dimensions.py               # 244D semantic space
│   │   ├─ system_id.py                # Trajectory prediction
│   │   ├─ flow_calculus.py            # Flow dynamics
│   │   ├─ integrator.py               # Geometric integration
│   │   └─ ...
│   │
│   ├─ policy/                         # Decision making
│   │   ├─ unified.py                  # Neural policy engine
│   │   └─ semantic_nudging.py         # Semantic steering
│   │
│   ├─ warp/math/                      # Mathematical operations
│   │   ├─ meaning_synthesizer.py     # Math → language
│   │   ├─ operation_selector.py      # Smart operation selection
│   │   └─ ...
│   │
│   ├─ reflection/                     # Learning system
│   │   ├─ buffer.py                   # Reflection buffer
│   │   ├─ rewards.py                  # Reward extraction
│   │   └─ ppo_trainer.py              # PPO training
│   │
│   ├─ embedding/                      # Embedding utilities
│   ├─ memory/                         # Memory backends
│   ├─ fabric/                         # Spacetime weaving
│   ├─ loom/                           # Pattern cards
│   └─ ...                             # Other core modules
│
├─ apps/                               # 📦 Specialized Applications
│   │
│   ├─ darkTrace/                      # 🌑 Security Research
│   │   ├─ README.md
│   │   ├─ pyproject.toml             # Deps: hololoom>=1.0
│   │   ├─ darkTrace/
│   │   │   ├─ __init__.py
│   │   │   ├─ cli.py                 # Standalone CLI
│   │   │   ├─ api.py                 # API for dashboard
│   │   │   │
│   │   │   ├─ observers/             # Layer 1: Observation
│   │   │   │   ├─ semantic_observer.py
│   │   │   │   ├─ trajectory_recorder.py
│   │   │   │   ├─ dimension_tracker.py
│   │   │   │   └─ flow_analyzer.py
│   │   │   │
│   │   │   ├─ analyzers/             # Layer 2: Analysis
│   │   │   │   ├─ trajectory_predictor.py
│   │   │   │   ├─ pattern_recognizer.py
│   │   │   │   ├─ attractor_detector.py
│   │   │   │   └─ fingerprint_generator.py
│   │   │   │
│   │   │   ├─ controllers/           # Layer 3: Control
│   │   │   │   ├─ embedding_manipulator.py
│   │   │   │   ├─ semantic_nudger.py
│   │   │   │   ├─ control_vector.py
│   │   │   │   └─ attack_library.py
│   │   │   │
│   │   │   ├─ exploits/              # Layer 4: Exploitation
│   │   │   │   ├─ jailbreaker.py
│   │   │   │   ├─ behavior_cloner.py
│   │   │   │   ├─ adversarial_generator.py
│   │   │   │   └─ safety_analyzer.py
│   │   │   │
│   │   │   ├─ datasets/              # Training data
│   │   │   ├─ models/                # Learned models
│   │   │   ├─ plugins/               # Extension system
│   │   │   └─ utils/
│   │   │
│   │   ├─ examples/
│   │   ├─ tests/
│   │   └─ docs/
│   │
│   ├─ Promptly/                       # 📝 Prompt Management
│   │   ├─ README.md
│   │   ├─ pyproject.toml             # Deps: hololoom>=1.0
│   │   ├─ promptly/
│   │   │   ├─ __init__.py
│   │   │   ├─ cli.py                 # promptly CLI
│   │   │   ├─ api.py                 # API for dashboard
│   │   │   │
│   │   │   ├─ promptly.py            # Core engine
│   │   │   ├─ execution_engine.py    # Orchestration
│   │   │   ├─ loop_composition.py    # Loop DSL
│   │   │   ├─ recursive_loops.py     # Recursive support
│   │   │   ├─ package_manager.py     # Skill management
│   │   │   │
│   │   │   ├─ tools/                 # Utilities
│   │   │   │   ├─ ab_testing.py
│   │   │   │   ├─ llm_judge.py
│   │   │   │   ├─ llm_judge_enhanced.py
│   │   │   │   ├─ cost_tracker.py
│   │   │   │   ├─ prompt_analytics.py
│   │   │   │   └─ ...
│   │   │   │
│   │   │   ├─ integrations/
│   │   │   │   ├─ hololoom_bridge.py # HoloLoom integration
│   │   │   │   └─ mcp_server.py      # MCP support
│   │   │   │
│   │   │   ├─ ui/                    # UI components
│   │   │   │   ├─ terminal_app.py
│   │   │   │   └─ web_app.py
│   │   │   │
│   │   │   └─ skill_templates/       # Skill scaffolding
│   │   │
│   │   ├─ demos/
│   │   ├─ tests/
│   │   ├─ docs/
│   │   └─ templates/
│   │
│   └─ narrative_analyzer/             # 📖 Literary Analysis
│       ├─ README.md
│       ├─ pyproject.toml             # Deps: hololoom>=1.0
│       ├─ narrative_analyzer/
│       │   ├─ __init__.py
│       │   ├─ cli.py                 # Standalone CLI
│       │   ├─ api.py                 # API for dashboard
│       │   │
│       │   ├─ matryoshka_depth.py    # Depth analysis
│       │   ├─ archetypes.py          # Archetypal patterns
│       │   ├─ emotional_arc.py       # Emotional trajectories
│       │   └─ ...
│       │
│       ├─ examples/
│       ├─ tests/
│       └─ docs/
│
└─ mythRL_ui/                          # 🧠 Smart Dashboard
    ├─ README.md
    ├─ pyproject.toml                  # Deps: darktrace, promptly, narrative_analyzer
    │
    ├─ backend/                        # FastAPI Backend
    │   ├─ main.py                     # FastAPI app
    │   │
    │   ├─ smart_agent/                # 🧠 Intelligence Layer
    │   │   ├─ __init__.py
    │   │   ├─ intent_analyzer.py     # Semantic intent understanding
    │   │   ├─ insight_generator.py   # Proactive insight generation
    │   │   ├─ layout_optimizer.py    # Adaptive layout
    │   │   ├─ context_predictor.py   # Predict user needs
    │   │   └─ learning_loop.py       # Dashboard learning system
    │   │
    │   ├─ orchestrator/               # WeavingShuttle Integration
    │   │   ├─ dashboard_weaver.py    # Each view is a "weaving cycle"
    │   │   └─ unified_semantic_view.py # Cross-app semantic aggregation
    │   │
    │   ├─ integrations/               # App Connectors
    │   │   ├─ base.py                # AppConnector base class
    │   │   ├─ darktrace_connector.py # darkTrace integration
    │   │   ├─ promptly_connector.py  # Promptly integration
    │   │   ├─ narrative_connector.py # narrative_analyzer integration
    │   │   └─ plugin_loader.py       # Dynamic app loading
    │   │
    │   ├─ reflection/                 # Dashboard Learning
    │   │   ├─ user_patterns.py       # Learn user workflows
    │   │   ├─ view_performance.py    # Track view effectiveness
    │   │   └─ recommendation_engine.py # Proactive recommendations
    │   │
    │   ├─ routers/                    # API Routes
    │   │   ├─ darktrace.py           # /api/darktrace/*
    │   │   ├─ promptly.py            # /api/promptly/*
    │   │   ├─ narrative.py           # /api/narrative/*
    │   │   └─ unified.py             # /api/unified/* (cross-app)
    │   │
    │   ├─ services/                   # Business Logic
    │   │   ├─ cross_app_comparison.py
    │   │   ├─ semantic_timeline.py
    │   │   └─ llm_comparison.py
    │   │
    │   └─ models/                     # Data models
    │       ├─ semantic_state.py
    │       ├─ user_context.py
    │       └─ insight.py
    │
    ├─ frontend/                       # React Frontend
    │   ├─ src/
    │   │   ├─ App.tsx
    │   │   │
    │   │   ├─ adaptive_layout/        # Self-Organizing UI
    │   │   │   ├─ SmartGrid.tsx      # Adaptive grid layout
    │   │   │   ├─ ContextualPanel.tsx # Context-aware panels
    │   │   │   ├─ InsightCard.tsx    # AI-generated insights
    │   │   │   └─ AdaptiveView.tsx   # Views that adapt to content
    │   │   │
    │   │   ├─ views/                  # Main Views
    │   │   │   ├─ IntelligentDashboard.tsx   # Main smart view
    │   │   │   ├─ UnifiedTimeline.tsx        # Cross-app timeline
    │   │   │   ├─ CrossAppComparison.tsx     # Compare apps
    │   │   │   ├─ RealTimeMonitoring.tsx     # Real-time LLM monitoring
    │   │   │   ├─ LLMComparison.tsx          # Compare different LLMs
    │   │   │   │
    │   │   │   ├─ darktrace/          # darkTrace-specific views
    │   │   │   │   ├─ ObservationView.tsx
    │   │   │   │   ├─ PredictionView.tsx
    │   │   │   │   └─ FingerprintView.tsx
    │   │   │   │
    │   │   │   ├─ promptly/           # Promptly-specific views
    │   │   │   │   ├─ PromptLibrary.tsx
    │   │   │   │   ├─ LoopComposer.tsx
    │   │   │   │   └─ ABTestingView.tsx
    │   │   │   │
    │   │   │   └─ narrative/          # narrative_analyzer views
    │   │   │       ├─ DepthAnalysis.tsx
    │   │   │       ├─ ArchetypeView.tsx
    │   │   │       └─ EmotionalArc.tsx
    │   │   │
    │   │   ├─ agents/                 # Frontend Intelligence
    │   │   │   ├─ intent_detector.ts  # Client-side intent
    │   │   │   ├─ insight_renderer.ts # Render AI insights
    │   │   │   └─ adaptive_behavior.ts # Client learning
    │   │   │
    │   │   ├─ components/             # Reusable Components
    │   │   │   ├─ SemanticGraph.tsx
    │   │   │   ├─ TrajectoryChart.tsx
    │   │   │   ├─ DimensionHeatmap.tsx
    │   │   │   └─ InsightPanel.tsx
    │   │   │
    │   │   └─ hooks/                  # Custom Hooks
    │   │       ├─ useSmartLayout.ts
    │   │       ├─ useSemanticState.ts
    │   │       └─ useAdaptiveBehavior.ts
    │   │
    │   ├─ package.json
    │   └─ tsconfig.json
    │
    ├─ docs/
    │   ├─ ARCHITECTURE.md
    │   ├─ API_REFERENCE.md
    │   ├─ SMART_FEATURES.md
    │   └─ INTEGRATION_GUIDE.md
    │
    └─ tests/
```

---

## Apps Ecosystem

### App 1: darkTrace (Security Research)

**Purpose**: Semantic reverse engineering of LLMs

**Capabilities**:
- Real-time semantic observation
- Trajectory prediction
- LLM fingerprinting
- Embedding manipulation (research)
- Safety analysis

**Standalone Usage**:
```bash
# CLI
darktrace observe --llm claude --monitor
darktrace fingerprint --outputs dataset.json
darktrace predict --text "current output" --steps 10

# Python API
from darkTrace import SemanticObserver, TrajectoryPredictor

observer = SemanticObserver()
state = observer.observe(text)
```

**Dashboard Integration**:
```python
# backend/integrations/darktrace_connector.py
from darkTrace.api import DarkTraceAPI

class DarkTraceConnector(AppConnector):
    def __init__(self):
        self.api = DarkTraceAPI()

    def predict_trajectory(self, text: str, n_steps: int):
        return self.api.predict(text, n_steps)
```

---

### App 2: Promptly (Prompt Management)

**Purpose**: Prompt composition, testing, and optimization

**Capabilities**:
- Prompt library management
- Loop DSL composition
- A/B testing framework
- LLM judge evaluation
- Cost tracking & analytics
- HoloLoom memory integration
- MCP server

**Standalone Usage**:
```bash
# CLI
promptly list
promptly run research_loop --input "quantum computing"
promptly test prompt_v1 prompt_v2 --judge llm

# Python API
from promptly import Promptly

p = Promptly()
result = p.chain(["analyze", "refine", "optimize"])
```

**Dashboard Integration**:
```python
# backend/integrations/promptly_connector.py
from promptly.api import PromptlyAPI

class PromptlyConnector(AppConnector):
    def __init__(self):
        self.api = PromptlyAPI()

    def execute_loop(self, loop_definition: str):
        return self.api.execute_loop(loop_definition)
```

---

### App 3: narrative_analyzer (Literary Analysis)

**Purpose**: Deep narrative and literary analysis

**Capabilities**:
- Matryoshka depth analysis (SURFACE → COSMIC)
- Archetypal pattern recognition
- Emotional trajectory tracking
- 244D semantic analysis
- Domain-optimized analysis (FUSED_NARRATIVE)

**Standalone Usage**:
```bash
# CLI
narrative-analyzer analyze odyssey.txt --mode fused_narrative
narrative-analyzer depth odyssey.txt --levels all

# Python API
from narrative_analyzer import MatryoshkaNarrativeDepth

analyzer = MatryoshkaNarrativeDepth()
depth = analyzer.analyze(text)
```

**Dashboard Integration**:
```python
# backend/integrations/narrative_connector.py
from narrative_analyzer.api import NarrativeAnalyzerAPI

class NarrativeConnector(AppConnector):
    def __init__(self):
        self.api = NarrativeAnalyzerAPI()

    def analyze_depth(self, text: str):
        return self.api.analyze_depth(text)
```

---

## Smart Dashboard Architecture

### Philosophy

The dashboard is **NOT** a traditional UI - it's an **intelligent agent** that:

1. **Understands Intent** - Uses semantic analysis to understand what user wants
2. **Generates Insights** - Proactively generates actionable insights
3. **Adapts Layout** - Reorganizes based on usage patterns
4. **Predicts Needs** - Uses darkTrace to predict what user needs next
5. **Learns Continuously** - Improves from every interaction

### Core Components

#### 1. Intent Analyzer

```python
# backend/smart_agent/intent_analyzer.py
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer

class DashboardIntentAnalyzer:
    """
    Understands user intent from actions using semantic analysis.

    Maps user actions → semantic intent → appropriate apps/views
    """

    def __init__(self):
        self.analyzer = create_semantic_analyzer(embed_fn, config)

    def analyze_user_action(self, action: UserAction) -> Intent:
        """
        Semantic analysis of user action.

        Examples:
        - Paste "Then Odysseus wept" → NARRATIVE_ANALYSIS intent
        - Ask "Compare these LLMs" → MODEL_COMPARISON intent
        - Select technical text → TECHNICAL_ANALYSIS intent
        """
        result = self.analyzer.analyze_text(action.text)
        dominant = result['semantic_forces']['dominant_velocity']

        # Map dominant dimensions to intent
        if self._has_narrative_dimensions(dominant):
            return Intent(
                type="NARRATIVE_ANALYSIS",
                confidence=0.9,
                recommended_apps=["narrative_analyzer"],
                recommended_views=["depth_analysis", "emotional_arc"]
            )
        elif self._has_comparative_pattern(action.context):
            return Intent(
                type="MODEL_COMPARISON",
                confidence=0.85,
                recommended_apps=["darkTrace", "narrative_analyzer"],
                recommended_views=["llm_comparison", "fingerprint_comparison"]
            )

        return intent
```

**How It Works**:
```
User Action → Semantic Analysis → Intent Detection → App Selection → View Recommendation
```

#### 2. Insight Generator

```python
# backend/smart_agent/insight_generator.py
from HoloLoom.warp.math.meaning_synthesizer import MeaningSynthesizer
from darkTrace.analyzers import TrajectoryPredictor

class SmartInsightGenerator:
    """
    Generates proactive insights by combining results from multiple apps.

    Uses:
    - darkTrace prediction
    - narrative_analyzer patterns
    - Promptly analytics
    - HoloLoom reflection
    """

    def generate_insights(
        self,
        narrative_result: dict,
        darktrace_result: dict,
        promptly_result: dict
    ) -> List[Insight]:
        """
        Generate cross-app insights.

        Examples:
        - "High semantic alignment (94%) across all analyzers - confident"
        - "darkTrace predicts climax approaching (82% confidence)"
        - "Pattern matches 3 past Odyssey analyses from narrative_analyzer"
        - "Promptly suggests similar refinement loop worked before"
        """
        insights = []

        # Cross-app agreement
        alignment = self._compute_alignment(narrative_result, darktrace_result)
        if alignment > 0.9:
            insights.append(Insight(
                type="CROSS_APP_AGREEMENT",
                text=f"All analyzers agree ({alignment:.0%}) - high confidence",
                confidence=alignment,
                priority="HIGH",
                actions=["trust_analysis", "proceed_with_confidence"]
            ))

        # Predictive insight from darkTrace
        prediction = darktrace_result.get('prediction')
        if prediction and prediction['confidence'] > 0.8:
            insights.append(Insight(
                type="PREDICTION",
                text=f"Predicted next: {prediction['dominant'][0]} ({prediction['confidence']:.0%})",
                confidence=prediction['confidence'],
                priority="MEDIUM",
                actions=["prepare_for_change", "monitor_trajectory"]
            ))

        # Historical pattern from reflection
        similar = self.reflection.find_similar(narrative_result['forces'])
        if similar:
            insights.append(Insight(
                type="HISTORICAL_PATTERN",
                text=f"Matches {len(similar)} past analyses - recognized pattern",
                confidence=0.75,
                priority="LOW",
                actions=["review_similar_cases", "apply_learned_patterns"]
            ))

        return insights
```

**Output Example**:
```
┌─────────────────────────────────────────────────────┐
│ 💡 SMART INSIGHTS (Auto-generated)                  │
├─────────────────────────────────────────────────────┤
│ 🎯 High Priority:                                   │
│   • All analyzers agree (94%) - high confidence     │
│     → Action: Trust analysis, proceed confidently   │
│                                                      │
│   • Emotional climax approaching (darkTrace)        │
│     → Action: Monitor for narrative peak            │
│                                                      │
│ 📊 Medium Priority:                                 │
│   • Predicted: Recognition (82% confidence)         │
│     → Action: Prepare for revelation scene          │
│                                                      │
│ 📚 Context:                                         │
│   • Pattern matches 3 past Odyssey analyses         │
│   • Promptly found similar refinement successful    │
└─────────────────────────────────────────────────────┘
```

#### 3. Layout Optimizer

```python
# backend/smart_agent/layout_optimizer.py
from HoloLoom.reflection.buffer import ReflectionBuffer

class SmartLayoutOptimizer:
    """
    Learns optimal layouts from user behavior.

    Tracks:
    - Which views user looks at most
    - Which insights lead to actions
    - Time spent per view
    - Successful workflow patterns
    """

    def __init__(self):
        self.reflection = ReflectionBuffer(persist_path="./dashboard_memory")
        self.user_patterns = self._load_patterns()

    def optimize_layout(self, context: AnalysisContext) -> LayoutConfig:
        """
        Determine optimal layout based on:
        1. Current analysis context (semantic)
        2. User's past behavior patterns
        3. What's been helpful historically
        """

        # Get user preferences from past interactions
        metrics = self.reflection.get_metrics()

        # Build layout based on learned patterns
        if context.content_type == "narrative":
            # User historically looks at depth analysis first
            if metrics.view_usage['depth_analysis'] > 10:
                layout = LayoutConfig(
                    main_view="depth_analysis",
                    sidebar=["emotional_arc", "insights"],
                    bottom=["predictions", "comparisons"]
                )

        # Adaptive: Sort by actual usage frequency
        layout.sort_by_usage(self.user_patterns['view_usage'])

        return layout
```

**Adaptive Behavior**:
```
Week 1: Default layout
  → User always checks predictions first

Week 2: Dashboard learns
  → Moves predictions to top
  → Pre-loads prediction data

Week 3: Full adaptation
  → Only shows views user actually uses
  → Hides panels user ignores
  → Pre-fetches data for likely next view
```

#### 4. Context Predictor

```python
# backend/smart_agent/context_predictor.py
from darkTrace.analyzers import TrajectoryPredictor

class DashboardContextPredictor:
    """
    Predicts what user likely wants to see next.

    Uses darkTrace trajectory prediction ON USER BEHAVIOR!
    """

    def predict_next_action(
        self,
        current_view: str,
        analysis_history: list,
        user_trajectory: list  # Past user actions as semantic trajectory
    ) -> Prediction:
        """
        Predict user's next likely action using semantic trajectory analysis.

        Example:
        - User viewed narrative analysis
        - Historically, 80% of time they compare with darkTrace next
        - Prediction: 85% probability they want comparison view
        - Dashboard: Pre-loads comparison data
        """

        # Use darkTrace predictor on user behavior
        predictor = TrajectoryPredictor()

        # Learn from user's action patterns (treating them as text)
        action_descriptions = [a.description for a in user_trajectory]
        predictor.learn_from_outputs(action_descriptions)

        # Predict next action
        prediction = predictor.predict(
            current_text=f"User viewed {current_view}",
            n_steps=3
        )

        # Pre-fetch if confident
        if prediction.confidence > 0.7:
            predicted_views = self._map_dimensions_to_views(
                prediction.dominant_dimensions
            )
            for view in predicted_views:
                self._prefetch_data(view)

        return prediction
```

**User Experience**:
```
User opens narrative analysis view
  ↓
Dashboard predicts (85% confidence): User will compare with darkTrace
  ↓
Dashboard pre-loads:
  - darkTrace fingerprint
  - Comparison charts
  - Cross-app insights
  ↓
User clicks "Compare" → Instant load! (already prepared)
```

#### 5. Learning Loop

```python
# backend/reflection/user_patterns.py
from HoloLoom.reflection.buffer import ReflectionBuffer
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace

class DashboardLearningLoop:
    """
    Dashboard learns from every interaction.

    Uses HoloLoom reflection system to:
    - Track helpful vs. unhelpful views
    - Learn successful workflow patterns
    - Generate recommendations
    - Adapt behavior over time
    """

    def __init__(self):
        self.reflection = ReflectionBuffer(persist_path="./dashboard_memory")

    async def record_interaction(self, interaction: UserInteraction):
        """
        Record interaction as Spacetime artifact.

        Tracks:
        - View opened (tool_selected)
        - Time spent (duration)
        - Actions taken (result)
        - User feedback (explicit or implicit)
        """

        spacetime = Spacetime(
            query=interaction.context,
            tool_selected=interaction.view_name,
            result={
                'action_taken': interaction.action_taken,
                'duration_ms': interaction.duration_ms,
                'insights_clicked': interaction.insights_clicked
            },
            trace=WeavingTrace(
                timestamp=datetime.now(),
                duration_ms=interaction.duration_ms,
                metadata={
                    'helpful': interaction.user_rating,
                    'engagement': interaction.engagement_score
                }
            )
        )

        # Store with feedback
        await self.reflection.store(
            spacetime,
            feedback={
                'helpful': interaction.helpful,
                'led_to_action': interaction.action_taken is not None,
                'engagement': interaction.duration_ms / 1000.0  # seconds
            }
        )

    def get_recommendations(self) -> List[Recommendation]:
        """
        Generate recommendations based on learning.

        Examples:
        - "Try darkTrace fingerprinting - you haven't used it yet"
        - "Narrative depth analysis works great for your content (92% success)"
        - "Consider real-time monitoring for LLM analysis"
        """

        metrics = self.reflection.get_metrics()
        recs = []

        # Unused features (exploration)
        all_views = ['depth_analysis', 'fingerprint', 'comparison', 'monitoring']
        used_views = set(metrics.tool_usage_counts.keys())
        unused = set(all_views) - used_views

        for view in unused:
            recs.append(Recommendation(
                text=f"Try {view} - you haven't explored it yet",
                confidence=0.7,
                category="FEATURE_DISCOVERY",
                priority="LOW"
            ))

        # Successful patterns (exploitation)
        if metrics.tool_success_rates:
            best_view = max(
                metrics.tool_success_rates.items(),
                key=lambda x: x[1]
            )
            if best_view[1] > 0.9:
                recs.append(Recommendation(
                    text=f"'{best_view[0]}' works great ({best_view[1]:.0%} success)",
                    confidence=best_view[1],
                    category="CONFIRMATION",
                    priority="HIGH"
                ))

        return recs
```

**Learning Cycle**:
```
User Interaction
  ↓
Record as Spacetime
  ↓
Extract Learning Signals
  ↓
Update User Patterns
  ↓
Generate Recommendations
  ↓
Adapt Layout & Behavior
  ↓
(Repeat)
```

---

## Integration Patterns

### Pattern 1: Unified Semantic Timeline

**Goal**: Show semantic trajectory from ALL apps on same timeline

```python
# backend/services/semantic_timeline.py
from darkTrace import SemanticObserver
from narrative_analyzer import MatryoshkaNarrativeDepth

class UnifiedSemanticTimeline:
    """
    Combines semantic states from all apps into unified view.
    """

    def create_timeline(self, text: str) -> Timeline:
        """
        Analyze text with all apps and create unified timeline.
        """

        # Run all analyzers in parallel
        darktrace_state = darktrace_observer.observe(text)
        narrative_state = narrative_analyzer.analyze(text)

        # Combine into unified timeline
        timeline = Timeline()

        for word_idx in range(len(words)):
            point = TimelinePoint(
                word_index=word_idx,
                word=words[word_idx],

                # darkTrace data
                semantic_velocity=darktrace_state[word_idx].velocity,
                predicted_next=darktrace_state[word_idx].prediction,

                # narrative_analyzer data
                depth_level=narrative_state[word_idx].depth,
                archetypes=narrative_state[word_idx].archetypes,

                # Cross-app insights
                alignment=self._compute_alignment(
                    darktrace_state[word_idx],
                    narrative_state[word_idx]
                )
            )

            timeline.add_point(point)

        return timeline
```

**UI Display**:
```
┌─────────────────────────────────────────────────────────┐
│  UNIFIED SEMANTIC TIMELINE                              │
├─────────────────────────────────────────────────────────┤
│  Text: "Then throwing his arms around his father..."    │
│                                                          │
│  Word-by-word analysis:                                 │
│                                                          │
│  [Then] →                                               │
│    darkTrace: Velocity 0.3, Prediction: emotional↑      │
│    Narrative: SURFACE → SYMBOLIC transition             │
│    Alignment: 87%                                       │
│                                                          │
│  [throwing] →                                           │
│    darkTrace: Velocity 0.6↑, Prediction: physical      │
│    Narrative: Action cluster detected                   │
│    Alignment: 92%                                       │
│                                                          │
│  [arms] →                                               │
│    darkTrace: Velocity 0.7↑, Prediction: embrace       │
│    Narrative: Longing (0.95), Recognition (0.71)       │
│    Alignment: 94% ← HIGH AGREEMENT                      │
│                                                          │
│  [around] →                                             │
│    darkTrace: Velocity 0.8↑, Curvature spike           │
│    Narrative: ARCHETYPAL level reached                  │
│    Alignment: 96%                                       │
│                                                          │
│  [father] →                                             │
│    darkTrace: Attractor detected: "Reunion"            │
│    Narrative: MYTHIC depth, Father archetype            │
│    Alignment: 98% ← PEAK ALIGNMENT                      │
│                                                          │
│  💡 Insight: All apps detect emotional climax           │
└─────────────────────────────────────────────────────────┘
```

### Pattern 2: Cross-App Comparison

**Goal**: Use all apps to compare different LLM outputs

```python
# backend/services/llm_comparison.py

class LLMComparisonService:
    """
    Use all apps to comprehensively compare LLM outputs.
    """

    def compare_llms(
        self,
        prompt: str,
        llm_outputs: Dict[str, str]  # {llm_name: output}
    ) -> ComparisonResult:
        """
        Compare LLM outputs using all available apps.
        """

        results = {}

        for llm_name, output in llm_outputs.items():
            # darkTrace analysis
            darktrace_fp = darktrace_analyzer.fingerprint(output)
            darktrace_pred = darktrace_predictor.predict(output)

            # narrative_analyzer analysis
            narrative_depth = narrative_analyzer.analyze_depth(output)
            narrative_emotional = narrative_analyzer.emotional_arc(output)

            # Promptly analysis (optional - if evaluating prompts)
            if promptly_available:
                promptly_score = promptly_judge.evaluate(prompt, output)

            results[llm_name] = {
                'fingerprint': darktrace_fp,
                'prediction': darktrace_pred,
                'depth': narrative_depth,
                'emotional_arc': narrative_emotional,
                'score': promptly_score if promptly_available else None
            }

        # Generate comparison insights
        insights = self._generate_comparison_insights(results)

        return ComparisonResult(
            llm_results=results,
            insights=insights,
            winner=self._determine_winner(results)
        )
```

### Pattern 3: App Plugin System

**Goal**: Dynamically load new apps without modifying dashboard

```python
# backend/integrations/plugin_loader.py

class AppPluginLoader:
    """
    Dynamically discover and load app integrations.

    Apps register via entry points:

    [darktrace.dashboard]
    connector = darkTrace.integrations.dashboard:DashboardConnector
    """

    def discover_apps(self) -> List[AppPlugin]:
        """
        Discover all installed apps with dashboard integration.
        """
        import pkg_resources

        plugins = []
        for entry_point in pkg_resources.iter_entry_points('mythrl.apps'):
            try:
                plugin = entry_point.load()
                plugins.append(plugin())
            except Exception as e:
                logger.error(f"Failed to load {entry_point.name}: {e}")

        return plugins

    def load_app(self, app_name: str) -> AppConnector:
        """
        Load specific app connector.
        """
        if app_name == 'darktrace':
            from darkTrace.integrations.dashboard import DashboardConnector
            return DashboardConnector()
        elif app_name == 'promptly':
            from promptly.integrations.dashboard import DashboardConnector
            return DashboardConnector()
        elif app_name == 'narrative_analyzer':
            from narrative_analyzer.integrations.dashboard import DashboardConnector
            return DashboardConnector()
        else:
            raise ValueError(f"Unknown app: {app_name}")
```

---

## Data Flow

### Real-Time Analysis Flow

```
User inputs text
  ↓
1. Intent Analysis (semantic)
  ↓ determines which apps to use
  ↓
2. Multi-App Analysis (parallel)
  ├─ darkTrace observation
  ├─ Promptly evaluation
  └─ narrative_analyzer depth
  ↓ all return results
  ↓
3. Unified Semantic View
  ├─ Combine trajectories
  ├─ Compute alignment
  └─ Generate timeline
  ↓
4. Insight Generation
  ├─ Cross-app patterns
  ├─ Predictions
  └─ Recommendations
  ↓
5. Adaptive Display
  ├─ Optimal layout
  ├─ Context-aware panels
  └─ Proactive insights
  ↓
6. Learning Loop
  └─ Record interaction
      ↓
      Update patterns
      ↓
      Adapt behavior
```

### Learning Data Flow

```
User Interaction
  ↓
Spacetime Artifact Created
  ├─ Context: What user was doing
  ├─ Tool: Which view/app used
  ├─ Result: What happened
  ├─ Duration: Time spent
  └─ Feedback: Implicit/explicit
  ↓
Reflection Buffer Storage
  ↓
Pattern Extraction
  ├─ Successful workflows
  ├─ Preferred views
  ├─ Effective app combinations
  └─ Temporal patterns
  ↓
Learning Signals Generated
  ├─ Layout adjustments
  ├─ Recommendations
  ├─ Predictive pre-fetching
  └─ Insight prioritization
  ↓
Dashboard Adapts
```

---

## Development Roadmap

### Phase 1: Foundation ✅ (Current)
- [x] Architecture documentation
- [x] Ecosystem structure defined
- [ ] Core module organization
- [ ] App separation strategy

### Phase 2: Apps Structure (Weeks 1-2)
- [ ] Create `apps/` directory
- [ ] Move Promptly to `apps/Promptly/`
- [ ] Create `apps/darkTrace/` structure
- [ ] Update `apps/narrative_analyzer/` (if needed)
- [ ] Ensure each app has:
  - [ ] `pyproject.toml` with hololoom dependency
  - [ ] Standalone CLI
  - [ ] `api.py` for dashboard integration
  - [ ] Documentation

### Phase 3: darkTrace Layer 1-2 (Weeks 3-6)
- [ ] Implement darkTrace observers
- [ ] Implement darkTrace analyzers
- [ ] System identification integration
- [ ] Flow calculus integration
- [ ] Standalone CLI working
- [ ] API interface defined

### Phase 4: Smart Backend (Weeks 7-9)
- [ ] Intent analyzer implementation
- [ ] Insight generator implementation
- [ ] Layout optimizer implementation
- [ ] Context predictor implementation
- [ ] Learning loop integration
- [ ] App connectors for all three apps

### Phase 5: Dashboard Foundation (Weeks 10-12)
- [ ] FastAPI backend setup
- [ ] Basic app connectors working
- [ ] Unified semantic view service
- [ ] Cross-app comparison service
- [ ] Reflection integration

### Phase 6: Frontend (Weeks 13-16)
- [ ] React setup
- [ ] Adaptive layout components
- [ ] Basic views for each app
- [ ] Unified timeline view
- [ ] Cross-app comparison view

### Phase 7: Intelligence (Weeks 17-20)
- [ ] Frontend intent detection
- [ ] Adaptive behavior implementation
- [ ] Insight rendering
- [ ] Real-time updates
- [ ] Learning visualization

### Phase 8: Polish & Launch (Weeks 21-24)
- [ ] Performance optimization
- [ ] Documentation complete
- [ ] Examples and tutorials
- [ ] User testing
- [ ] Public release

---

## Technical Specifications

### Backend Stack

**Language**: Python 3.10+
**Framework**: FastAPI
**Database**: SQLite (dev), PostgreSQL (prod)
**Caching**: Redis
**Task Queue**: Celery (for long-running analyses)

**Key Dependencies**:
```toml
[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.104"
uvicorn = "^0.24"
pydantic = "^2.5"
redis = "^5.0"
celery = "^5.3"

# HoloLoom and apps
hololoom = {path = "../HoloLoom", develop = true}
darktrace = {path = "../apps/darkTrace", develop = true}
promptly = {path = "../apps/Promptly", develop = true}
narrative-analyzer = {path = "../apps/narrative_analyzer", develop = true}
```

### Frontend Stack

**Language**: TypeScript
**Framework**: React 18+
**Build Tool**: Vite
**State Management**: Zustand
**Data Fetching**: TanStack Query
**Visualization**: D3.js, Recharts
**UI Components**: shadcn/ui

**Key Dependencies**:
```json
{
  "dependencies": {
    "react": "^18.2",
    "typescript": "^5.2",
    "zustand": "^4.4",
    "@tanstack/react-query": "^5.8",
    "d3": "^7.8",
    "recharts": "^2.9",
    "@radix-ui/react-*": "latest"
  }
}
```

### API Design

**RESTful Endpoints**:

```
/api/v1/
  ├─ /apps/
  │   ├─ GET  /list                    # List available apps
  │   └─ GET  /{app_name}/info         # App metadata
  │
  ├─ /darktrace/
  │   ├─ POST /observe                 # Observe LLM output
  │   ├─ POST /predict                 # Predict trajectory
  │   ├─ POST /fingerprint             # Generate fingerprint
  │   └─ GET  /models                  # List saved models
  │
  ├─ /promptly/
  │   ├─ GET  /prompts                 # List prompts
  │   ├─ POST /execute                 # Execute prompt/loop
  │   ├─ POST /test                    # A/B test
  │   └─ GET  /analytics               # Get analytics
  │
  ├─ /narrative/
  │   ├─ POST /analyze                 # Analyze text
  │   ├─ POST /depth                   # Depth analysis
  │   └─ POST /emotional-arc           # Emotional arc
  │
  ├─ /unified/
  │   ├─ POST /timeline                # Unified timeline
  │   ├─ POST /compare                 # Cross-app comparison
  │   └─ POST /insights                # Generate insights
  │
  └─ /learning/
      ├─ POST /record                  # Record interaction
      ├─ GET  /recommendations         # Get recommendations
      └─ GET  /patterns                # Get learned patterns
```

**WebSocket Endpoints**:
```
/ws/realtime
  - Real-time LLM monitoring
  - Live semantic state updates
  - Streaming insights
```

---

## Success Metrics

### For Apps (Individual)
- **darkTrace**: Prediction accuracy > 70%, fingerprint distinctiveness > 85%
- **Promptly**: Prompt execution success > 95%, judge accuracy > 80%
- **narrative_analyzer**: Depth detection accuracy > 85%, archetype recognition > 80%

### For Dashboard (Integration)
- **Intent Recognition**: > 90% accuracy in detecting user intent
- **Insight Relevance**: > 75% of insights rated helpful by users
- **Layout Adaptation**: Reduces clicks by > 30% over time
- **Prediction Accuracy**: > 70% accuracy in predicting next user action
- **Learning Rate**: Noticeable improvement within 2 weeks of use

### For Ecosystem
- **App Independence**: Each app works standalone without dashboard
- **Integration Ease**: New app can be integrated in < 1 day
- **Performance**: Analysis completes in < 2s for typical inputs
- **Scalability**: Handles 100+ concurrent users

---

## Conclusion

The mythRL ecosystem provides a **complete intelligent platform** for:

1. **Specialized Analysis**: Three powerful apps for different domains
2. **Unified Intelligence**: Smart dashboard that learns and adapts
3. **Clean Architecture**: Reusable core, independent apps, optional UI
4. **Continuous Learning**: Every component improves from interactions

**Next Steps**:
1. Create `apps/` directory structure
2. Implement darkTrace Layer 1-2
3. Build smart backend foundation
4. Launch V1 dashboard

---

**Status**: Design Phase Complete
**Ready for**: Implementation Phase
**Target**: Q2 2025 Launch

