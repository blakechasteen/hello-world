# Jenny Roadmap: What's Next + MRF Integration

**Status**: MVP Complete (Week 4), Phase 2.1-2.2 Implemented
**Date**: December 2025
**Updated**: December 2025

## Current State (MVP Complete + Phase 2 Progress)

Jenny Week 1-4 MVP delivered:
- **Week 1**: Compiler + Spec model (JennySpec, QueryAnalysis)
- **Week 2**: Lifecycle + Renderer (NASCENT→STABLE→DISSOLVING→ARCHIVED)
- **Week 3**: Actions + Streaming (PIN, DISMISS, WHY, SSE/WebSocket)
- **Week 4**: Runtime + Orchestrator Integration (JennyRuntime, WeavingOrchestrator)

**Phase 2 Progress** (December 2025):
- ✅ **2.1**: MRF-Enhanced Panel Generation (jenny_mrf.py)
- ✅ **2.2**: Thompson Sampling Panel Type Selection (PanelTypeLearner)
- 🔲 **2.3**: LLM-Based Panel Compilation (planned)

**Test Coverage**: 257 tests passing (239 unit + 18 integration)
- Jenny unit tests: 198 (original) + 41 (MRF) = 239
- Jenny integration tests: 18

---

## Phase 2: Intelligence Enhancement (In Progress)

### 2.1 MRF-Enhanced Panel Generation ✅ IMPLEMENTED

**Status**: ✅ Complete (December 2025)
**Location**: `HoloLoom/visualization/jenny_mrf.py`

**Goal**: Use Metaprompt Refinement Framework to generate higher-quality panel content.

**What Was Implemented**:
- `generate_why_panel_mrf()` - MRF-enhanced WHY panel using ELEGANCE strategy
- `JennyMRFCompiler` - Full MRF-enhanced compiler extending base JennyCompiler
- Rule-based fallback when MRF unavailable (graceful degradation)

**Integration Points**:

| Jenny Component | MRF Enhancement | Benefit |
|-----------------|-----------------|---------|
| `_detect_jenny_panel_type()` | Thompson Sampling learning | Learn optimal panel type per query type |
| WHY meta-panel | ELEGANCE strategy | Clarity → Simplicity → Beauty |
| REASONING panel | VERIFY strategy | Accuracy + Completeness + Consistency passes |
| SOURCES panel | ELEGANCE strategy | Clearer attribution |
| Query analysis | MRF prompt analysis | Better intent detection |

**Usage**:

```python
from HoloLoom.visualization.jenny_mrf import JennyMRFCompiler

# Create MRF-enhanced compiler
compiler = JennyMRFCompiler(enable_learning=True)

# Compile with MRF enhancement
specs = await compiler.compile(spacetime)

# After user interaction (pin/dismiss), update learning
compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

# View learning statistics
stats = compiler.get_learning_statistics()
```

### 2.2 Thompson Sampling for Panel Type Selection ✅ IMPLEMENTED

**Status**: ✅ Complete (December 2025)
**Location**: `HoloLoom/visualization/jenny_mrf.py`

**Goal**: Learn which panel types work best for different query types.

**What Was Implemented**:
- `PanelTypePrior` - Beta distribution wrapper with sample/update methods
- `PanelTypeLearner` - Full Thompson Sampling learner with persistence
- Exploration bonus for encouraging diversity
- State persistence to disk (JSON format)
- Complete statistics and history tracking

**Key Features**:
- Beta(α, β) priors per (query_type, panel_type) pair
- Success: α ← α + confidence
- Failure: β ← β + (1 - confidence)
- Expected value: E[X] = α / (α + β)
- Exploration bonus parameter for forced diversity

**Usage**:

```python
from HoloLoom.visualization.jenny_mrf import PanelTypeLearner, PanelTypeJenny

# Create learner with persistence
learner = PanelTypeLearner(persist_path="./jenny_learning.json")

# Select panel type using Thompson Sampling
candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CONFIDENCE, PanelTypeJenny.GRAPH]
selected = learner.select("factual", candidates)

# Update based on user interaction
learner.update("factual", selected, success=True, confidence=0.85)

# Get best panel type (without sampling)
best = learner.get_best_panel_type("factual")

# View statistics
stats = learner.get_statistics()
print(f"Total selections: {stats['total_selections']}")
```

**Tests**: 41 unit tests covering:
- Prior initialization and updates
- Thompson Sampling selection
- Learning improves over time
- Exploration bonus
- State persistence
- Graceful handling of corrupted state

### 2.3 LLM-Based Panel Compilation

**Goal**: Replace rule-based panel generation with LLM-generated content.

**Current State**: Heuristic-based (regex patterns, threshold checks)
**Target State**: LLM understands intent and generates appropriate visualization

**Implementation**:

```python
# In jenny_compiler.py
class LLMJennyCompiler(JennyCompiler):
    """LLM-based panel compiler using MRF."""

    async def compile_with_llm(self, spacetime, llm_provider: str = "claude") -> JennySpec:
        """Use LLM to determine best panel type and content."""

        # Build MRF-enhanced prompt for panel selection
        prompt = await self.mrf.refine_prompt(
            original_prompt=f"""
            Analyze this query response and determine the best visualization:

            Query: {spacetime.trace.query if hasattr(spacetime.trace, 'query') else 'unknown'}
            Response: {spacetime.response[:500]}
            Confidence: {spacetime.confidence}
            Threads: {len(getattr(spacetime.trace, 'threads_activated', []))}
            Duration: {getattr(spacetime.trace, 'duration_ms', 0)}ms

            Available panel types: TEXT, CODE, GRAPH, CONFIDENCE, TIMELINE, METRIC, REASONING, SOURCES

            Output JSON: {{"panel_type": "...", "reason": "...", "content_structure": {{...}}}}
            """,
            strategy=RefinementStrategyType.VERIFY
        )

        # Call LLM
        result = await self.llm_client.generate(prompt["enhanced_prompt"])

        return self._parse_llm_response(result)
```

---

## Phase 3: Advanced Renderers

### 3.1 React Component Renderer

**Goal**: Generate React component props instead of static HTML.

```python
class ReactRenderer(JennyRendererBase):
    """Render Jenny panels as React component props."""

    @property
    def supported_targets(self) -> List[RenderTarget]:
        return [RenderTarget.REACT]

    def render(self, spec: JennySpec) -> str:
        """Generate React component JSON props."""
        return json.dumps({
            "component": f"Jenny{spec.panel_type.value.title()}Panel",
            "props": {
                "id": spec.spec_id,
                "lifecycle": spec.lifecycle.value,
                "content": spec.content,
                "actions": [a.__dict__ for a in spec.actions],
                "position": spec.position,
                "size": spec.size.value,
            }
        })
```

### 3.2 AR Spatial Renderer

**Goal**: Generate spatial overlays for AR glasses.

```python
class ARRenderer(JennyRendererBase):
    """Render Jenny panels as AR spatial overlays."""

    @property
    def supported_targets(self) -> List[RenderTarget]:
        return [RenderTarget.AR]

    def render(self, spec: JennySpec) -> str:
        """Generate AR overlay specification."""
        return json.dumps({
            "overlay_type": "floating_panel",
            "anchor": "head_locked",  # Or world_locked for persistent panels
            "distance_m": 1.5,  # 1.5m from user
            "size": self._map_size_to_meters(spec.size),
            "content": self._render_ar_content(spec),
            "gestures": ["pinch_to_pin", "swipe_to_dismiss"],
        })
```

---

## Phase 4: Collaborative Panels

### 4.1 Multi-User Shared Panels

**Goal**: Panels that multiple users can see and interact with simultaneously.

**Features**:
- Shared lifecycle across sessions
- Conflict resolution for simultaneous actions
- Real-time synchronization via WebSocket
- User attribution for actions

### 4.2 Panel Templates

**Goal**: User-defined panel templates for domain-specific visualizations.

```python
@dataclass(frozen=True)
class PanelTemplate:
    """User-defined panel template."""
    template_id: str
    name: str
    panel_type: PanelTypeJenny
    layout: Dict[str, Any]
    default_actions: List[JennyAction]
    css_overrides: Optional[str] = None
```

---

## Phase 5: Accessibility & Animation

### 5.1 Full ARIA Support

**Goal**: Screen reader optimization for all panel types.

- Semantic HTML structure
- ARIA live regions for streaming updates
- Keyboard navigation
- Focus management during transitions

### 5.2 Animation System

**Goal**: Smooth transitions between lifecycle states.

```css
/* Lifecycle animations */
.jenny-panel[data-lifecycle="nascent"] {
    animation: spawn-in 300ms ease-out;
}

.jenny-panel[data-lifecycle="dissolving"] {
    animation: dissolve-out 300ms ease-in;
}

@keyframes spawn-in {
    from { opacity: 0; transform: scale(0.8) translateY(20px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes dissolve-out {
    from { opacity: 1; transform: scale(1); }
    to { opacity: 0; transform: scale(0.9); filter: blur(4px); }
}
```

---

## Implementation Timeline

| Phase | Features | Estimated Effort |
|-------|----------|------------------|
| **2.1** | MRF-Enhanced Panel Generation | 2-3 days |
| **2.2** | Thompson Sampling Learning | 1-2 days |
| **2.3** | LLM-Based Compilation | 3-4 days |
| **3.1** | React Renderer | 2-3 days |
| **3.2** | AR Renderer | 4-5 days |
| **4.1** | Multi-User Panels | 1 week |
| **4.2** | Panel Templates | 2-3 days |
| **5.1** | Accessibility | 2-3 days |
| **5.2** | Animation System | 1-2 days |

**Total Phase 2-5**: ~4-5 weeks

---

## Priority Recommendations

### High Priority (Phase 2 - Start Now)

1. **MRF Integration for WHY panels** - Immediate quality improvement
2. **Thompson Sampling for panel selection** - Self-improving system
3. **LLM-based query analysis** - Better intent detection

### Medium Priority (Phase 3)

4. **React Renderer** - Modern web app integration
5. **Accessibility** - WCAG compliance

### Lower Priority (Phase 4-5)

6. **AR Renderer** - Future-proofing
7. **Multi-User Panels** - Enterprise features
8. **Animation System** - Polish

---

## Quick Win: MRF WHY Panel Integration

**Immediate implementation** (can start today):

```python
# In jenny_compiler.py - enhance _generate_why_panel()
async def _generate_why_panel_mrf(self, spec: JennySpec, spacetime: Spacetime) -> Dict[str, Any]:
    """Generate WHY panel content using MRF for better explanations."""

    from HoloLoom.prompting import UnifiedMRF

    mrf = UnifiedMRF()

    # Use ELEGANCE strategy for clear, simple, beautiful explanations
    enhanced = await mrf.refine_prompt(
        original_prompt=f"""
        Explain why this UI was chosen:
        - Panel Type: {spec.panel_type.value}
        - Confidence: {spacetime.confidence}
        - Threads: {len(getattr(spacetime.trace, 'threads_activated', []))}
        - Response length: {len(spacetime.response)} chars
        """,
        strategy=RefinementStrategyType.ELEGANCE,
        context={"audience": "end_user", "tone": "helpful"}
    )

    return {
        "explanation": enhanced["enhanced_prompt"],
        "factors": self._extract_decision_factors(spec, spacetime),
        "alternatives": self._list_alternative_panels(spec.panel_type),
    }
```

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Panel selection accuracy | 75% (heuristic) | 90%+ (learned) | User satisfaction surveys |
| WHY panel helpfulness | N/A | 4.5/5 rating | In-panel feedback |
| Time to first panel | ~150ms | ~100ms | Trace metrics |
| Panel pin rate | 15% | 30% | SpecLedger analytics |
| Session replay usage | N/A | 20% of sessions | SpecLedger analytics |

---

*Last Updated: December 2025*