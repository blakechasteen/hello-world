# Semantic State → Policy Integration Plan

**Status**: Week 1-2 of 4-week roadmap
**Goal**: Enable policy to make semantic-aware decisions

---

## 🎯 Integration Architecture

```
Query Text
    ↓
[Matryoshka Streaming] → MatryoshkaSnapshot (244D)
    ↓
[SemanticState] → Compact 8D feature vector
    ↓
[Policy.decide(semantic_state)] → Tool selection + reasoning
```

---

## ✅ Completed (Phase 1)

### 1. SemanticState Foundation
**File**: `HoloLoom/semantic_calculus/semantic_state.py` (507 lines)

**Key Classes**:
- `SemanticState`: Converts 244D → 8D for policy
- `SemanticToolSelector`: Maps dimensions → tools

**Features**:
- ✅ Momentum computation (alignment across scales)
- ✅ Complexity computation (diversity of dimensions)
- ✅ Dominant dimension extraction (top 5)
- ✅ Topic shift detection (threshold-based)
- ✅ 8D feature vector for policy

---

## 🚧 In Progress (Phase 2)

### 2. WeavingOrchestrator Integration

**Approach**: Minimal invasive integration

**Integration Points**:

#### A. Feature Extraction (Step 4)
Location: `weaving_orchestrator.py:~1279-1350`

```python
async def step4_feature_extraction():
    # ... existing feature extraction ...

    # NEW: Extract semantic state if semantic_calculus available
    semantic_state = None
    if resonance_shed.semantic_calculus:
        # Use existing semantic analysis from ResonanceShed
        semantic_proj = features.metadata.get('semantic_projection')
        if semantic_proj:
            # Convert to SemanticState
            from HoloLoom.semantic_calculus.semantic_state import SemanticState
            semantic_state = SemanticState.from_projection(
                semantic_proj,
                spectrum=self.semantic_spectrum
            )

            # Store in features metadata for policy
            features.metadata['semantic_state'] = semantic_state

    return features
```

#### B. Policy Decision (Step 7)
Location: `weaving_orchestrator.py:~1533`

```python
# Extract semantic state from features
semantic_state = features.metadata.get('semantic_state')

# Pass to policy
action_plan = await asyncio.wait_for(
    policy.decide(
        features=features,
        context=context,
        semantic_state=semantic_state  # NEW!
    ),
    timeout=0.2
)
```

---

## ⏳ Pending (Phase 3)

### 3. NeuralCore Enhancement

**File**: `HoloLoom/policy/unified.py`

**Changes**:

#### A. Add semantic_state parameter
```python
class NeuralCore(nn.Module):
    def __init__(self, ..., use_semantic_state=True):
        super().__init__()
        # ... existing init ...

        if use_semantic_state:
            self.semantic_feature_dim = 8
            self.semantic_mlp = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )

    def forward(self, motifs, embeddings, context, semantic_state=None):
        # ... existing processing ...

        # NEW: Process semantic features
        if semantic_state is not None:
            semantic_vec = torch.from_numpy(
                semantic_state.to_feature_vector()
            ).float()

            semantic_encoded = self.semantic_mlp(semantic_vec)

            # Fuse with existing features
            combined = torch.cat([core_features, semantic_encoded], dim=-1)
        else:
            combined = core_features

        # ... rest of forward pass ...
```

#### B. Update decide() method
```python
async def decide(
    self,
    features: Features,
    context: Context,
    semantic_state: Optional['SemanticState'] = None  # NEW!
) -> ActionPlan:
    # ... existing processing ...

    # Pass semantic_state to NeuralCore.forward()
    tool_logits = self.core.forward(
        motifs_tensor,
        embeddings_tensor,
        context_tensor,
        semantic_state=semantic_state  # NEW!
    )

    # ... rest of decision logic ...
```

---

## 🎬 Demo (Phase 4)

### 4. Topic Shift Detection Demo

**File**: `demos/demo_semantic_state.py`

**Test Scenarios**:

```python
# Scenario 1: Focused conversation (high momentum)
queries = [
    "What is Thompson Sampling?",
    "How does it balance exploration and exploitation?",
    "Can you give me an example?"
]
# Expected: momentum=0.8, complexity=0.4, NO topic shift

# Scenario 2: Topic shift (low momentum)
queries = [
    "What is Thompson Sampling?",
    "Let's talk about cooking instead.",  # SHIFT!
    "How do I make pasta?"
]
# Expected: momentum=0.2, complexity=0.7, topic_shift_detected=True

# Scenario 3: Complex multi-topic (high complexity)
queries = [
    "Explain reinforcement learning, neural networks, "
    "Bayesian optimization, and genetic algorithms."
]
# Expected: momentum=0.5, complexity=0.9
```

**Visualization**:
```
Query 1: What is Thompson Sampling?
  Momentum: ████████░░ 0.85
  Complexity: ████░░░░░░ 0.42
  Dominant: [Exploration, Learning, Optimization]
  Status: ✓ Focused

Query 2: Let's talk about cooking instead
  Momentum: ██░░░░░░░░ 0.18 ⚠️ LOW
  Complexity: ███████░░░ 0.73
  Dominant: [Transformation, Creation, Action]
  Status: ⚠️ TOPIC SHIFT DETECTED
  Suggestion: Branch new thread?
```

---

## 📊 Success Metrics

### Phase 2 (Orchestrator Integration)
- [ ] Semantic state extracted from query
- [ ] Stored in features.metadata
- [ ] Passed to policy.decide()
- [ ] No breaking changes to existing code
- [ ] Graceful fallback if semantic_calculus disabled

### Phase 3 (Policy Enhancement)
- [ ] NeuralCore processes semantic features
- [ ] 8D semantic vector fused with existing features
- [ ] Tool selection influenced by semantic state
- [ ] Tests pass with semantic_state=None (backward compat)

### Phase 4 (Demo & Testing)
- [ ] Demo shows topic shift detection working
- [ ] Demo shows momentum/complexity tracking
- [ ] Demo shows tool suggestions based on semantics
- [ ] Integration test with full pipeline

---

## 🚨 Risk Mitigation

### Graceful Degradation
```python
# If semantic_calculus not available
if not semantic_state:
    # Policy uses existing features only
    # No semantic-aware decisions
    # System continues to work
```

### Backward Compatibility
```python
# Old code without semantic_state parameter
policy.decide(features, context)  # Still works!

# New code with semantic_state
policy.decide(features, context, semantic_state)  # Enhanced!
```

### Performance Impact
- Semantic state extraction: <5ms (one-time per query)
- Policy MLP processing: <1ms (minimal overhead)
- **Total overhead**: <6ms per query (acceptable)

---

## 🗓️ Timeline

| Phase | Tasks | Duration | Status |
|-------|-------|----------|--------|
| **Phase 1** | SemanticState foundation | 2 days | ✅ Complete |
| **Phase 2** | Orchestrator integration | 1 day | 🚧 In Progress |
| **Phase 3** | Policy enhancement | 2 days | ⏳ Pending |
| **Phase 4** | Demo & testing | 1 day | ⏳ Pending |
| **Total** | | **6 days** | **Week 1-2** |

---

## 🎯 Next Actions

**Immediate (Next 2 hours)**:
1. ✅ Create SemanticState module
2. 🚧 Wire into WeavingOrchestrator (in progress)
3. ⏳ Test orchestrator integration
4. ⏳ Enhance NeuralCore

**Tomorrow**:
5. Create demo script
6. Test topic shift detection
7. Validate tool suggestions

**This Week**:
8. Integration tests
9. Performance benchmarks
10. Documentation updates

---

## 📚 References

- **Roadmap**: `docs/architecture/PRIORITY_ROADMAP.md`
- **SemanticState**: `HoloLoom/semantic_calculus/semantic_state.py`
- **Orchestrator**: `HoloLoom/weaving_orchestrator.py`
- **Policy**: `HoloLoom/policy/unified.py`
- **Matryoshka Streaming**: `HoloLoom/semantic_calculus/matryoshka_streaming.py`

---

**Last Updated**: November 7, 2025
**Status**: Phase 2 in progress
**Next**: Complete orchestrator integration
