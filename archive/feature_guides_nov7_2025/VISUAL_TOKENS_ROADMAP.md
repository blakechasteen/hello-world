# Visual Tokens Integration Roadmap

**Date**: January 2025
**Status**: Future Enhancement (Post Phase 1)
**Priority**: High (after Tesseract installation complete)

---

## Vision

Integrate **DeepSeek-OCR native vision tokens** with **YarnGraph structural encoding** for:
- 15x context compression vs raw text
- Lossless YarnGraph reconstruction
- Visual layout preservation
- Voice-editable structured memory

---

## Phase 1: Foundation (Current - CPU Only) ✅

**Goal**: Get voice correction system working with Tesseract

**Tasks**:
- [x] Design schema-aware receipt processing
- [x] Implement voice correction system
- [x] Build web dashboard UI
- [ ] Install Tesseract OCR (in progress)
- [ ] Test with real receipts
- [ ] Validate pattern learning

**Hardware**: CPU-only (current system)
**OCR**: Tesseract (85-95% accuracy, 1 Wh/receipt)
**Compression**: Text-based (no visual tokens yet)

**Expected Completion**: Week 1

---

## Phase 2: Structural Visual Tokens (Next - CPU Only)

**Goal**: Add YarnGraph → structural visual token encoding

**Tasks**:
- [ ] Implement `YarnGraphVisualEncoder` class
- [ ] Implement `YarnGraphVisualDecoder` class
- [ ] Integrate with `SchemaAwareReceiptSpinner`
- [ ] Add compression metrics to demos
- [ ] Benchmark compression ratios (target: 6-10x)

**Architecture**:
```python
# YarnGraph → Structural Visual Token
class YarnGraphVisualEncoder:
    def encode(self, yarn_graph: KG) -> YarnVisualToken:
        # Node embeddings: (N, 384)
        # Edge embeddings: (E, 384)
        # Adjacency matrix: (N, N)
        # Metadata: JSON (for perfect reconstruction)
        pass
```

**Hardware**: CPU-only (works on current system)
**Compression**: 6-10x vs text
**Files**:
- `HoloLoom/embedding/yarn_visual_encoder.py`
- `HoloLoom/embedding/yarn_visual_decoder.py`
- Update `SchemaAwareReceiptSpinner`

**Expected Completion**: Week 2

---

## Phase 3: DeepSeek-OCR Integration (Future - Requires GPU)

**Goal**: Add native vision tokens from DeepSeek-OCR

**Prerequisites**:
- NVIDIA GPU (8GB+ VRAM)
- CUDA Toolkit 11.8+
- PyTorch with CUDA support
- vLLM 0.8.5+

**Tasks**:
- [ ] Upgrade hardware (GPU)
- [ ] Install CUDA toolkit
- [ ] Reinstall PyTorch with CUDA
- [ ] Install DeepSeek-OCR
- [ ] Implement `DeepSeekVisualYarnSpinner`
- [ ] Preserve native vision tokens in MemoryShard
- [ ] Benchmark total compression (target: 15x)

**Architecture**:
```python
# Image → DeepSeek vision tokens + YarnGraph structural tokens
class DeepSeekVisualYarnSpinner(SchemaAwareReceiptSpinner):
    async def spin(self, image_path):
        # 1. Extract with DeepSeek (preserve vision tokens)
        ocr_result = await deepseek.extract(image_path, return_vision_tokens=True)

        # 2. Build YarnGraph (canonical structure)
        yarn_graph = transform_to_graph(ocr_result.text)

        # 3. Encode structural tokens
        structural_tokens = yarn_encoder.encode(yarn_graph)

        # 4. Store BOTH
        return MemoryShard(
            vision_tokens=ocr_result.vision_tokens,      # 170 tokens (layout)
            structural_tokens=structural_tokens,         # 73 tokens (structure)
            yarn_graph=yarn_graph                        # Editable
        )
```

**Hardware**: GPU required (8GB+ VRAM)
**OCR**: DeepSeek (95-99% accuracy, ~10 Wh/receipt)
**Compression**: 15x total (2.6x vision + 6x structural)

**Files**:
- `HoloLoom/spinningWheel/deepseek_visual_yarn_spinner.py`
- Update OCR backend factory
- Update web UI to show vision tokens

**Expected Completion**: After GPU upgrade (TBD)

---

## Phase 4: Hybrid Auto-Selection (Production)

**Goal**: Smart backend selection based on hardware and receipt complexity

**Tasks**:
- [ ] Implement `HybridReceiptExtractor`
- [ ] Add complexity analysis heuristics
- [ ] Auto-select Tesseract (CPU) vs DeepSeek (GPU)
- [ ] Add fallback logic (Tesseract → DeepSeek for low confidence)
- [ ] Performance monitoring and metrics

**Architecture**:
```python
class HybridReceiptExtractor:
    async def extract(self, image_path):
        # 1. Analyze complexity
        complexity = analyze_complexity(image_path)

        # 2. Choose backend
        if complexity < 0.3 and cpu_only:
            backend = TesseractOCRBackend()
        elif complexity < 0.7 and has_gpu:
            backend = TesseractOCRBackend()  # Try fast path
            if result.confidence < 0.7:
                backend = DeepSeekOCRBackend()  # Fallback
        else:
            backend = DeepSeekOCRBackend()  # Complex, use best

        # 3. Always add structural tokens
        result = await backend.extract(image_path)
        yarn_graph = build_graph(result)
        structural_tokens = encode(yarn_graph)
```

**Decision Matrix**:

| Receipt Type | Backend | Reason |
|--------------|---------|--------|
| Simple, clear | Tesseract | Fast, efficient (1 Wh) |
| Handwritten | DeepSeek | Better accuracy (95-99%) |
| Low confidence | DeepSeek | Fallback for errors |
| Complex layout | DeepSeek | Preserves spatial structure |

**Expected Completion**: After Phase 3

---

## Phase 5: Visual Query & Composition (Advanced)

**Goal**: Query receipts by visual similarity and compose tokens

**Tasks**:
- [ ] Implement visual similarity search
- [ ] Add "find receipts like this" query
- [ ] Compositional token merging (graph union)
- [ ] Visual thumbnails in voice UI
- [ ] Image-to-image relationship graphs

**Features**:

### Visual Similarity Search
```python
# Query by visual appearance
results = await find_similar_receipts(
    reference_image="receipt.jpg",
    k=5,
    use_vision_tokens=True
)
# Returns receipts with similar layout/merchant
```

### Compositional Reasoning
```python
# Merge multiple receipts
receipt1_token = encode(receipt1_graph)
receipt2_token = encode(receipt2_graph)

merged_graph = receipt1_graph + receipt2_graph  # Graph union
merged_token = encode(merged_graph)

# Query merged context
total = query_graph(merged_graph, "total spending at Whole Foods?")
```

### Visual Thumbnails in UI
```python
# Voice correction with visual context
await websocket.send_json({
    'type': 'correction_context',
    'thumbnail': render_visual_token(vision_tokens),
    'structure': render_graph_diagram(yarn_graph)
})
```

**Expected Completion**: Month 2+

---

## Hardware Requirements by Phase

| Phase | CPU | RAM | GPU | VRAM | Storage |
|-------|-----|-----|-----|------|---------|
| **1: Foundation** | ✅ Intel | 8GB+ | ❌ None | - | 10GB |
| **2: Structural Tokens** | ✅ Intel | 8GB+ | ❌ None | - | 20GB |
| **3: DeepSeek-OCR** | ✅ Intel | 16GB+ | ✅ NVIDIA | 8GB+ | 50GB |
| **4: Hybrid** | ✅ Intel | 16GB+ | ✅ NVIDIA | 8GB+ | 50GB |
| **5: Advanced** | ✅ Intel | 16GB+ | ✅ NVIDIA | 8GB+ | 100GB |

**Your Current System**: Phase 1-2 ready (CPU-only)
**GPU Upgrade Needed**: Phase 3+ (NVIDIA GPU with 8GB+ VRAM)

---

## Performance Targets

### Phase 1: Foundation (Tesseract)
- Latency: ~250ms per receipt
- Energy: 1 Wh per receipt
- Accuracy: 85-95%
- Compression: None (text output)
- **Status**: Achievable on current hardware ✅

### Phase 2: Structural Tokens (CPU)
- Latency: ~300ms per receipt (+50ms for encoding)
- Energy: 1.05 Wh per receipt (+0.05 for encoding)
- Accuracy: 85-95% (same as Tesseract)
- Compression: 6-10x vs text
- **Status**: Achievable on current hardware ✅

### Phase 3: DeepSeek + Structural (GPU)
- Latency: ~350ms per receipt (300ms OCR + 50ms encoding)
- Energy: 10.05 Wh per receipt (10 + 0.05)
- Accuracy: 95-99% (DeepSeek)
- Compression: 15x total (2.6x vision + 6x structural)
- **Status**: Requires GPU upgrade ⏳

### Phase 4: Hybrid (CPU + GPU)
- Latency: 250-350ms (auto-select based on complexity)
- Energy: 1-10 Wh (90% Tesseract, 10% DeepSeek)
- Accuracy: 90-99% (adaptive)
- Compression: 8-15x (weighted average)
- **Status**: Requires GPU upgrade ⏳

---

## Cost Analysis

### Phase 1-2: CPU-Only (Current Hardware)
- **Hardware**: $0 (existing computer)
- **Energy**: 1 Wh × 1000 receipts/month = 1 kWh/month = $0.15/month
- **Software**: $0 (Tesseract open-source)
- **Total**: **$0.15/month** ✅

### Phase 3-5: GPU-Enabled
- **Hardware**: $400-800 (RTX 3060/4060 with 8-12GB VRAM, one-time)
- **Energy**: 10 Wh × 1000 receipts/month = 10 kWh/month = $1.50/month
- **Software**: $0 (DeepSeek-OCR open-source)
- **Total**: **$400-800 one-time + $1.50/month** ⏳

### ROI Calculation

**Benefits of GPU upgrade**:
- 10-14% better accuracy (85-95% → 95-99%)
- 15x compression (vs 6x CPU-only)
- Visual layout preservation
- Handles complex/handwritten receipts

**Break-even**:
- If processing >50k receipts, accuracy improvement pays for GPU
- If building a product, 15x compression = better UX = worth investment
- If energy matters, CPU-only is 10x more efficient (stick with Phase 2)

**Recommendation**: Start with Phase 1-2 (CPU-only), evaluate need for GPU after 3 months

---

## Success Metrics

### Phase 1: Foundation
- ✅ Voice correction system working
- ✅ Pattern learning with >60% confidence
- ✅ Web UI real-time updates
- ✅ >80% extraction accuracy (Tesseract)

### Phase 2: Structural Tokens
- ✅ 6-10x compression vs text
- ✅ Lossless YarnGraph reconstruction
- ✅ <50ms encoding latency
- ✅ Compositional token merging working

### Phase 3: DeepSeek Integration
- ✅ 95-99% extraction accuracy
- ✅ 15x total compression
- ✅ Vision tokens preserved
- ✅ <350ms total latency

### Phase 4: Hybrid Production
- ✅ Auto-backend selection working
- ✅ 90%+ receipts use fast path (Tesseract)
- ✅ 10% fallback to DeepSeek for complex cases
- ✅ <10 Wh average energy per receipt

---

## Documentation Updates

### Files to Create

**Phase 2**:
- `HoloLoom/embedding/README_VISUAL_TOKENS.md` - Visual token encoding guide
- `demos/demo_visual_token_compression.py` - Compression benchmarks
- Update `SCHEMA_AWARE_FOUNDATION.md` with visual token section

**Phase 3**:
- `HoloLoom/spinningWheel/README_DEEPSEEK_VISUAL.md` - DeepSeek integration guide
- `demos/demo_deepseek_visual_tokens.py` - End-to-end demo
- Update `INSTALL_OCR_BACKENDS.md` with DeepSeek vision token path

**Phase 4**:
- `HoloLoom/spinningWheel/README_HYBRID_OCR.md` - Hybrid extractor guide
- `demos/demo_hybrid_extraction.py` - Auto-selection demo
- Update `COMPLETE_IMPLEMENTATION_SUMMARY.md` with hybrid system

### Files to Update

**Phase 2**:
- `YARNGRAPH_VISUAL_TOKENS.md` - Add implementation status
- `VLM_VS_STRUCTURAL_ANALYSIS.md` - Update with Phase 2 results
- `QUICK_START_GUIDE.md` - Add structural token section

**Phase 3**:
- `DEEPSEEK_OCR_INTEGRATION.md` - Update with implementation
- `INSTALL_OCR_BACKENDS.md` - Add GPU setup guide
- `VISUAL_TOKENS_PROPOSAL.md` - Mark as implemented

**Phase 4**:
- `CLAUDE.md` - Update with hybrid OCR guide
- `README.md` - Add visual tokens to feature list
- All demo files - Update to use hybrid extractor

---

## Integration with Existing Systems

### Voice Correction System
```python
# Phase 1: Text-based corrections
"the merchant is Whole Foods Market"

# Phase 2: Corrections update YarnGraph, regenerate structural tokens
yarn_graph.update_node('merchant', name='Whole Foods Market')
structural_token = encode(yarn_graph)  # Instant regeneration

# Phase 3: Corrections keep original vision tokens as reference
display(vision_tokens)  # Show thumbnail
apply_correction(yarn_graph)  # Edit structure
```

### Web Dashboard UI
```python
# Phase 1: Text display
display_text(extracted_text)

# Phase 2: Graph visualization
display_graph_diagram(yarn_graph)

# Phase 3: Visual thumbnail + graph
display_thumbnail(vision_tokens)  # Original receipt
display_graph_diagram(yarn_graph)  # Structured view
```

### Memory System (YarnGraph)
```python
# Phase 1: Text-based memory shards
shard = MemoryShard(content=text)

# Phase 2: Add structural tokens
shard = MemoryShard(
    content=text,
    metadata={'structural_tokens': encode(yarn_graph)}
)

# Phase 3: Add native vision tokens
shard = MemoryShard(
    content=text,
    metadata={
        'vision_tokens': deepseek_vision_tokens,
        'structural_tokens': encode(yarn_graph)
    }
)
```

---

## Timeline

| Phase | Duration | Start | End | Dependencies |
|-------|----------|-------|-----|--------------|
| **1: Foundation** | 1 week | Now | Week 1 | Install Tesseract |
| **2: Structural Tokens** | 1 week | Week 2 | Week 3 | Phase 1 complete |
| **3: DeepSeek-OCR** | 2 weeks | TBD | TBD | GPU hardware |
| **4: Hybrid System** | 1 week | TBD | TBD | Phase 3 complete |
| **5: Advanced Features** | 4 weeks | TBD | TBD | Phase 4 complete |

**Critical Path**: GPU acquisition blocks Phase 3+

**Alternative Path**: Skip Phase 3, stay on Phase 2 indefinitely if:
- Budget constraints (no GPU purchase)
- Energy efficiency priority (CPU 10x better)
- Accuracy sufficient (85-95% good enough)

---

## Decision Points

### After Phase 1 (Week 1)
**Question**: Is Tesseract accuracy good enough for our receipts?

**If YES**: Continue to Phase 2 (structural tokens)
**If NO**: Consider cloud VLM API as interim solution ($0.02/receipt)

### After Phase 2 (Week 3)
**Question**: Is 6x compression + CPU efficiency sufficient?

**If YES**: Skip Phase 3, productionize Phase 2
**If NO**: Plan GPU upgrade, proceed to Phase 3

### After Phase 3 (TBD)
**Question**: Is DeepSeek accuracy worth 10x energy cost?

**If YES**: Deploy Phase 4 hybrid (90% Tesseract, 10% DeepSeek)
**If NO**: Revert to Phase 2 (CPU-only)

---

## Risk Mitigation

### Risk 1: GPU Upgrade Too Expensive
**Mitigation**: Phase 2 (structural tokens) works great on CPU, provides 6x compression without GPU

### Risk 2: DeepSeek-OCR Installation Issues
**Mitigation**: Hybrid system has Tesseract fallback, never blocks on DeepSeek

### Risk 3: Energy Cost Too High
**Mitigation**: Smart backend selection routes 90% to Tesseract (1 Wh), only 10% to DeepSeek (10 Wh)

### Risk 4: Accuracy Not Good Enough
**Mitigation**: Voice correction system learns from every correction, improves over time regardless of OCR backend

---

## References

**Core Documents**:
- `YARNGRAPH_VISUAL_TOKENS.md` - Architectural design
- `VLM_VS_STRUCTURAL_ANALYSIS.md` - Energy efficiency analysis
- `DEEPSEEK_OCR_INTEGRATION.md` - DeepSeek integration plan
- `VISUAL_TOKENS_PROPOSAL.md` - Original proposal

**Papers**:
- DeepSeek-OCR Paper (https://arxiv.org/abs/deepseek-ocr)
- Matryoshka Representation Learning (https://arxiv.org/abs/2205.13147)

**Code**:
- `HoloLoom/spinningWheel/schema_aware_receipt_spinner.py` - Foundation
- `HoloLoom/memory/graph.py` - YarnGraph implementation
- `HoloLoom/embedding/spectral.py` - Matryoshka embeddings

---

## Next Actions

**Immediate** (this week):
1. ✅ Install Tesseract (see `install_tesseract.bat`)
2. ✅ Run demos with real OCR
3. ✅ Validate voice correction system
4. ✅ Test pattern learning

**Short-term** (Week 2):
1. Implement `YarnGraphVisualEncoder`
2. Integrate with `SchemaAwareReceiptSpinner`
3. Benchmark compression ratios
4. Add unit tests

**Long-term** (Month 2+):
1. Evaluate GPU upgrade need
2. Install DeepSeek-OCR (if GPU acquired)
3. Implement hybrid system
4. Productionize

---

**Status**: Roadmap complete, ready for phased implementation 🚀

**Current Phase**: Phase 1 (Foundation with Tesseract)
**Next Milestone**: Install Tesseract, test with real receipts
