# Input Processing vs Spinning Wheel - Architecture Clarification

**Question**: Why do we have both `HoloLoom/input/` and `HoloLoom/spinning_wheel/`?

**Answer**: They serve different architectural layers and are **complementary**, not redundant.

---

## Architecture Layers

```
User Input
    ↓
┌─────────────────────────────────────┐
│ HoloLoom/spinning_wheel/             │  🎯 Application Layer
│ - High-level spinners                │
│ - Converts to MemoryShards           │
│ - Adds context & enrichment          │
│ - User-facing API                    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ HoloLoom/input/                      │  ⚙️ Infrastructure Layer
│ - Low-level processors               │
│ - Modality detection & routing       │
│ - Feature extraction                 │
│ - Multi-modal fusion                 │
└──────────────┬──────────────────────┘
               ↓
        ProcessedInput
               ↓
          Orchestrator
```

---

## HoloLoom/input/ (Infrastructure Layer)

**Purpose**: Low-level input processing infrastructure

**Responsibilities**:
- Automatic modality detection (text/image/audio/structured)
- Feature extraction from raw inputs
- Multi-modal fusion
- Protocol definitions for input processing

**Key Components**:
- `InputRouter` - Detects input type and routes to correct processor
- `TextProcessor`, `ImageProcessor`, `AudioProcessor`, `StructuredDataProcessor`
- `MultiModalFusion` - Combines features from multiple modalities
- `protocol.py` - Defines `ProcessedInput`, `ModalityType`, etc.

**Output**: `ProcessedInput` objects with extracted features

**Example**:
```python
from HoloLoom.input import InputRouter

router = InputRouter()
processed = await router.process(raw_input)
# Returns ProcessedInput with features, modality, metadata
```

---

## HoloLoom/spinning_wheel/ (Application Layer)

**Purpose**: High-level input adapters for HoloLoom orchestrator

**Responsibilities**:
- Convert various input types → `MemoryShard` objects
- Add domain-specific context and enrichment
- Provide user-friendly APIs
- Handle chunking, batching, crawling

**Key Components**:
- `TextSpinner`, `AudioSpinner`, `YouTubeSpinner`, `CodeSpinner`, `WebsiteSpinner`
- `RecursiveCrawler` - Web crawling with importance gating
- `BrowserHistoryReader` - Browser history ingestion
- `spin()` - Universal "ruthlessly elegant" API

**Output**: `List[MemoryShard]` ready for orchestrator

**Example**:
```python
from HoloLoom.spinning_wheel import YouTubeSpinner

spinner = YouTubeSpinner()
shards = await spinner.spin({'url': 'VIDEO_ID'})
# Returns MemoryShards with text, metadata, timestamps
```

---

## Relationship

**spinning_wheel USES input** (layered architecture):

```python
# Inside multimodal_spinner.py
from HoloLoom.input import InputRouter, MultiModalFusion

class MultiModalSpinner:
    def __init__(self):
        self.router = InputRouter()  # Uses infrastructure layer
        self.fusion = MultiModalFusion()

    async def spin(self, raw_data) -> List[MemoryShard]:
        # 1. Use input layer for processing
        processed = await self.router.process(raw_data)

        # 2. Convert to MemoryShards (application layer)
        shards = self._create_shards(processed)

        return shards
```

---

## When to Use Which?

### Use `HoloLoom/input/` when:
- Building a new modality processor
- Need low-level feature extraction
- Implementing custom multi-modal fusion
- Extending the input protocol

### Use `HoloLoom/spinning_wheel/` when:
- Ingesting data into HoloLoom
- Need MemoryShards for orchestrator
- Want high-level API (`spin()`)
- Processing YouTube, websites, code, etc.

---

## Key Differences

| Aspect | input/ | spinning_wheel/ |
|--------|--------|-----------------|
| **Layer** | Infrastructure | Application |
| **Output** | ProcessedInput | MemoryShard |
| **Focus** | Feature extraction | Memory creation |
| **Users** | Spinner developers | HoloLoom users |
| **Abstraction** | Low-level | High-level |
| **Examples** | TextProcessor | YouTubeSpinner |

---

## Design Benefits

1. **Separation of Concerns**
   - Input processing logic separate from memory creation
   - Clean architectural boundaries

2. **Reusability**
   - Input processors can be used independently
   - Spinners share common infrastructure

3. **Extensibility**
   - Add new processors without touching spinners
   - Add new spinners using existing processors

4. **Testability**
   - Test feature extraction independently
   - Test memory creation separately

---

## Example: Full Pipeline

```python
# Low-level (if you need fine control)
from HoloLoom.input import InputRouter
router = InputRouter()
processed = await router.process(raw_text)
# Use processed.features for custom logic

# High-level (typical usage)
from HoloLoom.spinning_wheel import spin
shards = await spin(raw_text)  # Handles everything
# Use shards directly with orchestrator
```

---

## Conclusion

**Both directories are needed and complementary**:

- `input/` = **Infrastructure** (how to process inputs)
- `spinning_wheel/` = **Application** (how to create memories)

This is **proper layered architecture**, not duplication!

Think of it like:
- `input/` is the HTTP library
- `spinning_wheel/` is the REST API framework

Both are essential, serving different purposes.

---

**Status**: ✅ Architecture is correct and well-designed
**Action**: No consolidation needed - clarify in documentation
