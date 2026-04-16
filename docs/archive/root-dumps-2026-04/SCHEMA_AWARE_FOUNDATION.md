# Schema-Aware Foundation - Elegant & Extensible Design

**Date**: January 2025
**Status**: Design Phase
**Goal**: Build rock-solid foundation for schema-aware spinners with voice commands + self-tuning

---

## The Vision

> **"Wool becomes yarn through conversation, not configuration."**

Users should be able to:
1. Drop any file (receipt, note, invoice)
2. Say "extract this as expenses" (voice command)
3. System learns patterns automatically (self-tuning)
4. Knowledge graph updates in real-time
5. Corrections improve future extractions

---

## Core Philosophy

### 1. Zero Configuration
- No schema files to write
- No field mappings to define
- System learns from examples

### 2. Voice-First
- Natural language commands
- Real-time corrections
- Conversational refinement

### 3. Self-Tuning
- Learns from corrections
- Adapts to user patterns
- Improves over time

### 4. Provenance Always
- Every transformation tracked
- Every decision explained
- Corrections propagate

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│             LAYER 7: VOICE INTERFACE                    │
│  "Extract this as expenses"                             │
│  "The merchant is actually Whole Foods"                 │
│  "Always map 'amt' to 'total'"                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Intent → Command
                     ▼
┌─────────────────────────────────────────────────────────┐
│          LAYER 6: SELF-TUNING ENGINE                    │
│  - Pattern learning (corrections → rules)               │
│  - Confidence tracking (what works?)                    │
│  - Schema evolution (add fields dynamically)            │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Learned Patterns
                     ▼
┌─────────────────────────────────────────────────────────┐
│          LAYER 5: SCHEMA REGISTRY (RAG)                 │
│  - Semantic schema matching                             │
│  - Field mapping suggestions                            │
│  - Validation + constraints                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Schema + Mappings
                     ▼
┌─────────────────────────────────────────────────────────┐
│     LAYER 4: TRANSFORMATION ENGINE (Protocol-Based)     │
│  - Extract structured data                              │
│  - Create graph nodes/edges                             │
│  - Apply learned transformations                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Structured Data
                     ▼
┌─────────────────────────────────────────────────────────┐
│        LAYER 3: SPECIALIZED SPINNERS (Domain)           │
│  - ReceiptSpinner                                       │
│  - HandwrittenSpinner                                   │
│  - InvoiceSpinner, FormSpinner, etc.                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Parsed Data
                     ▼
┌─────────────────────────────────────────────────────────┐
│         LAYER 2: OCR PROTOCOL (Multi-Backend)           │
│  - DeepSeek (excellent)                                 │
│  - Tesseract (good)                                     │
│  - Fallback (poor)                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Raw Text
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LAYER 1: RAW INPUT (Wool)                  │
│  - Images, PDFs, Audio, etc.                            │
└─────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Voice Command Interface (NEW)

**Protocol**:
```python
class VoiceCommandProtocol(Protocol):
    """Handle natural language commands for schema operations."""

    async def parse_intent(self, voice_text: str) -> Intent:
        """Parse user intent from voice/text command."""
        ...

    async def execute_correction(self, correction: Correction) -> None:
        """Apply user correction and learn from it."""
        ...

    async def suggest_schema(self, description: str) -> SchemaDefinition:
        """Generate schema from natural language description."""
        ...
```

**Example Commands**:
```python
# Schema selection
"extract this as expenses"
"process as receipt"
"treat as handwritten note with tasks"

# Field corrections
"the merchant is actually Whole Foods"
"total should be 45.99 not 4599"
"map 'amt' field to 'total'"

# Schema evolution
"add a 'tip' field to receipts"
"merchant should also have 'phone number'"
"create a new category called 'medical'"

# Batch operations
"reprocess all receipts from last week"
"apply new schema to everything"
```

### 2. Self-Tuning Engine (NEW)

**Core Mechanism**:
```python
@dataclass
class CorrectionPattern:
    """Learned pattern from user correction."""
    pattern_type: str  # field_mapping, value_fix, schema_addition
    source_pattern: str  # What triggers this
    target_action: str  # What to do
    confidence: float  # How often it works
    usage_count: int  # Times applied
    success_rate: float  # Times it was correct

class SelfTuningEngine:
    """Learn from corrections to improve future extractions."""

    async def learn_from_correction(
        self,
        original_data: Dict,
        corrected_data: Dict,
        context: Dict
    ) -> CorrectionPattern:
        """
        Learn pattern from user correction.

        Example:
        - User corrects "amt" -> "total" 5 times
        - System learns: field_name.contains("amt") -> map to "total"
        - Future receipts with "amt" automatically map to "total"
        """
        ...

    async def apply_learned_patterns(
        self,
        extracted_data: Dict,
        schema_name: str
    ) -> Dict:
        """Apply learned patterns before user sees data."""
        ...

    def get_confidence(self, pattern: CorrectionPattern) -> float:
        """Get confidence that pattern should be applied."""
        ...
```

**Learning Categories**:

1. **Field Mapping Patterns**
   - "amt" always maps to "total"
   - "qty" always maps to "quantity"
   - "desc" always maps to "description"

2. **Value Transformation Patterns**
   - "WH FOODS" -> "Whole Foods"
   - "4599" (when context is $) -> "45.99"
   - "01/15/25" -> "2025-01-15"

3. **Schema Evolution Patterns**
   - User adds "tip" field 3 times -> add to schema permanently
   - User adds "phone" to merchant 5 times -> add to schema
   - User creates "medical" category -> add to category enum

4. **Extraction Patterns**
   - Items always between "---" lines
   - Merchant name is always first line
   - Total always follows "Total:"

### 3. Enhanced Schema Registry

**Add Learning Capabilities**:
```python
class EnhancedSchemaRegistry(SchemaRegistry):
    """Schema registry with self-tuning and voice commands."""

    def __init__(self, tuning_engine: SelfTuningEngine):
        super().__init__()
        self.tuning_engine = tuning_engine
        self.correction_history = []

    async def apply_voice_correction(
        self,
        voice_command: str,
        transformation_id: str
    ) -> None:
        """
        Apply correction from voice command.

        Examples:
        - "the merchant is Whole Foods" -> fix merchant field
        - "add tip field" -> evolve schema
        - "this should be category medical" -> add category
        """
        intent = await self.parse_voice_intent(voice_command)

        if intent.type == IntentType.FIELD_CORRECTION:
            # Fix specific field
            await self._apply_field_correction(
                transformation_id,
                intent.field,
                intent.value
            )

        elif intent.type == IntentType.SCHEMA_EVOLUTION:
            # Add field to schema
            await self._evolve_schema(
                intent.schema_name,
                intent.field_definition
            )

        elif intent.type == IntentType.MAPPING_RULE:
            # Learn mapping rule
            await self.tuning_engine.learn_mapping_rule(
                intent.source_field,
                intent.target_field
            )

        # Store for future learning
        self.correction_history.append({
            'command': voice_command,
            'intent': intent,
            'timestamp': time.time(),
            'transformation_id': transformation_id
        })

    async def get_correction_suggestions(
        self,
        transformation: GraphTransformation
    ) -> List[str]:
        """
        Suggest common corrections based on learned patterns.

        Returns voice-friendly suggestions like:
        - "Did you mean Whole Foods instead of WH FOODS?"
        - "Should this be category 'grocery'?"
        - "Total seems high, confirm $459.90?"
        """
        ...
```

### 4. Transformation Provenance (ENHANCED)

**Track Everything**:
```python
@dataclass
class TransformationProvenance:
    """Complete history of a transformation."""
    transformation_id: str
    timestamp: float

    # Input
    source_file: Path
    ocr_backend: str
    ocr_confidence: float
    raw_text: str

    # Processing
    schema_name: str
    schema_version: str
    field_mappings: Dict[str, FieldMapping]
    patterns_applied: List[CorrectionPattern]

    # Output
    nodes_created: List[Dict]
    edges_created: List[KGEdge]
    validation_errors: List[str]

    # Corrections (if any)
    corrections: List[Correction] = field(default_factory=list)
    correction_count: int = 0

    # Learning
    learned_patterns: List[CorrectionPattern] = field(default_factory=list)

    def add_correction(self, correction: Correction) -> None:
        """Add user correction and learn from it."""
        self.corrections.append(correction)
        self.correction_count += 1

        # Extract pattern
        pattern = self._extract_pattern(correction)
        if pattern:
            self.learned_patterns.append(pattern)

    def get_confidence_score(self) -> float:
        """
        Get overall confidence in this transformation.

        Factors:
        - OCR confidence
        - Field mapping confidence
        - Pattern match confidence
        - Validation success
        - Number of corrections needed
        """
        ...
```

### 5. Unified Schema-Aware Spinner (PROTOCOL)

**Base Protocol for All Schema-Aware Spinners**:
```python
class SchemaAwareSpinnerProtocol(Protocol):
    """Protocol for all schema-aware spinners."""

    yarn_graph: KG
    schema_registry: EnhancedSchemaRegistry
    tuning_engine: SelfTuningEngine

    async def spin_with_schema(
        self,
        source: Any,
        schema_hint: Optional[str] = None,  # Optional voice hint
        **kwargs
    ) -> Tuple[SpinResult, GraphTransformation]:
        """
        Process source with schema-aware transformation.

        Args:
            source: Input data
            schema_hint: Optional schema name from voice ("extract as expenses")
            **kwargs: Additional options

        Returns:
            (SpinResult, GraphTransformation) with full provenance
        """
        ...

    async def apply_correction(
        self,
        transformation_id: str,
        voice_command: str
    ) -> None:
        """Apply correction via voice command."""
        ...

    async def suggest_improvements(
        self,
        transformation: GraphTransformation
    ) -> List[str]:
        """Get voice-friendly improvement suggestions."""
        ...

    def get_learned_patterns(self) -> List[CorrectionPattern]:
        """Get patterns learned by this spinner."""
        ...
```

**Concrete Implementation**:
```python
class UnifiedSchemaAwareSpinner(BaseSpinner):
    """
    Unified spinner with voice + self-tuning.

    Combines:
    - Schema-aware transformation
    - Voice command interface
    - Self-tuning from corrections
    - Complete provenance tracking
    """

    def __init__(
        self,
        domain_spinner: BaseSpinner,  # Receipt, Handwritten, etc.
        yarn_graph: KG,
        schema_registry: EnhancedSchemaRegistry,
        tuning_engine: SelfTuningEngine,
        voice_interface: Optional[VoiceCommandInterface] = None
    ):
        self.domain_spinner = domain_spinner
        self.yarn_graph = yarn_graph
        self.schema_registry = schema_registry
        self.tuning_engine = tuning_engine
        self.voice_interface = voice_interface

        # Track all transformations
        self.transformations: Dict[str, TransformationProvenance] = {}

    async def spin_with_schema(
        self,
        source: Any,
        schema_hint: Optional[str] = None,
        voice_command: Optional[str] = None,
        **kwargs
    ) -> Tuple[SpinResult, GraphTransformation]:
        """
        Complete wool->yarn pipeline with learning.

        Pipeline:
        1. Parse voice command (if provided)
        2. Domain spinner extracts structured data
        3. Apply learned patterns (self-tuning)
        4. Find/select schema (RAG or voice hint)
        5. Map fields with learned mappings
        6. Transform to graph nodes/edges
        7. Insert into Yarn Graph
        8. Track provenance
        9. Suggest improvements
        """

        transformation_id = self._generate_id()

        # Step 1: Parse voice command
        intent = None
        if voice_command:
            intent = await self.voice_interface.parse_intent(voice_command)
            if intent.schema_name:
                schema_hint = intent.schema_name

        # Step 2: Domain extraction
        domain_result = await self.domain_spinner.spin(source, **kwargs)

        # Step 3: Apply learned patterns
        for shard in domain_result.shards:
            structured_data = shard.metadata.get('structured_data')
            if structured_data:
                # Apply self-tuning improvements
                improved_data = await self.tuning_engine.apply_learned_patterns(
                    structured_data,
                    schema_hint or 'auto'
                )
                shard.metadata['structured_data'] = improved_data
                shard.metadata['patterns_applied'] = self.tuning_engine.last_patterns_applied

        # Step 4-7: Schema transformation (existing code)
        transformation = await self._transform_with_schema(
            domain_result,
            schema_hint,
            transformation_id
        )

        # Step 8: Track provenance
        provenance = TransformationProvenance(
            transformation_id=transformation_id,
            timestamp=time.time(),
            source_file=Path(source),
            schema_name=transformation.schema_name,
            nodes_created=transformation.nodes_created,
            edges_created=transformation.edges_created,
            patterns_applied=self.tuning_engine.last_patterns_applied,
            # ... full context
        )

        self.transformations[transformation_id] = provenance

        # Step 9: Suggest improvements
        if transformation.validation_errors or transformation.validation_warnings:
            suggestions = await self.suggest_improvements(transformation)
            transformation.metadata['suggestions'] = suggestions

        return domain_result, transformation

    async def apply_voice_correction(
        self,
        transformation_id: str,
        voice_command: str
    ) -> None:
        """
        Apply correction via voice command.

        Examples:
        - "the merchant is Whole Foods"
        - "total should be 45.99"
        - "add category medical"
        """

        provenance = self.transformations.get(transformation_id)
        if not provenance:
            raise ValueError(f"Transformation {transformation_id} not found")

        # Parse intent
        intent = await self.voice_interface.parse_intent(voice_command)

        # Create correction
        correction = Correction(
            transformation_id=transformation_id,
            voice_command=voice_command,
            intent=intent,
            timestamp=time.time()
        )

        # Apply correction to graph
        await self._apply_correction_to_graph(correction, provenance)

        # Learn pattern
        pattern = await self.tuning_engine.learn_from_correction(
            provenance.original_data,
            correction.corrected_data,
            context={'schema': provenance.schema_name}
        )

        # Update provenance
        provenance.add_correction(correction)

        # Re-apply to similar transformations?
        if pattern.confidence > 0.8:
            await self._propagate_correction(pattern)

    async def _propagate_correction(self, pattern: CorrectionPattern) -> None:
        """Apply learned pattern to similar past transformations."""
        for trans_id, provenance in self.transformations.items():
            if pattern.applies_to(provenance):
                # Auto-correct similar mistakes
                await self._apply_pattern_to_transformation(pattern, provenance)
```

---

## Usage Examples

### Example 1: Zero Config + Voice

```python
from HoloLoom.spinningWheel import UnifiedSchemaAwareSpinner
from HoloLoom.memory.graph import KG

# Setup (once)
spinner = UnifiedSchemaAwareSpinner(
    domain_spinner=ReceiptSpinner(),
    yarn_graph=KG(),
    schema_registry=EnhancedSchemaRegistry(tuning_engine=SelfTuningEngine()),
    tuning_engine=SelfTuningEngine(),
    voice_interface=VoiceCommandInterface()
)

# Process with voice command
result, transformation = await spinner.spin_with_schema(
    "receipt.jpg",
    voice_command="extract this as expenses"
)

# System automatically:
# - Detects it's a receipt
# - Finds 'expenses' schema
# - Extracts structured data
# - Creates graph nodes/edges
# - Returns suggestions if unsure

print(f"Nodes created: {transformation.node_count}")
print(f"Confidence: {transformation.confidence:.2f}")

# Correct via voice
await spinner.apply_voice_correction(
    transformation.transformation_id,
    "the merchant is actually Whole Foods Market"
)

# System learns: "WH FOODS" -> "Whole Foods Market"
# Future receipts with "WH FOODS" automatically corrected!
```

### Example 2: Self-Tuning Over Time

```python
# Day 1: Process receipts (some errors)
for receipt in receipts_day1:
    result, trans = await spinner.spin_with_schema(receipt)

    # User corrects: "amt" should be "total"
    await spinner.apply_voice_correction(trans.transformation_id, "amt means total")

# Day 2: Process more receipts (automatic improvement!)
for receipt in receipts_day2:
    result, trans = await spinner.spin_with_schema(receipt)
    # "amt" automatically mapped to "total" now!
    # No correction needed

# Day 7: System is highly accurate
patterns = spinner.get_learned_patterns()
print(f"Learned {len(patterns)} patterns")
for pattern in patterns[:5]:
    print(f"  {pattern.source_pattern} -> {pattern.target_action}")
    print(f"    Confidence: {pattern.confidence:.2f}")
    print(f"    Used: {pattern.usage_count} times")
```

### Example 3: Schema Evolution

```python
# User notices receipts have tips
result, trans = await spinner.spin_with_schema("receipt_with_tip.jpg")

# Add tip field via voice
await spinner.apply_voice_correction(
    trans.transformation_id,
    "add a tip field to receipts, value is 8.50"
)

# System evolves schema:
# - Adds 'tip' field to expense schema
# - Re-extracts data with new field
# - Updates graph node
# - Future receipts include tip automatically!

schema = spinner.schema_registry.get_schema("expenses")
print(schema.nodes['Transaction']['properties'])
# {'date', 'total', 'tax', 'tip'} <- 'tip' added!
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1) - IN PROGRESS

- [x] OCR Protocol
- [x] Specialized Spinners (Receipt, Handwritten)
- [x] Schema Registry (basic RAG)
- [x] SchemaAwareReceiptSpinner
- [ ] Fix OCR backend (install Tesseract)
- [ ] Complete end-to-end testing

### Phase 2: Voice Interface (Week 2)

- [ ] VoiceCommandProtocol
- [ ] Intent parsing (LLM-based)
- [ ] Correction application
- [ ] Voice UI component (web dashboard)

### Phase 3: Self-Tuning (Week 3)

- [ ] SelfTuningEngine
- [ ] Pattern learning from corrections
- [ ] Confidence tracking
- [ ] Pattern application

### Phase 4: Provenance (Week 4)

- [ ] TransformationProvenance tracking
- [ ] Correction history
- [ ] Pattern propagation
- [ ] Improvement suggestions

### Phase 5: Integration (Week 5)

- [ ] UnifiedSchemaAwareSpinner
- [ ] EnhancedSchemaRegistry
- [ ] Schema evolution
- [ ] Batch correction propagation

### Phase 6: Production (Week 6)

- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Deployment guide

---

## Critical Design Decisions

### 1. Protocol-Based Everything

**Why**: Extensibility
- Easy to swap components
- Easy to add new spinners
- Easy to add new learning strategies

### 2. Voice-First Interface

**Why**: User Experience
- Natural interaction
- No schema files
- No configuration
- Corrections in context

### 3. Self-Tuning is Core

**Why**: Intelligence
- System improves over time
- Learns user patterns
- Reduces manual work
- Adapts to edge cases

### 4. Provenance Always

**Why**: Trust & Debug
- Know why decisions made
- Trace corrections
- Debug issues
- Audit trail

### 5. RAG for Everything

**Why**: Semantic Intelligence
- Schema matching
- Field mapping
- Pattern learning
- Natural language understanding

---

## Next Steps (Immediate)

1. **Fix OCR Backend**
   ```bash
   # Windows (using Chocolatey)
   choco install tesseract

   # Or download from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Test End-to-End**
   ```bash
   PYTHONPATH=. python demos/demo_schema_aware_receipt.py
   ```

3. **Design Voice Interface**
   - Intent parsing with LLM
   - Correction application
   - Pattern learning

4. **Prototype Self-Tuning**
   - CorrectionPattern dataclass
   - Pattern learning algorithm
   - Confidence tracking

5. **Build UnifiedSchemaAwareSpinner**
   - Combine all components
   - Voice + self-tuning + provenance
   - End-to-end demo

---

## Success Metrics

### User Experience
- **Zero Config**: User never writes schema files
- **Voice-First**: 80%+ operations via voice
- **Self-Improving**: <5% corrections after 100 receipts

### System Performance
- **Accuracy**: >95% field extraction accuracy
- **Latency**: <500ms per receipt (with real OCR)
- **Learning**: Patterns learned after 3-5 examples

### Engineering Quality
- **Protocol-Based**: 100% swappable components
- **Provenance**: 100% traceable transformations
- **Testing**: >90% code coverage

---

**Status**: Design Complete - Ready for Implementation
**Next**: Install Tesseract → Test end-to-end → Build voice interface
