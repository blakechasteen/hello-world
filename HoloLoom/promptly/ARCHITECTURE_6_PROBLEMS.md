# Promptly Architecture: Supporting the 6 Problems Framework

**Design Philosophy**: Architecture should enable features, not constrain them.

We need a flexible, modular architecture that makes room for solving the 6 common AI problems while maintaining simplicity and performance.

---

## Current State Analysis

### What We Have Now

```
HoloLoom/promptly/
├── workflow_store.py                    # Workflow persistence
├── dspy_bridge.py                       # DSPy integration
├── dspy_workflow_adapter.py             # Multi-step workflows
├── beginner_prompts.py                  # Chat-based optimization
├── metrics_system.py                    # Evaluation metrics
└── examples/                            # Example workflows
```

**Current capabilities**:
- ✅ DSPy optimization
- ✅ Multi-step workflows
- ✅ Metrics evaluation
- ✅ Beginner-friendly interface

**Missing for 6 Problems**:
- ❌ Schema system
- ❌ Surgical editing
- ❌ Staged reasoning with gates
- ❌ Confidence tracking
- ❌ Consistency enforcement
- ❌ Context optimization

---

## Architectural Principles

### 1. **Separation of Concerns**
Each problem gets its own module with clear boundaries.

### 2. **Composition Over Inheritance**
Features compose together, not forced into hierarchy.

### 3. **Progressive Enhancement**
Basic features work alone, advanced features layer on top.

### 4. **API-First Design**
Every feature has programmatic API, UI comes second.

### 5. **Zero Breaking Changes**
New architecture extends existing, doesn't replace.

---

## Proposed Architecture

### Layer Model (7 Layers)

```
┌─────────────────────────────────────────────────┐
│  Layer 7: User Interfaces                      │
│  - CLI, Web UI, VSCode Extension               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Layer 6: Workflow Orchestration                │
│  - Multi-step pipelines, staged execution      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Layer 5: Problem Solvers (THE 6 MODULES)      │
│  - Schema, Surgical, Staged, Confidence,       │
│    Consistency, Context                         │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Layer 4: Execution Engine                      │
│  - DSPy bridge, LM providers, tool calls        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Layer 3: State Management                      │
│  - Memory, cache, versioning, persistence       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Layer 2: Core Primitives                       │
│  - Types, protocols, utilities                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Layer 1: Foundation (HoloLoom Integration)     │
│  - Config, embeddings, memory backends          │
└─────────────────────────────────────────────────┘
```

---

## Detailed Architecture

### Layer 1: Foundation (HoloLoom Integration)

**Purpose**: Connect Promptly to HoloLoom's infrastructure

```python
# HoloLoom/promptly/foundation/
├── __init__.py
├── config.py                  # Promptly-specific config
├── integration.py             # HoloLoom bridge
└── providers.py               # LM provider abstractions

# Key classes
class PromptlyConfig:
    """Extends HoloLoom Config with Promptly settings"""
    hololoom_config: Config
    enable_schema_validation: bool = True
    enable_confidence_tracking: bool = True
    enable_consistency_mode: bool = True
    default_context_budget: int = 5000

class HoloLoomIntegration:
    """Bridge to HoloLoom memory, embeddings, config"""
    def __init__(self, config: PromptlyConfig):
        self.hololoom = HoloLoom(config.hololoom_config)

    async def retrieve_context(self, query: str) -> List[MemoryShard]:
        return await self.hololoom.recall(query)

    async def store_result(self, result: Any) -> None:
        await self.hololoom.experience(result)
```

**Files**:
- `foundation/config.py` - Configuration system
- `foundation/integration.py` - HoloLoom bridge
- `foundation/providers.py` - LM provider abstractions (OpenAI, Anthropic, local)

---

### Layer 2: Core Primitives

**Purpose**: Shared types, protocols, utilities used by all layers

```python
# HoloLoom/promptly/core/
├── __init__.py
├── types.py                   # Core data types
├── protocols.py               # Interface definitions
├── errors.py                  # Exception hierarchy
└── utils.py                   # Shared utilities

# Key types
@dataclass
class PromptlyRequest:
    """Universal request format"""
    task: str
    inputs: Dict[str, Any]
    schema: Optional[Schema] = None
    confidence_threshold: float = 0.7
    context_budget: Optional[int] = None
    deterministic: bool = False

@dataclass
class PromptlyResponse:
    """Universal response format"""
    outputs: Dict[str, Any]
    confidence: float
    verification_status: VerificationStatus
    context_used: int  # tokens
    metadata: Dict[str, Any]

# Key protocols
class SchemaValidator(Protocol):
    def validate(self, data: Any, schema: Schema) -> ValidationResult: ...

class ConfidenceTracker(Protocol):
    def score(self, response: Any) -> float: ...

class ContextOptimizer(Protocol):
    def optimize(self, context: str, task: str) -> str: ...
```

**Files**:
- `core/types.py` - Data classes for requests, responses, schemas
- `core/protocols.py` - Interface definitions for all modules
- `core/errors.py` - Custom exceptions
- `core/utils.py` - Shared helper functions

---

### Layer 3: State Management

**Purpose**: Persistence, caching, versioning, history

```python
# HoloLoom/promptly/state/
├── __init__.py
├── cache.py                   # Result caching
├── history.py                 # Revision history
├── storage.py                 # Persistent storage
└── versioning.py              # Version control

# Key classes
class PromptlyCache:
    """Intelligent caching with drift detection"""
    def __init__(self, backend: CacheBackend):
        self.backend = backend

    async def get(self, key: str) -> Optional[PromptlyResponse]:
        """Retrieve cached result if exists"""

    async def set(self, key: str, value: PromptlyResponse) -> None:
        """Cache result with metadata"""

    async def check_drift(self, key: str, new_result: PromptlyResponse) -> DriftReport:
        """Compare new result with cached, detect drift"""

class RevisionHistory:
    """Git-like version control for AI outputs"""
    def __init__(self, storage: Storage):
        self.storage = storage
        self.graph = RevisionGraph()

    async def commit(self, result: PromptlyResponse, message: str) -> Revision:
        """Create revision snapshot"""

    async def rollback(self, revision_id: str) -> PromptlyResponse:
        """Revert to previous revision"""

    async def diff(self, rev_a: str, rev_b: str) -> Diff:
        """Compare two revisions"""

    async def branch(self, from_revision: str, name: str) -> Branch:
        """Create experimental branch"""
```

**Files**:
- `state/cache.py` - Caching system with drift detection
- `state/history.py` - Revision history and rollback
- `state/storage.py` - Persistent storage backends
- `state/versioning.py` - Git-like version control

---

### Layer 4: Execution Engine

**Purpose**: Execute prompts through various providers (DSPy, direct LM, etc.)

```python
# HoloLoom/promptly/execution/
├── __init__.py
├── engine.py                  # Main execution engine
├── dspy_executor.py           # DSPy-based execution
├── direct_executor.py         # Direct LM execution
└── tool_executor.py           # Tool calling support

# Key classes
class ExecutionEngine:
    """Unified execution interface"""
    def __init__(self, config: PromptlyConfig):
        self.config = config
        self.executors = {
            "dspy": DSPyExecutor(config),
            "direct": DirectExecutor(config)
        }

    async def execute(
        self,
        request: PromptlyRequest,
        executor_type: str = "dspy"
    ) -> PromptlyResponse:
        """Execute request through chosen executor"""
        executor = self.executors[executor_type]

        # Pre-execution: context optimization (Layer 5)
        optimized_context = await self.optimize_context(request)

        # Execution
        raw_response = await executor.execute(optimized_context)

        # Post-execution: validation, confidence scoring (Layer 5)
        validated_response = await self.validate_response(raw_response, request)

        return validated_response

class DSPyExecutor:
    """Execute through DSPy with optimization"""
    def __init__(self, config: PromptlyConfig):
        self.bridge = DSPyHoloLoom(config)

    async def execute(self, request: PromptlyRequest) -> Any:
        # Use existing dspy_bridge.py
        program = await self.bridge.execute(...)
        return program
```

**Files**:
- `execution/engine.py` - Unified execution interface
- `execution/dspy_executor.py` - DSPy integration (uses existing bridge)
- `execution/direct_executor.py` - Direct LM calls
- `execution/tool_executor.py` - Tool calling and agentic execution

---

### Layer 5: Problem Solvers (THE 6 MODULES)

**Purpose**: Solve each of the 6 common problems

This is the **core innovation layer**. Each problem gets its own module.

```python
# HoloLoom/promptly/solvers/
├── __init__.py
├── schema/                    # Problem 1: Projection Trap
│   ├── __init__.py
│   ├── builder.py             # Schema construction
│   ├── validator.py           # Schema validation
│   ├── templates.py           # Pre-built schemas
│   └── compiler.py            # Schema → prompt compilation
│
├── surgical/                  # Problem 2: Revision Loop
│   ├── __init__.py
│   ├── editor.py              # Diff-based editing
│   ├── differ.py              # Diff generation
│   ├── patcher.py             # Patch application
│   └── locker.py              # Field locking
│
├── staged/                    # Problem 3: Planning Illusion
│   ├── __init__.py
│   ├── workflow.py            # Multi-stage workflows
│   ├── gates.py               # Validation gates
│   ├── tools.py               # Tool contracts
│   └── reasoning.py           # Reasoning quality metrics
│
├── confidence/                # Problem 4: Confidence Illusion
│   ├── __init__.py
│   ├── tracker.py             # Confidence scoring
│   ├── verifier.py            # Chain of verification
│   ├── sources.py             # Source citation tracking
│   └── uncertainty.py         # "I don't know" handling
│
├── consistency/               # Problem 5: Drift Problem
│   ├── __init__.py
│   ├── enforcer.py            # Consistency enforcement
│   ├── normalizer.py          # Input normalization
│   ├── rules.py               # Rule compilation
│   └── drift_detector.py      # Drift detection
│
└── context/                   # Problem 6: Cognitive Bandwidth Trap
    ├── __init__.py
    ├── optimizer.py           # Smart context loading
    ├── cleaner.py             # Context cleaning
    ├── budget.py              # Budget management
    └── analyzer.py            # Context impact analysis
```

#### Module 1: Schema (Projection Trap)

```python
# HoloLoom/promptly/solvers/schema/builder.py

class SchemaBuilder:
    """Build output schemas visually or programmatically"""

    def __init__(self):
        self.fields: Dict[str, Field] = {}

    def add_field(
        self,
        name: str,
        type: FieldType,
        required: bool = True,
        constraints: Optional[Constraints] = None,
        description: Optional[str] = None
    ) -> "SchemaBuilder":
        """Add field to schema"""
        self.fields[name] = Field(
            name=name,
            type=type,
            required=required,
            constraints=constraints,
            description=description
        )
        return self

    def build(self) -> Schema:
        """Compile to Schema object"""
        return Schema(fields=self.fields)

    def to_prompt(self) -> str:
        """Generate schema-constrained prompt"""
        # Converts schema to prompt instructions
        return generate_schema_prompt(self.fields)

# Example usage
schema = SchemaBuilder()
schema.add_field("title", FieldType.STRING, max_length=100)
schema.add_field("audience", FieldType.ENUM, values=["executive", "technical"])
schema.add_field("summary", FieldType.STRING, required=True)

compiled = schema.build()
prompt = schema.to_prompt()
```

#### Module 2: Surgical (Revision Loop)

```python
# HoloLoom/promptly/solvers/surgical/editor.py

class SurgicalEditor:
    """Perform precise edits without full rewrites"""

    def __init__(self, differ: Differ, patcher: Patcher):
        self.differ = differ
        self.patcher = patcher

    async def edit(
        self,
        document: Document,
        snippet: Snippet,
        instruction: str,
        freeze_rest: bool = True
    ) -> EditResult:
        """Edit only the specified snippet"""

        # Generate diff for snippet
        diff = await self.differ.generate_diff(
            original=snippet.text,
            instruction=instruction
        )

        # Verify diff doesn't touch frozen content
        if freeze_rest:
            self._validate_diff_scope(diff, snippet)

        # Apply patch
        edited_document = self.patcher.apply(document, diff)

        return EditResult(
            document=edited_document,
            diff=diff,
            snippet_modified=snippet
        )

    def _validate_diff_scope(self, diff: Diff, allowed_snippet: Snippet):
        """Ensure diff only touches allowed snippet"""
        for change in diff.changes:
            if not allowed_snippet.contains(change.location):
                raise FrozenContentViolation(
                    f"Diff attempts to modify frozen content at {change.location}"
                )
```

#### Module 3: Staged (Planning Illusion)

```python
# HoloLoom/promptly/solvers/staged/workflow.py

class StagedWorkflow:
    """Multi-stage reasoning with validation gates"""

    def __init__(self, name: str):
        self.name = name
        self.stages: List[Stage] = []

    def add_stage(
        self,
        name: str,
        task: str,
        required_outputs: Dict[str, Any],
        validation_gates: List[ValidationGate],
        tools: Optional[List[str]] = None
    ) -> "StagedWorkflow":
        """Add stage with explicit outputs and gates"""
        stage = Stage(
            name=name,
            task=task,
            required_outputs=required_outputs,
            validation_gates=validation_gates,
            tools=tools or []
        )
        self.stages.append(stage)
        return self

    async def execute(self, initial_inputs: Dict[str, Any]) -> WorkflowResult:
        """Execute stages sequentially with gating"""
        context = initial_inputs.copy()
        trace = []

        for stage in self.stages:
            # Execute stage
            stage_result = await self._execute_stage(stage, context)

            # Validate outputs
            validation_result = self._validate_stage(stage_result, stage)

            if not validation_result.passed:
                # Gate failed - retry or abort
                return WorkflowResult(
                    success=False,
                    failed_stage=stage.name,
                    validation_error=validation_result.error,
                    trace=trace
                )

            # Update context for next stage
            context.update(stage_result.outputs)
            trace.append(stage_result)

        return WorkflowResult(
            success=True,
            final_outputs=context,
            trace=trace
        )
```

#### Module 4: Confidence (Confidence Illusion)

```python
# HoloLoom/promptly/solvers/confidence/tracker.py

class ConfidenceTracker:
    """Track and enforce confidence requirements"""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    async def score(
        self,
        response: Any,
        evidence: List[str],
        sources: List[str]
    ) -> ConfidenceScore:
        """Calculate confidence score for response"""

        # Multiple confidence signals
        signals = []

        # Signal 1: Evidence quality
        evidence_score = self._score_evidence(evidence)
        signals.append(("evidence", evidence_score))

        # Signal 2: Source reliability
        source_score = self._score_sources(sources)
        signals.append(("sources", source_score))

        # Signal 3: Internal consistency
        consistency_score = self._check_consistency(response)
        signals.append(("consistency", consistency_score))

        # Aggregate
        overall = sum(s[1] for s in signals) / len(signals)

        return ConfidenceScore(
            overall=overall,
            signals=signals,
            meets_threshold=overall >= self.min_confidence
        )

    async def verify_claims(
        self,
        response: Any,
        verification_mode: VerificationMode = VerificationMode.CHAIN
    ) -> VerificationResult:
        """Chain of verification for all claims"""

        # Stage 1: Extract claims
        claims = await self._extract_claims(response)

        # Stage 2: Find sources for each
        claims_with_sources = await self._find_sources(claims)

        # Stage 3: Verify against sources
        verified = await self._verify_against_sources(claims_with_sources)

        # Stage 4: Flag unverified
        flagged = [c for c in verified if not c.verified]

        return VerificationResult(
            total_claims=len(claims),
            verified_count=len([c for c in verified if c.verified]),
            unverified=flagged,
            verification_score=len([c for c in verified if c.verified]) / len(claims)
        )
```

#### Module 5: Consistency (Drift Problem)

```python
# HoloLoom/promptly/solvers/consistency/enforcer.py

class ConsistencyEnforcer:
    """Enforce deterministic, consistent outputs"""

    def __init__(self, cache: PromptlyCache, normalizer: InputNormalizer):
        self.cache = cache
        self.normalizer = normalizer

    async def execute_deterministic(
        self,
        request: PromptlyRequest,
        rules: ConsistencyRules
    ) -> PromptlyResponse:
        """Execute with zero drift tolerance"""

        # Normalize input for consistency
        normalized_input = self.normalizer.normalize(request.inputs)

        # Check cache for identical input
        cache_key = self._compute_hash(normalized_input, rules)
        cached = await self.cache.get(cache_key)

        if cached:
            # Check if we're in strict mode
            if rules.force_cache_match:
                return cached

        # Execute with zero temperature
        response = await self._execute_with_zero_temp(
            normalized_input,
            rules
        )

        # Detect drift if cached result exists
        if cached:
            drift = await self.cache.check_drift(cache_key, response)
            if drift.detected and rules.zero_drift_tolerance:
                raise DriftDetected(
                    f"Output drifted from cached result: {drift.report}"
                )

        # Cache for future
        await self.cache.set(cache_key, response)

        return response
```

#### Module 6: Context (Cognitive Bandwidth Trap)

```python
# HoloLoom/promptly/solvers/context/optimizer.py

class ContextOptimizer:
    """Optimize context for quality and efficiency"""

    def __init__(self, cleaner: ContextCleaner, budget_mgr: BudgetManager):
        self.cleaner = cleaner
        self.budget_mgr = budget_mgr

    async def optimize(
        self,
        context: str,
        task: str,
        budget: Optional[int] = None
    ) -> OptimizedContext:
        """Load minimal context needed for task"""

        # Analyze task to determine required context
        requirements = await self._analyze_requirements(task)

        # Extract only relevant sections
        relevant = await self._extract_relevant(context, requirements)

        # Clean extracted context
        cleaned = self.cleaner.clean(relevant)

        # Check against budget
        if budget:
            if len(cleaned) > budget:
                cleaned = await self._compress_to_budget(cleaned, budget)

        return OptimizedContext(
            content=cleaned,
            original_size=len(context),
            optimized_size=len(cleaned),
            reduction_pct=(len(context) - len(cleaned)) / len(context),
            requirements_met=self._verify_requirements(cleaned, requirements)
        )

    async def _extract_relevant(
        self,
        context: str,
        requirements: ContextRequirements
    ) -> str:
        """Extract only context sections relevant to task"""

        sections = []

        # Required context (must include)
        for req in requirements.required:
            section = self._find_section(context, req)
            if section:
                sections.append(("required", section))

        # Relevant context (include if space allows)
        for rel in requirements.relevant:
            section = self._find_section(context, rel)
            if section:
                sections.append(("relevant", section))

        # Combine sections
        combined = self._combine_sections(sections)
        return combined
```

---

### Layer 6: Workflow Orchestration

**Purpose**: Compose problem solvers into complete workflows

```python
# HoloLoom/promptly/orchestration/
├── __init__.py
├── orchestrator.py            # Main orchestrator
├── pipeline.py                # Pipeline builder
└── compositions.py            # Pre-built compositions

# Key classes
class PromptlyOrchestrator:
    """Orchestrate multiple problem solvers"""

    def __init__(self, config: PromptlyConfig):
        self.config = config
        self.execution_engine = ExecutionEngine(config)

        # Initialize all 6 solvers
        self.schema = SchemaModule(config)
        self.surgical = SurgicalModule(config)
        self.staged = StagedModule(config)
        self.confidence = ConfidenceModule(config)
        self.consistency = ConsistencyModule(config)
        self.context = ContextModule(config)

    async def execute(
        self,
        request: PromptlyRequest,
        enable_solvers: Optional[List[str]] = None
    ) -> PromptlyResponse:
        """Execute request through enabled solvers"""

        # Default: enable all solvers based on request
        if enable_solvers is None:
            enable_solvers = self._auto_detect_solvers(request)

        # Solver pipeline
        processed_request = request

        # 1. Schema validation (if schema provided)
        if "schema" in enable_solvers and request.schema:
            processed_request = await self.schema.validate_request(processed_request)

        # 2. Context optimization (if context large)
        if "context" in enable_solvers:
            processed_request = await self.context.optimize_request(processed_request)

        # 3. Consistency enforcement (if deterministic)
        if "consistency" in enable_solvers and request.deterministic:
            processed_request = await self.consistency.normalize_request(processed_request)

        # 4. Execute (with staging if needed)
        if "staged" in enable_solvers and self._is_complex_task(request):
            response = await self.staged.execute_workflow(processed_request)
        else:
            response = await self.execution_engine.execute(processed_request)

        # 5. Confidence tracking (post-execution)
        if "confidence" in enable_solvers:
            response = await self.confidence.score_response(response)

        # 6. Schema validation (post-execution)
        if "schema" in enable_solvers and request.schema:
            response = await self.schema.validate_response(response, request.schema)

        return response
```

**Files**:
- `orchestration/orchestrator.py` - Main orchestrator composing all solvers
- `orchestration/pipeline.py` - Pipeline builder for custom workflows
- `orchestration/compositions.py` - Pre-built workflow templates

---

### Layer 7: User Interfaces

**Purpose**: Multiple interfaces for different users

```python
# HoloLoom/promptly/interfaces/
├── __init__.py
├── cli/                       # Command-line interface
│   ├── __init__.py
│   ├── commands.py
│   └── repl.py
│
├── api/                       # REST API (FastAPI)
│   ├── __init__.py
│   ├── server.py
│   ├── routes/
│   └── models.py
│
├── sdk/                       # Python SDK
│   ├── __init__.py
│   ├── client.py
│   └── builders.py
│
└── web/                       # Web UI (future)
    ├── components/
    └── dashboard/
```

**CLI Example**:
```bash
# Schema-first workflow
promptly schema create my_qa_schema \
  --field "question:string" \
  --field "answer:string:required" \
  --field "confidence:number"

# Execute with schema
promptly execute my_qa_schema \
  --input question="What is Thompson Sampling?" \
  --enable-confidence-tracking \
  --min-confidence 0.8

# Surgical edit
promptly edit document.txt \
  --snippet "paragraph 3" \
  --instruction "Fix grammar" \
  --freeze-rest

# Staged workflow
promptly workflow create analysis_pipeline \
  --stage "data_review" \
  --stage "root_cause" \
  --stage "recommendations" \
  --validation-gates
```

**Python SDK Example**:
```python
from promptly import Promptly

# Initialize
promptly = Promptly(config=PromptlyConfig())

# Schema-first approach
schema = promptly.schema.build() \
    .add_field("title", type="string", max_length=100) \
    .add_field("summary", type="string", required=True) \
    .compile()

# Execute with all solvers enabled
result = await promptly.execute(
    task="Summarize this article",
    inputs={"article": article_text},
    schema=schema,
    enable_confidence=True,
    enable_consistency=True,
    min_confidence=0.8
)

print(f"Confidence: {result.confidence}")
print(f"Verified: {result.verification_status}")
print(f"Summary: {result.outputs['summary']}")
```

---

## Integration Points

### With Existing Promptly

```python
# Backward compatibility
from HoloLoom.promptly import DSPyHoloLoom, DSPyWorkflowAdapter

# Old API still works
bridge = DSPyHoloLoom(config=Config.fused())
result = await bridge.execute(signature, **inputs)

# New API extends it
from promptly.orchestration import PromptlyOrchestrator

orchestrator = PromptlyOrchestrator(config=PromptlyConfig())
result = await orchestrator.execute(
    request=PromptlyRequest(task="...", inputs=inputs),
    enable_solvers=["schema", "confidence", "context"]
)

# Old workflows work in new system
from HoloLoom.promptly import load_workflow
workflow = await load_workflow("qa_workflow.yaml")

# Enhance with new solvers
enhanced = orchestrator.enhance_workflow(
    workflow,
    enable_schema=True,
    enable_confidence=True
)
```

### With HoloLoom

```python
# Promptly uses HoloLoom for:
# 1. Memory/context retrieval
context = await hololoom_integration.retrieve_context(query)

# 2. Result storage
await hololoom_integration.store_result(result)

# 3. Embeddings for context optimization
embeddings = hololoom_integration.get_embeddings(text)

# 4. Configuration
promptly_config = PromptlyConfig(
    hololoom_config=Config.fused()  # Inherit HoloLoom config
)
```

---

## File Structure

### Complete Directory Layout

```
HoloLoom/promptly/
├── __init__.py                          # Public API exports
├── README.md                            # Main documentation
│
├── foundation/                          # Layer 1: Foundation
│   ├── __init__.py
│   ├── config.py
│   ├── integration.py
│   └── providers.py
│
├── core/                                # Layer 2: Core Primitives
│   ├── __init__.py
│   ├── types.py
│   ├── protocols.py
│   ├── errors.py
│   └── utils.py
│
├── state/                               # Layer 3: State Management
│   ├── __init__.py
│   ├── cache.py
│   ├── history.py
│   ├── storage.py
│   └── versioning.py
│
├── execution/                           # Layer 4: Execution Engine
│   ├── __init__.py
│   ├── engine.py
│   ├── dspy_executor.py                # Uses existing dspy_bridge.py
│   ├── direct_executor.py
│   └── tool_executor.py
│
├── solvers/                             # Layer 5: Problem Solvers
│   ├── __init__.py
│   ├── schema/                         # Problem 1
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── validator.py
│   │   ├── templates.py
│   │   └── compiler.py
│   ├── surgical/                       # Problem 2
│   │   ├── __init__.py
│   │   ├── editor.py
│   │   ├── differ.py
│   │   ├── patcher.py
│   │   └── locker.py
│   ├── staged/                         # Problem 3
│   │   ├── __init__.py
│   │   ├── workflow.py
│   │   ├── gates.py
│   │   ├── tools.py
│   │   └── reasoning.py
│   ├── confidence/                     # Problem 4
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   ├── verifier.py
│   │   ├── sources.py
│   │   └── uncertainty.py
│   ├── consistency/                    # Problem 5
│   │   ├── __init__.py
│   │   ├── enforcer.py
│   │   ├── normalizer.py
│   │   ├── rules.py
│   │   └── drift_detector.py
│   └── context/                        # Problem 6
│       ├── __init__.py
│       ├── optimizer.py
│       ├── cleaner.py
│       ├── budget.py
│       └── analyzer.py
│
├── orchestration/                       # Layer 6: Workflow Orchestration
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── pipeline.py
│   └── compositions.py
│
├── interfaces/                          # Layer 7: User Interfaces
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── commands.py
│   │   └── repl.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── routes/
│   └── sdk/
│       ├── __init__.py
│       ├── client.py
│       └── builders.py
│
├── legacy/                              # Existing code (preserved)
│   ├── workflow_store.py               # Keep for backward compatibility
│   ├── dspy_bridge.py                  # Wrapped by execution/dspy_executor.py
│   ├── dspy_workflow_adapter.py        # Wrapped by orchestration/
│   ├── beginner_prompts.py             # CLI interface, now part of interfaces/
│   └── metrics_system.py               # Integrated into confidence/
│
├── examples/                            # Example workflows and schemas
│   ├── schemas/
│   ├── workflows/
│   └── scripts/
│
├── tests/                               # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── docs/                                # Documentation
    ├── architecture/
    ├── guides/
    ├── api/
    └── tutorials/
```

---

## Implementation Strategy

### Phase 0: Foundation (Week 1-2)

**Goal**: Set up architecture without breaking existing code

```yaml
tasks:
  - Create directory structure
  - Define core types and protocols (core/)
  - Create foundation layer (foundation/)
  - Ensure backward compatibility

deliverables:
  - Empty module structure
  - Type definitions
  - Protocol interfaces
  - Integration tests pass
```

### Phase 1: First Solver (Week 3-4)

**Goal**: Implement Problem 1 (Schema) as proof of concept

```yaml
tasks:
  - Implement schema/builder.py
  - Implement schema/validator.py
  - Integrate with execution engine
  - Create CLI interface
  - Write tests

deliverables:
  - Working schema builder
  - Schema validation
  - CLI: promptly schema create
  - 20+ unit tests
```

### Phase 2: Orchestration (Week 5-6)

**Goal**: Build orchestrator that composes solvers

```yaml
tasks:
  - Implement orchestration/orchestrator.py
  - Add solver auto-detection
  - Create pipeline builder
  - Test with schema solver

deliverables:
  - Working orchestrator
  - Auto-detection logic
  - Pipeline composition
  - Integration tests
```

### Phase 3: Additional Solvers (Week 7-18)

**Goal**: Implement remaining 5 solvers

```yaml
week_7_8: Problem 2 (Surgical)
week_9_10: Problem 3 (Staged)
week_11_12: Problem 4 (Confidence)
week_13_14: Problem 5 (Consistency)
week_15_16: Problem 6 (Context)
week_17_18: Integration and polish
```

### Phase 4: State & Interfaces (Week 19-22)

**Goal**: Add persistence, caching, advanced interfaces

```yaml
week_19: State management (cache, history)
week_20: Web API (FastAPI)
week_21: Python SDK
week_22: Web UI (basic)
```

---

## Design Decisions

### Why 7 Layers?

**Layer 1-4**: Infrastructure (foundation, primitives, state, execution)
**Layer 5**: Problem solvers (the innovation)
**Layer 6**: Orchestration (composition)
**Layer 7**: Interfaces (user-facing)

**Benefits**:
- Clear separation of concerns
- Each solver is independent
- Easy to add new solvers
- Easy to test in isolation

### Why "Solvers" Not "Plugins"?

**Solvers** implies:
- Each module solves a specific problem
- Composable but not dependent
- Can work alone or together

**Plugins** implies:
- Optional extensions
- Not core to system

We want solvers to feel like **first-class citizens**, not optional add-ons.

### Why Keep Legacy Code?

**Backward compatibility** is critical:
- Existing DSPy integrations keep working
- Beginner prompts system stays usable
- Workflow adapter still valid

New architecture **wraps and extends**, doesn't replace.

### Why Protocol-Based?

**Protocols** enable:
- Swap implementations (e.g., different cache backends)
- Mock for testing
- Clear contracts between layers
- No tight coupling

Example:
```python
class SchemaValidator(Protocol):
    def validate(self, data: Any, schema: Schema) -> ValidationResult: ...

# Can have multiple implementations
class DSPySchemaValidator: ...
class JSONSchemaValidator: ...
class CustomSchemaValidator: ...

# Orchestrator doesn't care which
validator: SchemaValidator = get_validator()
result = validator.validate(data, schema)
```

---

## Migration Path

### For Existing Users

**No migration needed!** Old API keeps working:

```python
# Old way (still works)
from HoloLoom.promptly import DSPyHoloLoom
bridge = DSPyHoloLoom(config)
result = await bridge.execute(signature, **inputs)

# New way (opt-in)
from promptly import Promptly
promptly = Promptly(config)
result = await promptly.execute(request)

# Gradual migration
# Step 1: Use old API
# Step 2: Add schema validation
result = await promptly.with_schema(schema).execute_dspy(signature, **inputs)
# Step 3: Add confidence tracking
result = await promptly.with_schema(schema).with_confidence(min=0.8).execute_dspy(...)
# Step 4: Full new API
result = await promptly.execute(PromptlyRequest(...))
```

### For New Users

**Start with new API:**

```python
from promptly import Promptly, SchemaBuilder

# Clean, simple API
promptly = Promptly()

# Build schema
schema = SchemaBuilder() \
    .add_field("answer", "string", required=True) \
    .add_field("confidence", "number", min=0.0, max=1.0) \
    .build()

# Execute
result = await promptly.execute(
    task="Answer this question",
    inputs={"question": "What is X?"},
    schema=schema,
    enable_confidence=True
)
```

---

## Testing Strategy

### Unit Tests (Per Module)

```python
# tests/unit/solvers/schema/test_builder.py
def test_schema_builder_basic():
    schema = SchemaBuilder()
    schema.add_field("name", FieldType.STRING)
    compiled = schema.build()

    assert "name" in compiled.fields
    assert compiled.fields["name"].type == FieldType.STRING

# tests/unit/solvers/surgical/test_editor.py
async def test_surgical_edit_frozen_content():
    editor = SurgicalEditor(differ, patcher)

    with pytest.raises(FrozenContentViolation):
        await editor.edit(
            document=doc,
            snippet=snippet,
            instruction="Change everything",  # Too broad
            freeze_rest=True
        )
```

### Integration Tests (Solver + Engine)

```python
# tests/integration/test_schema_integration.py
async def test_schema_validation_end_to_end():
    schema = SchemaBuilder() \
        .add_field("answer", "string", required=True) \
        .build()

    orchestrator = PromptlyOrchestrator(config)

    result = await orchestrator.execute(
        PromptlyRequest(
            task="Answer question",
            inputs={"q": "What is X?"},
            schema=schema
        )
    )

    # Should have validated output
    assert "answer" in result.outputs
    assert isinstance(result.outputs["answer"], str)
```

### E2E Tests (Full Workflows)

```python
# tests/e2e/test_complete_workflow.py
async def test_qa_with_all_solvers():
    """Test Q&A with schema, confidence, context optimization"""

    promptly = Promptly(config)

    # Define schema
    schema = promptly.schema.build() \
        .add_field("answer", "string", required=True) \
        .add_field("confidence", "number", required=True) \
        .add_field("sources", "array", required=True) \
        .build()

    # Execute with all solvers
    result = await promptly.execute(
        task="What is Thompson Sampling?",
        inputs={"context": large_document},
        schema=schema,
        enable_confidence=True,
        enable_context_optimization=True,
        min_confidence=0.8
    )

    # Verify all solvers worked
    assert result.outputs["answer"]  # Schema enforced
    assert result.confidence >= 0.8  # Confidence tracked
    assert result.context_used < len(large_document) * 0.3  # Context optimized
    assert result.verification_status == VerificationStatus.VERIFIED
```

---

## Performance Considerations

### Caching Strategy

```python
# Layer 3: State Management
class PromptlyCache:
    """Multi-level caching"""

    def __init__(self):
        self.l1_memory = {}  # In-memory (fast)
        self.l2_disk = DiskCache()  # Persistent (slower)
        self.l3_redis = RedisCache()  # Distributed (optional)

    async def get(self, key: str) -> Optional[PromptlyResponse]:
        # L1: Memory
        if key in self.l1_memory:
            return self.l1_memory[key]

        # L2: Disk
        result = await self.l2_disk.get(key)
        if result:
            self.l1_memory[key] = result  # Promote to L1
            return result

        # L3: Redis (if available)
        if self.l3_redis:
            result = await self.l3_redis.get(key)
            if result:
                self.l1_memory[key] = result
                return result

        return None
```

### Lazy Loading

```python
# Only load solvers when needed
class PromptlyOrchestrator:
    def __init__(self, config: PromptlyConfig):
        self.config = config
        self._solvers = {}  # Lazy-loaded

    def _get_solver(self, name: str):
        """Lazy load solvers on first use"""
        if name not in self._solvers:
            self._solvers[name] = self._load_solver(name)
        return self._solvers[name]

    async def execute(self, request: PromptlyRequest):
        # Only load needed solvers
        for solver_name in self._detect_needed_solvers(request):
            solver = self._get_solver(solver_name)
            # Use solver...
```

### Parallel Execution

```python
# Execute independent solvers in parallel
async def execute(self, request: PromptlyRequest):
    # These can run in parallel
    context_task = asyncio.create_task(
        self.context.optimize(request)
    )
    consistency_task = asyncio.create_task(
        self.consistency.normalize(request)
    )

    # Wait for both
    optimized_context, normalized_input = await asyncio.gather(
        context_task,
        consistency_task
    )

    # Continue with execution...
```

---

## Success Metrics

### Architecture Quality

- **Modularity**: Each solver can be tested independently
- **Extensibility**: New solvers can be added without modifying existing code
- **Performance**: <10ms overhead per solver
- **Maintainability**: Clear boundaries, protocols, documentation

### Developer Experience

- **Time to first feature**: <4 hours for new solver
- **Test coverage**: >85% for all solvers
- **API clarity**: Beginner can use in <15 minutes
- **Documentation**: Complete API docs + tutorials

### User Experience

- **Backward compatibility**: 100% of old code works
- **Migration path**: Optional, gradual, no breaking changes
- **CLI usability**: Intuitive commands, helpful errors
- **SDK ergonomics**: Fluent API, chainable methods

---

## Conclusion

This architecture provides:

✅ **Room for the 6 Problems**: Each gets its own module (solvers/)
✅ **Backward Compatibility**: Legacy code preserved and wrapped
✅ **Extensibility**: Easy to add new solvers
✅ **Composability**: Solvers work alone or together
✅ **Testability**: Clear boundaries, protocol-based
✅ **Performance**: Caching, lazy loading, parallelism
✅ **Multiple Interfaces**: CLI, API, SDK, Web UI

**Next Steps**:
1. Get feedback on architecture
2. Start Phase 0 (foundation)
3. Implement first solver (schema) as proof of concept
4. Iterate based on learnings

The architecture is **designed to grow** with Promptly's ambitions while staying simple and maintainable.

---

**Ready to build?** 🚀
