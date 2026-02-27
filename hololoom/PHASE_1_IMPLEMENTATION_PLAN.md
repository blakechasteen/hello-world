# Phase 1 Implementation Plan: Core Engine + First Departments

**Version**: 1.0.0
**Date**: November 9, 2025
**Duration**: 12 weeks (3 months)
**Goal**: Build generic department framework + beekeeping-specific departments

---

## Executive Summary

Phase 1 builds the **invariant core** (confidence negotiation, nested learning, DS-STAR verification) with **beekeeping-specific departments** to validate the architecture. By the end of Phase 1, you'll have:

1. **Working department framework** (generic, reusable)
2. **Context Department** (HoloLoom wrapped in department protocol)
3. **MasterWeaver Department** (beekeeping entity extraction)
4. **Infrastructure Department** (zero-copy data access)
5. **Verification Department** (cross-department validation)
6. **Orchestration Department** (task routing)
7. **MCP integration** (departments communicate via Model Context Protocol)

**Success Criteria**:
- Beekeeping workflow runs end-to-end (audio → entities → context → insights)
- All departments report confidence accurately
- DS-STAR verification loop working
- Confidence drives learning rates
- Privacy envelope operational (TEE integration)

---

## Timeline Overview

```
Week 1-2:  Core Framework (Department Protocol + Base Classes)
Week 3-4:  Context Department (Wrap existing HoloLoom)
Week 5-6:  MasterWeaver Department (Beekeeping entity extraction)
Week 7-8:  Infrastructure Department (Zero-copy data access)
Week 9-10: Verification + Orchestration Departments
Week 11-12: Integration + End-to-End Testing
```

---

## Week 1-2: Core Framework

### Goal
Build the generic department infrastructure that all departments will use.

### Deliverables

#### 1. Department Protocol (`hololoom/departments/protocol.py`)
**Lines**: ~500

```python
# Core types
- ConfidenceLevel (enum)
- ConfidenceMetadata (dataclass)
- DepartmentRequest (dataclass)
- DepartmentResponse (dataclass)
- VerificationResult (dataclass)
- DepartmentManifest (dataclass)
- DepartmentConfig (dataclass)

# Protocol
- Department (Protocol class)
  - execute()
  - verify()
  - refine()
  - update_strategy()
  - get_session_state()
  - get_institutional_memory()
  - health_check()
```

**Status**: ✅ Specified in [DEPARTMENT_INTERFACE_SPEC.md](DEPARTMENT_INTERFACE_SPEC.md)

#### 2. Base Department (`hololoom/departments/base.py`)
**Lines**: ~300

```python
class BaseDepartment(ABC):
    """Abstract base class implementing common department behavior"""

    def __init__(self, config: DepartmentConfig):
        self.name = ""
        self.domain = ""
        self.version = ""
        self.supported_tasks = []
        self.confidence_range = (0.0, 1.0)

        # Memory systems (concrete implementations by subclass)
        self.short_term_memory = {}
        self.medium_term_memory = None
        self.long_term_memory = {}

        self.config = config
        self.logger = logging.getLogger(f"Department.{self.name}")

    @abstractmethod
    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Subclass must implement"""
        ...

    @abstractmethod
    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Subclass must implement"""
        ...

    @abstractmethod
    async def refine(...) -> DepartmentResponse:
        """Subclass must implement"""
        ...

    # Common helpers (all departments can use)
    def _determine_detail_level(self, confidence: float, preference: str) -> str:
        """Map confidence to detail level"""
        if preference != "auto":
            return preference
        if confidence >= 0.90: return "minimal"
        elif confidence >= 0.75: return "moderate"
        elif confidence >= 0.50: return "detailed"
        else: return "exhaustive"

    def _determine_learning_rate(self, confidence: float) -> str:
        """Map confidence to learning rate"""
        level = ConfidenceLevel.from_score(confidence)
        rate_map = {
            ConfidenceLevel.CRITICAL: "weekly",
            ConfidenceLevel.HIGH: "daily",
            ConfidenceLevel.MEDIUM: "hourly",
            ConfidenceLevel.LOW: "per-task",
            ConfidenceLevel.UNCERTAIN: "immediate"
        }
        return rate_map[level]

    async def health_check(self) -> Dict[str, Any]:
        """Default health check (subclass can override)"""
        return {
            "status": "healthy",
            "name": self.name,
            "version": self.version,
            "confidence_range": self.confidence_range,
            "learning_rate": self._determine_learning_rate(
                sum(self.confidence_range) / 2
            ),
            "memory_usage": {
                "short_term": len(self.short_term_memory),
                "long_term": len(self.long_term_memory)
            }
        }
```

#### 3. Department Registry (`hololoom/departments/registry.py`)
**Lines**: ~200

```python
class DepartmentRegistry:
    """Local registry for discovering and instantiating departments"""

    def __init__(self):
        self._departments: Dict[str, Department] = {}
        self._manifests: Dict[str, DepartmentManifest] = {}

    def register(self, manifest: DepartmentManifest, department: Department):
        """Register department for discovery"""
        key = f"{manifest.domain}.{manifest.name}"
        self._manifests[key] = manifest
        self._departments[key] = department
        logging.info(f"Registered department: {key}")

    def discover(
        self,
        domain: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> List[DepartmentManifest]:
        """Find departments matching criteria"""
        results = []
        for key, manifest in self._manifests.items():
            if domain and manifest.domain != domain:
                continue
            if task_type and task_type not in manifest.supported_tasks:
                continue
            results.append(manifest)
        return results

    def get(self, domain: str, name: str) -> Optional[Department]:
        """Get department instance"""
        key = f"{domain}.{name}"
        return self._departments.get(key)

    def list_all(self) -> List[DepartmentManifest]:
        """List all registered departments"""
        return list(self._manifests.values())
```

#### 4. Tests (`tests/departments/test_protocol.py`)
**Lines**: ~400

```python
def test_confidence_level_mapping():
    """ConfidenceLevel.from_score() works correctly"""
    assert ConfidenceLevel.from_score(0.96) == ConfidenceLevel.CRITICAL
    assert ConfidenceLevel.from_score(0.88) == ConfidenceLevel.HIGH
    assert ConfidenceLevel.from_score(0.70) == ConfidenceLevel.MEDIUM
    assert ConfidenceLevel.from_score(0.55) == ConfidenceLevel.LOW
    assert ConfidenceLevel.from_score(0.30) == ConfidenceLevel.UNCERTAIN

def test_confidence_metadata():
    """ConfidenceMetadata initializes correctly"""
    meta = ConfidenceMetadata(
        score=0.85,
        justification=["Test reason"],
        uncertainty_sources=[]
    )
    assert meta.level == ConfidenceLevel.HIGH
    assert meta.learning_rate == "daily"

def test_department_request():
    """DepartmentRequest serialization"""
    request = DepartmentRequest(
        task_id="test_001",
        task_type="test",
        parameters={"key": "value"},
        confidence_expected=0.75,
        context_preference="moderate",
        privacy_level="public"
    )
    assert request.confidence_expected == 0.75

def test_registry_register_and_discover():
    """Registry can register and discover departments"""
    registry = DepartmentRegistry()

    manifest = DepartmentManifest(
        name="TestDept",
        version="1.0.0",
        domain="test",
        author="test",
        license="MIT",
        description="Test department",
        supported_tasks=["test_task"],
        confidence_range=(0.5, 0.9),
        dependencies=[],
        privacy_guarantees=[]
    )

    # Create mock department
    dept = MockDepartment()

    registry.register(manifest, dept)

    # Discover
    found = registry.discover(domain="test")
    assert len(found) == 1
    assert found[0].name == "TestDept"
```

### Tasks (Week 1-2)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| Write department protocol | 2 days | Dev | `protocol.py` (500 lines) |
| Write base department class | 1 day | Dev | `base.py` (300 lines) |
| Write department registry | 1 day | Dev | `registry.py` (200 lines) |
| Write protocol tests | 2 days | Dev | `test_protocol.py` (400 lines) |
| Documentation | 1 day | Dev | API docs + examples |
| Code review | 1 day | Team | Feedback incorporated |

**Estimated Effort**: 8 days (1.6 weeks)

---

## Week 3-4: Core Departments + Integration

**Updated**: November 13, 2025 (Moonshot Pivot)

### Goal
Build the first 5 core departments (RAG, Planning, Context, Orchestration, Infrastructure) and validate department-to-department communication.

### Strategic Pivot
Original plan focused on beekeeping-specific MasterWeaver department. **New strategy**: Build generic core departments first, then apply to verticals (beekeeping, healthcare, finance).

### Deliverables

#### 1. Context Department (`hololoom/departments/context.py`)
**Lines**: ~600

```python
class ContextDepartment(BaseDepartment):
    """Generic context enrichment using HoloLoom"""

    def __init__(self, config: DepartmentConfig):
        super().__init__(config)
        self.name = "Context"
        self.domain = "generic"
        self.version = "1.0.0"
        self.supported_tasks = [
            "enrich_context",
            "retrieve_memories",
            "expand_knowledge_graph",
            "multi_scale_embedding"
        ]
        self.confidence_range = (0.55, 0.88)

        # HoloLoom orchestrator
        from hololoom.weaving_orchestrator import WeavingOrchestrator
        from hololoom.config import Config
        self.orchestrator = WeavingOrchestrator(cfg=Config.fused())

        # Memory systems
        from hololoom.reflection.buffer import ReflectionBuffer
        self.medium_term_memory = ReflectionBuffer(
            capacity=1000,
            persist_path="./context_sessions"
        )

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute context enrichment"""
        # (See HOLOLOOM_CONTEXT_DEPARTMENT_MAPPING.md for full implementation)
        ...

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Verify context quality"""
        ...

    async def refine(...) -> DepartmentResponse:
        """Apply refinements"""
        ...

    async def update_strategy(...):
        """Learn from patterns"""
        ...
```

**Status**: ✅ Fully specified in [HOLOLOOM_CONTEXT_DEPARTMENT_MAPPING.md](HOLOLOOM_CONTEXT_DEPARTMENT_MAPPING.md)

#### 2. Tests (`tests/departments/test_context.py`)
**Lines**: ~500

```python
@pytest.mark.asyncio
async def test_context_execute():
    """Context department executes with HoloLoom"""
    dept = ContextDepartment(DepartmentConfig())

    request = DepartmentRequest(
        task_id="ctx_001",
        task_type="enrich_context",
        parameters={"query": "What is Thompson Sampling?"},
        confidence_expected=0.75,
        context_preference="moderate",
        privacy_level="public"
    )

    response = await dept.execute(request)

    assert response.task_id == "ctx_001"
    assert 0.55 <= response.confidence.score <= 0.88
    assert response.result is not None

@pytest.mark.asyncio
async def test_context_verification_loop():
    """Context department can verify and refine"""
    dept = ContextDepartment(DepartmentConfig())

    request = DepartmentRequest(
        task_id="ctx_002",
        task_type="enrich_context",
        parameters={"query": "Ambiguous query with low confidence"},
        confidence_expected=0.85,
        context_preference="auto",
        privacy_level="public"
    )

    response = await dept.execute(request)
    verification = await dept.verify(response)

    if not verification.sufficient:
        refined = await dept.refine(request, response, verification)
        assert refined.confidence.score > response.confidence.score

@pytest.mark.asyncio
async def test_context_learning():
    """Context department learns from patterns"""
    dept = ContextDepartment(DepartmentConfig())

    # Execute 10 tasks
    for i in range(10):
        request = DepartmentRequest(
            task_id=f"ctx_{i}",
            task_type="enrich_context",
            parameters={"query": f"Query {i}"},
            confidence_expected=0.75,
            context_preference="auto",
            privacy_level="public"
        )
        await dept.execute(request)

    # Update strategy
    await dept.update_strategy([])

    # Check institutional memory
    memory = await dept.get_institutional_memory("successful_strategies")
    assert "preferred_refinements" in memory
```

### Tasks (Week 3-4)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| Implement `execute()` | 2 days | Dev | Wraps HoloLoom orchestrator |
| Implement `verify()` | 1 day | Dev | Quality checks |
| Implement `refine()` | 1 day | Dev | Integrate recursive refinement |
| Implement `update_strategy()` | 1 day | Dev | Pattern learning |
| Memory system integration | 2 days | Dev | Short/medium/long-term |
| Write tests | 2 days | Dev | `test_context.py` (500 lines) |
| Integration testing | 1 day | Dev | Test with real HoloLoom |

**Estimated Effort**: 10 days (2 weeks) - Context Department only

---

### Additional Departments (Week 3-4 Extension)

**Note**: The original Week 3-4 plan focused on Context Department only. Adding RAG, Planning, Orchestration, and Infrastructure departments extends this to **Week 3-5** (3 weeks total).

#### 2. RAG Department (`hololoom/departments/rag.py`)
**Lines**: ~500

**Key Features**:
- Multi-scale Matryoshka embeddings (96, 192, 384 dims)
- BM25 + semantic hybrid retrieval
- Citation tracking for responses
- Confidence calibration based on retrieval quality

#### 3. Planning Department (`hololoom/departments/planning.py`)
**Lines**: ~450

**Key Features**:
- Hierarchical task decomposition
- Dependency detection between steps
- Resource estimation per step
- Plan validation and feasibility checks

#### 4. Orchestration Department (`hololoom/departments/orchestration.py`)
**Lines**: ~550

**Key Features**:
- Task routing based on department capabilities
- Parallel execution coordination
- Result aggregation with confidence weighting
- Fallback strategies for department failures

#### 5. Infrastructure Department (`hololoom/departments/infrastructure.py`)
**Lines**: ~400

**Key Features**:
- Zero-copy memory-mapped data access
- Performance profiling across departments
- Resource usage tracking
- Health check aggregation

---

### Integration & Documentation (Week 3-5)

#### Integration Testing
**Tests**: `tests/integration/test_multi_department.py` (~600 lines)

**Key Test Scenarios**:
- Planning → RAG workflow (multi-step research)
- Orchestration routes to appropriate departments
- Confidence aggregation across department chains
- Fallback behavior when departments fail
- Privacy envelope handling across departments

#### Developer Documentation

**Deliverables**:
1. **Developer Guide** (`hololoom/DEVELOPER_GUIDE.md`) - ~1,500 lines
   - How to build a custom department
   - Protocol requirements and best practices
   - Testing strategies
   - Example: Building a healthcare-specific department

2. **API Reference** (`hololoom/API_REFERENCE.md`) - ~800 lines
   - Complete protocol API documentation
   - All dataclass fields and methods
   - Type signatures and examples

3. **Architecture Diagrams** (`hololoom/ARCHITECTURE_DIAGRAMS.md`) - ~400 lines
   - Department interaction flows
   - Confidence negotiation sequences
   - DS-STAR verification loop
   - Multi-department orchestration patterns

---

### Revised Tasks (Week 3-5)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| **Core Departments (5 total)** | | | |
| Context Department | 2 days | Dev | Wraps HoloLoom orchestrator (DONE Week 3-4) |
| RAG Department | 2 days | Dev | Multi-scale retrieval + generation |
| Planning Department | 1.5 days | Dev | Goal decomposition + validation |
| Orchestration Department | 2 days | Dev | Task routing + coordination |
| Infrastructure Department | 1.5 days | Dev | Zero-copy + monitoring |
| **Integration & Testing** | | | |
| Unit tests (5 × ~200 lines) | 2 days | Dev | `test_*.py` for each department |
| Integration tests | 2 days | Dev | Multi-department workflows |
| End-to-end tests | 1 day | Dev | Complete pipeline validation |
| **Documentation** | | | |
| Developer Guide | 1.5 days | Dev | How to build custom departments |
| API Reference | 1 day | Dev | Complete protocol documentation |
| Architecture Diagrams | 0.5 days | Dev | Visual documentation |

**Estimated Effort**: 16 days (3+ weeks, Week 3-5)

---

### Success Criteria (Week 3-5)

- ✅ **Week 1-2 Complete**: Core framework (protocol, base, registry) - DONE November 13, 2025
- ⏳ **Week 3-5 Goals**:
  - All 5 core departments implement `Department` protocol
  - Department-to-department communication working
  - All departments report accurate confidence metadata
  - DS-STAR verification working for each department
  - Integration tests passing (100% coverage)
  - Developer guide complete with examples
  - API reference published

---

### Next Steps (Week 6+)

After Week 3-5 core departments, proceed with:

#### Week 6-7: Beekeeping Suite (First Vertical)
- **Goal**: Build beekeeping-specific departments using core framework
- **MasterWeaver Department**: Beekeeping entity extraction
- **Hive Monitoring Workflow**: Audio → entities → insights pipeline
- **Target**: $1,200/yr beekeeping SaaS product
- **Validation**: Domain expert feedback

#### Week 8-9: Healthcare Vertical Exploration
- **Goal**: Validate framework with second vertical (healthcare)
- **Patient Data Departments**: HIPAA-compliant data handling
- **Medical Knowledge Graph**: Clinical decision support
- **Privacy Envelope**: TEE integration for HIPAA compliance
- **Target**: $10,000/yr healthcare suite

#### Week 10-11: B2B Marketplace Preparation
- **Goal**: Enable third-party developers to build departments
- **Department Packaging**: Docker + deployment automation
- **Pricing Models**: Usage-based, tier-based, enterprise
- **Developer Onboarding**: Documentation + examples
- **Marketplace Portal**: Department discovery + installation

#### Week 12: Production Deployment
- **Goal**: Ship beekeeping beta to first customers
- **Infrastructure**: Kubernetes, load balancing, monitoring
- **Load Testing**: 1,000 concurrent users target
- **Monitoring**: Performance metrics, error tracking
- **Launch**: Beekeeping beta with 10 pilot customers

---

## Week 5-6: MasterWeaver Department (Beekeeping)

### Goal
Build domain-specific department for extracting beekeeping knowledge from audio transcripts.

### Deliverables

#### 1. MasterWeaver Department (`hololoom/departments/beekeeping/masterweaver.py`)
**Lines**: ~800

```python
class MasterWeaverDepartment(BaseDepartment):
    """Beekeeping entity extraction from audio transcripts"""

    def __init__(self, config: DepartmentConfig):
        super().__init__(config)
        self.name = "MasterWeaver"
        self.domain = "beekeeping"
        self.version = "1.0.0"
        self.supported_tasks = [
            "extract_entities",
            "classify_behavior",
            "identify_problems",
            "suggest_actions"
        ]
        self.confidence_range = (0.40, 0.75)  # Lower confidence (learning domain)

        # SpinningWheel for transcript processing
        from hololoom.spinningWheel import AudioSpinner
        self.audio_spinner = AudioSpinner()

        # LLM for entity extraction (Ollama or OpenAI)
        self.llm_client = self._initialize_llm(config)

        # Entity taxonomy (beekeeping-specific)
        self.entity_types = {
            "queen_behavior": ["laying", "supercedure", "swarming"],
            "colony_health": ["strong", "weak", "queenless"],
            "problems": ["disease", "pest", "starvation"],
            "actions": ["feed", "treat", "split", "requeen"]
        }

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Extract beekeeping entities from transcript"""

        transcript = request.parameters.get("transcript", "")

        # Step 1: Process audio transcript
        shards = await self.audio_spinner.spin({"transcript": transcript})

        # Step 2: Extract entities using LLM
        entities = await self._extract_entities_llm(shards)

        # Step 3: Validate entities against taxonomy
        validated_entities = self._validate_entities(entities)

        # Step 4: Calculate confidence
        confidence = self._calculate_extraction_confidence(validated_entities, entities)

        # Build response
        response = DepartmentResponse(
            task_id=request.task_id,
            result=validated_entities,
            confidence=ConfidenceMetadata(
                score=confidence,
                justification=[
                    f"Extracted {len(validated_entities)} entities",
                    f"Taxonomy match: {len(validated_entities) / max(len(entities), 1) * 100:.0f}%"
                ],
                uncertainty_sources=[
                    "Transcript quality" if confidence < 0.60 else "",
                    "Novel entity types" if len(entities) > len(validated_entities) else ""
                ]
            ),
            detail_level=self._determine_detail_level(confidence, request.context_preference),
            learning_signals={
                "novel_entities": [e for e in entities if e not in validated_entities],
                "taxonomy_coverage": len(validated_entities) / len(self._get_all_taxonomy_entities())
            }
        )

        return response

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Verify entity extraction quality"""

        entities = response.result
        confidence = response.confidence.score

        # Check 1: Minimum entities extracted
        min_entities = 3
        if len(entities) < min_entities:
            return VerificationResult(
                sufficient=False,
                confidence_valid=False,
                reasoning_sound=False,
                alternative_paths=["Re-process with more context"],
                refinement_suggestions={"expand_context": True}
            )

        # Check 2: Taxonomy coverage
        taxonomy_coverage = response.learning_signals["taxonomy_coverage"]
        if taxonomy_coverage < 0.50:
            return VerificationResult(
                sufficient=False,
                confidence_valid=True,
                reasoning_sound=False,
                alternative_paths=["Expand entity taxonomy"],
                refinement_suggestions={"learn_new_entities": True}
            )

        # Check 3: Confidence matches entity count
        expected_confidence = min(0.75, 0.40 + (len(entities) / 10) * 0.35)
        if abs(confidence - expected_confidence) > 0.15:
            return VerificationResult(
                sufficient=False,
                confidence_valid=False,
                reasoning_sound=True,
                alternative_paths=["Recalibrate confidence model"],
                refinement_suggestions={"calibrate": True}
            )

        return VerificationResult(
            sufficient=True,
            confidence_valid=True,
            reasoning_sound=True,
            alternative_paths=[]
        )

    async def refine(...):
        """Apply refinements (expand context, learn new entities)"""
        ...

    async def update_strategy(...):
        """Learn from extraction patterns"""
        # Analyze novel entities
        # Expand taxonomy if patterns emerge
        # Recalibrate confidence model
        ...

    def _initialize_llm(self, config):
        """Initialize LLM client (Ollama or OpenAI)"""
        import os
        lm_model = os.getenv("LM_MODEL", "ollama/llama3.2:3b")

        if lm_model.startswith("ollama/"):
            import ollama
            return ollama.Client()
        else:
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def _extract_entities_llm(self, shards):
        """Use LLM to extract entities"""
        # Build prompt with taxonomy
        prompt = f"""Extract beekeeping entities from this inspection note.

Taxonomy:
{json.dumps(self.entity_types, indent=2)}

Transcript:
{shards[0].content}

Return JSON: {{"entities": [{{"type": "...", "value": "...", "confidence": 0.0-1.0}}]}}
"""

        response = await self._call_llm(prompt)
        entities = json.loads(response)["entities"]
        return entities

    def _validate_entities(self, entities):
        """Validate extracted entities against taxonomy"""
        validated = []
        for entity in entities:
            if self._is_valid_entity(entity):
                validated.append(entity)
        return validated

    def _calculate_extraction_confidence(self, validated, all_entities):
        """Calculate confidence in extraction"""
        if not all_entities:
            return 0.40  # Minimum confidence

        # Base confidence on validation rate
        validation_rate = len(validated) / len(all_entities)
        base_confidence = 0.40 + (validation_rate * 0.35)  # 0.40-0.75 range

        return min(0.75, base_confidence)  # Cap at upper range
```

#### 2. Beekeeping Taxonomy (`hololoom/departments/beekeeping/taxonomy.json`)
**Lines**: ~200

```json
{
  "queen_behavior": {
    "laying": ["laying well", "good pattern", "solid brood"],
    "supercedure": ["queen cells", "emergency cells"],
    "swarming": ["queen cells", "congested", "backfilling"]
  },
  "colony_health": {
    "strong": ["10+ frames", "good population", "active"],
    "weak": ["<5 frames", "low population", "listless"],
    "queenless": ["no eggs", "laying workers", "multiple eggs"]
  },
  "problems": {
    "disease": ["foulbrood", "nosema", "chalkbrood"],
    "pest": ["varroa", "hive beetle", "wax moth"],
    "starvation": ["no stores", "light hive", "clustering"]
  },
  "actions": {
    "feed": ["sugar water", "fondant", "pollen patty"],
    "treat": ["oxalic acid", "formic acid", "apivar"],
    "split": ["divide", "make nuc", "walk-away split"],
    "requeen": ["new queen", "queen cell", "combine"]
  }
}
```

#### 3. Tests (`tests/departments/beekeeping/test_masterweaver.py`)
**Lines**: ~400

```python
@pytest.mark.asyncio
async def test_masterweaver_extract_entities():
    """MasterWeaver extracts beekeeping entities"""
    dept = MasterWeaverDepartment(DepartmentConfig())

    transcript = """
    Inspected hive 5 today. Queen laying well with good pattern.
    Saw 10 frames of bees. Added a super. No signs of varroa.
    """

    request = DepartmentRequest(
        task_id="mw_001",
        task_type="extract_entities",
        parameters={"transcript": transcript},
        confidence_expected=0.65,
        context_preference="detailed",
        privacy_level="tee_only"
    )

    response = await dept.execute(request)

    # Check entities extracted
    entities = response.result
    assert len(entities) >= 3

    # Check entity types
    entity_types = [e["type"] for e in entities]
    assert "queen_behavior" in entity_types
    assert "colony_health" in entity_types

    # Check confidence in range
    assert 0.40 <= response.confidence.score <= 0.75
```

### Tasks (Week 5-6)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| Define beekeeping taxonomy | 2 days | Domain Expert | `taxonomy.json` (200 lines) |
| Implement LLM integration | 2 days | Dev | Ollama + OpenAI clients |
| Implement entity extraction | 3 days | Dev | `_extract_entities_llm()` |
| Implement validation | 1 day | Dev | `_validate_entities()` |
| Implement verification | 1 day | Dev | `verify()` method |
| Write tests | 2 days | Dev | `test_masterweaver.py` (400 lines) |
| Manual testing with real data | 1 day | Domain Expert | Validation report |

**Estimated Effort**: 12 days (2.4 weeks)

---

## Week 7-8: Infrastructure Department

### Goal
Zero-copy data access with confidence-aware querying.

### Deliverables

#### 1. Infrastructure Department (`hololoom/departments/beekeeping/infrastructure.py`)
**Lines**: ~500

```python
class InfrastructureDepartment(BaseDepartment):
    """Zero-copy data access for beekeeping datasets"""

    def __init__(self, config: DepartmentConfig):
        super().__init__(config)
        self.name = "Infrastructure"
        self.domain = "beekeeping"
        self.version = "1.0.0"
        self.supported_tasks = [
            "query_neo4j",
            "query_qdrant",
            "diagnose_performance",
            "provision_access"
        ]
        self.confidence_range = (0.70, 0.95)  # Higher confidence (deterministic)

        # Database clients
        from hololoom.memory.neo4j_graph import Neo4jGraph
        from hololoom.memory.qdrant_backend import QdrantBackend

        self.neo4j = Neo4jGraph(uri=config.neo4j_uri)
        self.qdrant = QdrantBackend(host=config.qdrant_host)

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute database query with zero-copy"""

        task_type = request.task_type

        if task_type == "query_neo4j":
            result = await self._query_neo4j(request.parameters)
            confidence = 0.95  # Deterministic query
        elif task_type == "query_qdrant":
            result = await self._query_qdrant(request.parameters)
            confidence = 0.90  # Semantic similarity has some uncertainty
        elif task_type == "diagnose_performance":
            result = await self._diagnose_performance()
            confidence = 0.85  # Performance analysis has interpretation
        else:
            raise ValueError(f"Unsupported task: {task_type}")

        return DepartmentResponse(
            task_id=request.task_id,
            result=result,
            confidence=ConfidenceMetadata(
                score=confidence,
                justification=[
                    f"Task type: {task_type}",
                    "Deterministic query" if task_type == "query_neo4j" else "Semantic similarity"
                ],
                uncertainty_sources=[]
            ),
            detail_level=self._determine_detail_level(confidence, request.context_preference),
            learning_signals={
                "query_time_ms": 0,  # TODO: Track query time
                "cache_hit": False
            }
        )

    async def _query_neo4j(self, parameters):
        """Execute Cypher query"""
        query = parameters["query"]
        params = parameters.get("params", {})
        return await self.neo4j.execute_query(query, params)

    async def _query_qdrant(self, parameters):
        """Execute vector similarity search"""
        vector = parameters["vector"]
        collection = parameters["collection"]
        limit = parameters.get("limit", 10)
        return await self.qdrant.search(collection, vector, limit=limit)

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        """Verify query results"""
        # Infrastructure queries are typically deterministic
        # Verify: results non-empty, confidence matches task type
        result = response.result

        if not result:
            return VerificationResult(
                sufficient=False,
                confidence_valid=False,
                reasoning_sound=False,
                alternative_paths=["Check query syntax"],
                refinement_suggestions={"validate_query": True}
            )

        return VerificationResult(
            sufficient=True,
            confidence_valid=True,
            reasoning_sound=True,
            alternative_paths=[]
        )
```

### Tasks (Week 7-8)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| Implement Neo4j integration | 2 days | Dev | `_query_neo4j()` |
| Implement Qdrant integration | 2 days | Dev | `_query_qdrant()` |
| Implement performance diagnostics | 1 day | Dev | `_diagnose_performance()` |
| Implement verification | 1 day | Dev | `verify()` method |
| Write tests | 2 days | Dev | `test_infrastructure.py` |
| Docker setup | 1 day | Dev | Neo4j + Qdrant containers |
| Manual testing | 1 day | Dev | Validation report |

**Estimated Effort**: 10 days (2 weeks)

---

## Week 9-10: Verification + Orchestration Departments

### Goal
Build cross-department validation and task routing.

### Deliverables

#### 1. Verification Department (`hololoom/departments/verification.py`)
**Lines**: ~600

```python
class VerificationDepartment(BaseDepartment):
    """Cross-department validation and confidence checking"""

    def __init__(self, config: DepartmentConfig):
        super().__init__(config)
        self.name = "Verification"
        self.domain = "generic"
        self.version = "1.0.0"
        self.supported_tasks = [
            "validate_response",
            "check_confidence",
            "detect_overconfidence",
            "suggest_refinements"
        ]
        self.confidence_range = (0.60, 0.90)

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Validate another department's response"""

        # Get response to validate
        other_response: DepartmentResponse = request.parameters["response"]
        expectations = request.parameters.get("expectations", {})

        # Run validation checks
        checks = {
            "confidence_valid": self._check_confidence_validity(other_response),
            "reasoning_sound": self._check_reasoning_soundness(other_response),
            "meets_expectations": self._check_meets_expectations(other_response, expectations),
            "detail_level_appropriate": self._check_detail_level(other_response)
        }

        # Calculate overall validation confidence
        validation_confidence = sum(checks.values()) / len(checks)

        # Build verification result
        result = VerificationResult(
            sufficient=all(checks.values()),
            confidence_valid=checks["confidence_valid"],
            reasoning_sound=checks["reasoning_sound"],
            alternative_paths=self._generate_alternative_paths(checks, other_response),
            refinement_suggestions=self._generate_refinement_suggestions(checks, other_response) if not all(checks.values()) else None,
            escalation_needed=validation_confidence < 0.50
        )

        return DepartmentResponse(
            task_id=request.task_id,
            result=result,
            confidence=ConfidenceMetadata(
                score=validation_confidence,
                justification=[f"{k}: {v}" for k, v in checks.items()],
                uncertainty_sources=[]
            ),
            detail_level="detailed",
            learning_signals={"checks": checks}
        )

    def _check_confidence_validity(self, response: DepartmentResponse) -> bool:
        """Check if confidence matches quality indicators"""
        # Compare reported confidence vs actual quality
        # (Quality heuristics: result completeness, reasoning depth, etc.)
        return True  # TODO: Implement heuristics

    def _check_reasoning_soundness(self, response: DepartmentResponse) -> bool:
        """Check if reasoning chain is logical"""
        if not response.reasoning:
            return response.confidence.score >= 0.90  # High confidence can skip reasoning
        # TODO: Check reasoning for logical consistency
        return True

    def _check_meets_expectations(self, response: DepartmentResponse, expectations: Dict) -> bool:
        """Check if response meets requester's expectations"""
        expected_conf = expectations.get("confidence_expected", 0.0)
        return response.confidence.score >= expected_conf - 0.10  # Allow 10% tolerance

    def _check_detail_level(self, response: DepartmentResponse) -> bool:
        """Check if detail level matches confidence"""
        confidence = response.confidence.score
        detail = response.detail_level

        expected_detail = self._determine_detail_level(confidence, "auto")
        return detail == expected_detail
```

#### 2. Orchestration Department (`hololoom/departments/orchestration.py`)
**Lines**: ~700

```python
class OrchestrationDepartment(BaseDepartment):
    """Task routing and multi-department coordination"""

    def __init__(self, config: DepartmentConfig, registry: DepartmentRegistry):
        super().__init__(config)
        self.name = "Orchestration"
        self.domain = "generic"
        self.version = "1.0.0"
        self.supported_tasks = [
            "route_task",
            "coordinate_workflow",
            "manage_consensus"
        ]
        self.confidence_range = (0.80, 0.98)

        self.registry = registry

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Route task to appropriate department"""

        task_type = request.parameters["task_type"]
        task_request = request.parameters["task_request"]

        # Find department that supports this task
        dept = self._route_to_department(task_type, task_request)

        if not dept:
            return DepartmentResponse(
                task_id=request.task_id,
                result=None,
                confidence=ConfidenceMetadata(
                    score=0.30,
                    justification=["No department found for task"],
                    uncertainty_sources=["Unknown task type"]
                ),
                detail_level="minimal",
                learning_signals={"routing_failed": True}
            )

        # Execute task with department
        dept_response = await dept.execute(task_request)

        # Verify response
        verification_request = DepartmentRequest(
            task_id=f"{request.task_id}_verify",
            task_type="validate_response",
            parameters={"response": dept_response},
            confidence_expected=0.75,
            context_preference="moderate",
            privacy_level=request.privacy_level
        )

        verification_dept = self.registry.get("generic", "Verification")
        verification_response = await verification_dept.execute(verification_request)
        verification_result = verification_response.result

        # If insufficient, trigger refinement
        if not verification_result.sufficient:
            dept_response = await dept.refine(
                task_request,
                dept_response,
                verification_result
            )

        return DepartmentResponse(
            task_id=request.task_id,
            result=dept_response,
            confidence=ConfidenceMetadata(
                score=0.90,  # High confidence in routing
                justification=[f"Routed to {dept.name}", "Verified and refined"],
                uncertainty_sources=[]
            ),
            detail_level="moderate",
            learning_signals={
                "routed_to": dept.name,
                "verification": verification_result.sufficient,
                "refined": not verification_result.sufficient
            }
        )

    def _route_to_department(self, task_type: str, task_request: DepartmentRequest) -> Optional[Department]:
        """Find department that supports task"""
        # Discover departments that support this task
        manifests = self.registry.discover(task_type=task_type)

        if not manifests:
            return None

        # Pick department (for now, just first match)
        # TODO: More intelligent routing (confidence, load balancing, etc.)
        manifest = manifests[0]
        return self.registry.get(manifest.domain, manifest.name)
```

### Tasks (Week 9-10)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| Implement Verification department | 3 days | Dev | `verification.py` (600 lines) |
| Implement Orchestration department | 3 days | Dev | `orchestration.py` (700 lines) |
| Write tests | 2 days | Dev | Test files (500 lines) |
| Integration testing | 2 days | Dev | Multi-department workflows |

**Estimated Effort**: 10 days (2 weeks)

---

## Week 11-12: Integration + End-to-End Testing

### Goal
Full beekeeping workflow running end-to-end with all departments.

### Deliverables

#### 1. End-to-End Workflow (`demos/demo_beekeeping_workflow.py`)
**Lines**: ~300

```python
async def demo_beekeeping_workflow():
    """Complete beekeeping workflow: audio → entities → context → insights"""

    # Setup
    registry = DepartmentRegistry()

    # Register departments
    context = ContextDepartment(DepartmentConfig())
    masterweaver = MasterWeaverDepartment(DepartmentConfig())
    infrastructure = InfrastructureDepartment(DepartmentConfig())
    verification = VerificationDepartment(DepartmentConfig())
    orchestration = OrchestrationDepartment(DepartmentConfig(), registry)

    registry.register(context.manifest(), context)
    registry.register(masterweaver.manifest(), masterweaver)
    registry.register(infrastructure.manifest(), infrastructure)
    registry.register(verification.manifest(), verification)
    registry.register(orchestration.manifest(), orchestration)

    # Step 1: Extract entities from inspection audio
    transcript = "Inspected hive 5. Queen laying well, 10 frames of bees. No varroa."

    masterweaver_request = DepartmentRequest(
        task_id="demo_001_extract",
        task_type="extract_entities",
        parameters={"transcript": transcript},
        confidence_expected=0.65,
        context_preference="detailed",
        privacy_level="tee_only"
    )

    entities_response = await orchestration.execute(DepartmentRequest(
        task_id="demo_001_route",
        task_type="route_task",
        parameters={
            "task_type": "extract_entities",
            "task_request": masterweaver_request
        },
        confidence_expected=0.85,
        context_preference="moderate",
        privacy_level="tee_only"
    ))

    entities = entities_response.result.result
    print(f"Extracted {len(entities)} entities with confidence {entities_response.result.confidence.score:.2f}")

    # Step 2: Enrich with context
    context_request = DepartmentRequest(
        task_id="demo_001_context",
        task_type="enrich_context",
        parameters={
            "query": f"Provide insights for: {entities}",
            "entities": entities
        },
        confidence_expected=0.75,
        context_preference="detailed",
        privacy_level="public"
    )

    context_response = await orchestration.execute(DepartmentRequest(
        task_id="demo_001_route_context",
        task_type="route_task",
        parameters={
            "task_type": "enrich_context",
            "task_request": context_request
        },
        confidence_expected=0.85,
        context_preference="moderate",
        privacy_level="public"
    ))

    insights = context_response.result.result
    print(f"Generated insights with confidence {context_response.result.confidence.score:.2f}")
    print(insights)

    # Step 3: Store to knowledge graph
    infrastructure_request = DepartmentRequest(
        task_id="demo_001_store",
        task_type="query_neo4j",
        parameters={
            "query": "CREATE (i:Inspection {date: $date, hive: $hive}) RETURN i",
            "params": {"date": "2025-11-09", "hive": "hive_5"}
        },
        confidence_expected=0.95,
        context_preference="minimal",
        privacy_level="confidential"
    )

    store_response = await orchestration.execute(DepartmentRequest(
        task_id="demo_001_route_store",
        task_type="route_task",
        parameters={
            "task_type": "query_neo4j",
            "task_request": infrastructure_request
        },
        confidence_expected=0.95,
        context_preference="minimal",
        privacy_level="confidential"
    ))

    print(f"Stored to knowledge graph with confidence {store_response.result.confidence.score:.2f}")

if __name__ == "__main__":
    asyncio.run(demo_beekeeping_workflow())
```

#### 2. Integration Tests (`tests/integration/test_full_workflow.py`)
**Lines**: ~400

```python
@pytest.mark.asyncio
async def test_full_beekeeping_workflow():
    """End-to-end workflow runs successfully"""
    # (Same as demo, but with assertions)
    ...

@pytest.mark.asyncio
async def test_confidence_negotiation():
    """Departments negotiate confidence correctly"""
    # Test: Low confidence triggers verification → refinement
    ...

@pytest.mark.asyncio
async def test_learning_rates():
    """Departments update at different learning rates"""
    # Test: High confidence → weekly updates, Low confidence → per-task updates
    ...

@pytest.mark.asyncio
async def test_privacy_envelope():
    """Sensitive data stays in TEE"""
    # Test: MasterWeaver processes transcript in TEE, only outputs entities
    ...
```

#### 3. Documentation (`hololoom/PHASE_1_COMPLETE.md`)
**Lines**: ~500

Summary of Phase 1 achievements, known issues, next steps for Phase 2.

### Tasks (Week 11-12)

| Task | Duration | Owner | Output |
|------|----------|-------|--------|
| Build end-to-end demo | 2 days | Dev | `demo_beekeeping_workflow.py` |
| Write integration tests | 3 days | Dev | `test_full_workflow.py` (400 lines) |
| Manual testing with real data | 2 days | Domain Expert | Validation report |
| Performance benchmarking | 1 day | Dev | Latency, throughput metrics |
| Documentation | 2 days | Dev | `PHASE_1_COMPLETE.md` (500 lines) |
| Code review | 1 day | Team | Feedback incorporated |
| Bug fixes | 1 day | Dev | Fix issues found in testing |

**Estimated Effort**: 12 days (2.4 weeks)

---

## Phase 1 Summary

### Total Effort
- **Week 1-2**: Core Framework (8 days)
- **Week 3-4**: Context Department (10 days)
- **Week 5-6**: MasterWeaver Department (12 days)
- **Week 7-8**: Infrastructure Department (10 days)
- **Week 9-10**: Verification + Orchestration (10 days)
- **Week 11-12**: Integration + Testing (12 days)

**Total**: 62 days (~12.4 weeks, or 3.1 months with buffer)

### Deliverables
1. **Generic department framework** (protocol, base class, registry)
2. **5 departments**:
   - Context (generic)
   - MasterWeaver (beekeeping)
   - Infrastructure (beekeeping)
   - Verification (generic)
   - Orchestration (generic)
3. **Complete test suite** (~2,200 lines)
4. **End-to-end workflow** (beekeeping audio → insights)
5. **Documentation** (~2,000 lines)

### Success Criteria

✅ **Beekeeping workflow runs end-to-end**
- Audio transcript → Entities → Context → Insights → Knowledge graph

✅ **Confidence negotiation working**
- Departments report confidence accurately
- Low confidence triggers verification → refinement
- High confidence runs with minimal detail

✅ **DS-STAR verification loop operational**
- Verify → Refine → Learn cycle works
- Router intelligently selects refinement strategies

✅ **Multi-timescale learning**
- High confidence departments update weekly
- Low confidence departments update per-task
- Learning signals accumulate in institutional memory

✅ **Privacy envelope functional**
- MasterWeaver processes sensitive transcripts in TEE
- Only privacy-preserved insights shared
- Verifiable output generated

---

## Phase 2 Preview

**Goal**: B2B marketplace + additional domain sets

**Deliverables**:
1. **Department Marketplace**:
   - Web UI for discovering departments
   - One-click installation
   - Third-party department support
   - Rating & reviews

2. **Additional Domain Sets**:
   - Healthcare departments (extract medical entities, HIPAA compliance)
   - Finance departments (transaction analysis, risk modeling)
   - Manufacturing departments (quality control, supply chain)

3. **Cross-Domain Features**:
   - Department composition (workflows span domains)
   - Federated learning (departments learn from each other)
   - Multi-tenant architecture

**Duration**: 16 weeks (4 months)

---

**Next Document**: [B2B_PRODUCT_ARCHITECTURE.md](B2B_PRODUCT_ARCHITECTURE.md)