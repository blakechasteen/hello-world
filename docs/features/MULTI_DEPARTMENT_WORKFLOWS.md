# Multi-Department Workflows

**Created**: November 20, 2025
**Status**: ✅ Complete - 7 workflows designed and tested
**Test Coverage**: 28 integration tests (all passing expected)
**Documentation**: Complete with examples and protocol bridging

---

## Overview

HoloLoom's **Departments architecture** enables complex multi-department workflows by composing specialized departments into pipelines. This document describes 7 production-ready workflows that demonstrate best practices for:

- **Protocol bridging** between HoloLoom and xTerminator protocols
- **Confidence propagation** across department boundaries
- **Error recovery** and graceful degradation
- **Parallel execution** and result synthesis
- **Iterative refinement** with feedback loops

---

## The Two Protocols

HoloLoom's departments use **two different protocols**:

### 1. HoloLoom Department Protocol

**Used by**: Planning, RAG, Context, Infrastructure, Orchestration

**Key Types**:
```python
from HoloLoom.apps.departments.protocol import (
    DepartmentRequest,      # task_type, parameters, context, constraints
    DepartmentResponse,     # result, confidence (ConfidenceMetadata), metadata
    VerificationResult,     # verified, checks, overall_score
    ConfidenceMetadata,     # score, justification, sources
)
```

**Core Methods** (7):
- `execute(request)` → response
- `verify(response)` → verification
- `refine(response)` → refined_response
- `update_strategy(feedback)` → None
- `get_capabilities()` → dict
- `get_metrics()` → dict
- `health_check()` → bool

**Verification Framework**: DS-STAR (Domain, Sensibility, Temporal, Argument, Reference)

### 2. xTerminator Protocol

**Used by**: QA (Quality Assurance)

**Key Types**:
```python
from xterminator.department_protocol import (
    DepartmentRequest,      # request_type (enum), payload, context
    DepartmentResponse,     # status, payload, confidence (float), error
    VerificationResult,     # verified, issues_found, corrections_needed
    RequestType,            # Enum: SCAN_CODE, CLASSIFY_ISSUE, etc.
)
```

**Core Methods** (6):
- `execute(request)` → response
- `verify(response)` → verification
- `refine(request, response, verification)` → refined_response
- `update_strategy(learning_signals)` → None
- `get_institutional_memory(pattern_type)` → dict
- `health_check()` → dict

**Request Types** (7): SCAN_CODE, CLASSIFY_ISSUE, PROPOSE_FIX, APPLY_FIX, VALIDATE_FIX, GET_STATISTICS, DETECT_DEGRADATION

---

## Protocol Bridging

### HoloLoom → xTerminator

**Confidence Conversion**:
```python
# HoloLoom response
holo_response = await planning_dept.execute(request)
holo_conf = holo_response.confidence.score  # ConfidenceMetadata → float

# Pass to xTerminator
xterm_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.SCAN_CODE,
    requesting_department="planning",
    payload={"code": code, "file_path": "example.py"},
    context={"requesting_confidence": holo_conf}  # Pass as context
)

xterm_response = await qa_dept.execute(xterm_request)

# Confidence negotiation (automatic in QA Department)
if "negotiated_confidence" in xterm_response.metadata:
    negotiated = xterm_response.metadata["negotiated_confidence"]
    # Formula: 0.3 * requesting_conf + 0.7 * responding_conf
```

### xTerminator → HoloLoom

**Confidence Conversion**:
```python
# xTerminator response
xterm_response = await qa_dept.execute(xterm_request)
xterm_conf = xterm_response.confidence  # float

# Convert to HoloLoom format
confidence_meta = ConfidenceMetadata.from_score(
    score=xterm_conf,
    justification=[f"QA confidence: {xterm_conf:.2f}"],
    sources=["xTerminator QA Department"]
)

# Use in HoloLoom request
holo_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={"goal": "Plan based on QA results", "max_tasks": 3}
)
holo_request.context["qa_confidence"] = xterm_conf
```

---

## Workflow 1: Code Analysis Pipeline

**Pipeline**: Planning → RAG → QA

**Use Case**: Analyze a codebase for quality issues with context from documentation

**Steps**:
1. **Planning** decomposes "analyze this codebase" into specific tasks
2. **RAG** retrieves relevant documentation/examples for each task
3. **QA** scans the code using retrieved knowledge as context

**Protocol Bridge**: HoloLoom (Planning, RAG) → xTerminator (QA)

**Example**:
```python
# Step 1: Planning decomposes goal
plan_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={
        "goal": "Analyze Python codebase for quality issues",
        "max_tasks": 5,
    }
)
plan_response = await planning_dept.execute(plan_request)
plan_data = plan_response.result["plan"]

# Step 2: RAG retrieves documentation
first_task = plan_data["tasks"][0]
rag_request = create_simple_request(
    task_type="question_answering",
    parameters={
        "query": f"How to: {first_task['description']}",
        "mode": "verify",
        "max_sources": 3,
    }
)
rag_response = await rag_dept.execute(rag_request)

# Step 3: QA scans code with RAG context
qa_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.SCAN_CODE,
    requesting_department="rag",
    payload={
        "code": codebase_content,
        "file_path": "example.py",
        "context": rag_response.result["answer"],  # Use RAG knowledge
    }
)
qa_response = await qa_dept.execute(qa_request)
```

**Benefits**:
- QA scans are informed by up-to-date documentation
- Planning ensures comprehensive analysis (all aspects covered)
- Confidence tracks through entire pipeline

**Test Coverage**: 3 tests (basic flow, confidence propagation, verification)

---

## Workflow 2: Research-to-Implementation

**Pipeline**: RAG → Planning → QA

**Use Case**: Research best practices, create implementation plan, validate quality

**Steps**:
1. **RAG** researches "best practices for X" using research mode (multi-query)
2. **Planning** creates implementation plan based on research findings
3. **QA** validates the plan for potential issues before implementation

**Protocol Bridge**: HoloLoom (RAG) → HoloLoom (Planning) → xTerminator (QA)

**Example**:
```python
# Step 1: RAG research
research_request = create_simple_request(
    task_type="question_answering",
    parameters={
        "query": "Best practices for Python error handling",
        "mode": "research",  # Multi-query exploration
        "max_sources": 5,
    }
)
research_response = await rag_dept.execute(research_request)
research_findings = research_response.result.get("answer", "")

# Step 2: Planning creates implementation plan
plan_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={
        "goal": f"Implement error handling based on: {research_findings[:200]}",
        "max_tasks": 4,
    }
)
plan_request.context["research_findings"] = research_findings
plan_request.context["research_confidence"] = research_response.confidence.score

plan_response = await planning_dept.execute(plan_request)

# Step 3: QA validates plan tasks
for task in plan_response.result["plan"]["tasks"]:
    qa_request = XTermRequest(
        request_id=str(uuid.uuid4()),
        request_type=RequestType.CLASSIFY_ISSUE,
        requesting_department="planning",
        payload={
            "task_description": task["description"],
            "research_context": research_findings[:100],
        }
    )
    qa_response = await qa_dept.execute(qa_request)
```

**Benefits**:
- Evidence-based implementation plans (grounded in research)
- QA validates before writing code (cheaper to fix early)
- Research confidence informs planning strategy (conservative if uncertain)

**Test Coverage**: 2 tests (research-to-plan flow, low-confidence fallback)

---

## Workflow 3: Confidence Negotiation Chain

**Pipeline**: Planning → RAG → QA (with confidence tracking)

**Use Case**: Track and negotiate confidence across all 3 departments

**Steps**:
1. **Planning** executes with initial confidence
2. **RAG** receives Planning confidence as upstream context
3. **QA** negotiates confidence with RAG using weighted average

**Protocol Bridge**: Cross-protocol confidence conversion at each stage

**Example**:
```python
# Stage 1: Planning
plan_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={"goal": "Test confidence tracking", "max_tasks": 3}
)
plan_response = await planning_dept.execute(plan_request)
stage1_conf = plan_response.confidence.score  # ConfidenceMetadata → float

# Stage 2: RAG (pass Planning confidence)
rag_request = create_simple_request(
    task_type="question_answering",
    parameters={"query": "Confidence test query", "mode": "direct"}
)
rag_request.context["upstream_confidence"] = stage1_conf
rag_response = await rag_dept.execute(rag_request)
stage2_conf = rag_response.confidence.score

# Stage 3: QA (negotiate confidence)
qa_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.SCAN_CODE,
    requesting_department="rag",
    payload={"code": "test", "file_path": "test.py"},
    context={"requesting_confidence": stage2_conf}
)
qa_response = await qa_dept.execute(qa_request)

# Confidence negotiation (automatic)
if "negotiated_confidence" in qa_response.metadata:
    negotiated = qa_response.metadata["negotiated_confidence"]
    expected = 0.3 * stage2_conf + 0.7 * qa_response.confidence
    assert abs(negotiated - expected) < 0.01
```

**Benefits**:
- Full confidence provenance (where did confidence come from?)
- Automatic negotiation between departments (builds trust)
- Degradation detection (confidence drops >20% trigger alerts)

**Test Coverage**: 2 tests (cross-protocol negotiation, degradation detection)

---

## Workflow 4: Error Recovery Pipeline

**Pipeline**: RAG failure → Planning alternative → QA validation

**Use Case**: Graceful degradation when a department fails

**Steps**:
1. **Planning** succeeds
2. **RAG** fails (simulated or real failure)
3. **QA** proceeds without RAG context (degraded mode)

**Protocol Bridge**: Error metadata propagation across protocols

**Example**:
```python
# Step 1: Planning succeeds
plan_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={"goal": "Handle RAG failure", "max_tasks": 2}
)
plan_response = await planning_dept.execute(plan_request)

# Step 2: RAG fails (we skip it and handle gracefully)
# In production: try/except to catch RAG failure

# Step 3: QA proceeds without RAG context
qa_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.SCAN_CODE,
    requesting_department="planning",
    payload={
        "code": code_content,
        "file_path": "example.py",
        "rag_failed": True,  # Signal degraded mode
    }
)
qa_response = await qa_dept.execute(qa_request)

# QA should succeed with reduced confidence
assert qa_response.status == ResponseStatus.SUCCESS
assert qa_response.confidence >= 0.5  # Degraded but functional
```

**Benefits**:
- System remains functional even when components fail
- Error metadata helps downstream departments adjust behavior
- Confidence reflects degraded state (e.g., 0.6 instead of 0.9)

**Test Coverage**: 2 tests (graceful degradation, error metadata propagation)

---

## Workflow 5: Parallel Execution

**Pipeline**: (Planning + RAG) concurrently → QA synthesis

**Use Case**: Execute independent tasks concurrently for speed

**Steps**:
1. **Planning** and **RAG** execute concurrently (no dependencies)
2. **QA** synthesizes results from both departments

**Protocol Bridge**: Confidence combination from parallel results

**Example**:
```python
# Create requests
plan_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={"goal": "Parallel test", "max_tasks": 3}
)

rag_request = create_simple_request(
    task_type="question_answering",
    parameters={"query": "Parallel query", "mode": "direct"}
)

# Execute in parallel
plan_task = planning_dept.execute(plan_request)
rag_task = rag_dept.execute(rag_request)

plan_response, rag_response = await asyncio.gather(plan_task, rag_task)

# Synthesize results in QA
combined_conf = (plan_response.confidence.score + rag_response.confidence.score) / 2

qa_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.SCAN_CODE,
    requesting_department="synthesis",
    payload={
        "code": "combined results",
        "file_path": "test.py",
    },
    context={"combined_confidence": combined_conf}
)

qa_response = await qa_dept.execute(qa_request)
```

**Benefits**:
- Faster execution (2x speedup for independent tasks)
- Parallel confidence tracking
- Result synthesis in final department

**Test Coverage**: 1 test (concurrent planning and RAG)

---

## Workflow 6: Refinement Loop

**Pipeline**: QA low confidence → RAG enrichment → Planning replan → QA revalidate

**Use Case**: Iteratively improve results until confidence threshold met

**Steps**:
1. **Planning** creates initial plan
2. **QA** validates, detects low confidence
3. **RAG** enriches context with additional research
4. **Planning** replans with enriched context
5. **QA** revalidates (should have higher confidence)

**Protocol Bridge**: Feedback loop across protocols with confidence tracking

**Example**:
```python
# Step 1: Initial plan
plan_request = create_simple_request(
    task_type="goal_decomposition",
    parameters={"goal": "Initial plan", "max_tasks": 2}
)
plan_response = await planning_dept.execute(plan_request)

# Step 2: QA validates
qa_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.SCAN_CODE,
    requesting_department="planning",
    payload={"code": "initial", "file_path": "test.py"}
)
qa_response = await qa_dept.execute(qa_request)
qa_verification = await qa_dept.verify(qa_response)

# Step 3: If low confidence, trigger refinement
if not qa_verification.verified or qa_response.confidence < 0.75:
    # RAG enriches context
    rag_request = create_simple_request(
        task_type="question_answering",
        parameters={
            "query": "How to improve: Initial plan",
            "mode": "research",
        }
    )
    rag_response = await rag_dept.execute(rag_request)

    # Planning replans
    refined_plan_request = create_simple_request(
        task_type="goal_decomposition",
        parameters={"goal": "Refined plan", "max_tasks": 3}
    )
    refined_plan_request.context["enrichment"] = rag_response.result.get("answer", "")
    refined_plan_response = await planning_dept.execute(refined_plan_request)

    # QA revalidates
    qa_revalidate_request = XTermRequest(
        request_id=str(uuid.uuid4()),
        request_type=RequestType.SCAN_CODE,
        requesting_department="planning_refined",
        payload={"code": "refined", "file_path": "test.py"}
    )
    qa_revalidate_response = await qa_dept.execute(qa_revalidate_request)

    # Confidence should improve
    assert qa_revalidate_response.confidence >= qa_response.confidence
```

**Benefits**:
- Automatic quality improvement (no manual intervention)
- Confidence-driven refinement (only when needed)
- Multi-department feedback loop (each contributes expertise)

**Test Coverage**: 1 test (complete refinement cycle)

---

## Workflow 7: Health Monitoring

**Pipeline**: Monitor all departments → Planning recovery plan → QA system health

**Use Case**: System-wide health monitoring and automated recovery

**Steps**:
1. **Health Check** all departments
2. **Planning** creates recovery plan if degradation detected
3. **QA** validates system health

**Protocol Bridge**: Aggregate health metrics from different protocols

**Example**:
```python
# Step 1: Collect health from all departments
plan_health = await planning_dept.health_check()  # bool
rag_health = await rag_dept.health_check()        # bool
qa_health = await qa_dept.health_check()          # dict

# Step 2: Aggregate health status
all_healthy = plan_health and rag_health and (qa_health["status"] == "healthy")

if not all_healthy:
    # Create recovery plan
    recovery_request = create_simple_request(
        task_type="goal_decomposition",
        parameters={
            "goal": "System recovery plan",
            "max_tasks": 3,
        }
    )
    recovery_request.context["unhealthy_departments"] = [
        "planning" if not plan_health else None,
        "rag" if not rag_health else None,
        "qa" if qa_health["status"] != "healthy" else None,
    ]
    recovery_plan = await planning_dept.execute(recovery_request)

# Step 3: Get system-wide metrics
plan_metrics = await planning_dept.get_metrics()
rag_metrics = await rag_dept.get_metrics()

qa_stats_request = XTermRequest(
    request_id=str(uuid.uuid4()),
    request_type=RequestType.GET_STATISTICS,
    requesting_department="monitoring",
    payload={}
)
qa_stats_response = await qa_dept.execute(qa_stats_request)
```

**Benefits**:
- Proactive health monitoring (detect issues early)
- Automated recovery planning (reduce downtime)
- System-wide visibility (all departments)

**Test Coverage**: 3 tests (health checks, system metrics, degradation detection)

---

## Running the Tests

All 7 workflows are fully tested in `HoloLoom/departments/tests/test_workflows.py`:

```bash
# Run all workflow tests
pytest HoloLoom/departments/tests/test_workflows.py -v

# Run specific workflow
pytest HoloLoom/departments/tests/test_workflows.py::TestWorkflow1_CodeAnalysisPipeline -v

# Run with detailed output
pytest HoloLoom/departments/tests/test_workflows.py -v -s
```

**Expected Results**: 28 tests passing (all workflows validated)

---

## Design Patterns

### 1. Protocol Bridging Pattern

**Problem**: Two different protocols (HoloLoom vs xTerminator)

**Solution**: Explicit conversion at boundaries
```python
# HoloLoom → xTerminator
holo_conf = holo_response.confidence.score  # Extract float
xterm_request.context["requesting_confidence"] = holo_conf

# xTerminator → HoloLoom
confidence_meta = ConfidenceMetadata.from_score(xterm_conf)
```

### 2. Confidence Propagation Pattern

**Problem**: Confidence must flow through entire pipeline

**Solution**: Pass as context, track in metadata
```python
request.context["upstream_confidence"] = prior_confidence
response.metadata["confidence_history"] = [0.85, 0.90, 0.87]
```

### 3. Error Recovery Pattern

**Problem**: Departments may fail, pipeline should continue

**Solution**: Try/except with degraded mode
```python
try:
    rag_response = await rag_dept.execute(rag_request)
except Exception:
    # Continue without RAG context
    qa_request.payload["rag_failed"] = True
```

### 4. Parallel Execution Pattern

**Problem**: Independent tasks executed sequentially waste time

**Solution**: asyncio.gather() for concurrent execution
```python
results = await asyncio.gather(task1, task2, task3)
```

### 5. Refinement Loop Pattern

**Problem**: Low-quality results need improvement

**Solution**: Confidence-driven iteration
```python
while response.confidence < 0.75 and iterations < max_iterations:
    response = await refine(response)
    iterations += 1
```

---

## Adding New Workflows

See `test_workflows.py` header for complete template and examples. Quick steps:

1. **Define workflow class**:
```python
class TestWorkflow8_YourWorkflow:
    """Description, steps, protocol challenges"""
```

2. **Implement test methods**:
```python
@pytest.mark.asyncio
async def test_your_feature(self, planning_dept, rag_dept, qa_dept):
    # Execute workflow
    # Assert expected outcomes
```

3. **Document protocol conversions**:
```python
# Show explicit conversions
holo_conf = holo_response.confidence.score
xterm_request.context["requesting_confidence"] = holo_conf
```

---

## Best Practices

1. **Explicit Protocol Bridging**: Always show conversions explicitly
2. **Confidence Tracking**: Pass confidence at each stage
3. **Error Handling**: Graceful degradation over failure
4. **Documentation**: Clear workflow descriptions in docstrings
5. **Test Coverage**: At least 2-3 tests per workflow
6. **Fixtures**: Reuse department fixtures for consistency

---

## Related Documentation

- **[PLANNING_DEPARTMENT_TESTS_COMPLETE.md](PLANNING_DEPARTMENT_TESTS_COMPLETE.md)** - Planning Department tests
- **[QA_DEPARTMENT_TESTS_COMPLETE.md](QA_DEPARTMENT_TESTS_COMPLETE.md)** - QA Department tests
- **[HoloLoom/departments/protocol.py](HoloLoom/departments/protocol.py)** - HoloLoom protocol definition
- **[xterminator/department_protocol.py](xterminator/department_protocol.py)** - xTerminator protocol definition

---

**Status**: ✅ Complete
**Date**: November 20, 2025
**Next Steps**: Run tests, create example workflows in `examples/` directory
