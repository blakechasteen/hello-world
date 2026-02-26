# Department Guide

Complete guide to HoloLoom's multi-department architecture - 5 specialized departments handling distinct domains.

---

## Overview

HoloLoom implements a **multi-department architecture** where specialized departments handle different aspects of the system:

| Department | Responsibility | Tasks |
|------------|----------------|-------|
| **RAG** | Retrieval-Augmented Generation | question_answering, document_search, batch_processing |
| **Planning** | Goal decomposition & execution | goal_decomposition, execution_planning, resource_estimation |
| **Orchestration** | Cross-department coordination | sequential_workflow, parallel_execution, result_aggregation |
| **Infrastructure** | Resource management & scaling | health_check, auto_scale, monitor_resources, deploy_service |
| **Context** | Contextual intelligence & privacy | context_enrichment, privacy_enforcement, session_tracking |

---

## Why Multi-Department?

**Problem**: Monolithic systems become unmaintainable as they grow.

**Solution**: Separate concerns into specialized departments with clear responsibilities.

**Benefits**:
- ✅ Clear separation of concerns
- ✅ Independent scaling
- ✅ B2B customization (customer-specific policies)
- ✅ Fault tolerance (isolated failures)
- ✅ Testability (unit test departments independently)

See [ADR-001: Multi-Department Architecture](../../architecture/decisions/ADR-001-multi-department.md) for design rationale.

---

## Department Details

### RAG Department

**Responsibility**: Retrieval-Augmented Generation

**Supported Tasks**:
- `question_answering` - Answer queries using retrieved knowledge
- `document_search` - Search documents by semantic similarity
- `batch_processing` - Process multiple queries efficiently
- `retrieve_context` - Retrieve relevant context for query

**Example**:
```python
from HoloLoom.departments import get_department

rag_dept = get_department("rag")

request = {
    "task_type": "question_answering",
    "parameters": {
        "query": "What is Thompson Sampling?",
        "max_sources": 5
    }
}

response = await rag_dept.process(request)
print(f"Answer: {response['result']['answer']}")
print(f"Confidence: {response['confidence']:.2f}")
print(f"Sources: {len(response['result']['sources'])}")
```

**Performance**:
- Latency: ~150ms (cold), <1ms (cached)
- Quality: 85-95% confidence
- Scalability: 100+ QPS

**Documentation**: See [RAG Department Guide](rag.md) (coming soon)

---

### Planning Department

**Responsibility**: Goal decomposition & execution planning

**Supported Tasks**:
- `goal_decomposition` - Break goals into sub-goals
- `execution_planning` - Create execution plans
- `resource_estimation` - Estimate required resources
- `risk_assessment` - Assess execution risks

**Example**:
```python
from HoloLoom.departments import get_department

planning_dept = get_department("planning")

request = {
    "task_type": "goal_decomposition",
    "parameters": {
        "goal": "Implement Thompson Sampling in production",
        "depth": 3
    }
}

response = await planning_dept.process(request)
print(f"Sub-goals: {response['result']['sub_goals']}")
print(f"Execution order: {response['result']['execution_order']}")
```

**Performance**:
- Latency: ~200ms
- Quality: 80-90% confidence
- Scalability: 50+ QPS

**Documentation**: See [Planning Department Guide](planning.md) (coming soon)

---

### Orchestration Department

**Responsibility**: Cross-department coordination

**Supported Tasks**:
- `sequential_workflow` - Execute tasks in sequence
- `parallel_execution` - Execute tasks concurrently
- `result_aggregation` - Aggregate multiple results
- `route_task` - Route task to appropriate department

**Example**:
```python
from HoloLoom.departments import get_department

orchestration_dept = get_department("orchestration")

# Parallel execution (RAG + Planning)
request = {
    "task_type": "parallel_execution",
    "parameters": {
        "departments": [
            {"department_id": "rag", "task_type": "retrieve_context", ...},
            {"department_id": "planning", "task_type": "goal_decomposition", ...},
        ],
    },
}

response = await orchestration_dept.process(request)
print(f"Results: {len(response['result']['results'])}")
```

**Performance**:
- Latency: ~50ms (overhead), depends on sub-tasks
- Scalability: 100+ QPS

**Documentation**: See [Orchestration Department Guide](orchestration.md) (coming soon)

---

### Infrastructure Department

**Responsibility**: Resource management & scaling

**Supported Tasks**:
- `health_check` - Check system health
- `auto_scale` - Auto-scale based on metrics
- `monitor_resources` - Monitor CPU, memory, disk, network
- `deploy_service` - Deploy services
- `validate_deployment` - Validate deployments

**Example**:
```python
from HoloLoom.departments import get_department

infra_dept = get_department("infrastructure")

# Monitor resources
request = {
    "task_type": "monitor_resources",
    "parameters": {
        "services": ["hololoom", "neo4j", "qdrant"]
    }
}

response = await infra_dept.process(request)
print(f"CPU: {response['result']['cpu_percent']:.1f}%")
print(f"Memory: {response['result']['memory_percent']:.1f}%")

# Auto-scale if needed
if response['result']['cpu_percent'] > 80:
    scale_request = {
        "task_type": "auto_scale",
        "parameters": {
            "service": "hololoom",
            "metrics": response['result']
        }
    }
    scale_response = await infra_dept.process(scale_request)
    print(f"Scaling: {scale_response['result']['recommendation']}")
```

**Performance**:
- Latency: ~100ms
- Accuracy: 90%+ scaling decisions
- Scalability: 20+ QPS (resource-intensive)

**Documentation**: See [Infrastructure Department Guide](infrastructure.md) (coming soon)

---

### Context Department

**Responsibility**: Contextual intelligence & privacy

**Supported Tasks**:
- `context_enrichment` - Enrich requests with context
- `privacy_enforcement` - Enforce privacy policies
- `session_tracking` - Track user sessions
- `access_control` - RBAC validation

**Example**:
```python
from HoloLoom.departments import get_department
from HoloLoom.apps.departments.protocol import PrivacyEnvelope, PrivacyLevel

context_dept = get_department("context")

# HIPAA-compliant PHI access
phi_data = PrivacyEnvelope(
    data={"patient_id": "P12345", "diagnosis": "diabetes"},
    privacy_level=PrivacyLevel.CRITICAL,
    allowed_roles=["physician"]
)

request = {
    "task_type": "context_enrichment",
    "parameters": {
        "data": phi_data,
        "user_context": {"role": "physician"}
    }
}

response = await context_dept.process(request)
print(f"Enriched: {response['result']}")
```

**Performance**:
- Latency: ~80ms
- Compliance: HIPAA, SOX, GDPR
- Scalability: 100+ QPS

**Documentation**: See [Context Department Guide](context.md) (coming soon)

---

## Cross-Department Workflows

Departments work together to handle complex tasks. See [Workflow Examples](../../examples/workflows/cross-department.md) for 5 production-ready patterns:

1. **Research & Analysis** - Context → RAG → Planning
2. **Deployment** - Context → Planning → Infrastructure
3. **Intelligent Routing** - Context → Orchestration → RAG/Planning
4. **Monitoring & Scaling** - Infrastructure → Planning → Context
5. **Customer Onboarding** - All 5 departments

---

## Department Protocol

All departments implement `BaseDepartment` protocol with 7 mandatory methods:

```python
class BaseDepartment:
    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute department-specific task"""
        ...

    async def health_check(self) -> bool:
        """Check department health"""
        ...

    async def batch_execute(self, requests: List[DepartmentRequest]) -> List[DepartmentResponse]:
        """Execute multiple requests efficiently"""
        ...

    def get_supported_tasks(self) -> List[str]:
        """Return list of supported task types"""
        ...

    def get_capabilities(self) -> Dict[str, Any]:
        """Return department capabilities"""
        ...

    async def validate_request(self, request: DepartmentRequest) -> bool:
        """Validate request before execution"""
        ...

    def get_confidence_range(self) -> Tuple[float, float]:
        """Return min/max confidence for this department"""
        ...
```

---

## Request/Response Protocol

```python
@dataclass
class DepartmentRequest:
    task_type: str                      # e.g., "question_answering"
    parameters: Dict[str, Any]          # Task-specific parameters
    priority: int = 0                   # Priority (0=normal, 1=high, 2=critical)
    timeout_ms: Optional[int] = None    # Optional timeout
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DepartmentResponse:
    result: Dict[str, Any]              # Task result
    confidence: float                   # 0.0-1.0 confidence score
    metadata: ConfidenceMetadata        # Provenance, timing, sources
    status: str = "success"             # success | partial_failure | failure
    error: Optional[str] = None
```

---

## Creating Custom Departments

To create a new department:

1. **Inherit from BaseDepartment**:
```python
from HoloLoom.apps.departments.protocol import BaseDepartment, DepartmentRequest, DepartmentResponse

class MyDepartment(BaseDepartment):
    def __init__(self, department_id: str):
        super().__init__(
            name=department_id,
            domain="my_domain",
            version="1.0.0",
            supported_tasks=["task1", "task2"],
            confidence_range=(0.7, 0.95),
            config=DepartmentConfig(name=department_id, domain="my_domain")
        )

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        # Implement task execution
        result = await self._process_task(request)
        return DepartmentResponse(
            result=result,
            confidence=0.9,
            metadata=ConfidenceMetadata(...)
        )
```

2. **Register Department**:
```python
from HoloLoom.departments import register_department

register_department("my_dept", MyDepartment("my_dept"))
```

3. **Use Department**:
```python
dept = get_department("my_dept")
response = await dept.process(request)
```

---

## Testing Departments

```python
import pytest
from HoloLoom.departments import get_department

@pytest.mark.asyncio
async def test_department_health():
    """Test department health check"""
    dept = get_department("rag")
    assert await dept.health_check()

@pytest.mark.asyncio
async def test_department_task():
    """Test department task execution"""
    dept = get_department("rag")
    request = {
        "task_type": "question_answering",
        "parameters": {"query": "Test query"}
    }
    response = await dept.process(request)
    assert response["status"] == "success"
    assert 0.0 <= response["confidence"] <= 1.0
```

---

## Performance Guidelines

**Latency Targets** (per department):
- RAG: <200ms
- Planning: <300ms
- Orchestration: <100ms (overhead)
- Infrastructure: <150ms
- Context: <100ms

**Scalability Targets**:
- 100+ QPS per department (except Infrastructure: 20+ QPS)
- 1000+ concurrent requests (with proper async handling)

**Memory Usage**:
- <500MB per department
- <2GB total for all 5 departments

---

## Troubleshooting

### Department Health Check Fails

**Symptom**: `await dept.health_check()` returns False

**Causes**:
- Department not initialized
- Backend services unavailable (Neo4j, Qdrant)
- Configuration error

**Solution**:
```bash
# Check Docker services
docker ps
# Should show: neo4j, qdrant

# Restart if needed
docker-compose restart
```

### High Latency

**Symptom**: Department responses taking >500ms

**Causes**:
- Cold cache (first query)
- Large memory (1M+ memories)
- No indexing on vector store

**Solution**:
```python
# Enable query cache
config = Config.fast()
config.enable_query_cache = True

# Use FAST mode (not FUSED)
config = Config.fast()  # ~150ms
# config = Config.fused()  # ~300ms
```

### Department Not Found

**Symptom**: `ValueError: Department not found: my_dept`

**Cause**: Department not registered

**Solution**:
```python
from HoloLoom.departments import register_department, get_department

# Register before using
register_department("my_dept", MyDepartment("my_dept"))

# Then get
dept = get_department("my_dept")
```

---

## Next Steps

- [RAG Department Guide](rag.md) - Detailed RAG documentation (coming soon)
- [Planning Department Guide](planning.md) - Detailed Planning documentation (coming soon)
- [Workflow Examples](../../examples/workflows/cross-department.md) - Real-world patterns
- [API Reference](../../api/departments.md) - Complete API documentation (coming soon)

---

**Last Updated**: 2025-11-22 | **Documentation Version**: 1.1.0
