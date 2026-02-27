# ADR-001: Multi-Department Architecture

**Status**: Accepted
**Date**: 2025-09-15
**Author**: HoloLoom Core Team
**Updated**: 2025-11-22 (Moonshot Task 1 - Cross-Department Workflows)

---

## Context

HoloLoom needed a scalable architecture for handling diverse enterprise workloads (RAG, planning, infrastructure management, compliance) while maintaining:
- Clear separation of concerns
- Independent scaling of components
- B2B customization (customer-specific policies)
- Fault tolerance (component failures don't crash entire system)

**Alternatives Considered**:
1. **Monolithic Orchestrator** - Single massive class handling all functionality
2. **Microservices** - Separate processes communicating via HTTP/gRPC
3. **Multi-Department (Chosen)** - Protocol-based departments within single process

---

## Decision

We will implement a **multi-department architecture** where specialized departments handle distinct domains:

### 5 Core Departments

| Department | Responsibility | Example Tasks |
|------------|----------------|---------------|
| **RAG** | Retrieval-Augmented Generation | question_answering, document_search, batch_processing |
| **Planning** | Goal decomposition & execution | goal_decomposition, execution_planning, resource_estimation |
| **Orchestration** | Cross-department coordination | sequential_workflow, parallel_execution, result_aggregation |
| **Infrastructure** | Resource management & scaling | health_check, auto_scale, monitor_resources, deploy_service |
| **Context** | Contextual intelligence & privacy | context_enrichment, privacy_enforcement, session_tracking |

### Department Protocol

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

### Request/Response Protocol

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

## Consequences

### Positive

**✓ Clear Separation of Concerns**
- Each department has a single, well-defined responsibility
- Reduces cognitive load for developers
- Easier to reason about system behavior

**✓ Independent Scaling**
- Can scale departments independently based on workload
- E.g., scale RAG department during high query volume, Infrastructure during deployments

**✓ B2B Customization**
- Customer-specific policies per department (e.g., HIPAA-compliant RAG, SOX-compliant audit)
- Marketplace tiers (Bronze/Silver/Gold/Platinum) with different department capabilities

**✓ Fault Tolerance**
- Department failures are isolated (e.g., Planning failure doesn't affect RAG)
- Graceful degradation possible

**✓ Testability**
- Each department can be unit-tested in isolation
- Integration tests validate cross-department workflows
- Mock departments for testing

**✓ Extensibility**
- New departments can be added without modifying existing ones
- Third-party departments can implement `BaseDepartment` protocol

### Negative

**✗ Protocol Overhead**
- ~0.5ms per department call for request/response serialization
- Negligible compared to department execution time (50-500ms)

**✗ Complexity for Simple Use Cases**
- Single-query workflows still go through department routing
- Mitigated by `HoloLoom` unified API for simple cases

**✗ Cross-Department Dependencies**
- Some workflows require sequential department execution (e.g., Context → RAG → Planning)
- Managed by Orchestration Department with workflow patterns

---

## Comparison to Alternatives

### vs. Monolithic Orchestrator

| Aspect | Monolithic | Multi-Department |
|--------|-----------|------------------|
| **Complexity** | Lower (single class) | Higher (5 departments) |
| **Testability** | Hard (test entire system) | Easy (test departments independently) |
| **Scaling** | All or nothing | Per-department |
| **B2B Customization** | Difficult (monolith modification) | Easy (per-department policies) |
| **Fault Tolerance** | None (single point of failure) | Good (isolated failures) |

**Verdict**: Multi-department wins for enterprise B2B use cases.

### vs. Microservices

| Aspect | Microservices | Multi-Department |
|--------|--------------|------------------|
| **Latency** | High (network overhead) | Low (in-process calls) |
| **Deployment** | Complex (5 services) | Simple (single process) |
| **Development** | Complex (service coordination) | Medium (protocol-based) |
| **Fault Tolerance** | Excellent (process isolation) | Good (graceful degradation) |
| **Scaling** | Excellent (Kubernetes) | Good (thread/process pools) |

**Verdict**: Multi-department is better for single-node deployments. Microservices for multi-node clusters.

**Hybrid Approach** (Future): Multi-department for single-node, microservices for distributed clusters.

---

## Implementation

### Registry Pattern

```python
# Department registry
_DEPARTMENT_REGISTRY: Dict[str, Department] = {}

def register_department(department_id: str, department: Department) -> None:
    """Register a department"""
    _DEPARTMENT_REGISTRY[department_id] = department

def get_department(department_id: str) -> Department:
    """Get department by ID"""
    if department_id not in _DEPARTMENT_REGISTRY:
        raise ValueError(f"Department not found: {department_id}")
    return _DEPARTMENT_REGISTRY[department_id]

# Register core departments
register_department("rag", RAGDepartment())
register_department("planning", PlanningDepartment())
register_department("orchestration", OrchestrationDepartment())
register_department("infrastructure", InfrastructureDepartment())
register_department("context", ContextDepartment())
```

### Cross-Department Workflow (from Moonshot Task 1)

```python
# Research & Analysis Pipeline (Context → RAG → Planning)
async def research_workflow():
    context_dept = get_department("context")
    orchestration_dept = get_department("orchestration")
    planning_dept = get_department("planning")

    # Step 1: Enrich context
    context_response = await context_dept.execute(context_request)
    enrichment = context_response.result

    # Step 2: Parallel execution (RAG + Planning)
    parallel_request = DepartmentRequest(
        task_type="parallel_execution",
        parameters={
            "departments": [
                {"department_id": "rag", "task_type": "retrieve_context", ...},
                {"department_id": "planning", "task_type": "goal_decomposition", ...},
            ],
        },
    )

    parallel_response = await orchestration_dept.execute(parallel_request)
    rag_result = parallel_response.result["results"][0]
    planning_result = parallel_response.result["results"][1]

    # Step 3: Create execution plan
    plan_request = DepartmentRequest(
        task_type="create_execution_plan",
        parameters={
            "sources": rag_result["sources"],
            "sub_goals": planning_result["sub_goals"]
        }
    )

    plan_response = await planning_dept.execute(plan_request)
    return plan_response.result
```

**5 Workflow Patterns Demonstrated** (Moonshot Task 1):
1. Research & Analysis - Context enrichment + parallel execution
2. Deployment - Sequential workflow with health monitoring
3. Intelligent Routing - Auto-routing with learning
4. Performance Monitoring - Resource monitoring + auto-scaling
5. Customer Onboarding - All 5 departments working together

---

## Metrics

**Performance** (Moonshot Task 1 Implementation):

| Workflow | Departments | Latency | Status |
|----------|-------------|---------|--------|
| Research & Analysis | Context, RAG, Planning | ~800ms | ✓ |
| Deployment | Context, Planning, Infrastructure | ~1,200ms | ✓ |
| Intelligent Routing | Context, Orchestration, RAG/Planning | ~400ms | ✓ |
| Monitoring & Scaling | Infrastructure, Planning, Context | ~600ms | ✓ |
| Customer Onboarding | All 5 departments | ~1,500ms | ✓ |

**Test Coverage**:
- 9/9 structural tests passing (Task 1)
- 25/25 Context Department tests (Production Hardening)
- 100+ department protocol tests

---

## Related ADRs

- [ADR-002: Thompson Sampling for Routing](ADR-002-thompson-sampling.md) - Query routing across departments
- [ADR-004: Alignment Framework Integration](ADR-004-alignment-framework.md) - Safety across all departments

---

## References

- **Implementation**: `hololoom/departments/`
- **Protocol**: `hololoom/departments/protocol.py` (BaseDepartment)
- **Registry**: `hololoom/departments/registry.py`
- **Workflows**: `hololoom/departments/examples/workflow_examples.py` (730 lines, Moonshot Task 1)
- **Documentation**: `hololoom/departments/MOONSHOT_TASK_1_COMPLETE.md`

---

**Last Updated**: 2025-11-22 | **Status**: Production Ready | **Version**: 1.1.0
