#!/usr/bin/env python3
"""
MCP Department Registry - Federated Agent Architecture

Each department exposes its capabilities as an MCP server with:
- Tool definitions (standardized signatures)
- Permission requirements (what can this department do?)
- Session management (stateless with session IDs)
- Error handling (escalation protocols)

This implements Conway's Law for agents: architecture mirrors organization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class PermissionLevel(Enum):
    """Permission levels for department operations"""
    READ = "read"           # Can query/read data
    WRITE = "write"         # Can modify data
    EXECUTE = "execute"     # Can run code/tasks
    DEPLOY = "deploy"       # Can deploy to production
    ADMIN = "admin"         # Can modify system config


class PermissionMode(Enum):
    """How permissions are enforced"""
    HARD = "hard"   # Tool not visible if no permission
    SOFT = "soft"   # Tool visible, but call fails with log


@dataclass
class MCPTool:
    """Definition of a tool exposed by a department"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    required_permissions: list[PermissionLevel]
    permission_mode: PermissionMode
    returns: dict[str, Any]  # JSON Schema


@dataclass
class DepartmentCharter:
    """Charter defining a department's role and capabilities"""
    name: str
    role: str  # High-level purpose
    capabilities: list[str]  # What it can do
    tools: list[MCPTool]  # MCP-exposed tools
    permissions: list[PermissionLevel]  # What this department is allowed to do
    dependencies: list[str]  # Other departments this depends on
    session_state: Literal["stateless", "session_id", "stateful"]
    context_budget: int  # Max tokens for this department's context


# =============================================================================
# DEPARTMENT DEFINITIONS
# =============================================================================

MASTER_WEAVER = DepartmentCharter(
    name="MasterWeaver",
    role="Entity extraction, domain understanding, knowledge structuring",
    capabilities=[
        "Extract entities from multimodal input",
        "Validate entity consistency",
        "Query domain ontologies",
        "Build knowledge graphs from raw data"
    ],
    tools=[
        MCPTool(
            name="extract_entities",
            description="Extract structured entities from raw input (text, audio transcript, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "input_data": {"type": "string", "description": "Raw input data"},
                    "domain": {"type": "string", "description": "Domain context (beekeeping, automotive, etc.)"},
                    "session_id": {"type": "string", "description": "Session identifier for context continuity"},
                    "confidence_threshold": {"type": "number", "default": 0.75}
                },
                "required": ["input_data", "domain", "session_id"]
            },
            required_permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "items": {"type": "object"}},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string", "description": "Compact justification"},
                    "session_id": {"type": "string"}
                }
            }
        ),
        MCPTool(
            name="validate_entity_consistency",
            description="Check if extracted entities are internally consistent",
            parameters={
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "items": {"type": "object"}},
                    "session_id": {"type": "string"}
                },
                "required": ["entities", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "is_consistent": {"type": "boolean"},
                    "inconsistencies": {"type": "array"},
                    "confidence": {"type": "number"}
                }
            }
        ),
        MCPTool(
            name="query_domain_ontology",
            description="Query domain-specific knowledge graph",
            parameters={
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "concept": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["domain", "concept", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "definition": {"type": "string"},
                    "relationships": {"type": "array"},
                    "examples": {"type": "array"}
                }
            }
        )
    ],
    permissions=[PermissionLevel.READ, PermissionLevel.WRITE],  # Can read Neo4j, write entities
    dependencies=["Infrastructure"],  # Needs Infrastructure for Neo4j access
    session_state="session_id",  # Stateless but accepts session context
    context_budget=50000  # 50k tokens for processing large datasets
)


VERIFICATION = DepartmentCharter(
    name="Verification",
    role="Quality assurance, confidence validation, alignment checking",
    capabilities=[
        "Validate confidence claims",
        "Cross-check department outputs",
        "Generate alignment reports",
        "Request re-runs with stricter parameters"
    ],
    tools=[
        MCPTool(
            name="validate_confidence_claim",
            description="Validate that a department's confidence claim matches actual quality",
            parameters={
                "type": "object",
                "properties": {
                    "claim": {"type": "object", "description": "Department's output with confidence"},
                    "evidence": {"type": "object", "description": "Supporting data for validation"},
                    "session_id": {"type": "string"}
                },
                "required": ["claim", "evidence", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "validation_result": {"type": "boolean"},
                    "actual_confidence": {"type": "number"},
                    "claimed_confidence": {"type": "number"},
                    "recommendation": {"type": "string"}
                }
            }
        ),
        MCPTool(
            name="request_rerun",
            description="Request another department re-run a task with stricter parameters",
            parameters={
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Target department name"},
                    "task_id": {"type": "string"},
                    "new_params": {"type": "object", "description": "Updated parameters"},
                    "reason": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["department", "task_id", "new_params", "reason", "session_id"]
            },
            required_permissions=[PermissionLevel.EXECUTE],
            permission_mode=PermissionMode.SOFT,  # Logged, can be denied
            returns={
                "type": "object",
                "properties": {
                    "accepted": {"type": "boolean"},
                    "new_task_id": {"type": "string", "nullable": True},
                    "denial_reason": {"type": "string", "nullable": True}
                }
            }
        ),
        MCPTool(
            name="cross_check_departments",
            description="Compare outputs from multiple departments for consistency",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "departments": {"type": "array", "items": {"type": "string"}},
                    "session_id": {"type": "string"}
                },
                "required": ["task_id", "departments", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "consistent": {"type": "boolean"},
                    "conflicts": {"type": "array"},
                    "resolution_recommendation": {"type": "string"}
                }
            }
        )
    ],
    permissions=[PermissionLevel.READ, PermissionLevel.EXECUTE],  # Read all, execute validation tools
    dependencies=["MasterWeaver", "Infrastructure", "Execution"],  # Can validate any department
    session_state="session_id",
    context_budget=30000  # 30k tokens for validation logic
)


INFRASTRUCTURE = DepartmentCharter(
    name="Infrastructure",
    role="Data systems, performance optimization, zero-copy queries",
    capabilities=[
        "Manage Neo4j and Qdrant",
        "Optimize query performance",
        "Provision zero-copy access",
        "Diagnose system bottlenecks"
    ],
    tools=[
        MCPTool(
            name="query_neo4j",
            description="Execute Cypher query on Neo4j knowledge graph",
            parameters={
                "type": "object",
                "properties": {
                    "cypher_query": {"type": "string"},
                    "parameters": {"type": "object", "default": {}},
                    "session_id": {"type": "string"}
                },
                "required": ["cypher_query", "session_id"]
            },
            required_permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "execution_time_ms": {"type": "number"}
                }
            }
        ),
        MCPTool(
            name="query_qdrant",
            description="Execute semantic search on Qdrant vector database",
            parameters={
                "type": "object",
                "properties": {
                    "query_vector": {"type": "array", "items": {"type": "number"}},
                    "collection": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "session_id": {"type": "string"}
                },
                "required": ["query_vector", "collection", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                    "scores": {"type": "array"}
                }
            }
        ),
        MCPTool(
            name="diagnose_performance",
            description="Analyze query performance and suggest optimizations",
            parameters={
                "type": "object",
                "properties": {
                    "query_type": {"type": "string", "enum": ["neo4j", "qdrant"]},
                    "query": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["query_type", "query", "session_id"]
            },
            required_permissions=[PermissionLevel.READ, PermissionLevel.ADMIN],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "bottlenecks": {"type": "array"},
                    "recommendations": {"type": "array"},
                    "estimated_improvement": {"type": "string"}
                }
            }
        )
    ],
    permissions=[PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.ADMIN],
    dependencies=[],  # No dependencies (foundational service)
    session_state="session_id",
    context_budget=20000  # 20k tokens for system diagnostics
)


EXECUTION = DepartmentCharter(
    name="Execution",
    role="Task execution, code running, deployment orchestration",
    capabilities=[
        "Execute Claude Code tasks",
        "Monitor task status",
        "Retrieve execution logs",
        "Deploy to staging/production"
    ],
    tools=[
        MCPTool(
            name="run_claude_code_task",
            description="Execute a task via Claude Code",
            parameters={
                "type": "object",
                "properties": {
                    "task_spec": {"type": "string", "description": "Natural language task description"},
                    "context_files": {"type": "array", "items": {"type": "string"}},
                    "session_id": {"type": "string"}
                },
                "required": ["task_spec", "session_id"]
            },
            required_permissions=[PermissionLevel.EXECUTE],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "initial_response": {"type": "string"}
                }
            }
        ),
        MCPTool(
            name="check_task_status",
            description="Check status of running/completed task",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["task_id", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["running", "completed", "failed"]},
                    "progress": {"type": "number"},
                    "result": {"type": "string", "nullable": True}
                }
            }
        )
    ],
    permissions=[PermissionLevel.READ, PermissionLevel.EXECUTE],
    dependencies=["Infrastructure"],  # Needs Infrastructure for memory access
    session_state="session_id",
    context_budget=40000  # 40k tokens for complex task execution
)


CONTEXT_HOLOLOOM = DepartmentCharter(
    name="Context",
    role="Multi-pass context enrichment, missing context detection",
    capabilities=[
        "Enrich context with multi-modal data",
        "Retrieve relevant context for decisions",
        "Detect missing context before execution",
        "Perform multi-pass graph traversal"
    ],
    tools=[
        MCPTool(
            name="enrich_context",
            description="Perform multi-pass context enrichment on raw data",
            parameters={
                "type": "object",
                "properties": {
                    "raw_data": {"type": "string"},
                    "context_request": {"type": "object", "description": "What context is needed"},
                    "passes": {"type": "integer", "default": 3, "description": "Number of enrichment passes"},
                    "session_id": {"type": "string"}
                },
                "required": ["raw_data", "context_request", "session_id"]
            },
            required_permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "enriched_context": {"type": "object"},
                    "confidence": {"type": "number"},
                    "sources": {"type": "array"}
                }
            }
        ),
        MCPTool(
            name="detect_missing_context",
            description="Analyze a decision and detect what context is missing",
            parameters={
                "type": "object",
                "properties": {
                    "decision": {"type": "object"},
                    "available_context": {"type": "object"},
                    "session_id": {"type": "string"}
                },
                "required": ["decision", "available_context", "session_id"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "missing_context": {"type": "array"},
                    "impact_severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "recommendations": {"type": "array"}
                }
            }
        )
    ],
    permissions=[PermissionLevel.READ, PermissionLevel.WRITE],
    dependencies=["Infrastructure", "MasterWeaver"],
    session_state="session_id",
    context_budget=60000  # 60k tokens for multi-pass enrichment
)


ORCHESTRATION = DepartmentCharter(
    name="Orchestration",
    role="Task routing, department coordination, session management",
    capabilities=[
        "Route tasks to appropriate departments",
        "Manage global session state",
        "Escalate decisions requiring consensus",
        "Update roadmap status"
    ],
    tools=[
        MCPTool(
            name="route_task",
            description="Route a task to the appropriate department(s)",
            parameters={
                "type": "object",
                "properties": {
                    "task_spec": {"type": "object", "description": "Task specification"},
                    "candidate_departments": {"type": "array", "items": {"type": "string"}},
                    "session_id": {"type": "string"}
                },
                "required": ["task_spec", "session_id"]
            },
            required_permissions=[PermissionLevel.EXECUTE],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "assigned_department": {"type": "string"},
                    "routing_reason": {"type": "string"},
                    "estimated_duration": {"type": "string"}
                }
            }
        ),
        MCPTool(
            name="get_session_context",
            description="Retrieve compact session context for a department",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "department": {"type": "string", "description": "Requesting department"}
                },
                "required": ["session_id", "department"]
            },
            required_permissions=[PermissionLevel.READ],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "context": {"type": "object"},
                    "roadmap_phase": {"type": "string"},
                    "relevant_decisions": {"type": "array"}
                }
            }
        ),
        MCPTool(
            name="escalate_decision",
            description="Escalate a decision requiring human input or multi-department consensus",
            parameters={
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "reason": {"type": "string"},
                    "affected_departments": {"type": "array", "items": {"type": "string"}},
                    "session_id": {"type": "string"}
                },
                "required": ["issue", "reason", "session_id"]
            },
            required_permissions=[PermissionLevel.ADMIN],
            permission_mode=PermissionMode.HARD,
            returns={
                "type": "object",
                "properties": {
                    "escalation_id": {"type": "string"},
                    "status": {"type": "string"},
                    "resolution_channel": {"type": "string"}
                }
            }
        )
    ],
    permissions=[PermissionLevel.READ, PermissionLevel.EXECUTE, PermissionLevel.ADMIN],
    dependencies=[],  # No dependencies (top-level coordinator)
    session_state="stateful",  # Maintains global session state
    context_budget=100000  # 100k tokens for global coordination
)


# =============================================================================
# DEPARTMENT REGISTRY
# =============================================================================

DEPARTMENT_REGISTRY: dict[str, DepartmentCharter] = {
    "MasterWeaver": MASTER_WEAVER,
    "Verification": VERIFICATION,
    "Infrastructure": INFRASTRUCTURE,
    "Execution": EXECUTION,
    "Context": CONTEXT_HOLOLOOM,
    "Orchestration": ORCHESTRATION
}


def get_department(name: str) -> DepartmentCharter | None:
    """Get department charter by name"""
    return DEPARTMENT_REGISTRY.get(name)


def list_departments() -> list[str]:
    """List all registered departments"""
    return list(DEPARTMENT_REGISTRY.keys())


def get_department_tools(department_name: str) -> list[MCPTool]:
    """Get all tools exposed by a department"""
    dept = get_department(department_name)
    return dept.tools if dept else []


def check_permission(department_name: str, required_permission: PermissionLevel) -> bool:
    """Check if a department has a specific permission"""
    dept = get_department(department_name)
    if not dept:
        return False
    return required_permission in dept.permissions


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=== MCP Department Registry ===\n")

    for dept_name in list_departments():
        dept = get_department(dept_name)
        print(f"Department: {dept.name}")
        print(f"  Role: {dept.role}")
        print(f"  Tools: {len(dept.tools)}")
        print(f"  Permissions: {[p.value for p in dept.permissions]}")
        print(f"  Context Budget: {dept.context_budget} tokens")
        print(f"  Session State: {dept.session_state}")
        print()
