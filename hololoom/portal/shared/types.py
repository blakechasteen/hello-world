"""
Shared Pydantic models for Portal components.

These types define the contract between Portal Server, Node Daemon, and Shuttle Bot.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeCapabilities(BaseModel):
    """Hardware capabilities reported by a node."""

    cpu_cores: int = Field(ge=1, description="Number of CPU cores")
    memory_mb: int = Field(ge=128, description="Available memory in MB")
    gpu: bool = Field(default=False, description="GPU available for compute")
    wasm_runtime: str = Field(default="wasmtime", description="WASM runtime name")

    class Config:
        json_schema_extra = {
            "example": {
                "cpu_cores": 8,
                "memory_mb": 16384,
                "gpu": False,
                "wasm_runtime": "wasmtime"
            }
        }


class NodeStatus(str, Enum):
    """Node health status."""

    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    UNKNOWN = "unknown"


class NodeRecord(BaseModel):
    """A registered node in the Loom."""

    node_id: str = Field(description="Unique node identifier")
    address: str = Field(description="Node HTTP address (host:port)")
    capabilities: NodeCapabilities
    status: NodeStatus = Field(default=NodeStatus.UNKNOWN)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    current_jobs: int = Field(default=0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "node_id": "desktop-1",
                "address": "192.168.1.100:9090",
                "capabilities": {
                    "cpu_cores": 8,
                    "memory_mb": 16384,
                    "gpu": False
                },
                "status": "online",
                "last_heartbeat": "2025-12-03T10:30:00Z",
                "current_jobs": 0
            }
        }


class JobRequest(BaseModel):
    """Request to execute a WASM job on a node."""

    job_id: str = Field(description="Unique job identifier")
    module_id: str = Field(description="WASM module identifier")
    wasm_base64: str | None = Field(
        default=None,
        description="Base64-encoded WASM bytes (if not using registered module)"
    )
    entry_function: str = Field(default="run", description="Function to call")
    input_json: Any = Field(description="Input data as JSON")
    timeout_seconds: int = Field(default=60, ge=1, le=600)

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-123",
                "module_id": "add-v1",
                "entry_function": "run",
                "input_json": {"a": 2, "b": 3},
                "timeout_seconds": 60
            }
        }


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class JobResult(BaseModel):
    """Result of a WASM job execution."""

    job_id: str
    status: JobStatus
    output_json: Any | None = None
    error: str | None = None
    execution_time_ms: float | None = None
    node_id: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-123",
                "status": "completed",
                "output_json": {"result": 5},
                "execution_time_ms": 12.5,
                "node_id": "desktop-1"
            }
        }


class LoomStatus(BaseModel):
    """Overall status of the Loom (all nodes)."""

    loom_id: str
    nodes_online: int = Field(ge=0)
    nodes_offline: int = Field(ge=0)
    total_nodes: int = Field(ge=0)
    total_cpu_cores: int = Field(ge=0)
    total_memory_mb: int = Field(ge=0)
    jobs_in_progress: int = Field(ge=0)
    uptime_seconds: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "loom_id": "loom-blake",
                "nodes_online": 2,
                "nodes_offline": 1,
                "total_nodes": 3,
                "total_cpu_cores": 24,
                "total_memory_mb": 49152,
                "jobs_in_progress": 0,
                "uptime_seconds": 3600.5
            }
        }


class NodeRegistration(BaseModel):
    """Request to register a node with the Portal."""

    node_id: str
    address: str
    capabilities: NodeCapabilities
    shared_secret: str = Field(description="Shared secret for auth")


class ModuleManifest(BaseModel):
    """WASM module manifest describing inputs/outputs."""

    id: str = Field(description="Unique module identifier")
    name: str = Field(description="Human-readable name")
    version: str = Field(default="1.0.0")
    entry_function: str = Field(default="run")
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    description: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "add-v1",
                "name": "Simple Adder",
                "version": "1.0.0",
                "entry_function": "run",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    }
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "number"}
                    }
                }
            }
        }
