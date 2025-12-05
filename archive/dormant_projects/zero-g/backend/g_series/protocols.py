"""
G-Series Protocol Definitions

The G-Series framework provides staged data onboarding with a multiphase rocket metaphor.
Each G-level represents increasing altitude and capability in the data access journey.

G-Series Stages:
- G1: Contact - Zero-move access, MCP connectors, API layer, safety
- G2: Lift-Off - Deeper connectors, schema discovery, index scaffolding
- G3: Orbital Ops - Real-time streaming, hot/cold paging, yarn stabilization
- G4: Space Walk - Heddle tensioning, graph realignment, semantic tuning
- G5+: Legacy Repair - Database rewriting, table repair, structure modernization

Core Principle: **Data that cannot move stays in place**
Zero-G accesses data via APIs, connectors, and metadata-only operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==================== Data Source Types ====================

class DataSourceType(Enum):
    """Types of data sources Zero-G can access"""
    LOCAL_FILE = "local_file"
    HTTP_API = "http_api"
    DATABASE_SQL = "database_sql"
    DATABASE_NOSQL = "database_nosql"
    STREAMING = "streaming"
    LEGACY_FTP = "legacy_ftp"
    CLOUD_STORAGE = "cloud_storage"
    IMMOVABLE_ARCHIVE = "immovable_archive"


class AccessMode(Enum):
    """How Zero-G accesses the data source"""
    ZERO_MOVE = "zero_move"      # Data stays in place, metadata only
    ZERO_COPY = "zero_copy"      # Read-only access, no copying
    READ_ONLY = "read_only"      # Can read but not modify
    READ_WRITE = "read_write"    # Full access (use with caution)
    STREAMING = "streaming"      # Continuous stream access


class DataSensitivity(Enum):
    """Data sensitivity classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    IMMOVABLE = "immovable"  # Cannot be moved due to compliance/regulations


# ==================== Data Structures ====================

@dataclass
class DataSource:
    """A single data source to be accessed"""
    source_id: str
    source_type: DataSourceType
    access_mode: AccessMode
    sensitivity: DataSensitivity
    connection_params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    region_lock: Optional[str] = None  # Geographic restriction (e.g., "US-WEST")
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SchemaElement:
    """A single element in a discovered schema"""
    element_id: str
    element_type: str  # table, collection, file, stream
    name: str
    properties: Dict[str, Any]
    relationships: List[str] = field(default_factory=list)


@dataclass
class SchemaDiscoveryResult:
    """Results of schema discovery"""
    source_id: str
    elements: List[SchemaElement]
    conflicts: List[Dict[str, Any]]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IndexStrategy:
    """Strategy for indexing a data source"""
    source_id: str
    index_type: str  # vector, keyword, hybrid, graph
    fields_to_index: List[str]
    embedding_model: Optional[str] = None
    batch_size: int = 100
    estimated_time_seconds: Optional[float] = None


@dataclass
class StreamConfig:
    """Configuration for streaming data access"""
    source_id: str
    stream_type: str  # websocket, sse, polling, kafka
    poll_interval_seconds: Optional[float] = None
    buffer_size: int = 1000
    checkpoint_strategy: str = "time_based"  # time_based, count_based, manual


@dataclass
class ConnectorHealth:
    """Health status of a connector"""
    connector_id: str
    status: str  # healthy, degraded, down
    latency_ms: Optional[float] = None
    error_rate: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== G1: Contact Layer ====================

class G1ZeroMoveProtocol(Protocol):
    """
    G1.1: Zero-Move Access - Data that must remain in place

    Principles:
    - Data never leaves its original location
    - Only metadata, headers, and summaries are retrieved
    - No mutation, no copying, no downloads
    - Suitable for compliance-restricted, massive, or legacy data
    """

    async def connect(self, source: DataSource) -> ConnectorHealth:
        """Establish connection to data source (zero-move)"""
        ...

    async def get_metadata(self, source_id: str) -> Dict[str, Any]:
        """Retrieve metadata only (no actual data)"""
        ...

    async def get_summary(
        self,
        source_id: str,
        summary_type: str = "count"  # count, schema, sample
    ) -> Dict[str, Any]:
        """Get summary statistics without reading data"""
        ...


class G1MCPProtocol(Protocol):
    """
    G1.2: MCP (Model Context Protocol) Connector Stack

    MCP provides standardized connectors for:
    - Local files
    - HTTP APIs
    - Databases
    - Cloud storage
    - Legacy systems
    """

    async def register_connector(
        self,
        connector_id: str,
        connector_type: DataSourceType,
        config: Dict[str, Any]
    ) -> None:
        """Register a new MCP connector"""
        ...

    async def list_connectors(self) -> List[str]:
        """List all registered connectors"""
        ...

    async def test_connector(
        self,
        connector_id: str
    ) -> ConnectorHealth:
        """Test connector health"""
        ...

    async def invoke_connector(
        self,
        connector_id: str,
        operation: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke a connector operation"""
        ...


class G1APIProtocol(Protocol):
    """
    G1.3: API Layer Foundations

    Provides stable APIs for:
    - REST endpoints
    - GraphQL queries
    - gRPC services
    - WebSocket streaming
    """

    async def create_endpoint(
        self,
        endpoint_id: str,
        method: str,  # GET, POST, etc.
        handler: Any,  # Callable
        schema: Dict[str, Any]
    ) -> None:
        """Create a new API endpoint"""
        ...

    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all available endpoints"""
        ...


class G1SafetyProtocol(Protocol):
    """
    G1.4: Safety & OpSec Layer

    Provides:
    - Region locks
    - Encryption at rest and in transit
    - Access control and authorization
    - Audit logging
    - Rollback capabilities
    """

    async def verify_region_lock(
        self,
        source_id: str,
        requested_region: str
    ) -> bool:
        """Verify data access complies with region lock"""
        ...

    async def encrypt_connection(
        self,
        source_id: str,
        encryption_config: Dict[str, Any]
    ) -> bool:
        """Enable encryption for data connection"""
        ...

    async def audit_access(
        self,
        astronaut_id: str,
        source_id: str,
        operation: str,
        timestamp: datetime
    ) -> None:
        """Log data access for audit trail"""
        ...

    async def create_rollback_point(
        self,
        source_id: str,
        description: str
    ) -> str:
        """Create a rollback point (returns rollback_id)"""
        ...

    async def execute_rollback(
        self,
        rollback_id: str
    ) -> bool:
        """Execute rollback to a previous point"""
        ...


# ==================== G2: Lift-Off Layer ====================

class G2SchemaDiscoveryProtocol(Protocol):
    """
    G2: Schema Discovery and Inference

    Responsibilities:
    - Infer schema from data sources
    - Detect conflicts across schemas
    - Propose unification strategies
    """

    async def discover_schema(
        self,
        source_id: str,
        sample_size: int = 1000
    ) -> SchemaDiscoveryResult:
        """Discover schema from data source"""
        ...

    async def detect_conflicts(
        self,
        schemas: List[SchemaDiscoveryResult]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts across multiple schemas"""
        ...

    async def propose_unification(
        self,
        schemas: List[SchemaDiscoveryResult]
    ) -> Dict[str, Any]:
        """Propose a unified schema across sources"""
        ...


class G2IndexingProtocol(Protocol):
    """
    G2: Indexing and Scaffolding

    Responsibilities:
    - Plan indexing strategy
    - Execute vector/keyword/graph indexing
    - Monitor indexing progress
    """

    async def plan_indexing(
        self,
        source_id: str,
        schema: SchemaDiscoveryResult
    ) -> IndexStrategy:
        """Plan indexing strategy for a data source"""
        ...

    async def execute_indexing(
        self,
        strategy: IndexStrategy,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Execute indexing according to strategy"""
        ...

    async def get_index_status(
        self,
        source_id: str
    ) -> Dict[str, Any]:
        """Get status of indexing operation"""
        ...


# ==================== G3: Orbital Operations ====================

class G3StreamingProtocol(Protocol):
    """
    G3: Real-Time Streaming Operations

    Responsibilities:
    - Stream data from live sources
    - Handle backpressure and buffering
    - Checkpoint and resume
    """

    async def create_stream(
        self,
        config: StreamConfig
    ) -> str:
        """Create a new stream (returns stream_id)"""
        ...

    async def consume_stream(
        self,
        stream_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Consume items from a stream"""
        ...

    async def pause_stream(
        self,
        stream_id: str
    ) -> None:
        """Pause stream consumption"""
        ...

    async def resume_stream(
        self,
        stream_id: str,
        from_checkpoint: Optional[str] = None
    ) -> None:
        """Resume stream from checkpoint"""
        ...


class G3PagingProtocol(Protocol):
    """
    G3: Hot/Cold Memory Paging

    Responsibilities:
    - Classify data as hot/warm/cold
    - Page data between fast/slow storage
    - Optimize access patterns
    """

    async def classify_data(
        self,
        source_id: str,
        access_pattern: Dict[str, Any]
    ) -> str:
        """Classify data as hot/warm/cold"""
        ...

    async def page_in(
        self,
        source_id: str,
        data_id: str
    ) -> Dict[str, Any]:
        """Page data into hot storage"""
        ...

    async def page_out(
        self,
        source_id: str,
        data_id: str
    ) -> None:
        """Page data to cold storage"""
        ...


# ==================== G4: Space Walk Layer ====================

class G4HeddleProtocol(Protocol):
    """
    G4: Heddle Tensioning and Graph Realignment

    Heddles are structural pathways in the Loom (graph connections).
    Tensioning adjusts semantic relationships, associations, and routing.

    This is advanced, manual intervention mode.
    """

    async def identify_heddles(
        self,
        source_id: str
    ) -> List[Dict[str, Any]]:
        """Identify all heddles (graph structures) for a source"""
        ...

    async def analyze_tension(
        self,
        heddle_id: str
    ) -> Dict[str, Any]:
        """Analyze current tension (strength) of a heddle"""
        ...

    async def adjust_tension(
        self,
        heddle_id: str,
        new_weight: float,
        justification: str
    ) -> bool:
        """Adjust heddle tension (graph edge weight)"""
        ...

    async def rethread_heddle(
        self,
        heddle_id: str,
        new_source: str,
        new_target: str,
        justification: str
    ) -> bool:
        """Rethread a heddle (change graph connection)"""
        ...


# ==================== G5+: Legacy Repair ====================

class G5LegacyRepairProtocol(Protocol):
    """
    G5+: Legacy Database Repair and Modernization

    Advanced B2B tooling for:
    - Safe database schema migration
    - Table repair and cleanup
    - Structure modernization
    - Real-time AI augmentation layers

    WARNING: These operations can modify source data.
    Use only with explicit authorization and backups.
    """

    async def analyze_legacy_schema(
        self,
        source_id: str
    ) -> Dict[str, Any]:
        """Analyze legacy schema for repair opportunities"""
        ...

    async def propose_modernization(
        self,
        source_id: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Propose modernization plan"""
        ...

    async def execute_repair(
        self,
        source_id: str,
        repair_plan: Dict[str, Any],
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Execute repair plan (dry_run=True for preview)"""
        ...

    async def rollback_repair(
        self,
        repair_id: str
    ) -> bool:
        """Rollback a repair operation"""
        ...


# ==================== G-Series Coordinator ====================

class GSeriesCoordinator(ABC):
    """
    Main G-Series coordinator - orchestrates data onboarding across all G-levels

    The coordinator manages the staged onboarding process:
    G1 (Contact) → G2 (Lift-Off) → G3 (Orbital) → [G4 (EVA)] → [G5+ (Repair)]
    """

    def __init__(
        self,
        g1_zero_move: G1ZeroMoveProtocol,
        g1_mcp: G1MCPProtocol,
        g1_api: G1APIProtocol,
        g1_safety: G1SafetyProtocol,
        g2_schema: G2SchemaDiscoveryProtocol,
        g2_indexing: G2IndexingProtocol,
        g3_streaming: G3StreamingProtocol,
        g3_paging: G3PagingProtocol,
        g4_heddle: G4HeddleProtocol,
        g5_repair: Optional[G5LegacyRepairProtocol] = None
    ):
        self.g1_zero_move = g1_zero_move
        self.g1_mcp = g1_mcp
        self.g1_api = g1_api
        self.g1_safety = g1_safety
        self.g2_schema = g2_schema
        self.g2_indexing = g2_indexing
        self.g3_streaming = g3_streaming
        self.g3_paging = g3_paging
        self.g4_heddle = g4_heddle
        self.g5_repair = g5_repair

        self.sources: Dict[str, DataSource] = {}
        self.schemas: Dict[str, SchemaDiscoveryResult] = {}
        self.streams: Dict[str, StreamConfig] = {}

    @abstractmethod
    async def onboard_source(
        self,
        source: DataSource,
        target_g_level: str = "G3"  # G1, G2, G3, G4, G5
    ) -> Dict[str, Any]:
        """
        Onboard a data source through staged G-series process

        Process:
        1. G1 Contact: Zero-move connection, safety checks
        2. G2 Lift-Off: Schema discovery, indexing
        3. G3 Orbital: Streaming setup (if applicable)
        4. G4 EVA: Manual tuning (optional)
        5. G5 Repair: Legacy modernization (optional)
        """
        ...

    @abstractmethod
    async def get_source_status(
        self,
        source_id: str
    ) -> Dict[str, Any]:
        """Get current status of a data source"""
        ...

    @abstractmethod
    async def offboard_source(
        self,
        source_id: str,
        cleanup: bool = False
    ) -> bool:
        """Offboard a data source (cleanup indexes, streams, etc.)"""
        ...
