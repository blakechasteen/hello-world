"""
Launch System Protocol Definitions

This module defines the interfaces for Zero-G's Spaceflight-metaphor lifecycle orchestration.
The Launch System manages the formalized, NASA-style sequence from Preflight through EVA.

Lifecycle Stages:
- GROUNDED: System not yet initialized
- PREFLIGHT: Suit fitting, training, physical exam, briefing
- COUNTDOWN: T-10 to T-0 launch sequence
- LIFTOFF: Non-invasive metadata scanning (no writes)
- BOOST: Schema discovery, indexing, graph alignment
- ORBIT: Stable operation, streaming, app interop
- EVA: Manual intervention, heddle tensioning
- LANDING: Graceful shutdown
- ABORT: Emergency rollback at any stage
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==================== Launch Stages ====================

class LaunchStage(Enum):
    """Lifecycle stages in the spaceflight metaphor"""
    GROUNDED = "grounded"
    PREFLIGHT = "preflight"
    COUNTDOWN = "countdown"
    LIFTOFF = "liftoff"
    BOOST = "boost"
    ORBIT = "orbit"
    EVA = "eva"
    LANDING = "landing"
    ABORT = "abort"


class PreflightPhase(Enum):
    """Substages of preflight operations"""
    SUIT_FITTING = "suit_fitting"          # Permissions, privacy, access checks
    ASTRONAUT_TRAINING = "training"        # UI walkthrough, concept introduction
    PHYSICAL_EXAM = "physical_exam"        # Connection stability, environment health
    MISSION_BRIEFING = "mission_briefing"  # What will/won't be accessed


class CountdownEvent(Enum):
    """T-minus countdown events (T-10 to T-0)"""
    T_MINUS_10 = "t_minus_10"  # Final authorization request
    T_MINUS_9 = "t_minus_9"    # Atmospheric telemetry stable
    T_MINUS_8 = "t_minus_8"    # Zero-copy channels warming
    T_MINUS_7 = "t_minus_7"    # Thread anchors confirmed
    T_MINUS_6 = "t_minus_6"    # WarpSpace initialized
    T_MINUS_5 = "t_minus_5"    # YarnGraph scaffolding detected
    T_MINUS_4 = "t_minus_4"    # ResonanceShed fusion online
    T_MINUS_3 = "t_minus_3"    # Rift safety locks engaged
    T_MINUS_2 = "t_minus_2"    # ConvergenceEngine synchronized
    T_MINUS_1 = "t_minus_1"    # Mission Control releases hold
    T_0 = "t_0"                # Ignition - Lift-Off


class SafetyStatus(Enum):
    """Safety interlock status"""
    GREEN = "green"      # All systems nominal
    YELLOW = "yellow"    # Caution - proceed with monitoring
    RED = "red"          # Hold - do not proceed
    ABORT = "abort"      # Emergency - rollback immediately


# ==================== Data Structures ====================

@dataclass
class LaunchCheckpoint:
    """A checkpoint in the launch sequence for rollback"""
    stage: LaunchStage
    timestamp: datetime
    state_snapshot: Dict[str, Any]
    rollback_available: bool = True


@dataclass
class SafetyCheck:
    """A single safety verification check"""
    check_name: str
    status: SafetyStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    """Results of preflight verification"""
    suit_fitting: SafetyCheck
    training_complete: SafetyCheck
    physical_exam: SafetyCheck
    briefing_acknowledged: SafetyCheck
    overall_status: SafetyStatus
    can_proceed: bool


@dataclass
class CountdownStatus:
    """Current countdown state"""
    current_event: CountdownEvent
    events_completed: List[CountdownEvent]
    events_remaining: List[CountdownEvent]
    holds: List[SafetyCheck]  # Any holds preventing progression
    estimated_liftoff: Optional[datetime] = None


@dataclass
class OrbitStatus:
    """Status while in stable orbit"""
    orbital_altitude: str  # Metaphorical - represents system capability level
    apps_docked: List[str]  # Apps currently connected
    streaming_active: bool
    yarn_stable: bool
    warp_aligned: bool
    uptime_seconds: float


@dataclass
class EVASession:
    """An EVA (space walk) session for manual intervention"""
    eva_id: str
    astronaut_id: str  # User performing EVA
    start_time: datetime
    heddles_tensioned: List[str]  # Graph structures adjusted
    adjustments_made: Dict[str, Any]
    end_time: Optional[datetime] = None
    status: str = "active"  # active | complete | aborted


@dataclass
class TelemetrySnapshot:
    """Current system telemetry"""
    stage: LaunchStage
    timestamp: datetime
    loom_health: Dict[str, Any]
    g_series_status: Dict[str, Any]
    safety_status: SafetyStatus
    performance_metrics: Dict[str, float]
    active_missions: int
    errors: List[str] = field(default_factory=list)


# ==================== Protocol Definitions ====================

class PreflightProtocol(ABC):
    """
    Preflight operations - verification before launch

    Responsibilities:
    - Validate permissions and access controls
    - Check data source connectivity
    - Verify safety constraints
    - Brief astronaut on mission parameters
    """

    @abstractmethod
    async def run_suit_fitting(
        self,
        astronaut_id: str,
        permissions: Dict[str, Any]
    ) -> SafetyCheck:
        """Verify permissions and access controls"""
        ...

    @abstractmethod
    async def run_training(
        self,
        astronaut_id: str,
        concepts: List[str]
    ) -> SafetyCheck:
        """Complete astronaut training (UI walkthrough)"""
        ...

    @abstractmethod
    async def run_physical_exam(
        self,
        environment: Dict[str, Any]
    ) -> SafetyCheck:
        """Check system health and connectivity"""
        ...

    @abstractmethod
    async def run_mission_briefing(
        self,
        mission_params: Dict[str, Any]
    ) -> SafetyCheck:
        """Brief on what will/won't be accessed"""
        ...

    @abstractmethod
    async def generate_preflight_report(self) -> PreflightReport:
        """Aggregate all preflight checks into a report"""
        ...


class CountdownProtocol(ABC):
    """
    Launch countdown orchestration (T-10 to T-0)

    Responsibilities:
    - Execute countdown sequence in order
    - Verify each step completes successfully
    - Handle holds and scrubs
    - Coordinate subsystem initialization
    """

    @abstractmethod
    async def start_countdown(self) -> CountdownStatus:
        """Begin the countdown sequence"""
        ...

    @abstractmethod
    async def execute_countdown_event(
        self,
        event: CountdownEvent
    ) -> SafetyCheck:
        """Execute a single countdown event"""
        ...

    @abstractmethod
    async def hold_countdown(
        self,
        reason: str,
        severity: SafetyStatus
    ) -> None:
        """Pause the countdown for safety or verification"""
        ...

    @abstractmethod
    async def resume_countdown(self) -> CountdownStatus:
        """Resume countdown after a hold"""
        ...

    @abstractmethod
    async def scrub_launch(self, reason: str) -> None:
        """Cancel launch and return to preflight"""
        ...


class LiftoffProtocol(ABC):
    """
    Lift-off operations - non-invasive data access

    Responsibilities:
    - Metadata-only scanning
    - No writes to source systems
    - Initial graph preview
    - Warp alignment without indexing
    """

    @abstractmethod
    async def scan_metadata(
        self,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Scan metadata from data sources (no reads of actual data)"""
        ...

    @abstractmethod
    async def preview_graph(self) -> Dict[str, Any]:
        """Generate initial graph preview from metadata"""
        ...

    @abstractmethod
    async def align_warp(self) -> SafetyCheck:
        """Align WarpSpace without full indexing"""
        ...


class BoostProtocol(ABC):
    """
    Booster operations - assertive but safe data access

    Responsibilities:
    - Schema inference and discovery
    - Conflict detection
    - Indexing strategies
    - Cross-dataset tension analysis
    """

    @abstractmethod
    async def infer_schema(
        self,
        sources: List[str]
    ) -> Dict[str, Any]:
        """Infer schema from data sources"""
        ...

    @abstractmethod
    async def detect_conflicts(
        self,
        schemas: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect schema conflicts across sources"""
        ...

    @abstractmethod
    async def plan_indexing(
        self,
        schemas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan indexing strategy"""
        ...

    @abstractmethod
    async def execute_indexing(
        self,
        plan: Dict[str, Any]
    ) -> SafetyCheck:
        """Execute the indexing plan"""
        ...


class OrbitProtocol(ABC):
    """
    Orbital operations - stable, continuous operation

    Responsibilities:
    - Maintain stable orbit (streaming, querying)
    - App docking and interoperation
    - Yarn stabilization
    - Event fusion
    """

    @abstractmethod
    async def achieve_orbit(self) -> OrbitStatus:
        """Transition to stable orbit"""
        ...

    @abstractmethod
    async def dock_app(
        self,
        app_id: str,
        app_manifest: Dict[str, Any]
    ) -> SafetyCheck:
        """Dock an app to the orbital loom"""
        ...

    @abstractmethod
    async def undock_app(self, app_id: str) -> None:
        """Undock an app from the orbital loom"""
        ...

    @abstractmethod
    async def get_orbit_status(self) -> OrbitStatus:
        """Get current orbital status"""
        ...


class EVAProtocol(ABC):
    """
    EVA (Extra-Vehicular Activity) - manual intervention mode

    Responsibilities:
    - Airlock cycling (enter/exit EVA mode)
    - Heddle tensioning (graph structure adjustment)
    - Manual semantic tuning
    - Safety verification for manual operations
    """

    @abstractmethod
    async def cycle_airlock(
        self,
        direction: str  # "enter" | "exit"
    ) -> SafetyCheck:
        """Cycle airlock to enter/exit EVA mode"""
        ...

    @abstractmethod
    async def start_eva(
        self,
        astronaut_id: str,
        purpose: str
    ) -> EVASession:
        """Begin an EVA session"""
        ...

    @abstractmethod
    async def tension_heddle(
        self,
        eva_session: EVASession,
        heddle_id: str,
        adjustment: Dict[str, Any]
    ) -> SafetyCheck:
        """Adjust a heddle (graph structure)"""
        ...

    @abstractmethod
    async def end_eva(
        self,
        eva_session: EVASession
    ) -> EVASession:
        """End an EVA session and return to cabin"""
        ...


class MissionControlProtocol(ABC):
    """
    Mission Control - monitoring, safety, and support

    Responsibilities:
    - Real-time telemetry
    - Anomaly detection
    - Emergency rollback
    - Safety overrides
    - Audit logging
    """

    @abstractmethod
    async def get_telemetry(self) -> TelemetrySnapshot:
        """Get current system telemetry"""
        ...

    @abstractmethod
    async def detect_anomalies(
        self,
        telemetry: TelemetrySnapshot
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in telemetry"""
        ...

    @abstractmethod
    async def emergency_rollback(
        self,
        target_checkpoint: Optional[LaunchCheckpoint] = None
    ) -> SafetyCheck:
        """Execute emergency rollback to a previous checkpoint"""
        ...

    @abstractmethod
    async def override_safety(
        self,
        astronaut_id: str,
        reason: str,
        authorization: str
    ) -> SafetyCheck:
        """Override a safety hold (requires high-level authorization)"""
        ...


# ==================== Launch System Orchestrator ====================

class LaunchSystemOrchestrator(ABC):
    """
    Main launch system orchestrator - coordinates entire lifecycle

    The orchestrator is a state machine that manages transitions between:
    GROUNDED → PREFLIGHT → COUNTDOWN → LIFTOFF → BOOST → ORBIT → [EVA] → LANDING

    At any point, the system can enter ABORT for emergency rollback.
    """

    def __init__(
        self,
        preflight: PreflightProtocol,
        countdown: CountdownProtocol,
        liftoff: LiftoffProtocol,
        boost: BoostProtocol,
        orbit: OrbitProtocol,
        eva: EVAProtocol,
        mission_control: MissionControlProtocol
    ):
        self.preflight = preflight
        self.countdown = countdown
        self.liftoff = liftoff
        self.boost = boost
        self.orbit = orbit
        self.eva = eva
        self.mission_control = mission_control

        self.current_stage = LaunchStage.GROUNDED
        self.checkpoints: List[LaunchCheckpoint] = []
        self.telemetry_callbacks: List[Callable] = []

    @abstractmethod
    async def execute_preflight(
        self,
        mission_params: Dict[str, Any]
    ) -> PreflightReport:
        """Execute preflight checks"""
        ...

    @abstractmethod
    async def execute_countdown(self) -> CountdownStatus:
        """Execute launch countdown"""
        ...

    @abstractmethod
    async def execute_liftoff(self) -> SafetyCheck:
        """Execute lift-off operations"""
        ...

    @abstractmethod
    async def execute_boost(self) -> SafetyCheck:
        """Execute booster operations"""
        ...

    @abstractmethod
    async def achieve_orbit(self) -> OrbitStatus:
        """Achieve stable orbit"""
        ...

    @abstractmethod
    async def enter_eva_mode(
        self,
        astronaut_id: str,
        purpose: str
    ) -> EVASession:
        """Enter EVA mode from orbit"""
        ...

    @abstractmethod
    async def land(self) -> SafetyCheck:
        """Graceful shutdown and landing"""
        ...

    @abstractmethod
    async def abort_mission(
        self,
        reason: str,
        rollback_to: Optional[LaunchStage] = None
    ) -> SafetyCheck:
        """Emergency abort and rollback"""
        ...

    @abstractmethod
    async def create_checkpoint(self) -> LaunchCheckpoint:
        """Create a rollback checkpoint at current state"""
        ...

    @abstractmethod
    async def get_current_stage(self) -> LaunchStage:
        """Get current launch stage"""
        ...

    @abstractmethod
    async def register_telemetry_callback(
        self,
        callback: Callable[[TelemetrySnapshot], None]
    ) -> None:
        """Register a callback for telemetry updates"""
        ...
