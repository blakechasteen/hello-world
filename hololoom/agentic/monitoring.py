"""
Real-Time Agent Monitoring System

Provides live visibility into agent reasoning with:
- Tree structure visualization of reasoning steps
- WebSocket broadcasting for real-time updates
- Multi-agent concurrent tracking
- Performance metrics per agent

Status: Production Ready (November 2025)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import asyncio
import logging
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"  # Blocked on dependency
    VERIFY = "verify"    # Verification mode
    RESEARCH = "research"  # Research mode
    COMPLETED = "completed"
    FAILED = "failed"


class StepType(Enum):
    """Reasoning step types"""
    INITIAL_ANSWER = "initial_answer"
    VERIFICATION = "verification"
    RESEARCH_QUERY = "research_query"
    SYNTHESIS = "synthesis"
    PLAN = "plan"
    EXECUTION = "execution"


@dataclass
class AgentTreeNode:
    """Node in the agent reasoning tree"""
    node_id: str
    agent_id: str
    step_type: StepType
    query: Optional[str] = None
    confidence: Optional[float] = None
    epistemic_confidence: Optional[float] = None
    finding: Optional[str] = None
    tool_used: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Tree structure
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

    # Status
    status: AgentStatus = AgentStatus.RUNNING
    error: Optional[str] = None


@dataclass
class AgentSession:
    """Tracks a single agent reasoning session"""
    agent_id: str
    project: str
    query: str
    mode: str  # DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE
    status: AgentStatus

    # Tree structure
    root_node_id: Optional[str] = None
    nodes: Dict[str, AgentTreeNode] = field(default_factory=dict)

    # Files being worked on
    files: List[str] = field(default_factory=list)

    # Two-line feed
    feed_line1: str = ""
    feed_line2: str = ""

    # Performance
    total_steps: int = 0
    current_step: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    total_duration_ms: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentTreeBuilder:
    """Converts flat step list to tree structure"""

    @staticmethod
    def build_tree(agent_id: str, steps: List[Dict[str, Any]]) -> Dict[str, AgentTreeNode]:
        """
        Build tree from flat steps list.

        Tree structure:
        - DIRECT mode: Single root node
        - VERIFY mode: Root + N verification child nodes
        - RESEARCH mode: Root + N research query child nodes
        - PLAN_EXECUTE mode: Root (plan) + N execution child nodes
        """
        nodes = {}
        root_node_id = None

        for i, step in enumerate(steps):
            node_id = f"{agent_id}_step_{i}"

            # Determine step type
            step_type_str = step.get('type', 'initial_answer')
            try:
                step_type = StepType(step_type_str)
            except ValueError:
                # Fallback for unknown types
                step_type = StepType.INITIAL_ANSWER

            # Create node
            node = AgentTreeNode(
                node_id=node_id,
                agent_id=agent_id,
                step_type=step_type,
                query=step.get('query'),
                confidence=step.get('confidence'),
                epistemic_confidence=step.get('epistemic_confidence'),
                finding=step.get('finding'),
                tool_used=step.get('tool_used'),
                latency_ms=step.get('latency_ms'),
                status=AgentStatus.COMPLETED if step.get('completed') else AgentStatus.RUNNING
            )

            # Determine parent relationship
            if i == 0:
                # First node is root
                root_node_id = node_id
                node.parent_id = None
            else:
                # Subsequent nodes are children of root (for VERIFY/RESEARCH)
                # Or sequential (for PLAN_EXECUTE)
                prev_node_id = f"{agent_id}_step_{i-1}"

                if step_type in [StepType.VERIFICATION, StepType.RESEARCH_QUERY]:
                    # Multi-query modes: attach to root
                    node.parent_id = root_node_id
                    if root_node_id and root_node_id in nodes:
                        nodes[root_node_id].children.append(node_id)
                else:
                    # Sequential modes: chain
                    node.parent_id = prev_node_id
                    if prev_node_id in nodes:
                        nodes[prev_node_id].children.append(node_id)

            nodes[node_id] = node

        return nodes

    @staticmethod
    def get_tree_depth(nodes: Dict[str, AgentTreeNode], root_id: str) -> int:
        """Calculate tree depth"""
        if not root_id or root_id not in nodes:
            return 0

        max_depth = 1
        for child_id in nodes[root_id].children:
            child_depth = AgentTreeBuilder.get_tree_depth(nodes, child_id)
            max_depth = max(max_depth, 1 + child_depth)

        return max_depth


class AgentMonitor:
    """
    Real-time agent monitoring system.

    Responsibilities:
    - Track active agent sessions
    - Build tree structures from reasoning steps
    - Broadcast updates to WebSocket clients
    - Collect performance metrics
    """

    def __init__(self):
        # Active sessions
        self.sessions: Dict[str, AgentSession] = {}

        # Project-based grouping
        self.projects: Dict[str, Set[str]] = defaultdict(set)  # project -> agent_ids

        # WebSocket connections
        self.ws_connections: List[Any] = []  # WebSocket instances

        # Performance metrics
        self.total_agents_started: int = 0
        self.total_agents_completed: int = 0
        self.total_agents_failed: int = 0
        self.latencies: deque = deque(maxlen=1000)

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start monitoring (background cleanup)"""
        logger.info("Starting agent monitoring system")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop monitoring"""
        logger.info("Stopping agent monitoring system")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.debug("Cleanup task cancelled successfully")
                pass

    async def _cleanup_loop(self):
        """Periodic cleanup of old completed sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Remove sessions older than 1 hour
                now = datetime.now()
                to_remove = []

                for agent_id, session in self.sessions.items():
                    if session.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
                        start_time = datetime.fromisoformat(session.start_time)
                        age_seconds = (now - start_time).total_seconds()

                        if age_seconds > 3600:  # 1 hour
                            to_remove.append(agent_id)

                if to_remove:
                    logger.info(f"Cleanup: Removing {len(to_remove)} old sessions")

                for agent_id in to_remove:
                    session = self.sessions.pop(agent_id)
                    self.projects[session.project].discard(agent_id)
                    logger.debug(
                        f"Cleanup: Removed session {agent_id} "
                        f"(project={session.project}, status={session.status.value})"
                    )

            except asyncio.CancelledError:
                logger.debug("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup failed: {e}", exc_info=True)

    def register_ws_connection(self, ws):
        """Register WebSocket connection"""
        self.ws_connections.append(ws)
        logger.info(f"WebSocket connected (total: {len(self.ws_connections)})")

    def unregister_ws_connection(self, ws):
        """Unregister WebSocket connection"""
        if ws in self.ws_connections:
            self.ws_connections.remove(ws)
            logger.info(f"WebSocket disconnected (remaining: {len(self.ws_connections)})")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients"""
        disconnected = []

        for ws in self.ws_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Cleanup disconnected clients
        for ws in disconnected:
            self.unregister_ws_connection(ws)

    async def agent_started(
        self,
        agent_id: str,
        project: str,
        query: str,
        mode: str,
        files: Optional[List[str]] = None
    ):
        """Agent started reasoning"""
        self.total_agents_started += 1

        # Create session
        session = AgentSession(
            agent_id=agent_id,
            project=project,
            query=query,
            mode=mode,
            status=AgentStatus.RUNNING,
            files=files or []
        )

        self.sessions[agent_id] = session
        self.projects[project].add(agent_id)

        logger.info(
            f"Agent started: {agent_id} (project={project}, mode={mode}, "
            f"query='{query[:50]}...')"
        )

        # Broadcast
        await self.broadcast({
            'type': 'agent_started',
            'agent_id': agent_id,
            'project': project,
            'query': query,
            'mode': mode,
            'files': files or [],
            'timestamp': datetime.now().isoformat()
        })

    async def agent_step(
        self,
        agent_id: str,
        step: int,
        total_steps: int,
        step_type: str,
        confidence: Optional[float] = None,
        epistemic: Optional[float] = None,
        finding: Optional[str] = None,
        latency_ms: Optional[float] = None
    ):
        """Agent completed a step"""
        if agent_id not in self.sessions:
            return

        session = self.sessions[agent_id]
        session.current_step = step
        session.total_steps = total_steps

        # Track latency
        if latency_ms is not None:
            self.latencies.append(latency_ms)

        # Broadcast
        await self.broadcast({
            'type': 'agent_step',
            'agent_id': agent_id,
            'step': step,
            'total_steps': total_steps,
            'step_type': step_type,
            'confidence': confidence,
            'epistemic': epistemic,
            'finding': finding,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        })

    async def agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        files: Optional[List[str]] = None
    ):
        """Agent status changed"""
        if agent_id not in self.sessions:
            return

        session = self.sessions[agent_id]
        session.status = status

        if files is not None:
            session.files = files

        # Update metrics
        if status == AgentStatus.COMPLETED:
            self.total_agents_completed += 1
        elif status == AgentStatus.FAILED:
            self.total_agents_failed += 1

        # Broadcast
        await self.broadcast({
            'type': 'agent_status',
            'agent_id': agent_id,
            'status': status.value,
            'files': session.files,
            'timestamp': datetime.now().isoformat()
        })

    async def agent_feed(
        self,
        agent_id: str,
        line1: str,
        line2: str = ""
    ):
        """Update agent's two-line feed"""
        if agent_id not in self.sessions:
            return

        session = self.sessions[agent_id]
        session.feed_line1 = line1
        session.feed_line2 = line2

        # Broadcast
        await self.broadcast({
            'type': 'agent_feed',
            'agent_id': agent_id,
            'line1': line1,
            'line2': line2,
            'timestamp': datetime.now().isoformat()
        })

    async def agent_completed(
        self,
        agent_id: str,
        steps: List[Dict[str, Any]],
        total_duration_ms: float
    ):
        """Agent completed reasoning"""
        if agent_id not in self.sessions:
            logger.warning(f"Agent completed but session not found: {agent_id}")
            return

        session = self.sessions[agent_id]
        session.status = AgentStatus.COMPLETED
        session.total_duration_ms = total_duration_ms

        # Build tree structure
        session.nodes = AgentTreeBuilder.build_tree(agent_id, steps)
        if session.nodes:
            session.root_node_id = f"{agent_id}_step_0"

        # Track metrics
        self.total_agents_completed += 1
        self.latencies.append(total_duration_ms)

        logger.info(
            f"Agent completed: {agent_id} (project={session.project}, "
            f"duration={total_duration_ms:.1f}ms, steps={len(steps)})"
        )

        # Broadcast
        await self.broadcast({
            'type': 'agent_completed',
            'agent_id': agent_id,
            'total_duration_ms': total_duration_ms,
            'tree': self._serialize_tree(session),
            'timestamp': datetime.now().isoformat()
        })

    async def agent_failed(
        self,
        agent_id: str,
        error: str
    ):
        """Agent failed with error"""
        if agent_id not in self.sessions:
            logger.warning(f"Agent failed but session not found: {agent_id}")
            return

        session = self.sessions[agent_id]
        session.status = AgentStatus.FAILED

        # Track metrics
        self.total_agents_failed += 1

        logger.error(
            f"Agent failed: {agent_id} (project={session.project}, "
            f"error='{error[:100]}...')"
        )

        # Broadcast
        await self.broadcast({
            'type': 'agent_failed',
            'agent_id': agent_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

    def _serialize_tree(self, session: AgentSession) -> Dict[str, Any]:
        """Serialize tree structure for WebSocket transmission"""
        if not session.root_node_id or not session.nodes:
            return {}

        def serialize_node(node_id: str) -> Dict[str, Any]:
            if node_id not in session.nodes:
                return {}

            node = session.nodes[node_id]
            return {
                'node_id': node.node_id,
                'step_type': node.step_type.value,
                'query': node.query,
                'confidence': node.confidence,
                'epistemic_confidence': node.epistemic_confidence,
                'finding': node.finding,
                'tool_used': node.tool_used,
                'latency_ms': node.latency_ms,
                'status': node.status.value,
                'children': [serialize_node(child_id) for child_id in node.children]
            }

        return serialize_node(session.root_node_id)

    def get_session(self, agent_id: str) -> Optional[AgentSession]:
        """Get session by agent ID"""
        return self.sessions.get(agent_id)

    def get_project_agents(self, project: str) -> List[AgentSession]:
        """Get all agents for a project"""
        agent_ids = self.projects.get(project, set())
        return [self.sessions[aid] for aid in agent_ids if aid in self.sessions]

    def get_all_sessions(self) -> List[AgentSession]:
        """Get all active sessions"""
        return list(self.sessions.values())

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_agents_started': self.total_agents_started,
            'total_agents_completed': self.total_agents_completed,
            'total_agents_failed': self.total_agents_failed,
            'active_agents': len(self.sessions),
            'avg_latency_ms': sum(self.latencies) / len(self.latencies) if self.latencies else 0,
            'success_rate': (
                self.total_agents_completed / self.total_agents_started
                if self.total_agents_started > 0 else 0
            ),
            'projects': list(self.projects.keys()),
            'ws_connections': len(self.ws_connections)
        }


# Global monitor instance
_monitor: Optional[AgentMonitor] = None


def get_monitor() -> AgentMonitor:
    """Get or create global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = AgentMonitor()
    return _monitor


async def start_monitoring():
    """Start global monitoring"""
    monitor = get_monitor()
    await monitor.start()


async def stop_monitoring():
    """Stop global monitoring"""
    global _monitor
    if _monitor:
        await _monitor.stop()
        _monitor = None
