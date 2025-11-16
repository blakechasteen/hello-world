"""
SOAR Core Engine - Security Orchestration, Automation, and Response

Automated incident response playbooks with trigger-based execution.

**Implemented: 2025-11-16**
**Phase**: 4 - Security Pipeline
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Set
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PlaybookSeverity(str, Enum):
    """Playbook severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PlaybookStatus(str, Enum):
    """Playbook execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionMode(str, Enum):
    """Playbook execution modes"""
    PRODUCTION = "production"  # Execute all actions
    DRY_RUN = "dry_run"  # Simulate without executing
    TEST = "test"  # Execute with test fixtures


@dataclass
class SecurityEvent:
    """Security event that triggers playbooks"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""  # "security.attack.sql_injection"
    severity: PlaybookSeverity = PlaybookSeverity.INFO
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    payload: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity.value,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "payload": self.payload,
            "metadata": self.metadata,
        }


@dataclass
class PlaybookResult:
    """Result of playbook execution"""
    playbook_name: str
    execution_id: str
    status: PlaybookStatus
    actions_taken: List[str] = field(default_factory=list)
    actions_failed: List[str] = field(default_factory=list)
    incident_id: Optional[str] = None
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if playbook succeeded"""
        return self.status == PlaybookStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "playbook_name": self.playbook_name,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "success": self.success,
            "actions_taken": self.actions_taken,
            "actions_failed": self.actions_failed,
            "incident_id": self.incident_id,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class PlaybookMetadata:
    """Metadata for playbook registration"""
    name: str
    description: str
    severity: PlaybookSeverity
    triggers: List[str]  # Event types that trigger this playbook
    version: str = "1.0.0"
    author: str = "HoloLoom Security Team"
    requires_approval: bool = False  # Manual approval required
    timeout_seconds: float = 300.0
    tags: List[str] = field(default_factory=list)


@dataclass
class PlaybookExecution:
    """Track playbook execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    playbook_name: str = ""
    event: Optional[SecurityEvent] = None
    status: PlaybookStatus = PlaybookStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actions_log: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[PlaybookResult] = None
    mode: ExecutionMode = ExecutionMode.PRODUCTION


class PlaybookRegistry:
    """Registry for playbook functions"""

    def __init__(self):
        self._playbooks: Dict[str, Callable] = {}
        self._metadata: Dict[str, PlaybookMetadata] = {}
        self._trigger_map: Dict[str, List[str]] = defaultdict(list)

    def register(self, metadata: PlaybookMetadata, func: Callable):
        """Register a playbook"""
        self._playbooks[metadata.name] = func
        self._metadata[metadata.name] = metadata

        # Build trigger map
        for trigger in metadata.triggers:
            self._trigger_map[trigger].append(metadata.name)

        logger.info(f"Registered playbook: {metadata.name} (triggers: {metadata.triggers})")

    def get_playbook(self, name: str) -> Optional[Callable]:
        """Get playbook function by name"""
        return self._playbooks.get(name)

    def get_metadata(self, name: str) -> Optional[PlaybookMetadata]:
        """Get playbook metadata"""
        return self._metadata.get(name)

    def get_playbooks_for_trigger(self, trigger: str) -> List[str]:
        """Get playbook names that match a trigger"""
        # Exact match
        if trigger in self._trigger_map:
            return self._trigger_map[trigger].copy()

        # Prefix match (e.g., "security.attack.*" matches "security.attack.sql_injection")
        matches = []
        for registered_trigger, playbooks in self._trigger_map.items():
            if self._matches_trigger(trigger, registered_trigger):
                matches.extend(playbooks)

        return list(set(matches))  # Deduplicate

    def _matches_trigger(self, event_type: str, trigger_pattern: str) -> bool:
        """Check if event type matches trigger pattern"""
        # Exact match
        if event_type == trigger_pattern:
            return True

        # Wildcard match (e.g., "security.attack.*")
        if trigger_pattern.endswith("*"):
            prefix = trigger_pattern[:-1]
            return event_type.startswith(prefix)

        return False

    def list_playbooks(self) -> List[str]:
        """List all registered playbooks"""
        return list(self._playbooks.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_playbooks": len(self._playbooks),
            "total_triggers": len(self._trigger_map),
            "by_severity": {
                severity.value: len([m for m in self._metadata.values() if m.severity == severity])
                for severity in PlaybookSeverity
            },
        }


class SOAREngine:
    """
    Security Orchestration, Automation, and Response Engine

    Executes automated incident response playbooks based on security events.
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        enable_audit_log: bool = True,
        audit_log_path: Optional[Path] = None,
    ):
        """
        Initialize SOAR engine

        Args:
            mode: Execution mode (production/dry_run/test)
            enable_audit_log: Enable audit logging
            audit_log_path: Path for audit logs
        """
        self.mode = mode
        self.enable_audit_log = enable_audit_log
        self.audit_log_path = audit_log_path or Path("./soar_audit.jsonl")

        self.registry = PlaybookRegistry()
        self.executions: Dict[str, PlaybookExecution] = {}
        self.pending_approvals: Set[str] = set()

        # Statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "dry_run_executions": 0,
        }

        logger.info(f"SOAR Engine initialized (mode: {mode.value})")

    def register_playbook(self, metadata: PlaybookMetadata):
        """
        Decorator to register a playbook

        Usage:
            @soar.register_playbook(PlaybookMetadata(
                name="SQL Injection Response",
                triggers=["security.attack.sql_injection"],
                severity=PlaybookSeverity.CRITICAL
            ))
            async def sql_injection_response(event: SecurityEvent, actions):
                await actions.block_ip(event.source_ip)
                return PlaybookResult(...)
        """
        def decorator(func: Callable):
            self.registry.register(metadata, func)
            return func
        return decorator

    async def process_event(
        self,
        event: SecurityEvent,
        override_mode: Optional[ExecutionMode] = None,
    ) -> List[PlaybookResult]:
        """
        Process security event and execute matching playbooks

        Args:
            event: Security event
            override_mode: Override execution mode

        Returns:
            List of playbook results
        """
        mode = override_mode or self.mode

        # Find matching playbooks
        playbook_names = self.registry.get_playbooks_for_trigger(event.event_type)

        if not playbook_names:
            logger.debug(f"No playbooks match event: {event.event_type}")
            return []

        logger.info(
            f"Event {event.event_id} ({event.event_type}) triggered "
            f"{len(playbook_names)} playbook(s): {playbook_names}"
        )

        # Execute playbooks
        results = []
        for playbook_name in playbook_names:
            result = await self.execute_playbook(playbook_name, event, mode)
            results.append(result)

        return results

    async def execute_playbook(
        self,
        playbook_name: str,
        event: SecurityEvent,
        mode: Optional[ExecutionMode] = None,
    ) -> PlaybookResult:
        """
        Execute a specific playbook

        Args:
            playbook_name: Name of playbook to execute
            event: Security event
            mode: Execution mode

        Returns:
            Playbook result
        """
        mode = mode or self.mode

        # Get playbook
        playbook_func = self.registry.get_playbook(playbook_name)
        metadata = self.registry.get_metadata(playbook_name)

        if not playbook_func or not metadata:
            error_msg = f"Playbook not found: {playbook_name}"
            logger.error(error_msg)
            return PlaybookResult(
                playbook_name=playbook_name,
                execution_id=str(uuid.uuid4()),
                status=PlaybookStatus.FAILED,
                error_message=error_msg,
            )

        # Create execution record
        execution = PlaybookExecution(
            playbook_name=playbook_name,
            event=event,
            mode=mode,
        )
        self.executions[execution.execution_id] = execution

        # Check if manual approval required
        if metadata.requires_approval and mode == ExecutionMode.PRODUCTION:
            execution.status = PlaybookStatus.PENDING
            self.pending_approvals.add(execution.execution_id)
            logger.warning(
                f"Playbook '{playbook_name}' requires manual approval "
                f"(execution_id: {execution.execution_id})"
            )
            return PlaybookResult(
                playbook_name=playbook_name,
                execution_id=execution.execution_id,
                status=PlaybookStatus.PENDING,
                metadata={"requires_approval": True},
            )

        # Execute playbook
        execution.status = PlaybookStatus.RUNNING
        execution.started_at = datetime.utcnow()

        try:
            # Import actions module here to avoid circular import
            from .actions import ActionExecutor

            # Create action executor
            actions = ActionExecutor(
                mode=mode,
                execution_id=execution.execution_id,
                event=event,
            )

            # Execute with timeout
            result = await asyncio.wait_for(
                playbook_func(event, actions),
                timeout=metadata.timeout_seconds,
            )

            # Update execution
            execution.status = PlaybookStatus.SUCCESS
            execution.completed_at = datetime.utcnow()
            execution.actions_log = actions.get_action_log()
            execution.result = result

            # Update result
            result.execution_id = execution.execution_id
            result.status = PlaybookStatus.SUCCESS
            result.duration_ms = (
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )

            # Update statistics
            self.stats["total_executions"] += 1
            if mode == ExecutionMode.DRY_RUN:
                self.stats["dry_run_executions"] += 1
            else:
                self.stats["successful_executions"] += 1

            # Audit log
            if self.enable_audit_log:
                await self._write_audit_log(execution, result)

            logger.info(
                f"Playbook '{playbook_name}' executed successfully "
                f"(mode: {mode.value}, duration: {result.duration_ms:.0f}ms, "
                f"actions: {len(result.actions_taken)})"
            )

            return result

        except asyncio.TimeoutError:
            execution.status = PlaybookStatus.TIMEOUT
            execution.completed_at = datetime.utcnow()

            error_msg = f"Playbook timeout ({metadata.timeout_seconds}s)"
            logger.error(f"{error_msg}: {playbook_name}")

            self.stats["total_executions"] += 1
            self.stats["failed_executions"] += 1

            return PlaybookResult(
                playbook_name=playbook_name,
                execution_id=execution.execution_id,
                status=PlaybookStatus.TIMEOUT,
                error_message=error_msg,
            )

        except Exception as e:
            execution.status = PlaybookStatus.FAILED
            execution.completed_at = datetime.utcnow()

            error_msg = f"Playbook error: {str(e)}"
            logger.exception(f"Playbook '{playbook_name}' failed: {e}")

            self.stats["total_executions"] += 1
            self.stats["failed_executions"] += 1

            return PlaybookResult(
                playbook_name=playbook_name,
                execution_id=execution.execution_id,
                status=PlaybookStatus.FAILED,
                error_message=error_msg,
            )

    async def approve_execution(self, execution_id: str) -> bool:
        """Manually approve a pending playbook execution"""
        if execution_id not in self.pending_approvals:
            logger.warning(f"Execution not pending approval: {execution_id}")
            return False

        execution = self.executions.get(execution_id)
        if not execution or not execution.event:
            logger.error(f"Execution not found: {execution_id}")
            return False

        self.pending_approvals.remove(execution_id)

        # Re-execute without approval check
        metadata = self.registry.get_metadata(execution.playbook_name)
        if metadata:
            metadata.requires_approval = False

        await self.execute_playbook(
            execution.playbook_name,
            execution.event,
            execution.mode,
        )

        return True

    async def _write_audit_log(self, execution: PlaybookExecution, result: PlaybookResult):
        """Write execution to audit log"""
        try:
            log_entry = {
                "execution_id": execution.execution_id,
                "playbook_name": execution.playbook_name,
                "event": execution.event.to_dict() if execution.event else None,
                "status": execution.status.value,
                "mode": execution.mode.value,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_ms": result.duration_ms,
                "actions_taken": result.actions_taken,
                "actions_failed": result.actions_failed,
                "incident_id": result.incident_id,
                "error_message": result.error_message,
            }

            # Append to JSONL file
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_execution(self, execution_id: str) -> Optional[PlaybookExecution]:
        """Get execution by ID"""
        return self.executions.get(execution_id)

    def list_executions(
        self,
        status: Optional[PlaybookStatus] = None,
        playbook_name: Optional[str] = None,
    ) -> List[PlaybookExecution]:
        """List executions with optional filters"""
        executions = list(self.executions.values())

        if status:
            executions = [e for e in executions if e.status == status]

        if playbook_name:
            executions = [e for e in executions if e.playbook_name == playbook_name]

        return sorted(
            executions,
            key=lambda e: e.started_at or datetime.min,
            reverse=True,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.stats,
            "registry": self.registry.get_statistics(),
            "pending_approvals": len(self.pending_approvals),
            "total_stored_executions": len(self.executions),
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cleanup if needed
        pass


# Factory function
def create_soar_engine(
    mode: ExecutionMode = ExecutionMode.PRODUCTION,
    enable_audit_log: bool = True,
    audit_log_path: Optional[Path] = None,
) -> SOAREngine:
    """
    Create SOAR engine

    Args:
        mode: Execution mode
        enable_audit_log: Enable audit logging
        audit_log_path: Custom audit log path

    Returns:
        SOAREngine instance
    """
    return SOAREngine(
        mode=mode,
        enable_audit_log=enable_audit_log,
        audit_log_path=audit_log_path,
    )
