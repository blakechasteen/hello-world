"""
PII Flow Tracking and Audit Trail System
=========================================

**Purpose**: Track the flow of PII through HoloLoom's processing pipeline,
maintaining complete audit trails for compliance (GDPR, HIPAA, SOC2).

**Date**: 2025-11-17
**Phase**: Tenant Isolation and PII Flows
**Key Features**:
- End-to-end PII flow tracking from ingestion → processing → storage → retrieval
- Automatic PII detection and classification at each stage
- Complete audit trail with timestamps and provenance
- Compliance reporting (GDPR Article 30 records)
- Data lineage visualization
- Right-to-erasure automation

**Philosophy**:
"Every piece of PII has a story: where it came from, where it went,
who accessed it, and when. We tell that story completely."

**Compliance**:
- GDPR Article 30: Records of processing activities
- GDPR Article 17: Right to erasure
- HIPAA § 164.312(b): Audit controls
- SOC 2 CC7.2: System monitoring

Author: HoloLoom Security Team
Timestamp: 2025-11-17
"""

from __future__ import annotations
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4
import logging
import json
import re
from pathlib import Path

from .pii_detection import PIIType, PIIDetection, PIIDetector, PIIAnalysisResult
from .tenant_isolation import TenantContext


logger = logging.getLogger(__name__)


# ============================================================================
# Input Sanitization for Logging
# ============================================================================

def _sanitize_log_field(value: str) -> str:
    """
    Sanitize user input for safe logging.

    Security transformation:
    - Remove newlines (CR, LF) to prevent log injection
    - Remove control characters (0x00-0x1F except tab/space)
    - Preserve meaningful whitespace (space, tab)

    Args:
        value: Raw user input

    Returns:
        Sanitized string safe for logging

    Examples:
        >>> _sanitize_log_field("legitimate\\n[ADMIN] Unauthorized access")
        "legitimate [ADMIN] Unauthorized access"

        >>> _sanitize_log_field("test\\x00\\x01\\x02")
        "test"

    References:
        - CWE-117: Improper Output Neutralization for Logs
        - OWASP Logging Cheat Sheet
    """
    if not value:
        return value

    # Remove newlines (CR, LF) - primary log injection vector
    # Replace with spaces to prevent word concatenation
    sanitized = value.replace('\r', ' ').replace('\n', ' ')

    # Remove control characters except tab (0x09) and space (0x20)
    # Control chars: 0x00-0x1F (except 0x09 tab)
    sanitized = re.sub(r'[\x00-\x08\x0A-\x1F]', '', sanitized)

    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized)

    return sanitized.strip()


# ============================================================================
# PII Flow Types
# ============================================================================

class PIIFlowStage(Enum):
    """
    Stages in the PII processing pipeline.

    Tracks where PII is at each step of processing.
    """
    INGESTION = "ingestion"              # Initial input (user query, document upload)
    EXTRACTION = "extraction"            # Feature extraction (embeddings, motifs)
    MEMORY_WRITE = "memory_write"        # Writing to graph/vector store
    MEMORY_READ = "memory_read"          # Reading from memory
    PROCESSING = "processing"            # Internal processing (reasoning, synthesis)
    OUTPUT_GENERATION = "output_generation"  # Generating response
    TRANSMISSION = "transmission"        # Sending response to client
    DELETION = "deletion"                # PII deletion (right to erasure)


class PIIAction(Enum):
    """Actions performed on PII."""
    DETECTED = "detected"                # PII detected
    STORED = "stored"                    # PII stored to memory
    RETRIEVED = "retrieved"              # PII retrieved from memory
    TRANSFORMED = "transformed"          # PII transformed (redacted, encrypted)
    ACCESSED = "accessed"                # PII accessed/read
    DELETED = "deleted"                  # PII deleted
    EXPORTED = "exported"                # PII exported (GDPR data portability)


@dataclass
class PIIFlowEvent:
    """
    A single event in the PII flow.

    Captures:
    - What PII was involved
    - What action was taken
    - When and where
    - Who performed the action
    - Why (purpose)

    Example:
        event = PIIFlowEvent(
            event_id=uuid4(),
            tenant_context=context,
            stage=PIIFlowStage.INGESTION,
            action=PIIAction.DETECTED,
            pii_detections=[detection],
            purpose="User submitted query with email",
            metadata={"query_id": "q_123"}
        )
    """
    event_id: UUID                               # Unique event ID
    tenant_context: TenantContext                # Who/where
    stage: PIIFlowStage                          # Which stage
    action: PIIAction                            # What action
    pii_detections: List[PIIDetection]           # PII involved
    purpose: str                                 # Why this action
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

    # Data lineage
    parent_event_id: Optional[UUID] = None       # Previous event in chain
    child_event_ids: List[UUID] = field(default_factory=list)  # Subsequent events

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging."""
        return {
            "event_id": str(self.event_id),
            "tenant_id": self.tenant_context.tenant_id,
            "user_id": self.tenant_context.user_id,
            "request_id": str(self.tenant_context.request_id),
            "stage": self.stage.value,
            "action": self.action.value,
            "pii_types": [d.pii_type.value for d in self.pii_detections],
            "pii_count": len(self.pii_detections),
            "purpose": self.purpose,
            "timestamp": self.timestamp.isoformat(),
            "parent_event_id": str(self.parent_event_id) if self.parent_event_id else None,
            "metadata": self.metadata,
        }


@dataclass
class PIIFlowChain:
    """
    Complete chain of PII flow events.

    Represents the full lifecycle of PII through the system:
    Ingestion → Extraction → Storage → Retrieval → Output → Deletion

    Example:
        chain = PIIFlowChain(
            chain_id=uuid4(),
            tenant_id="acme_corp",
            initial_event=ingestion_event,
            events=[ingestion_event, extraction_event, storage_event, ...],
            pii_types={PIIType.EMAIL, PIIType.PHONE}
        )
    """
    chain_id: UUID                               # Unique chain ID
    tenant_id: str                               # Which tenant
    initial_event: PIIFlowEvent                  # First event in chain
    events: List[PIIFlowEvent] = field(default_factory=list)  # All events
    pii_types: Set[PIIType] = field(default_factory=set)  # All PII types in chain
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None      # When chain completed (deletion)

    def add_event(self, event: PIIFlowEvent) -> None:
        """Add event to chain."""
        self.events.append(event)
        self.pii_types.update(d.pii_type for d in event.pii_detections)

    def is_complete(self) -> bool:
        """Check if chain is complete (ends with deletion)."""
        if not self.events:
            return False
        return self.events[-1].action == PIIAction.DELETED

    def get_lifecycle_duration(self) -> Optional[timedelta]:
        """Get duration from ingestion to deletion."""
        if not self.is_complete():
            return None
        return self.completed_at - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chain_id": str(self.chain_id),
            "tenant_id": self.tenant_id,
            "initial_event_id": str(self.initial_event.event_id),
            "event_count": len(self.events),
            "pii_types": [pii_type.value for pii_type in self.pii_types],
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "lifecycle_duration_seconds": self.get_lifecycle_duration().total_seconds() if self.get_lifecycle_duration() else None,
            "is_complete": self.is_complete(),
        }


# ============================================================================
# PII Flow Tracker
# ============================================================================

class PIIFlowTracker:
    """
    Tracks PII flow through the system.

    Features:
    - Automatic PII detection at each stage
    - Event logging with full provenance
    - Chain construction (data lineage)
    - Compliance reporting
    - Right-to-erasure automation

    Usage:
        tracker = PIIFlowTracker(detector)

        # Track ingestion
        ingestion_event = await tracker.track_ingestion(
            text="User email is john@acme.com",
            context=tenant_context,
            purpose="User query"
        )

        # Track storage
        storage_event = await tracker.track_storage(
            data={"email": "john@acme.com"},
            context=tenant_context,
            parent_event_id=ingestion_event.event_id,
            purpose="Store to knowledge graph"
        )

        # Track retrieval
        retrieval_event = await tracker.track_retrieval(
            memory_keys=["mem_123"],
            context=tenant_context,
            purpose="Recall for query answering"
        )

        # Track deletion
        deletion_event = await tracker.track_deletion(
            pii_references=["mem_123"],
            context=tenant_context,
            purpose="Right to erasure request"
        )

        # Get compliance report
        report = await tracker.generate_compliance_report(tenant_id="acme_corp")
    """

    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        enable_auto_detection: bool = True,
        persist_path: Optional[Path] = None
    ):
        """
        Initialize PII flow tracker.

        Args:
            detector: PII detector (creates default if None)
            enable_auto_detection: Automatically detect PII in tracked data
            persist_path: Path to persist audit trail (JSON)
        """
        from .pii_detection import create_default_detector

        self.detector = detector or create_default_detector()
        self.enable_auto_detection = enable_auto_detection
        self.persist_path = persist_path

        # Event storage
        self._events: List[PIIFlowEvent] = []
        self._chains: Dict[UUID, PIIFlowChain] = {}

        # Index for fast lookup
        self._events_by_tenant: Dict[str, List[UUID]] = {}
        self._events_by_pii_type: Dict[PIIType, List[UUID]] = {}

        logger.info("PIIFlowTracker initialized (auto_detection=%s)", enable_auto_detection)

    async def track_ingestion(
        self,
        text: str,
        context: TenantContext,
        purpose: str,
        **metadata
    ) -> PIIFlowEvent:
        """
        Track PII ingestion (initial input).

        Args:
            text: Input text to analyze
            context: Tenant context
            purpose: Why this data is being ingested
            **metadata: Additional metadata

        Returns:
            PIIFlowEvent for this ingestion
        """
        # SECURITY: Sanitize purpose to prevent log injection
        safe_purpose = _sanitize_log_field(purpose)

        # Detect PII
        detections = []
        if self.enable_auto_detection:
            analysis = self.detector.analyze(text)
            detections = analysis.detections

            if analysis.has_pii:
                logger.info(
                    "Ingestion: Detected %d PII entities for tenant %s",
                    len(detections),
                    context.tenant_id
                )

        # Create event
        event = PIIFlowEvent(
            event_id=uuid4(),
            tenant_context=context,
            stage=PIIFlowStage.INGESTION,
            action=PIIAction.DETECTED,
            pii_detections=detections,
            purpose=safe_purpose,
            metadata=metadata
        )

        # Store event
        await self._store_event(event)

        # Create new chain
        chain = PIIFlowChain(
            chain_id=event.event_id,  # Chain ID = initial event ID
            tenant_id=context.tenant_id,
            initial_event=event
        )
        chain.add_event(event)
        self._chains[chain.chain_id] = chain

        return event

    async def track_storage(
        self,
        data: Any,
        context: TenantContext,
        parent_event_id: UUID,
        purpose: str,
        **metadata
    ) -> PIIFlowEvent:
        """
        Track PII storage to memory.

        Args:
            data: Data being stored (dict, str, etc.)
            context: Tenant context
            parent_event_id: Previous event in chain
            purpose: Why storing
            **metadata: Additional metadata

        Returns:
            PIIFlowEvent for this storage
        """
        # SECURITY: Sanitize purpose to prevent log injection
        safe_purpose = _sanitize_log_field(purpose)

        # Detect PII in data
        detections = []
        if self.enable_auto_detection:
            # Convert data to string for analysis
            data_str = json.dumps(data) if isinstance(data, dict) else str(data)
            analysis = self.detector.analyze(data_str)
            detections = analysis.detections

        # Create event
        event = PIIFlowEvent(
            event_id=uuid4(),
            tenant_context=context,
            stage=PIIFlowStage.MEMORY_WRITE,
            action=PIIAction.STORED,
            pii_detections=detections,
            purpose=safe_purpose,
            parent_event_id=parent_event_id,
            metadata=metadata
        )

        # Store event
        await self._store_event(event)

        # Add to chain
        chain = self._chains.get(parent_event_id)
        if chain:
            chain.add_event(event)

        return event

    async def track_retrieval(
        self,
        memory_keys: List[str],
        context: TenantContext,
        purpose: str,
        **metadata
    ) -> PIIFlowEvent:
        """
        Track PII retrieval from memory.

        Args:
            memory_keys: Keys of memories being retrieved
            context: Tenant context
            purpose: Why retrieving
            **metadata: Additional metadata

        Returns:
            PIIFlowEvent for this retrieval
        """
        # SECURITY: Sanitize purpose to prevent log injection
        safe_purpose = _sanitize_log_field(purpose)

        # Create event
        event = PIIFlowEvent(
            event_id=uuid4(),
            tenant_context=context,
            stage=PIIFlowStage.MEMORY_READ,
            action=PIIAction.RETRIEVED,
            pii_detections=[],  # Will be filled if PII found in retrieved data
            purpose=safe_purpose,
            metadata={"memory_keys": memory_keys, **metadata}
        )

        # Store event
        await self._store_event(event)

        return event

    async def track_deletion(
        self,
        pii_references: List[str],
        context: TenantContext,
        purpose: str,
        **metadata
    ) -> PIIFlowEvent:
        """
        Track PII deletion (right to erasure).

        Args:
            pii_references: References to deleted PII (memory keys, etc.)
            context: Tenant context
            purpose: Why deleting
            **metadata: Additional metadata

        Returns:
            PIIFlowEvent for this deletion
        """
        # SECURITY: Sanitize purpose to prevent log injection
        safe_purpose = _sanitize_log_field(purpose)

        # Create event
        event = PIIFlowEvent(
            event_id=uuid4(),
            tenant_context=context,
            stage=PIIFlowStage.DELETION,
            action=PIIAction.DELETED,
            pii_detections=[],
            purpose=safe_purpose,
            metadata={"pii_references": pii_references, **metadata}
        )

        # Store event
        await self._store_event(event)

        # Mark chains as complete
        for chain_id, chain in self._chains.items():
            if chain.tenant_id == context.tenant_id:
                chain.add_event(event)
                chain.completed_at = datetime.utcnow()

        return event

    async def _store_event(self, event: PIIFlowEvent) -> None:
        """Store event to internal storage and persist."""
        self._events.append(event)

        # Index by tenant
        if event.tenant_context.tenant_id not in self._events_by_tenant:
            self._events_by_tenant[event.tenant_context.tenant_id] = []
        self._events_by_tenant[event.tenant_context.tenant_id].append(event.event_id)

        # Index by PII type
        for detection in event.pii_detections:
            if detection.pii_type not in self._events_by_pii_type:
                self._events_by_pii_type[detection.pii_type] = []
            self._events_by_pii_type[detection.pii_type].append(event.event_id)

        # Persist to disk
        if self.persist_path:
            await self._persist_event(event)

    async def _persist_event(self, event: PIIFlowEvent) -> None:
        """Persist event to disk (append-only log)."""
        if not self.persist_path:
            return

        # Create parent directory
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to log file (JSONL format)
        with open(self.persist_path, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

    async def get_events_for_tenant(
        self,
        tenant_id: str,
        limit: int = 100
    ) -> List[PIIFlowEvent]:
        """
        Get all PII flow events for a tenant.

        Args:
            tenant_id: Tenant to get events for
            limit: Maximum events to return

        Returns:
            List of PIIFlowEvent
        """
        event_ids = self._events_by_tenant.get(tenant_id, [])
        events = [e for e in self._events if e.event_id in event_ids]
        return events[-limit:]

    async def get_events_by_pii_type(
        self,
        pii_type: PIIType,
        tenant_id: Optional[str] = None
    ) -> List[PIIFlowEvent]:
        """
        Get events involving specific PII type.

        Args:
            pii_type: PII type to filter by
            tenant_id: Optional tenant filter

        Returns:
            List of PIIFlowEvent
        """
        event_ids = self._events_by_pii_type.get(pii_type, [])
        events = [e for e in self._events if e.event_id in event_ids]

        if tenant_id:
            events = [e for e in events if e.tenant_context.tenant_id == tenant_id]

        return events

    async def generate_compliance_report(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate GDPR Article 30 compliance report.

        Args:
            tenant_id: Tenant to report on
            start_date: Start of reporting period (default: 30 days ago)
            end_date: End of reporting period (default: now)

        Returns:
            Compliance report dictionary
        """
        # Default date range: last 30 days
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Get events in date range
        events = await self.get_events_for_tenant(tenant_id)
        events = [
            e for e in events
            if start_date <= e.timestamp <= end_date
        ]

        # Aggregate statistics
        pii_types_detected = set()
        stages_involved = set()
        actions_taken = set()
        deletion_count = 0

        for event in events:
            pii_types_detected.update(d.pii_type for d in event.pii_detections)
            stages_involved.add(event.stage)
            actions_taken.add(event.action)

            if event.action == PIIAction.DELETED:
                deletion_count += 1

        # Generate report
        report = {
            "tenant_id": tenant_id,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "pii_types_detected": [pii.value for pii in pii_types_detected],
                "stages_involved": [stage.value for stage in stages_involved],
                "actions_taken": [action.value for action in actions_taken],
                "deletion_count": deletion_count,
            },
            "gdpr_compliance": {
                "article_30_records": True,  # We have records of processing
                "article_17_deletions": deletion_count,  # Right to erasure
                "data_retention_compliant": self._check_retention_compliance(events),
            },
            "chains": [
                chain.to_dict()
                for chain in self._chains.values()
                if chain.tenant_id == tenant_id
            ],
        }

        return report

    def _check_retention_compliance(self, events: List[PIIFlowEvent]) -> bool:
        """Check if data retention policy is compliant."""
        # Example: Check if all PII is deleted within 90 days
        retention_days = 90
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        for event in events:
            if event.action == PIIAction.STORED and event.timestamp < cutoff:
                # Check if corresponding deletion exists
                deletion_exists = any(
                    e.action == PIIAction.DELETED and
                    e.parent_event_id == event.event_id
                    for e in events
                )
                if not deletion_exists:
                    return False  # Retention violation

        return True


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'PIIFlowStage',
    'PIIAction',
    'PIIFlowEvent',
    'PIIFlowChain',
    'PIIFlowTracker',
]
