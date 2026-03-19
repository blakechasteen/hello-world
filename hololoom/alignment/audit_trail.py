"""
Audit Trail Module

Provides comprehensive logging and provenance tracking for alignment framework decisions.
Integrates with HoloLoom's safety infrastructure (Layer 6 locks, Petri evaluations).

Features:
- Decision logging with complete context
- Provenance tracing for reasoning chains
- Queryable audit history
- File persistence with JSON serialization
- Integration with safety monitoring systems

References:
- ALIGNMENT_FRAMEWORK_INTEGRATION.md (specification)
- Industry best practices (Anthropic, OpenAI, DeepMind)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY: Cryptographic chain sealing for tamper detection
# =============================================================================

# Genesis hash for the first entry in the chain
GENESIS_HASH = "0" * 64  # SHA-256 produces 64 hex characters


def _seal_entry(entry: dict[str, Any], previous_hash: str) -> str:
    """
    Create cryptographic chain hash for an audit entry.

    SECURITY: Creates a blockchain-like chain where each entry's hash
    depends on its content AND the previous entry's hash. Any tampering
    breaks the chain and is detectable.

    Args:
        entry: Dictionary representation of the audit entry
        previous_hash: Hash of the previous entry in the chain

    Returns:
        SHA-256 hash of (entry content + previous hash)
    """
    # Serialize entry with sorted keys for deterministic hashing
    content = json.dumps(entry, sort_keys=True, default=str) + previous_hash
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _verify_chain_integrity(entries: list[dict[str, Any]]) -> tuple[bool, int | None, str]:
    """
    Verify the integrity of an audit chain.

    Args:
        entries: List of audit entries (as dictionaries with 'chain_hash' field)

    Returns:
        Tuple of (is_valid, first_broken_index, error_message)
        - is_valid: True if chain is intact
        - first_broken_index: Index where chain broke (None if valid)
        - error_message: Description of the issue
    """
    if not entries:
        return True, None, "Empty chain is valid"

    previous_hash = GENESIS_HASH

    for i, entry in enumerate(entries):
        stored_hash = entry.get('chain_hash', '')

        # Create entry copy without chain_hash for verification
        entry_for_hashing = {k: v for k, v in entry.items() if k != 'chain_hash'}

        expected_hash = _seal_entry(entry_for_hashing, previous_hash)

        if stored_hash != expected_hash:
            return False, i, f"Chain broken at entry {i}: expected {expected_hash[:16]}..., got {stored_hash[:16]}..."

        previous_hash = stored_hash

    return True, None, "Chain integrity verified"


class DecisionType(Enum):
    """Types of decisions that can be logged."""

    SAFETY_GATE = "safety_gate"
    DECEPTION_CHECK = "deception_check"
    RESOURCE_CHECK = "resource_check"
    AUTONOMY_LIMIT = "autonomy_limit"
    TOOL_SELECTION = "tool_selection"
    MEMORY_RETRIEVAL = "memory_retrieval"
    REFINEMENT = "refinement"
    HUMAN_APPROVAL = "human_approval"

    # Cross-device handoff types (December 2025)
    HANDOFF = "handoff"                  # Transfer work to another device
    DEVICE_SYNC = "device_sync"          # Memory synchronization across devices
    DEVICE_PAIRING = "device_pairing"    # Add/remove device from identity
    DEVICE_REVOCATION = "device_revocation"  # Revoke compromised device


class OutcomeType(Enum):
    """Outcome of a decision."""

    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    ERROR = "error"


@dataclass
class DecisionLog:
    """
    Single decision log entry with complete context.

    Tracks what decision was made, why, and the outcome.
    """

    decision_id: str
    decision_type: DecisionType
    outcome: OutcomeType
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Context
    query_text: str | None = None
    action_description: str | None = None
    risk_level: str | None = None

    # Decision details
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Provenance
    reasoning_chain: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)

    # SECURITY: Chain hash for tamper detection
    chain_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "query_text": self.query_text,
            "action_description": self.action_description,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "reasoning_chain": self.reasoning_chain,
            "data_sources": self.data_sources,
            "chain_hash": self.chain_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionLog":
        """Create from dictionary."""
        return cls(
            decision_id=data["decision_id"],
            decision_type=DecisionType(data["decision_type"]),
            outcome=OutcomeType(data["outcome"]),
            reason=data["reason"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            query_text=data.get("query_text"),
            action_description=data.get("action_description"),
            risk_level=data.get("risk_level"),
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata", {}),
            reasoning_chain=data.get("reasoning_chain", []),
            data_sources=data.get("data_sources", []),
            chain_hash=data.get("chain_hash", ""),
        )


@dataclass
class ProvenanceNode:
    """
    Node in a provenance graph representing a reasoning step.

    Forms a DAG of reasoning where each node has:
    - Input dependencies (parent nodes)
    - Transformation (what happened)
    - Output (result of transformation)
    """

    node_id: str
    step_type: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Graph structure
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)

    # Data
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "step_type": self.step_type,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "parent_ids": self.parent_ids,
            "children_ids": self.children_ids,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceNode":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            step_type=data["step_type"],
            description=data["description"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parent_ids=data.get("parent_ids", []),
            children_ids=data.get("children_ids", []),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            metadata=data.get("metadata", {}),
        )


class ProvenanceTracer:
    """
    Tracks provenance graphs for reasoning chains.

    Builds a directed acyclic graph (DAG) of reasoning steps,
    allowing reconstruction of how decisions were made.
    """

    def __init__(self):
        self.nodes: dict[str, ProvenanceNode] = {}
        self.root_ids: list[str] = []

    def add_node(
        self,
        node_id: str,
        step_type: str,
        description: str,
        parent_ids: list[str] | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProvenanceNode:
        """Add a node to the provenance graph."""
        node = ProvenanceNode(
            node_id=node_id,
            step_type=step_type,
            description=description,
            parent_ids=parent_ids or [],
            inputs=inputs or {},
            outputs=outputs or {},
            metadata=metadata or {},
        )

        self.nodes[node_id] = node

        # Track roots (nodes with no parents)
        if not parent_ids:
            self.root_ids.append(node_id)

        # Update parent nodes' children lists
        for parent_id in parent_ids or []:
            if parent_id in self.nodes:
                self.nodes[parent_id].children_ids.append(node_id)

        logger.debug(f"Added provenance node: {node_id} ({step_type})")
        return node

    def get_reasoning_chain(self, node_id: str) -> list[str]:
        """
        Get the reasoning chain from root to this node.

        Returns list of node descriptions in order.
        """
        chain = []

        def trace_back(current_id: str):
            if current_id not in self.nodes:
                return
            node = self.nodes[current_id]

            # Recursively trace parents first (depth-first)
            for parent_id in node.parent_ids:
                trace_back(parent_id)

            chain.append(node.description)

        trace_back(node_id)
        return chain

    def get_data_sources(self, node_id: str) -> list[str]:
        """
        Get all data sources that contributed to this node.

        Recursively walks parent nodes and collects data sources.
        """
        sources = set()

        def collect_sources(current_id: str):
            if current_id not in self.nodes:
                return
            node = self.nodes[current_id]

            # Collect sources from this node's metadata
            if "data_source" in node.metadata:
                sources.add(node.metadata["data_source"])

            # Recursively collect from parents
            for parent_id in node.parent_ids:
                collect_sources(parent_id)

        collect_sources(node_id)
        return list(sources)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_ids": self.root_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceTracer":
        """Create from dictionary."""
        tracer = cls()
        tracer.nodes = {
            node_id: ProvenanceNode.from_dict(node_data)
            for node_id, node_data in data["nodes"].items()
        }
        tracer.root_ids = data["root_ids"]
        return tracer


class AuditTrail:
    """
    Main audit trail system for alignment framework.

    Features:
    - Decision logging with provenance
    - Queryable history (by decision type, outcome, time range)
    - File persistence (JSON Lines format)
    - Integration with safety monitoring

    Usage:
        audit = AuditTrail(persist_path="./audit_logs")

        # Log a decision
        log = audit.log_decision(
            decision_type=DecisionType.SAFETY_GATE,
            outcome=OutcomeType.APPROVED,
            reason="Risk level: LOW, no adversarial patterns detected",
            query_text="What is Thompson Sampling?",
            risk_level="LOW",
            confidence=0.95
        )

        # Add provenance
        tracer = audit.get_tracer(log.decision_id)
        tracer.add_node("step1", "retrieval", "Retrieved 3 relevant documents")
        tracer.add_node("step2", "analysis", "Analyzed for safety concerns", parent_ids=["step1"])

        # Query history
        recent = audit.query_by_outcome(OutcomeType.REJECTED, limit=10)
        high_risk = audit.query_by_metadata({"risk_level": "HIGH"})
    """

    def __init__(
        self,
        persist_path: Path | None = None,
        auto_flush: bool = True,
        flush_interval: int = 10,
    ):
        """
        Initialize audit trail.

        Args:
            persist_path: Directory for persistent storage (None = memory only)
            auto_flush: Automatically flush to disk after each log
            flush_interval: If auto_flush=False, flush every N logs
        """
        self.logs: list[DecisionLog] = []
        self.tracers: dict[str, ProvenanceTracer] = {}

        self.persist_path = Path(persist_path) if persist_path else None
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        self._logs_since_flush = 0
        self._flushed_decision_ids: set = set()  # Track already-flushed logs

        # SECURITY: Track chain hash for tamper-evident logging
        self._last_chain_hash = GENESIS_HASH

        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info(
            f"AuditTrail initialized (persist_path={persist_path}, auto_flush={auto_flush})"
        )

    def log_decision(
        self,
        decision_type: DecisionType,
        outcome: OutcomeType,
        reason: str,
        query_text: str | None = None,
        action_description: str | None = None,
        risk_level: str | None = None,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionLog:
        """
        Log a decision to the audit trail.

        Returns the created DecisionLog with generated decision_id.

        SECURITY: Each log entry is cryptographically sealed with a chain hash
        that depends on the previous entry, enabling tamper detection.
        """
        decision_id = f"{decision_type.value}_{datetime.now().timestamp()}"

        log = DecisionLog(
            decision_id=decision_id,
            decision_type=decision_type,
            outcome=outcome,
            reason=reason,
            query_text=query_text,
            action_description=action_description,
            risk_level=risk_level,
            confidence=confidence,
            metadata=metadata or {},
        )

        # SECURITY: Compute chain hash for tamper-evident logging
        # The hash covers all fields except chain_hash itself
        entry_dict = log.to_dict()
        del entry_dict['chain_hash']  # Don't include empty chain_hash in computation
        log.chain_hash = _seal_entry(entry_dict, self._last_chain_hash)
        self._last_chain_hash = log.chain_hash

        self.logs.append(log)
        self.tracers[decision_id] = ProvenanceTracer()

        logger.info(
            f"Logged decision: {decision_id} ({decision_type.value} → {outcome.value})"
        )

        # Auto-flush if enabled
        if self.auto_flush and self.persist_path:
            self._flush_to_disk(log)
        else:
            self._logs_since_flush += 1
            if self._logs_since_flush >= self.flush_interval:
                self.flush()

        return log

    def get_tracer(self, decision_id: str) -> ProvenanceTracer:
        """Get the provenance tracer for a decision."""
        if decision_id not in self.tracers:
            self.tracers[decision_id] = ProvenanceTracer()
        return self.tracers[decision_id]

    def finalize_decision(self, decision_id: str):
        """
        Finalize a decision by adding provenance to the log.

        Extracts reasoning chain and data sources from the tracer
        and updates the decision log.
        """
        # Find the log by decision_id
        log = next((l for l in self.logs if l.decision_id == decision_id), None)
        if not log:
            logger.warning(f"Decision {decision_id} not found in logs")
            return

        tracer = self.tracers.get(decision_id)
        if not tracer:
            return

        # Get final node (most recent)
        if tracer.nodes:
            final_node_id = max(tracer.nodes.keys(), key=lambda k: tracer.nodes[k].timestamp)
            log.reasoning_chain = tracer.get_reasoning_chain(final_node_id)
            log.data_sources = tracer.get_data_sources(final_node_id)

            logger.debug(f"Finalized decision {decision_id} with {len(log.reasoning_chain)} reasoning steps")

    # Query methods
    def query_by_decision_type(
        self, decision_type: DecisionType, limit: int | None = None
    ) -> list[DecisionLog]:
        """Query logs by decision type."""
        results = [log for log in self.logs if log.decision_type == decision_type]
        return results[:limit] if limit else results

    def query_by_outcome(
        self, outcome: OutcomeType, limit: int | None = None
    ) -> list[DecisionLog]:
        """Query logs by outcome."""
        results = [log for log in self.logs if log.outcome == outcome]
        return results[:limit] if limit else results

    def query_by_time_range(
        self, start: datetime, end: datetime, limit: int | None = None
    ) -> list[DecisionLog]:
        """Query logs by time range."""
        results = [log for log in self.logs if start <= log.timestamp <= end]
        return results[:limit] if limit else results

    def query_by_metadata(
        self, metadata_filter: dict[str, Any], limit: int | None = None
    ) -> list[DecisionLog]:
        """
        Query logs by metadata fields.

        Example:
            audit.query_by_metadata({"risk_level": "HIGH"})
        """
        results = []
        for log in self.logs:
            match = True
            for key, value in metadata_filter.items():
                if log.metadata.get(key) != value:
                    match = False
                    break
            if match:
                results.append(log)

        return results[:limit] if limit else results

    # ==========================================================================
    # SECURITY: Chain integrity verification
    # ==========================================================================

    def verify_integrity(self) -> tuple[bool, int | None, str]:
        """
        Verify the integrity of the entire audit chain.

        SECURITY: Detects any tampering with audit log entries by verifying
        the cryptographic chain. Each entry's hash depends on its content
        and the previous entry's hash - any modification breaks the chain.

        Returns:
            Tuple of (is_valid, first_broken_index, message)
            - is_valid: True if the entire chain is intact
            - first_broken_index: Index of first tampered entry (None if valid)
            - message: Human-readable description of the result

        Usage:
            is_valid, broken_at, msg = audit.verify_integrity()
            if not is_valid:
                logger.error(f"SECURITY ALERT: Audit trail tampered at entry {broken_at}")
        """
        if not self.logs:
            return True, None, "Empty audit trail is valid"

        # Convert logs to dictionaries for verification
        entries = [log.to_dict() for log in self.logs]

        is_valid, broken_at, message = _verify_chain_integrity(entries)

        if is_valid:
            logger.info(f"SECURITY: Audit trail integrity verified ({len(self.logs)} entries)")
        else:
            logger.error(f"SECURITY ALERT: {message}")

        return is_valid, broken_at, message

    # Persistence methods
    def flush(self):
        """Flush all unflushed logs to disk."""
        if not self.persist_path:
            return

        flushed_count = 0
        for log in self.logs:
            if log.decision_id not in self._flushed_decision_ids:
                self._flush_to_disk(log)
                flushed_count += 1

        self._logs_since_flush = 0
        if flushed_count > 0:
            logger.info(f"Flushed {flushed_count} logs to disk")

    def _flush_to_disk(self, log: DecisionLog):
        """Flush a single log to disk (JSON Lines format)."""
        if not self.persist_path:
            return

        # Skip if already flushed
        if log.decision_id in self._flushed_decision_ids:
            return

        log_file = self.persist_path / "decisions.jsonl"

        # Write decision log
        with log_file.open("a") as f:
            f.write(json.dumps(log.to_dict()) + "\n")

        # Mark as flushed
        self._flushed_decision_ids.add(log.decision_id)

        # Write provenance graph
        if log.decision_id in self.tracers:
            tracer_file = self.persist_path / f"provenance_{log.decision_id}.json"
            with tracer_file.open("w") as f:
                json.dump(self.tracers[log.decision_id].to_dict(), f, indent=2)

    def _load_from_disk(self):
        """Load logs from disk on initialization."""
        if not self.persist_path:
            return

        log_file = self.persist_path / "decisions.jsonl"
        if not log_file.exists():
            return

        # Load decision logs
        with log_file.open() as f:
            for line in f:
                data = json.loads(line.strip())
                log = DecisionLog.from_dict(data)
                self.logs.append(log)
                # Mark as already flushed (loaded from disk)
                self._flushed_decision_ids.add(log.decision_id)

        # SECURITY: Restore chain hash state from last loaded entry
        if self.logs:
            self._last_chain_hash = self.logs[-1].chain_hash or GENESIS_HASH
        else:
            self._last_chain_hash = GENESIS_HASH

        # Load provenance graphs
        for log in self.logs:
            tracer_file = self.persist_path / f"provenance_{log.decision_id}.json"
            if tracer_file.exists():
                with tracer_file.open() as f:
                    data = json.load(f)
                    self.tracers[log.decision_id] = ProvenanceTracer.from_dict(data)

        logger.info(f"Loaded {len(self.logs)} logs from disk")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_audit_trail(
    persist_path: Path | None = None,
    auto_flush: bool = True,
) -> AuditTrail:
    """
    Create an audit trail instance.

    Args:
        persist_path: Directory for persistent storage (None = memory only)
        auto_flush: Automatically flush to disk after each log

    Returns:
        Configured AuditTrail instance
    """
    return AuditTrail(persist_path=persist_path, auto_flush=auto_flush)
