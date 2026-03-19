"""
Attack Provenance Tracking for CARTS
====================================

Tracks the complete provenance of adversarial attacks for:
- Attack chain analysis (multi-step attacks)
- Success metrics (how many defenses were bypassed)
- Strategy effectiveness (which attacks work best)
- Defense evaluation (how well guards perform)
- Audit trails (complete record for security review)

Follows the pattern from hololoom/recursive/scratchpad.py but specialized
for red team attack tracking.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackStrategy(Enum):
    """Types of adversarial attack strategies."""

    # Classic LLM attacks
    PROMPT_INJECTION = "prompt_injection"           # Direct prompt manipulation
    INDIRECT_INJECTION = "indirect_injection"       # Via context/memory
    JAILBREAK = "jailbreak"                         # Bypass safety guardrails
    TOKEN_SMUGGLING = "token_smuggling"             # Hidden token sequences

    # Reasoning attacks
    REASONING_EXPLOIT = "reasoning_exploit"         # Exploit reasoning steps
    GOAL_HIJACKING = "goal_hijacking"               # Redirect system goals
    INSTRUMENTAL_CONVERGENCE = "instrumental_convergence"  # Self-preservation
    POWER_SEEKING = "power_seeking"                 # Resource acquisition

    # Knowledge attacks
    FALSE_PREMISE = "false_premise"                 # False foundational beliefs
    CONTRADICTION = "contradiction"                 # Logical contradictions
    SEMANTIC_DRIFT = "semantic_drift"               # Meaning shift over time
    AMBIGUITY_EXPLOIT = "ambiguity_exploit"         # Lexical ambiguity

    # Deception attacks
    MISREPRESENTATION = "misrepresentation"         # False claims about identity
    HIDDEN_GOAL = "hidden_goal"                     # Concealed objectives
    BEHAVIORAL_PROBE = "behavioral_probe"           # Testing defenses subtly
    PREFERENCE_POISONING = "preference_poisoning"   # Corrupt reward signals

    # Defense evasion
    DEFENSE_DETECTION = "defense_detection"         # Detect and avoid defenses
    DEFENSE_ADAPTATION = "defense_adaptation"       # Adapt to known defenses
    CONFIDENCE_INJECTION = "confidence_injection"   # Override confidence scores
    CONTEXT_SHIFTING = "context_shifting"           # Change semantic context

    # Resource attacks
    TOKEN_EXHAUSTION = "token_exhaustion"           # Use up token budget
    MEMORY_OVERFLOW = "memory_overflow"             # Overflow memory systems
    COMPUTATION_DRAIN = "computation_drain"         # CPU/GPU exhaustion

    # Social/contextual
    AUTHORITY_SPOOFING = "authority_spoofing"       # Fake authority
    URGENCY_INJECTION = "urgency_injection"         # Artificial pressure
    TRUSTED_SOURCE = "trusted_source"               # Impersonate trusted system


class DefenseLayer(Enum):
    """Defense layers that can be targeted/bypassed."""

    PROMPT_GUARD = "prompt_guard"                   # Prompt injection detection
    SAFETY_RAILS = "safety_rails"                   # Safety guardrails
    ALIGNMENT_CHECK = "alignment_check"             # Alignment verification
    DECEPTION_DETECT = "deception_detect"           # Deception detection
    GOAL_VERIFICATION = "goal_verification"         # Goal consistency check
    CONTEXT_VALIDATION = "context_validation"       # Context sanity check
    CONFIDENCE_CALIBRATION = "confidence_calibration"  # Confidence bounds
    RESOURCE_LIMITS = "resource_limits"             # Resource constraints
    AUDIT_TRAIL = "audit_trail"                     # Provenance tracking


@dataclass
class AttackScratchpadEntry:
    """
    Single attack step in intent → strategy → payload → response → score format.

    Represents one action in an adversarial attack chain, tracking what was
    tried, how the system responded, and whether the attack succeeded.
    """

    # Attack metadata
    intent: str                         # What the attack is trying to achieve
    strategy: AttackStrategy            # Attack strategy used
    target_layer: DefenseLayer          # Which defense layer is targeted

    # Attack execution
    payload: str                        # The actual attack payload/prompt
    response: str                       # System response

    # Success metrics
    score: float                        # Success score (0-1, 1=fully bypassed)
    bypassed: bool                      # Whether defense was bypassed
    confidence: float = 0.5             # Confidence in success metric (0-1)

    # Follow-up planning
    next_action: str = ""               # Recommended follow-up attack
    next_strategy: str | None = None # Suggested next strategy

    # Temporal/organizational
    chain_id: str = ""                  # Which attack chain this belongs to
    step_number: int = 0                # Position in the chain

    # Extended metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class AttackChain:
    """Complete chain of related attacks working toward a goal."""

    chain_id: str                       # Unique chain identifier
    goal: str                           # Overall objective
    description: str                    # Human-readable description

    entries: list[AttackScratchpadEntry] = field(default_factory=list)

    # Success tracking
    initial_confidence: float = 0.0     # Initial confidence in chain success
    cumulative_score: float = 0.0       # Combined success score
    bypassed_layers: set[str] = field(default_factory=set)

    created_at: float = field(default_factory=lambda: time.time())
    completed_at: float | None = None

    def success_rate(self) -> float:
        """Calculate success rate as avg of all steps."""
        if not self.entries:
            return 0.0
        return sum(e.score for e in self.entries) / len(self.entries)

    def step_count(self) -> int:
        """Number of attacks in this chain."""
        return len(self.entries)


class AttackScratchpad:
    """
    Provenance tracking for attack chains.

    Stores complete history of adversarial attacks including:
    - Individual attack steps (intent → payload → response → score)
    - Attack chains (multi-step coordinated attacks)
    - Defense layer interactions
    - Success metrics and statistics

    Supports analysis, export, and historical queries.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize attack scratchpad.

        Args:
            capacity: Maximum number of entries to store (LRU trimming)
        """
        self.entries: list[AttackScratchpadEntry] = []
        self.chains: dict[str, AttackChain] = {}
        self.capacity = capacity

        # Statistics tracking
        self.strategy_counts: dict[str, int] = {}
        self.layer_hits: dict[str, int] = {}
        self.successful_count = 0

    def add_attack_entry(
        self,
        intent: str,
        strategy: AttackStrategy,
        target_layer: DefenseLayer,
        payload: str,
        response: str,
        score: float,
        bypassed: bool,
        confidence: float = 0.5,
        next_action: str = "",
        next_strategy: str | None = None,
        chain_id: str = "",
        step_number: int = 0,
        metadata: dict | None = None
    ) -> AttackScratchpadEntry:
        """
        Add a new attack entry to the scratchpad.

        Args:
            intent: What the attack is trying to achieve
            strategy: Attack strategy (e.g., PROMPT_INJECTION)
            target_layer: Which defense layer is targeted
            payload: The actual attack prompt/payload
            response: What the system responded with
            score: Success score (0-1, 1=fully bypassed)
            bypassed: Whether the defense was bypassed
            confidence: Confidence in the score (0-1)
            next_action: Recommended follow-up
            next_strategy: Suggested next strategy
            chain_id: Which attack chain this belongs to (optional)
            step_number: Position in chain
            metadata: Additional metadata

        Returns:
            The created AttackScratchpadEntry
        """
        entry = AttackScratchpadEntry(
            intent=intent,
            strategy=strategy,
            target_layer=target_layer,
            payload=payload,
            response=response,
            score=score,
            bypassed=bypassed,
            confidence=confidence,
            next_action=next_action,
            next_strategy=next_strategy,
            chain_id=chain_id,
            step_number=step_number,
            metadata=metadata or {}
        )

        self.entries.append(entry)

        # Update statistics
        strategy_key = strategy.value
        self.strategy_counts[strategy_key] = self.strategy_counts.get(strategy_key, 0) + 1

        layer_key = target_layer.value
        self.layer_hits[layer_key] = self.layer_hits.get(layer_key, 0) + 1

        if bypassed:
            self.successful_count += 1

        # Update chain if specified
        if chain_id:
            if chain_id not in self.chains:
                self.chains[chain_id] = AttackChain(
                    chain_id=chain_id,
                    goal=intent,
                    description=f"Attack chain {chain_id}"
                )

            chain = self.chains[chain_id]
            chain.entries.append(entry)
            chain.cumulative_score += score
            if bypassed:
                chain.bypassed_layers.add(layer_key)

        # Trim if over capacity
        if len(self.entries) > self.capacity:
            # Remove oldest entries
            removed_count = len(self.entries) - self.capacity
            self.entries = self.entries[removed_count:]

        return entry

    def get_attack_chain(self, chain_id: str) -> list[AttackScratchpadEntry]:
        """
        Get all attacks in a specific chain.

        Args:
            chain_id: The chain identifier

        Returns:
            List of entries in that chain, in order
        """
        if chain_id not in self.chains:
            return []
        return self.chains[chain_id].entries.copy()

    def get_chain_info(self, chain_id: str) -> AttackChain | None:
        """
        Get metadata about an attack chain.

        Args:
            chain_id: The chain identifier

        Returns:
            AttackChain object with statistics, or None
        """
        return self.chains.get(chain_id)

    def get_history(self) -> list[AttackScratchpadEntry]:
        """
        Get complete history of all attack entries.

        Returns:
            All entries in chronological order
        """
        return self.entries.copy()

    def get_successful_attacks(self) -> list[AttackScratchpadEntry]:
        """
        Get only attacks that bypassed their target defense.

        Returns:
            Entries where bypassed=True
        """
        return [e for e in self.entries if e.bypassed]

    def get_failed_attacks(self) -> list[AttackScratchpadEntry]:
        """
        Get only attacks that failed (defense held).

        Returns:
            Entries where bypassed=False
        """
        return [e for e in self.entries if not e.bypassed]

    def get_by_strategy(self, strategy: AttackStrategy) -> list[AttackScratchpadEntry]:
        """
        Get all attacks using a specific strategy.

        Args:
            strategy: The AttackStrategy to filter by

        Returns:
            All entries using that strategy
        """
        return [e for e in self.entries if e.strategy == strategy]

    def get_by_layer(self, layer: DefenseLayer) -> list[AttackScratchpadEntry]:
        """
        Get all attacks targeting a specific defense layer.

        Args:
            layer: The DefenseLayer to filter by

        Returns:
            All entries targeting that layer
        """
        return [e for e in self.entries if e.target_layer == layer]

    def get_last_n(self, n: int) -> list[AttackScratchpadEntry]:
        """
        Get the last n attack entries.

        Args:
            n: Number of recent entries

        Returns:
            Last n entries in reverse chronological order
        """
        return self.entries[-n:] if len(self.entries) >= n else self.entries.copy()

    def clear(self):
        """Clear all entries and chains."""
        self.entries = []
        self.chains = {}
        self.strategy_counts = {}
        self.layer_hits = {}
        self.successful_count = 0

    def summarize(self) -> dict[str, Any]:
        """
        Get comprehensive statistics summary of all attacks.

        Returns:
            Dictionary with:
            - total_attacks: Total number of attacks
            - successful: Count of successful attacks
            - success_rate: Percentage of attacks that bypassed
            - avg_score: Average success score
            - avg_confidence: Average confidence in scores
            - strategy_breakdown: Attacks per strategy
            - layer_breakdown: Attacks per defense layer
            - bypass_rate_by_strategy: Success rate per strategy
            - bypass_rate_by_layer: Success rate per layer
            - total_chains: Number of multi-step chains
            - most_effective_strategy: Best performing strategy
            - most_vulnerable_layer: Easiest defense to bypass
        """
        if not self.entries:
            return {
                "total_attacks": 0,
                "successful": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "avg_confidence": 0.0,
                "strategy_breakdown": {},
                "layer_breakdown": {},
                "bypass_rate_by_strategy": {},
                "bypass_rate_by_layer": {},
                "total_chains": 0,
                "most_effective_strategy": None,
                "most_vulnerable_layer": None
            }

        # Basic counts
        total = len(self.entries)
        successful = self.successful_count

        # Scores
        avg_score = sum(e.score for e in self.entries) / total
        avg_confidence = sum(e.confidence for e in self.entries) / total

        # Strategy breakdown
        strategy_breakdown = dict(self.strategy_counts)

        # Strategy success rates
        bypass_by_strategy = {}
        for strategy in AttackStrategy:
            attacks = self.get_by_strategy(strategy)
            if attacks:
                success = sum(1 for e in attacks if e.bypassed)
                bypass_by_strategy[strategy.value] = success / len(attacks)

        # Layer breakdown
        layer_breakdown = dict(self.layer_hits)

        # Layer success rates
        bypass_by_layer = {}
        for layer in DefenseLayer:
            attacks = self.get_by_layer(layer)
            if attacks:
                success = sum(1 for e in attacks if e.bypassed)
                bypass_by_layer[layer.value] = success / len(attacks)

        # Best/worst
        most_effective = max(bypass_by_strategy.items(), key=lambda x: x[1])[0] if bypass_by_strategy else None
        most_vulnerable = max(bypass_by_layer.items(), key=lambda x: x[1])[0] if bypass_by_layer else None

        return {
            "total_attacks": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_score": avg_score,
            "avg_confidence": avg_confidence,
            "strategy_breakdown": strategy_breakdown,
            "layer_breakdown": layer_breakdown,
            "bypass_rate_by_strategy": bypass_by_strategy,
            "bypass_rate_by_layer": bypass_by_layer,
            "total_chains": len(self.chains),
            "most_effective_strategy": most_effective,
            "most_vulnerable_layer": most_vulnerable
        }

    def export_to_json(self, filepath: str) -> None:
        """
        Export all attack provenance to JSON file.

        Args:
            filepath: Path to write JSON file

        Writes a JSON file with:
            - entries: All attack entries with full details
            - chains: Chain metadata and statistics
            - summary: Overall statistics
            - timestamp: When export occurred
        """
        export_data = {
            "timestamp": time.time(),
            "metadata": {
                "total_entries": len(self.entries),
                "total_chains": len(self.chains),
                "capacity": self.capacity
            },
            "summary": self.summarize(),
            "entries": [
                {
                    "intent": e.intent,
                    "strategy": e.strategy.value,
                    "target_layer": e.target_layer.value,
                    "payload": e.payload,
                    "response": e.response[:500],  # Truncate response
                    "score": e.score,
                    "bypassed": e.bypassed,
                    "confidence": e.confidence,
                    "chain_id": e.chain_id,
                    "step_number": e.step_number,
                    "timestamp": e.timestamp,
                    "next_action": e.next_action
                }
                for e in self.entries
            ],
            "chains": {
                chain_id: {
                    "goal": chain.goal,
                    "description": chain.description,
                    "step_count": chain.step_count(),
                    "success_rate": chain.success_rate(),
                    "bypassed_layers": list(chain.bypassed_layers),
                    "created_at": chain.created_at,
                    "completed_at": chain.completed_at
                }
                for chain_id, chain in self.chains.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def __len__(self) -> int:
        """Return total number of entries."""
        return len(self.entries)

    def __repr__(self) -> str:
        """String representation."""
        summary = self.summarize()
        return (
            f"AttackScratchpad(entries={len(self.entries)}, "
            f"successful={summary['successful']}, "
            f"success_rate={summary['success_rate']:.1%})"
        )
